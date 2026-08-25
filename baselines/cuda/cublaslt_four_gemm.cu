#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t status_ = (expr);                                            \
        if (status_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA failure line %d: %s: %s\n",            \
                         __LINE__, #expr, cudaGetErrorString(status_));           \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(expr)                                                       \
    do {                                                                        \
        cublasStatus_t status_ = (expr);                                         \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                  \
            std::fprintf(stderr, "cuBLASLt failure line %d: %s (code=%d)\n",   \
                         __LINE__, #expr, static_cast<int>(status_));             \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (0)

struct Options {
    int batch = 256;
    int m = 1024;
    int n = 1024;
    int k = 512;
    int warmup = 20;
    int iterations = 100;
    bool check = false;
    size_t workspace_bytes = 128ull << 20;
};

int parse_int(const char* text, const char* name) {
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (!end || *end != '\0' || value <= 0 || value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "invalid %s: %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        auto need = [&](const char* name) -> const char* {
            if (index + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(EXIT_FAILURE);
            }
            return argv[++index];
        };
        if (std::strcmp(argv[index], "--batch") == 0)
            options.batch = parse_int(need("--batch"), "batch");
        else if (std::strcmp(argv[index], "--m") == 0)
            options.m = parse_int(need("--m"), "m");
        else if (std::strcmp(argv[index], "--n") == 0)
            options.n = parse_int(need("--n"), "n");
        else if (std::strcmp(argv[index], "--k") == 0)
            options.k = parse_int(need("--k"), "k");
        else if (std::strcmp(argv[index], "--warmup") == 0)
            options.warmup = parse_int(need("--warmup"), "warmup");
        else if (std::strcmp(argv[index], "--iterations") == 0)
            options.iterations = parse_int(need("--iterations"), "iterations");
        else if (std::strcmp(argv[index], "--check") == 0)
            options.check = true;
        else {
            std::fprintf(stderr, "unknown option: %s\n", argv[index]);
            std::exit(EXIT_FAILURE);
        }
    }
    return options;
}

__device__ __forceinline__ uint32_t mix32(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

__global__ void fill_half(__half* output, size_t count, uint32_t seed) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint32_t bits = mix32(static_cast<uint32_t>(index) ^ seed);
    output[index] = __float2half(static_cast<float>(bits & 0xffffu) / 32768.0f - 1.0f);
}

template <typename Callable>
float time_callable(int warmup, int iterations, const Callable& callable) {
    for (int index = 0; index < warmup; ++index) callable();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int index = 0; index < iterations; ++index) callable();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return milliseconds / iterations;
}

struct Plan {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t c = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
};

void destroy(Plan& plan) {
    if (plan.workspace) cudaFree(plan.workspace);
    if (plan.c) cublasLtMatrixLayoutDestroy(plan.c);
    if (plan.b) cublasLtMatrixLayoutDestroy(plan.b);
    if (plan.a) cublasLtMatrixLayoutDestroy(plan.a);
    if (plan.operation) cublasLtMatmulDescDestroy(plan.operation);
    if (plan.handle) cublasLtDestroy(plan.handle);
}

void set_batch(cublasLtMatrixLayout_t layout, int count, int64_t stride) {
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride, sizeof(stride)));
}

Plan make_plan(const Options& options, const __half* a_data,
               const __half* b_data, float* c_data) {
    Plan plan;
    plan.workspace_bytes = options.workspace_bytes;
    CUBLAS_CHECK(cublasLtCreate(&plan.handle));
    CUBLAS_CHECK(cublasLtMatmulDescCreate(
        &plan.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t op_a = CUBLAS_OP_N;
    cublasOperation_t op_b = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        plan.operation, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        plan.operation, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &plan.a, CUDA_R_16F, options.m, options.k, options.k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &plan.b, CUDA_R_16F, options.n, options.k, options.k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &plan.c, CUDA_R_32F, options.m, options.n, options.n));
    cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
    for (auto layout : {plan.a, plan.b, plan.c}) {
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
            layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)));
    }
    set_batch(plan.a, options.batch, static_cast<int64_t>(options.m) * options.k);
    set_batch(plan.b, options.batch, static_cast<int64_t>(options.n) * options.k);
    set_batch(plan.c, options.batch, static_cast<int64_t>(options.m) * options.n);
    CUDA_CHECK(cudaMalloc(&plan.workspace, plan.workspace_bytes));
    cublasLtMatmulPreference_t preference = nullptr;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &plan.workspace_bytes, sizeof(plan.workspace_bytes)));
    constexpr int max_algorithms = 16;
    cublasLtMatmulHeuristicResult_t results[max_algorithms];
    int returned = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        plan.handle, plan.operation, plan.a, plan.b, plan.c, plan.c,
        preference, max_algorithms, results, &returned));
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    if (returned == 0) {
        std::fprintf(stderr, "cuBLASLt returned no algorithm\n");
        std::exit(EXIT_FAILURE);
    }
    float alpha = 1.0f, beta = 0.0f;
    float best = std::numeric_limits<float>::infinity();
    for (int index = 0; index < returned; ++index) {
        auto call = [&]() {
            CUBLAS_CHECK(cublasLtMatmul(
                plan.handle, plan.operation, &alpha, a_data, plan.a,
                b_data, plan.b, &beta, c_data, plan.c, c_data, plan.c,
                &results[index].algo, plan.workspace, plan.workspace_bytes, nullptr));
        };
        float milliseconds = time_callable(2, 6, call);
        if (milliseconds < best) {
            best = milliseconds;
            plan.algorithm = results[index].algo;
        }
    }
    return plan;
}

void gemm(const Plan& plan, const __half* a, const __half* b, float* c,
          float alpha, float beta) {
    CUBLAS_CHECK(cublasLtMatmul(
        plan.handle, plan.operation, &alpha, a, plan.a, b, plan.b,
        &beta, c, plan.c, c, plan.c, &plan.algorithm,
        plan.workspace, plan.workspace_bytes, nullptr));
}

void cpu_check(const Options& options,
               const std::vector<__half>& wr, const std::vector<__half>& wi,
               const std::vector<__half>& xr, const std::vector<__half>& xi,
               const std::vector<float>& yr, const std::vector<float>& yi) {
    size_t output_stride = static_cast<size_t>(options.m) * options.n;
    size_t a_stride = static_cast<size_t>(options.m) * options.k;
    size_t b_stride = static_cast<size_t>(options.n) * options.k;
    float max_abs = 0.0f;
    size_t bad = 0;
    for (int batch = 0; batch < options.batch; ++batch) {
        for (int m = 0; m < options.m; ++m) {
            for (int n = 0; n < options.n; ++n) {
                float ref_r = 0.0f, ref_i = 0.0f;
                for (int k = 0; k < options.k; ++k) {
                    float ar = __half2float(wr[batch * a_stride + m * options.k + k]);
                    float ai = __half2float(wi[batch * a_stride + m * options.k + k]);
                    float br = __half2float(xr[batch * b_stride + n * options.k + k]);
                    float bi = __half2float(xi[batch * b_stride + n * options.k + k]);
                    ref_r += ar * br + ai * bi;
                    ref_i += ar * bi - ai * br;
                }
                size_t index = batch * output_stride + m * options.n + n;
                float difference = std::max(std::fabs(ref_r - yr[index]),
                                            std::fabs(ref_i - yi[index]));
                max_abs = std::max(max_abs, difference);
                if (difference > 0.02f) ++bad;
            }
        }
    }
    std::printf("{\"kind\":\"cublaslt_complex_correctness\","
                "\"max_abs_diff\":%.9g,\"bad\":%zu,\"pass\":%s}\n",
                max_abs, bad, bad == 0 ? "true" : "false");
    if (bad != 0) std::exit(EXIT_FAILURE);
}

}  // namespace

int main(int argc, char** argv) {
    Options options = parse_options(argc, argv);
    size_t a_count = static_cast<size_t>(options.batch) * options.m * options.k;
    size_t b_count = static_cast<size_t>(options.batch) * options.n * options.k;
    size_t c_count = static_cast<size_t>(options.batch) * options.m * options.n;
    __half *wr = nullptr, *wi = nullptr, *xr = nullptr, *xi = nullptr;
    float *yr = nullptr, *yi = nullptr;
    CUDA_CHECK(cudaMalloc(&wr, a_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&wi, a_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&xr, b_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&xi, b_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&yr, c_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&yi, c_count * sizeof(float)));
    int threads = 256;
    fill_half<<<static_cast<int>((a_count + threads - 1) / threads), threads>>>(wr, a_count, 1u);
    fill_half<<<static_cast<int>((a_count + threads - 1) / threads), threads>>>(wi, a_count, 2u);
    fill_half<<<static_cast<int>((b_count + threads - 1) / threads), threads>>>(xr, b_count, 3u);
    fill_half<<<static_cast<int>((b_count + threads - 1) / threads), threads>>>(xi, b_count, 4u);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    Plan plan = make_plan(options, wr, xr, yr);
    auto complex_path = [&]() {
        gemm(plan, wr, xr, yr, 1.0f, 0.0f);
        gemm(plan, wi, xi, yr, 1.0f, 1.0f);
        gemm(plan, wr, xi, yi, 1.0f, 0.0f);
        gemm(plan, wi, xr, yi, -1.0f, 1.0f);
    };
    complex_path();
    CUDA_CHECK(cudaDeviceSynchronize());
    if (options.check) {
        std::vector<__half> hwr(a_count), hwi(a_count), hxr(b_count), hxi(b_count);
        std::vector<float> hyr(c_count), hyi(c_count);
        CUDA_CHECK(cudaMemcpy(hwr.data(), wr, a_count * sizeof(__half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hwi.data(), wi, a_count * sizeof(__half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hxr.data(), xr, b_count * sizeof(__half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hxi.data(), xi, b_count * sizeof(__half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hyr.data(), yr, c_count * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hyi.data(), yi, c_count * sizeof(float), cudaMemcpyDeviceToHost));
        cpu_check(options, hwr, hwi, hxr, hxi, hyr, hyi);
    }
    float milliseconds = time_callable(options.warmup, options.iterations, complex_path);
    double useful_ops = 8.0 * options.batch * options.m * options.n * options.k;
    double tops = useful_ops / (milliseconds * 1.0e9);
    std::printf(
        "{\"kind\":\"cublaslt_complex_baseline\",\"batch\":%d,"
        "\"m\":%d,\"n\":%d,\"k\":%d,\"warmup\":%d,"
        "\"iterations\":%d,\"milliseconds\":%.9g,"
        "\"effective_complex_tops\":%.9g,\"check\":%s}\n",
        options.batch, options.m, options.n, options.k, options.warmup,
        options.iterations, milliseconds, tops, options.check ? "true" : "false");
    destroy(plan);
    CUDA_CHECK(cudaFree(yi)); CUDA_CHECK(cudaFree(yr));
    CUDA_CHECK(cudaFree(xi)); CUDA_CHECK(cudaFree(xr));
    CUDA_CHECK(cudaFree(wi)); CUDA_CHECK(cudaFree(wr));
    return EXIT_SUCCESS;
}
