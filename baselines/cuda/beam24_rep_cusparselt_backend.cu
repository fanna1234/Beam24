#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparseLt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t status_ = (expr);                                            \
        if (status_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA failed at line %d: %s: %s (%d)\n",      \
                         __LINE__, #expr, cudaGetErrorString(status_),            \
                         static_cast<int>(status_));                              \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (0)

#define CUSPARSELT_CHECK(expr)                                                   \
    do {                                                                        \
        cusparseStatus_t status_ = (expr);                                       \
        if (status_ != CUSPARSE_STATUS_SUCCESS) {                                \
            std::fprintf(stderr,                                                 \
                         "cuSPARSELt failed at line %d: %s: %s (%d)\n",         \
                         __LINE__, #expr, cusparseLtGetErrorString(status_),       \
                         static_cast<int>(status_));                              \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (0)

struct Options {
    int m = 1024;
    int n = 1024;
    int k = 64;
    int batch = 1;
    int warmup = 20;
    int iterations = 100;
    std::string order = "AB";
    bool check = false;
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
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--m") == 0)
            options.m = parse_int(need("--m"), "m");
        else if (std::strcmp(argv[i], "--n") == 0)
            options.n = parse_int(need("--n"), "n");
        else if (std::strcmp(argv[i], "--k") == 0)
            options.k = parse_int(need("--k"), "k");
        else if (std::strcmp(argv[i], "--batch") == 0)
            options.batch = parse_int(need("--batch"), "batch");
        else if (std::strcmp(argv[i], "--warmup") == 0)
            options.warmup = parse_int(need("--warmup"), "warmup");
        else if (std::strcmp(argv[i], "--iterations") == 0)
            options.iterations = parse_int(need("--iterations"), "iterations");
        else if (std::strcmp(argv[i], "--order") == 0)
            options.order = need("--order");
        else if (std::strcmp(argv[i], "--check") == 0)
            options.check = true;
        else {
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            std::exit(EXIT_FAILURE);
        }
    }
    if (options.k % 4 != 0) {
        std::fprintf(stderr, "K must be divisible by four\n");
        std::exit(EXIT_FAILURE);
    }
    if (options.order != "AB" && options.order != "BA") {
        std::fprintf(stderr, "order must be AB or BA\n");
        std::exit(EXIT_FAILURE);
    }
    return options;
}

uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__global__ void fill_half_kernel(__half* output, size_t count, uint32_t seed) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint32_t value = static_cast<uint32_t>(index) ^ seed;
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    float normalized = static_cast<float>(value & 0xffffu) / 32768.0f - 1.0f;
    output[index] = __float2half(normalized);
}

__global__ void local_f4_kernel(const __half* input_real,
                                const __half* input_imag,
                                __half* output_real,
                                __half* output_imag,
                                size_t complex_groups) {
    size_t group = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (group >= complex_groups) return;
    size_t base = group * 4;
    float r0 = __half2float(input_real[base + 0]);
    float r1 = __half2float(input_real[base + 1]);
    float r2 = __half2float(input_real[base + 2]);
    float r3 = __half2float(input_real[base + 3]);
    float i0 = __half2float(input_imag[base + 0]);
    float i1 = __half2float(input_imag[base + 1]);
    float i2 = __half2float(input_imag[base + 2]);
    float i3 = __half2float(input_imag[base + 3]);
    float ar = r0 + r2, ai = i0 + i2;
    float br = r0 - r2, bi = i0 - i2;
    float cr = r1 + r3, ci = i1 + i3;
    float dr = r1 - r3, di = i1 - i3;
    output_real[base + 0] = __float2half_rn(0.5f * (ar + cr));
    output_imag[base + 0] = __float2half_rn(0.5f * (ai + ci));
    output_real[base + 1] = __float2half_rn(0.5f * (br - di));
    output_imag[base + 1] = __float2half_rn(0.5f * (bi + dr));
    output_real[base + 2] = __float2half_rn(0.5f * (ar - cr));
    output_imag[base + 2] = __float2half_rn(0.5f * (ai - ci));
    output_real[base + 3] = __float2half_rn(0.5f * (br + di));
    output_imag[base + 3] = __float2half_rn(0.5f * (bi - dr));
}

__global__ void promote_complex_kernel(const __half* input_real,
                                       const __half* input_imag,
                                       float* output_real,
                                       float* output_imag,
                                       size_t count) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output_real[index] = __half2float(input_real[index]);
    output_imag[index] = __half2float(input_imag[index]);
}

struct ComplexWeights {
    std::vector<__half> real;
    std::vector<__half> imag;
};

ComplexWeights make_joint_sparse_weights(int m, int k) {
    static constexpr int masks[6][2] = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3},
    };
    ComplexWeights weights{
        std::vector<__half>(static_cast<size_t>(m) * k, __float2half(0.0f)),
        std::vector<__half>(static_cast<size_t>(m) * k, __float2half(0.0f))};
    for (int row = 0; row < m; ++row) {
        for (int group = 0; group < k / 4; ++group) {
            int mask = static_cast<int>(mix32(static_cast<uint32_t>(row * 65537 + group)) % 6u);
            for (int slot = 0; slot < 2; ++slot) {
                int column = 4 * group + masks[mask][slot];
                uint32_t real_bits = mix32(static_cast<uint32_t>(row * k + column + 17));
                uint32_t imag_bits = mix32(static_cast<uint32_t>(row * k + column + 97));
                float real = static_cast<float>(real_bits & 0xffffu) / 32768.0f - 1.0f;
                float imag = static_cast<float>(imag_bits & 0xffffu) / 32768.0f - 1.0f;
                if (std::fabs(real) < 0.05f) real = real < 0.0f ? -0.05f : 0.05f;
                if (std::fabs(imag) < 0.05f) imag = imag < 0.0f ? -0.05f : 0.05f;
                size_t index = static_cast<size_t>(row) * k + column;
                weights.real[index] = __float2half(real);
                weights.imag[index] = __float2half(imag);
            }
        }
    }
    return weights;
}

template <typename Callable>
float time_callable(int warmup, int iterations, const Callable& callable) {
    for (int i = 0; i < warmup; ++i) callable();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) callable();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / iterations;
}

struct SparsePlan {
    cusparseLtHandle_t handle{};
    cusparseLtMatDescriptor_t weight{};
    cusparseLtMatDescriptor_t data{};
    cusparseLtMatDescriptor_t output{};
    cusparseLtMatmulDescriptor_t operation{};
    cusparseLtMatmulAlgSelection_t selection{};
    cusparseLtMatmulPlan_t plan{};
    __half* compressed = nullptr;
    void* compression_buffer = nullptr;
    void* workspace = nullptr;
    size_t compressed_bytes = 0;
    size_t workspace_bytes = 0;
};

void destroy_sparse(SparsePlan& plan) {
    if (plan.workspace) cudaFree(plan.workspace);
    if (plan.compression_buffer) cudaFree(plan.compression_buffer);
    if (plan.compressed) cudaFree(plan.compressed);
    cusparseLtMatmulPlanDestroy(&plan.plan);
    cusparseLtMatmulAlgSelectionDestroy(&plan.selection);
    cusparseLtMatDescriptorDestroy(&plan.output);
    cusparseLtMatDescriptorDestroy(&plan.data);
    cusparseLtMatDescriptorDestroy(&plan.weight);
    cusparseLtDestroy(&plan.handle);
}

SparsePlan make_sparse_plan(const Options& options, __half* weights,
                            const __half* data, __half* output) {
    SparsePlan sparse;
    CUSPARSELT_CHECK(cusparseLtInit(&sparse.handle));
    auto order = CUSPARSE_ORDER_ROW;
    auto op_a = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto op_b = CUSPARSE_OPERATION_TRANSPOSE;
    constexpr unsigned alignment = 16;
    CUSPARSELT_CHECK(cusparseLtStructuredDescriptorInit(
        &sparse.handle, &sparse.weight, options.m, options.k, options.k,
        alignment, CUDA_R_16F, order, CUSPARSELT_SPARSITY_50_PERCENT));
    CUSPARSELT_CHECK(cusparseLtDenseDescriptorInit(
        &sparse.handle, &sparse.data, options.n, options.k, options.k,
        alignment, CUDA_R_16F, order));
    CUSPARSELT_CHECK(cusparseLtDenseDescriptorInit(
        &sparse.handle, &sparse.output, options.m, options.n, options.n,
        alignment, CUDA_R_16F, order));

    int32_t batch = options.batch;
    int64_t weight_stride = 0;
    int64_t data_stride = static_cast<int64_t>(options.n) * options.k;
    int64_t output_stride = static_cast<int64_t>(options.m) * options.n;
    for (auto* descriptor : {&sparse.weight, &sparse.data, &sparse.output}) {
        CUSPARSELT_CHECK(cusparseLtMatDescSetAttribute(
            &sparse.handle, descriptor, CUSPARSELT_MAT_NUM_BATCHES,
            &batch, sizeof(batch)));
    }
    CUSPARSELT_CHECK(cusparseLtMatDescSetAttribute(
        &sparse.handle, &sparse.weight, CUSPARSELT_MAT_BATCH_STRIDE,
        &weight_stride, sizeof(weight_stride)));
    CUSPARSELT_CHECK(cusparseLtMatDescSetAttribute(
        &sparse.handle, &sparse.data, CUSPARSELT_MAT_BATCH_STRIDE,
        &data_stride, sizeof(data_stride)));
    CUSPARSELT_CHECK(cusparseLtMatDescSetAttribute(
        &sparse.handle, &sparse.output, CUSPARSELT_MAT_BATCH_STRIDE,
        &output_stride, sizeof(output_stride)));

    CUSPARSELT_CHECK(cusparseLtMatmulDescriptorInit(
        &sparse.handle, &sparse.operation, op_a, op_b, &sparse.weight,
        &sparse.data, &sparse.output, &sparse.output, CUSPARSE_COMPUTE_32F));
    CUSPARSELT_CHECK(cusparseLtMatmulDescSetAttribute(
        &sparse.handle, &sparse.operation, CUSPARSELT_MATMUL_SPARSE_MAT_POINTER,
        &weights, sizeof(weights)));
    CUSPARSELT_CHECK(cusparseLtMatmulAlgSelectionInit(
        &sparse.handle, &sparse.selection, &sparse.operation,
        CUSPARSELT_MATMUL_ALG_DEFAULT));
    CUSPARSELT_CHECK(cusparseLtMatmulPlanInit(
        &sparse.handle, &sparse.plan, &sparse.operation, &sparse.selection));

    int* valid = nullptr;
    CUDA_CHECK(cudaMalloc(&valid, sizeof(int)));
    CUSPARSELT_CHECK(cusparseLtSpMMAPruneCheck(
        &sparse.handle, &sparse.operation, weights, valid, nullptr));
    int host_valid = -1;
    CUDA_CHECK(cudaMemcpy(&host_valid, valid, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(valid));
    if (host_valid != 0) {
        std::fprintf(stderr, "custom matrix A failed exact 2:4 legality check\n");
        std::exit(EXIT_FAILURE);
    }

    size_t compression_buffer_bytes = 0;
    CUSPARSELT_CHECK(cusparseLtSpMMACompressedSize(
        &sparse.handle, &sparse.plan, &sparse.compressed_bytes,
        &compression_buffer_bytes));
    CUDA_CHECK(cudaMalloc(&sparse.compressed, sparse.compressed_bytes));
    CUDA_CHECK(cudaMalloc(&sparse.compression_buffer, compression_buffer_bytes));
    CUSPARSELT_CHECK(cusparseLtSpMMACompress(
        &sparse.handle, &sparse.plan, weights, sparse.compressed,
        sparse.compression_buffer, nullptr));

    float alpha = 1.0f, beta = 0.0f;
    CUSPARSELT_CHECK(cusparseLtMatmulSearch(
        &sparse.handle, &sparse.plan, &alpha, sparse.compressed, data, &beta,
        output, output, nullptr, nullptr, 0));
    CUSPARSELT_CHECK(cusparseLtMatmulGetWorkspace(
        &sparse.handle, &sparse.plan, &sparse.workspace_bytes));
    CUDA_CHECK(cudaMalloc(&sparse.workspace, sparse.workspace_bytes));
    return sparse;
}

void sparse_gemm(SparsePlan& sparse, const __half* data, __half* output,
                 float alpha, float beta) {
    CUSPARSELT_CHECK(cusparseLtMatmul(
        &sparse.handle, &sparse.plan, &alpha, sparse.compressed, data,
        &beta, output, output, sparse.workspace, nullptr, 0));
}

void check_composed(const Options& options, const ComplexWeights& weights,
                    const __half* input_real, const __half* input_imag,
                    const __half* transformed_real, const __half* transformed_imag,
                    const float* output_real, const float* output_imag) {
    size_t input_count = static_cast<size_t>(options.batch) * options.n * options.k;
    size_t output_count = static_cast<size_t>(options.batch) * options.m * options.n;
    if (static_cast<unsigned long long>(options.batch) * options.m * options.n *
            options.k > 100000000ull) {
        std::fprintf(stderr, "--check is limited to 100M complex MACs\n");
        std::exit(EXIT_FAILURE);
    }
    std::vector<__half> xr(input_count), xi(input_count), zr(input_count), zi(input_count);
    std::vector<float> yr(output_count), yi(output_count);
    CUDA_CHECK(cudaMemcpy(xr.data(), input_real, input_count * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(xi.data(), input_imag, input_count * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(zr.data(), transformed_real, input_count * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(zi.data(), transformed_imag, input_count * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(yr.data(), output_real, output_count * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(yi.data(), output_imag, output_count * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float f4_max_abs = 0.0f;
    for (size_t base = 0; base < input_count; base += 4) {
        float r0 = __half2float(xr[base + 0]), r1 = __half2float(xr[base + 1]);
        float r2 = __half2float(xr[base + 2]), r3 = __half2float(xr[base + 3]);
        float i0 = __half2float(xi[base + 0]), i1 = __half2float(xi[base + 1]);
        float i2 = __half2float(xi[base + 2]), i3 = __half2float(xi[base + 3]);
        float expected_r[4] = {
            0.5f * (r0 + r1 + r2 + r3),
            0.5f * (r0 - r2 - i1 + i3),
            0.5f * (r0 - r1 + r2 - r3),
            0.5f * (r0 - r2 + i1 - i3)};
        float expected_i[4] = {
            0.5f * (i0 + i1 + i2 + i3),
            0.5f * (i0 - i2 + r1 - r3),
            0.5f * (i0 - i1 + i2 - i3),
            0.5f * (i0 - i2 - r1 + r3)};
        for (int slot = 0; slot < 4; ++slot) {
            float rounded_r = __half2float(__float2half(expected_r[slot]));
            float rounded_i = __half2float(__float2half(expected_i[slot]));
            f4_max_abs = std::max(f4_max_abs,
                std::fabs(rounded_r - __half2float(zr[base + slot])));
            f4_max_abs = std::max(f4_max_abs,
                std::fabs(rounded_i - __half2float(zi[base + slot])));
        }
    }

    float gemm_max_abs = 0.0f;
    unsigned long long bad = 0;
    size_t output_stride = static_cast<size_t>(options.m) * options.n;
    size_t data_stride = static_cast<size_t>(options.n) * options.k;
    for (int batch = 0; batch < options.batch; ++batch) {
        for (int row = 0; row < options.m; ++row) {
            for (int column = 0; column < options.n; ++column) {
                float reference_real = 0.0f, reference_imag = 0.0f;
                for (int k = 0; k < options.k; ++k) {
                    size_t weight_index = static_cast<size_t>(row) * options.k + k;
                    size_t data_index = static_cast<size_t>(batch) * data_stride +
                                        static_cast<size_t>(column) * options.k + k;
                    float ur = __half2float(weights.real[weight_index]);
                    float ui = __half2float(weights.imag[weight_index]);
                    float zrv = __half2float(zr[data_index]);
                    float ziv = __half2float(zi[data_index]);
                    reference_real += ur * zrv + ui * ziv;
                    reference_imag += ur * ziv - ui * zrv;
                }
                size_t output_index = static_cast<size_t>(batch) * output_stride +
                                      static_cast<size_t>(row) * options.n + column;
                float difference = std::max(std::fabs(reference_real - yr[output_index]),
                                            std::fabs(reference_imag - yi[output_index]));
                gemm_max_abs = std::max(gemm_max_abs, difference);
                float reference_abs = std::max(std::fabs(reference_real),
                                               std::fabs(reference_imag));
                if (difference > 0.125f + 0.02f * reference_abs) ++bad;
            }
        }
    }
    bool passed = f4_max_abs == 0.0f && bad == 0;
    std::printf(
        "{\"kind\":\"beam24_rep_cusparselt_correctness\","
        "\"f4_max_abs_diff\":%.9g,\"gemm_max_abs_diff\":%.9g,"
        "\"bad\":%llu,\"pass\":%s}\n",
        f4_max_abs, gemm_max_abs, bad, passed ? "true" : "false");
    if (!passed) std::exit(EXIT_FAILURE);
}

}  // namespace

int main(int argc, char** argv) {
    Options options = parse_options(argc, argv);
    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    CUDA_CHECK(cudaSetDevice(device));

    size_t weight_count = static_cast<size_t>(options.m) * options.k;
    size_t data_count = static_cast<size_t>(options.batch) * options.n * options.k;
    size_t output_count = static_cast<size_t>(options.batch) * options.m * options.n;
    size_t f4_groups = data_count / 4;
    ComplexWeights host_weights = make_joint_sparse_weights(options.m, options.k);

    __half *weight_real = nullptr, *weight_imag = nullptr;
    __half *input_real = nullptr, *input_imag = nullptr;
    __half *transformed_real = nullptr, *transformed_imag = nullptr;
    __half *half_output_real = nullptr, *half_output_imag = nullptr;
    float *output_real = nullptr, *output_imag = nullptr;
    CUDA_CHECK(cudaMalloc(&weight_real, weight_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&weight_imag, weight_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&input_real, data_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&input_imag, data_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&transformed_real, data_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&transformed_imag, data_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&half_output_real, output_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&half_output_imag, output_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&output_real, output_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output_imag, output_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(weight_real, host_weights.real.data(),
                          weight_count * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weight_imag, host_weights.imag.data(),
                          weight_count * sizeof(__half), cudaMemcpyHostToDevice));

    int threads = 256;
    int data_blocks = static_cast<int>((data_count + threads - 1) / threads);
    fill_half_kernel<<<data_blocks, threads>>>(input_real, data_count, 0x12345678u);
    fill_half_kernel<<<data_blocks, threads>>>(input_imag, data_count, 0x87654321u);
    int f4_blocks = static_cast<int>((f4_groups + threads - 1) / threads);
    local_f4_kernel<<<f4_blocks, threads>>>(
        input_real, input_imag, transformed_real, transformed_imag, f4_groups);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    SparsePlan real_plan = make_sparse_plan(
        options, weight_real, transformed_real, half_output_real);
    SparsePlan imag_plan = make_sparse_plan(
        options, weight_imag, transformed_imag, half_output_real);
    int output_blocks = static_cast<int>((output_count + threads - 1) / threads);
    auto composed_call = [&]() {
        local_f4_kernel<<<f4_blocks, threads>>>(
            input_real, input_imag, transformed_real, transformed_imag, f4_groups);
        sparse_gemm(real_plan, transformed_real, half_output_real, 1.0f, 0.0f);
        sparse_gemm(imag_plan, transformed_imag, half_output_real, 1.0f, 1.0f);
        sparse_gemm(real_plan, transformed_imag, half_output_imag, 1.0f, 0.0f);
        sparse_gemm(imag_plan, transformed_real, half_output_imag, -1.0f, 1.0f);
        promote_complex_kernel<<<output_blocks, threads>>>(
            half_output_real, half_output_imag, output_real, output_imag, output_count);
    };

    composed_call();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (options.check) {
        check_composed(options, host_weights, input_real, input_imag,
                       transformed_real, transformed_imag, output_real, output_imag);
    }

    float milliseconds = time_callable(options.warmup, options.iterations, composed_call);
    double useful_ops = 8.0 * options.batch * options.m * options.n * options.k;
    double tops = useful_ops / (milliseconds * 1.0e9);
    int driver = 0, runtime = 0, cusparselt_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUSPARSELT_CHECK(cusparseLtGetVersion(&real_plan.handle, &cusparselt_version));
    std::printf(
        "{\"kind\":\"beam24_rep_cusparselt_backend\",\"gpu\":\"%s\","
        "\"cc\":\"%d.%d\",\"driver\":%d,\"runtime\":%d,"
        "\"cusparselt\":%d,\"batch\":%d,\"m\":%d,\"n\":%d,\"k\":%d,"
        "\"warmup\":%d,\"iterations\":%d,\"milliseconds\":%.9g,"
        "\"effective_complex_tops\":%.9g,\"dynamic_f4\":true,"
        "\"sparse_gemm_calls\":4,\"intermediate\":\"fp16\","
        "\"output\":\"fp32\",\"check\":%s}\n",
        properties.name, properties.major, properties.minor, driver, runtime,
        cusparselt_version, options.batch, options.m, options.n, options.k,
        options.warmup, options.iterations, milliseconds, tops,
        options.check ? "true" : "false");

    destroy_sparse(imag_plan);
    destroy_sparse(real_plan);
    CUDA_CHECK(cudaFree(output_imag));
    CUDA_CHECK(cudaFree(output_real));
    CUDA_CHECK(cudaFree(half_output_imag));
    CUDA_CHECK(cudaFree(half_output_real));
    CUDA_CHECK(cudaFree(transformed_imag));
    CUDA_CHECK(cudaFree(transformed_real));
    CUDA_CHECK(cudaFree(input_imag));
    CUDA_CHECK(cudaFree(input_real));
    CUDA_CHECK(cudaFree(weight_imag));
    CUDA_CHECK(cudaFree(weight_real));
    return EXIT_SUCCESS;
}
