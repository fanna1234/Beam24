#include <cublasLt.h>
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

[[noreturn]] void fail(const char* kind, const char* expression, int line, int code) {
    std::fprintf(stderr, "%s failed at line %d: %s (code=%d)\n", kind, line,
                 expression, code);
    std::exit(EXIT_FAILURE);
}

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

#define CUBLAS_CHECK(expr)                                                       \
    do {                                                                        \
        cublasStatus_t status_ = (expr);                                         \
        if (status_ != CUBLAS_STATUS_SUCCESS)                                    \
            fail("cuBLASLt", #expr, __LINE__, static_cast<int>(status_));       \
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
    size_t workspace_bytes = 64ull << 20;
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

__global__ void compare_half_kernel(const __half* reference, const __half* candidate,
                                    size_t count, float atol, float rtol,
                                    unsigned int* max_abs_bits,
                                    unsigned int* max_ref_bits,
                                    unsigned long long* mismatch_count) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float ref = __half2float(reference[index]);
    float got = __half2float(candidate[index]);
    float diff = fabsf(ref - got);
    atomicMax(max_abs_bits, __float_as_uint(diff));
    atomicMax(max_ref_bits, __float_as_uint(fabsf(ref)));
    if (diff > atol + rtol * fabsf(ref)) atomicAdd(mismatch_count, 1ull);
}

std::vector<__half> make_sparse_a(int m, int k) {
    static constexpr int masks[6][2] = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3},
    };
    std::vector<__half> values(static_cast<size_t>(m) * k, __float2half(0.0f));
    for (int row = 0; row < m; ++row) {
        for (int group = 0; group < k / 4; ++group) {
            int mask = static_cast<int>(mix32(static_cast<uint32_t>(row * 65537 + group)) % 6u);
            for (int slot = 0; slot < 2; ++slot) {
                int column = 4 * group + masks[mask][slot];
                uint32_t bits = mix32(static_cast<uint32_t>(row * k + column + 17));
                float value = static_cast<float>(bits & 0xffffu) / 32768.0f - 1.0f;
                if (std::fabs(value) < 0.05f) value = value < 0.0f ? -0.05f : 0.05f;
                values[static_cast<size_t>(row) * k + column] = __float2half(value);
            }
        }
    }
    return values;
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

struct DensePlan {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t data_layout = nullptr;
    cublasLtMatrixLayout_t weight_layout = nullptr;
    cublasLtMatrixLayout_t output_layout = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    bool has_algorithm = false;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
};

void destroy_dense(DensePlan& plan) {
    if (plan.workspace) cudaFree(plan.workspace);
    if (plan.output_layout) cublasLtMatrixLayoutDestroy(plan.output_layout);
    if (plan.weight_layout) cublasLtMatrixLayoutDestroy(plan.weight_layout);
    if (plan.data_layout) cublasLtMatrixLayoutDestroy(plan.data_layout);
    if (plan.operation) cublasLtMatmulDescDestroy(plan.operation);
    if (plan.handle) cublasLtDestroy(plan.handle);
}

DensePlan make_dense_plan(const Options& options, const __half* data,
                          const __half* weights, __half* output) {
    DensePlan plan;
    plan.workspace_bytes = options.workspace_bytes;
    CUBLAS_CHECK(cublasLtCreate(&plan.handle));
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan.operation, CUBLAS_COMPUTE_32F,
                                          CUDA_R_32F));
    cublasOperation_t transposed = CUBLAS_OP_T;
    cublasOperation_t normal = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        plan.operation, CUBLASLT_MATMUL_DESC_TRANSA, &transposed, sizeof(transposed)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        plan.operation, CUBLASLT_MATMUL_DESC_TRANSB, &normal, sizeof(normal)));

    // Row-major data [N,K] is column-major [K,N]. Row-major weights [M,K]
    // are column-major [K,M]. Compute C_col[N,M] = data^T[N,K] * weight[K,M].
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.data_layout, CUDA_R_16F,
                                            options.k, options.n, options.k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.weight_layout, CUDA_R_16F,
                                            options.k, options.m, options.k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.output_layout, CUDA_R_16F,
                                            options.n, options.m, options.n));
    int32_t batch = options.batch;
    int64_t data_stride = static_cast<int64_t>(options.n) * options.k;
    int64_t weight_stride = 0;
    int64_t output_stride = static_cast<int64_t>(options.m) * options.n;
    for (auto layout : {plan.data_layout, plan.weight_layout, plan.output_layout}) {
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
            layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
    }
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        plan.data_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &data_stride, sizeof(data_stride)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        plan.weight_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &weight_stride, sizeof(weight_stride)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        plan.output_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &output_stride, sizeof(output_stride)));

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
        plan.handle, plan.operation, plan.data_layout, plan.weight_layout,
        plan.output_layout, plan.output_layout, preference, max_algorithms,
        results, &returned));
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    if (returned <= 0) {
        std::fprintf(stderr, "cuBLASLt returned no heuristic algorithms\n");
        std::exit(EXIT_FAILURE);
    }

    float alpha = 1.0f, beta = 0.0f;
    float best_ms = std::numeric_limits<float>::infinity();
    for (int index = 0; index < returned; ++index) {
        auto call = [&]() {
            CUBLAS_CHECK(cublasLtMatmul(
                plan.handle, plan.operation, &alpha, data, plan.data_layout,
                weights, plan.weight_layout, &beta, output, plan.output_layout,
                output, plan.output_layout, &results[index].algo, plan.workspace,
                plan.workspace_bytes, nullptr));
        };
        float ms = time_callable(2, 8, call);
        if (ms < best_ms) {
            best_ms = ms;
            plan.algorithm = results[index].algo;
            plan.has_algorithm = true;
        }
    }
    return plan;
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
    auto host_weights = make_sparse_a(options.m, options.k);

    __half *weights = nullptr, *data = nullptr, *dense_output = nullptr,
           *sparse_output = nullptr;
    CUDA_CHECK(cudaMalloc(&weights, weight_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&data, data_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dense_output, output_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&sparse_output, output_count * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(weights, host_weights.data(), weight_count * sizeof(__half),
                          cudaMemcpyHostToDevice));
    int threads = 256;
    int blocks = static_cast<int>((data_count + threads - 1) / threads);
    fill_half_kernel<<<blocks, threads>>>(data, data_count, 0x12345678u);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    DensePlan dense = make_dense_plan(options, data, weights, dense_output);
    SparsePlan sparse = make_sparse_plan(options, weights, data, sparse_output);
    float alpha = 1.0f, beta = 0.0f;
    auto dense_call = [&]() {
        CUBLAS_CHECK(cublasLtMatmul(
            dense.handle, dense.operation, &alpha, data, dense.data_layout,
            weights, dense.weight_layout, &beta, dense_output,
            dense.output_layout, dense_output, dense.output_layout,
            dense.has_algorithm ? &dense.algorithm : nullptr, dense.workspace,
            dense.workspace_bytes, nullptr));
    };
    auto sparse_call = [&]() {
        CUSPARSELT_CHECK(cusparseLtMatmul(
            &sparse.handle, &sparse.plan, &alpha, sparse.compressed, data,
            &beta, sparse_output, sparse_output, sparse.workspace, nullptr, 0));
    };

    dense_call();
    sparse_call();
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned int *max_abs_bits = nullptr, *max_ref_bits = nullptr;
    unsigned long long* mismatch_count = nullptr;
    CUDA_CHECK(cudaMalloc(&max_abs_bits, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&max_ref_bits, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&mismatch_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(max_abs_bits, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(max_ref_bits, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(mismatch_count, 0, sizeof(unsigned long long)));
    blocks = static_cast<int>((output_count + threads - 1) / threads);
    compare_half_kernel<<<blocks, threads>>>(
        dense_output, sparse_output, output_count, 0.125f, 0.01f,
        max_abs_bits, max_ref_bits, mismatch_count);
    CUDA_CHECK(cudaGetLastError());
    unsigned int host_max_abs_bits = 0, host_max_ref_bits = 0;
    unsigned long long host_mismatches = 0;
    CUDA_CHECK(cudaMemcpy(&host_max_abs_bits, max_abs_bits, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&host_max_ref_bits, max_ref_bits, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&host_mismatches, mismatch_count,
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    float max_abs = 0.0f, max_ref = 0.0f;
    std::memcpy(&max_abs, &host_max_abs_bits, sizeof(float));
    std::memcpy(&max_ref, &host_max_ref_bits, sizeof(float));
    bool correct = host_mismatches == 0;
    if (!correct) {
        std::fprintf(stderr,
                     "correctness failed: mismatches=%llu max_abs=%g max_ref=%g\n",
                     host_mismatches, max_abs, max_ref);
        return EXIT_FAILURE;
    }

    float dense_ms = 0.0f, sparse_ms = 0.0f;
    if (options.order == "AB") {
        dense_ms = time_callable(options.warmup, options.iterations, dense_call);
        sparse_ms = time_callable(options.warmup, options.iterations, sparse_call);
    } else {
        sparse_ms = time_callable(options.warmup, options.iterations, sparse_call);
        dense_ms = time_callable(options.warmup, options.iterations, dense_call);
    }
    double logical_flops = 2.0 * options.batch * options.m * options.n * options.k;
    double dense_tflops = logical_flops / (dense_ms * 1.0e9);
    double sparse_effective_tflops = logical_flops / (sparse_ms * 1.0e9);

    int driver = 0, runtime = 0, cusparselt_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUSPARSELT_CHECK(cusparseLtGetVersion(&sparse.handle, &cusparselt_version));
    std::printf(
        "{\"kind\":\"real_gemm_consumer_screen\",\"gpu\":\"%s\","
        "\"cc\":\"%d.%d\",\"driver\":%d,\"runtime\":%d,"
        "\"cusparselt\":%d,\"m\":%d,\"n\":%d,\"k\":%d,"
        "\"batch\":%d,\"warmup\":%d,\"iterations\":%d,"
        "\"order\":\"%s\",\"dense_ms\":%.9g,\"sparse_ms\":%.9g,"
        "\"speedup\":%.9g,\"dense_effective_tflops\":%.9g,"
        "\"sparse_effective_tflops\":%.9g,\"correct\":true,"
        "\"max_abs_diff\":%.9g,\"max_reference_abs\":%.9g,"
        "\"mismatch_count\":0,\"output_elements\":%zu,"
        "\"sparse_compressed_bytes\":%zu,\"sparse_workspace_bytes\":%zu}\n",
        properties.name, properties.major, properties.minor, driver, runtime,
        cusparselt_version, options.m, options.n, options.k, options.batch,
        options.warmup, options.iterations, options.order.c_str(), dense_ms,
        sparse_ms, dense_ms / sparse_ms, dense_tflops, sparse_effective_tflops,
        max_abs, max_ref, output_count, sparse.compressed_bytes,
        sparse.workspace_bytes);

    CUDA_CHECK(cudaFree(mismatch_count));
    CUDA_CHECK(cudaFree(max_ref_bits));
    CUDA_CHECK(cudaFree(max_abs_bits));
    destroy_sparse(sparse);
    destroy_dense(dense);
    CUDA_CHECK(cudaFree(sparse_output));
    CUDA_CHECK(cudaFree(dense_output));
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(weights));
    return EXIT_SUCCESS;
}
