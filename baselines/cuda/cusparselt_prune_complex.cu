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
    std::string prune = "strip";
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
        else if (std::strcmp(argv[i], "--prune") == 0)
            options.prune = need("--prune");
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
    if (options.prune != "strip" && options.prune != "tile") {
        std::fprintf(stderr, "prune must be strip or tile\n");
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

ComplexWeights make_dense_complex_weights(int m, int k) {
    ComplexWeights weights{
        std::vector<__half>(static_cast<size_t>(m) * k, __float2half(0.0f)),
        std::vector<__half>(static_cast<size_t>(m) * k, __float2half(0.0f))};
    for (int row = 0; row < m; ++row) {
        for (int column = 0; column < k; ++column) {
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
    return weights;
}

void prune_strip_cpu(std::vector<__half>& values, int m, int k) {
    for (int row = 0; row < m; ++row) {
        for (int column = 0; column < k; column += 4) {
            int order[4] = {0, 1, 2, 3};
            std::sort(order, order + 4, [&](int left, int right) {
                return std::fabs(__half2float(values[static_cast<size_t>(row) * k + column + left])) >
                       std::fabs(__half2float(values[static_cast<size_t>(row) * k + column + right]));
            });
            bool keep[4] = {false, false, false, false};
            keep[order[0]] = true;
            keep[order[1]] = true;
            for (int slot = 0; slot < 4; ++slot) {
                if (!keep[slot]) {
                    values[static_cast<size_t>(row) * k + column + slot] = __float2half(0.0f);
                }
            }
        }
    }
}

void prune_tile_cpu(std::vector<__half>& values, int m, int k) {
    static constexpr int masks[6][2] = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3},
    };
    for (int row0 = 0; row0 < m; row0 += 4) {
        for (int column0 = 0; column0 < k; column0 += 4) {
            float best_score = -1.0f;
            int best[4] = {0, 0, 0, 0};
            for (int a = 0; a < 6; ++a) {
                for (int b = 0; b < 6; ++b) {
                    for (int c = 0; c < 6; ++c) {
                        for (int d = 0; d < 6; ++d) {
                            int choices[4] = {a, b, c, d};
                            int column_counts[4] = {0, 0, 0, 0};
                            float score = 0.0f;
                            for (int row = 0; row < 4; ++row) {
                                for (int slot = 0; slot < 2; ++slot) {
                                    int column = masks[choices[row]][slot];
                                    ++column_counts[column];
                                    size_t index = static_cast<size_t>(row0 + row) * k + column0 + column;
                                    score += std::fabs(__half2float(values[index]));
                                }
                            }
                            if (column_counts[0] != 2 || column_counts[1] != 2 ||
                                column_counts[2] != 2 || column_counts[3] != 2) continue;
                            if (score > best_score) {
                                best_score = score;
                                for (int row = 0; row < 4; ++row) best[row] = choices[row];
                            }
                        }
                    }
                }
            }
            for (int row = 0; row < 4; ++row) {
                bool keep[4] = {false, false, false, false};
                keep[masks[best[row]][0]] = true;
                keep[masks[best[row]][1]] = true;
                for (int column = 0; column < 4; ++column) {
                    if (!keep[column]) {
                        values[static_cast<size_t>(row0 + row) * k + column0 + column] =
                            __float2half(0.0f);
                    }
                }
            }
        }
    }
}

void check_prune_oracle(const Options& options, ComplexWeights expected,
                        const ComplexWeights& actual) {
    auto prune = options.prune == "strip" ? prune_strip_cpu : prune_tile_cpu;
    prune(expected.real, options.m, options.k);
    prune(expected.imag, options.m, options.k);
    size_t mismatches = 0;
    for (size_t index = 0; index < expected.real.size(); ++index) {
        mismatches += __half_as_ushort(expected.real[index]) != __half_as_ushort(actual.real[index]);
        mismatches += __half_as_ushort(expected.imag[index]) != __half_as_ushort(actual.imag[index]);
    }
    std::printf(
        "{\"kind\":\"cusparselt_prune_oracle\",\"algorithm\":\"%s\","
        "\"mismatches\":%zu,\"pass\":%s}\n",
        options.prune.c_str(), mismatches, mismatches == 0 ? "true" : "false");
    if (mismatches != 0) std::exit(EXIT_FAILURE);
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

SparsePlan make_pruned_sparse_plan(const Options& options, __half* weights,
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

    cusparseLtPruneAlg_t prune_algorithm = options.prune == "strip"
        ? CUSPARSELT_PRUNE_SPMMA_STRIP
        : CUSPARSELT_PRUNE_SPMMA_TILE;
    CUSPARSELT_CHECK(cusparseLtSpMMAPrune(
        &sparse.handle, &sparse.operation, weights, weights,
        prune_algorithm, nullptr));

    int* valid = nullptr;
    CUDA_CHECK(cudaMalloc(&valid, sizeof(int)));
    CUSPARSELT_CHECK(cusparseLtSpMMAPruneCheck(
        &sparse.handle, &sparse.operation, weights, valid, nullptr));
    int host_valid = -1;
    CUDA_CHECK(cudaMemcpy(&host_valid, valid, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(valid));
    if (host_valid != 0) {
        std::fprintf(stderr, "cuSPARSELt pruning failed exact 2:4 legality check\n");
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

void check_pruned(const Options& options, const ComplexWeights& weights,
                  const __half* input_real, const __half* input_imag,
                  const float* output_real, const float* output_imag) {
    size_t input_count = static_cast<size_t>(options.batch) * options.n * options.k;
    size_t output_count = static_cast<size_t>(options.batch) * options.m * options.n;
    if (static_cast<unsigned long long>(options.batch) * options.m * options.n *
            options.k > 100000000ull) {
        std::fprintf(stderr, "--check is limited to 100M complex MACs\n");
        std::exit(EXIT_FAILURE);
    }
    std::vector<__half> xr(input_count), xi(input_count);
    std::vector<float> yr(output_count), yi(output_count);
    CUDA_CHECK(cudaMemcpy(xr.data(), input_real, input_count * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(xi.data(), input_imag, input_count * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(yr.data(), output_real, output_count * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(yi.data(), output_imag, output_count * sizeof(float),
                          cudaMemcpyDeviceToHost));

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
                    float xrv = __half2float(xr[data_index]);
                    float xiv = __half2float(xi[data_index]);
                    reference_real += ur * xrv + ui * xiv;
                    reference_imag += ur * xiv - ui * xrv;
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
    bool passed = bad == 0;
    std::printf(
        "{\"kind\":\"cusparselt_prune_complex_correctness\","
        "\"gemm_max_abs_diff\":%.9g,"
        "\"bad\":%llu,\"pass\":%s}\n",
        gemm_max_abs, bad, passed ? "true" : "false");
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
    ComplexWeights host_weights = make_dense_complex_weights(options.m, options.k);
    ComplexWeights dense_host_weights = host_weights;

    __half *weight_real = nullptr, *weight_imag = nullptr;
    __half *input_real = nullptr, *input_imag = nullptr;
    __half *half_output_real = nullptr, *half_output_imag = nullptr;
    float *output_real = nullptr, *output_imag = nullptr;
    CUDA_CHECK(cudaMalloc(&weight_real, weight_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&weight_imag, weight_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&input_real, data_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&input_imag, data_count * sizeof(__half)));
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
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    SparsePlan real_plan = make_pruned_sparse_plan(
        options, weight_real, input_real, half_output_real);
    SparsePlan imag_plan = make_pruned_sparse_plan(
        options, weight_imag, input_imag, half_output_real);
    CUDA_CHECK(cudaMemcpy(host_weights.real.data(), weight_real,
                          weight_count * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_weights.imag.data(), weight_imag,
                          weight_count * sizeof(__half), cudaMemcpyDeviceToHost));
    if (options.check) check_prune_oracle(options, dense_host_weights, host_weights);
    int output_blocks = static_cast<int>((output_count + threads - 1) / threads);
    auto composed_call = [&]() {
        sparse_gemm(real_plan, input_real, half_output_real, 1.0f, 0.0f);
        sparse_gemm(imag_plan, input_imag, half_output_real, 1.0f, 1.0f);
        sparse_gemm(real_plan, input_imag, half_output_imag, 1.0f, 0.0f);
        sparse_gemm(imag_plan, input_real, half_output_imag, -1.0f, 1.0f);
        promote_complex_kernel<<<output_blocks, threads>>>(
            half_output_real, half_output_imag, output_real, output_imag, output_count);
    };

    composed_call();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (options.check) {
        check_pruned(options, host_weights, input_real, input_imag,
                     output_real, output_imag);
    }

    float milliseconds = time_callable(options.warmup, options.iterations, composed_call);
    double useful_ops = 8.0 * options.batch * options.m * options.n * options.k;
    double tops = useful_ops / (milliseconds * 1.0e9);
    int driver = 0, runtime = 0, cusparselt_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUSPARSELT_CHECK(cusparseLtGetVersion(&real_plan.handle, &cusparselt_version));
    std::printf(
        "{\"kind\":\"cusparselt_prune_complex\",\"gpu\":\"%s\","
        "\"cc\":\"%d.%d\",\"driver\":%d,\"runtime\":%d,"
        "\"cusparselt\":%d,\"batch\":%d,\"m\":%d,\"n\":%d,\"k\":%d,"
        "\"warmup\":%d,\"iterations\":%d,\"milliseconds\":%.9g,"
        "\"effective_complex_tops\":%.9g,\"prune\":\"%s\","
        "\"sparse_gemm_calls\":4,\"intermediate\":\"fp16\","
        "\"output\":\"fp32\",\"check\":%s}\n",
        properties.name, properties.major, properties.minor, driver, runtime,
        cusparselt_version, options.batch, options.m, options.n, options.k,
        options.warmup, options.iterations, milliseconds, tops,
        options.prune.c_str(),
        options.check ? "true" : "false");

    destroy_sparse(imag_plan);
    destroy_sparse(real_plan);
    CUDA_CHECK(cudaFree(output_imag));
    CUDA_CHECK(cudaFree(output_real));
    CUDA_CHECK(cudaFree(half_output_imag));
    CUDA_CHECK(cudaFree(half_output_real));
    CUDA_CHECK(cudaFree(input_imag));
    CUDA_CHECK(cudaFree(input_real));
    CUDA_CHECK(cudaFree(weight_imag));
    CUDA_CHECK(cudaFree(weight_real));
    return EXIT_SUCCESS;
}
