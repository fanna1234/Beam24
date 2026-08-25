#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <ccglib/ccglib.hpp>
#include <ccglib/common/helper.h>
#include <cudawrappers/cu.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

__global__ void power_reduce_planar_bcmn(
    const float* __restrict__ complex_output,
    float* __restrict__ power,
    size_t m, size_t n) {
    const size_t row_linear = blockIdx.x;
    const size_t batch_index = row_linear / m;
    const size_t row = row_linear - batch_index * m;
    const size_t plane_size = m * n;
    const float* real = complex_output + (batch_index * 2 + 0) * plane_size + row * n;
    const float* imag = complex_output + (batch_index * 2 + 1) * plane_size + row * n;
    float sum = 0.0f;
    for (size_t column = threadIdx.x; column < n; column += blockDim.x) {
        const float r = real[column];
        const float i = imag[column];
        sum = fmaf(r, r, sum);
        sum = fmaf(i, i, sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    __shared__ float warp_sums[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) power[row_linear] = sum;
    }
}

__device__ __forceinline__ bool better_power(float candidate_value,
                                              uint32_t candidate_index,
                                              float current_value,
                                              uint32_t current_index) {
    return candidate_value > current_value ||
           (candidate_value == current_value && candidate_index < current_index);
}

__global__ void argmax_power_kernel(
    const float* __restrict__ power,
    float* __restrict__ max_power,
    uint32_t* __restrict__ max_index,
    int M) {
    const int batch_index = blockIdx.x;
    const float* row = power + static_cast<size_t>(batch_index) * M;
    float best_value = -CUDART_INF_F;
    uint32_t best_index = 0xffffffffu;
    for (int index = threadIdx.x; index < M; index += blockDim.x) {
        const float value = row[index];
        if (better_power(value, static_cast<uint32_t>(index), best_value, best_index)) {
            best_value = value;
            best_index = static_cast<uint32_t>(index);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_value = __shfl_down_sync(0xffffffffu, best_value, offset);
        const uint32_t other_index = __shfl_down_sync(0xffffffffu, best_index, offset);
        if (better_power(other_value, other_index, best_value, best_index)) {
            best_value = other_value;
            best_index = other_index;
        }
    }
    __shared__ float warp_values[8];
    __shared__ uint32_t warp_indices[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_values[warp] = best_value;
        warp_indices[warp] = best_index;
    }
    __syncthreads();
    if (warp == 0) {
        best_value = lane < 8 ? warp_values[lane] : -CUDART_INF_F;
        best_index = lane < 8 ? warp_indices[lane] : 0xffffffffu;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_value = __shfl_down_sync(0xffffffffu, best_value, offset);
            const uint32_t other_index = __shfl_down_sync(0xffffffffu, best_index, offset);
            if (better_power(other_value, other_index, best_value, best_index)) {
                best_value = other_value;
                best_index = other_index;
            }
        }
        if (lane == 0) {
            max_power[batch_index] = best_value;
            max_index[batch_index] = best_index;
        }
    }
}

const float* device_float_ptr(const cu::DeviceMemory& memory) {
    return reinterpret_cast<const float*>(static_cast<CUdeviceptr>(memory));
}

float* device_float_ptr(cu::DeviceMemory& memory) {
    return reinterpret_cast<float*>(static_cast<CUdeviceptr>(memory));
}

cudaStream_t runtime_stream(cu::Stream& stream) {
    return reinterpret_cast<cudaStream_t>(static_cast<CUstream>(stream));
}

struct Options {
    size_t batch = 1;
    size_t m = 1024;
    size_t n = 1024;
    size_t k = 512;
    size_t warmup = 20;
    size_t iterations = 100;
    std::string mode = "basic";
    bool check = false;
};

void launch_power(cu::DeviceMemory& complex_output, cu::DeviceMemory& power,
                  const Options& options, cu::Stream& stream) {
    const size_t rows = options.batch * options.m;
    power_reduce_planar_bcmn<<<static_cast<unsigned int>(rows), 256, 0,
                               runtime_stream(stream)>>>(
        device_float_ptr(complex_output), device_float_ptr(power),
        options.m, options.n);
}

void launch_argmax(cu::DeviceMemory& power, cu::DeviceMemory& max_power,
                   cu::DeviceMemory& max_index, const Options& options,
                   cu::Stream& stream) {
    argmax_power_kernel<<<static_cast<unsigned int>(options.batch), 256, 0,
                           runtime_stream(stream)>>>(
        device_float_ptr(power), device_float_ptr(max_power),
        reinterpret_cast<uint32_t*>(static_cast<CUdeviceptr>(max_index)),
        static_cast<int>(options.m));
}

size_t parse_size(const char* text, const char* name) {
    char* end = nullptr;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (!end || *end != '\0' || value == 0) {
        std::fprintf(stderr, "invalid %s: %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<size_t>(value);
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
            options.batch = parse_size(need("--batch"), "batch");
        else if (std::strcmp(argv[index], "--m") == 0)
            options.m = parse_size(need("--m"), "m");
        else if (std::strcmp(argv[index], "--n") == 0)
            options.n = parse_size(need("--n"), "n");
        else if (std::strcmp(argv[index], "--k") == 0)
            options.k = parse_size(need("--k"), "k");
        else if (std::strcmp(argv[index], "--warmup") == 0)
            options.warmup = parse_size(need("--warmup"), "warmup");
        else if (std::strcmp(argv[index], "--iterations") == 0)
            options.iterations = parse_size(need("--iterations"), "iterations");
        else if (std::strcmp(argv[index], "--mode") == 0)
            options.mode = need("--mode");
        else if (std::strcmp(argv[index], "--check") == 0)
            options.check = true;
        else {
            std::fprintf(stderr, "unknown option: %s\n", argv[index]);
            std::exit(EXIT_FAILURE);
        }
    }
    if (options.mode != "basic" && options.mode != "opt_static_a" &&
        options.mode != "opt_full_pipeline") {
        std::fprintf(stderr, "mode must be basic, opt_static_a, or opt_full_pipeline\n");
        std::exit(EXIT_FAILURE);
    }
    return options;
}

float time_callable(cu::Stream& stream, size_t warmup, size_t iterations,
                    const std::function<void()>& callable) {
    for (size_t index = 0; index < warmup; ++index) callable();
    stream.synchronize();
    cu::Event start, stop;
    stream.record(start);
    for (size_t index = 0; index < iterations; ++index) callable();
    stream.record(stop);
    stream.synchronize();
    return stop.elapsedTime(start) / static_cast<float>(iterations);
}

struct Path {
    std::function<void()> run;
    float static_a_ms = 0.0f;
    std::unique_ptr<ccglib::mma::GEMM> gemm;
    std::unique_ptr<ccglib::pipeline::Pipeline> pipeline;
    std::unique_ptr<ccglib::transpose::Transpose> transpose_a;
    std::unique_ptr<ccglib::transpose::Transpose> transpose_b;
    std::unique_ptr<cu::DeviceMemory> a_tiled;
    std::unique_ptr<cu::DeviceMemory> b_tiled;
};

Path make_path(const Options& options, cu::Device& device, cu::Stream& stream,
               cu::DeviceMemory& a, cu::DeviceMemory& b,
               cu::DeviceMemory& c) {
    const ccglib::Precision precision(ccglib::ValueType::float16,
                                      ccglib::ValueType::float32);
    Path path;
    if (options.mode == "basic") {
        path.gemm = std::make_unique<ccglib::mma::GEMM>(
            options.batch, options.m, options.n, options.k, device, stream,
            precision, ccglib::mma::basic, ccglib::complex_planar,
            ccglib::mma::row_major, ccglib::mma::row_major,
            ccglib::mma::col_major);
        auto* gemm = path.gemm.get();
        path.run = [gemm, &a, &b, &c]() { gemm->Run(a, b, c); };
        return path;
    }

    if (options.mode == "opt_full_pipeline") {
        path.pipeline = std::make_unique<ccglib::pipeline::Pipeline>(
            options.batch, options.m, options.n, options.k, device, stream,
            ccglib::complex_planar, ccglib::complex_planar,
            ccglib::mma::row_major, ccglib::mma::col_major,
            ccglib::mma::row_major, ccglib::ValueType::float16,
            ccglib::ValueType::float32, ccglib::mma::opt);
        auto* pipeline = path.pipeline.get();
        path.run = [pipeline, &a, &b, &c]() { pipeline->Run(a, b, c); };
        return path;
    }

    const dim3 dimensions = ccglib::mma::GEMM::GetDimensions(
        precision, ccglib::mma::opt);
    const size_t m_padded = dimensions.x * ccglib::helper::ceildiv(options.m, size_t(dimensions.x));
    const size_t n_padded = dimensions.y * ccglib::helper::ceildiv(options.n, size_t(dimensions.y));
    const size_t k_padded = dimensions.z * ccglib::helper::ceildiv(options.k, size_t(dimensions.z));
    constexpr size_t complex = 2;
    constexpr size_t input_bytes = 2;
    path.a_tiled = std::make_unique<cu::DeviceMemory>(
        options.batch * complex * m_padded * k_padded * input_bytes);
    path.b_tiled = std::make_unique<cu::DeviceMemory>(
        options.batch * complex * n_padded * k_padded * input_bytes);
    path.transpose_a = std::make_unique<ccglib::transpose::Transpose>(
        options.batch, options.m, options.k, dimensions.x, dimensions.z,
        16, device, stream, ccglib::complex_planar);
    path.transpose_b = std::make_unique<ccglib::transpose::Transpose>(
        options.batch, options.n, options.k, dimensions.y, dimensions.z,
        16, device, stream, ccglib::complex_planar);
    path.gemm = std::make_unique<ccglib::mma::GEMM>(
        options.batch, options.m, options.n, options.k, device, stream,
        precision, ccglib::mma::opt, ccglib::complex_planar,
        ccglib::mma::row_major, ccglib::mma::row_major,
        ccglib::mma::col_major);

    cu::Event start, stop;
    stream.record(start);
    path.transpose_a->Run(a, *path.a_tiled);
    stream.record(stop);
    stream.synchronize();
    path.static_a_ms = stop.elapsedTime(start);
    auto* transpose_b = path.transpose_b.get();
    auto* gemm = path.gemm.get();
    auto* a_tiled = path.a_tiled.get();
    auto* b_tiled = path.b_tiled.get();
    path.run = [transpose_b, gemm, a_tiled, b_tiled, &b, &c]() {
        transpose_b->Run(b, *b_tiled);
        gemm->Run(*a_tiled, *b_tiled, c);
    };
    return path;
}

void fill_inputs(const Options& options, cu::DeviceMemory& a,
                 cu::DeviceMemory& b, cu::Stream& stream) {
    if (!options.check) {
        stream.memsetAsync(a, static_cast<unsigned char>(0x5a), a.size());
        stream.memsetAsync(b, static_cast<unsigned char>(0xa5), b.size());
        stream.synchronize();
        return;
    }
    std::mt19937 generator(20260823u);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<half> host_a(a.size() / sizeof(half));
    std::vector<half> host_b(b.size() / sizeof(half));
    for (half& value : host_a) value = __float2half(distribution(generator));
    for (half& value : host_b) value = __float2half(distribution(generator));
    cu::memcpyHtoD(a, host_a.data(), a.size());
    cu::memcpyHtoD(b, host_b.data(), b.size());
}

void check_against_basic(const Options& options, cu::Device& device,
                         cu::Stream& stream, cu::DeviceMemory& a,
                         cu::DeviceMemory& b, cu::DeviceMemory& output) {
    const size_t output_values = options.batch * 2 * options.m * options.n;
    cu::DeviceMemory reference(output_values * sizeof(float));
    const ccglib::Precision precision(ccglib::ValueType::float16,
                                      ccglib::ValueType::float32);
    ccglib::mma::GEMM basic(
        options.batch, options.m, options.n, options.k, device, stream,
        precision, ccglib::mma::basic, ccglib::complex_planar,
        ccglib::mma::row_major, ccglib::mma::row_major,
        ccglib::mma::col_major);
    basic.Run(a, b, reference);
    stream.synchronize();
    std::vector<float> host_reference(output_values);
    std::vector<float> host_output(output_values);
    cu::memcpyDtoH(host_reference.data(), reference, reference.size());
    cu::memcpyDtoH(host_output.data(), output, output.size());
    float max_abs = 0.0f;
    size_t bad = 0;
    for (size_t index = 0; index < output_values; ++index) {
        float difference = std::fabs(host_reference[index] - host_output[index]);
        max_abs = std::max(max_abs, difference);
        if (difference > 0.02f) ++bad;
    }
    std::printf("{\"kind\":\"ccglib_baseline_correctness\","
                "\"mode\":\"%s\",\"max_abs_diff\":%.9g,"
                "\"bad\":%zu,\"values\":%zu,\"pass\":%s}\n",
                options.mode.c_str(), max_abs, bad, output_values,
                bad == 0 ? "true" : "false");
    if (bad != 0) std::exit(EXIT_FAILURE);
}

void check_power_output(const Options& options, cu::DeviceMemory& complex_output,
                        cu::DeviceMemory& power) {
    const size_t plane_size = options.m * options.n;
    const size_t complex_values = options.batch * 2 * plane_size;
    const size_t power_values = options.batch * options.m;
    std::vector<float> host_complex(complex_values);
    std::vector<float> host_power(power_values);
    cu::memcpyDtoH(host_complex.data(), complex_output, complex_output.size());
    cu::memcpyDtoH(host_power.data(), power, power.size());
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t bad = 0;
    for (size_t batch_index = 0; batch_index < options.batch; ++batch_index) {
        const float* real = host_complex.data() + (batch_index * 2 + 0) * plane_size;
        const float* imag = host_complex.data() + (batch_index * 2 + 1) * plane_size;
        for (size_t row = 0; row < options.m; ++row) {
            double expected = 0.0;
            for (size_t column = 0; column < options.n; ++column) {
                const size_t index = row * options.n + column;
                expected += static_cast<double>(real[index]) * real[index]
                          + static_cast<double>(imag[index]) * imag[index];
            }
            const float observed = host_power[batch_index * options.m + row];
            const float difference = std::fabs(observed - static_cast<float>(expected));
            const float relative = difference / std::max(std::fabs(static_cast<float>(expected)), 1.0e-6f);
            max_abs = std::max(max_abs, difference);
            max_rel = std::max(max_rel, relative);
            if (difference > 0.05f + 1.0e-4f * std::fabs(static_cast<float>(expected))) ++bad;
        }
    }
    std::printf("{\"kind\":\"beampower_correctness\","
                "\"max_abs_diff\":%.9g,\"max_rel_diff\":%.9g,"
                "\"bad\":%zu,\"values\":%zu,\"pass\":%s}\n",
                max_abs, max_rel, bad, power_values,
                bad == 0 ? "true" : "false");
    if (bad != 0) std::exit(EXIT_FAILURE);
}

void check_doa_output(const Options& options, cu::DeviceMemory& power,
                      cu::DeviceMemory& max_power, cu::DeviceMemory& max_index) {
    std::vector<float> host_power(options.batch * options.m);
    std::vector<float> host_max_power(options.batch);
    std::vector<uint32_t> host_max_index(options.batch);
    cu::memcpyDtoH(host_power.data(), power, power.size());
    cu::memcpyDtoH(host_max_power.data(), max_power, max_power.size());
    cu::memcpyDtoH(host_max_index.data(), max_index, max_index.size());
    size_t bad = 0;
    float max_abs = 0.0f;
    for (size_t batch_index = 0; batch_index < options.batch; ++batch_index) {
        float expected_value = -std::numeric_limits<float>::infinity();
        uint32_t expected_index = 0xffffffffu;
        for (size_t row = 0; row < options.m; ++row) {
            const float value = host_power[batch_index * options.m + row];
            if (value > expected_value ||
                (value == expected_value && row < expected_index)) {
                expected_value = value;
                expected_index = static_cast<uint32_t>(row);
            }
        }
        max_abs = std::max(max_abs, std::fabs(host_max_power[batch_index] - expected_value));
        if (host_max_index[batch_index] != expected_index ||
            std::fabs(host_max_power[batch_index] - expected_value) >
                1.0e-5f * std::max(std::fabs(expected_value), 1.0f)) {
            ++bad;
        }
    }
    std::printf("{\"kind\":\"doa_correctness\","
                "\"max_power_abs\":%.9g,\"bad\":%zu,"
                "\"values\":%zu,\"pass\":%s}\n",
                max_abs, bad, options.batch, bad == 0 ? "true" : "false");
    if (bad != 0) std::exit(EXIT_FAILURE);
}

}  // namespace

int main(int argc, char** argv) {
    Options options = parse_options(argc, argv);
    cu::init();
    cu::Device device(0);
    cu::Context context(CU_CTX_BLOCKING_SYNC, device);
    cu::Stream stream;
    constexpr size_t complex = 2;
    const size_t a_bytes = options.batch * complex * options.m * options.k * sizeof(half);
    const size_t b_bytes = options.batch * complex * options.n * options.k * sizeof(half);
    const size_t c_bytes = options.batch * complex * options.m * options.n * sizeof(float);
    const size_t power_bytes = options.batch * options.m * sizeof(float);
    cu::DeviceMemory a(a_bytes), b(b_bytes), c(c_bytes), power(power_bytes),
        max_power(options.batch * sizeof(float)),
        max_index(options.batch * sizeof(uint32_t));
    fill_inputs(options, a, b, stream);
    c.zero(c_bytes);
    power.zero(power_bytes);
    Path path = make_path(options, device, stream, a, b, c);
    path.run();
    launch_power(c, power, options, stream);
    launch_argmax(power, max_power, max_index, options, stream);
    stream.synchronize();
    if (options.check) {
        check_against_basic(options, device, stream, a, b, c);
        check_power_output(options, c, power);
        check_doa_output(options, power, max_power, max_index);
    }
    const std::function<void()> system_run = [&]() {
        path.run();
        launch_power(c, power, options, stream);
        launch_argmax(power, max_power, max_index, options, stream);
    };
    float milliseconds = time_callable(
        stream, options.warmup, options.iterations, system_run);
    double useful_ops = 8.0 * options.batch * options.m * options.n * options.k;
    double tops = useful_ops / (milliseconds * 1.0e9);
    std::printf(
        "{\"kind\":\"ccglib_doa_e2e\",\"mode\":\"%s\","
        "\"batch\":%zu,\"m\":%zu,\"n\":%zu,\"k\":%zu,"
        "\"warmup\":%zu,\"iterations\":%zu,\"milliseconds\":%.9g,"
        "\"effective_complex_tops\":%.9g,\"static_a_ms\":%.9g,"
        "\"power_values\":%zu,\"doa_values\":%zu,\"check\":%s}\n",
        options.mode.c_str(), options.batch, options.m, options.n, options.k,
        options.warmup, options.iterations, milliseconds, tops,
        path.static_a_ms, options.batch * options.m, options.batch,
        options.check ? "true" : "false");
    return EXIT_SUCCESS;
}
