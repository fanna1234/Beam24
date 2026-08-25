#include <cuda.h>

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
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

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
    cu::DeviceMemory a(a_bytes), b(b_bytes), c(c_bytes);
    fill_inputs(options, a, b, stream);
    c.zero(c_bytes);
    Path path = make_path(options, device, stream, a, b, c);
    path.run();
    stream.synchronize();
    if (options.check) check_against_basic(options, device, stream, a, b, c);
    float milliseconds = time_callable(
        stream, options.warmup, options.iterations, path.run);
    double useful_ops = 8.0 * options.batch * options.m * options.n * options.k;
    double tops = useful_ops / (milliseconds * 1.0e9);
    std::printf(
        "{\"kind\":\"ccglib_same_entry_baseline\",\"mode\":\"%s\","
        "\"batch\":%zu,\"m\":%zu,\"n\":%zu,\"k\":%zu,"
        "\"warmup\":%zu,\"iterations\":%zu,\"milliseconds\":%.9g,"
        "\"effective_complex_tops\":%.9g,\"static_a_ms\":%.9g,"
        "\"check\":%s}\n",
        options.mode.c_str(), options.batch, options.m, options.n, options.k,
        options.warmup, options.iterations, milliseconds, tops,
        path.static_a_ms, options.check ? "true" : "false");
    return EXIT_SUCCESS;
}
