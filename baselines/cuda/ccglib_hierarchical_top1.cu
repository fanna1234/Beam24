#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <ccglib/ccglib.hpp>
#include <cudawrappers/cu.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#define CHECK_RT(call)                                                         \
    do {                                                                       \
        cudaError_t error_ = (call);                                           \
        if (error_ != cudaSuccess) {                                           \
            std::fprintf(                                                      \
                stderr, "CUDA Runtime error %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(error_));               \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

namespace hierarchy_support {

constexpr int FULL_M = 1024;
constexpr int FULL_K = 512;
constexpr int COARSE_M = 128;
constexpr int COARSE_K = 128;
constexpr int TOP_SECTORS = 8;
constexpr int REFINEMENT_SLOTS = 16;
constexpr int FINE_M = TOP_SECTORS * REFINEMENT_SLOTS;
constexpr int SUBARRAY_START = (FULL_K - COARSE_K) / 2;

struct DenseCodebook {
    int rows = 0;
    int sensors = 0;
    std::vector<half> real;
    std::vector<half> imag;
};

template <typename T>
__global__ void replicate_batches_kernel(
    const T* __restrict__ source,
    T* __restrict__ destination,
    size_t elements_per_batch,
    size_t total_elements) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        destination[index] = source[index % elements_per_batch];
    }
}
__device__ __forceinline__ bool pair_better(
    float lhs_value,
    uint32_t lhs_index,
    float rhs_value,
    uint32_t rhs_index) {
    return lhs_value > rhs_value ||
           (lhs_value == rhs_value && lhs_index < rhs_index);
}

__global__ void top8_and_candidates_kernel(
    const float* __restrict__ coarse_power,
    uint32_t* __restrict__ top_sectors,
    uint32_t* __restrict__ candidate_rows) {
    const int batch = blockIdx.x;
    const int thread = threadIdx.x;
    __shared__ float values[COARSE_M];
    __shared__ uint32_t indices[COARSE_M];
    values[thread] =
        coarse_power[static_cast<size_t>(batch) * COARSE_M + thread];
    indices[thread] = static_cast<uint32_t>(thread);
    __syncthreads();

    #pragma unroll
    for (int length = 2; length <= COARSE_M; length <<= 1) {
        #pragma unroll
        for (int stride = length >> 1; stride > 0; stride >>= 1) {
            const int peer = thread ^ stride;
            if (peer > thread) {
                const bool descending = (thread & length) == 0;
                const float lhs_value = values[thread];
                const float rhs_value = values[peer];
                const uint32_t lhs_index = indices[thread];
                const uint32_t rhs_index = indices[peer];
                const bool swap = descending
                    ? pair_better(
                          rhs_value, rhs_index, lhs_value, lhs_index)
                    : pair_better(
                          lhs_value, lhs_index, rhs_value, rhs_index);
                if (swap) {
                    values[thread] = rhs_value;
                    values[peer] = lhs_value;
                    indices[thread] = rhs_index;
                    indices[peer] = lhs_index;
                }
            }
            __syncthreads();
        }
    }

    if (thread < TOP_SECTORS) {
        top_sectors[
            static_cast<size_t>(batch) * TOP_SECTORS + thread] =
            indices[thread];
    }
    const int sector = thread / REFINEMENT_SLOTS;
    const int slot = thread % REFINEMENT_SLOTS;
    const int coarse_index = static_cast<int>(indices[sector]);
    const int center =
        (coarse_index * (FULL_M - 1) + (COARSE_M - 1) / 2) /
        (COARSE_M - 1);
    constexpr int coarse_stride =
        ((FULL_M - 1) + (COARSE_M - 1) / 2) / (COARSE_M - 1);
    constexpr int radius = coarse_stride / 2;
    const int start = max(0, center - radius);
    const int stop = min(FULL_M - 1, center + radius);
    const int count = stop - start + 1;
    const int candidate = slot < count ? start + slot : center;
    candidate_rows[
        static_cast<size_t>(batch) * FINE_M + thread] =
        static_cast<uint32_t>(candidate);
}

__device__ __forceinline__ bool better_power(
    float candidate_value,
    uint32_t candidate_index,
    float current_value,
    uint32_t current_index) {
    return candidate_value > current_value ||
           (candidate_value == current_value &&
            candidate_index < current_index);
}

__global__ void argmax_power_kernel(
    const float* __restrict__ power,
    float* __restrict__ max_power,
    uint32_t* __restrict__ max_index,
    int rows) {
    const int batch = blockIdx.x;
    const float* values = power + static_cast<size_t>(batch) * rows;
    float best_value = -CUDART_INF_F;
    uint32_t best_index = 0xffffffffu;
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        const float value = values[row];
        if (better_power(
                value, static_cast<uint32_t>(row),
                best_value, best_index)) {
            best_value = value;
            best_index = static_cast<uint32_t>(row);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_value =
            __shfl_down_sync(0xffffffffu, best_value, offset);
        const uint32_t other_index =
            __shfl_down_sync(0xffffffffu, best_index, offset);
        if (better_power(
                other_value, other_index, best_value, best_index)) {
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
            const float other_value =
                __shfl_down_sync(0xffffffffu, best_value, offset);
            const uint32_t other_index =
                __shfl_down_sync(0xffffffffu, best_index, offset);
            if (better_power(
                    other_value, other_index,
                    best_value, best_index)) {
                best_value = other_value;
                best_index = other_index;
            }
        }
        if (lane == 0) {
            max_power[batch] = best_value;
            max_index[batch] = best_index;
        }
    }
}

__global__ void mapped_argmax_kernel(
    const float* __restrict__ fine_power,
    const uint32_t* __restrict__ candidate_rows,
    float* __restrict__ max_power,
    uint32_t* __restrict__ max_index) {
    const int batch = blockIdx.x;
    const size_t base = static_cast<size_t>(batch) * FINE_M;
    float best_value = -CUDART_INF_F;
    uint32_t best_index = 0xffffffffu;
    for (int row = threadIdx.x; row < FINE_M; row += blockDim.x) {
        const float value = fine_power[base + row];
        const uint32_t full_index = candidate_rows[base + row];
        if (better_power(
                value, full_index, best_value, best_index)) {
            best_value = value;
            best_index = full_index;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_value =
            __shfl_down_sync(0xffffffffu, best_value, offset);
        const uint32_t other_index =
            __shfl_down_sync(0xffffffffu, best_index, offset);
        if (better_power(
                other_value, other_index, best_value, best_index)) {
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
            const float other_value =
                __shfl_down_sync(0xffffffffu, best_value, offset);
            const uint32_t other_index =
                __shfl_down_sync(0xffffffffu, best_index, offset);
            if (better_power(
                    other_value, other_index,
                    best_value, best_index)) {
                best_value = other_value;
                best_index = other_index;
            }
        }
        if (lane == 0) {
            max_power[batch] = best_value;
            max_index[batch] = best_index;
        }
    }
}

DenseCodebook make_projected_ula_codebook(
    int rows,
    int sensors,
    int physical_sensor_start,
    double minimum_angle,
    double maximum_angle) {
    DenseCodebook result;
    result.rows = rows;
    result.sensors = sensors;
    result.real.resize(static_cast<size_t>(rows) * sensors);
    result.imag.resize(static_cast<size_t>(rows) * sensors);
    constexpr double pi =
        3.141592653589793238462643383279502884;
    for (int row = 0; row < rows; ++row) {
        std::vector<std::complex<double>> projected(sensors);
        double dense_energy = 0.0;
        double retained_energy = 0.0;
        const double angle =
            minimum_angle +
            (maximum_angle - minimum_angle) * row / (rows - 1);
        const double spatial_phase =
            pi * std::sin(angle * pi / 180.0);
        for (int group = 0; group < sensors / 4; ++group) {
            std::complex<double> weights[4];
            std::complex<double> coefficient[4];
            double magnitude[4];
            for (int element = 0; element < 4; ++element) {
                const int physical_sensor =
                    physical_sensor_start + group * 4 + element;
                weights[element] = std::polar(
                    1.0 / static_cast<double>(sensors),
                    physical_sensor * spatial_phase);
                dense_energy += std::norm(weights[element]);
            }
            for (int mode = 0; mode < 4; ++mode) {
                coefficient[mode] = {0.0, 0.0};
                for (int element = 0; element < 4; ++element) {
                    coefficient[mode] += weights[element] * std::polar(
                        0.5, 2.0 * pi * element * mode / 4.0);
                }
                magnitude[mode] = std::norm(coefficient[mode]);
            }
            int first = 0;
            int second = 1;
            if (magnitude[second] > magnitude[first]) {
                std::swap(first, second);
            }
            for (int mode = 2; mode < 4; ++mode) {
                if (magnitude[mode] > magnitude[first]) {
                    second = first;
                    first = mode;
                } else if (magnitude[mode] > magnitude[second]) {
                    second = mode;
                }
            }
            bool retained[4] = {false, false, false, false};
            retained[first] = true;
            retained[second] = true;
            retained_energy +=
                magnitude[first] + magnitude[second];
            for (int element = 0; element < 4; ++element) {
                std::complex<double> value(0.0, 0.0);
                for (int mode = 0; mode < 4; ++mode) {
                    if (retained[mode]) {
                        value += coefficient[mode] * std::polar(
                            0.5,
                            -2.0 * pi * mode * element / 4.0);
                    }
                }
                projected[group * 4 + element] = value;
            }
        }
        const double gain_scale = dense_energy / retained_energy;
        for (int sensor = 0; sensor < sensors; ++sensor) {
            const std::complex<double> value =
                projected[sensor] * gain_scale;
            const size_t output =
                static_cast<size_t>(row) * sensors + sensor;
            result.real[output] =
                __float2half_rn(static_cast<float>(value.real()));
            result.imag[output] =
                __float2half_rn(static_cast<float>(value.imag()));
        }
    }
    return result;
}

void fill_plane_wave_input(
    std::vector<half>& real,
    std::vector<half>& imag,
    int batches,
    int snapshots) {
    const size_t elements =
        static_cast<size_t>(batches) * snapshots * FULL_K;
    real.resize(elements);
    imag.resize(elements);
    constexpr double pi =
        3.141592653589793238462643383279502884;
    std::mt19937 generator(20260823u);
    std::normal_distribution<float> noise(0.0f, 0.002f);
    std::uniform_real_distribution<float> amplitude(-1.0f, 1.0f);
    for (int batch = 0; batch < batches; ++batch) {
        const int source_row =
            (batch * 73 + (batch % 3) * 7) % FULL_M;
        const double angle =
            -60.0 + 120.0 * source_row / (FULL_M - 1);
        const double spatial_phase =
            pi * std::sin(angle * pi / 180.0);
        for (int snapshot = 0; snapshot < snapshots; ++snapshot) {
            const std::complex<float> gain(
                amplitude(generator), amplitude(generator));
            for (int sensor = 0; sensor < FULL_K; ++sensor) {
                const std::complex<float> signal = gain * std::polar(
                    1.0f,
                    static_cast<float>(sensor * spatial_phase));
                const size_t index =
                    (static_cast<size_t>(batch) * snapshots +
                     snapshot) *
                        FULL_K +
                    sensor;
                real[index] = __float2half_rn(
                    signal.real() + noise(generator));
                imag[index] = __float2half_rn(
                    signal.imag() + noise(generator));
            }
        }
    }
}

bool half_bits_equal(half lhs, half rhs) {
    uint16_t left = 0;
    uint16_t right = 0;
    std::memcpy(&left, &lhs, sizeof(left));
    std::memcpy(&right, &rhs, sizeof(right));
    return left == right;
}

}  // namespace hierarchy_support

namespace external_hierarchy {

constexpr int FULL_M = hierarchy_support::FULL_M;
constexpr int FULL_K = hierarchy_support::FULL_K;
constexpr int COARSE_M = hierarchy_support::COARSE_M;
constexpr int COARSE_K = hierarchy_support::COARSE_K;
constexpr int FINE_M = hierarchy_support::FINE_M;
constexpr int TOP_SECTORS = hierarchy_support::TOP_SECTORS;
constexpr int SUBARRAY_START = hierarchy_support::SUBARRAY_START;

template <typename T>
T* device_ptr(cu::DeviceMemory& memory) {
    return reinterpret_cast<T*>(static_cast<CUdeviceptr>(memory));
}

template <typename T>
const T* device_ptr(const cu::DeviceMemory& memory) {
    return reinterpret_cast<const T*>(static_cast<CUdeviceptr>(memory));
}

cudaStream_t runtime_stream(cu::Stream& stream) {
    return reinterpret_cast<cudaStream_t>(static_cast<CUstream>(stream));
}

__device__ __forceinline__ uint32_t mix32(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

__global__ void fill_half_planar(
    half* output, size_t count, uint32_t seed) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const uint32_t bits = mix32(static_cast<uint32_t>(index) ^ seed);
    output[index] = __float2half_rn(
        static_cast<float>(bits & 0xffffu) / 32768.0f - 1.0f);
}

__global__ void gather_center_input_kernel(
    const uint4* __restrict__ full,
    uint4* __restrict__ compact,
    int full_chunks,
    int compact_chunks,
    int snapshots,
    size_t total_chunks) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total_chunks) return;
    const int chunk = static_cast<int>(index % compact_chunks);
    const size_t row_linear = index / compact_chunks;
    const int snapshot = static_cast<int>(row_linear % snapshots);
    const size_t batch_plane = row_linear / snapshots;
    constexpr int start_chunk =
        SUBARRAY_START * static_cast<int>(sizeof(half)) /
        static_cast<int>(sizeof(uint4));
    const size_t source =
        (batch_plane * snapshots + snapshot) * full_chunks +
        start_chunk + chunk;
    compact[index] = full[source];
}

__global__ void gather_dense_a_planar_kernel(
    const uint4* __restrict__ source,
    const uint32_t* __restrict__ candidate_rows,
    uint4* __restrict__ destination,
    int chunks_per_row,
    size_t total_chunks) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total_chunks) return;
    const int chunk = static_cast<int>(index % chunks_per_row);
    const size_t row_linear = index / chunks_per_row;
    const int fine_row = static_cast<int>(row_linear % FINE_M);
    const size_t batch_plane = row_linear / FINE_M;
    const int plane = static_cast<int>(batch_plane % 2);
    const int batch = static_cast<int>(batch_plane / 2);
    const uint32_t source_row =
        candidate_rows[static_cast<size_t>(batch) * FINE_M + fine_row];
    const size_t source_index =
        (static_cast<size_t>(plane) * FULL_M + source_row) *
            chunks_per_row + chunk;
    destination[index] = source[source_index];
}

__global__ void power_reduce_planar_bcmn(
    const float* __restrict__ complex_output,
    float* __restrict__ power,
    int M,
    int N) {
    const size_t row_linear = blockIdx.x;
    const size_t batch = row_linear / M;
    const size_t row = row_linear - batch * M;
    const size_t plane_size = static_cast<size_t>(M) * N;
    const float* real =
        complex_output + (batch * 2 + 0) * plane_size + row * N;
    const float* imag =
        complex_output + (batch * 2 + 1) * plane_size + row * N;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < N; column += blockDim.x) {
        const float r = real[column];
        const float i = imag[column];
        sum = fmaf(r, r, sum);
        sum = fmaf(i, i, sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    __shared__ float warp_sums[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) power[row_linear] = sum;
    }
}

std::vector<half> planar_codebook(
    const hierarchy_support::DenseCodebook& codebook) {
    const size_t plane = static_cast<size_t>(codebook.rows) * codebook.sensors;
    std::vector<half> output(2 * plane);
    std::copy(codebook.real.begin(), codebook.real.end(), output.begin());
    std::copy(codebook.imag.begin(), codebook.imag.end(), output.begin() + plane);
    return output;
}

std::vector<half> planar_batched_input(
    const std::vector<half>& real,
    const std::vector<half>& imag,
    int batches,
    int snapshots) {
    const size_t plane = static_cast<size_t>(snapshots) * FULL_K;
    std::vector<half> output(static_cast<size_t>(batches) * 2 * plane);
    for (int batch = 0; batch < batches; ++batch) {
        std::copy_n(
            real.begin() + static_cast<size_t>(batch) * plane,
            plane,
            output.begin() + static_cast<size_t>(batch * 2) * plane);
        std::copy_n(
            imag.begin() + static_cast<size_t>(batch) * plane,
            plane,
            output.begin() + static_cast<size_t>(batch * 2 + 1) * plane);
    }
    return output;
}

std::unique_ptr<ccglib::mma::GEMM> make_basic_gemm(
    size_t batch,
    size_t m,
    size_t n,
    size_t k,
    cu::Device& device,
    cu::Stream& stream) {
    const ccglib::Precision precision(
        ccglib::ValueType::float16, ccglib::ValueType::float32);
    return std::make_unique<ccglib::mma::GEMM>(
        batch, m, n, k, device, stream, precision, ccglib::mma::basic,
        ccglib::complex_planar, ccglib::mma::row_major,
        ccglib::mma::row_major, ccglib::mma::col_major);
}

template <typename Callable>
float time_pipeline(
    cudaStream_t stream,
    int warmup,
    int iterations,
    const Callable& callable) {
    for (int iteration = 0; iteration < warmup; ++iteration) callable();
    CHECK_RT(cudaStreamSynchronize(stream));
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    CHECK_RT(cudaEventCreate(&begin));
    CHECK_RT(cudaEventCreate(&end));
    CHECK_RT(cudaEventRecord(begin, stream));
    for (int iteration = 0; iteration < iterations; ++iteration) callable();
    CHECK_RT(cudaEventRecord(end, stream));
    CHECK_RT(cudaEventSynchronize(end));
    float milliseconds = 0.0f;
    CHECK_RT(cudaEventElapsedTime(&milliseconds, begin, end));
    CHECK_RT(cudaEventDestroy(begin));
    CHECK_RT(cudaEventDestroy(end));
    return milliseconds / iterations;
}

}  // namespace external_hierarchy

int main(int argc, char** argv) {
    using namespace external_hierarchy;

    const int iterations = argc > 1 ? std::atoi(argv[1]) : 100;
    const int warmup = argc > 2 ? std::atoi(argv[2]) : 20;
    const int check = argc > 3 ? std::atoi(argv[3]) : 0;
    const int batches = argc > 4 ? std::atoi(argv[4]) : 256;
    const int snapshots = argc > 5 ? std::atoi(argv[5]) : 1024;
    if (iterations <= 0 || warmup < 0 || batches <= 0 || snapshots <= 0 ||
        (check != 0 && check != 1)) {
        std::fprintf(
            stderr,
            "Usage: %s [iterations warmup check batch snapshots]\n",
            argv[0]);
        return 2;
    }

    cu::init();
    cu::Device device(0);
    cu::Context context(CU_CTX_BLOCKING_SYNC, device);
    cu::Stream stream;
    cudaStream_t cuda_stream = runtime_stream(stream);

    const hierarchy_support::DenseCodebook full =
        hierarchy_support::make_projected_ula_codebook(
        FULL_M, FULL_K, 0, -60.0, 60.0);
    const hierarchy_support::DenseCodebook coarse =
        hierarchy_support::make_projected_ula_codebook(
        COARSE_M, COARSE_K, SUBARRAY_START, -60.0, 60.0);
    const std::vector<half> full_planar = planar_codebook(full);
    const std::vector<half> coarse_planar = planar_codebook(coarse);

    const size_t full_input_values =
        static_cast<size_t>(batches) * 2 * snapshots * FULL_K;
    const size_t coarse_input_values =
        static_cast<size_t>(batches) * 2 * snapshots * COARSE_K;
    const size_t coarse_a_values =
        static_cast<size_t>(batches) * 2 * COARSE_M * COARSE_K;
    const size_t fine_a_values =
        static_cast<size_t>(batches) * 2 * FINE_M * FULL_K;
    const size_t complex_output_values =
        static_cast<size_t>(batches) * 2 * FINE_M * snapshots;
    const size_t power_values = static_cast<size_t>(batches) * FINE_M;

    cu::DeviceMemory d_full_static(full_planar.size() * sizeof(half));
    cu::DeviceMemory d_coarse_static(coarse_planar.size() * sizeof(half));
    cu::DeviceMemory d_full_input(full_input_values * sizeof(half));
    cu::DeviceMemory d_coarse_input(coarse_input_values * sizeof(half));
    cu::DeviceMemory d_coarse_a(coarse_a_values * sizeof(half));
    cu::DeviceMemory d_fine_a(fine_a_values * sizeof(half));
    cu::DeviceMemory d_complex_output(
        complex_output_values * sizeof(float));
    cu::DeviceMemory d_coarse_power(power_values * sizeof(float));
    cu::DeviceMemory d_fine_power(power_values * sizeof(float));
    cu::DeviceMemory d_max_power(static_cast<size_t>(batches) * sizeof(float));
    cu::DeviceMemory d_max_index(
        static_cast<size_t>(batches) * sizeof(uint32_t));
    cu::DeviceMemory d_top(
        static_cast<size_t>(batches) * TOP_SECTORS * sizeof(uint32_t));
    cu::DeviceMemory d_candidates(
        static_cast<size_t>(batches) * FINE_M * sizeof(uint32_t));

    cu::memcpyHtoD(
        d_full_static, full_planar.data(), d_full_static.size());
    cu::memcpyHtoD(
        d_coarse_static, coarse_planar.data(), d_coarse_static.size());

    std::vector<half> host_full_input;
    if (check) {
        std::vector<half> real;
        std::vector<half> imag;
        hierarchy_support::fill_plane_wave_input(
            real, imag, batches, snapshots);
        host_full_input =
            planar_batched_input(real, imag, batches, snapshots);
        cu::memcpyHtoD(
            d_full_input, host_full_input.data(), d_full_input.size());
    } else {
        fill_half_planar<<<
            (full_input_values + 255) / 256, 256, 0, cuda_stream>>>(
            device_ptr<half>(d_full_input), full_input_values, 20260824u);
    }

    hierarchy_support::replicate_batches_kernel<<<
        (coarse_a_values + 255) / 256, 256, 0, cuda_stream>>>(
        device_ptr<half>(d_coarse_static),
        device_ptr<half>(d_coarse_a),
        coarse_planar.size(),
        coarse_a_values);
    CHECK_RT(cudaGetLastError());
    CHECK_RT(cudaStreamSynchronize(cuda_stream));

    auto stage1 = make_basic_gemm(
        batches, COARSE_M, snapshots, COARSE_K, device, stream);
    auto stage2 = make_basic_gemm(
        batches, FINE_M, snapshots, FULL_K, device, stream);

    const int full_chunks = FULL_K * sizeof(half) / sizeof(uint4);
    const int coarse_chunks = COARSE_K * sizeof(half) / sizeof(uint4);
    const size_t coarse_input_chunks =
        static_cast<size_t>(batches) * 2 * snapshots * coarse_chunks;
    const int a_chunks = FULL_K * sizeof(half) / sizeof(uint4);
    const size_t fine_a_chunks =
        static_cast<size_t>(batches) * 2 * FINE_M * a_chunks;

    auto enqueue_hierarchy = [&]() {
        gather_center_input_kernel<<<
            (coarse_input_chunks + 255) / 256, 256, 0, cuda_stream>>>(
            reinterpret_cast<const uint4*>(
                device_ptr<half>(d_full_input)),
            reinterpret_cast<uint4*>(
                device_ptr<half>(d_coarse_input)),
            full_chunks,
            coarse_chunks,
            snapshots,
            coarse_input_chunks);
        stage1->Run(d_coarse_a, d_coarse_input, d_complex_output);
        power_reduce_planar_bcmn<<<
            batches * COARSE_M, 256, 0, cuda_stream>>>(
            device_ptr<float>(d_complex_output),
            device_ptr<float>(d_coarse_power),
            COARSE_M,
            snapshots);
        hierarchy_support::top8_and_candidates_kernel<<<
            batches, COARSE_M, 0, cuda_stream>>>(
            device_ptr<float>(d_coarse_power),
            device_ptr<uint32_t>(d_top),
            device_ptr<uint32_t>(d_candidates));
        gather_dense_a_planar_kernel<<<
            (fine_a_chunks + 255) / 256, 256, 0, cuda_stream>>>(
            reinterpret_cast<const uint4*>(
                device_ptr<half>(d_full_static)),
            device_ptr<uint32_t>(d_candidates),
            reinterpret_cast<uint4*>(
                device_ptr<half>(d_fine_a)),
            a_chunks,
            fine_a_chunks);
        stage2->Run(d_fine_a, d_full_input, d_complex_output);
        power_reduce_planar_bcmn<<<
            batches * FINE_M, 256, 0, cuda_stream>>>(
            device_ptr<float>(d_complex_output),
            device_ptr<float>(d_fine_power),
            FINE_M,
            snapshots);
        hierarchy_support::mapped_argmax_kernel<<<
            batches, 256, 0, cuda_stream>>>(
            device_ptr<float>(d_fine_power),
            device_ptr<uint32_t>(d_candidates),
            device_ptr<float>(d_max_power),
            device_ptr<uint32_t>(d_max_index));
    };

    CHECK_RT(cudaGetLastError());
    enqueue_hierarchy();
    CHECK_RT(cudaGetLastError());
    CHECK_RT(cudaStreamSynchronize(cuda_stream));

    int failure = 0;
    if (check) {
        const size_t full_a_values =
            static_cast<size_t>(batches) * 2 * FULL_M * FULL_K;
        const size_t full_output_values =
            static_cast<size_t>(batches) * 2 * FULL_M * snapshots;
        cu::DeviceMemory d_full_a(full_a_values * sizeof(half));
        cu::DeviceMemory d_full_output(full_output_values * sizeof(float));
        cu::DeviceMemory d_full_power(
            static_cast<size_t>(batches) * FULL_M * sizeof(float));
        cu::DeviceMemory d_full_max_power(
            static_cast<size_t>(batches) * sizeof(float));
        cu::DeviceMemory d_full_max_index(
            static_cast<size_t>(batches) * sizeof(uint32_t));
        hierarchy_support::replicate_batches_kernel<<<
            (full_a_values + 255) / 256, 256, 0, cuda_stream>>>(
            device_ptr<half>(d_full_static),
            device_ptr<half>(d_full_a),
            full_planar.size(),
            full_a_values);
        auto exhaustive = make_basic_gemm(
            batches, FULL_M, snapshots, FULL_K, device, stream);
        exhaustive->Run(d_full_a, d_full_input, d_full_output);
        power_reduce_planar_bcmn<<<
            batches * FULL_M, 256, 0, cuda_stream>>>(
            device_ptr<float>(d_full_output),
            device_ptr<float>(d_full_power),
            FULL_M,
            snapshots);
        hierarchy_support::argmax_power_kernel<<<batches, 256, 0, cuda_stream>>>(
            device_ptr<float>(d_full_power),
            device_ptr<float>(d_full_max_power),
            device_ptr<uint32_t>(d_full_max_index),
            FULL_M);
        enqueue_hierarchy();
        CHECK_RT(cudaGetLastError());
        CHECK_RT(cudaStreamSynchronize(cuda_stream));

        std::vector<uint32_t> exhaustive_indices(batches);
        std::vector<uint32_t> hierarchy_indices(batches);
        std::vector<float> exhaustive_powers(batches);
        std::vector<float> hierarchy_powers(batches);
        std::vector<uint32_t> candidates(
            static_cast<size_t>(batches) * FINE_M);
        cu::memcpyDtoH(
            exhaustive_indices.data(),
            d_full_max_index,
            d_full_max_index.size());
        cu::memcpyDtoH(
            hierarchy_indices.data(),
            d_max_index,
            d_max_index.size());
        cu::memcpyDtoH(
            exhaustive_powers.data(),
            d_full_max_power,
            d_full_max_power.size());
        cu::memcpyDtoH(
            hierarchy_powers.data(),
            d_max_power,
            d_max_power.size());
        cu::memcpyDtoH(
            candidates.data(), d_candidates, d_candidates.size());

        int missing = 0;
        int index_mismatch = 0;
        float max_relative_power = 0.0f;
        for (int batch = 0; batch < batches; ++batch) {
            const auto first =
                candidates.begin() + static_cast<size_t>(batch) * FINE_M;
            const auto last = first + FINE_M;
            if (std::find(first, last, exhaustive_indices[batch]) == last) {
                ++missing;
            }
            if (hierarchy_indices[batch] != exhaustive_indices[batch]) {
                ++index_mismatch;
            }
            const float denominator =
                std::max(std::fabs(exhaustive_powers[batch]), 1.0e-6f);
            max_relative_power = std::max(
                max_relative_power,
                std::fabs(
                    hierarchy_powers[batch] - exhaustive_powers[batch]) /
                    denominator);
        }

        std::vector<half> fine_a(fine_a_values);
        std::vector<half> coarse_input(coarse_input_values);
        cu::memcpyDtoH(fine_a.data(), d_fine_a, d_fine_a.size());
        cu::memcpyDtoH(
            coarse_input.data(), d_coarse_input, d_coarse_input.size());
        size_t a_mismatch = 0;
        for (int batch = 0; batch < batches; ++batch) {
            for (int plane = 0; plane < 2; ++plane) {
                for (int row = 0; row < FINE_M; ++row) {
                    const uint32_t source_row =
                        candidates[
                            static_cast<size_t>(batch) * FINE_M + row];
                    for (int sensor = 0; sensor < FULL_K; ++sensor) {
                        const size_t destination =
                            ((static_cast<size_t>(batch) * 2 + plane) *
                                 FINE_M +
                             row) *
                                FULL_K +
                            sensor;
                        const size_t source =
                            (static_cast<size_t>(plane) * FULL_M +
                             source_row) *
                                FULL_K +
                            sensor;
                        if (!hierarchy_support::half_bits_equal(
                                fine_a[destination],
                                full_planar[source])) {
                            ++a_mismatch;
                        }
                    }
                }
            }
        }

        size_t input_mismatch = 0;
        for (int batch = 0; batch < batches; ++batch) {
            for (int plane = 0; plane < 2; ++plane) {
                for (int snapshot = 0; snapshot < snapshots; ++snapshot) {
                    for (int sensor = 0; sensor < COARSE_K; ++sensor) {
                        const size_t destination =
                            ((static_cast<size_t>(batch) * 2 + plane) *
                                 snapshots +
                             snapshot) *
                                COARSE_K +
                            sensor;
                        const size_t source =
                            ((static_cast<size_t>(batch) * 2 + plane) *
                                 snapshots +
                             snapshot) *
                                FULL_K +
                            SUBARRAY_START + sensor;
                        if (!hierarchy_support::half_bits_equal(
                                coarse_input[destination],
                                host_full_input[source])) {
                            ++input_mismatch;
                        }
                    }
                }
            }
        }

        const bool passed =
            missing == 0 && index_mismatch == 0 &&
            max_relative_power <= 1.0e-4f && a_mismatch == 0 &&
            input_mismatch == 0;
        std::printf(
            "{\"kind\":\"ccglib_same_hierarchy_correctness\","
            "\"winner_missing\":%d,\"index_mismatch\":%d,"
            "\"max_relative_power\":%.9g,\"dense_a_mismatch\":%zu,"
            "\"subarray_input_mismatch\":%zu,"
            "\"status\":\"%s\"}\n",
            missing,
            index_mismatch,
            max_relative_power,
            a_mismatch,
            input_mismatch,
            passed ? "PASSED" : "FAILED");
        failure = passed ? 0 : 1;
    }

    const float milliseconds =
        time_pipeline(cuda_stream, warmup, iterations, enqueue_hierarchy);
    const size_t visible_workspace =
        d_coarse_input.size() + d_fine_a.size() +
        d_complex_output.size() + d_coarse_power.size() +
        d_fine_power.size() + d_top.size() + d_candidates.size() +
        d_max_power.size() + d_max_index.size();
    std::printf(
        "{\"kind\":\"ccglib_same_hierarchy_e2e\","
        "\"milliseconds\":%.9g,\"batch\":%d,\"m\":%d,\"n\":%d,"
        "\"k\":%d,\"coarse_m\":%d,\"coarse_k\":%d,"
        "\"fine_m\":%d,\"warmup\":%d,\"iterations\":%d,"
        "\"visible_workspace_bytes\":%zu,\"engine\":\"ccglib_basic\","
        "\"launch_mode\":\"direct\"}\n",
        milliseconds,
        batches,
        FULL_M,
        snapshots,
        FULL_K,
        COARSE_M,
        COARSE_K,
        FINE_M,
        warmup,
        iterations,
        visible_workspace);
    return failure;
}
