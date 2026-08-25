// Reuse the evidence-bound Beam24 kernel implementation without editing its
// source or changing the hash associated with the exhaustive reference.
#define main beam24_embedded_reference_main
#include "beam24_system.cu"
#undef main

#include <complex>
#include <cstring>
#include <functional>
#include <string>

#ifdef BEAM24_FINITE_QUALITY
#include <cublas_v2.h>
#endif

#ifndef BEAM24_DEFAULT_MODE
#define BEAM24_DEFAULT_MODE 1
#endif

namespace hierarchy {

#ifdef BEAM24_FINITE_QUALITY
#define CHECK_BLAS(call)                                                       \
    do {                                                                       \
        const cublasStatus_t status_ = (call);                                 \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                \
            std::fprintf(stderr, "cuBLAS error %s:%d: %d\n",                \
                         __FILE__, __LINE__, static_cast<int>(status_));        \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)
#endif

constexpr int FULL_M = 1024;
constexpr int FULL_K = 512;
constexpr int COARSE_M = 128;
constexpr int COARSE_K = 128;
constexpr int TOP_SECTORS = 8;
constexpr int REFINEMENT_SLOTS = 16;
constexpr int FINE_M = TOP_SECTORS * REFINEMENT_SLOTS;
constexpr int SUBARRAY_START = (FULL_K - COARSE_K) / 2;
constexpr size_t HIERARCHY_MIN_BATCH_SNAPSHOTS = 3072;

static_assert(FULL_M % BM == 0 && COARSE_M == BM && FINE_M == BM);
static_assert(FULL_K % BK == 0 && COARSE_K % BK == 0);
static_assert(SUBARRAY_START % 4 == 0);

struct Codebook {
    int rows = 0;
    int sensors = 0;
    std::vector<half> real_packed;
    std::vector<half> imag_packed;
    std::vector<uint32_t> row_meta;
    std::vector<uint32_t> tile_meta;
};

template <typename T>
__global__ void replicate_batches_kernel(
    const T* __restrict__ source,
    T* __restrict__ destination,
    size_t elements_per_batch,
    size_t total_elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        destination[index] = source[index % elements_per_batch];
    }
}

__global__ void initialize_input_kernel(
    half* real, half* imag, size_t elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t value = static_cast<uint32_t>(index) ^ 0x9e3779b9u;
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    const float r = (static_cast<int>(value & 0xffffu) - 32768) / 32768.0f;
    const float i = (static_cast<int>((value >> 16) & 0xffffu) - 32768) / 32768.0f;
    real[index] = __float2half_rn(r);
    imag[index] = __float2half_rn(i);
}

#ifdef BEAM24_FINITE_QUALITY
struct FiniteQualityParams {
    float snr_db;
    float gain_sigma_db;
    float phase_sigma_deg;
    float position_sigma_lambda;
    float reflection_db;
};

constexpr int FINITE_QUALITY_CONDITIONS = 11;

const char* finite_quality_condition_name(int condition) {
    static const char* names[FINITE_QUALITY_CONDITIONS] = {
        "clean", "snr10", "snr0", "gain05", "gain1", "phase1",
        "phase5", "position001", "position005", "reflection10", "combined"};
    return condition >= 0 && condition < FINITE_QUALITY_CONDITIONS
        ? names[condition] : "invalid";
}

FiniteQualityParams finite_quality_params(int condition) {
    const float none = std::numeric_limits<float>::infinity();
    switch (condition) {
        case 0: return {none, 0.0f, 0.0f, 0.0f, none};
        case 1: return {10.0f, 0.0f, 0.0f, 0.0f, none};
        case 2: return {0.0f, 0.0f, 0.0f, 0.0f, none};
        case 3: return {none, 0.5f, 0.0f, 0.0f, none};
        case 4: return {none, 1.0f, 0.0f, 0.0f, none};
        case 5: return {none, 0.0f, 1.0f, 0.0f, none};
        case 6: return {none, 0.0f, 5.0f, 0.0f, none};
        case 7: return {none, 0.0f, 0.0f, 0.001f, none};
        case 8: return {none, 0.0f, 0.0f, 0.005f, none};
        case 9: return {none, 0.0f, 0.0f, 0.0f, -10.0f};
        case 10: return {10.0f, 1.0f, 5.0f, 0.005f, -10.0f};
        default: return {none, 0.0f, 0.0f, 0.0f, none};
    }
}

__host__ __device__ __forceinline__ uint32_t finite_mix32(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

__host__ __device__ __forceinline__ float finite_uniform(
    uint32_t seed, uint32_t trial, uint32_t item, uint32_t tag) {
    uint32_t value = seed ^ (trial * 0x9e3779b9u) ^
        (item * 0x85ebca6bu) ^ (tag * 0xc2b2ae35u);
    value = finite_mix32(value);
    return (static_cast<float>(value) + 0.5f) * (1.0f / 4294967296.0f);
}

__device__ __forceinline__ float finite_normal(
    uint32_t seed, uint32_t trial, uint32_t item, uint32_t tag) {
    const float u1 = fmaxf(finite_uniform(seed, trial, item, tag), 1.0e-7f);
    const float u2 = finite_uniform(seed, trial, item, tag + 1u);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * CUDART_PI_F * u2);
}

__host__ __device__ __forceinline__ float finite_source_angle(
    uint32_t seed, uint32_t trial) {
    return -55.0f + 110.0f * finite_uniform(seed, trial, 0u, 1u);
}

__device__ __forceinline__ float2 finite_mul(float2 lhs, float2 rhs) {
    return make_float2(
        fmaf(-lhs.y, rhs.y, lhs.x * rhs.x),
        fmaf(lhs.x, rhs.y, lhs.y * rhs.x));
}

__device__ __forceinline__ float2 finite_polar(float magnitude, float phase) {
    float sine = 0.0f;
    float cosine = 0.0f;
    sincosf(phase, &sine, &cosine);
    return make_float2(magnitude * cosine, magnitude * sine);
}

__global__ void prepare_finite_sensor_state_kernel(
    float2* source_state,
    float2* reflection_state,
    int batches,
    uint32_t seed,
    uint32_t trial_offset,
    FiniteQualityParams params) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(batches) * FULL_K;
    if (linear >= total) return;
    const int sensor = static_cast<int>(linear % FULL_K);
    const int batch = static_cast<int>(linear / FULL_K);
    const uint32_t trial = trial_offset + static_cast<uint32_t>(batch);
    const float angle = finite_source_angle(seed, trial);
    const float gain_db = params.gain_sigma_db == 0.0f ? 0.0f :
        params.gain_sigma_db * finite_normal(seed, trial, sensor, 11u);
    const float gain = exp2f(gain_db * (log2f(10.0f) / 20.0f));
    const float phase_error = params.phase_sigma_deg == 0.0f ? 0.0f :
        params.phase_sigma_deg * finite_normal(seed, trial, sensor, 17u) *
            (CUDART_PI_F / 180.0f);
    const float position_error = params.position_sigma_lambda == 0.0f ? 0.0f :
        params.position_sigma_lambda * finite_normal(seed, trial, sensor, 23u);
    const float position = 0.5f * sensor + position_error;
    const float spatial_phase = 2.0f * CUDART_PI_F * position *
        sinf(angle * CUDART_PI_F / 180.0f);
    const float2 calibration = finite_polar(gain, phase_error);
    source_state[linear] = finite_mul(calibration, finite_polar(1.0f, spatial_phase));

    if (isfinite(params.reflection_db)) {
        const float direction = finite_uniform(seed, trial, 0u, 31u) < 0.5f ? -1.0f : 1.0f;
        const float offset = direction *
            (5.0f + 20.0f * finite_uniform(seed, trial, 0u, 37u));
        const float reflection_angle = fminf(55.0f, fmaxf(-55.0f, angle + offset));
        const float reflection_phase = 2.0f * CUDART_PI_F *
            finite_uniform(seed, trial, 0u, 41u);
        const float reflection_spatial = 2.0f * CUDART_PI_F * position *
            sinf(reflection_angle * CUDART_PI_F / 180.0f);
        reflection_state[linear] = finite_mul(
            calibration, finite_polar(1.0f, reflection_spatial + reflection_phase));
    } else {
        reflection_state[linear] = make_float2(0.0f, 0.0f);
    }
}

__global__ void prepare_finite_waveform_kernel(
    float2* waveform,
    int batches,
    int snapshots,
    uint32_t seed,
    uint32_t trial_offset) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(batches) * snapshots;
    if (linear >= total) return;
    const int snapshot = static_cast<int>(linear % snapshots);
    const int batch = static_cast<int>(linear / snapshots);
    const uint32_t trial = trial_offset + static_cast<uint32_t>(batch);
    const float phase = 2.0f * CUDART_PI_F *
        finite_uniform(seed, trial, static_cast<uint32_t>(snapshot), 47u);
    waveform[linear] = finite_polar(1.0f, phase);
}

__global__ void generate_finite_snapshot_input_kernel(
    half* real,
    half* imag,
    const float2* source_state,
    const float2* reflection_state,
    const float2* waveform,
    int batches,
    int snapshots,
    uint32_t seed,
    uint32_t trial_offset,
    FiniteQualityParams params) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(batches) * snapshots * FULL_K;
    if (linear >= total) return;
    size_t cursor = linear;
    const int sensor = static_cast<int>(cursor % FULL_K); cursor /= FULL_K;
    const int snapshot = static_cast<int>(cursor % snapshots); cursor /= snapshots;
    const int batch = static_cast<int>(cursor);
    const uint32_t trial = trial_offset + static_cast<uint32_t>(batch);
    const size_t sensor_index = static_cast<size_t>(batch) * FULL_K + sensor;
    const size_t waveform_index = static_cast<size_t>(batch) * snapshots + snapshot;
    const float reflection_scale = isfinite(params.reflection_db)
        ? exp2f(params.reflection_db * (log2f(10.0f) / 20.0f)) : 0.0f;
    float2 manifold = source_state[sensor_index];
    manifold.x = fmaf(reflection_scale, reflection_state[sensor_index].x, manifold.x);
    manifold.y = fmaf(reflection_scale, reflection_state[sensor_index].y, manifold.y);
    float2 value = finite_mul(waveform[waveform_index], manifold);
    if (isfinite(params.snr_db)) {
        const float variance = exp2f(-params.snr_db * (log2f(10.0f) / 10.0f));
        const float sigma = sqrtf(0.5f * variance);
        const uint32_t item = static_cast<uint32_t>(snapshot * FULL_K + sensor);
        value.x += sigma * finite_normal(seed, trial, item, 53u);
        value.y += sigma * finite_normal(seed, trial, item, 59u);
    }
    real[linear] = __float2half_rn(value.x);
    imag[linear] = __float2half_rn(value.y);
}

__global__ void power_reduce_column_major_kernel(
    const float* real,
    const float* imag,
    float* power,
    int batches,
    int rows,
    int columns) {
    const size_t row_linear = blockIdx.x;
    const int batch = static_cast<int>(row_linear / rows);
    const int row = static_cast<int>(row_linear % rows);
    if (batch >= batches) return;
    const size_t base = static_cast<size_t>(batch) * rows * columns;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const size_t index = base + row + static_cast<size_t>(column) * rows;
        sum = fmaf(real[index], real[index], sum);
        sum = fmaf(imag[index], imag[index], sum);
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

void make_dense_ula_codebook(std::vector<half>& real, std::vector<half>& imag) {
    real.resize(static_cast<size_t>(FULL_M) * FULL_K);
    imag.resize(static_cast<size_t>(FULL_M) * FULL_K);
    constexpr double pi = 3.141592653589793238462643383279502884;
    for (int row = 0; row < FULL_M; ++row) {
        const double angle = -60.0 + 120.0 * row / (FULL_M - 1);
        const double spatial_phase = pi * std::sin(angle * pi / 180.0);
        for (int sensor = 0; sensor < FULL_K; ++sensor) {
            const std::complex<double> value = std::polar(
                1.0 / static_cast<double>(FULL_K), sensor * spatial_phase);
            const size_t index = static_cast<size_t>(row) * FULL_K + sensor;
            real[index] = __float2half_rn(static_cast<float>(value.real()));
            imag[index] = __float2half_rn(static_cast<float>(value.imag()));
        }
    }
}
#endif

__device__ __forceinline__ bool pair_better(
    float lhs_value, uint32_t lhs_index,
    float rhs_value, uint32_t rhs_index) {
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
    values[thread] = coarse_power[static_cast<size_t>(batch) * COARSE_M + thread];
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
                    ? pair_better(rhs_value, rhs_index, lhs_value, lhs_index)
                    : pair_better(lhs_value, lhs_index, rhs_value, rhs_index);
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
        top_sectors[static_cast<size_t>(batch) * TOP_SECTORS + thread] = indices[thread];
    }

    const int sector = thread / REFINEMENT_SLOTS;
    const int slot = thread % REFINEMENT_SLOTS;
    const int coarse_index = static_cast<int>(indices[sector]);
    const int center =
        (coarse_index * (FULL_M - 1) + (COARSE_M - 1) / 2) / (COARSE_M - 1);
    constexpr int stride =
        ((FULL_M - 1) + (COARSE_M - 1) / 2) / (COARSE_M - 1);
    constexpr int radius = stride / 2;
    const int start = max(0, center - radius);
    const int stop = min(FULL_M - 1, center + radius);
    const int count = stop - start + 1;
    const int candidate = slot < count ? start + slot : center;
    candidate_rows[static_cast<size_t>(batch) * FINE_M + thread] =
        static_cast<uint32_t>(candidate);
}

__global__ void gather_packed_a_vec_kernel(
    const uint4* __restrict__ source_real,
    const uint4* __restrict__ source_imag,
    const uint32_t* __restrict__ candidate_rows,
    uint4* __restrict__ destination_real,
    uint4* __restrict__ destination_imag,
    int chunks_per_row,
    size_t total_chunks) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total_chunks) return;
    const int chunk = static_cast<int>(index % chunks_per_row);
    const size_t row_linear = index / chunks_per_row;
    const uint32_t source_row = candidate_rows[row_linear];
    const size_t source_index = static_cast<size_t>(source_row) * chunks_per_row + chunk;
    destination_real[index] = source_real[source_index];
    destination_imag[index] = source_imag[source_index];
}

__global__ void repack_fine_metadata_kernel(
    const uint32_t* __restrict__ full_row_meta,
    const uint32_t* __restrict__ candidate_rows,
    uint32_t* __restrict__ fine_tile_meta,
    int k_tiles,
    size_t logical_pairs) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= logical_pairs) return;

    size_t cursor = index;
    const int group = static_cast<int>(cursor % 8); cursor /= 8;
    const int segment = static_cast<int>(cursor % M16_SEGMENTS); cursor /= M16_SEGMENTS;
    const int sparse_k = static_cast<int>(cursor % SPARSE_K_STEPS); cursor /= SPARSE_K_STEPS;
    const int k_tile = static_cast<int>(cursor % k_tiles); cursor /= k_tiles;
    const int batch = static_cast<int>(cursor);

    const int fine_row0 = segment * MMA_M + group;
    const int fine_row1 = fine_row0 + 8;
    const uint32_t source_row0 =
        candidate_rows[static_cast<size_t>(batch) * FINE_M + fine_row0];
    const uint32_t source_row1 =
        candidate_rows[static_cast<size_t>(batch) * FINE_M + fine_row1];
    const int words_per_row = k_tiles * SPARSE_K_STEPS;
    const int word = k_tile * SPARSE_K_STEPS + sparse_k;
    const uint32_t meta0 = full_row_meta[
        static_cast<size_t>(source_row0) * words_per_row + word];
    const uint32_t meta1 = full_row_meta[
        static_cast<size_t>(source_row1) * words_per_row + word];
    const uint32_t front = (meta0 & 0xffffu) | ((meta1 & 0xffffu) << 16);
    const uint32_t back = ((meta0 >> 16) & 0xffffu) | (meta1 & 0xffff0000u);

    const size_t tile_base =
        (static_cast<size_t>(batch) * k_tiles + k_tile) * META_WORDS_PER_KTILE;
    const size_t output = tile_base
        + sparse_k * META_WORDS_PER_SP_K
        + segment * META_WORDS_PER_WARP
        + group * 2;
    fine_tile_meta[output] = front;
    fine_tile_meta[output + 1] = back;
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
        if (better_power(value, full_index, best_value, best_index)) {
            best_value = value;
            best_index = full_index;
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
            max_power[batch] = best_value;
            max_index[batch] = best_index;
        }
    }
}

void create_tma_3d_f16_strided(
    CUtensorMap* tensor_map,
    void* data,
    int dim0,
    int dim1,
    int batches,
    uint64_t stride1_bytes,
    uint64_t stride2_bytes,
    int box0,
    int box1,
    CUtensorMapSwizzle swizzle) {
    uint64_t dimensions[3] = {
        static_cast<uint64_t>(dim0),
        static_cast<uint64_t>(dim1),
        static_cast<uint64_t>(batches)};
    uint64_t strides[2] = {stride1_bytes, stride2_bytes};
    uint32_t box[3] = {
        static_cast<uint32_t>(box0),
        static_cast<uint32_t>(box1), 1};
    uint32_t element_strides[3] = {1, 1, 1};
    CHECK_DRV(cuTensorMapEncodeTiled(
        tensor_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, data,
        dimensions, strides, box, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

Codebook make_ula_codebook(
    int rows,
    int sensors,
    int physical_sensor_start,
    double minimum_angle,
    double maximum_angle) {
    Codebook result;
    result.rows = rows;
    result.sensors = sensors;
    const int packed_k = sensors / 2;
    const int words_per_row = sensors / MMA_K_LOGICAL;
    result.real_packed.resize(static_cast<size_t>(rows) * packed_k);
    result.imag_packed.resize(static_cast<size_t>(rows) * packed_k);
    result.row_meta.assign(static_cast<size_t>(rows) * words_per_row, 0u);

    constexpr double pi = 3.141592653589793238462643383279502884;
    for (int row = 0; row < rows; ++row) {
        std::vector<std::complex<double>> packed_values(packed_k);
        double retained_energy = 0.0;
        double dense_energy = 0.0;
        const double angle = rows == 1
            ? minimum_angle
            : minimum_angle + (maximum_angle - minimum_angle) * row / (rows - 1);
        const double spatial_phase = pi * std::sin(angle * pi / 180.0);
        for (int group = 0; group < sensors / 4; ++group) {
            std::complex<double> weights[4];
            for (int element = 0; element < 4; ++element) {
                const int physical_sensor = physical_sensor_start + group * 4 + element;
                weights[element] = std::polar(
                    1.0 / static_cast<double>(sensors),
                    physical_sensor * spatial_phase);
                dense_energy += std::norm(weights[element]);
            }
            std::complex<double> coefficient[4];
            double magnitude[4];
            for (int mode = 0; mode < 4; ++mode) {
                coefficient[mode] = {0.0, 0.0};
                for (int element = 0; element < 4; ++element) {
                    const double phase = 2.0 * pi * element * mode / 4.0;
                    coefficient[mode] += weights[element] * std::polar(0.5, phase);
                }
                magnitude[mode] = std::norm(coefficient[mode]);
            }
            int first = 0;
            int second = 1;
            if (magnitude[second] > magnitude[first]) std::swap(first, second);
            for (int mode = 2; mode < 4; ++mode) {
                if (magnitude[mode] > magnitude[first]) {
                    second = first;
                    first = mode;
                } else if (magnitude[mode] > magnitude[second]) {
                    second = mode;
                }
            }
            if (first > second) std::swap(first, second);
            const uint32_t nibble = static_cast<uint32_t>(first | (second << 2));
            const int packed_base = group * 2;
            packed_values[packed_base] = coefficient[first];
            packed_values[packed_base + 1] = coefficient[second];
            retained_energy += magnitude[first] + magnitude[second];
            const int word = group / CHUNKS_PER_MMA;
            const int chunk = group % CHUNKS_PER_MMA;
            result.row_meta[static_cast<size_t>(row) * words_per_row + word] |=
                nibble << (chunk * 4);
        }
        const double gain_scale = dense_energy / retained_energy;
        for (int packed_k_index = 0; packed_k_index < packed_k; ++packed_k_index) {
            // The maintained producer computes an unnormalized four-point
            // transform, so the unitary beamspace coefficient carries an
            // additional factor of one half at the sparse-MMA input.
            const std::complex<double> value =
                packed_values[packed_k_index] * (0.5 * gain_scale);
            const size_t output = static_cast<size_t>(row) * packed_k + packed_k_index;
            result.real_packed[output] =
                __float2half_rn(static_cast<float>(value.real()));
            result.imag_packed[output] =
                __float2half_rn(static_cast<float>(value.imag()));
        }
    }
    build_meta_words_per_tile(rows, sensors, result.row_meta, result.tile_meta);
    return result;
}

void fill_plane_wave_input(
    std::vector<half>& real,
    std::vector<half>& imag,
    int batches,
    int snapshots) {
    const size_t elements = static_cast<size_t>(batches) * snapshots * FULL_K;
    real.resize(elements);
    imag.resize(elements);
    constexpr double pi = 3.141592653589793238462643383279502884;
    std::mt19937 generator(20260823u);
    std::normal_distribution<float> noise(0.0f, 0.002f);
    std::uniform_real_distribution<float> amplitude(-1.0f, 1.0f);
    for (int batch = 0; batch < batches; ++batch) {
        const int source_row = (batch * 73 + (batch % 3) * 7) % FULL_M;
        const double angle = -60.0 + 120.0 * source_row / (FULL_M - 1);
        const double spatial_phase = pi * std::sin(angle * pi / 180.0);
        for (int snapshot = 0; snapshot < snapshots; ++snapshot) {
            const std::complex<float> gain(amplitude(generator), amplitude(generator));
            for (int sensor = 0; sensor < FULL_K; ++sensor) {
                const std::complex<float> signal = gain * std::polar(
                    1.0f, static_cast<float>(sensor * spatial_phase));
                const size_t index =
                    (static_cast<size_t>(batch) * snapshots + snapshot) * FULL_K + sensor;
                real[index] = __float2half_rn(signal.real() + noise(generator));
                imag[index] = __float2half_rn(signal.imag() + noise(generator));
            }
        }
    }
}

template <typename T>
T* allocate_and_copy(const std::vector<T>& host) {
    T* device = nullptr;
    CHECK_RT(cudaMalloc(&device, host.size() * sizeof(T)));
    CHECK_RT(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

template <typename T>
void replicate_to_batches(
    const T* source,
    T* destination,
    size_t elements_per_batch,
    int batches,
    cudaStream_t stream) {
    const size_t total = elements_per_batch * batches;
    replicate_batches_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        source, destination, elements_per_batch, total);
}

cudaGraphExec_t capture_graph(cudaStream_t stream, const std::function<void()>& enqueue) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CHECK_RT(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    enqueue();
    CHECK_RT(cudaStreamEndCapture(stream, &graph));
    CHECK_RT(cudaGraphInstantiate(&executable, graph, 0));
    CHECK_RT(cudaGraphDestroy(graph));
    return executable;
}

bool half_bits_equal(half lhs, half rhs) {
    uint16_t left = 0;
    uint16_t right = 0;
    std::memcpy(&left, &lhs, sizeof(left));
    std::memcpy(&right, &rhs, sizeof(right));
    return left == right;
}

}  // namespace hierarchy

int main(int argc, char** argv) {
    using namespace hierarchy;
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 100;
    const int warmup = argc > 2 ? std::atoi(argv[2]) : 20;
    const int check = argc > 3 ? std::atoi(argv[3]) : 0;
    const int batches = argc > 4 ? std::atoi(argv[4]) : 256;
    const int snapshots = argc > 5 ? std::atoi(argv[5]) : 1024;
    const int use_graph = argc > 6 ? std::atoi(argv[6]) : 1;
    const int pointer_pad_mib = argc > 7 ? std::atoi(argv[7]) : 0;
#ifdef BEAM24_FINITE_QUALITY
    const int quality_condition = argc > 8 ? std::atoi(argv[8]) : -1;
    const uint32_t quality_seed = argc > 9
        ? static_cast<uint32_t>(std::strtoul(argv[9], nullptr, 10)) : 0u;
    const int quality_trial_offset = argc > 10 ? std::atoi(argv[10]) : 0;
    const int quality_replays = argc > 11 ? std::atoi(argv[11]) : 5;
    const int quality_dense_oracle = argc > 12 ? std::atoi(argv[12]) : 1;
    const bool quality_mode = quality_condition >= 0;
#else
    const bool quality_mode = false;
#endif
    const int requested_mode = BEAM24_DEFAULT_MODE;
    // The integrated hierarchy has a fixed multi-launch floor. Locked SM120a
    // crossover measurements retain exhaustive Beam24 below B*N=3072.
    const int mode =
        requested_mode == 1 &&
                static_cast<size_t>(batches) * snapshots <
                    HIERARCHY_MIN_BATCH_SNAPSHOTS
            ? 0
            : requested_mode;
    if (iterations <= 0 || warmup < 0 || batches <= 0 ||
        snapshots <= 0 || snapshots % BN != 0 ||
        pointer_pad_mib < 0 ||
        (check != 0 && check != 1) || (use_graph != 0 && use_graph != 1) ||
        (requested_mode != 0 && requested_mode != 1)
#ifdef BEAM24_FINITE_QUALITY
        || quality_condition >= FINITE_QUALITY_CONDITIONS ||
        (quality_mode && (check != 1 || quality_seed == 0u ||
                          quality_trial_offset < 0 || quality_replays <= 0 ||
                          (quality_dense_oracle != 0 && quality_dense_oracle != 1)))
#endif
        ) {
        std::fprintf(stderr,
            "Usage: %s [iterations warmup check batch snapshots graph pointer_pad_mib"
#ifdef BEAM24_FINITE_QUALITY
            " quality_condition quality_seed trial_offset quality_replays dense_oracle"
#endif
            "], "
            "with snapshots divisible by %d and compile-time mode 0/1\n",
            argv[0], BN);
        return 2;
    }

    CHECK_DRV(cuInit(0));
    cudaDeviceProp properties{};
    CHECK_RT(cudaGetDeviceProperties(&properties, 0));
    if (properties.major != 12 || properties.minor != 0) {
        std::fprintf(stderr, "This experiment requires compute capability 12.0\n");
        return 3;
    }
    cudaStream_t stream = nullptr;
    CHECK_RT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    void* d_pointer_pad = nullptr;
    if (pointer_pad_mib > 0) {
        CHECK_RT(cudaMalloc(
            &d_pointer_pad, static_cast<size_t>(pointer_pad_mib) << 20));
    }

    const Codebook full = make_ula_codebook(FULL_M, FULL_K, 0, -60.0, 60.0);
    const Codebook coarse = make_ula_codebook(
        COARSE_M, COARSE_K, SUBARRAY_START, -60.0, 60.0);

    half* d_full_real_static = allocate_and_copy(full.real_packed);
    half* d_full_imag_static = allocate_and_copy(full.imag_packed);
    uint32_t* d_full_row_meta = allocate_and_copy(full.row_meta);
    half* d_coarse_real_static = allocate_and_copy(coarse.real_packed);
    half* d_coarse_imag_static = allocate_and_copy(coarse.imag_packed);
    uint32_t* d_coarse_tile_meta_static = allocate_and_copy(coarse.tile_meta);

    const size_t input_elements =
        static_cast<size_t>(batches) * snapshots * FULL_K;
    half* d_input_real = nullptr;
    half* d_input_imag = nullptr;
    CHECK_RT(cudaMalloc(&d_input_real, input_elements * sizeof(half)));
    CHECK_RT(cudaMalloc(&d_input_imag, input_elements * sizeof(half)));
#ifdef BEAM24_FINITE_QUALITY
    float2* d_quality_source_state = nullptr;
    float2* d_quality_reflection_state = nullptr;
    float2* d_quality_waveform = nullptr;
    if (quality_mode) {
        const FiniteQualityParams params = finite_quality_params(quality_condition);
        const size_t sensor_state_elements = static_cast<size_t>(batches) * FULL_K;
        const size_t waveform_elements = static_cast<size_t>(batches) * snapshots;
        CHECK_RT(cudaMalloc(&d_quality_source_state,
                            sensor_state_elements * sizeof(float2)));
        CHECK_RT(cudaMalloc(&d_quality_reflection_state,
                            sensor_state_elements * sizeof(float2)));
        CHECK_RT(cudaMalloc(&d_quality_waveform,
                            waveform_elements * sizeof(float2)));
        prepare_finite_sensor_state_kernel<<<
            (sensor_state_elements + 255) / 256, 256, 0, stream>>>(
            d_quality_source_state, d_quality_reflection_state, batches,
            quality_seed, static_cast<uint32_t>(quality_trial_offset), params);
        prepare_finite_waveform_kernel<<<
            (waveform_elements + 255) / 256, 256, 0, stream>>>(
            d_quality_waveform, batches, snapshots, quality_seed,
            static_cast<uint32_t>(quality_trial_offset));
        generate_finite_snapshot_input_kernel<<<
            (input_elements + 255) / 256, 256, 0, stream>>>(
            d_input_real, d_input_imag,
            d_quality_source_state, d_quality_reflection_state,
            d_quality_waveform, batches, snapshots, quality_seed,
            static_cast<uint32_t>(quality_trial_offset), params);
    } else
#endif
    if (check) {
        std::vector<half> host_real;
        std::vector<half> host_imag;
        fill_plane_wave_input(host_real, host_imag, batches, snapshots);
        CHECK_RT(cudaMemcpyAsync(
            d_input_real, host_real.data(), input_elements * sizeof(half),
            cudaMemcpyHostToDevice, stream));
        CHECK_RT(cudaMemcpyAsync(
            d_input_imag, host_imag.data(), input_elements * sizeof(half),
            cudaMemcpyHostToDevice, stream));
    } else {
        initialize_input_kernel<<<(input_elements + 255) / 256, 256, 0, stream>>>(
            d_input_real, d_input_imag, input_elements);
    }

#ifdef BEAM24_FINITE_QUALITY
    half* d_dense_weight_real = nullptr;
    half* d_dense_weight_imag = nullptr;
    float* d_dense_output_real = nullptr;
    float* d_dense_output_imag = nullptr;
    float* d_dense_power = nullptr;
    float* d_dense_max_power = nullptr;
    uint32_t* d_dense_max_index = nullptr;
    cublasHandle_t dense_handle = nullptr;
    if (quality_mode && quality_dense_oracle) {
        std::vector<half> dense_weight_real;
        std::vector<half> dense_weight_imag;
        make_dense_ula_codebook(dense_weight_real, dense_weight_imag);
        d_dense_weight_real = allocate_and_copy(dense_weight_real);
        d_dense_weight_imag = allocate_and_copy(dense_weight_imag);
        const size_t dense_output_elements =
            static_cast<size_t>(batches) * FULL_M * snapshots;
        CHECK_RT(cudaMalloc(&d_dense_output_real,
                            dense_output_elements * sizeof(float)));
        CHECK_RT(cudaMalloc(&d_dense_output_imag,
                            dense_output_elements * sizeof(float)));
        CHECK_RT(cudaMalloc(&d_dense_power,
                            static_cast<size_t>(batches) * FULL_M * sizeof(float)));
        CHECK_RT(cudaMalloc(&d_dense_max_power, batches * sizeof(float)));
        CHECK_RT(cudaMalloc(&d_dense_max_index, batches * sizeof(uint32_t)));
        CHECK_BLAS(cublasCreate(&dense_handle));
        CHECK_BLAS(cublasSetStream(dense_handle, stream));
        CHECK_BLAS(cublasSetMathMode(dense_handle, CUBLAS_TENSOR_OP_MATH));
    }
#endif

    const bool need_exhaustive = mode == 0 || check;
    const bool need_hierarchy = mode == 1 || check;
    half* d_full_real_batched = nullptr;
    half* d_full_imag_batched = nullptr;
    uint32_t* d_full_tile_meta_batched = nullptr;
    float* d_exhaustive_power = nullptr;
    float* d_exhaustive_max_power = nullptr;
    uint32_t* d_exhaustive_max_index = nullptr;
    CUtensorMap tma_full_real{}, tma_full_imag{};

    if (need_exhaustive) {
        const size_t full_a_elements = full.real_packed.size() * batches;
        const size_t full_meta_elements = full.tile_meta.size() * batches;
        CHECK_RT(cudaMalloc(&d_full_real_batched, full_a_elements * sizeof(half)));
        CHECK_RT(cudaMalloc(&d_full_imag_batched, full_a_elements * sizeof(half)));
        CHECK_RT(cudaMalloc(&d_full_tile_meta_batched, full_meta_elements * sizeof(uint32_t)));
        replicate_to_batches(
            d_full_real_static, d_full_real_batched, full.real_packed.size(), batches, stream);
        replicate_to_batches(
            d_full_imag_static, d_full_imag_batched, full.imag_packed.size(), batches, stream);
        uint32_t* d_full_tile_meta_static = allocate_and_copy(full.tile_meta);
        replicate_to_batches(
            d_full_tile_meta_static, d_full_tile_meta_batched,
            full.tile_meta.size(), batches, stream);
        CHECK_RT(cudaFree(d_full_tile_meta_static));
        CHECK_RT(cudaMalloc(&d_exhaustive_power,
                            static_cast<size_t>(batches) * FULL_M * sizeof(float)));
        CHECK_RT(cudaMalloc(&d_exhaustive_max_power, batches * sizeof(float)));
        CHECK_RT(cudaMalloc(&d_exhaustive_max_index, batches * sizeof(uint32_t)));
        create_tma_3d_f16(
            &tma_full_real, d_full_real_batched, FULL_K / 2, FULL_M, batches,
            BKP, BM, CU_TENSOR_MAP_SWIZZLE_64B);
        create_tma_3d_f16(
            &tma_full_imag, d_full_imag_batched, FULL_K / 2, FULL_M, batches,
            BKP, BM, CU_TENSOR_MAP_SWIZZLE_64B);
    }

    half* d_coarse_real_batched = nullptr;
    half* d_coarse_imag_batched = nullptr;
    uint32_t* d_coarse_tile_meta_batched = nullptr;
    half* d_fine_real = nullptr;
    half* d_fine_imag = nullptr;
    uint32_t* d_fine_tile_meta = nullptr;
    float* d_coarse_power = nullptr;
    float* d_fine_power = nullptr;
    float* d_hierarchy_max_power = nullptr;
    uint32_t* d_hierarchy_max_index = nullptr;
    uint32_t* d_top_sectors = nullptr;
    uint32_t* d_candidate_rows = nullptr;
    CUtensorMap tma_coarse_real{}, tma_coarse_imag{};
    CUtensorMap tma_fine_real{}, tma_fine_imag{};

    if (need_hierarchy) {
        const size_t coarse_a_elements = coarse.real_packed.size() * batches;
        const size_t coarse_meta_elements = coarse.tile_meta.size() * batches;
        const size_t fine_a_elements =
            static_cast<size_t>(batches) * FINE_M * (FULL_K / 2);
        const size_t fine_meta_elements =
            static_cast<size_t>(batches) * (FULL_K / BK) * META_WORDS_PER_KTILE;
        CHECK_RT(cudaMalloc(&d_coarse_real_batched, coarse_a_elements * sizeof(half)));
        CHECK_RT(cudaMalloc(&d_coarse_imag_batched, coarse_a_elements * sizeof(half)));
        CHECK_RT(cudaMalloc(&d_coarse_tile_meta_batched,
                            coarse_meta_elements * sizeof(uint32_t)));
        replicate_to_batches(
            d_coarse_real_static, d_coarse_real_batched,
            coarse.real_packed.size(), batches, stream);
        replicate_to_batches(
            d_coarse_imag_static, d_coarse_imag_batched,
            coarse.imag_packed.size(), batches, stream);
        replicate_to_batches(
            d_coarse_tile_meta_static, d_coarse_tile_meta_batched,
            coarse.tile_meta.size(), batches, stream);
        CHECK_RT(cudaMalloc(&d_fine_real, fine_a_elements * sizeof(half)));
        CHECK_RT(cudaMalloc(&d_fine_imag, fine_a_elements * sizeof(half)));
        CHECK_RT(cudaMalloc(&d_fine_tile_meta, fine_meta_elements * sizeof(uint32_t)));
        CHECK_RT(cudaMalloc(&d_coarse_power,
                            static_cast<size_t>(batches) * COARSE_M * sizeof(float)));
        CHECK_RT(cudaMalloc(&d_fine_power,
                            static_cast<size_t>(batches) * FINE_M * sizeof(float)));
        CHECK_RT(cudaMalloc(&d_hierarchy_max_power, batches * sizeof(float)));
        CHECK_RT(cudaMalloc(&d_hierarchy_max_index, batches * sizeof(uint32_t)));
        CHECK_RT(cudaMalloc(&d_top_sectors,
                            static_cast<size_t>(batches) * TOP_SECTORS * sizeof(uint32_t)));
        CHECK_RT(cudaMalloc(&d_candidate_rows,
                            static_cast<size_t>(batches) * FINE_M * sizeof(uint32_t)));
        create_tma_3d_f16(
            &tma_coarse_real, d_coarse_real_batched, COARSE_K / 2, COARSE_M, batches,
            BKP, BM, CU_TENSOR_MAP_SWIZZLE_64B);
        create_tma_3d_f16(
            &tma_coarse_imag, d_coarse_imag_batched, COARSE_K / 2, COARSE_M, batches,
            BKP, BM, CU_TENSOR_MAP_SWIZZLE_64B);
        create_tma_3d_f16(
            &tma_fine_real, d_fine_real, FULL_K / 2, FINE_M, batches,
            BKP, BM, CU_TENSOR_MAP_SWIZZLE_64B);
        create_tma_3d_f16(
            &tma_fine_imag, d_fine_imag, FULL_K / 2, FINE_M, batches,
            BKP, BM, CU_TENSOR_MAP_SWIZZLE_64B);
    }

    CUtensorMap tma_input_full_real{}, tma_input_full_imag{};
    CUtensorMap tma_input_coarse_real{}, tma_input_coarse_imag{};
    create_tma_3d_f16(
        &tma_input_full_real, d_input_real, FULL_K, snapshots, batches,
        BK, BN, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_f16(
        &tma_input_full_imag, d_input_imag, FULL_K, snapshots, batches,
        BK, BN, CU_TENSOR_MAP_SWIZZLE_128B);
    if (need_hierarchy) {
        create_tma_3d_f16_strided(
            &tma_input_coarse_real, d_input_real + SUBARRAY_START,
            COARSE_K, snapshots, batches,
            static_cast<uint64_t>(FULL_K) * sizeof(half),
            static_cast<uint64_t>(FULL_K) * snapshots * sizeof(half),
            BK, BN, CU_TENSOR_MAP_SWIZZLE_128B);
        create_tma_3d_f16_strided(
            &tma_input_coarse_imag, d_input_imag + SUBARRAY_START,
            COARSE_K, snapshots, batches,
            static_cast<uint64_t>(FULL_K) * sizeof(half),
            static_cast<uint64_t>(FULL_K) * snapshots * sizeof(half),
            BK, BN, CU_TENSOR_MAP_SWIZZLE_128B);
    }

    CHECK_RT(cudaGetLastError());
    CHECK_RT(cudaStreamSynchronize(stream));

    const dim3 block(THREADS);
    const dim3 full_grid(snapshots / BN, FULL_M / BM, batches);
    const dim3 coarse_grid(snapshots / BN, 1, batches);
    const dim3 fine_grid(snapshots / BN, 1, batches);
    auto enqueue_exhaustive = [&]() {
        CHECK_RT(cudaMemsetAsync(
            d_exhaustive_power, 0,
            static_cast<size_t>(batches) * FULL_M * sizeof(float), stream));
        sm120_sp_complex_f4_v10_half2_io_kernel<true><<<full_grid, block, 0, stream>>>(
            tma_full_real, tma_full_imag,
            tma_input_full_real, tma_input_full_imag,
            d_full_tile_meta_batched,
            nullptr, nullptr, d_exhaustive_power,
            FULL_M, snapshots, FULL_K);
        argmax_power_kernel<<<batches, 256, 0, stream>>>(
            d_exhaustive_power, d_exhaustive_max_power,
            d_exhaustive_max_index, FULL_M);
    };

    const int chunks_per_row = (FULL_K / 2) * sizeof(half) / sizeof(uint4);
    const size_t total_chunks =
        static_cast<size_t>(batches) * FINE_M * chunks_per_row;
    const size_t logical_meta_pairs =
        static_cast<size_t>(batches) * (FULL_K / BK) * SPARSE_K_STEPS
        * M16_SEGMENTS * 8;
    auto enqueue_hierarchy = [&]() {
        CHECK_RT(cudaMemsetAsync(
            d_coarse_power, 0,
            static_cast<size_t>(batches) * COARSE_M * sizeof(float), stream));
        sm120_sp_complex_f4_v10_half2_io_kernel<true><<<coarse_grid, block, 0, stream>>>(
            tma_coarse_real, tma_coarse_imag,
            tma_input_coarse_real, tma_input_coarse_imag,
            d_coarse_tile_meta_batched,
            nullptr, nullptr, d_coarse_power,
            COARSE_M, snapshots, COARSE_K);
        top8_and_candidates_kernel<<<batches, COARSE_M, 0, stream>>>(
            d_coarse_power, d_top_sectors, d_candidate_rows);
        gather_packed_a_vec_kernel<<<(total_chunks + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const uint4*>(d_full_real_static),
            reinterpret_cast<const uint4*>(d_full_imag_static),
            d_candidate_rows,
            reinterpret_cast<uint4*>(d_fine_real),
            reinterpret_cast<uint4*>(d_fine_imag),
            chunks_per_row, total_chunks);
        repack_fine_metadata_kernel<<<
            (logical_meta_pairs + 255) / 256, 256, 0, stream>>>(
            d_full_row_meta, d_candidate_rows, d_fine_tile_meta,
            FULL_K / BK, logical_meta_pairs);
        CHECK_RT(cudaMemsetAsync(
            d_fine_power, 0,
            static_cast<size_t>(batches) * FINE_M * sizeof(float), stream));
        sm120_sp_complex_f4_v10_half2_io_kernel<true><<<fine_grid, block, 0, stream>>>(
            tma_fine_real, tma_fine_imag,
            tma_input_full_real, tma_input_full_imag,
            d_fine_tile_meta,
            nullptr, nullptr, d_fine_power,
            FINE_M, snapshots, FULL_K);
        mapped_argmax_kernel<<<batches, 256, 0, stream>>>(
            d_fine_power, d_candidate_rows,
            d_hierarchy_max_power, d_hierarchy_max_index);
    };

    cudaGraphExec_t exhaustive_graph = nullptr;
    cudaGraphExec_t hierarchy_graph = nullptr;
    if (use_graph && need_exhaustive) {
        exhaustive_graph = capture_graph(stream, enqueue_exhaustive);
    }
    if (use_graph && need_hierarchy) {
        hierarchy_graph = capture_graph(stream, enqueue_hierarchy);
    }
    auto launch_exhaustive = [&]() {
        if (use_graph) CHECK_RT(cudaGraphLaunch(exhaustive_graph, stream));
        else enqueue_exhaustive();
    };
    auto launch_hierarchy = [&]() {
        if (use_graph) CHECK_RT(cudaGraphLaunch(hierarchy_graph, stream));
        else enqueue_hierarchy();
    };

#ifdef BEAM24_FINITE_QUALITY
    auto launch_dense_quality = [&]() {
        if (!quality_mode || !quality_dense_oracle) return;
        const float one = 1.0f;
        const float minus_one = -1.0f;
        const float zero = 0.0f;
        const long long input_stride = static_cast<long long>(FULL_K) * snapshots;
        const long long output_stride = static_cast<long long>(FULL_M) * snapshots;
        const cublasGemmAlgo_t algorithm = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        CHECK_BLAS(cublasGemmStridedBatchedEx(
            dense_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            FULL_M, snapshots, FULL_K,
            &one,
            d_dense_weight_real, CUDA_R_16F, FULL_K, 0,
            d_input_real, CUDA_R_16F, FULL_K, input_stride,
            &zero,
            d_dense_output_real, CUDA_R_32F, FULL_M, output_stride,
            batches, CUBLAS_COMPUTE_32F, algorithm));
        CHECK_BLAS(cublasGemmStridedBatchedEx(
            dense_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            FULL_M, snapshots, FULL_K,
            &one,
            d_dense_weight_imag, CUDA_R_16F, FULL_K, 0,
            d_input_imag, CUDA_R_16F, FULL_K, input_stride,
            &one,
            d_dense_output_real, CUDA_R_32F, FULL_M, output_stride,
            batches, CUBLAS_COMPUTE_32F, algorithm));
        CHECK_BLAS(cublasGemmStridedBatchedEx(
            dense_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            FULL_M, snapshots, FULL_K,
            &one,
            d_dense_weight_real, CUDA_R_16F, FULL_K, 0,
            d_input_imag, CUDA_R_16F, FULL_K, input_stride,
            &zero,
            d_dense_output_imag, CUDA_R_32F, FULL_M, output_stride,
            batches, CUBLAS_COMPUTE_32F, algorithm));
        CHECK_BLAS(cublasGemmStridedBatchedEx(
            dense_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            FULL_M, snapshots, FULL_K,
            &minus_one,
            d_dense_weight_imag, CUDA_R_16F, FULL_K, 0,
            d_input_real, CUDA_R_16F, FULL_K, input_stride,
            &one,
            d_dense_output_imag, CUDA_R_32F, FULL_M, output_stride,
            batches, CUBLAS_COMPUTE_32F, algorithm));
        power_reduce_column_major_kernel<<<
            static_cast<unsigned int>(static_cast<size_t>(batches) * FULL_M),
            256, 0, stream>>>(
            d_dense_output_real, d_dense_output_imag, d_dense_power,
            batches, FULL_M, snapshots);
        argmax_power_kernel<<<batches, 256, 0, stream>>>(
            d_dense_power, d_dense_max_power, d_dense_max_index, FULL_M);
    };
#endif

    auto launch_selected = [&]() {
        if (mode == 0) launch_exhaustive();
        else launch_hierarchy();
    };
    for (int iteration = 0; iteration < warmup; ++iteration) launch_selected();
    CHECK_RT(cudaGetLastError());
    CHECK_RT(cudaStreamSynchronize(stream));
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    CHECK_RT(cudaEventCreate(&begin));
    CHECK_RT(cudaEventCreate(&end));
    CHECK_RT(cudaEventRecord(begin, stream));
    for (int iteration = 0; iteration < iterations; ++iteration) launch_selected();
    CHECK_RT(cudaEventRecord(end, stream));
    CHECK_RT(cudaEventSynchronize(end));
    float milliseconds = 0.0f;
    CHECK_RT(cudaEventElapsedTime(&milliseconds, begin, end));
    milliseconds /= iterations;

    const size_t candidate_workspace =
        static_cast<size_t>(batches) * FINE_M * (FULL_K / 2) * sizeof(half) * 2
        + static_cast<size_t>(batches) * (FULL_K / BK) * META_WORDS_PER_KTILE
            * sizeof(uint32_t)
        + static_cast<size_t>(batches) * (COARSE_M + FINE_M) * sizeof(float)
        + static_cast<size_t>(batches) *
            (TOP_SECTORS + FINE_M + 2) * sizeof(uint32_t)
        + static_cast<size_t>(batches) * sizeof(float);
    std::printf(
        "{\"kind\":\"%s\",\"milliseconds\":%.9g,"
        "\"batch\":%d,\"m\":%d,\"n\":%d,\"k\":%d,"
        "\"coarse_m\":%d,\"coarse_k\":%d,\"fine_m\":%d,"
        "\"top_sectors\":%d,\"refinement_slots\":%d,"
        "\"graph\":%s,\"iterations\":%d,\"warmup\":%d,"
        "\"dynamic_workspace_bytes\":%zu,\"pointer_pad_mib\":%d,"
        "\"dispatch\":\"%s\"}\n",
        mode == 0 ? "beam24_exhaustive_e2e" : "beam24_hierarchical_e2e",
        milliseconds, batches, FULL_M, snapshots, FULL_K,
        COARSE_M, COARSE_K, FINE_M, TOP_SECTORS, REFINEMENT_SLOTS,
        use_graph ? "true" : "false", iterations, warmup,
        mode == 0 ? size_t(0) : candidate_workspace, pointer_pad_mib,
        mode == 0 && requested_mode == 1
            ? "exhaustive_small_workload"
            : (mode == 0 ? "exhaustive" : "hierarchical"));

    int failure = 0;
    if (check) {
        launch_exhaustive();
        launch_hierarchy();
#ifdef BEAM24_FINITE_QUALITY
        if (quality_mode && quality_dense_oracle) launch_dense_quality();
#endif
        CHECK_RT(cudaGetLastError());
        CHECK_RT(cudaStreamSynchronize(stream));
        std::vector<uint32_t> exhaustive_indices(batches);
        std::vector<uint32_t> hierarchy_indices(batches);
        std::vector<float> exhaustive_powers(batches);
        std::vector<float> hierarchy_powers(batches);
        std::vector<uint32_t> candidates(static_cast<size_t>(batches) * FINE_M);
        CHECK_RT(cudaMemcpy(
            exhaustive_indices.data(), d_exhaustive_max_index,
            batches * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CHECK_RT(cudaMemcpy(
            hierarchy_indices.data(), d_hierarchy_max_index,
            batches * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CHECK_RT(cudaMemcpy(
            exhaustive_powers.data(), d_exhaustive_max_power,
            batches * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_RT(cudaMemcpy(
            hierarchy_powers.data(), d_hierarchy_max_power,
            batches * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_RT(cudaMemcpy(
            candidates.data(), d_candidate_rows,
            candidates.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

#ifdef BEAM24_FINITE_QUALITY
        std::vector<uint32_t> dense_indices;
        std::vector<float> dense_powers;
        std::vector<float> dense_power_map;
        int replay_unstable = 0;
        if (quality_mode && quality_dense_oracle) {
            dense_indices.resize(batches);
            dense_powers.resize(batches);
            dense_power_map.resize(static_cast<size_t>(batches) * FULL_M);
            CHECK_RT(cudaMemcpy(
                dense_indices.data(), d_dense_max_index,
                batches * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CHECK_RT(cudaMemcpy(
                dense_powers.data(), d_dense_max_power,
                batches * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_RT(cudaMemcpy(
                dense_power_map.data(), d_dense_power,
                dense_power_map.size() * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<uint32_t> replay_indices(batches);
            for (int replay = 1; replay < quality_replays; ++replay) {
                launch_hierarchy();
                CHECK_RT(cudaGetLastError());
                CHECK_RT(cudaStreamSynchronize(stream));
                CHECK_RT(cudaMemcpy(
                    replay_indices.data(), d_hierarchy_max_index,
                    batches * sizeof(uint32_t), cudaMemcpyDeviceToHost));
                for (int batch = 0; batch < batches; ++batch) {
                    replay_unstable +=
                        replay_indices[batch] != hierarchy_indices[batch];
                }
            }
        }
#endif

        int missing = 0;
        int index_mismatch = 0;
        float maximum_relative_power = 0.0f;
        for (int batch = 0; batch < batches; ++batch) {
            const auto begin_candidate = candidates.begin() + static_cast<size_t>(batch) * FINE_M;
            const auto end_candidate = begin_candidate + FINE_M;
            if (std::find(begin_candidate, end_candidate, exhaustive_indices[batch]) ==
                end_candidate) ++missing;
            if (hierarchy_indices[batch] != exhaustive_indices[batch]) ++index_mismatch;
            const float denominator = std::max(std::fabs(exhaustive_powers[batch]), 1.0e-6f);
            maximum_relative_power = std::max(
                maximum_relative_power,
                std::fabs(hierarchy_powers[batch] - exhaustive_powers[batch]) / denominator);
        }

        const size_t fine_a_elements =
            static_cast<size_t>(batches) * FINE_M * (FULL_K / 2);
        std::vector<half> fine_real(fine_a_elements);
        std::vector<half> fine_imag(fine_a_elements);
        CHECK_RT(cudaMemcpy(
            fine_real.data(), d_fine_real,
            fine_a_elements * sizeof(half), cudaMemcpyDeviceToHost));
        CHECK_RT(cudaMemcpy(
            fine_imag.data(), d_fine_imag,
            fine_a_elements * sizeof(half), cudaMemcpyDeviceToHost));
        size_t a_mismatch = 0;
        for (int batch = 0; batch < batches; ++batch) {
            for (int row = 0; row < FINE_M; ++row) {
                const uint32_t source_row =
                    candidates[static_cast<size_t>(batch) * FINE_M + row];
                for (int packed_k = 0; packed_k < FULL_K / 2; ++packed_k) {
                    const size_t destination =
                        (static_cast<size_t>(batch) * FINE_M + row) * (FULL_K / 2)
                        + packed_k;
                    const size_t source =
                        static_cast<size_t>(source_row) * (FULL_K / 2) + packed_k;
                    if (!half_bits_equal(fine_real[destination], full.real_packed[source]) ||
                        !half_bits_equal(fine_imag[destination], full.imag_packed[source])) {
                        ++a_mismatch;
                    }
                }
            }
        }

        const size_t fine_meta_elements =
            static_cast<size_t>(batches) * (FULL_K / BK) * META_WORDS_PER_KTILE;
        std::vector<uint32_t> fine_meta(fine_meta_elements);
        CHECK_RT(cudaMemcpy(
            fine_meta.data(), d_fine_tile_meta,
            fine_meta_elements * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        size_t metadata_mismatch = 0;
        const int k_tiles = FULL_K / BK;
        const int words_per_row = k_tiles * SPARSE_K_STEPS;
        for (int batch = 0; batch < batches; ++batch) {
            for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
                for (int sparse_k = 0; sparse_k < SPARSE_K_STEPS; ++sparse_k) {
                    for (int segment = 0; segment < M16_SEGMENTS; ++segment) {
                        for (int group = 0; group < 8; ++group) {
                            const uint32_t row0 = candidates[
                                static_cast<size_t>(batch) * FINE_M
                                + segment * MMA_M + group];
                            const uint32_t row1 = candidates[
                                static_cast<size_t>(batch) * FINE_M
                                + segment * MMA_M + group + 8];
                            const int word = k_tile * SPARSE_K_STEPS + sparse_k;
                            const uint32_t meta0 =
                                full.row_meta[static_cast<size_t>(row0) * words_per_row + word];
                            const uint32_t meta1 =
                                full.row_meta[static_cast<size_t>(row1) * words_per_row + word];
                            const uint32_t expected_front =
                                (meta0 & 0xffffu) | ((meta1 & 0xffffu) << 16);
                            const uint32_t expected_back =
                                ((meta0 >> 16) & 0xffffu) | (meta1 & 0xffff0000u);
                            const size_t output =
                                (static_cast<size_t>(batch) * k_tiles + k_tile)
                                    * META_WORDS_PER_KTILE
                                + sparse_k * META_WORDS_PER_SP_K
                                + segment * META_WORDS_PER_WARP + group * 2;
                            if (fine_meta[output] != expected_front) ++metadata_mismatch;
                            if (fine_meta[output + 1] != expected_back) ++metadata_mismatch;
                        }
                    }
                }
            }
        }

#ifdef BEAM24_FINITE_QUALITY
        int quality_nonfinite = 0;
        if (quality_mode && quality_dense_oracle) {
            int local_dense_exact = 0;
            int hierarchy_local_exact = 0;
            int hierarchy_dense_exact = 0;
            int hierarchy_dense_within_one = 0;
            int local_dense_max_shift_grid = 0;
            int hierarchy_dense_max_shift_grid = 0;
            double dense_true_doa_sum = 0.0;
            double hierarchy_true_doa_sum = 0.0;
            double dense_true_doa_max = 0.0;
            double hierarchy_true_doa_max = 0.0;
            double dense_margin_sum = 0.0;
            double dense_margin_min = std::numeric_limits<double>::infinity();
            double miss_margin_sum = 0.0;
            double miss_margin_max = 0.0;
            int miss_count = 0;
            for (int batch = 0; batch < batches; ++batch) {
                const uint32_t dense_index = dense_indices[batch];
                const uint32_t local_index = exhaustive_indices[batch];
                const uint32_t hierarchy_index = hierarchy_indices[batch];
                local_dense_exact += local_index == dense_index;
                hierarchy_local_exact += hierarchy_index == local_index;
                hierarchy_dense_exact += hierarchy_index == dense_index;
                const int local_shift = std::abs(
                    static_cast<int>(local_index) - static_cast<int>(dense_index));
                const int hierarchy_shift = std::abs(
                    static_cast<int>(hierarchy_index) - static_cast<int>(dense_index));
                hierarchy_dense_within_one += hierarchy_shift <= 1;
                local_dense_max_shift_grid = std::max(
                    local_dense_max_shift_grid, local_shift);
                hierarchy_dense_max_shift_grid = std::max(
                    hierarchy_dense_max_shift_grid, hierarchy_shift);
                const double source_angle = finite_source_angle(
                    quality_seed,
                    static_cast<uint32_t>(quality_trial_offset + batch));
                const double dense_angle = -60.0 +
                    120.0 * dense_index / static_cast<double>(FULL_M - 1);
                const double hierarchy_angle = -60.0 +
                    120.0 * hierarchy_index / static_cast<double>(FULL_M - 1);
                const double dense_doa = std::fabs(dense_angle - source_angle);
                const double hierarchy_doa = std::fabs(hierarchy_angle - source_angle);
                dense_true_doa_sum += dense_doa;
                hierarchy_true_doa_sum += hierarchy_doa;
                dense_true_doa_max = std::max(dense_true_doa_max, dense_doa);
                hierarchy_true_doa_max = std::max(
                    hierarchy_true_doa_max, hierarchy_doa);

                float second = -std::numeric_limits<float>::infinity();
                const size_t power_base = static_cast<size_t>(batch) * FULL_M;
                for (int row = 0; row < FULL_M; ++row) {
                    if (static_cast<uint32_t>(row) == dense_index) continue;
                    second = std::max(second, dense_power_map[power_base + row]);
                }
                const double denominator = std::max(
                    std::fabs(static_cast<double>(dense_powers[batch])), 1.0e-20);
                const double margin =
                    (static_cast<double>(dense_powers[batch]) - second) / denominator;
                dense_margin_sum += margin;
                dense_margin_min = std::min(dense_margin_min, margin);
                if (hierarchy_index != dense_index) {
                    miss_margin_sum += margin;
                    miss_margin_max = std::max(miss_margin_max, margin);
                    ++miss_count;
                    std::printf(
                        "{\"kind\":\"beam24_finite_snapshot_failure\","
                        "\"condition\":\"%s\",\"seed\":%u,\"trial\":%u,"
                        "\"source_angle_deg\":%.9g,\"dense_index\":%u,"
                        "\"local_index\":%u,\"hierarchy_index\":%u,"
                        "\"dense_margin\":%.9g}\n",
                        finite_quality_condition_name(quality_condition), quality_seed,
                        static_cast<uint32_t>(quality_trial_offset + batch), source_angle,
                        dense_index, local_index, hierarchy_index, margin);
                }
                quality_nonfinite +=
                    !std::isfinite(dense_powers[batch]) ||
                    !std::isfinite(exhaustive_powers[batch]) ||
                    !std::isfinite(hierarchy_powers[batch]) ||
                    !std::isfinite(margin);
            }
            std::printf(
                "{\"kind\":\"beam24_finite_snapshot_quality\","
                "\"condition\":\"%s\",\"condition_index\":%d,"
                "\"seed\":%u,\"trial_offset\":%d,\"trials\":%d,"
                "\"snapshots\":%d,\"local_dense_exact\":%d,"
                "\"hierarchy_local_exact\":%d,"
                "\"hierarchy_dense_exact\":%d,"
                "\"hierarchy_dense_within_one\":%d,"
                "\"local_dense_max_shift_grid\":%d,"
                "\"hierarchy_dense_max_shift_grid\":%d,"
                "\"replay_unstable\":%d,\"replays\":%d,"
                "\"dense_margin_min\":%.9g,\"dense_margin_mean\":%.9g,"
                "\"miss_margin_mean\":%.9g,\"miss_margin_max\":%.9g,"
                "\"dense_true_doa_mean_deg\":%.9g,"
                "\"dense_true_doa_max_deg\":%.9g,"
                "\"hierarchy_true_doa_mean_deg\":%.9g,"
                "\"hierarchy_true_doa_max_deg\":%.9g,"
                "\"packed_a_mismatch\":%zu,\"metadata_mismatch\":%zu,"
                "\"nonfinite\":%d}\n",
                finite_quality_condition_name(quality_condition), quality_condition,
                quality_seed, quality_trial_offset, batches, snapshots,
                local_dense_exact, hierarchy_local_exact, hierarchy_dense_exact,
                hierarchy_dense_within_one, local_dense_max_shift_grid,
                hierarchy_dense_max_shift_grid, replay_unstable, quality_replays,
                dense_margin_min, dense_margin_sum / batches,
                miss_count ? miss_margin_sum / miss_count : 0.0, miss_margin_max,
                dense_true_doa_sum / batches, dense_true_doa_max,
                hierarchy_true_doa_sum / batches, hierarchy_true_doa_max,
                a_mismatch, metadata_mismatch, quality_nonfinite);
        }
#endif

        const bool algorithm_passed = missing == 0 && index_mismatch == 0 &&
            maximum_relative_power <= 1.0e-4f;
        bool passed = algorithm_passed &&
            a_mismatch == 0 && metadata_mismatch == 0;
#ifdef BEAM24_FINITE_QUALITY
        if (quality_mode) {
            passed = a_mismatch == 0 && metadata_mismatch == 0 &&
                replay_unstable == 0 && quality_nonfinite == 0;
        }
#endif
        std::printf(
            "{\"kind\":\"beam24_hierarchy_correctness\","
            "\"winner_missing\":%d,\"index_mismatch\":%d,"
            "\"max_relative_power\":%.9g,\"packed_a_mismatch\":%zu,"
            "\"metadata_mismatch\":%zu,\"batch\":%d,\"status\":\"%s\"}\n",
            missing, index_mismatch, maximum_relative_power,
            a_mismatch, metadata_mismatch, batches,
            quality_mode ? (passed ? "MEASURED" : "FAILED") :
                (passed ? "PASSED" : "FAILED"));
        failure = passed ? 0 : 1;
    }

    CHECK_RT(cudaStreamSynchronize(stream));
#ifdef BEAM24_FINITE_QUALITY
    if (dense_handle != nullptr) CHECK_BLAS(cublasDestroy(dense_handle));
    if (d_dense_weight_real != nullptr) CHECK_RT(cudaFree(d_dense_weight_real));
    if (d_dense_weight_imag != nullptr) CHECK_RT(cudaFree(d_dense_weight_imag));
    if (d_dense_output_real != nullptr) CHECK_RT(cudaFree(d_dense_output_real));
    if (d_dense_output_imag != nullptr) CHECK_RT(cudaFree(d_dense_output_imag));
    if (d_dense_power != nullptr) CHECK_RT(cudaFree(d_dense_power));
    if (d_dense_max_power != nullptr) CHECK_RT(cudaFree(d_dense_max_power));
    if (d_dense_max_index != nullptr) CHECK_RT(cudaFree(d_dense_max_index));
    if (d_quality_source_state != nullptr) CHECK_RT(cudaFree(d_quality_source_state));
    if (d_quality_reflection_state != nullptr) CHECK_RT(cudaFree(d_quality_reflection_state));
    if (d_quality_waveform != nullptr) CHECK_RT(cudaFree(d_quality_waveform));
#endif
    if (d_pointer_pad != nullptr) CHECK_RT(cudaFree(d_pointer_pad));
    return failure;
}
