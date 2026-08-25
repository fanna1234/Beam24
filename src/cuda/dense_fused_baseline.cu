// Internal dense complex FP16 Tensor-Core attribution control with
// accumulator-resident beam power.
// Target: sm_120a, BM=64, BN=128, BK=64, two-stage TMA, 4P+8C.

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#define CHECK_DRV(call)                                                        \
    do {                                                                       \
        CUresult err_ = (call);                                                \
        if (err_ != CUDA_SUCCESS) {                                            \
            const char* name = nullptr;                                        \
            cuGetErrorName(err_, &name);                                       \
            std::fprintf(stderr, "CUDA Driver error %s:%d: %s\n",             \
                         __FILE__, __LINE__, name ? name : "(unknown)");       \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CHECK_RT(call)                                                         \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA Runtime error %s:%d: %s\n",            \
                         __FILE__, __LINE__, cudaGetErrorString(err_));         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

namespace {

// ---- Tile constants ----
constexpr int MMA_M  = 16;
constexpr int MMA_N8 = 8;   // half of MMA_N, for m16n8 sub-tile
constexpr int MMA_N  = 16;  // two m16n8 MMAs per m16n16 tile
constexpr int MMA_K = 16;

constexpr int BM  = 64;
constexpr int BN  = 128;
constexpr int BK  = 64;

constexpr int ROW_BYTES_A = BK * int(sizeof(half));  // 128 -> SW128B
constexpr int ROW_BYTES_B = BK * int(sizeof(half));  // 128 -> SW128B

constexpr int DENSE_K_STEPS = BK / MMA_K;  // 4

constexpr int PRODUCER_WARPS   = 4;
constexpr int CONSUMER_WARPS   = 8;
constexpr int WARPS_TOTAL      = PRODUCER_WARPS + CONSUMER_WARPS;  // 12
constexpr int THREADS          = WARPS_TOTAL * 32;                 // 384
constexpr int CONSUMER_THREADS = CONSUMER_WARPS * 32;              // 256
constexpr int PRODUCER_THREADS = PRODUCER_WARPS * 32;              // 128

// Each consumer warp covers 32x32 of the 64x128 CTA tile.
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
static_assert(WARPS_M * WARPS_N == CONSUMER_WARPS);

constexpr int WM_TILES = BM / (WARPS_M * MMA_M);  // 128/(2*16)=4
constexpr int WN_TILES = BN / (WARPS_N * MMA_N);   // 128/(4*16)=2

constexpr int PIPE_STAGES = 2;

// A/B layouts are [rows, BK] in shared memory with SW128B.

// ---- Device helpers ----

__device__ __forceinline__ uint32_t cvta_shared_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void ldmatrix_x4_shared_b16(uint32_t regs[4],
                                                        uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_shared_b16(uint32_t regs[2],
                                                        uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(regs[0]), "=r"(regs[1])
        : "r"(addr));
}

// Dense FP16 MMA: m16n8k16.
__device__ __forceinline__ void mma_m16n8k16_f16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
}

__device__ __forceinline__ void init_barrier(uint64_t* bar, int count) {
    uint64_t p = static_cast<uint64_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "l"(p), "r"(count) : "memory");
}

__device__ __forceinline__ void arrive_barrier_tx(uint64_t* bar, uint32_t bytes) {
    uint64_t p = static_cast<uint64_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :: "l"(p), "r"(bytes) : "memory");
}

__device__ __forceinline__ void arrive_barrier(uint64_t* bar) {
    uint64_t p = static_cast<uint64_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" :: "l"(p) : "memory");
}

__device__ __forceinline__ void wait_barrier(uint64_t* bar, int parity) {
    uint64_t p = static_cast<uint64_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT_%=:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n"
        "@P1 bra.uni DONE_%=;\n"
        "bra.uni LAB_WAIT_%=;\n"
        "DONE_%=:\n"
        "}\n"
        :: "l"(p), "r"(parity) : "memory");
}

__device__ __forceinline__ half* sw128_b_ptr(half* base, int row, int k) {
    const int linear_byte = row * ROW_BYTES_B + k * int(sizeof(half));
    const int swizzle_mask = (row & 7) << 4;
    return reinterpret_cast<half*>(
        reinterpret_cast<char*>(base) + (linear_byte ^ swizzle_mask));
}

__global__ void power_reduce_separate_planes(
    const float* __restrict__ real,
    const float* __restrict__ imag,
    float* __restrict__ power,
    int M, int N) {
    const size_t row_linear = blockIdx.x;
    const size_t batch_index = row_linear / static_cast<size_t>(M);
    const int row = static_cast<int>(row_linear - batch_index * M);
    const size_t base = batch_index * static_cast<size_t>(M) * N
                      + static_cast<size_t>(row) * N;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < N; column += blockDim.x) {
        const float r = real[base + column];
        const float i = imag[base + column];
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

// ---- Main kernel (4P + 8C warp specialization) ----

template <bool FUSED_POWER>
__global__ __launch_bounds__(THREADS, 1)
void sm120_dense_complex_fused_power_kernel(
    const __grid_constant__ CUtensorMap tmaAReal,
    const __grid_constant__ CUtensorMap tmaAImag,
    const __grid_constant__ CUtensorMap tmaBReal,
    const __grid_constant__ CUtensorMap tmaBImag,
    float* __restrict__ CReal,
    float* __restrict__ CImag,
    float* __restrict__ Power,
    int M, int N, int K)
{
    static __shared__ __align__(128) half smAReal[PIPE_STAGES][BM * BK];
    static __shared__ __align__(128) half smAImag[PIPE_STAGES][BM * BK];
    static __shared__ __align__(128) half smBReal[PIPE_STAGES][BN * BK]; // 2 x 16 KB
    static __shared__ __align__(128) half smBImag[PIPE_STAGES][BN * BK]; // 2 x 16 KB
    static __shared__ __align__(8)   uint64_t mbar_tma[PIPE_STAGES];
    static __shared__ __align__(8)   uint64_t mbar_ready[PIPE_STAGES];
    static __shared__ __align__(8)   uint64_t mbar_empty[PIPE_STAGES];

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid & 31;

    const int block_m  = blockIdx.y * BM;
    const int block_n  = blockIdx.x * BN;
    const int batch_index = blockIdx.z;
    const int k_tiles  = K / BK;
    constexpr uint32_t TMA_A_BYTES = BM * BK * sizeof(half);  // 8192
    constexpr uint32_t TMA_B_BYTES = BN * BK  * sizeof(half);  // 16384
    constexpr uint32_t TOTAL_TMA   = 2 * TMA_A_BYTES + 2 * TMA_B_BYTES;

    // ---- Init barriers ----
    if (tid < PIPE_STAGES) {
        init_barrier(&mbar_tma[tid], 1);
        init_barrier(&mbar_ready[tid], PRODUCER_THREADS);
        init_barrier(&mbar_empty[tid], CONSUMER_THREADS);  // 128
    }
    __syncthreads();

    // ---- Prologue: TMA prefetch stage 0 (A + B + Meta all async) ----
    if (tid == 0) {
        arrive_barrier_tx(&mbar_tma[0], TOTAL_TMA);
        // TMA A/B real and imaginary planes.
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cta.global.tile"
            ".mbarrier::complete_tx::bytes [%0], [%1, {%2,%3,%4}], [%5];\n"
            :: "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&smAReal[0][0]))),
               "l"(reinterpret_cast<uint64_t>(&tmaAReal)),
               "r"(0), "r"(block_m), "r"(batch_index),
               "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&mbar_tma[0])))
            : "memory");
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cta.global.tile"
            ".mbarrier::complete_tx::bytes [%0], [%1, {%2,%3,%4}], [%5];\n"
            :: "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&smAImag[0][0]))),
               "l"(reinterpret_cast<uint64_t>(&tmaAImag)),
               "r"(0), "r"(block_m), "r"(batch_index),
               "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&mbar_tma[0])))
            : "memory");
        // TMA B real/imag: full K, dim0 coord = kt*BK, dim1 coord = block_n
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cta.global.tile"
            ".mbarrier::complete_tx::bytes [%0], [%1, {%2,%3,%4}], [%5];\n"
            :: "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&smBReal[0][0]))),
               "l"(reinterpret_cast<uint64_t>(&tmaBReal)),
               "r"(0), "r"(block_n), "r"(batch_index),
               "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&mbar_tma[0])))
            : "memory");
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cta.global.tile"
            ".mbarrier::complete_tx::bytes [%0], [%1, {%2,%3,%4}], [%5];\n"
            :: "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&smBImag[0][0]))),
               "l"(reinterpret_cast<uint64_t>(&tmaBImag)),
               "r"(0), "r"(block_n), "r"(batch_index),
               "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&mbar_tma[0])))
            : "memory");
    }

    // ================================================================
    if (warp_id < PRODUCER_WARPS) {
        // ===== PRODUCER PATH =====
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");

        for (int kt = 0; kt < k_tiles; ++kt) {
            const int cur = kt % PIPE_STAGES;
            const int nxt = (kt + 1) % PIPE_STAGES;
            const int phase = (kt / PIPE_STAGES) & 1;
            wait_barrier(&mbar_tma[cur], phase);

            // Lane 0 schedules the next tile before publishing the current stage.
            if (tid == 0 && kt + 1 < k_tiles) {
                if (kt + 1 >= PIPE_STAGES) {
                    const int empty_phase =
                        ((kt + 1 - PIPE_STAGES) / PIPE_STAGES) & 1;
                    wait_barrier(&mbar_empty[nxt], empty_phase);
                }
                const int k_next  = (kt + 1) * BK;

                arrive_barrier_tx(&mbar_tma[nxt], TOTAL_TMA);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.tile"
                    ".mbarrier::complete_tx::bytes [%0], [%1, {%2,%3,%4}], [%5];\n"
                    :: "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&smAReal[nxt][0]))),
                       "l"(reinterpret_cast<uint64_t>(&tmaAReal)),
                       "r"(k_next), "r"(block_m), "r"(batch_index),
                       "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&mbar_tma[nxt])))
                    : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.tile"
                    ".mbarrier::complete_tx::bytes [%0], [%1, {%2,%3,%4}], [%5];\n"
                    :: "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&smAImag[nxt][0]))),
                       "l"(reinterpret_cast<uint64_t>(&tmaAImag)),
                       "r"(k_next), "r"(block_m), "r"(batch_index),
                       "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&mbar_tma[nxt])))
                    : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.tile"
                    ".mbarrier::complete_tx::bytes [%0], [%1, {%2,%3,%4}], [%5];\n"
                    :: "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&smBReal[nxt][0]))),
                       "l"(reinterpret_cast<uint64_t>(&tmaBReal)),
                       "r"(k_next), "r"(block_n), "r"(batch_index),
                       "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&mbar_tma[nxt])))
                    : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.tile"
                    ".mbarrier::complete_tx::bytes [%0], [%1, {%2,%3,%4}], [%5];\n"
                    :: "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&smBImag[nxt][0]))),
                       "l"(reinterpret_cast<uint64_t>(&tmaBImag)),
                       "r"(k_next), "r"(block_n), "r"(batch_index),
                       "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&mbar_tma[nxt])))
                    : "memory");
            }

            // All producer threads publish the TMA-complete stage.
            arrive_barrier(&mbar_ready[cur]);
        }
    } else {
        // ===== CONSUMER PATH =====
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");

        const int cons_id = warp_id - PRODUCER_WARPS;  // 0..7
        const int cons_wm = cons_id / WARPS_N;         // 0..1
        const int cons_wn = cons_id % WARPS_N;         // 0..3

        const int group_id     = lane >> 2;    // 0..7
        const int tid_in_group = lane & 3;     // 0..3

        // B ldmatrix addressing (SW128B)
        const int b_offk  = (lane / 8) * 8;
        const int lane_n8 = lane % 8;

        // Two complex output planes: 2 x WM=4 x WN=2 x 8 = 128 floats.
        float c_real[WM_TILES][WN_TILES][8];
        float c_imag[WM_TILES][WN_TILES][8];
        #pragma unroll
        for (int wm = 0; wm < WM_TILES; ++wm) {
            #pragma unroll
            for (int wn = 0; wn < WN_TILES; ++wn) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    c_real[wm][wn][i] = 0.0f;
                    c_imag[wm][wn][i] = 0.0f;
                }
            }
        }

        // ---- Main K-loop ----
        for (int kt = 0; kt < k_tiles; ++kt) {
            const int cur = kt % PIPE_STAGES;
            const int phase = (kt / PIPE_STAGES) & 1;

            wait_barrier(&mbar_ready[cur], phase);

            const uint32_t smAReal_base = cvta_shared_u32(&smAReal[cur][0]);
            const uint32_t smAImag_base = cvta_shared_u32(&smAImag[cur][0]);
            const uint32_t smBReal_base = cvta_shared_u32(&smBReal[cur][0]);
            const uint32_t smBImag_base = cvta_shared_u32(&smBImag[cur][0]);


            // ---- Compute: four dense K16 steps ----
            #pragma unroll
            for (int dense_k = 0; dense_k < DENSE_K_STEPS; ++dense_k) {
                const int b_k_byte_base =
                    b_offk * int(sizeof(half)) + dense_k * MMA_K * int(sizeof(half));

                uint32_t br_lo[WN_TILES][2];
                uint32_t br_hi[WN_TILES][2];
                uint32_t bi_lo[WN_TILES][2];
                uint32_t bi_hi[WN_TILES][2];

                #pragma unroll
                for (int wn = 0; wn < WN_TILES; ++wn) {
                    const int n_base = cons_wn * WN_TILES * MMA_N + wn * MMA_N;
                    const int row_lo = n_base + lane_n8;
                    const int row_hi = n_base + MMA_N8 + lane_n8;
                    const int mask_lo = (row_lo & 7) << 4;  // SW128B de-swizzle
                    const int mask_hi = (row_hi & 7) << 4;
                    const int lin_lo = row_lo * ROW_BYTES_B + b_k_byte_base;
                    const int lin_hi = row_hi * ROW_BYTES_B + b_k_byte_base;
                    const uint32_t lo_offset = static_cast<uint32_t>(lin_lo ^ mask_lo);
                    const uint32_t hi_offset = static_cast<uint32_t>(lin_hi ^ mask_hi);
                    ldmatrix_x2_shared_b16(br_lo[wn], smBReal_base + lo_offset);
                    ldmatrix_x2_shared_b16(br_hi[wn], smBReal_base + hi_offset);
                    ldmatrix_x2_shared_b16(bi_lo[wn], smBImag_base + lo_offset);
                    ldmatrix_x2_shared_b16(bi_hi[wn], smBImag_base + hi_offset);
                }

                // Compute: WM_TILES x WN_TILES x 2 (lo/hi N8) dense MMAs.
                #pragma unroll
                for (int wm = 0; wm < WM_TILES; ++wm) {
                    const int a_row = cons_wm * WM_TILES * MMA_M + wm * MMA_M + (lane & 15);
                    const int a_k_byte =
                        ((lane >> 4) * 8 + dense_k * MMA_K) * int(sizeof(half));
                    const int a_lin = a_row * ROW_BYTES_A + a_k_byte;
                    const int a_mask = (a_row & 7) << 4;
                    const uint32_t a_offset = static_cast<uint32_t>(a_lin ^ a_mask);

                    uint32_t ar[4];
                    uint32_t ai[4];
                    ldmatrix_x4_shared_b16(ar, smAReal_base + a_offset);
                    ldmatrix_x4_shared_b16(ai, smAImag_base + a_offset);
                    uint32_t ai_neg[4];
                    #pragma unroll
                    for (int reg = 0; reg < 4; ++reg) {
                        ai_neg[reg] = ai[reg] ^ 0x80008000u;
                    }

                    // conj(A)*B: real=Ar*Br+Ai*Bi, imag=Ar*Bi-Ai*Br.
                    #pragma unroll
                    for (int wn = 0; wn < WN_TILES; ++wn) {
                        mma_m16n8k16_f16(
                            c_real[wm][wn][0], c_real[wm][wn][1],
                            c_real[wm][wn][2], c_real[wm][wn][3],
                            ar[0], ar[1], ar[2], ar[3],
                            br_lo[wn][0], br_lo[wn][1]);
                        mma_m16n8k16_f16(
                            c_real[wm][wn][0], c_real[wm][wn][1],
                            c_real[wm][wn][2], c_real[wm][wn][3],
                            ai[0], ai[1], ai[2], ai[3],
                            bi_lo[wn][0], bi_lo[wn][1]);
                        mma_m16n8k16_f16(
                            c_imag[wm][wn][0], c_imag[wm][wn][1],
                            c_imag[wm][wn][2], c_imag[wm][wn][3],
                            ar[0], ar[1], ar[2], ar[3],
                            bi_lo[wn][0], bi_lo[wn][1]);
                        mma_m16n8k16_f16(
                            c_imag[wm][wn][0], c_imag[wm][wn][1],
                            c_imag[wm][wn][2], c_imag[wm][wn][3],
                            ai_neg[0], ai_neg[1], ai_neg[2], ai_neg[3],
                            br_lo[wn][0], br_lo[wn][1]);

                        mma_m16n8k16_f16(
                            c_real[wm][wn][4], c_real[wm][wn][5],
                            c_real[wm][wn][6], c_real[wm][wn][7],
                            ar[0], ar[1], ar[2], ar[3],
                            br_hi[wn][0], br_hi[wn][1]);
                        mma_m16n8k16_f16(
                            c_real[wm][wn][4], c_real[wm][wn][5],
                            c_real[wm][wn][6], c_real[wm][wn][7],
                            ai[0], ai[1], ai[2], ai[3],
                            bi_hi[wn][0], bi_hi[wn][1]);
                        mma_m16n8k16_f16(
                            c_imag[wm][wn][4], c_imag[wm][wn][5],
                            c_imag[wm][wn][6], c_imag[wm][wn][7],
                            ar[0], ar[1], ar[2], ar[3],
                            bi_hi[wn][0], bi_hi[wn][1]);
                        mma_m16n8k16_f16(
                            c_imag[wm][wn][4], c_imag[wm][wn][5],
                            c_imag[wm][wn][6], c_imag[wm][wn][7],
                            ai_neg[0], ai_neg[1], ai_neg[2], ai_neg[3],
                            br_hi[wn][0], br_hi[wn][1]);
                    }
                }
            }

            arrive_barrier(&mbar_empty[cur]);
        }

        if constexpr (FUSED_POWER) {
            #pragma unroll
            for (int wm = 0; wm < WM_TILES; ++wm) {
                float power_lo = 0.0f;
                float power_hi = 0.0f;
                #pragma unroll
                for (int wn = 0; wn < WN_TILES; ++wn) {
                    const float* cr = c_real[wm][wn];
                    const float* ci = c_imag[wm][wn];
                    #pragma unroll
                    for (int value = 0; value < 2; ++value) {
                        power_lo = fmaf(cr[value], cr[value], power_lo);
                        power_lo = fmaf(ci[value], ci[value], power_lo);
                        power_hi = fmaf(cr[value + 2], cr[value + 2], power_hi);
                        power_hi = fmaf(ci[value + 2], ci[value + 2], power_hi);
                        power_lo = fmaf(cr[value + 4], cr[value + 4], power_lo);
                        power_lo = fmaf(ci[value + 4], ci[value + 4], power_lo);
                        power_hi = fmaf(cr[value + 6], cr[value + 6], power_hi);
                        power_hi = fmaf(ci[value + 6], ci[value + 6], power_hi);
                    }
                }
                power_lo += __shfl_down_sync(0xffffffffu, power_lo, 1, 4);
                power_lo += __shfl_down_sync(0xffffffffu, power_lo, 2, 4);
                power_hi += __shfl_down_sync(0xffffffffu, power_hi, 1, 4);
                power_hi += __shfl_down_sync(0xffffffffu, power_hi, 2, 4);
                if (tid_in_group == 0) {
                    const int row_lo = block_m + cons_wm * WM_TILES * MMA_M
                                     + wm * MMA_M + group_id;
                    const int row_hi = row_lo + 8;
                    const size_t power_base = static_cast<size_t>(batch_index) * M;
                    if (row_lo < M) atomicAdd(Power + power_base + row_lo, power_lo);
                    if (row_hi < M) atomicAdd(Power + power_base + row_hi, power_hi);
                }
            }
        } else {
            // ---- Store complex C ----
            const size_t output_base = static_cast<size_t>(batch_index) * M * N;
            #pragma unroll
            for (int wm = 0; wm < WM_TILES; ++wm) {
                const int row_lo = block_m + cons_wm * WM_TILES * MMA_M + wm * MMA_M + group_id;
                const int row_hi = row_lo + 8;
                #pragma unroll
                for (int wn = 0; wn < WN_TILES; ++wn) {
                    const int col = block_n + cons_wn * WN_TILES * MMA_N
                                  + wn * MMA_N + tid_in_group * 2;
                    const float* cr = c_real[wm][wn];
                    const float* ci = c_imag[wm][wn];
                    if (row_lo < M) {
                        if (col + 1 < N) {
                            const size_t index = output_base + row_lo * N + col;
                            *reinterpret_cast<float2*>(CReal + index) = make_float2(cr[0], cr[1]);
                            *reinterpret_cast<float2*>(CImag + index) = make_float2(ci[0], ci[1]);
                        }
                        if (col + 9 < N) {
                            const size_t index = output_base + row_lo * N + col + 8;
                            *reinterpret_cast<float2*>(CReal + index) = make_float2(cr[4], cr[5]);
                            *reinterpret_cast<float2*>(CImag + index) = make_float2(ci[4], ci[5]);
                        }
                    }
                    if (row_hi < M) {
                        if (col + 1 < N) {
                            const size_t index = output_base + row_hi * N + col;
                            *reinterpret_cast<float2*>(CReal + index) = make_float2(cr[2], cr[3]);
                            *reinterpret_cast<float2*>(CImag + index) = make_float2(ci[2], ci[3]);
                        }
                        if (col + 9 < N) {
                            const size_t index = output_base + row_hi * N + col + 8;
                            *reinterpret_cast<float2*>(CReal + index) = make_float2(cr[6], cr[7]);
                            *reinterpret_cast<float2*>(CImag + index) = make_float2(ci[6], ci[7]);
                        }
                    }
                }
            }
        }
    }
}

// ---- Host helpers ----

void create_tma_2d_f16(CUtensorMap* tmap, void* data,
                       int dim0, int dim1, int box0, int box1,
                       CUtensorMapSwizzle swz) {
    uint64_t gDim[2]    = { uint64_t(dim0), uint64_t(dim1) };
    uint64_t gStride[1] = { uint64_t(dim0) * sizeof(half) };
    uint32_t bDim[2]    = { uint32_t(box0), uint32_t(box1) };
    uint32_t eStride[2] = { 1, 1 };
    CHECK_DRV(cuTensorMapEncodeTiled(
        tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, data,
        gDim, gStride, bDim, eStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swz,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

void create_tma_3d_f16(CUtensorMap* tmap, void* data,
                       int dim0, int dim1, int batches,
                       int box0, int box1, CUtensorMapSwizzle swz) {
    uint64_t gDim[3] = {uint64_t(dim0), uint64_t(dim1), uint64_t(batches)};
    uint64_t gStride[2] = {
        uint64_t(dim0) * sizeof(half),
        uint64_t(dim0) * uint64_t(dim1) * sizeof(half)};
    uint32_t bDim[3] = {uint32_t(box0), uint32_t(box1), 1};
    uint32_t eStride[3] = {1, 1, 1};
    CHECK_DRV(cuTensorMapEncodeTiled(
        tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, data,
        gDim, gStride, bDim, eStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swz,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

struct DiffStat { float max_abs = 0, max_rel = 0; int bad = 0; };

DiffStat compare(const std::vector<float>& out,
                 const std::vector<float>& ref, float tol) {
    DiffStat s;
    for (size_t i = 0; i < out.size(); ++i) {
        float ae = std::fabs(out[i] - ref[i]);
        float re = ae / std::max(std::fabs(ref[i]), 1e-6f);
        s.max_abs = std::max(s.max_abs, ae);
        s.max_rel = std::max(s.max_rel, re);
        if (ae > tol) ++s.bad;
    }
    return s;
}

// CPU reference: C = A_dense * B_row (standard GEMM with dense A that has zeros)
void cpu_ref_gemm(const std::vector<half>& A_dense,
                  const std::vector<float>& B_row,
                  std::vector<float>& C_ref,
                  int M, int N, int K) {
    C_ref.assign(static_cast<size_t>(M) * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            float a = __half2float(A_dense[static_cast<size_t>(m) * K + k]);
            if (a == 0.0f) continue;
            for (int n = 0; n < N; ++n)
                C_ref[static_cast<size_t>(m) * N + n] += a * B_row[static_cast<size_t>(k) * N + n];
        }
}

void cpu_ref_complex(const std::vector<half>& A_real,
                     const std::vector<half>& A_imag,
                     const std::vector<float>& Z_real,
                     const std::vector<float>& Z_imag,
                     std::vector<float>& C_real,
                     std::vector<float>& C_imag,
                     int batches, int M, int N, int K) {
    const size_t a_stride = static_cast<size_t>(M) * K;
    const size_t z_stride = static_cast<size_t>(K) * N;
    const size_t c_stride = static_cast<size_t>(M) * N;
    C_real.assign(static_cast<size_t>(batches) * c_stride, 0.0f);
    C_imag.assign(static_cast<size_t>(batches) * c_stride, 0.0f);
    for (int batch = 0; batch < batches; ++batch) {
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < K; ++k) {
                float ar = __half2float(A_real[batch * a_stride + static_cast<size_t>(m) * K + k]);
                float ai = __half2float(A_imag[batch * a_stride + static_cast<size_t>(m) * K + k]);
                if (ar == 0.0f && ai == 0.0f) continue;
                for (int n = 0; n < N; ++n) {
                    float zr = Z_real[batch * z_stride + static_cast<size_t>(k) * N + n];
                    float zi = Z_imag[batch * z_stride + static_cast<size_t>(k) * N + n];
                    C_real[batch * c_stride + static_cast<size_t>(m) * N + n] += ar * zr + ai * zi;
                    C_imag[batch * c_stride + static_cast<size_t>(m) * N + n] += ar * zi - ai * zr;
                }
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    int M = (argc > 1) ? std::atoi(argv[1]) : 1024;
    int N = (argc > 2) ? std::atoi(argv[2]) : 1024;
    int K = (argc > 3) ? std::atoi(argv[3]) : 1024;
    int iters  = (argc > 4) ? std::atoi(argv[4]) : 10;
    int warmup = (argc > 5) ? std::atoi(argv[5]) : 3;
    int check  = (argc > 6) ? std::atoi(argv[6]) : 1;
    float tol  = (argc > 7) ? std::atof(argv[7]) : 1e-2f;
    int batches = (argc > 8) ? std::atoi(argv[8]) : 1;

    if (M <= 0 || N <= 0 || K <= 0 || batches <= 0 || iters <= 0 ||
        warmup < 0 || (check != 0 && check != 1)) {
        std::printf("Usage: %s [M N K [iters warmup check(0|1) tol batch]]\n", argv[0]);
        return 1;
    }
    if ((M % BM) || (N % BN) || (K % BK)) {
        std::printf("Requires M%%%d==0 N%%%d==0 K%%%d==0\n", BM, BN, BK);
        return 2;
    }

    CHECK_DRV(cuInit(0));
    cudaDeviceProp prop{}; CHECK_RT(cudaGetDeviceProperties(&prop, 0));

    std::printf("=== Internal dense FP16 Tensor-Core fused power/top-1 control ===\n");
    std::printf("Tile: BM=%d BN=%d BK=%d, %d threads, 4P+8C\n",
                BM, BN, BK, THREADS);
    std::printf("GPU: %s | CC %d.%d\n", prop.name, prop.major, prop.minor);
    std::printf("B=%d M=%d N=%d K=%d | iters=%d warmup=%d check=%d tol=%g\n",
                batches, M, N, K, iters, warmup, check, tol);
    std::printf("Consumer: %dM x %dN, per-warp: %dx%d m16n16 tiles\n",
                WARPS_M, WARPS_N, WM_TILES, WN_TILES);
    std::printf("Producer: planar complex A/B TMA; no transform or metadata\n");
    std::printf("Consumer: real=Ar*Br+Ai*Bi, imag=Ar*Bi-Ai*Br\n");
    std::printf("Dense: %d K-steps/tile, MMA m16n8k16\n", DENSE_K_STEPS);

    const size_t szA = static_cast<size_t>(batches) * M * K;
    std::vector<half> hARealDense(szA), hAImagDense(szA);
    std::mt19937 rng(20260317u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t index = 0; index < szA; ++index) {
        hARealDense[index] = __float2half(dist(rng));
        hAImagDense[index] = __float2half(dist(rng));
    }

    // B input is stored [N,K] for TMA and mirrored [K,N] for the CPU oracle.
    const size_t szBPerBatch = size_t(K) * N;
    const size_t szB = static_cast<size_t>(batches) * szBPerBatch;
    std::vector<half> hBRealCol(szB);
    std::vector<half> hBImagCol(szB);
    std::vector<float> hZRealRow;
    std::vector<float> hZImagRow;
    if (check) {
        hZRealRow.resize(szB);
        hZImagRow.resize(szB);
    }
    std::mt19937 rng_b(20260318u);
    for (int batch = 0; batch < batches; ++batch) {
      const size_t batch_col_base = static_cast<size_t>(batch) * N * K;
      const size_t batch_row_base = static_cast<size_t>(batch) * K * N;
      for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            const size_t col_index = batch_col_base + size_t(n) * K + k;
            hBRealCol[col_index] = __float2half(dist(rng_b));
            hBImagCol[col_index] = __float2half(dist(rng_b));
            if (check) {
                hZRealRow[batch_row_base + size_t(k) * N + n] =
                    __half2float(hBRealCol[col_index]);
                hZImagRow[batch_row_base + size_t(k) * N + n] =
                    __half2float(hBImagCol[col_index]);
            }
        }
      }
    }

    const size_t szC = static_cast<size_t>(batches) * M * N;
    std::vector<float> hCReal, hCImag, hRefReal, hRefImag;
    if (check) {
        hCReal.assign(szC, 0.0f);
        hCImag.assign(szC, 0.0f);
        hRefReal.assign(szC, 0.0f);
        hRefImag.assign(szC, 0.0f);
    }

    // ---- Device allocations ----
    half *dAReal, *dAImag, *dBReal, *dBImag;
    float *dCReal = nullptr, *dCImag = nullptr;
    float *dPower, *dPowerSeparate = nullptr, *dMaxPower;
    uint32_t* dMaxIndex;
    CHECK_RT(cudaMalloc(&dAReal, sizeof(half) * hARealDense.size()));
    CHECK_RT(cudaMalloc(&dAImag, sizeof(half) * hAImagDense.size()));
    CHECK_RT(cudaMalloc(&dBReal,    sizeof(half) * hBRealCol.size()));
    CHECK_RT(cudaMalloc(&dBImag,    sizeof(half) * hBImagCol.size()));
    if (check) {
        CHECK_RT(cudaMalloc(&dCReal, sizeof(float) * szC));
        CHECK_RT(cudaMalloc(&dCImag, sizeof(float) * szC));
    }
    CHECK_RT(cudaMalloc(&dPower,    sizeof(float) * static_cast<size_t>(batches) * M));
    if (check) {
        CHECK_RT(cudaMalloc(&dPowerSeparate,
                            sizeof(float) * static_cast<size_t>(batches) * M));
    }
    CHECK_RT(cudaMalloc(&dMaxPower, sizeof(float) * batches));
    CHECK_RT(cudaMalloc(&dMaxIndex, sizeof(uint32_t) * batches));

    CHECK_RT(cudaMemcpy(dAReal, hARealDense.data(), sizeof(half) * hARealDense.size(), cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(dAImag, hAImagDense.data(), sizeof(half) * hAImagDense.size(), cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(dBReal,    hBRealCol.data(),  sizeof(half) * hBRealCol.size(),    cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(dBImag,    hBImagCol.data(),  sizeof(half) * hBImagCol.size(),    cudaMemcpyHostToDevice));

    // TMA: A [K,M,B] and B [K,N,B].
    CUtensorMap tmaAReal{}, tmaAImag{}, tmaBReal{}, tmaBImag{};
    create_tma_3d_f16(&tmaAReal, dAReal, K, M, batches, BK, BM, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_f16(&tmaAImag, dAImag, K, M, batches, BK, BM, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_f16(&tmaBReal, dBReal, K, N, batches, BK, BN, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_f16(&tmaBImag, dBImag, K, N, batches, BK, BN, CU_TENSOR_MAP_SWIZZLE_128B);

    dim3 grid(N / BN, M / BM, batches);
    dim3 block(THREADS);

    std::printf("Grid: %d x %d x %d = %d blocks\n",
                grid.x, grid.y, grid.z, grid.x * grid.y * grid.z);

    const dim3 power_grid(static_cast<unsigned int>(static_cast<size_t>(batches) * M));
    const size_t power_bytes = sizeof(float) * static_cast<size_t>(batches) * M;
    // Warmup fused beamforming + atomic power epilogue.
    for (int i = 0; i < warmup; ++i) {
        CHECK_RT(cudaMemsetAsync(dPower, 0, power_bytes));
        sm120_dense_complex_fused_power_kernel<true><<<grid, block>>>(
            tmaAReal, tmaAImag, tmaBReal, tmaBImag,
            dCReal, dCImag, dPower, M, N, K);
        argmax_power_kernel<<<batches, 256>>>(dPower, dMaxPower, dMaxIndex, M);
    }
    CHECK_RT(cudaGetLastError());
    CHECK_RT(cudaDeviceSynchronize());

    // Bench fused beamforming + atomic power epilogue, including output clear.
    cudaEvent_t ev0, ev1;
    CHECK_RT(cudaEventCreate(&ev0));
    CHECK_RT(cudaEventCreate(&ev1));
    CHECK_RT(cudaEventRecord(ev0));
    for (int i = 0; i < iters; ++i) {
        CHECK_RT(cudaMemsetAsync(dPower, 0, power_bytes));
        sm120_dense_complex_fused_power_kernel<true><<<grid, block>>>(
            tmaAReal, tmaAImag, tmaBReal, tmaBImag,
            dCReal, dCImag, dPower, M, N, K);
        argmax_power_kernel<<<batches, 256>>>(dPower, dMaxPower, dMaxIndex, M);
    }
    CHECK_RT(cudaGetLastError());
    CHECK_RT(cudaEventRecord(ev1));
    CHECK_RT(cudaEventSynchronize(ev1));

    float kern_ms = 0;
    CHECK_RT(cudaEventElapsedTime(&kern_ms, ev0, ev1));
    kern_ms /= iters;
    // Complex useful work: four real products = 8*M*N*K operations.
    double kern_tflops = 8.0 * batches * M * N * K / (kern_ms * 1e-3) / 1e12;

    std::printf("\nDense fused beamforming+power+top1 : %.9f ms  %.2f TOps/s (beamforming useful work, logical K=%d)\n", kern_ms, kern_tflops, K);
    std::printf(
        "{\"kind\":\"dense_fused_power_top1_e2e\",\"m\":%d,\"n\":%d,"
        "\"k\":%d,\"batch\":%d,\"iterations\":%d,\"warmup\":%d,"
        "\"milliseconds\":%.9g,\"effective_complex_tops\":%.9g,"
        "\"power_values\":%zu,\"top1_values\":%d,"
        "\"complex_workspace_bytes\":%zu}\n",
        M, N, K, batches, iters, warmup, kern_ms, kern_tflops,
        static_cast<size_t>(batches) * M, batches,
        check ? 2 * sizeof(float) * szC : size_t(0));

    int check_failed = 0;
    // Correctness check
    if (check) {
        std::vector<float> hPowerFused(static_cast<size_t>(batches) * M);
        CHECK_RT(cudaMemcpy(hPowerFused.data(), dPower,
                            sizeof(float) * hPowerFused.size(), cudaMemcpyDeviceToHost));
        std::vector<float> hMaxPower(batches);
        std::vector<uint32_t> hMaxIndex(batches);
        CHECK_RT(cudaMemcpy(hMaxPower.data(), dMaxPower,
                            sizeof(float) * batches, cudaMemcpyDeviceToHost));
        CHECK_RT(cudaMemcpy(hMaxIndex.data(), dMaxIndex,
                            sizeof(uint32_t) * batches, cudaMemcpyDeviceToHost));

        // Build the independent S0 reference using the full complex output and
        // the standalone power reducer.
        sm120_dense_complex_fused_power_kernel<false><<<grid, block>>>(
            tmaAReal, tmaAImag, tmaBReal, tmaBImag,
            dCReal, dCImag, dPowerSeparate, M, N, K);
        power_reduce_separate_planes<<<power_grid, 256>>>(
            dCReal, dCImag, dPowerSeparate, M, N);
        CHECK_RT(cudaGetLastError());
        CHECK_RT(cudaDeviceSynchronize());

        std::printf("Computing candidate CPU reference...\n");
        cpu_ref_complex(hARealDense, hAImagDense, hZRealRow, hZImagRow,
                        hRefReal, hRefImag, batches, M, N, K);

        CHECK_RT(cudaMemcpy(hCReal.data(), dCReal, sizeof(float) * szC, cudaMemcpyDeviceToHost));
        CHECK_RT(cudaMemcpy(hCImag.data(), dCImag, sizeof(float) * szC, cudaMemcpyDeviceToHost));
        std::vector<float> hPowerSeparate(static_cast<size_t>(batches) * M);
        CHECK_RT(cudaMemcpy(hPowerSeparate.data(), dPowerSeparate,
                            sizeof(float) * hPowerSeparate.size(), cudaMemcpyDeviceToHost));
        DiffStat st_real = compare(hCReal, hRefReal, tol);
        DiffStat st_imag = compare(hCImag, hRefImag, tol);
        int bad_total = st_real.bad + st_imag.bad;
        std::printf("Check real: max_abs=%.6f max_rel=%.6f bad=%d/%zu\n",
                    st_real.max_abs, st_real.max_rel, st_real.bad, hCReal.size());
        std::printf("Check imag: max_abs=%.6f max_rel=%.6f bad=%d/%zu -> %s\n",
                    st_imag.max_abs, st_imag.max_rel, st_imag.bad, hCImag.size(),
                    bad_total == 0 ? "PASSED" : "FAILED");
        check_failed |= (bad_total != 0);

        float power_max_abs = 0.0f;
        float power_max_rel = 0.0f;
        int power_bad = 0;
        for (int batch_index = 0; batch_index < batches; ++batch_index) {
            for (int row = 0; row < M; ++row) {
                double expected = 0.0;
                const size_t row_base = (static_cast<size_t>(batch_index) * M + row) * N;
                for (int column = 0; column < N; ++column) {
                    const float r = hCReal[row_base + column];
                    const float i = hCImag[row_base + column];
                    expected += static_cast<double>(r) * r + static_cast<double>(i) * i;
                }
                const float observed = hPowerSeparate[static_cast<size_t>(batch_index) * M + row];
                const float expected_f = static_cast<float>(expected);
                const float difference = std::fabs(observed - expected_f);
                const float relative = difference / std::max(std::fabs(expected_f), 1.0e-6f);
                power_max_abs = std::max(power_max_abs, difference);
                power_max_rel = std::max(power_max_rel, relative);
                if (difference > 0.05f + 1.0e-4f * std::fabs(expected_f)) ++power_bad;
            }
        }
        std::printf("Power: max_abs=%.6f max_rel=%.6f bad=%d/%zu -> %s\n",
                    power_max_abs, power_max_rel, power_bad, hPowerSeparate.size(),
                    power_bad == 0 ? "PASSED" : "FAILED");
        check_failed |= (power_bad != 0);

        float fused_max_abs = 0.0f;
        float fused_max_rel = 0.0f;
        int fused_bad = 0;
        for (size_t index = 0; index < hPowerFused.size(); ++index) {
            const float difference = std::fabs(hPowerFused[index] - hPowerSeparate[index]);
            const float relative = difference /
                std::max(std::fabs(hPowerSeparate[index]), 1.0e-6f);
            fused_max_abs = std::max(fused_max_abs, difference);
            fused_max_rel = std::max(fused_max_rel, relative);
            if (difference > 0.10f + 2.0e-4f * std::fabs(hPowerSeparate[index])) ++fused_bad;
        }
        std::printf("Fused-vs-separate power: max_abs=%.6f max_rel=%.6f bad=%d/%zu -> %s\n",
                    fused_max_abs, fused_max_rel, fused_bad, hPowerFused.size(),
                    fused_bad == 0 ? "PASSED" : "FAILED");
        check_failed |= (fused_bad != 0);

        int top1_bad = 0;
        float top1_max_abs = 0.0f;
        for (int batch_index = 0; batch_index < batches; ++batch_index) {
            float expected_value = -std::numeric_limits<float>::infinity();
            uint32_t expected_index = 0xffffffffu;
            for (int row = 0; row < M; ++row) {
                const float value = hPowerFused[static_cast<size_t>(batch_index) * M + row];
                if (value > expected_value ||
                    (value == expected_value && static_cast<uint32_t>(row) < expected_index)) {
                    expected_value = value;
                    expected_index = static_cast<uint32_t>(row);
                }
            }
            top1_max_abs = std::max(top1_max_abs,
                                    std::fabs(hMaxPower[batch_index] - expected_value));
            if (hMaxIndex[batch_index] != expected_index ||
                std::fabs(hMaxPower[batch_index] - expected_value) >
                    1.0e-5f * std::max(std::fabs(expected_value), 1.0f)) {
                ++top1_bad;
            }
        }
        std::printf("Top1: max_abs=%.6f bad=%d/%d -> %s\n",
                    top1_max_abs, top1_bad, batches,
                    top1_bad == 0 ? "PASSED" : "FAILED");
        check_failed |= (top1_bad != 0);
        if (bad_total) {
            int shown = 0;
            for (int i = 0; i < M && shown < 8; ++i)
                for (int j = 0; j < N && shown < 8; ++j) {
                    size_t idx = size_t(i) * N + j;
                    float d = std::max(std::fabs(hCReal[idx] - hRefReal[idx]),
                                       std::fabs(hCImag[idx] - hRefImag[idx]));
                    if (d > tol) {
                        std::printf("  [%d,%d] out=(%.6f,%.6f) ref=(%.6f,%.6f) maxdiff=%.6f\n",
                                    i, j, hCReal[idx], hCImag[idx],
                                    hRefReal[idx], hRefImag[idx], d);
                        ++shown;
                    }
                }
        }
    }

    CHECK_RT(cudaEventDestroy(ev0));
    CHECK_RT(cudaEventDestroy(ev1));
    CHECK_RT(cudaFree(dAReal)); CHECK_RT(cudaFree(dAImag));
    CHECK_RT(cudaFree(dBReal)); CHECK_RT(cudaFree(dBImag));
    if (dCReal) CHECK_RT(cudaFree(dCReal));
    if (dCImag) CHECK_RT(cudaFree(dCImag));
    CHECK_RT(cudaFree(dPower));
    if (dPowerSeparate) CHECK_RT(cudaFree(dPowerSeparate));
    CHECK_RT(cudaFree(dMaxPower)); CHECK_RT(cudaFree(dMaxIndex));
    return check_failed ? 3 : 0;
}
