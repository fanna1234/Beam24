// sm120_sp_complex_f4_v10_half2_io.cu
//
// Diagnostic fused-producer prototype for Beam24 on sm_120a.
//
// Batched full complex fused candidate. The producer loads planar complex B=(Br,Bi),
// applies a unitary local DFT4 in the TMA shared-memory tile, and consumers
// compute conj(U)*Z with four sparse MMA contributions sharing the transformed
// B tile and metadata.
//
// Base: sm_120a FP16 2:4 Sparse GEMM v7c — 4P+4C warp specialization.
// BM=128 BN=128 BK=64(logical), BKP=32(packed), TMA SW64B(A) SW128B(B) + ldmatrix
//
// Warps 0-3: Producer (TMA A+B + GMEM meta load), setmaxnreg.dec 40
// Warps 4-7: Consumer (compute), setmaxnreg.inc 232 → large tile
// Consumer: 2M×2N, each warp 64×64 (WM=4 WN=4), 128 float accumulators
//
// Sparse MMA: mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32
// 2 sparse K-steps per BK=64 tile (each step = one m16n8k32 MMA)
//
// Build:
//   nvcc -gencode arch=compute_120a,code=sm_120a -O3 \
//        sm120_sp_fp16_tiled_v7c.cu -o sm120_sp_fp16_v7c -lcuda
//
// Run:
//   ./sm120_sp_complex_f4_v4_batch [M N K [iters warmup check tol batch]]

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
constexpr int MMA_K_LOGICAL = 32;   // sparse MMA logical K
constexpr int MMA_K_PACKED  = 16;   // sparse MMA packed K (2:4)

constexpr int BM  = 128;
constexpr int BN  = 128;
constexpr int BK  = 64;              // logical K per tile
constexpr int BKP = BK / 2;          // 32 — packed K per tile (2:4 compression)

constexpr int ROW_BYTES_A = BKP * int(sizeof(half));  // 64  → SW64B
constexpr int ROW_BYTES_B = BK  * int(sizeof(half));  // 128 → SW128B

constexpr int SPARSE_K_STEPS = BK / MMA_K_LOGICAL;  // 2

constexpr int PRODUCER_WARPS   = 4;
constexpr int CONSUMER_WARPS   = 8;
constexpr int WARPS_TOTAL      = PRODUCER_WARPS + CONSUMER_WARPS;  // 12
constexpr int THREADS          = WARPS_TOTAL * 32;                 // 384
constexpr int CONSUMER_THREADS = CONSUMER_WARPS * 32;              // 256
constexpr int PRODUCER_THREADS = PRODUCER_WARPS * 32;              // 128

// Consumer 2M x 4N: each warp covers 64x32 of the 128x128 tile.
// Two complex accumulator planes retain the v1 128-float/warp footprint.
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
static_assert(WARPS_M * WARPS_N == CONSUMER_WARPS);

constexpr int WM_TILES = BM / (WARPS_M * MMA_M);  // 128/(2*16)=4
constexpr int WN_TILES = BN / (WARPS_N * MMA_N);   // 128/(4*16)=2

constexpr int PIPE_STAGES = 2;

// B layout: [BN, BK] in SMEM, BK*sizeof(half)=128 bytes/row

// ---- Metadata constants ----
// FP16 2:4: 8 chunks per MMA (MMA_K_LOGICAL/4=8), 4-bit nibble each
// 8 chunks x 4bit = 32 bits/row → front/back split (PTX ISA Fig 125)
// Per m16 segment: 8 groups x 2 (front+back) = 16 words
constexpr int CHUNKS_PER_MMA = MMA_K_LOGICAL / 4;       // 8
constexpr int META_WORDS_PER_WARP = 16;                  // 8 groups x 2
constexpr int M16_SEGMENTS = BM / MMA_M;                 // 8
constexpr int META_WORDS_PER_SP_K = M16_SEGMENTS * META_WORDS_PER_WARP;  // 128
constexpr int META_WORDS_PER_KTILE = SPARSE_K_STEPS * META_WORDS_PER_SP_K;  // 256

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

// Sparse FP16 MMA: m16n8k32, 2:4 structured sparsity
// A: 4 regs (packed), B: 4 regs (full K=32), meta: 1 reg
__device__ __forceinline__ void mma_sp_m16n8k32_f16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
    uint32_t meta) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, "
        "{%0, %1, %2, %3}, "
        "%12, 0x0;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "r"(meta));
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
        if (better_power(value, static_cast<uint32_t>(index),
                         best_value, best_index)) {
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

__device__ __forceinline__ void local_dft4_inplace_sw128(
    half* real_tile, half* imag_tile, int logical_group) {
    constexpr int GROUPS_PER_ROW = BK / 4;
    const int row = logical_group / GROUPS_PER_ROW;
    const int k0 = (logical_group % GROUPS_PER_ROW) * 4;
    const half2 r01 = *reinterpret_cast<const half2*>(
        sw128_b_ptr(real_tile, row, k0 + 0));
    const half2 r23 = *reinterpret_cast<const half2*>(
        sw128_b_ptr(real_tile, row, k0 + 2));
    const half2 i01 = *reinterpret_cast<const half2*>(
        sw128_b_ptr(imag_tile, row, k0 + 0));
    const half2 i23 = *reinterpret_cast<const half2*>(
        sw128_b_ptr(imag_tile, row, k0 + 2));
    const float r0 = __half2float(__low2half(r01));
    const float r1 = __half2float(__high2half(r01));
    const float r2 = __half2float(__low2half(r23));
    const float r3 = __half2float(__high2half(r23));
    const float i0 = __half2float(__low2half(i01));
    const float i1 = __half2float(__high2half(i01));
    const float i2 = __half2float(__low2half(i23));
    const float i3 = __half2float(__high2half(i23));

    const float a_r = r0 + r2;
    const float a_i = i0 + i2;
    const float b_r = r0 - r2;
    const float b_i = i0 - i2;
    const float c_r = r1 + r3;
    const float c_i = i1 + i3;
    const float d_r = r1 - r3;
    const float d_i = i1 - i3;

    *reinterpret_cast<half2*>(sw128_b_ptr(real_tile, row, k0 + 0)) =
        __floats2half2_rn(a_r + c_r, b_r - d_i);
    *reinterpret_cast<half2*>(sw128_b_ptr(imag_tile, row, k0 + 0)) =
        __floats2half2_rn(a_i + c_i, b_i + d_r);
    *reinterpret_cast<half2*>(sw128_b_ptr(real_tile, row, k0 + 2)) =
        __floats2half2_rn(a_r - c_r, b_r + d_i);
    *reinterpret_cast<half2*>(sw128_b_ptr(imag_tile, row, k0 + 2)) =
        __floats2half2_rn(a_i - c_i, b_i - d_r);
}

// ---- Main kernel (4P + 8C warp specialization) ----

template <bool FUSED_POWER>
__global__ __launch_bounds__(THREADS, 1)
void sm120_sp_complex_f4_v10_half2_io_kernel(
    const __grid_constant__ CUtensorMap tmaAReal,
    const __grid_constant__ CUtensorMap tmaAImag,
    const __grid_constant__ CUtensorMap tmaBReal,
    const __grid_constant__ CUtensorMap tmaBImag,
    const uint32_t* __restrict__ gMeta,
    float* __restrict__ CReal,
    float* __restrict__ CImag,
    float* __restrict__ Power,
    int M, int N, int K)
{
    // A: planar complex packed; B: planar complex full.
    static __shared__ __align__(128) half smAReal[PIPE_STAGES][BM * BKP];
    static __shared__ __align__(128) half smAImag[PIPE_STAGES][BM * BKP];
    static __shared__ __align__(128) half smBReal[PIPE_STAGES][BN * BK]; // 2 x 16 KB
    static __shared__ __align__(128) half smBImag[PIPE_STAGES][BN * BK]; // 2 x 16 KB
    static __shared__ __align__(16)  uint32_t smMeta[PIPE_STAGES][META_WORDS_PER_KTILE];  // 2 x 1 KB
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
    const int mt_idx   = blockIdx.y;  // M-tile index for metadata

    constexpr uint32_t TMA_A_BYTES = BM * BKP * sizeof(half);  // 8192
    constexpr uint32_t TMA_B_BYTES = BN * BK  * sizeof(half);  // 16384
    constexpr uint32_t META_BYTES  = META_WORDS_PER_KTILE * sizeof(uint32_t);  // 1024
    constexpr uint32_t TOTAL_TMA   = 2 * TMA_A_BYTES + 2 * TMA_B_BYTES + META_BYTES;

    // ---- Init barriers ----
    if (tid < PIPE_STAGES) {
        init_barrier(&mbar_tma[tid], 1);
        init_barrier(&mbar_ready[tid], PRODUCER_THREADS);
        init_barrier(&mbar_empty[tid], CONSUMER_THREADS);  // 128
    }
    __syncthreads();

    // ---- Prologue: TMA prefetch stage 0 (A + B + Meta all async) ----
    if (tid == 0) {
        const int m_tiles = M / BM;
        const size_t meta_base =
            ((static_cast<size_t>(batch_index) * m_tiles + mt_idx) * k_tiles) *
            META_WORDS_PER_KTILE;
        const uint64_t gmeta0 = reinterpret_cast<uint64_t>(gMeta + meta_base);

        arrive_barrier_tx(&mbar_tma[0], TOTAL_TMA);
        // TMA A real/imag: packed, dim0 coord = kt*BKP, dim1 coord = block_m
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
        // Bulk copy metadata (1024 bytes, no TensorMap needed)
        asm volatile(
            "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
            :: "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&smMeta[0][0]))),
               "l"(gmeta0), "r"(uint32_t(META_BYTES)),
               "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&mbar_tma[0])))
            : "memory");
    }

    // ================================================================
    if (warp_id < PRODUCER_WARPS) {
        // ===== PRODUCER PATH =====
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");

        constexpr int F4_GROUPS_PER_TILE = BN * (BK / 4);
        for (int kt = 0; kt < k_tiles; ++kt) {
            const int cur = kt % PIPE_STAGES;
            const int nxt = (kt + 1) % PIPE_STAGES;
            const int phase = (kt / PIPE_STAGES) & 1;
            wait_barrier(&mbar_tma[cur], phase);

            // Lane 0 schedules the next tile before the cooperative transform,
            // allowing the next TMA transfer to overlap current F4 work.
            if (tid == 0 && kt + 1 < k_tiles) {
                if (kt + 1 >= PIPE_STAGES) {
                    const int empty_phase =
                        ((kt + 1 - PIPE_STAGES) / PIPE_STAGES) & 1;
                    wait_barrier(&mbar_empty[nxt], empty_phase);
                }
                const int kp_next = (kt + 1) * BKP;
                const int k_next  = (kt + 1) * BK;
                const int m_tiles = M / BM;
                const size_t meta_off =
                    (((static_cast<size_t>(batch_index) * m_tiles + mt_idx) * k_tiles) +
                     (kt + 1)) * META_WORDS_PER_KTILE;
                const uint64_t gmeta_nxt = reinterpret_cast<uint64_t>(gMeta + meta_off);

                arrive_barrier_tx(&mbar_tma[nxt], TOTAL_TMA);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.tile"
                    ".mbarrier::complete_tx::bytes [%0], [%1, {%2,%3,%4}], [%5];\n"
                    :: "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&smAReal[nxt][0]))),
                       "l"(reinterpret_cast<uint64_t>(&tmaAReal)),
                       "r"(kp_next), "r"(block_m), "r"(batch_index),
                       "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&mbar_tma[nxt])))
                    : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.tile"
                    ".mbarrier::complete_tx::bytes [%0], [%1, {%2,%3,%4}], [%5];\n"
                    :: "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&smAImag[nxt][0]))),
                       "l"(reinterpret_cast<uint64_t>(&tmaAImag)),
                       "r"(kp_next), "r"(block_m), "r"(batch_index),
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
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                    :: "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&smMeta[nxt][0]))),
                       "l"(gmeta_nxt), "r"(uint32_t(META_BYTES)),
                       "l"(static_cast<uint64_t>(__cvta_generic_to_shared(&mbar_tma[nxt])))
                    : "memory");
            }

            for (int group_index = tid; group_index < F4_GROUPS_PER_TILE;
                 group_index += PRODUCER_THREADS) {
                local_dft4_inplace_sw128(
                    &smBReal[cur][0], &smBImag[cur][0], group_index);
            }
            // All 128 producer threads arrive only after their disjoint K4
            // groups are transformed. Consumers wait on this ready barrier.
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


            // ---- Compute: 2 sparse K-steps (sp_k replaces dense kh) ----
            #pragma unroll
            for (int sp_k = 0; sp_k < SPARSE_K_STEPS; ++sp_k) {
                // B offset bytes: ((lane/8)*8 + sp_k*32) * sizeof(half) = (lane/8)*16 + sp_k*64
                const int b_k_byte_base = b_offk * int(sizeof(half)) + sp_k * 64;

                // Pre-load B via ldmatrix (SW128B)
                // ldmatrix.x4 loads 8N × 32K FP16 path for mma.sp m16n8k32
                uint32_t br_lo[WN_TILES][4];
                uint32_t br_hi[WN_TILES][4];
                uint32_t bi_lo[WN_TILES][4];
                uint32_t bi_hi[WN_TILES][4];

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
                    ldmatrix_x4_shared_b16(br_lo[wn], smBReal_base + lo_offset);
                    ldmatrix_x4_shared_b16(br_hi[wn], smBReal_base + hi_offset);
                    ldmatrix_x4_shared_b16(bi_lo[wn], smBImag_base + lo_offset);
                    ldmatrix_x4_shared_b16(bi_hi[wn], smBImag_base + hi_offset);
                }

                // Compute: WM_TILES x WN_TILES x 2 (lo/hi N8) sparse MMAs
                #pragma unroll
                for (int wm = 0; wm < WM_TILES; ++wm) {
                    // A via ldmatrix (SW64B): 16M × 16K_packed FP16 → 4 regs
                    const int a_row = cons_wm * WM_TILES * MMA_M + wm * MMA_M + (lane & 15);
                    const int a_k_byte = ((lane >> 4) * 8 + sp_k * MMA_K_PACKED) * int(sizeof(half));
                    const int a_lin = a_row * ROW_BYTES_A + a_k_byte;
                    const int a_mask = ((a_row >> 1) & 3) << 4;  // SW64B de-swizzle (64B rows, 128B bank cycle)
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

                    // Metadata: index into smMeta[cur]
                    const int mma_m_idx = cons_wm * WM_TILES + wm;  // 0..7
                    const uint32_t meta = smMeta[cur][
                        sp_k * META_WORDS_PER_SP_K +
                        mma_m_idx * META_WORDS_PER_WARP +
                        group_id * 2 + (tid_in_group & 1)];

                    // conj(U)*Z:
                    // real = Ur*Zr + Ui*Zi
                    // imag = Ur*Zi - Ui*Zr
                    #pragma unroll
                    for (int wn = 0; wn < WN_TILES; ++wn) {
                        mma_sp_m16n8k32_f16(
                            c_real[wm][wn][0], c_real[wm][wn][1],
                            c_real[wm][wn][2], c_real[wm][wn][3],
                            ar[0], ar[1], ar[2], ar[3],
                            br_lo[wn][0], br_lo[wn][1], br_lo[wn][2], br_lo[wn][3],
                            meta);
                        mma_sp_m16n8k32_f16(
                            c_real[wm][wn][0], c_real[wm][wn][1],
                            c_real[wm][wn][2], c_real[wm][wn][3],
                            ai[0], ai[1], ai[2], ai[3],
                            bi_lo[wn][0], bi_lo[wn][1], bi_lo[wn][2], bi_lo[wn][3],
                            meta);
                        mma_sp_m16n8k32_f16(
                            c_imag[wm][wn][0], c_imag[wm][wn][1],
                            c_imag[wm][wn][2], c_imag[wm][wn][3],
                            ar[0], ar[1], ar[2], ar[3],
                            bi_lo[wn][0], bi_lo[wn][1], bi_lo[wn][2], bi_lo[wn][3],
                            meta);
                        mma_sp_m16n8k32_f16(
                            c_imag[wm][wn][0], c_imag[wm][wn][1],
                            c_imag[wm][wn][2], c_imag[wm][wn][3],
                            ai_neg[0], ai_neg[1], ai_neg[2], ai_neg[3],
                            br_lo[wn][0], br_lo[wn][1], br_lo[wn][2], br_lo[wn][3],
                            meta);

                        mma_sp_m16n8k32_f16(
                            c_real[wm][wn][4], c_real[wm][wn][5],
                            c_real[wm][wn][6], c_real[wm][wn][7],
                            ar[0], ar[1], ar[2], ar[3],
                            br_hi[wn][0], br_hi[wn][1], br_hi[wn][2], br_hi[wn][3],
                            meta);
                        mma_sp_m16n8k32_f16(
                            c_real[wm][wn][4], c_real[wm][wn][5],
                            c_real[wm][wn][6], c_real[wm][wn][7],
                            ai[0], ai[1], ai[2], ai[3],
                            bi_hi[wn][0], bi_hi[wn][1], bi_hi[wn][2], bi_hi[wn][3],
                            meta);
                        mma_sp_m16n8k32_f16(
                            c_imag[wm][wn][4], c_imag[wm][wn][5],
                            c_imag[wm][wn][6], c_imag[wm][wn][7],
                            ar[0], ar[1], ar[2], ar[3],
                            bi_hi[wn][0], bi_hi[wn][1], bi_hi[wn][2], bi_hi[wn][3],
                            meta);
                        mma_sp_m16n8k32_f16(
                            c_imag[wm][wn][4], c_imag[wm][wn][5],
                            c_imag[wm][wn][6], c_imag[wm][wn][7],
                            ai_neg[0], ai_neg[1], ai_neg[2], ai_neg[3],
                            br_hi[wn][0], br_hi[wn][1], br_hi[wn][2], br_hi[wn][3],
                            meta);
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

static uint32_t compress_chunk_2_4(const float in4[4], half out2[2], float dense4[4]) {
    float mag[4];
    for (int i = 0; i < 4; ++i) {
        mag[i] = std::fabs(in4[i]);
    }

    int i0 = 0;
    int i1 = 1;
    if (mag[i0] < mag[i1]) {
        std::swap(i0, i1);
    }
    for (int i = 2; i < 4; ++i) {
        if (mag[i] > mag[i0]) {
            i1 = i0;
            i0 = i;
        } else if (mag[i] > mag[i1]) {
            i1 = i;
        }
    }
    if (i0 > i1) {
        std::swap(i0, i1);
    }

    for (int i = 0; i < 4; ++i) {
        dense4[i] = 0.0f;
    }
    dense4[i0] = in4[i0];
    dense4[i1] = in4[i1];

    out2[0] = __float2half(in4[i0]);
    out2[1] = __float2half(in4[i1]);
    return static_cast<uint32_t>((i0 & 0x3) | ((i1 & 0x3) << 2));
}

// Generate 2:4 sparse A matrix.
// A_dense: [M, K] with explicit zeros (for CPU reference)
// A_packed: [M, K/2] compressed payload
// row_meta: [M, K_tiles, SPARSE_K_STEPS] raw 32-bit words (8 chunks x 4 bits)
static void generate_sparse_a_complex_fp16_2_4(
    int M,
    int K,
    uint32_t seed,
    std::vector<half>& A_real_dense,
    std::vector<half>& A_imag_dense,
    std::vector<half>& A_real_packed,
    std::vector<half>& A_imag_packed,
    std::vector<uint32_t>& row_meta)
{
    const int K_tiles = K / BK;
    const int Kp = K / 2;

    A_real_dense.assign(static_cast<size_t>(M) * K, __float2half(0.0f));
    A_imag_dense.assign(static_cast<size_t>(M) * K, __float2half(0.0f));
    A_real_packed.resize(static_cast<size_t>(M) * Kp);
    A_imag_packed.resize(static_cast<size_t>(M) * Kp);
    row_meta.resize(static_cast<size_t>(M) * K_tiles * SPARSE_K_STEPS);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int m = 0; m < M; ++m) {
        for (int kt = 0; kt < K_tiles; ++kt) {
            for (int sp_k = 0; sp_k < SPARSE_K_STEPS; ++sp_k) {
                const int k_base = kt * BK + sp_k * MMA_K_LOGICAL;
                const int kp_base = kt * BKP + sp_k * MMA_K_PACKED;
                uint32_t meta = 0;

                for (int chunk = 0; chunk < CHUNKS_PER_MMA; ++chunk) {
                    float in4[4];
                    float imag4[4];
                    for (int i = 0; i < 4; ++i) {
                        in4[i] = dist(rng);
                        imag4[i] = dist(rng);
                    }

                    half packed2[2];
                    float dense4[4];
                    const uint32_t nib = compress_chunk_2_4(in4, packed2, dense4);
                    meta |= (nib << (chunk * 4));

                    const int k_chunk_base = k_base + chunk * 4;
                    const int idx0 = nib & 0x3;
                    const int idx1 = (nib >> 2) & 0x3;
                    for (int i = 0; i < 4; ++i) {
                        const size_t index = static_cast<size_t>(m) * K + k_chunk_base + i;
                        const half original_real = __float2half(dense4[i]);
                        const half original_imag = __float2half(
                            (i == idx0 || i == idx1) ? imag4[i] : 0.0f);
                        A_real_dense[index] = __float2half(
                            0.5f * __half2float(original_real));
                        A_imag_dense[index] = __float2half(
                            0.5f * __half2float(original_imag));
                    }

                    const int kp_chunk_base = kp_base + chunk * 2;
                    A_real_packed[static_cast<size_t>(m) * Kp + (kp_chunk_base + 0)] =
                        __float2half(0.5f * __half2float(packed2[0]));
                    A_real_packed[static_cast<size_t>(m) * Kp + (kp_chunk_base + 1)] =
                        __float2half(0.5f * __half2float(packed2[1]));
                    const half original_imag0 = __float2half(imag4[idx0]);
                    const half original_imag1 = __float2half(imag4[idx1]);
                    A_imag_packed[static_cast<size_t>(m) * Kp + (kp_chunk_base + 0)] =
                        __float2half(0.5f * __half2float(original_imag0));
                    A_imag_packed[static_cast<size_t>(m) * Kp + (kp_chunk_base + 1)] =
                        __float2half(0.5f * __half2float(original_imag1));
                }

                row_meta[static_cast<size_t>(m) * K_tiles * SPARSE_K_STEPS
                    + kt * SPARSE_K_STEPS + sp_k] = meta;
            }
        }
    }
}

// Build per-tile metadata words from row_meta.
// Layout: [M_tiles, K_tiles, SPARSE_K_STEPS, M16_SEGMENTS, META_WORDS_PER_WARP]
//         [M_tiles x K_tiles x META_WORDS_PER_KTILE]
// For each segment seg (0..7), groups 0..7 use rows:
//   row0 = m_base + seg*16 + group
//   row1 = row0 + 8
// front/back split follows PTX ordered-metadata format.
static void build_meta_words_per_tile(
    int M,
    int K,
    const std::vector<uint32_t>& row_meta,
    std::vector<uint32_t>& meta_words)
{
    const int K_tiles = K / BK;
    const int M_tiles = M / BM;
    meta_words.resize(static_cast<size_t>(M_tiles) * K_tiles * META_WORDS_PER_KTILE);

    for (int mt = 0; mt < M_tiles; ++mt) {
        const int m_base = mt * BM;
        for (int kt = 0; kt < K_tiles; ++kt) {
            for (int sp_k = 0; sp_k < SPARSE_K_STEPS; ++sp_k) {
                const size_t tile_base =
                    (static_cast<size_t>(mt) * K_tiles + kt) * META_WORDS_PER_KTILE
                    + sp_k * META_WORDS_PER_SP_K;

                for (int seg = 0; seg < M16_SEGMENTS; ++seg) {
                    const int seg_row_base = m_base + seg * MMA_M;
                    for (int group = 0; group < 8; ++group) {
                        const int row0 = seg_row_base + group;
                        const int row1 = row0 + 8;
                        const uint32_t m0 = row_meta[
                            static_cast<size_t>(row0) * K_tiles * SPARSE_K_STEPS
                            + kt * SPARSE_K_STEPS + sp_k];
                        const uint32_t m1 = row_meta[
                            static_cast<size_t>(row1) * K_tiles * SPARSE_K_STEPS
                            + kt * SPARSE_K_STEPS + sp_k];

                        // front/back split: chunks 0..3 in low16, chunks 4..7 in high16
                        const uint32_t front = (m0 & 0xFFFFu) | ((m1 & 0xFFFFu) << 16);
                        const uint32_t back  = ((m0 >> 16) & 0xFFFFu) | (m1 & 0xFFFF0000u);

                        const int idx = seg * META_WORDS_PER_WARP + group * 2;
                        meta_words[tile_base + idx + 0] = front;
                        meta_words[tile_base + idx + 1] = back;
                    }
                }
            }
        }
    }
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
    int use_graph = (argc > 9) ? std::atoi(argv[9]) : 0;

    if (M <= 0 || N <= 0 || K <= 0 || batches <= 0 || iters <= 0 ||
        warmup < 0 || (check != 0 && check != 1) ||
        (use_graph != 0 && use_graph != 1)) {
        std::printf("Usage: %s [M N K [iters warmup check(0|1) tol batch graph(0|1)]]\n", argv[0]);
        return 1;
    }
    if ((M % BM) || (N % BN) || (K % BK)) {
        std::printf("Requires M%%%d==0 N%%%d==0 K%%%d==0\n", BM, BN, BK);
        return 2;
    }

    CHECK_DRV(cuInit(0));
    cudaDeviceProp prop{}; CHECK_RT(cudaGetDeviceProperties(&prop, 0));

    std::printf("=== Beam24 fused local-DFT + complex Sparse FP16 GEMM v10 half2 F4 IO ===\n");
    std::printf("Tile: BM=%d BN=%d BK=%d BKP=%d, %d threads, 4P+8C\n",
                BM, BN, BK, BKP, THREADS);
    std::printf("GPU: %s | CC %d.%d\n", prop.name, prop.major, prop.minor);
    std::printf("B=%d M=%d N=%d K=%d | iters=%d warmup=%d check=%d tol=%g graph=%d\n",
                batches, M, N, K, iters, warmup, check, tol, use_graph);
    std::printf("Consumer: %dM x %dN, per-warp: %dx%d m16n16 tiles (128 accum regs)\n",
                WARPS_M, WARPS_N, WM_TILES, WN_TILES);
    std::printf("Producer: planar complex Br/Bi TMA + in-place unitary F4\n");
    std::printf("Consumer: real=Ur*Zr+Ui*Zi, imag=Ur*Zi-Ui*Zr\n");
    std::printf("Sparse: %d K-steps/tile, MMA m16n8k32, metadata %d words/tile\n",
                SPARSE_K_STEPS, META_WORDS_PER_KTILE);

    // ---- Generate sparse data ----
    const int Kp = K / 2;
    std::vector<half> hARealDense, hAImagDense, hARealPacked, hAImagPacked;
    std::vector<uint32_t> hMetaWords;
    if (check) {
        hARealDense.reserve(static_cast<size_t>(batches) * M * K);
        hAImagDense.reserve(static_cast<size_t>(batches) * M * K);
    }
    hARealPacked.reserve(static_cast<size_t>(batches) * M * Kp);
    hAImagPacked.reserve(static_cast<size_t>(batches) * M * Kp);
    for (int batch = 0; batch < batches; ++batch) {
        std::vector<half> ar_dense, ai_dense, ar_packed, ai_packed;
        std::vector<uint32_t> row_meta, meta_words;
        generate_sparse_a_complex_fp16_2_4(
            M, K, 20260317u + static_cast<uint32_t>(batch),
            ar_dense, ai_dense, ar_packed, ai_packed, row_meta);
        build_meta_words_per_tile(M, K, row_meta, meta_words);
        if (check) {
            hARealDense.insert(hARealDense.end(), ar_dense.begin(), ar_dense.end());
            hAImagDense.insert(hAImagDense.end(), ai_dense.begin(), ai_dense.end());
        }
        hARealPacked.insert(hARealPacked.end(), ar_packed.begin(), ar_packed.end());
        hAImagPacked.insert(hAImagPacked.end(), ai_packed.begin(), ai_packed.end());
        hMetaWords.insert(hMetaWords.end(), meta_words.begin(), meta_words.end());
    }

    // B input: planar complex [N,K] row-major for device TMA. The CPU
    // reference consumes the transformed real plane Zr as [K,N].
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
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int batch = 0; batch < batches; ++batch) {
      const size_t batch_col_base = static_cast<size_t>(batch) * N * K;
      const size_t batch_row_base = static_cast<size_t>(batch) * K * N;
      for (int n = 0; n < N; ++n) {
        for (int group = 0; group < K / 4; ++group) {
            float r[4], i[4];
            for (int lane = 0; lane < 4; ++lane) {
                r[lane] = dist(rng_b);
                i[lane] = dist(rng_b);
                const size_t col_index = batch_col_base + size_t(n) * K + group * 4 + lane;
                hBRealCol[col_index] = __float2half(r[lane]);
                hBImagCol[col_index] = __float2half(i[lane]);
                if (check) {
                    // Reference must start from the same rounded FP16 inputs.
                    r[lane] = __half2float(hBRealCol[col_index]);
                    i[lane] = __half2float(hBImagCol[col_index]);
                }
            }
            if (check) {
                float zr[4] = {
                    r[0] + r[1] + r[2] + r[3],
                    r[0] - i[1] - r[2] + i[3],
                    r[0] - r[1] + r[2] - r[3],
                    r[0] + i[1] - r[2] - i[3],
                };
                float zi[4] = {
                    i[0] + i[1] + i[2] + i[3],
                    i[0] + r[1] - i[2] - r[3],
                    i[0] - i[1] + i[2] - i[3],
                    i[0] - r[1] - i[2] + r[3],
                };
                for (int lane = 0; lane < 4; ++lane) {
                    // The device stores Zr as FP16 before sparse MMA consumption.
                    hZRealRow[batch_row_base + size_t(group * 4 + lane) * N + n] =
                        __half2float(__float2half(zr[lane]));
                    hZImagRow[batch_row_base + size_t(group * 4 + lane) * N + n] =
                        __half2float(__float2half(zi[lane]));
                }
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
    half *dARealPacked, *dAImagPacked, *dBReal, *dBImag;
    float *dCReal = nullptr, *dCImag = nullptr;
    float *dPower, *dPowerSeparate = nullptr, *dMaxPower;
    uint32_t* dMaxIndex;
    uint32_t *dMeta;
    CHECK_RT(cudaMalloc(&dARealPacked, sizeof(half) * hARealPacked.size()));
    CHECK_RT(cudaMalloc(&dAImagPacked, sizeof(half) * hAImagPacked.size()));
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
    CHECK_RT(cudaMalloc(&dMeta,     sizeof(uint32_t) * hMetaWords.size()));

    CHECK_RT(cudaMemcpy(dARealPacked, hARealPacked.data(), sizeof(half) * hARealPacked.size(), cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(dAImagPacked, hAImagPacked.data(), sizeof(half) * hAImagPacked.size(), cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(dBReal,    hBRealCol.data(),  sizeof(half) * hBRealCol.size(),    cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(dBImag,    hBImagCol.data(),  sizeof(half) * hBImagCol.size(),    cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(dMeta,     hMetaWords.data(), sizeof(uint32_t) * hMetaWords.size(), cudaMemcpyHostToDevice));

    // TMA: A_packed [Kp,M,B] and B [K,N,B].
    CUtensorMap tmaAReal{}, tmaAImag{}, tmaBReal{}, tmaBImag{};
    create_tma_3d_f16(&tmaAReal, dARealPacked, Kp, M, batches, BKP, BM, CU_TENSOR_MAP_SWIZZLE_64B);
    create_tma_3d_f16(&tmaAImag, dAImagPacked, Kp, M, batches, BKP, BM, CU_TENSOR_MAP_SWIZZLE_64B);
    create_tma_3d_f16(&tmaBReal, dBReal, K, N, batches, BK, BN, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_f16(&tmaBImag, dBImag, K, N, batches, BK, BN, CU_TENSOR_MAP_SWIZZLE_128B);

    dim3 grid(N / BN, M / BM, batches);
    dim3 block(THREADS);

    std::printf("Grid: %d x %d x %d = %d blocks\n",
                grid.x, grid.y, grid.z, grid.x * grid.y * grid.z);

    const dim3 power_grid(static_cast<unsigned int>(static_cast<size_t>(batches) * M));
    const size_t power_bytes = sizeof(float) * static_cast<size_t>(batches) * M;
    cudaStream_t exec_stream = nullptr;
    CHECK_RT(cudaStreamCreateWithFlags(&exec_stream, cudaStreamNonBlocking));

    auto enqueue_direct = [&]() {
        CHECK_RT(cudaMemsetAsync(dPower, 0, power_bytes, exec_stream));
        sm120_sp_complex_f4_v10_half2_io_kernel<true><<<grid, block, 0, exec_stream>>>(
            tmaAReal, tmaAImag, tmaBReal, tmaBImag, dMeta,
            dCReal, dCImag, dPower, M, N, K);
        argmax_power_kernel<<<batches, 256, 0, exec_stream>>>(
            dPower, dMaxPower, dMaxIndex, M);
    };

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    if (use_graph) {
        CHECK_RT(cudaStreamBeginCapture(exec_stream, cudaStreamCaptureModeGlobal));
        enqueue_direct();
        CHECK_RT(cudaStreamEndCapture(exec_stream, &graph));
        CHECK_RT(cudaGraphInstantiate(&graph_exec, graph, 0));
    }

    auto enqueue_frame = [&]() {
        if (use_graph)
            CHECK_RT(cudaGraphLaunch(graph_exec, exec_stream));
        else
            enqueue_direct();
    };

    // Warmup complete beamforming + power + DOA sequence.
    for (int i = 0; i < warmup; ++i) {
        enqueue_frame();
    }
    CHECK_RT(cudaGetLastError());
    CHECK_RT(cudaStreamSynchronize(exec_stream));

    // Bench the complete device-resident sequence.
    cudaEvent_t ev0, ev1;
    CHECK_RT(cudaEventCreate(&ev0));
    CHECK_RT(cudaEventCreate(&ev1));
    CHECK_RT(cudaEventRecord(ev0, exec_stream));
    for (int i = 0; i < iters; ++i) {
        enqueue_frame();
    }
    CHECK_RT(cudaGetLastError());
    CHECK_RT(cudaEventRecord(ev1, exec_stream));
    CHECK_RT(cudaEventSynchronize(ev1));

    float kern_ms = 0;
    CHECK_RT(cudaEventElapsedTime(&kern_ms, ev0, ev1));
    kern_ms /= iters;
    // Complex useful work: four real products = 8*M*N*K operations.
    double kern_tflops = 8.0 * batches * M * N * K / (kern_ms * 1e-3) / 1e12;

    std::printf("\nBeamforming+power+DOA%s : %.9f ms  %.2f TOps/s (beamforming useful work, logical K=%d)\n",
                use_graph ? " Graph" : " direct", kern_ms, kern_tflops, K);
    std::printf(
        "{\"kind\":\"beam24_doa_e2e\",\"m\":%d,\"n\":%d,"
        "\"k\":%d,\"batch\":%d,\"iterations\":%d,\"warmup\":%d,"
        "\"milliseconds\":%.9g,\"effective_complex_tops\":%.9g,"
        "\"power_values\":%zu,\"doa_values\":%d,\"graph\":%s,"
        "\"complex_workspace_bytes\":%zu}\n",
        M, N, K, batches, iters, warmup, kern_ms, kern_tflops,
        static_cast<size_t>(batches) * M, batches,
        use_graph ? "true" : "false",
        check ? 2 * sizeof(float) * szC : size_t(0));

    int check_failed = 0;
    // Correctness check
    if (check) {
        std::vector<float> hPowerFused(static_cast<size_t>(batches) * M);
        std::vector<float> hMaxPower(batches);
        std::vector<uint32_t> hMaxIndex(batches);
        CHECK_RT(cudaMemcpy(hPowerFused.data(), dPower,
                            sizeof(float) * hPowerFused.size(), cudaMemcpyDeviceToHost));
        CHECK_RT(cudaMemcpy(hMaxPower.data(), dMaxPower,
                            sizeof(float) * hMaxPower.size(), cudaMemcpyDeviceToHost));
        CHECK_RT(cudaMemcpy(hMaxIndex.data(), dMaxIndex,
                            sizeof(uint32_t) * hMaxIndex.size(), cudaMemcpyDeviceToHost));

        // Build the independent S0 reference using the full complex output and
        // the standalone power reducer.
        sm120_sp_complex_f4_v10_half2_io_kernel<false><<<grid, block, 0, exec_stream>>>(
            tmaAReal, tmaAImag, tmaBReal, tmaBImag, dMeta,
            dCReal, dCImag, dPowerSeparate, M, N, K);
        power_reduce_separate_planes<<<power_grid, 256, 0, exec_stream>>>(
            dCReal, dCImag, dPowerSeparate, M, N);
        CHECK_RT(cudaGetLastError());
        CHECK_RT(cudaStreamSynchronize(exec_stream));

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

        // Reconstruct the original normalized-F4 semantic contract. Power-of-two
        // scaling should recover the original FP16 A and Z values for normal
        // finite inputs, but this is checked rather than assumed.
        std::vector<half> hARealOriginal(hARealDense.size());
        std::vector<half> hAImagOriginal(hAImagDense.size());
        for (size_t index = 0; index < hARealDense.size(); ++index) {
            hARealOriginal[index] = __float2half(2.0f * __half2float(hARealDense[index]));
            hAImagOriginal[index] = __float2half(2.0f * __half2float(hAImagDense[index]));
        }
        std::vector<float> hZRealNormalized(hZRealRow.size());
        std::vector<float> hZImagNormalized(hZImagRow.size());
        for (size_t index = 0; index < hZRealRow.size(); ++index) {
            hZRealNormalized[index] = __half2float(__float2half(0.5f * hZRealRow[index]));
            hZImagNormalized[index] = __half2float(__float2half(0.5f * hZImagRow[index]));
        }
        std::vector<float> hSemanticReal(szC, 0.0f), hSemanticImag(szC, 0.0f);
        cpu_ref_complex(hARealOriginal, hAImagOriginal,
                        hZRealNormalized, hZImagNormalized,
                        hSemanticReal, hSemanticImag, batches, M, N, K);
        const DiffStat semantic_real = compare(hCReal, hSemanticReal, tol);
        const DiffStat semantic_imag = compare(hCImag, hSemanticImag, tol);
        const int semantic_bad = semantic_real.bad + semantic_imag.bad;
        std::printf("Semantic real: max_abs=%.6f max_rel=%.6f bad=%d/%zu\n",
                    semantic_real.max_abs, semantic_real.max_rel,
                    semantic_real.bad, hCReal.size());
        std::printf("Semantic imag: max_abs=%.6f max_rel=%.6f bad=%d/%zu -> %s\n",
                    semantic_imag.max_abs, semantic_imag.max_rel,
                    semantic_imag.bad, hCImag.size(),
                    semantic_bad == 0 ? "PASSED" : "FAILED");
        check_failed |= (semantic_bad != 0);

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

        int doa_bad = 0;
        float doa_power_max_abs = 0.0f;
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
            doa_power_max_abs = std::max(
                doa_power_max_abs, std::fabs(hMaxPower[batch_index] - expected_value));
            if (hMaxIndex[batch_index] != expected_index ||
                std::fabs(hMaxPower[batch_index] - expected_value) >
                    1.0e-5f * std::max(std::fabs(expected_value), 1.0f)) {
                ++doa_bad;
            }
        }
        std::printf("DOA top1: max_power_abs=%.6f bad=%d/%d -> %s\n",
                    doa_power_max_abs, doa_bad, batches,
                    doa_bad == 0 ? "PASSED" : "FAILED");
        check_failed |= (doa_bad != 0);
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
    if (graph_exec) CHECK_RT(cudaGraphExecDestroy(graph_exec));
    if (graph) CHECK_RT(cudaGraphDestroy(graph));
    CHECK_RT(cudaStreamDestroy(exec_stream));
    CHECK_RT(cudaFree(dARealPacked)); CHECK_RT(cudaFree(dAImagPacked));
    CHECK_RT(cudaFree(dBReal)); CHECK_RT(cudaFree(dBImag));
    if (dCReal) CHECK_RT(cudaFree(dCReal));
    if (dCImag) CHECK_RT(cudaFree(dCImag));
    CHECK_RT(cudaFree(dPower));
    if (dPowerSeparate) CHECK_RT(cudaFree(dPowerSeparate));
    CHECK_RT(cudaFree(dMaxPower)); CHECK_RT(cudaFree(dMaxIndex));
    CHECK_RT(cudaFree(dMeta));
    return check_failed ? 3 : 0;
}
