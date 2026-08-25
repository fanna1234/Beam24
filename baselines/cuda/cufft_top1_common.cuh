#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math_constants.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call) do {                                                   \
    cudaError_t error_ = (call);                                                \
    if (error_ != cudaSuccess) {                                                \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                     cudaGetErrorString(error_));                               \
        std::exit(1);                                                           \
    }                                                                           \
} while (0)

#define CHECK_CUFFT(call) do {                                                  \
    cufftResult error_ = (call);                                                \
    if (error_ != CUFFT_SUCCESS) {                                              \
        std::fprintf(stderr, "cuFFT error %s:%d: %d\n", __FILE__, __LINE__,   \
                     int(error_));                                              \
        std::exit(1);                                                           \
    }                                                                           \
} while (0)

__global__ void initialize_half_planes(half* real, half* imag, size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const uint32_t value = static_cast<uint32_t>(index) * 1664525u + 1013904223u;
    const float r = (float(value & 0xffffu) / 32768.0f - 1.0f) * 0.5f;
    const float i = (float((value >> 16) & 0xffffu) / 32768.0f - 1.0f) * 0.5f;
    real[index] = __float2half_rn(r);
    imag[index] = __float2half_rn(i);
}

__global__ void pack_zero_pad(const half* real, const half* imag,
                              cufftComplex* output, int sensors, int fft_length,
                              size_t transforms) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = transforms * static_cast<size_t>(fft_length);
    if (index >= total) return;
    const int column = int(index % fft_length);
    const size_t transform = index / fft_length;
    cufftComplex value{0.0f, 0.0f};
    if (column < sensors) {
        const size_t input_index = transform * sensors + column;
        value.x = __half2float(real[input_index]);
        value.y = __half2float(imag[input_index]);
    }
    output[index] = value;
}

__device__ __forceinline__ bool better(float value, uint32_t index,
                                        float best, uint32_t best_index) {
    return value > best || (value == best && index < best_index);
}

__global__ void argmax_kernel(const float* power, float* max_power,
                              uint32_t* max_index, int beams) {
    const int batch_index = int(blockIdx.x);
    const float* row = power + static_cast<size_t>(batch_index) * beams;
    float best_value = -CUDART_INF_F;
    uint32_t best_index = 0xffffffffu;
    for (int beam = threadIdx.x; beam < beams; beam += blockDim.x) {
        const float value = row[beam];
        if (better(value, uint32_t(beam), best_value, best_index)) {
            best_value = value;
            best_index = uint32_t(beam);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_value = __shfl_down_sync(0xffffffffu, best_value, offset);
        const uint32_t other_index =
            __shfl_down_sync(0xffffffffu, best_index, offset);
        if (better(other_value, other_index, best_value, best_index)) {
            best_value = other_value;
            best_index = other_index;
        }
    }
    __shared__ float values[8];
    __shared__ uint32_t indices[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        values[warp] = best_value;
        indices[warp] = best_index;
    }
    __syncthreads();
    if (warp == 0) {
        best_value = lane < 8 ? values[lane] : -CUDART_INF_F;
        best_index = lane < 8 ? indices[lane] : 0xffffffffu;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_value =
                __shfl_down_sync(0xffffffffu, best_value, offset);
            const uint32_t other_index =
                __shfl_down_sync(0xffffffffu, best_index, offset);
            if (better(other_value, other_index, best_value, best_index)) {
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

