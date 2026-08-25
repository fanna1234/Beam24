#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#ifndef BEAM24_CUFFT_HALF
#define BEAM24_CUFFT_HALF 0
#endif

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
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint32_t value = static_cast<uint32_t>(index) * 1664525u + 1013904223u;
    float r = (float(value & 0xffffu) / 32768.0f - 1.0f) * 0.5f;
    float i = (float((value >> 16) & 0xffffu) / 32768.0f - 1.0f) * 0.5f;
    real[index] = __float2half_rn(r);
    imag[index] = __float2half_rn(i);
}

__global__ void pack_zero_pad(const half* real, const half* imag,
                              cufftComplex* output, int K, int M,
                              size_t transforms) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = transforms * static_cast<size_t>(M);
    if (index >= total) return;
    int column = int(index % M);
    size_t transform = index / M;
    cufftComplex value{0.0f, 0.0f};
    if (column < K) {
        size_t input_index = transform * K + column;
        value.x = __half2float(real[input_index]);
        value.y = __half2float(imag[input_index]);
    }
    output[index] = value;
}

__global__ void pack_zero_pad_half(const half* real, const half* imag,
                                   half2* output, int K, int M,
                                   size_t transforms) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = transforms * static_cast<size_t>(M);
    if (index >= total) return;
    int column = int(index % M);
    size_t transform = index / M;
    half2 value = __float2half2_rn(0.0f);
    if (column < K) {
        size_t input_index = transform * K + column;
        value = __halves2half2(real[input_index], imag[input_index]);
    }
    output[index] = value;
}

__global__ void power_reduce_fft(const cufftComplex* spectrum, float* power,
                                 int batch, int snapshots, int beams) {
    size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= static_cast<size_t>(batch) * beams) return;
    int batch_index = int(row / beams);
    int beam = int(row % beams);
    float sum = 0.0f;
    for (int snapshot = threadIdx.x; snapshot < snapshots; snapshot += blockDim.x) {
        size_t index =
            (static_cast<size_t>(batch_index) * snapshots + snapshot) * beams + beam;
        cufftComplex value = spectrum[index];
        sum = fmaf(value.x, value.x, sum);
        sum = fmaf(value.y, value.y, sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    __shared__ float warp_sum[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sum[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? warp_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) power[row] = sum;
    }
}

__global__ void power_reduce_fft_half(const half2* spectrum, float* power,
                                      int batch, int snapshots, int beams) {
    size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= static_cast<size_t>(batch) * beams) return;
    int batch_index = int(row / beams);
    int beam = int(row % beams);
    float sum = 0.0f;
    for (int snapshot = threadIdx.x; snapshot < snapshots; snapshot += blockDim.x) {
        size_t index =
            (static_cast<size_t>(batch_index) * snapshots + snapshot) * beams + beam;
        float2 value = __half22float2(spectrum[index]);
        sum = fmaf(value.x, value.x, sum);
        sum = fmaf(value.y, value.y, sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    __shared__ float warp_sum[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sum[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? warp_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) power[row] = sum;
    }
}

__device__ __forceinline__ bool better(float value, uint32_t index,
                                        float best, uint32_t best_index) {
    return value > best || (value == best && index < best_index);
}

__global__ void argmax_kernel(const float* power, float* max_power,
                              uint32_t* max_index, int beams) {
    int batch_index = int(blockIdx.x);
    const float* row = power + static_cast<size_t>(batch_index) * beams;
    float best_value = -CUDART_INF_F;
    uint32_t best_index = 0xffffffffu;
    for (int beam = threadIdx.x; beam < beams; beam += blockDim.x) {
        float value = row[beam];
        if (better(value, uint32_t(beam), best_value, best_index)) {
            best_value = value;
            best_index = uint32_t(beam);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_value = __shfl_down_sync(0xffffffffu, best_value, offset);
        uint32_t other_index = __shfl_down_sync(0xffffffffu, best_index, offset);
        if (better(other_value, other_index, best_value, best_index)) {
            best_value = other_value;
            best_index = other_index;
        }
    }
    __shared__ float values[8];
    __shared__ uint32_t indices[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        values[warp] = best_value;
        indices[warp] = best_index;
    }
    __syncthreads();
    if (warp == 0) {
        best_value = lane < 8 ? values[lane] : -CUDART_INF_F;
        best_index = lane < 8 ? indices[lane] : 0xffffffffu;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_value = __shfl_down_sync(0xffffffffu, best_value, offset);
            uint32_t other_index = __shfl_down_sync(0xffffffffu, best_index, offset);
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

int main(int argc, char** argv) {
    int batch = argc > 1 ? std::atoi(argv[1]) : 256;
    int snapshots = argc > 2 ? std::atoi(argv[2]) : 1024;
    int K = argc > 3 ? std::atoi(argv[3]) : 512;
    int beams = argc > 4 ? std::atoi(argv[4]) : 1024;
    int iterations = argc > 5 ? std::atoi(argv[5]) : 10;
    int warmup = argc > 6 ? std::atoi(argv[6]) : 3;
    int check = argc > 7 ? std::atoi(argv[7]) : 0;
    if (batch <= 0 || snapshots <= 0 || K <= 0 || beams < K ||
        iterations <= 0 || warmup < 0) return 2;

    size_t transforms = static_cast<size_t>(batch) * snapshots;
    size_t input_count = transforms * K;
    size_t fft_count = transforms * beams;
    half *input_real = nullptr, *input_imag = nullptr;
    void* fft_data = nullptr;
    float *power = nullptr, *max_power = nullptr;
    uint32_t* max_index = nullptr;
    CHECK_CUDA(cudaMalloc(&input_real, input_count * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&input_imag, input_count * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&fft_data, fft_count *
        (BEAM24_CUFFT_HALF ? sizeof(half2) : sizeof(cufftComplex))));
    CHECK_CUDA(cudaMalloc(&power, static_cast<size_t>(batch) * beams * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&max_power, batch * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&max_index, batch * sizeof(uint32_t)));
    initialize_half_planes<<<(input_count + 255) / 256, 256>>>(
        input_real, input_imag, input_count);
    CHECK_CUDA(cudaGetLastError());

    cufftHandle plan;
    int length[1] = {beams};
#if BEAM24_CUFFT_HALF
    CHECK_CUFFT(cufftCreate(&plan));
    long long length64[1] = {beams};
    size_t work_size = 0;
    CHECK_CUFFT(cufftXtMakePlanMany(
        plan, 1, length64, nullptr, 1, beams, CUDA_C_16F,
        nullptr, 1, beams, CUDA_C_16F, transforms, &work_size, CUDA_C_16F));
#else
    CHECK_CUFFT(cufftPlanMany(&plan, 1, length, nullptr, 1, beams,
                              nullptr, 1, beams, CUFFT_C2C, int(transforms)));
#endif

    auto enqueue = [&]() {
#if BEAM24_CUFFT_HALF
        pack_zero_pad_half<<<(fft_count + 255) / 256, 256>>>(
            input_real, input_imag, static_cast<half2*>(fft_data), K, beams, transforms);
        CHECK_CUFFT(cufftXtExec(plan, fft_data, fft_data, CUFFT_FORWARD));
        power_reduce_fft_half<<<static_cast<unsigned int>(static_cast<size_t>(batch) * beams), 256>>>(
            static_cast<const half2*>(fft_data), power, batch, snapshots, beams);
#else
        pack_zero_pad<<<(fft_count + 255) / 256, 256>>>(
            input_real, input_imag, static_cast<cufftComplex*>(fft_data), K, beams, transforms);
        CHECK_CUFFT(cufftExecC2C(
            plan, static_cast<cufftComplex*>(fft_data),
            static_cast<cufftComplex*>(fft_data), CUFFT_FORWARD));
        power_reduce_fft<<<static_cast<unsigned int>(static_cast<size_t>(batch) * beams), 256>>>(
            static_cast<const cufftComplex*>(fft_data), power, batch, snapshots, beams);
#endif
        argmax_kernel<<<batch, 256>>>(power, max_power, max_index, beams);
    };
    for (int index = 0; index < warmup; ++index) enqueue();
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int index = 0; index < iterations; ++index) enqueue();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    milliseconds /= iterations;

    int failed = 0;
    if (check) {
        std::vector<half> host_real(K), host_imag(K);
        std::vector<cufftComplex> host_fft(beams);
        CHECK_CUDA(cudaMemcpy(host_real.data(), input_real, K * sizeof(half), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(host_imag.data(), input_imag, K * sizeof(half), cudaMemcpyDeviceToHost));
#if BEAM24_CUFFT_HALF
        std::vector<half2> host_fft_half(beams);
        CHECK_CUDA(cudaMemcpy(host_fft_half.data(), fft_data,
                              beams * sizeof(half2), cudaMemcpyDeviceToHost));
        for (int beam = 0; beam < beams; ++beam) {
            float2 value = __half22float2(host_fft_half[beam]);
            host_fft[beam] = cufftComplex{value.x, value.y};
        }
#else
        CHECK_CUDA(cudaMemcpy(host_fft.data(), fft_data,
                              beams * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
#endif
        float max_error = 0.0f;
        for (int beam = 0; beam < beams; ++beam) {
            double reference_real = 0.0, reference_imag = 0.0;
            for (int sensor = 0; sensor < K; ++sensor) {
                double angle = -2.0 * M_PI * beam * sensor / beams;
                double xr = __half2float(host_real[sensor]);
                double xi = __half2float(host_imag[sensor]);
                reference_real += xr * std::cos(angle) - xi * std::sin(angle);
                reference_imag += xr * std::sin(angle) + xi * std::cos(angle);
            }
            max_error = std::max(max_error,
                float(std::hypot(host_fft[beam].x - reference_real,
                                 host_fft[beam].y - reference_imag)));
        }
        std::printf("FFT check max_abs=%g -> %s\n", max_error,
                    max_error <= 0.02f ? "PASSED" : "FAILED");
        failed = max_error > 0.02f;
    }
    std::printf(
        "{\"kind\":\"cufft_spatial_top1\",\"batch\":%d,\"snapshots\":%d,"
        "\"k\":%d,\"beams\":%d,\"iterations\":%d,\"warmup\":%d,"
        "\"milliseconds\":%.9g,\"fft_workspace_values\":%zu,"
        "\"precision\":\"%s\"}\n",
        batch, snapshots, K, beams, iterations, warmup, milliseconds, fft_count,
        BEAM24_CUFFT_HALF ? "complex_fp16" : "complex_fp32");

    cufftDestroy(plan);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(input_real); cudaFree(input_imag); cudaFree(fft_data);
    cudaFree(power); cudaFree(max_power); cudaFree(max_index);
    return failed ? 3 : 0;
}
