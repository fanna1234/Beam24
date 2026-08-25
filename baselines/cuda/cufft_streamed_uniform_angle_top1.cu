#include "cufft_top1_common.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void interpolate_power_reduce_streamed(
    const cufftComplex* spectrum,
    const int* lower_index,
    const float* fraction,
    float* power,
    int batch,
    int snapshots,
    int beams,
    int fft_length,
    float inverse_k) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= static_cast<size_t>(batch) * beams) return;
    const int batch_index = int(row / beams);
    const int beam = int(row % beams);
    const int lower = lower_index[beam];
    const int upper = lower + 1 == fft_length ? 0 : lower + 1;
    const float weight = fraction[beam];
    float sum = 0.0f;
    for (int snapshot = threadIdx.x; snapshot < snapshots; snapshot += blockDim.x) {
        const size_t base =
            (static_cast<size_t>(batch_index) * snapshots + snapshot) * fft_length;
        const cufftComplex lhs = spectrum[base + lower];
        const cufftComplex rhs = spectrum[base + upper];
        const float real = fmaf(weight, rhs.x - lhs.x, lhs.x) * inverse_k;
        const float imag = fmaf(weight, rhs.y - lhs.y, lhs.y) * inverse_k;
        sum = fmaf(real, real, sum);
        sum = fmaf(imag, imag, sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    __shared__ float warp_sum[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sum[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? warp_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) power[row] = sum;
    }
}

int main(int argc, char** argv) {
    const int batch = argc > 1 ? std::atoi(argv[1]) : 256;
    const int snapshots = argc > 2 ? std::atoi(argv[2]) : 1024;
    const int K = argc > 3 ? std::atoi(argv[3]) : 512;
    const int beams = argc > 4 ? std::atoi(argv[4]) : 1024;
    const int fft_length = argc > 5 ? std::atoi(argv[5]) : 4096;
    const int chunk_batches = argc > 6 ? std::atoi(argv[6]) : 2;
    const int iterations = argc > 7 ? std::atoi(argv[7]) : 10;
    const int warmup = argc > 8 ? std::atoi(argv[8]) : 3;
    if (batch <= 0 || snapshots <= 0 || K <= 0 || beams <= 0 ||
        fft_length < K || chunk_batches <= 0 || batch % chunk_batches != 0 ||
        iterations <= 0 || warmup < 0) return 2;

    const size_t transforms = static_cast<size_t>(batch) * snapshots;
    const size_t input_count = transforms * K;
    const size_t chunk_transforms =
        static_cast<size_t>(chunk_batches) * snapshots;
    const size_t chunk_fft_count = chunk_transforms * fft_length;
    const size_t full_fft_count = transforms * fft_length;
    half *input_real = nullptr, *input_imag = nullptr;
    cufftComplex* fft_data = nullptr;
    float *power = nullptr, *max_power = nullptr;
    uint32_t* max_index = nullptr;
    int* lower_index = nullptr;
    float* fraction = nullptr;
    CHECK_CUDA(cudaMalloc(&input_real, input_count * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&input_imag, input_count * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&fft_data, chunk_fft_count * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc(&power, static_cast<size_t>(batch) * beams * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&max_power, batch * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&max_index, batch * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&lower_index, beams * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&fraction, beams * sizeof(float)));
    initialize_half_planes<<<(input_count + 255) / 256, 256>>>(
        input_real, input_imag, input_count);
    CHECK_CUDA(cudaGetLastError());

    std::vector<int> host_lower(beams);
    std::vector<float> host_fraction(beams);
    for (int beam = 0; beam < beams; ++beam) {
        const double angle = beams == 1
            ? 0.0
            : -60.0 + 120.0 * beam / (beams - 1);
        double coordinate = 0.5 * fft_length * std::sin(angle * M_PI / 180.0);
        coordinate = std::fmod(coordinate, double(fft_length));
        if (coordinate < 0.0) coordinate += fft_length;
        const double base = std::floor(coordinate);
        host_lower[beam] = int(base);
        host_fraction[beam] = float(coordinate - base);
    }
    CHECK_CUDA(cudaMemcpy(lower_index, host_lower.data(), beams * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(fraction, host_fraction.data(), beams * sizeof(float),
                          cudaMemcpyHostToDevice));

    cufftHandle plan;
    int length[1] = {fft_length};
    CHECK_CUFFT(cufftPlanMany(&plan, 1, length, nullptr, 1, fft_length,
                              nullptr, 1, fft_length, CUFFT_C2C,
                              int(chunk_transforms)));

    auto enqueue = [&]() {
        for (int batch_offset = 0; batch_offset < batch;
             batch_offset += chunk_batches) {
            const size_t input_offset =
                static_cast<size_t>(batch_offset) * snapshots * K;
            pack_zero_pad<<<(chunk_fft_count + 255) / 256, 256>>>(
                input_real + input_offset, input_imag + input_offset,
                fft_data, K, fft_length, chunk_transforms);
            CHECK_CUFFT(cufftExecC2C(plan, fft_data, fft_data, CUFFT_FORWARD));
            interpolate_power_reduce_streamed<<<chunk_batches * beams, 256>>>(
                fft_data, lower_index, fraction,
                power + static_cast<size_t>(batch_offset) * beams,
                chunk_batches, snapshots, beams, fft_length, 1.0f / K);
        }
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

    uint32_t first_top1 = 0;
    CHECK_CUDA(cudaMemcpy(&first_top1, max_index, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    std::printf(
        "{\"kind\":\"streamed_cufft_uniform_angle_top1\","
        "\"batch\":%d,\"snapshots\":%d,\"k\":%d,\"beams\":%d,"
        "\"fft_length\":%d,\"chunk_batches\":%d,"
        "\"iterations\":%d,\"warmup\":%d,\"milliseconds\":%.9g,"
        "\"streamed_workspace_bytes\":%zu,\"full_workspace_bytes\":%zu,"
        "\"materializes_full_output\":%s,\"first_top1\":%u,"
        "\"precision\":\"complex_fp32\","
        "\"consumer\":\"uniform_angle_interpolate_power_top1\"}\n",
        batch, snapshots, K, beams, fft_length, chunk_batches, iterations,
        warmup, milliseconds, chunk_fft_count * sizeof(cufftComplex),
        full_fft_count * sizeof(cufftComplex),
        chunk_batches == batch ? "true" : "false", first_top1);

    cufftDestroy(plan);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(input_real);
    cudaFree(input_imag);
    cudaFree(fft_data);
    cudaFree(power);
    cudaFree(max_power);
    cudaFree(max_index);
    cudaFree(lower_index);
    cudaFree(fraction);
    return 0;
}
