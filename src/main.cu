#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#define N 10000000

// ---------------- CPU ----------------
void vectorAddCPU(const float* A, const float* B, float* C) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// ---------------- GPU Kernel ----------------
__global__ void vectorAddGPU(const float* A, const float* B, float* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// ---------------- Main ----------------
int main() {
    size_t size = N * sizeof(float);

    // Host memory
    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C(N), h_C_gpu(N);

    // ================= CPU Timing =================
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_A.data(), h_B.data(), h_C.data());
    auto end_cpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_time.count() << " ms\n";

    // ================= GPU Memory =================
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // ================= H2D Timing =================
    cudaEvent_t h2d_start, h2d_stop;
    cudaEventCreate(&h2d_start);
    cudaEventCreate(&h2d_stop);

    cudaEventRecord(h2d_start);

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    cudaEventRecord(h2d_stop);
    cudaEventSynchronize(h2d_stop);

    float h2d_time = 0;
    cudaEventElapsedTime(&h2d_time, h2d_start, h2d_stop);
    std::cout << "H2D Time: " << h2d_time << " ms\n";

    // ================= Kernel Launch =================
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t k_start, k_stop;
    cudaEventCreate(&k_start);
    cudaEventCreate(&k_stop);

    cudaEventRecord(k_start);

    vectorAddGPU<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaEventRecord(k_stop);
    cudaEventSynchronize(k_stop);

    float kernel_time = 0;
    cudaEventElapsedTime(&kernel_time, k_start, k_stop);
    std::cout << "GPU Kernel Time: " << kernel_time << " ms\n";

    // ================= D2H Timing =================
    cudaEvent_t d2h_start, d2h_stop;
    cudaEventCreate(&d2h_start);
    cudaEventCreate(&d2h_stop);

    cudaEventRecord(d2h_start);

    cudaMemcpy(h_C_gpu.data(), d_C, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(d2h_stop);
    cudaEventSynchronize(d2h_stop);

    float d2h_time = 0;
    cudaEventElapsedTime(&d2h_time, d2h_start, d2h_stop);
    std::cout << "D2H Time: " << d2h_time << " ms\n";

    // ================= Verify =================
    for (int i = 0; i < 10; i++) {
        if (h_C[i] != h_C_gpu[i]) {
            std::cout << "Mismatch at " << i << "\n";
            break;
        }
    }

    // ================= Cleanup =================
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(h2d_start);
    cudaEventDestroy(h2d_stop);
    cudaEventDestroy(k_start);
    cudaEventDestroy(k_stop);
    cudaEventDestroy(d2h_start);
    cudaEventDestroy(d2h_stop);

    return 0;
}