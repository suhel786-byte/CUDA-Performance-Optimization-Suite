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

    // ---------------- CPU Timing ----------------
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_A.data(), h_B.data(), h_C.data());
    auto end_cpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_time.count() << " ms\n";

    // ---------------- GPU Memory ----------------
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // ---------------- Kernel Launch ----------------
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    auto start_gpu = std::chrono::high_resolution_clock::now();

    vectorAddGPU<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaDeviceSynchronize();

    auto end_gpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;
    std::cout << "GPU Time: " << gpu_time.count() << " ms\n";

    // ---------------- Copy Back ----------------
    cudaMemcpy(h_C_gpu.data(), d_C, size, cudaMemcpyDeviceToHost);

    // ---------------- Verify ----------------
    for (int i = 0; i < 10; i++) {
        if (h_C[i] != h_C_gpu[i]) {
            std::cout << "Mismatch at " << i << "\n";
            break;
        }
    }

    // ---------------- Cleanup ----------------
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}