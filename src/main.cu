#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>
#include "../include/matmul_shared.h"

#define N 10000000

// ================= VECTOR ADD =================

// CPU
void vectorAddCPU(const float* A, const float* B, float* C) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// GPU
__global__ void vectorAddGPU(const float* A, const float* B, float* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// ================= MATRIX MULTIPLICATION =================

// CPU
void matMulCPU(const float* A, const float* B, float* C, int M) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            float sum = 0.0f;
            for (int k = 0; k < M; k++) {
                sum += A[i * M + k] * B[k * M + j];
            }
            C[i * M + j] = sum;
        }
    }
}

// GPU
__global__ void matMulGPU(const float* A, const float* B, float* C, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < M) {
        float sum = 0.0f;
        for (int k = 0; k < M; k++) {
            sum += A[row * M + k] * B[k * M + col];
        }
        C[row * M + col] = sum;
    }
}

// ================= MAIN =================

int main() {

    // 🔥 Initialize CUDA (removes first-launch overhead)
    cudaFree(0);

    // ================= VECTOR ADD =================
    size_t size = N * sizeof(float);

    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C(N), h_C_gpu(N);

    // CPU timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_A.data(), h_B.data(), h_C.data());
    auto end_cpu = std::chrono::high_resolution_clock::now();

    std::cout << "CPU Time: "
              << std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count()
              << " ms\n";

    // GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 🔥 Warm-up kernel
    vectorAddGPU<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    // Kernel timing
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

    cudaMemcpy(h_C_gpu.data(), d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // ================= MATRIX MULTIPLICATION =================
    std::cout << "DEBUG: entering matrix section\n";
    std::cout << "\n===== MATRIX MULTIPLICATION =====\n";

    int M = 512;
    size_t mat_size = M * M * sizeof(float);

    std::vector<float> A(M * M, 1.0f);
    std::vector<float> B(M * M, 2.0f);
    std::vector<float> C_cpu(M * M);
    std::vector<float> C_gpu(M * M);

    // CPU
    auto start_cpu_mat = std::chrono::high_resolution_clock::now();
    matMulCPU(A.data(), B.data(), C_cpu.data(), M);
    auto end_cpu_mat = std::chrono::high_resolution_clock::now();

    std::cout << "Matrix CPU Time: "
              << std::chrono::duration<double, std::milli>(end_cpu_mat - start_cpu_mat).count()
              << " ms\n";

    // GPU memory
    float *d_A2, *d_B2, *d_C2;
    cudaMalloc(&d_A2, mat_size);
    cudaMalloc(&d_B2, mat_size);
    cudaMalloc(&d_C2, mat_size);

    cudaMemcpy(d_A2, A.data(), mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2, B.data(), mat_size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks2((M + 15) / 16, (M + 15) / 16);

    cudaEvent_t start_mat, stop_mat;
    cudaEventCreate(&start_mat);
    cudaEventCreate(&stop_mat);

    cudaEventRecord(start_mat);
    runMatrixShared(A.data(), B.data(), C_gpu.data(), M);
    cudaEventRecord(stop_mat);
    cudaEventSynchronize(stop_mat);

    float mat_time = 0;
    cudaEventElapsedTime(&mat_time, start_mat, stop_mat);

    std::cout << "Matrix GPU Time: " << mat_time << " ms\n";

    cudaMemcpy(C_gpu.data(), d_C2, mat_size, cudaMemcpyDeviceToHost);
    // 🔥 Verification 
        bool correct = true;

        for (int i = 0; i < M * M; i++) {
            if (abs(C_cpu[i] - C_gpu[i]) > 1e-5) {
                std::cout << "Mismatch at " << i << "\n";
                correct = false;
                break;
            }
        }

        if (correct) {
            std::cout << "Matrix result correct \n";
        }

    cudaFree(d_A2);
    cudaFree(d_B2);
    cudaFree(d_C2);

    return 0;
}