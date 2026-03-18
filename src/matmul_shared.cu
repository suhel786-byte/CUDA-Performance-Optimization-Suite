#include <iostream>
#include <cuda_runtime.h>
using namespace std;
#define TILE 16

__global__ void matMulSharedKernel(const float* A, const float* B, float* C, int M) {

    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (M + TILE - 1) / TILE; t++) {

        if (row < M && t * TILE + threadIdx.x < M)
            tileA[threadIdx.y][threadIdx.x] = A[row * M + t * TILE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0;

        if (col < M && t * TILE + threadIdx.y < M)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * M + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < M) {
        C[row * M + col] = sum;
    }
}

// Wrapper function (host-side)
void runMatrixShared(const float* A, const float* B, float* C, int M) {

    float *d_A, *d_B, *d_C;
    size_t size = M * M * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((M + 15) / 16, (M + 15) / 16);

    // 🔥 Warm-up
    matMulSharedKernel<<<blocks, threads>>>(d_A, d_B, d_C, M);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matMulSharedKernel<<<blocks, threads>>>(d_A, d_B, d_C, M);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0;
    cudaEventElapsedTime(&time, start, stop);

    cout << "Shared Memory GPU Time: " << time << " ms\n";

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}