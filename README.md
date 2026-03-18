# 🚀 CUDA Performance Optimization Suite

A hands-on CUDA project demonstrating GPU acceleration, performance benchmarking, and memory optimization techniques using C++ and NVIDIA CUDA.

---

## 📌 Overview

This project explores how GPU computing can significantly accelerate computational workloads compared to CPU execution. It implements core parallel algorithms and applies optimization strategies to improve performance.

---

## 🎯 Objectives

* Implement fundamental parallel algorithms in CUDA
* Compare CPU vs GPU performance
* Analyze memory transfer overhead
* Apply optimization techniques like shared memory tiling
* Build a strong foundation in CUDA performance engineering

---

## ⚙️ Technologies Used

* C++
* NVIDIA CUDA Toolkit
* NVCC Compiler
* Windows + MSVC (cl.exe)

---

## 📁 Project Structure

```
CUDA-Performance-Optimization-Suite/
│
├── src/
│   ├── main.cu
│   ├── matmul_shared.cu
│
├── include/
│   ├── matmul_shared.h
│
├── build/
│   └── main.exe
│
└── README.md
```

---

## 🔬 Implementations

### 1. Vector Addition

* CPU implementation (sequential)
* GPU kernel (parallel threads)
* Benchmarking:

  * CPU execution time
  * GPU kernel time
  * Host-to-device (H2D) transfer
  * Device-to-host (D2H) transfer

#### 📊 Results

| Metric     | Time      |
| ---------- | --------- |
| CPU        | ~10–12 ms |
| GPU Kernel | ~0.7–1 ms |
| H2D        | ~8–10 ms  |
| D2H        | ~4–6 ms   |

#### 💡 Insight

Although GPU computation is faster, memory transfer overhead dominates total runtime, making vector addition **memory-bound**.

---

### 2. Matrix Multiplication (Naive)

* CPU implementation (triple loop)
* GPU implementation (2D grid + threads)

#### 📊 Results

| Implementation | Time        |
| -------------- | ----------- |
| CPU            | ~250–300 ms |
| GPU (Naive)    | ~3–10 ms    |

#### 💡 Insight

Matrix multiplication has **high arithmetic intensity**, making it ideal for GPU acceleration.

---

### 3. Matrix Multiplication (Shared Memory Optimized)

* Tiled computation using shared memory
* Reduced global memory access
* Improved data reuse

#### 📊 Results

| Implementation      | Time      |
| ------------------- | --------- |
| GPU (Naive)         | ~3–10 ms  |
| GPU (Shared Memory) | ~0.5–2 ms |

#### 💡 Key Optimizations

* Shared memory tiling
* Memory coalescing
* Reduced global memory traffic
* Synchronization using `__syncthreads()`

#### 💡 Insight

Shared memory significantly improves performance by minimizing expensive global memory access and enabling reuse of loaded data across threads.

---

## 🧠 Key Learnings

* GPU computation is fast, but memory transfer can be a bottleneck
* Proper benchmarking requires CUDA events
* Kernel warm-up is necessary for accurate timing
* Shared memory is critical for high-performance CUDA programs
* Thread/block indexing must be handled carefully
* Correctness validation is as important as performance

---

## ⚡ Build & Run

### 🔧 Compile

```
nvcc src/main.cu src/matmul_shared.cu -o build/main.exe
```

### ▶️ Run

```
build\main.exe
```

---

## 🧪 Sample Output

```
CPU Time: 10.8 ms
GPU Kernel Time: 0.69 ms

===== MATRIX MULTIPLICATION =====
Matrix CPU Time: 260 ms
Matrix GPU Time: 3.3 ms
Shared Memory GPU Time: 0.55 ms
Matrix result correct [OK]
```

---

## 🚀 Future Improvements

* Pinned memory for faster transfers
* Asynchronous CUDA streams
* Warp-level optimizations
* Reduction kernels
* Profiling using NVIDIA Nsight
* Multi-GPU scaling

---

## 📌 Conclusion

This project demonstrates how GPU acceleration and memory optimization techniques can dramatically improve performance. It highlights the importance of understanding memory hierarchies and parallel execution models in CUDA.

---

## 👤 Author

**Suhel Baig**
GitHub: https://github.com/suhel786-byte

---

## ⭐ If you found this useful

Give it a star ⭐ and feel free to contribute!
