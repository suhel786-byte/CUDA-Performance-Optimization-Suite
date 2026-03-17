## Vector Addition Benchmark

### Results (RTX 3050)

| Metric | Time (ms) |
|-------|----------|
| CPU | ~12 ms |
| H2D Transfer | ~10 ms |
| GPU Kernel | ~1.3 ms |
| D2H Transfer | ~6.6 ms |

### Insight
Although GPU kernel execution is significantly faster, total runtime is dominated by memory transfer overhead. This demonstrates the importance of optimizing data movement in CUDA applications.