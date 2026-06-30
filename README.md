# Python Performance Optimization

> Demonstrating how to speed up Python for computationally intensive, high-performance,
> and low-level optimization workloads — with examples from numerical computing,
> simulation, and real-time data processing.


## Quick Decision Rule

| Bottleneck | Strategy |
|---|---|
| Waiting (network, disk) | `threading` (simple) or `asyncio` (high scale) |
| CPU-bound pure Python | `multiprocessing` |
| CPU-bound in NumPy/sklearn/XGBoost | `threading` works (GIL released in C extensions) |
| Memory-bound | Vectorize, reduce dtype, sequential access patterns |
| Single-thread hotspot | Profile → NumPy/Numba/Cython/Rust |

## Concurrency Model

|  | **threading** | **asyncio** | **multiprocessing** |
|---|---|---|---|
| **Best for** | I/O-bound, moderate concurrency | I/O-bound, high concurrency (1000s tasks) | CPU-bound, true parallelism (bypasses GIL) |
| **Memory** | Shared memory | Single thread, event loop | Separate processes, memory duplication |
| **Trade-offs** | GIL limits CPU work | Cooperative scheduling | Serialization overhead |

## Performance Hierarchy (Speed)

```
CPU Cache (auto)    →  nanoseconds
In-memory (dict)    →  microseconds
Redis/Memcached     →  ~1 ms (network)
Disk (file/SQLite)  →  ~1-10 ms
Object Storage (S3) →  ~50-200 ms
Recompute           →  seconds-minutes
```

## Running Examples

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmarks
python -m benchmarks.bench_concurrency
python -m benchmarks.bench_vectorization
python -m benchmarks.bench_monte_carlo

# Run specific examples
python -m examples.concurrency.01_gil_demo
python -m examples.vectorization.01_numpy_vs_loops

# Run tests
pytest tests/
```

## Notes on Performance Optimization

- **Profile first** — never optimize without measuring
- **Vectorize** — NumPy/pandas operate on contiguous memory blocks (cache-friendly)
- **Choose the right concurrency model** — threading for I/O, multiprocessing for CPU
- **GIL is a CPython detail** — C extensions (NumPy, sklearn) release it
- **Compiled extensions** — Numba (JIT), Cython, or Rust/C++ for true hotspots
- **Memory matters** — float32 vs float64, contiguous arrays, avoid copies

