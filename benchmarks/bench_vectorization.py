"""
Vectorization Benchmark: When does NumPy become faster?
=========================================================

Shows the crossover point where NumPy overhead is paid back by vectorization.
Key insight: NumPy has a fixed overhead per operation (~1µs function call)
but processes arrays in C. For small arrays, this overhead dominates.
"""

import time

import numpy as np


def benchmark_sizes():
    """Test NumPy vs loop for different array sizes."""
    sizes = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
    n_repeats = 1000

    print("=" * 70)
    print("VECTORIZATION: NumPy vs Python Loops at Different Sizes")
    print("=" * 70)
    print()
    print(f"  {'Size':>10} {'Loop (µs)':>12} {'NumPy (µs)':>12} {'Speedup':>10} {'Winner':>8}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")

    for size in sizes:
        data = np.random.randn(size)
        repeats = max(10, n_repeats // size)

        # Python loop: sum of squares
        start = time.perf_counter()
        for _ in range(repeats):
            total = 0.0
            for x in data:
                total += x * x
        loop_time = (time.perf_counter() - start) / repeats * 1e6

        # NumPy: sum of squares
        start = time.perf_counter()
        for _ in range(repeats):
            total = np.sum(data * data)
        numpy_time = (time.perf_counter() - start) / repeats * 1e6

        speedup = loop_time / numpy_time
        winner = "NumPy" if speedup > 1 else "Loop"

        print(f"  {size:>10,} {loop_time:>10.1f}\u00b5s {numpy_time:>10.1f}\u00b5s {speedup:>9.1f}x {winner:>8}")

    print()
    print("  → NumPy wins for n > ~50-100 elements")
    print("  → For very small arrays, function call overhead dominates")
    print("  → In quant finance, we almost always have large arrays → always use NumPy")
    print()


def benchmark_operations():
    """Compare different operation types: loop vs vectorized."""
    n = 500_000
    data = np.random.randn(n)
    data2 = np.random.randn(n)

    operations = [
        ("Element-wise multiply", lambda d: [x * 2 for x in d], lambda d: d * 2),
        ("Sum", lambda d: sum(d), lambda d: np.sum(d)),
        ("Exp", lambda d: [np.exp(x) for x in d[:10000]], lambda d: np.exp(d[:10000])),
        ("Dot product", lambda d: sum(a * b for a, b in zip(d, data2)), lambda d: np.dot(d, data2)),
    ]

    print("=" * 70)
    print(f"OPERATION COMPARISON (n={n:,})")
    print("=" * 70)
    print()
    print(f"  {'Operation':<22} {'Loop':>10} {'NumPy':>10} {'Speedup':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")

    for name, loop_fn, numpy_fn in operations:
        # Loop
        start = time.perf_counter()
        loop_fn(data)
        loop_time = time.perf_counter() - start

        # NumPy
        start = time.perf_counter()
        numpy_fn(data)
        numpy_time = time.perf_counter() - start

        speedup = loop_time / numpy_time
        print(f"  {name:<22} {loop_time:>8.4f}s {numpy_time:>8.6f}s {speedup:>9.0f}x")

    print()


def benchmark_memory_layout():
    """Show how memory layout affects performance (row-major vs column access)."""
    n = 5000

    # C-contiguous (row-major) — default for NumPy
    A = np.random.randn(n, n)  # C order (row-major)

    print("=" * 70)
    print("MEMORY LAYOUT: Cache-Friendly Access Patterns")
    print("=" * 70)
    print()

    # Row-wise sum (sequential memory access — cache friendly)
    start = time.perf_counter()
    row_sums = np.sum(A, axis=1)
    row_time = time.perf_counter() - start

    # Column-wise sum (strided access — cache unfriendly for C-order)
    start = time.perf_counter()
    col_sums = np.sum(A, axis=0)
    col_time = time.perf_counter() - start

    # Fortran-order array, column-wise sum (now cache friendly)
    A_f = np.asfortranarray(A)
    start = time.perf_counter()
    col_sums_f = np.sum(A_f, axis=0)
    col_f_time = time.perf_counter() - start

    print(f"  Array: {n}×{n} float64 ({A.nbytes / 1024 / 1024:.0f} MB)")
    print(f"  Row sum (C-order, sequential):     {row_time * 1000:.2f} ms")
    print(f"  Col sum (C-order, strided):        {col_time * 1000:.2f} ms")
    print(f"  Col sum (F-order, sequential):     {col_f_time * 1000:.2f} ms")
    print()
    print("  → Sequential memory access is faster (CPU cache prefetch)")
    print("  → Choose memory layout based on your access pattern")
    print("  → For row-wise operations: C order (default)")
    print("  → For column-wise operations: Fortran order")
    print()


if __name__ == "__main__":
    benchmark_sizes()
    benchmark_operations()
    benchmark_memory_layout()


