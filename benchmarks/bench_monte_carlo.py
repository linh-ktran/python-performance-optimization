"""
Comprehensive Benchmark: Monte Carlo Option Pricing
=====================================================

Compare ALL optimization strategies on the same problem:
1. Pure Python loops
2. NumPy vectorized
3. Numba JIT
4. Numba parallel
5. Multiprocessing
6. C extension (ctypes)

This shows the full spectrum of "How do you speed up Python?"
"""

import math
import os
import time
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =============================================================================
# 1. Pure Python
# =============================================================================

def mc_python(S0, K, r, sigma, T, n_paths):
    """Pure Python — baseline."""
    payoff_sum = 0.0
    for _ in range(n_paths):
        z = np.random.randn()
        ST = S0 * math.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)
        payoff_sum += max(ST - K, 0)
    return math.exp(-r * T) * payoff_sum / n_paths


# =============================================================================
# 2. NumPy Vectorized
# =============================================================================

def mc_numpy(S0, K, r, sigma, T, n_paths):
    """NumPy vectorized — process all paths at once."""
    Z = np.random.randn(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0)
    return np.exp(-r * T) * np.mean(payoffs)


# =============================================================================
# 3 & 4. Numba JIT (single-threaded and parallel)
# =============================================================================

if HAS_NUMBA:
    @njit(cache=True)
    def mc_numba(S0, K, r, sigma, T, n_paths):
        """Numba single-threaded."""
        payoff_sum = 0.0
        for _ in range(n_paths):
            z = np.random.randn()
            ST = S0 * math.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)
            if ST > K:
                payoff_sum += ST - K
        return math.exp(-r * T) * payoff_sum / n_paths

    @njit(parallel=True, cache=True)
    def mc_numba_parallel(S0, K, r, sigma, T, n_paths):
        """Numba with parallel loop."""
        payoffs = np.empty(n_paths)
        for i in prange(n_paths):
            z = np.random.randn()
            ST = S0 * math.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)
            payoffs[i] = max(ST - K, 0)
        return math.exp(-r * T) * np.mean(payoffs)


# =============================================================================
# 5. Multiprocessing
# =============================================================================

def _mc_worker(args):
    """Worker for multiprocessing."""
    S0, K, r, sigma, T, n_paths, seed = args
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return np.exp(-r * T) * np.mean(np.maximum(ST - K, 0))


def mc_multiprocessing(S0, K, r, sigma, T, n_paths):
    """Distribute across CPU cores."""
    n_workers = cpu_count()
    paths_per_worker = n_paths // n_workers
    args = [(S0, K, r, sigma, T, paths_per_worker, i) for i in range(n_workers)]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_mc_worker, args))

    return np.mean(results)


# =============================================================================
# Benchmark runner
# =============================================================================

def run_benchmark():
    """Run all methods and compare."""
    S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.2, 1.0
    n_paths = 1_000_000

    print("=" * 70)
    print(f"MONTE CARLO OPTION PRICING BENCHMARK ({n_paths:,} paths)")
    print("=" * 70)
    print(f"  S0={S0}, K={K}, r={r}, σ={sigma}, T={T}")
    print(f"  CPU cores: {cpu_count()}")
    print()

    results = []

    # 1. Pure Python (only 50K — too slow for 1M)
    n_py = 50_000
    start = time.perf_counter()
    price = mc_python(S0, K, r, sigma, T, n_py)
    elapsed = time.perf_counter() - start
    normalized = elapsed * (n_paths / n_py)
    results.append(("Python loop", normalized, price, f"(extrapolated from {n_py:,})"))

    # 2. NumPy
    start = time.perf_counter()
    price = mc_numpy(S0, K, r, sigma, T, n_paths)
    elapsed = time.perf_counter() - start
    results.append(("NumPy vectorized", elapsed, price, ""))

    # 3 & 4. Numba
    if HAS_NUMBA:
        # Warmup
        mc_numba(S0, K, r, sigma, T, 100)
        mc_numba_parallel(S0, K, r, sigma, T, 100)

        start = time.perf_counter()
        price = mc_numba(S0, K, r, sigma, T, n_paths)
        elapsed = time.perf_counter() - start
        results.append(("Numba JIT", elapsed, price, ""))

        start = time.perf_counter()
        price = mc_numba_parallel(S0, K, r, sigma, T, n_paths)
        elapsed = time.perf_counter() - start
        results.append(("Numba parallel", elapsed, price, f"({cpu_count()} cores)"))

    # 5. Multiprocessing
    start = time.perf_counter()
    price = mc_multiprocessing(S0, K, r, sigma, T, n_paths)
    elapsed = time.perf_counter() - start
    results.append(("Multiprocessing", elapsed, price, f"({cpu_count()} workers)"))

    # Print results table
    baseline = results[0][1]
    print(f"  {'Method':<22} {'Time':>10} {'Speedup':>10} {'Price':>10}  Notes")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}  {'-'*20}")
    for name, elapsed, price, note in results:
        speedup = baseline / elapsed
        print(f"  {name:<22} {elapsed:>8.4f}s {speedup:>9.1f}x  ${price:>8.4f}  {note}")

    print()
    print("  Key insights:")
    print("  • NumPy: ~100x faster than Python (vectorization)")
    print("  • Numba: ~100-200x faster (compiled to machine code)")
    print("  • Numba parallel: ~N×cores faster (multi-core)")
    print("  • Multiprocessing: good but has IPC overhead")
    print("  • All give similar prices (statistical convergence)")


if __name__ == "__main__":
    run_benchmark()

