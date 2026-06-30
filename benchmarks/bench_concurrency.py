"""
Concurrency Benchmark: Threading vs Asyncio vs Multiprocessing
================================================================

Side-by-side comparison on I/O-bound and CPU-bound workloads.
"""

import asyncio
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count

import numpy as np


# =============================================================================
# I/O-bound task
# =============================================================================

def io_task(task_id: int, duration: float = 0.1) -> int:
    """Simulate I/O-bound work (API call, DB query)."""
    time.sleep(duration)
    return task_id


async def io_task_async(task_id: int, duration: float = 0.1) -> int:
    """Async version of I/O task."""
    await asyncio.sleep(duration)
    return task_id


# =============================================================================
# CPU-bound task
# =============================================================================

def cpu_task(n: int = 2_000_000) -> float:
    """CPU-bound: compute sum of squares."""
    total = 0.0
    for i in range(n):
        total += i * i * 0.001
    return total


def cpu_task_numpy(n: int = 2_000_000) -> float:
    """CPU-bound with NumPy (releases GIL)."""
    arr = np.arange(n, dtype=np.float64)
    return float(np.sum(arr * arr * 0.001))


# =============================================================================
# Benchmarks
# =============================================================================

def bench_io_bound():
    """Compare concurrency models for I/O-bound work."""
    n_tasks = 50
    duration = 0.1

    print("=" * 70)
    print(f"I/O-BOUND: {n_tasks} tasks × {duration}s each")
    print("=" * 70)
    print(f"  Theoretical minimum: {duration:.1f}s (all concurrent)")
    print(f"  Sequential baseline: {n_tasks * duration:.1f}s")
    print()

    # Sequential
    start = time.perf_counter()
    [io_task(i, duration) for i in range(n_tasks)]
    seq_time = time.perf_counter() - start

    # Threading
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_tasks) as pool:
        list(pool.map(lambda i: io_task(i, duration), range(n_tasks)))
    thread_time = time.perf_counter() - start

    # Asyncio
    async def run_async():
        tasks = [io_task_async(i, duration) for i in range(n_tasks)]
        return await asyncio.gather(*tasks)

    start = time.perf_counter()
    asyncio.run(run_async())
    async_time = time.perf_counter() - start

    # Multiprocessing (overkill for I/O, shows overhead)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(n_tasks, cpu_count())) as pool:
        list(pool.map(lambda i: io_task(i, duration), range(n_tasks)))
    mp_time = time.perf_counter() - start

    print(f"  {'Method':<20} {'Time':>8} {'Speedup':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*10}")
    print(f"  {'Sequential':<20} {seq_time:>6.3f}s {1.0:>9.1f}x")
    print(f"  {'Threading':<20} {thread_time:>6.3f}s {seq_time/thread_time:>9.1f}x")
    print(f"  {'Asyncio':<20} {async_time:>6.3f}s {seq_time/async_time:>9.1f}x")
    print(f"  {'Multiprocessing':<20} {mp_time:>6.3f}s {seq_time/mp_time:>9.1f}x")
    print()
    print("  → Threading and asyncio both excellent for I/O")
    print("  → Multiprocessing has startup overhead (not ideal for I/O)")
    print()


def bench_cpu_bound():
    """Compare concurrency models for CPU-bound work."""
    n_tasks = 4

    print("=" * 70)
    print(f"CPU-BOUND: {n_tasks} pure Python tasks")
    print("=" * 70)
    print()

    # Sequential
    start = time.perf_counter()
    [cpu_task() for _ in range(n_tasks)]
    seq_time = time.perf_counter() - start

    # Threading (GIL limits this!)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_tasks) as pool:
        list(pool.map(lambda _: cpu_task(), range(n_tasks)))
    thread_time = time.perf_counter() - start

    # Multiprocessing (true parallelism)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_tasks) as pool:
        list(pool.map(lambda _: cpu_task(), range(n_tasks)))
    mp_time = time.perf_counter() - start

    print(f"  {'Method':<20} {'Time':>8} {'Speedup':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*10}")
    print(f"  {'Sequential':<20} {seq_time:>6.3f}s {1.0:>9.1f}x")
    print(f"  {'Threading':<20} {thread_time:>6.3f}s {seq_time/thread_time:>9.1f}x")
    print(f"  {'Multiprocessing':<20} {mp_time:>6.3f}s {seq_time/mp_time:>9.1f}x")
    print()
    print("  → Threading gives NO speedup (GIL!)")
    print("  → Multiprocessing gives ~Nx speedup (separate processes)")
    print()


def bench_numpy_threaded():
    """Show that NumPy + threading DOES work (GIL released in C)."""
    n_tasks = 4

    print("=" * 70)
    print(f"NUMPY + THREADING: {n_tasks} tasks (GIL released in C extensions)")
    print("=" * 70)
    print()

    # Sequential
    start = time.perf_counter()
    [cpu_task_numpy() for _ in range(n_tasks)]
    seq_time = time.perf_counter() - start

    # Threading (works because NumPy releases GIL!)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_tasks) as pool:
        list(pool.map(lambda _: cpu_task_numpy(), range(n_tasks)))
    thread_time = time.perf_counter() - start

    print(f"  {'Method':<20} {'Time':>8} {'Speedup':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*10}")
    print(f"  {'Sequential':<20} {seq_time:>6.3f}s {1.0:>9.1f}x")
    print(f"  {'Threading':<20} {thread_time:>6.3f}s {seq_time/thread_time:>9.1f}x")
    print()
    print("  → Threading works with NumPy! GIL released during C computation.")
    print("  → This is why sklearn/XGBoost inference can be parallelized with threads.")
    print()


if __name__ == "__main__":
    bench_io_bound()
    bench_cpu_bound()
    bench_numpy_threaded()

