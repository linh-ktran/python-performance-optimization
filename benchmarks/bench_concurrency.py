"""
Side-by-side comparison of threading, asyncio, and multiprocessing
on I/O-bound and CPU-bound workloads.
"""

import asyncio
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count

import numpy as np


IO_DURATION = 0.1  # used by multiprocessing (needs top-level picklable reference)


def io_task(task_id: int, duration: float = 0.1) -> int:
    """Simulate I/O-bound work (API call, DB query)."""
    time.sleep(duration)
    return task_id


def _io_task_default(task_id: int) -> int:
    """Top-level wrapper so ProcessPoolExecutor can pickle it."""
    time.sleep(IO_DURATION)
    return task_id


async def io_task_async(task_id: int, duration: float = 0.1) -> int:
    """Async version of I/O task."""
    await asyncio.sleep(duration)
    return task_id


def cpu_task(n: int = 2_000_000) -> float:
    """CPU-bound pure Python."""
    total = 0.0
    for i in range(n):
        total += i * i * 0.001
    return total


def _cpu_task_wrapper(_) -> float:
    """Top-level wrapper for ProcessPoolExecutor."""
    return cpu_task()


def cpu_task_numpy(n: int = 2_000_000) -> float:
    """CPU-bound with NumPy (releases GIL)."""
    arr = np.arange(n, dtype=np.float64)
    return float(np.sum(arr * arr * 0.001))


def bench_io_bound():
    """Compare concurrency models for I/O-bound work."""
    n_tasks = 50
    duration = 0.1

    print(f"I/O-bound ({n_tasks} tasks x {duration}s each):\n")

    # Sequential
    start = time.perf_counter()
    [io_task(i, duration) for i in range(n_tasks)]
    seq_time = time.perf_counter() - start

    # Threading
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_tasks) as pool:
        list(pool.map(_io_task_default, range(n_tasks)))
    thread_time = time.perf_counter() - start

    async def run_async():
        return await asyncio.gather(*[io_task_async(i, duration) for i in range(n_tasks)])

    start = time.perf_counter()
    asyncio.run(run_async())
    async_time = time.perf_counter() - start

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(n_tasks, cpu_count())) as pool:
        list(pool.map(_io_task_default, range(n_tasks)))
    mp_time = time.perf_counter() - start

    print(f"  {'Method':<20} {'Time':>8} {'Speedup':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*10}")
    print(f"  {'Sequential':<20} {seq_time:>6.3f}s {1.0:>9.1f}x")
    print(f"  {'Threading':<20} {thread_time:>6.3f}s {seq_time/thread_time:>9.1f}x")
    print(f"  {'Asyncio':<20} {async_time:>6.3f}s {seq_time/async_time:>9.1f}x")
    print(f"  {'Multiprocessing':<20} {mp_time:>6.3f}s {seq_time/mp_time:>9.1f}x")
    print()


def bench_cpu_bound():
    """Compare concurrency models for CPU-bound work."""
    n_tasks = 4

    print(f"CPU-bound ({n_tasks} pure Python tasks):\n")

    start = time.perf_counter()
    [cpu_task() for _ in range(n_tasks)]
    seq_time = time.perf_counter() - start

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_tasks) as pool:
        list(pool.map(lambda _: cpu_task(), range(n_tasks)))
    thread_time = time.perf_counter() - start

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_tasks) as pool:
        list(pool.map(_cpu_task_wrapper, range(n_tasks)))
    mp_time = time.perf_counter() - start

    print(f"  {'Method':<20} {'Time':>8} {'Speedup':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*10}")
    print(f"  {'Sequential':<20} {seq_time:>6.3f}s {1.0:>9.1f}x")
    print(f"  {'Threading':<20} {thread_time:>6.3f}s {seq_time/thread_time:>9.1f}x  (GIL blocks it)")
    print(f"  {'Multiprocessing':<20} {mp_time:>6.3f}s {seq_time/mp_time:>9.1f}x")
    print()


def bench_numpy_threaded():
    """Show that NumPy + threading DOES work (GIL released in C)."""
    n_tasks = 4

    print(f"NumPy + threading ({n_tasks} tasks, GIL released in C):\n")

    start = time.perf_counter()
    [cpu_task_numpy() for _ in range(n_tasks)]
    seq_time = time.perf_counter() - start

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_tasks) as pool:
        list(pool.map(lambda _: cpu_task_numpy(), range(n_tasks)))
    thread_time = time.perf_counter() - start

    print(f"  Sequential: {seq_time:.3f}s")
    print(f"  Threaded:   {thread_time:.3f}s  ({seq_time/thread_time:.2f}x)")
    print()


if __name__ == "__main__":
    bench_io_bound()
    bench_cpu_bound()
    bench_numpy_threaded()
