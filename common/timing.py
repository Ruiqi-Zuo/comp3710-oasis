"""Wall-clock timing that is safe to use around CUDA work.

CUDA kernel launches are asynchronous: without an explicit synchronise, a naive
`time.perf_counter()` around a GPU call measures only the launch overhead and
makes the GPU look impossibly fast. Part 1 asks you to rank three
implementations by speed, so getting this right matters.
"""

from __future__ import annotations

import time

import torch


class Timer:
    """Context manager measuring elapsed seconds, synchronising CUDA if needed.

    Example:
        >>> with Timer("naive DFT (GPU)") as t:
        ...     X = naive_dft_torch(signal)
        >>> print(t.seconds)
    """

    def __init__(self, label: str = "", sync: bool = True) -> None:
        self.label = label
        self.sync = sync
        self.seconds: float = float("nan")

    def _barrier(self) -> None:
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    def __enter__(self) -> "Timer":
        self._barrier()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        self._barrier()
        self.seconds = time.perf_counter() - self._start
        return None

    def __str__(self) -> str:
        return f"{self.label}: {self.seconds:.6f} s"


def benchmark(fn, *args, warmup: int = 1, repeats: int = 3, label: str = "", **kwargs):
    """Run ``fn`` a few times and return ``(result, best_seconds)``.

    A warm-up call absorbs one-off costs (CUDA context creation, kernel
    autotuning) that would otherwise dominate the first measurement.
    """
    for _ in range(warmup):
        fn(*args, **kwargs)

    best = float("inf")
    result = None
    for _ in range(repeats):
        with Timer(label) as timer:
            result = fn(*args, **kwargs)
        best = min(best, timer.seconds)
    return result, best
