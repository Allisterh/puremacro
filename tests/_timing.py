"""Wall-clock budgets for timing assertions.

Budgets are written for a developer machine. Shared CI runners are several
times slower and noisier (Windows runners measured 0.56 s against a 0.15 s
budget), so CI sets ``PUREMACRO_TIMING_SCALE`` to widen every budget at once.
A scaled budget still catches the non-terminating loops these tests guard.

Rules for timing tests:

* Every absolute wall-clock assertion goes through :func:`budget` (seconds) or
  :func:`budget_ms` (milliseconds), and its failure message prints the
  *scaled* value, so a CI failure reads the number that was actually enforced.
  :func:`assert_within` does both.
* Repeat a measurement with :func:`best_of` (the minimum after a warm-up), not
  the maximum of several runs: the maximum measures the noisiest moment of a
  shared runner, the minimum measures the code.
* Ratio checks (``t_slow / t_fast >= k``) are **not** scaled. Both timings come
  from the same machine in the same process, so a uniformly slower runner
  leaves the ratio unchanged; the scale factor applies only to absolute
  budgets.
* Pure performance checks carry ``@pytest.mark.benchmark``; hang guards (a
  generous ceiling that catches a non-terminating loop) do not.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable

__all__ = ["timing_scale", "budget", "budget_ms", "best_of", "assert_within"]


def timing_scale() -> float:
    """The multiplier applied to every absolute budget (``PUREMACRO_TIMING_SCALE``)."""
    return float(os.environ.get("PUREMACRO_TIMING_SCALE", "1"))


def budget(seconds: float) -> float:
    """A wall-clock budget in seconds, widened by ``PUREMACRO_TIMING_SCALE``."""
    return seconds * timing_scale()


def budget_ms(ms: float) -> float:
    """A wall-clock budget in milliseconds, widened by ``PUREMACRO_TIMING_SCALE``."""
    return ms * timing_scale()


def best_of(fn: Callable[[], Any], n: int = 3, *, warmup: int = 1) -> float:
    """Minimum ``perf_counter`` duration of ``fn()`` over ``n`` calls, in seconds.

    ``warmup`` untimed calls run first so caches, lazy imports and JIT-like
    memoisation do not land in the measurement.
    """
    if n < 1:
        raise ValueError("best_of needs n >= 1")
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def assert_within(elapsed: float, seconds: float, what: str) -> None:
    """Assert ``elapsed <= budget(seconds)``, printing the scaled budget on failure."""
    limit = budget(seconds)
    assert elapsed <= limit, (
        f"{what} took {elapsed:.4g}s > budget {limit:.4g}s "
        f"({seconds}s x PUREMACRO_TIMING_SCALE={timing_scale():g})")
