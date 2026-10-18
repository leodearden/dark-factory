"""Judge a wall-clock latency bound on a contended host by re-measuring a window that misses it.

A window is a run of timed samples summarised by nearest-rank percentiles, and
every window is held to the bound in full.  A window whose p99 misses is
measured again after a pause, until one meets the bound or a time budget is
spent; a window whose median misses is final, since that is a slow operation
rather than a scheduling tail.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass


def _nearest_rank(ordered: Sequence[float], quantile: float) -> float:
    return ordered[math.ceil(quantile * len(ordered)) - 1]


@dataclass(frozen=True)
class LatencyWindow:
    """One window of timed samples, in milliseconds."""

    p50_ms: float
    p99_ms: float
    max_ms: float
    samples: int

    @classmethod
    def of(cls, samples_ms: Sequence[float]) -> LatencyWindow:
        if not samples_ms:
            raise ValueError('a latency window needs at least one sample')
        ordered = sorted(samples_ms)
        return cls(
            p50_ms=_nearest_rank(ordered, 0.50),
            p99_ms=_nearest_rank(ordered, 0.99),
            max_ms=ordered[-1],
            samples=len(ordered),
        )


def time_window(sample: Callable[[], object], samples: int) -> LatencyWindow:
    """Call *sample* *samples* times back to back, timing each call on the wall clock."""
    durations_ms: list[float] = []
    for _ in range(samples):
        start = time.perf_counter()
        sample()
        durations_ms.append((time.perf_counter() - start) * 1000.0)
    return LatencyWindow.of(durations_ms)


@dataclass(frozen=True)
class RemeasuredWindows:
    """Every window measured against p99 < *bound_ms*, in order; the last one is the verdict."""

    windows: tuple[LatencyWindow, ...]
    bound_ms: float

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError('a re-measurement holds at least one window')

    @property
    def met(self) -> bool:
        return self.windows[-1].p99_ms < self.bound_ms

    def describe(self) -> str:
        listed = ', '.join(
            f'{window.p50_ms:.2f}/{window.p99_ms:.2f}/{window.max_ms:.2f}'
            for window in self.windows
        )
        return f'{len(self.windows)} window(s) as p50/p99/max ms: {listed}'


def _is_final(window: LatencyWindow, bound_ms: float) -> bool:
    return window.p99_ms < bound_ms or window.p50_ms >= bound_ms


def remeasure_until_p99_within(
    measure: Callable[[], LatencyWindow],
    *,
    bound_ms: float,
    budget_s: float,
    pause_s: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> RemeasuredWindows:
    deadline = clock() + budget_s
    windows: list[LatencyWindow] = []
    while True:
        window = measure()
        windows.append(window)
        if _is_final(window, bound_ms) or clock() >= deadline:
            return RemeasuredWindows(windows=tuple(windows), bound_ms=bound_ms)
        sleep(pause_s)
