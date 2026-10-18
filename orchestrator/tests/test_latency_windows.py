"""Tests for _latency_windows: judging a wall-clock latency bound by re-measuring a missed window."""
from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Iterable

import pytest
from _latency_windows import (
    LatencyWindow,
    RemeasuredWindows,
    remeasure_until_p99_within,
    time_window,
)

_BOUND_MS = 50.0


class _FakeClock:
    """A monotonic clock that only moves when a window is measured or the policy sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _window(*, p50_ms: float, p99_ms: float) -> LatencyWindow:
    return LatencyWindow(p50_ms=p50_ms, p99_ms=p99_ms, max_ms=p99_ms, samples=200)


_WITHIN = _window(p50_ms=1.0, p99_ms=12.0)
_OVER = _window(p50_ms=1.0, p99_ms=120.0)


def _scripted(
    clock: _FakeClock, windows: Iterable[LatencyWindow], *, cost_s: float = 0.0,
) -> Callable[[], LatencyWindow]:
    """A measure that hands out *windows* in order, each taking *cost_s* of the fake clock."""
    pending = iter(windows)

    def measure() -> LatencyWindow:
        clock.now += cost_s
        return next(pending)

    return measure


def _remeasure(
    clock: _FakeClock, measure: Callable[[], LatencyWindow], *, budget_s: float = 60.0,
    pause_s: float = 1.0,
) -> RemeasuredWindows:
    return remeasure_until_p99_within(
        measure, bound_ms=_BOUND_MS, budget_s=budget_s, pause_s=pause_s,
        clock=clock, sleep=clock.sleep,
    )


def test_of_takes_nearest_rank_p50_p99_and_max_whatever_the_sample_order() -> None:
    samples = [float(n) for n in range(1, 201)]
    random.Random(7090).shuffle(samples)

    window = LatencyWindow.of(samples)

    assert window == LatencyWindow(p50_ms=100.0, p99_ms=198.0, max_ms=200.0, samples=200)


def test_of_refuses_an_empty_sample() -> None:
    with pytest.raises(ValueError, match='at least one sample'):
        LatencyWindow.of([])


def test_remeasured_windows_refuse_to_hold_no_window() -> None:
    with pytest.raises(ValueError, match='at least one window'):
        RemeasuredWindows(windows=(), bound_ms=_BOUND_MS)


def test_a_first_window_within_the_bound_is_the_only_one_measured() -> None:
    clock = _FakeClock()

    remeasured = _remeasure(clock, _scripted(clock, [_WITHIN, _OVER]))

    assert remeasured.windows == (_WITHIN,)
    assert remeasured.met
    assert clock.sleeps == []


def test_missed_windows_are_remeasured_after_a_pause_until_one_is_within() -> None:
    clock = _FakeClock()
    first_over = _window(p50_ms=1.0, p99_ms=80.0)

    remeasured = _remeasure(
        clock, _scripted(clock, [first_over, _OVER, _WITHIN]), pause_s=0.25,
    )

    assert remeasured.windows == (first_over, _OVER, _WITHIN)
    assert remeasured.met
    assert clock.sleeps == [0.25, 0.25]


def test_a_p99_exactly_at_the_bound_misses_and_is_remeasured() -> None:
    clock = _FakeClock()
    at_bound = _window(p50_ms=1.0, p99_ms=_BOUND_MS)

    remeasured = _remeasure(clock, _scripted(clock, [at_bound, _WITHIN]))

    assert remeasured.windows == (at_bound, _WITHIN)
    assert remeasured.met
    assert len(clock.sleeps) == 1


def test_remeasuring_stops_after_the_first_window_ending_at_or_past_the_deadline() -> None:
    clock = _FakeClock()
    clock.now = 1000.0

    remeasured = _remeasure(
        clock, _scripted(clock, itertools.repeat(_OVER), cost_s=1.0), budget_s=5.0, pause_s=1.0,
    )

    assert remeasured.windows == (_OVER, _OVER, _OVER)
    assert not remeasured.met
    assert clock.now == 1005.0


@pytest.mark.parametrize('p50_ms', [_BOUND_MS, 4 * _BOUND_MS])
def test_a_window_whose_median_misses_the_bound_is_not_remeasured(p50_ms: float) -> None:
    clock = _FakeClock()
    slow_median = _window(p50_ms=p50_ms, p99_ms=5 * _BOUND_MS)

    remeasured = _remeasure(clock, _scripted(clock, [slow_median, _WITHIN]), budget_s=600.0)

    assert remeasured.windows == (slow_median,)
    assert not remeasured.met
    assert clock.sleeps == []


def test_time_window_times_each_of_its_samples() -> None:
    calls: list[None] = []

    window = time_window(lambda: calls.append(None), samples=7)

    assert len(calls) == 7
    assert window.samples == 7
    assert 0.0 <= window.p50_ms <= window.p99_ms <= window.max_ms


@pytest.mark.parametrize(('p99_ms', 'within'), [(49.99, True), (_BOUND_MS, False), (50.01, False)])
def test_a_window_is_within_the_bound_only_when_its_p99_is_strictly_below_it(
    p99_ms: float, within: bool,
) -> None:
    window = _window(p50_ms=1.0, p99_ms=p99_ms)

    assert window.p99_within(_BOUND_MS) is within
    assert RemeasuredWindows(windows=(window,), bound_ms=_BOUND_MS).met is within


def test_describe_lists_every_windows_p50_p99_and_max_to_two_decimals_in_order() -> None:
    remeasured = RemeasuredWindows(
        windows=(
            LatencyWindow(p50_ms=1.5, p99_ms=61.234, max_ms=99.9, samples=200),
            LatencyWindow(p50_ms=2.25, p99_ms=72.5, max_ms=130.0, samples=200),
            LatencyWindow(p50_ms=0.8, p99_ms=12.0, max_ms=14.1, samples=200),
        ),
        bound_ms=_BOUND_MS,
    )

    assert remeasured.describe() == (
        '3 window(s) as p50/p99/max ms: 1.50/61.23/99.90, 2.25/72.50/130.00, 0.80/12.00/14.10'
    )
