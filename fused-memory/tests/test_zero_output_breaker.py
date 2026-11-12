"""Tests for ``fused_memory.middleware.zero_output_breaker.ZeroOutputBreaker``.

Every call passes an explicit ``now``; no clock is patched.
"""

from __future__ import annotations

import pytest

from fused_memory.middleware.zero_output_breaker import ZeroOutputBreaker


def _breaker(threshold: int = 3, cooldown_seconds: float = 60.0) -> ZeroOutputBreaker:
    return ZeroOutputBreaker(threshold=threshold, cooldown_seconds=cooldown_seconds)


def test_a_new_breaker_is_closed_with_no_timeouts():
    breaker = _breaker()

    assert breaker.is_open(0.0) is False
    assert breaker.consecutive_timeouts == 0


def test_it_opens_on_the_timeout_that_reaches_the_threshold():
    breaker = _breaker(threshold=3)

    assert breaker.record_timeout(10.0) is False
    assert breaker.record_timeout(11.0) is False
    assert breaker.is_open(11.0) is False

    assert breaker.record_timeout(12.0) is True
    assert breaker.is_open(12.0) is True


def test_it_stays_open_until_the_cooldown_lapses():
    breaker = _breaker(threshold=1, cooldown_seconds=30.0)

    breaker.record_timeout(100.0)

    assert breaker.is_open(129.9) is True
    assert breaker.is_open(130.0) is False
    assert breaker.is_open(500.0) is False


def test_a_failed_probe_after_the_cooldown_reopens_without_a_fresh_run():
    breaker = _breaker(threshold=3, cooldown_seconds=30.0)
    for now in (0.0, 1.0, 2.0):
        breaker.record_timeout(now)
    assert breaker.is_open(40.0) is False

    assert breaker.record_timeout(40.0) is True
    assert breaker.is_open(40.0) is True
    assert breaker.is_open(69.9) is True
    assert breaker.is_open(70.0) is False


def test_close_ends_the_run_and_closes_the_breaker():
    breaker = _breaker(threshold=3, cooldown_seconds=30.0)
    for now in (0.0, 1.0, 2.0):
        breaker.record_timeout(now)

    breaker.close()

    assert breaker.is_open(3.0) is False
    assert breaker.consecutive_timeouts == 0
    assert breaker.record_timeout(4.0) is False
    assert breaker.record_timeout(5.0) is False
    assert breaker.record_timeout(6.0) is True


def test_consecutive_timeouts_reports_the_current_run_length():
    breaker = _breaker(threshold=2)

    for expected in (1, 2, 3):
        breaker.record_timeout(float(expected))
        assert breaker.consecutive_timeouts == expected

    breaker.close()
    assert breaker.consecutive_timeouts == 0


def test_cooldown_seconds_is_reported():
    assert _breaker(cooldown_seconds=45.0).cooldown_seconds == 45.0


def test_a_threshold_that_could_never_open_is_rejected():
    with pytest.raises(ValueError):
        _breaker(threshold=0)
