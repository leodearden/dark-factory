"""Tests for θ=1893: NoLandingsCircuitBreaker (pure detection logic).

Covers:
  step-01 RED  — No-trip healthy cases (< window samples, landings advancing,
                 disk stable/rising even with landings flat)
  step-03 RED  — Trip behavior: both conditions fire, AND semantics, idempotency
  step-05 RED  — Reset behavior while tripped (landing resume, disk recovery,
                 hysteresis below margin stays tripped)
  step-07 RED  — No-flap: full-window rebuild required after resume
"""
from __future__ import annotations

import pytest

from orchestrator.merge_queue import BreakerTrip, NoLandingsCircuitBreaker


# ---------------------------------------------------------------------------
# step-01: healthy / no-trip cases
# ---------------------------------------------------------------------------


class TestBreakerHealthyNoTrip:
    """NoLandingsCircuitBreaker.observe() returns None for all healthy inputs.

    RED until step-02 GREEN adds BreakerTrip + NoLandingsCircuitBreaker to
    merge_queue.py.
    """

    def test_single_sample_below_window_never_trips(self):
        """With fewer than window_samples observations, observe() is always None."""
        breaker = NoLandingsCircuitBreaker(window_samples=3, disk_margin_bytes=1000)
        # Only one sample — not enough to evaluate
        assert breaker.observe(0, 100_000) is None

    def test_two_samples_below_window_never_trips(self):
        """Two samples with window=3 is still below window — no trip."""
        breaker = NoLandingsCircuitBreaker(window_samples=3, disk_margin_bytes=1000)
        breaker.observe(0, 100_000)
        assert breaker.observe(0, 99_000) is None

    def test_landings_advancing_no_trip_even_with_falling_disk(self):
        """Landings increasing across window → None even when disk is falling."""
        breaker = NoLandingsCircuitBreaker(window_samples=3, disk_margin_bytes=1000)
        # Feed a full window where landings advance (condition 1 not met)
        # but disk falls (condition 2 would be met in isolation)
        breaker.observe(0, 100_000)
        breaker.observe(1, 99_000)   # landing happened
        result = breaker.observe(2, 98_000)   # another landing
        assert result is None, "landings advancing should never trip"

    def test_flat_landings_stable_disk_no_trip(self):
        """Landings flat + disk stable (not strictly falling) → None."""
        breaker = NoLandingsCircuitBreaker(window_samples=3, disk_margin_bytes=1000)
        breaker.observe(5, 100_000)
        breaker.observe(5, 100_000)  # stable disk
        result = breaker.observe(5, 100_000)  # still stable
        assert result is None, "stable disk should not trip even with flat landings"

    def test_flat_landings_rising_disk_no_trip(self):
        """Landings flat + disk rising → None (disk condition not met)."""
        breaker = NoLandingsCircuitBreaker(window_samples=3, disk_margin_bytes=1000)
        breaker.observe(5, 90_000)
        breaker.observe(5, 95_000)   # disk rising
        result = breaker.observe(5, 100_000)  # still rising
        assert result is None, "rising disk should not trip"

    def test_observe_returns_none_type(self):
        """observe() return type is None (not False or 0) when no trip."""
        breaker = NoLandingsCircuitBreaker(window_samples=2, disk_margin_bytes=500)
        result = breaker.observe(0, 50_000)
        assert result is None


# ---------------------------------------------------------------------------
# step-03: trip behavior
# ---------------------------------------------------------------------------


class TestBreakerTrip:
    """NoLandingsCircuitBreaker trips on flat-landings + strictly-falling disk.

    RED until step-04 GREEN implements trip detection in observe().
    """

    def _fill_and_trip(
        self,
        window: int = 3,
        margin: int = 1000,
        landing_start: int = 10,
        free_start: int = 200_000,
        drop: int = 5000,
    ) -> tuple[NoLandingsCircuitBreaker, BreakerTrip]:
        """Drive a breaker to trip state; return (breaker, trip)."""
        breaker = NoLandingsCircuitBreaker(
            window_samples=window, disk_margin_bytes=margin
        )
        trip = None
        for i in range(window):
            trip = breaker.observe(landing_start, free_start - i * drop)
        assert trip is not None, "expected a trip after full window of flat+falling"
        assert isinstance(trip, BreakerTrip)
        return breaker, trip

    def test_trip_fires_after_full_window_of_flat_landings_and_falling_disk(self):
        """Full window of flat landings + strictly falling disk → BreakerTrip(action='halt')."""
        _breaker, trip = self._fill_and_trip()
        assert trip.action == 'halt'

    def test_trip_carries_correct_window_samples(self):
        """BreakerTrip.window_samples equals the configured window."""
        _breaker, trip = self._fill_and_trip(window=4)
        assert trip.window_samples == 4

    def test_trip_carries_zero_landings_in_window(self):
        """BreakerTrip.landings_in_window is 0 when landings were flat."""
        _breaker, trip = self._fill_and_trip(landing_start=7)
        assert trip.landings_in_window == 0

    def test_trip_carries_free_start_and_end(self):
        """BreakerTrip.free_start / free_end match window first/last free-bytes."""
        window = 3
        free_start = 300_000
        drop = 10_000
        free_end = free_start - (window - 1) * drop
        _breaker, trip = self._fill_and_trip(
            window=window, free_start=free_start, drop=drop
        )
        assert trip.free_start == free_start
        assert trip.free_end == free_end

    def test_trip_reason_is_non_empty_string(self):
        """BreakerTrip.reason is a non-empty string (used in escalation message)."""
        _breaker, trip = self._fill_and_trip()
        assert isinstance(trip.reason, str)
        assert len(trip.reason) > 0

    # ── AND semantics ────────────────────────────────────────────────────────

    def test_flat_disk_no_trip_when_only_disk_condition_met(self):
        """Disk strictly falling but landings advancing → None (AND semantics)."""
        breaker = NoLandingsCircuitBreaker(window_samples=3, disk_margin_bytes=1000)
        # Landings advance each step, disk still falls
        breaker.observe(0, 100_000)
        breaker.observe(1, 95_000)
        result = breaker.observe(2, 90_000)
        assert result is None, "landing in window should prevent trip"

    def test_flat_landings_no_trip_when_only_landing_condition_met(self):
        """Landings flat but disk NOT strictly falling (plateau) → None."""
        breaker = NoLandingsCircuitBreaker(window_samples=3, disk_margin_bytes=1000)
        breaker.observe(5, 80_000)
        breaker.observe(5, 80_000)  # same disk = not strictly falling
        result = breaker.observe(5, 79_000)  # one fall at the end, not all pairs
        # The plateau in the middle (80k→80k) breaks the strict-monotone check
        assert result is None, "non-strictly-falling disk should not trip"

    # ── idempotency ──────────────────────────────────────────────────────────

    def test_halt_emitted_only_once_while_tripped(self):
        """Once tripped, further flat+falling samples return None (no double-halt)."""
        breaker, _trip = self._fill_and_trip(window=3, free_start=100_000, drop=1000)
        # Already tripped; feed more flat+falling samples
        r1 = breaker.observe(10, 90_000)
        r2 = breaker.observe(10, 89_000)
        r3 = breaker.observe(10, 88_000)
        assert r1 is None
        assert r2 is None
        assert r3 is None


# ---------------------------------------------------------------------------
# step-05: reset / resume behavior while tripped
# ---------------------------------------------------------------------------


class TestBreakerReset:
    """Reset (resume) behavior when the breaker is tripped.

    RED until step-06 GREEN implements reset in observe().
    """

    def _tripped_breaker(
        self,
        landings: int = 5,
        free_start: int = 200_000,
        drop: int = 10_000,
        window: int = 3,
        margin: int = 5_000,
    ) -> NoLandingsCircuitBreaker:
        """Return a breaker that is already in tripped state."""
        breaker = NoLandingsCircuitBreaker(
            window_samples=window, disk_margin_bytes=margin
        )
        for i in range(window):
            breaker.observe(landings, free_start - i * drop)
        return breaker

    def test_clean_landing_resumes(self):
        """Observe with landings_total > landings-at-trip → BreakerTrip(action='resume')."""
        breaker = self._tripped_breaker(landings=10)
        result = breaker.observe(11, 100_000)   # landing happened
        assert result is not None
        assert result.action == 'resume'

    def test_disk_recovery_above_margin_resumes(self):
        """free_bytes >= free-at-trip + disk_margin_bytes → 'resume'."""
        margin = 5_000
        free_start = 200_000
        drop = 10_000
        window = 3
        # After filling the window the trip-level free_bytes = free_start - (window-1)*drop
        free_at_trip = free_start - (window - 1) * drop  # = 180_000
        breaker = self._tripped_breaker(
            landings=5,
            free_start=free_start,
            drop=drop,
            window=window,
            margin=margin,
        )
        # Recovery: free_bytes rises to free_at_trip + margin (exactly at threshold)
        recovery_bytes = free_at_trip + margin
        result = breaker.observe(5, recovery_bytes)
        assert result is not None, "disk recovery to trip+margin should resume"
        assert result.action == 'resume'

    def test_disk_partial_recovery_below_margin_stays_tripped(self):
        """free_bytes rose but by less than disk_margin_bytes → None (stays tripped)."""
        margin = 5_000
        free_start = 200_000
        drop = 10_000
        window = 3
        free_at_trip = free_start - (window - 1) * drop  # = 180_000
        breaker = self._tripped_breaker(
            landings=5,
            free_start=free_start,
            drop=drop,
            window=window,
            margin=margin,
        )
        # Rise is only margin - 1 bytes above trip level (below threshold)
        partial_recovery = free_at_trip + margin - 1
        result = breaker.observe(5, partial_recovery)
        assert result is None, "partial disk recovery below margin should stay tripped"

    def test_after_resume_observe_returns_none(self):
        """After a 'resume', the breaker is no longer tripped."""
        breaker = self._tripped_breaker(landings=10)
        # Resume via landing
        r_resume = breaker.observe(11, 150_000)
        assert r_resume is not None and r_resume.action == 'resume'
        # Subsequent observe — breaker is fresh; does not immediately re-trip
        r_after = breaker.observe(11, 150_000)
        assert r_after is None


# ---------------------------------------------------------------------------
# step-07: no-flap — full-window rebuild required after resume
# ---------------------------------------------------------------------------


class TestBreakerNoFlap:
    """Anti-flap: a single bad sample after resume does NOT re-trip.

    RED until step-08 GREEN clears the sample buffer on resume.
    """

    def test_single_flat_falling_sample_after_resume_does_not_retip(self):
        """Immediately after resume, one flat+falling sample returns None."""
        # Drive to trip
        window = 3
        margin = 1_000
        breaker = NoLandingsCircuitBreaker(window_samples=window, disk_margin_bytes=margin)
        for i in range(window):
            breaker.observe(5, 100_000 - i * 5_000)
        # Resume via a clean landing
        r_resume = breaker.observe(6, 90_000)
        assert r_resume is not None and r_resume.action == 'resume', (
            f"expected resume, got {r_resume!r}"
        )
        # One flat+falling sample right after resume must NOT re-trip
        r_after = breaker.observe(6, 85_000)
        assert r_after is None, (
            "single flat+falling sample after resume should not re-trip (buffer not full)"
        )

    def test_full_window_after_resume_can_retip(self):
        """After resume, a fresh full window of flat+falling samples does re-trip."""
        window = 3
        margin = 1_000
        breaker = NoLandingsCircuitBreaker(window_samples=window, disk_margin_bytes=margin)
        # Trip
        for i in range(window):
            breaker.observe(5, 100_000 - i * 5_000)
        # Resume
        r = breaker.observe(6, 200_000)
        assert r is not None and r.action == 'resume'
        # Now accumulate a full new window of flat+falling  (buffer was cleared)
        trip2 = None
        for i in range(window):
            trip2 = breaker.observe(6, 200_000 - i * 5_000)
        assert trip2 is not None and trip2.action == 'halt', (
            "fresh full window after resume should re-trip"
        )

    def test_partial_window_after_resume_does_not_retip(self):
        """window-1 samples after resume are still insufficient to re-trip."""
        window = 4
        margin = 500
        breaker = NoLandingsCircuitBreaker(window_samples=window, disk_margin_bytes=margin)
        # Trip
        for i in range(window):
            breaker.observe(3, 80_000 - i * 2_000)
        # Resume
        breaker.observe(4, 200_000)
        # Only window-1 = 3 flat+falling samples (one short of window)
        result = None
        for i in range(window - 1):
            result = breaker.observe(4, 200_000 - i * 2_000)
        assert result is None, f"window-1 samples should not re-trip, got {result!r}"
