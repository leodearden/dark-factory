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
        breaker = NoLandingsCircuitBreaker(window_samples=3, disk_free_floor_bytes=1000)
        # Only one sample — not enough to evaluate
        assert breaker.observe(0, 100_000) is None

    def test_two_samples_below_window_never_trips(self):
        """Two samples with window=3 is still below window — no trip."""
        breaker = NoLandingsCircuitBreaker(window_samples=3, disk_free_floor_bytes=1000)
        breaker.observe(0, 100_000)
        assert breaker.observe(0, 99_000) is None

    def test_landings_advancing_no_trip_even_with_falling_disk(self):
        """Landings increasing across window → None even when disk is falling."""
        breaker = NoLandingsCircuitBreaker(window_samples=3, disk_free_floor_bytes=1000)
        # Feed a full window where landings advance (condition 1 not met)
        # but disk falls (condition 2 would be met in isolation)
        breaker.observe(0, 100_000)
        breaker.observe(1, 99_000)   # landing happened
        result = breaker.observe(2, 98_000)   # another landing
        assert result is None, "landings advancing should never trip"

    def test_flat_landings_stable_disk_no_trip(self):
        """Landings flat + disk stable (not strictly falling) → None."""
        breaker = NoLandingsCircuitBreaker(window_samples=3, disk_free_floor_bytes=1000)
        breaker.observe(5, 100_000)
        breaker.observe(5, 100_000)  # stable disk
        result = breaker.observe(5, 100_000)  # still stable
        assert result is None, "stable disk should not trip even with flat landings"

    def test_flat_landings_rising_disk_no_trip(self):
        """Landings flat + disk rising → None (disk condition not met)."""
        breaker = NoLandingsCircuitBreaker(window_samples=3, disk_free_floor_bytes=1000)
        breaker.observe(5, 90_000)
        breaker.observe(5, 95_000)   # disk rising
        result = breaker.observe(5, 100_000)  # still rising
        assert result is None, "rising disk should not trip"

    def test_observe_returns_none_type(self):
        """observe() return type is None (not False or 0) when no trip."""
        breaker = NoLandingsCircuitBreaker(window_samples=2, disk_free_floor_bytes=500)
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
            window_samples=window, disk_free_floor_bytes=margin
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
        breaker = NoLandingsCircuitBreaker(window_samples=3, disk_free_floor_bytes=1000)
        # Landings advance each step, disk still falls
        breaker.observe(0, 100_000)
        breaker.observe(1, 95_000)
        result = breaker.observe(2, 90_000)
        assert result is None, "landing in window should prevent trip"

    def test_flat_landings_no_trip_when_only_landing_condition_met(self):
        """Landings flat but disk NOT strictly falling (plateau) → None."""
        breaker = NoLandingsCircuitBreaker(window_samples=3, disk_free_floor_bytes=1000)
        breaker.observe(5, 80_000)
        breaker.observe(5, 80_000)  # same disk = not strictly falling
        result = breaker.observe(5, 79_000)  # one fall at the end, not all pairs
        # The plateau in the middle (80k→80k) breaks the strict-monotone check
        assert result is None, "non-strictly-falling disk should not trip"

    # ── idempotency ──────────────────────────────────────────────────────────

    def test_halt_emitted_only_once_while_tripped(self):
        """Once tripped, further flat+falling samples return None (no double-halt).

        Use a high disk_free_floor_bytes (above the observed values) so the
        absolute-floor resume condition is not triggered — we are testing the
        halt-idempotency path, not resume.
        """
        # free after trip: 100_000, 99_000, 98_000; observed values go down to
        # 88_000 — use floor=200_000 so none of those trigger disk-recovery resume
        breaker, _trip = self._fill_and_trip(
            window=3, free_start=100_000, drop=1000, margin=200_000
        )
        # Already tripped; feed more flat+falling samples (all below floor=200k)
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
            window_samples=window, disk_free_floor_bytes=margin
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

    def test_disk_recovery_to_absolute_floor_resumes(self):
        """free_bytes >= disk_free_floor_bytes resumes regardless of trip level."""
        floor = 100_000
        free_start = 200_000
        drop = 10_000
        window = 3
        # free_at_trip = 180_000; floor=100_000 < free_at_trip — proves absolute semantics
        breaker = self._tripped_breaker(
            landings=5,
            free_start=free_start,
            drop=drop,
            window=window,
            margin=floor,
        )
        # Recovery to exactly the floor (below free_at_trip)
        result = breaker.observe(5, floor)
        assert result is not None, "disk recovery to absolute floor should resume"
        assert result.action == 'resume'

    def test_disk_below_floor_stays_tripped(self):
        """free_bytes < disk_free_floor_bytes keeps the breaker tripped."""
        floor = 100_000
        free_start = 200_000
        drop = 10_000
        window = 3
        breaker = self._tripped_breaker(
            landings=5,
            free_start=free_start,
            drop=drop,
            window=window,
            margin=floor,
        )
        # Observe below the floor
        result = breaker.observe(5, floor - 1)
        assert result is None, "free below absolute floor should stay tripped"

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
        breaker = NoLandingsCircuitBreaker(window_samples=window, disk_free_floor_bytes=margin)
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
        breaker = NoLandingsCircuitBreaker(window_samples=window, disk_free_floor_bytes=margin)
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
        breaker = NoLandingsCircuitBreaker(window_samples=window, disk_free_floor_bytes=margin)
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


# ---------------------------------------------------------------------------
# task/1918 step-01: absolute-floor resume semantics
# ---------------------------------------------------------------------------


class TestBreakerAbsoluteFloorResume:
    """NoLandingsCircuitBreaker resumes on an absolute disk-free floor (not
    a relative margin above the trip level).

    RED until step-02 GREEN renames disk_margin_bytes -> disk_free_floor_bytes
    and changes the resume condition.
    """

    def _trip_with_floor(
        self,
        floor: int,
        landings: int = 5,
        free_start: int = 200_000,
        drop: int = 10_000,
        window: int = 3,
    ) -> NoLandingsCircuitBreaker:
        """Drive a breaker (constructed with disk_free_floor_bytes=floor) to
        tripped state and return it."""
        breaker = NoLandingsCircuitBreaker(
            window_samples=window, disk_free_floor_bytes=floor
        )
        for i in range(window):
            breaker.observe(landings, free_start - i * drop)
        assert breaker.is_tripped, "breaker should be tripped after fill"
        return breaker

    def test_disk_recovery_to_absolute_floor_resumes_below_trip_level(self):
        """free_bytes >= floor resumes even when free_bytes < free_at_trip.

        Scenario:
          window=3, floor=100_000
          Fill: free 200_000 -> 190_000 -> 180_000  (free_at_trip=180_000)
          Observe: free=150_000, same landings
          OLD behaviour: 150_000 < 180_000 + margin  -> stay tripped (relative)
          NEW behaviour: 150_000 >= 100_000 floor    -> resume (absolute)
        """
        # free_at_trip = 200_000 - (3-1)*10_000 = 180_000
        # floor = 100_000 < free_at_trip, so old relative margin would NOT resume
        breaker = self._trip_with_floor(
            floor=100_000, landings=5, free_start=200_000, drop=10_000, window=3
        )
        result = breaker.observe(5, 150_000)  # 150_000 >= 100_000 floor
        assert result is not None, (
            "disk recovery to absolute floor (150k >= 100k) should resume, "
            "even though 150k < free_at_trip=180k"
        )
        assert result.action == 'resume'

    def test_disk_below_floor_stays_tripped(self):
        """free_bytes < floor keeps the breaker tripped."""
        breaker = self._trip_with_floor(
            floor=100_000, landings=5, free_start=200_000, drop=10_000, window=3
        )
        result = breaker.observe(5, 90_000)  # 90_000 < 100_000 floor
        assert result is None, "free below floor should stay tripped"

    def test_clean_landing_still_resumes_with_floor(self):
        """A clean landing resumes regardless of the disk-free floor."""
        breaker = self._trip_with_floor(
            floor=500_000,  # unreachably high floor — disk path can never fire
            landings=5,
            free_start=200_000,
            drop=10_000,
            window=3,
        )
        # Increase landings — clean landing always resumes (OR semantics)
        result = breaker.observe(6, 100_000)  # 100_000 << 500_000 floor
        assert result is not None, "clean landing should resume even when disk is below floor"
        assert result.action == 'resume'


# ---------------------------------------------------------------------------
# task/1918 step-09: trip requires disk pressure (below absolute floor)
# ---------------------------------------------------------------------------


class TestBreakerTripRequiresDiskPressure:
    """Trip is gated on disk being below the absolute floor.

    Regression guard for the halt+escalate+resume flap identified in post-
    implementation review: a host with disk above the floor must never trip
    (and therefore can never self-resume in the next breaker interval).

    Use DIRECT NoLandingsCircuitBreaker construction — do NOT use the shared
    _fill_and_trip/_tripped_breaker/_trip_with_floor helpers, which are
    migrated in step-10.  Direct construction keeps these tests stable across
    that migration.

    RED (tests a and b) until step-10 GREEN adds 'last_free < disk_free_floor_bytes'
    as a third AND-condition in _check_trip.
    """

    def test_no_trip_when_disk_stays_above_floor(self):
        """Flat landings + strictly-falling disk ABOVE the floor → no trip.

        Current code: _check_trip fires on flat+falling alone (ignores disk
        level vs floor) → trips at the third observe() → r3 is BreakerTrip.
        Step-10 GREEN: adds 'last_free < disk_free_floor_bytes' as third
        AND-condition → no trip while disk is above floor.
        FAILS on current code (r3 is not None).
        """
        breaker = NoLandingsCircuitBreaker(window_samples=3, disk_free_floor_bytes=100_000)
        # Disk: 200_000 → 190_000 → 180_000 — strictly falling but ALL >= 100_000 floor
        r1 = breaker.observe(5, 200_000)
        r2 = breaker.observe(5, 190_000)
        r3 = breaker.observe(5, 180_000)  # last_free=180_000 >= floor=100_000
        assert r1 is None, f"first observe should return None, got {r1!r}"
        assert r2 is None, f"second observe should return None, got {r2!r}"
        assert r3 is None, (
            f"disk above floor should not trip (got {r3!r}); "
            "last_free=180_000 >= floor=100_000 means no disk pressure"
        )
        assert not breaker.is_tripped, (
            "breaker must not be tripped when disk is above floor"
        )

    def test_no_flap_trip_above_floor_never_self_resumes(self):
        """Long run of flat-landings + strictly-falling-but-above-floor: no halt emitted.

        Regression guard: without the disk-pressure gate, _check_trip fired at
        window completion and _check_resume immediately resumed (free >= floor)
        on the very next pass → halt+escalate+auto-resume flap each window cycle.
        FAILS on current code (halt emitted at the third sample).
        """
        breaker = NoLandingsCircuitBreaker(window_samples=3, disk_free_floor_bytes=100_000)
        halts = []
        # 10 samples falling from 200_000 to 110_000 in 10_000 steps — all >= floor=100_000
        for step in range(10):
            free = 200_000 - step * 10_000  # 200_000, 190_000, ..., 110_000
            result = breaker.observe(5, free)
            if result is not None and result.action == 'halt':
                halts.append(result)
        assert not halts, (
            f"no halt should be emitted when disk stays above floor (got {halts!r}); "
            "halt+auto-resume flap must be eliminated by the disk-pressure gate"
        )
        assert not breaker.is_tripped, (
            "breaker must remain untripped after a long above-floor run"
        )

    def test_trip_fires_when_disk_below_floor(self):
        """Flat landings + strictly-falling disk ALL below floor → trip fires.

        Positive case: documents that disk pressure (all samples below floor)
        still triggers the halt.  Passes on current code (no change to behaviour
        when disk is below the floor).
        """
        breaker = NoLandingsCircuitBreaker(window_samples=3, disk_free_floor_bytes=100_000)
        # Disk: 95_000 → 90_000 → 85_000 — strictly falling AND all < 100_000 floor
        breaker.observe(5, 95_000)
        breaker.observe(5, 90_000)
        result = breaker.observe(5, 85_000)  # last_free=85_000 < floor=100_000
        assert result is not None, (
            "flat landings + falling disk below floor should trip"
        )
        assert result.action == 'halt', f"expected 'halt', got {result!r}"

    def test_below_floor_spiral_stays_halted(self):
        """After a below-floor trip, further below-floor samples sustain the halt.

        Guards against regressing the fix into a no-op: a genuine no-landings
        spiral below the floor must keep the breaker tripped and not auto-resume.
        Passes on current code (resume only on landing or free >= floor, and all
        values here are below floor=100_000 with flat landings).
        """
        breaker = NoLandingsCircuitBreaker(window_samples=3, disk_free_floor_bytes=100_000)
        # Drive a below-floor trip
        breaker.observe(5, 95_000)
        breaker.observe(5, 90_000)
        trip = breaker.observe(5, 85_000)  # trips: last_free=85_000 < floor=100_000
        assert trip is not None and trip.action == 'halt', (
            "trip must fire (below floor) to set up this test"
        )
        # Continue spiral — still below floor, still flat landings
        r1 = breaker.observe(5, 80_000)
        r2 = breaker.observe(5, 75_000)
        r3 = breaker.observe(5, 70_000)
        assert r1 is None, "post-trip sample 1 should return None (no resume yet)"
        assert r2 is None, "post-trip sample 2 should return None"
        assert r3 is None, "post-trip sample 3 should return None"
        assert breaker.is_tripped, (
            "breaker must stay tripped throughout a below-floor no-landings spiral"
        )
