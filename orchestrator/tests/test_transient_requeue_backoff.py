"""Jittered exponential backoff for transient (5xx) API requeues.

Task 3317 / PRD ``plans/server-side-api-error-handling-prd.md`` task δ,
contract C3, resolved decision 7.

Origin incident (2026-07-29 provider outage): the flat 30s
``requeue_cooldown_secs`` turned a sustained provider 5xx into a retry
storm — 67 starts in a single half-hour bucket.  Contract C3 replaces the
flat cooldown *for transient-classified requeues only* with

    envelope(n) = min(base * 2**(n-1), cap)      # base=30.0, cap=900.0
    armed       = U(envelope/2, envelope)        # equal jitter

so the armed cooldowns grow ~30 → 480s over n=1..5 and clamp at 900s.
GENUINE requeues keep the flat ``requeue_cooldown_secs`` — the carve-out
that boundary row 4 pins.
"""
from __future__ import annotations

import pytest

from orchestrator.scheduler import transient_requeue_cooldown

# Contract C3 defaults, and the closed-form envelope they produce.  Every
# assertion below is exact arithmetic — no numeric tolerance guesswork.
BASE = 30.0
CAP = 900.0
# n:                       1     2      3      4      5      6
ENVELOPES = [30.0, 60.0, 120.0, 240.0, 480.0, 900.0]


class TestTransientRequeueCooldownFormula:
    """The pure ``transient_requeue_cooldown`` helper (no Scheduler involved)."""

    @pytest.mark.parametrize('n,expected', list(enumerate(ENVELOPES, start=1)))
    def test_envelope_is_exact_doubling_then_clamps(self, n, expected):
        """envelope(n) = min(base * 2**(n-1), cap), exactly.

        n=6 is the clamp boundary: 30 * 2**5 = 960 > 900, so it pins at the
        cap rather than overshooting.
        """
        _armed, envelope = transient_requeue_cooldown(
            n, base_secs=BASE, cap_secs=CAP, rng=lambda lo, hi: hi,
        )
        assert envelope == expected

    @pytest.mark.parametrize('n', [7, 8, 20])
    def test_envelope_stays_pinned_at_cap_beyond_the_clamp(self, n):
        """Past the clamp the schedule is monotone and capped — never unbounded."""
        _armed, envelope = transient_requeue_cooldown(
            n, base_secs=BASE, cap_secs=CAP, rng=lambda lo, hi: hi,
        )
        assert envelope == CAP

    @pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
    def test_lower_jitter_edge_is_exactly_half_the_envelope(self, n):
        """The jitter FLOOR is envelope/2 — 15.0s at n=1, never 0."""
        armed, envelope = transient_requeue_cooldown(
            n, base_secs=BASE, cap_secs=CAP, rng=lambda lo, hi: lo,
        )
        assert armed == envelope / 2
        assert armed == ENVELOPES[n - 1] / 2

    @pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
    def test_upper_jitter_edge_is_exactly_the_envelope(self, n):
        armed, envelope = transient_requeue_cooldown(
            n, base_secs=BASE, cap_secs=CAP, rng=lambda lo, hi: hi,
        )
        assert armed == envelope
        assert armed == ENVELOPES[n - 1]

    def test_jitter_floor_at_n1_is_fifteen_seconds(self):
        """Pinned explicitly: the shortest cooldown this can ever arm is 15s.

        The capability manifest states the floor; a regression that dropped
        the ``/2`` (drawing from ``[0, envelope]``) would reintroduce
        near-zero cooldowns — the exact tight-loop pathology this task exists
        to remove — while still passing every envelope assertion above.
        """
        armed, envelope = transient_requeue_cooldown(
            1, base_secs=BASE, cap_secs=CAP, rng=lambda lo, hi: lo,
        )
        assert (armed, envelope) == (15.0, 30.0)

    def test_rng_is_called_with_equal_jitter_bounds(self):
        """The draw interval is exactly ``(envelope/2, envelope)`` — equal jitter."""
        calls: list[tuple[float, float]] = []

        def _recording(lo: float, hi: float) -> float:
            calls.append((lo, hi))
            return hi

        transient_requeue_cooldown(3, base_secs=BASE, cap_secs=CAP, rng=_recording)
        assert calls == [(60.0, 120.0)]

    def test_default_rng_draws_live_values_inside_the_envelope(self):
        """``rng=None`` falls back to ``random.uniform`` and jitter is real.

        200 draws at n=2 must all land in [30, 60] (the envelope contract),
        and produce more than one distinct value (the jitter is live, not a
        constant that happens to sit inside the band).
        """
        draws = [
            transient_requeue_cooldown(2, base_secs=BASE, cap_secs=CAP)[0]
            for _ in range(200)
        ]
        assert all(30.0 <= d <= 60.0 for d in draws), (
            f'draw outside the n=2 envelope [30, 60]: {sorted(draws)[:5]}...'
        )
        assert len(set(draws)) > 1, 'rng default is not producing jittered values'

    @pytest.mark.parametrize('n', [0, -3])
    def test_degenerate_n_clamps_to_the_first_envelope(self, n):
        """n<1 must clamp to n=1 — never a zero or negative cooldown.

        A caller can only reach here with a post-increment count (>=1), but
        an arithmetic slip upstream must degrade to the *safe* end (a full
        base cooldown), not to a hot loop.
        """
        armed, envelope = transient_requeue_cooldown(
            n, base_secs=BASE, cap_secs=CAP, rng=lambda lo, hi: hi,
        )
        assert envelope == 30.0
        assert armed == 30.0

    def test_base_above_cap_yields_the_cap(self):
        """A mis-config (base > cap) resolves to the cap, not to base."""
        armed, envelope = transient_requeue_cooldown(
            1, base_secs=100.0, cap_secs=20.0, rng=lambda lo, hi: hi,
        )
        assert envelope == 20.0
        assert armed == 20.0
