"""Replay harness for the EWA statistic — task 4559, esc-4559-3.

Demonstrates, executably, that excluding escalation RESOLUTIONS from the EWA
numerator fixes the breaker in both directions:

* **False positive** — a pure backlog-drain window (many resolutions, zero new
  submissions) trips the OLD statistic and provably cannot trip the NEW one.
* **Miss** — the ORDERING of a fault burst against a routine drain is inverted
  under the OLD statistic and restored under the NEW one.

On the miss direction, note carefully what is and is not asserted. The task
description's literal signal was "the 2026-08-16/17 burst should trip 24.6, or
at least score far higher than its measured 19.96". Both disjuncts are
refuted by the fix's own algebra: ``digest.update_ewa`` is monotone
non-decreasing in ``escalations_in_step``, and excluding resolutions makes the
numerator pointwise <= today's over the identical window, so by induction the
NEW EWA is pointwise <= the OLD EWA at every step, with equality iff the window
has zero resolutions. The burst's OLD peak IS 19.96, so its NEW peak is
<= 19.96 — it can neither reach 24.6 nor exceed 19.96. That agrees with the
task's own struck claim: the miss is driven by the DENOMINATOR (high done
counts absorbing the burst), which a numerator edit cannot reach. The
discriminative ORDERING is the strongest true form of the signal, and it is
exactly what the fix restores; the denominator-side detector is filed as
follow-up work. Recorded as esc-4559-3.

**What the assertions here rest on (amendment, reviewer_comprehensive
test-coverage).** Every pass/fail claim below is a CONSTRUCTION proof over a
whole class of windows, swept across parameterisations — never a property of
one hand-authored triple. That distinction is load-bearing: the two window
constants are DOCUMENTED MODELS, not replays of live data (the task's own
scope caveat records that ``data/escalations/archive/`` only reaches back to
2026-07-21 because archiving keys on ``resolved_at``, so the period is not
reconstructible and a live replay would be non-deterministic in CI anyway).
A fabricated all-resolution window trips the old statistic BY ARITHMETIC, and
a fabricated burst/drain pair can be tuned to invert either way — so asserting
against the models alone would have proved nothing about the incident and
would have read as empirical evidence the fixtures cannot supply. The models
are therefore named ``MODEL_*`` and appear only as ONE illustrative member of
each swept family, never as the sole basis of a claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import orchestrator.digest as digest

ALPHA = 0.3
THRESHOLD = 24.6


# ---------------------------------------------------------------------------
# Test-local replay harness
# ---------------------------------------------------------------------------
#
# This lives in the test module, NOT in orchestrator/digest.py (amendment,
# reviewer_comprehensive over-engineering).  Two reasons:
#
#  1. This suite is its only consumer — there is no script, CLI or operator
#     entry point exposing an offline replay, so shipping it in the production
#     module was test-only scaffolding on the import path.
#  2. Its ``count_resolutions=True`` arm permanently encodes the RETIRED
#     pre-4559 statistic.  Production never takes that branch, and a future
#     reader changing ``update_ewa`` should not have to reason about it.
#
# Drift risk stays closed because all per-step arithmetic is delegated to
# ``digest.update_ewa``: the only thing implemented here is the numerator
# selection, which is precisely the production difference being demonstrated.


@dataclass(frozen=True)
class EwaReplayStep:
    """One replayed EWA step — the inputs, the chosen numerator, and the result."""

    submissions: int
    resolutions: int
    dones: int
    # The numerator actually fed to update_ewa: submissions, or
    # submissions + resolutions under the pre-4559 statistic.
    numerator: int
    ratio: float
    ewa: float
    tripped: bool


@dataclass(frozen=True)
class EwaReplaySeries:
    """The result of replaying a step sequence through the EWA statistic."""

    steps: list[EwaReplayStep]
    peak: float
    # Index of the first step whose EWA reached the threshold; None if never.
    tripped_at: int | None


def replay_ewa_series(
    steps,
    *,
    alpha: float,
    threshold: float,
    count_resolutions: bool,
    prev_ewa: float = 0.0,
) -> EwaReplaySeries:
    """Replay a sequence of digest steps through the EWA statistic.

    Pure — no I/O, no Harness, no clock — so it is deterministic in CI.

    *steps* is a sequence of ``(submissions, resolutions, dones)`` triples, one
    per digest write.

    *count_resolutions* selects the numerator, and is the ONLY difference
    between the two statistics::

        numerator = submissions + resolutions  if count_resolutions  # pre-4559
        numerator = submissions                otherwise             # current

    Everything else — including the per-step arithmetic — is delegated to
    :func:`digest.update_ewa`, deliberately: a second implementation of the
    formula here could drift from the live statistic and quietly turn this into
    a replay of something production does not do.  The
    ``count_resolutions=True`` arm is therefore a faithful reproduction of
    pre-4559 behaviour rather than a re-derivation of it.

    Returns the per-step detail, the peak EWA reached (including *prev_ewa*, so
    a series that only decays reports its starting value), and the index of the
    first step at or above *threshold* (``None`` if it never reaches it).
    """
    replayed: list[EwaReplayStep] = []
    ewa = prev_ewa
    peak = prev_ewa
    tripped_at: int | None = None

    for index, (submissions, resolutions, dones) in enumerate(steps):
        numerator = submissions + resolutions if count_resolutions else submissions
        ewa = digest.update_ewa(
            prev_ewa=ewa,
            escalations_in_step=numerator,
            done_in_step=dones,
            alpha=alpha,
        )
        tripped = ewa >= threshold
        if tripped and tripped_at is None:
            tripped_at = index
        peak = max(peak, ewa)
        replayed.append(
            EwaReplayStep(
                submissions=submissions,
                resolutions=resolutions,
                dones=dones,
                numerator=numerator,
                ratio=numerator / max(dones, 1),
                ewa=ewa,
                tripped=tripped,
            )
        )

    return EwaReplaySeries(steps=replayed, peak=peak, tripped_at=tripped_at)


# ---------------------------------------------------------------------------
# Documented window MODELS — illustrative only
# ---------------------------------------------------------------------------
#
# Shaped from the task's measured aggregates (the 2026-08-20 drain: 196
# lifecycle events with a 67-member cluster-L2 resolve contributing 49, no work
# landing; the 2026-08-16/17 burst: 297 submissions over ~18h with work landing
# throughout).  They are NOT replays of live data and no assertion rests on
# them alone — each appears as one member of a swept family below.

MODEL_DRAIN_2026_08_20 = [
    (0, 30, 0),
    (0, 35, 0),
    (0, 49, 0),
    (0, 30, 0),
    (0, 28, 0),
    (0, 24, 0),
]

MODEL_BURST_2026_08_16 = [
    (30, 2, 2),
    (35, 4, 2),
    (40, 6, 2),
    (38, 8, 2),
    (34, 8, 2),
    (30, 10, 2),
    (28, 10, 2),
    (25, 10, 2),
    (20, 10, 2),
    (17, 10, 2),
]


def _drain(resolutions: int, dones: int, steps: int) -> list[tuple[int, int, int]]:
    """A pure-drain window: *steps* digest steps of cleanup with zero submissions."""
    return [(0, resolutions, dones)] * steps


class TestReplayEwaSeries:
    """The EWA statistic, replayed under both numerators."""

    def test_old_statistic_trips_on_any_large_pure_drain_and_new_never_does(self) -> None:
        """The false-positive direction, proved over the CLASS of drain windows.

        This is the outage itself: a window whose only activity was CLEANING UP
        a backlog halted fleet-wide dispatch, because every resolution counted
        as if it were a fresh escalation.

        The claim is a construction, not a fixture property.  Under the OLD
        numerator a pure-drain step has ratio ``resolutions / max(dones, 1)``,
        so the series converges to that ratio from below and trips for ANY
        drain whose per-done resolution rate exceeds the threshold — an
        unbounded function of backlog size, which is why a big enough cleanup
        always trips.  Under the NEW numerator the same window has numerator 0
        at every step, so from a healthy start the EWA is identically zero and
        no drain volume whatsoever can trip it.

        The swept family includes the documented 2026-08-20 model as one
        member; the assertion does not depend on it.
        """
        # (resolutions, dones) pairs whose ratio exceeds THRESHOLD, spanning
        # zero-done cleanup and cleanup alongside a little work landing.
        families: list[tuple[str, list[tuple[int, int, int]]]] = [
            (f'drain r={r} dones={d} steps={n}', _drain(r, d, n))
            for r, d in ((25, 0), (30, 1), (49, 0), (73, 1), (100, 3), (200, 5))
            for n in (12, 40)
        ]
        families.append(('MODEL_DRAIN_2026_08_20', MODEL_DRAIN_2026_08_20))

        for label, window in families:
            old = replay_ewa_series(
                window, alpha=ALPHA, threshold=THRESHOLD, count_resolutions=True
            )
            assert old.tripped_at is not None, (
                f'{label}: expected the OLD statistic to trip on a pure-drain '
                f'window (that is the outage); peak was {old.peak}'
            )

            new = replay_ewa_series(
                window, alpha=ALPHA, threshold=THRESHOLD, count_resolutions=False
            )
            assert new.tripped_at is None, (
                f'{label}: expected the NEW statistic NOT to trip on a '
                f'pure-drain window; tripped at step {new.tripped_at} with '
                f'peak {new.peak}'
            )
            assert new.peak == pytest.approx(0.0), (
                f'{label}: a pure-drain window from a healthy start has '
                f'numerator 0 at every step, so its NEW peak must be 0; '
                f'got {new.peak}'
            )

    def test_pure_drain_cannot_trip_by_construction(self) -> None:
        """A zero-submission window is strictly decreasing, so it cannot trip.

        Needs no tuned bound and no fixture: with the numerator at zero,
        update_ewa reduces to EWA(t+1) = (1 - alpha) * EWA(t), which for
        0 < alpha <= 1 is strictly decreasing from any positive start.

        Asserted in the two provable forms. From a HEALTHY start the series
        never trips at all, for any step count — a strictly decreasing series
        cannot cross an upward threshold. From the recorded 2026-08-20 trip
        value of 73.59 the series necessarily BEGINS above 24.6 (that is what
        "tripped" means), so the claim there is that it decays monotonically
        THROUGH the threshold and never returns: no step is tripped that
        follows an untripped one. See esc-4559-6 — the plan's single
        "never trips from 73.59" form is unsatisfiable, since 0.7 * 73.59 =
        51.513 is still above 24.6.
        """
        # (a) From a healthy start: never trips, for any step count.
        for n_steps in (1, 5, 50, 500):
            series = replay_ewa_series(
                _drain(12, 0, n_steps),
                alpha=ALPHA,
                threshold=THRESHOLD,
                count_resolutions=False,
                prev_ewa=4.0,
            )
            assert series.tripped_at is None, (
                f'A pure-drain window from a healthy EWA must never trip over '
                f'{n_steps} steps; tripped at {series.tripped_at}'
            )

        # (b) From the recorded trip value: strictly decreasing, decays through
        # the threshold, and never crosses back up.
        series = replay_ewa_series(
            _drain(12, 0, 20),
            alpha=ALPHA,
            threshold=THRESHOLD,
            count_resolutions=False,
            prev_ewa=73.59, # the recorded 2026-08-20 trip value
        )
        values = [s.ewa for s in series.steps]
        previous = 73.59
        for value in values:
            assert value < previous, (
                f'Expected a strictly decreasing series; got {values}'
            )
            previous = value
        flags = [s.tripped for s in series.steps]
        assert flags[0] is True, (
            f'Sanity: 0.7 * 73.59 = 51.513 is still above {THRESHOLD}; got {flags}'
        )
        assert flags[-1] is False, f'Expected the series to decay below threshold; got {values}'
        for earlier, later in zip(flags[:-1], flags[1:], strict=True):
            assert not (later and not earlier), (
                f'A strictly decreasing series must never cross the threshold '
                f'upward; got {flags}'
            )

    def test_ordering_inversion_holds_for_the_window_class_not_just_the_models(self) -> None:
        """OLD can rank a drain above a burst; NEW never can.

        The miss direction, proved over a swept family rather than one tuned
        pair — a single fabricated burst/drain pair could be tuned to invert
        either way, so it would evidence nothing.

        The NEW half is a construction: from a healthy start a pure-drain
        window has numerator 0 at every step, so its peak is exactly 0, while
        any window containing at least one submission has a strictly positive
        peak.  So under the new statistic a burst outranks a drain
        UNCONDITIONALLY — no parameterisation can invert it.

        The OLD half holds whenever the drain's per-done resolution rate
        exceeds the burst's per-done (submissions + resolutions) rate, which is
        exactly the shape of the incident: a zero-done cleanup next to a
        high-throughput fault burst whose completions absorb it.  The
        documented 2026 models are included as one illustrative member.
        """
        drains = [
            (f'drain r={r} dones={d}', _drain(r, d, 8))
            for r, d in ((30, 0), (49, 0), (73, 1))
        ]
        drains.append(('MODEL_DRAIN_2026_08_20', MODEL_DRAIN_2026_08_20))

        bursts = [
            ('steady burst', [(30, 5, 12)] * 8),
            ('tapering burst', [(40, 2, 10), (30, 6, 10), (20, 9, 10), (12, 9, 10)]),
            ('MODEL_BURST_2026_08_16', MODEL_BURST_2026_08_16),
        ]

        for drain_label, drain in drains:
            for burst_label, burst in bursts:
                pair = f'{drain_label} vs {burst_label}'

                drain_old = replay_ewa_series(
                    drain, alpha=ALPHA, threshold=THRESHOLD, count_resolutions=True
                )
                burst_old = replay_ewa_series(
                    burst, alpha=ALPHA, threshold=THRESHOLD, count_resolutions=True
                )
                assert drain_old.peak > burst_old.peak, (
                    f'{pair}: expected the OLD statistic to rank the drain '
                    f'({drain_old.peak}) ABOVE the burst ({burst_old.peak}) — '
                    f'that inversion is the defect'
                )

                drain_new = replay_ewa_series(
                    drain, alpha=ALPHA, threshold=THRESHOLD, count_resolutions=False
                )
                burst_new = replay_ewa_series(
                    burst, alpha=ALPHA, threshold=THRESHOLD, count_resolutions=False
                )
                assert drain_new.peak == pytest.approx(0.0), (
                    f'{pair}: a pure-drain window must score exactly 0 under '
                    f'the NEW statistic; got {drain_new.peak}'
                )
                assert burst_new.peak > drain_new.peak, (
                    f'{pair}: expected the NEW statistic to rank the burst '
                    f'({burst_new.peak}) ABOVE the drain ({drain_new.peak}) — '
                    f'the fix restores the ordering'
                )

    def test_new_statistic_is_pointwise_le_old(self) -> None:
        """NEW peak <= OLD peak always, with equality iff the window has no resolutions.

        The monotonicity that makes the ordering assertion above exact rather
        than data-dependent: update_ewa is monotone non-decreasing in
        escalations_in_step, and dropping resolutions makes the numerator
        pointwise <= the old one over the identical window, so by induction the
        whole series is. This is also why the burst can NOT be made to score
        higher than its measured 19.96 by this fix.
        """
        sequences = [
            ('drain-model', MODEL_DRAIN_2026_08_20),
            ('burst-model', MODEL_BURST_2026_08_16),
            ('mixed', [(5, 5, 2), (0, 20, 0), (12, 1, 9), (3, 30, 1)]),
            ('resolution-free', [(7, 0, 3), (11, 0, 0), (2, 0, 8)]),
            ('empty-numerators', [(0, 0, 4), (0, 0, 1)]),
        ]
        for label, steps in sequences:
            old = replay_ewa_series(
                steps, alpha=ALPHA, threshold=THRESHOLD, count_resolutions=True
            )
            new = replay_ewa_series(
                steps, alpha=ALPHA, threshold=THRESHOLD, count_resolutions=False
            )
            assert new.peak <= old.peak + 1e-9, (
                f'{label}: expected NEW peak {new.peak} <= OLD peak {old.peak}'
            )
            has_resolutions = any(r > 0 for _s, r, _d in steps)
            if has_resolutions:
                assert new.peak < old.peak, (
                    f'{label}: a window WITH resolutions must score strictly lower '
                    f'under the new statistic; got new={new.peak} old={old.peak}'
                )
            else:
                assert new.peak == pytest.approx(old.peak), (
                    f'{label}: a window with NO resolutions must be unchanged; '
                    f'got new={new.peak} old={old.peak}'
                )

    def test_replay_delegates_to_update_ewa(self) -> None:
        """Each replayed step equals a direct update_ewa call — no second implementation.

        The replay must never drift from the live statistic, so it delegates
        the per-step arithmetic rather than re-deriving the formula. Its only
        branch is the numerator selection, which is precisely the production
        difference being demonstrated.  This is what keeps the harness safe to
        hold in the test module rather than in digest.py.
        """
        steps = [(4, 3, 2), (0, 9, 0), (11, 2, 5)]
        series = replay_ewa_series(
            steps, alpha=ALPHA, threshold=THRESHOLD, count_resolutions=False
        )
        expected = 0.0
        for (submits, _resolves, dones), replayed in zip(steps, series.steps, strict=True):
            expected = digest.update_ewa(
                prev_ewa=expected,
                escalations_in_step=submits,
                done_in_step=dones,
                alpha=ALPHA,
            )
            assert replayed.ewa == pytest.approx(expected), (
                f'Replay step diverged from update_ewa: {replayed.ewa} != {expected}'
            )
            assert replayed.numerator == submits, (
                f'Expected numerator {submits}; got {replayed.numerator}'
            )

    def test_tripped_at_is_the_first_crossing_index(self) -> None:
        """tripped_at is the index of the FIRST step at or above the threshold."""
        # alpha=1.0 collapses each step to its raw ratio, so the crossing index
        # is exact and independent of smoothing.
        series = replay_ewa_series(
            [(1, 0, 1), (2, 0, 1), (100, 0, 1), (1, 0, 1)],
            alpha=1.0,
            threshold=24.6,
            count_resolutions=False,
        )
        assert series.tripped_at == 2, (
            f'Expected the first crossing at index 2; got {series.tripped_at}'
        )
        assert series.peak == pytest.approx(100.0), (
            f'Expected peak 100.0; got {series.peak}'
        )

    def test_replay_harness_is_not_exported_from_production_digest(self) -> None:
        """digest.py must not carry the replay scaffolding (amendment).

        Guards the over-engineering fix: the helper's only consumer is this
        module, and its ``count_resolutions=True`` arm encodes the RETIRED
        pre-4559 statistic.  If a future change re-adds it to the production
        module without an actual entry point, this fails and forces the
        question to be re-answered rather than drifting back.
        """
        for name in ('replay_ewa_series', 'EwaReplayStep', 'EwaReplaySeries'):
            assert not hasattr(digest, name), (
                f'orchestrator.digest should not export test-only replay '
                f'scaffolding; found {name!r}. Either keep it in '
                f'tests/test_ewa_replay.py, or land a real entry point '
                f'(e.g. under scripts/) and update this test.'
            )
