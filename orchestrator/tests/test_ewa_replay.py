"""Replay harness for the EWA statistic — task 4559, esc-4559-3.

Demonstrates, executably, that excluding escalation RESOLUTIONS from the EWA
numerator fixes the breaker in both directions:

* **False positive** — a pure backlog-drain window (many resolutions, zero new
  submissions) trips the OLD statistic and provably cannot trip the NEW one.
* **Miss** — the ORDERING of the two 2026 windows is inverted under the OLD
  statistic (a routine drain outranks a genuine fault burst) and restored
  under the NEW one.

On the miss direction, note carefully what is and is not asserted.  The task
description's literal signal was "the 2026-08-16/17 burst should trip 24.6, or
at least score far higher than its measured 19.96".  Both disjuncts are
refuted by the fix's own algebra: ``digest.update_ewa`` is monotone
non-decreasing in ``escalations_in_step``, and excluding resolutions makes the
numerator pointwise <= today's over the identical window, so by induction the
NEW EWA is pointwise <= the OLD EWA at every step, with equality iff the window
has zero resolutions.  The burst's OLD peak IS 19.96, so its NEW peak is
<= 19.96 — it can neither reach 24.6 nor exceed 19.96.  That agrees with the
task's own struck claim: the miss is driven by the DENOMINATOR (high done
counts absorbing the burst), which a numerator edit cannot reach.  The
discriminative ORDERING is the strongest true form of the signal, and it is
exactly what the fix restores; the denominator-side detector is filed as
follow-up work.  Recorded as esc-4559-3.

The two window fixtures are DOCUMENTED MODELS, not replays of live data: the
task's own scope caveat records that ``data/escalations/archive/`` only reaches
back to 2026-07-21 because archiving keys on ``resolved_at``, so the full
period is not reconstructible and a live replay would be non-deterministic in
CI regardless.  Every constant below carries a source comment citing the
task's measured aggregates, and the assertions are restricted to those that
hold for any reasonable parameterisation — no tuned numeric bound is
introduced anywhere.
"""

from __future__ import annotations

import pytest

import orchestrator.digest as digest

# ---------------------------------------------------------------------------
# Window models — each constant sourced from task 4559's measurements
# ---------------------------------------------------------------------------

# Production defaults at the time of the incidents (config.py:4066, 4076).
ALPHA = 0.3
THRESHOLD = 24.6

# --- DRAIN_2026_08_20 ------------------------------------------------------
# The false-positive window.  Measured: 196 escalation-lifecycle events, of
# which a SINGLE cluster-L2 resolve with 67 members contributed 49; 84% of the
# resolutions were of escalations FILED BEFORE the window opened (so they are
# cleanup, not new signal); and the window coincided with a landing collapse,
# giving zero done transitions.  Modelled as: zero submissions throughout, the
# resolutions spread over steps with the 49-event cluster resolve as one spike,
# and dones=0 for every step.
#
# Steps are (submissions, resolutions, dones) triples, one per digest write.
# A digest step is NOT exactly digest_every_n_escalations events: the gate is a
# lower bound checked once per supervisor poll (harness.py, _maybe_write_digest
# step 3), so a step carries however many events accumulated since the last
# poll.  The partition below models six polls across the window with the
# cluster resolve landing in one of them.  No assertion depends on the exact
# partition — all are numerator-sign arguments — only on the submissions column
# being zero throughout.
DRAIN_2026_08_20: list[tuple[int, int, int]] = [
    (0, 30, 0),  # routine cleanup of pre-window escalations
    (0, 35, 0),
    (0, 49, 0),  # the single 67-member cluster-L2 resolve: 49 events at once
    (0, 30, 0),
    (0, 28, 0),
    (0, 24, 0),  # 196 lifecycle events total, every one a resolution
]

# --- BURST_2026_08_16 ------------------------------------------------------
# The miss window.  Measured: 297 escalation SUBMISSIONS over ~18h, with work
# landing throughout — it is the high done count that absorbs the burst and
# makes it score only 19.96 under the old statistic.  Modelled as submissions
# spread across the window with a matching non-zero done count per step, and a
# resolution stream alongside (escalations filed early in the burst get
# resolved later in it).
BURST_2026_08_16: list[tuple[int, int, int]] = [
    (30, 2, 2),
    (35, 4, 2),
    (40, 6, 2),
    (38, 8, 2),
    (34, 8, 2),
    (30, 10, 2),
    (28, 10, 2),
    (25, 10, 2),
    (20, 10, 2),
    (17, 10, 2),  # 297 submissions total; peaks at 18.56 under the old statistic,
]                 # close to the task's measured 19.96


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReplayEwaSeries:
    """digest.replay_ewa_series over the two 2026 windows."""

    def test_drain_window_trips_old_statistic_only(self) -> None:
        """The 2026-08-20 drain trips the OLD statistic and not the NEW one.

        This is the false-positive direction, and it is the outage itself: a
        window whose only activity was CLEANING UP a backlog halted fleet-wide
        dispatch, because every resolution counted as if it were a fresh
        escalation.
        """
        old = digest.replay_ewa_series(
            DRAIN_2026_08_20,
            alpha=ALPHA,
            threshold=THRESHOLD,
            count_resolutions=True,
        )
        assert old.tripped_at is not None, (
            f'Expected the old statistic to trip on the drain window '
            f'(that is the outage); peak was {old.peak}'
        )

        new = digest.replay_ewa_series(
            DRAIN_2026_08_20,
            alpha=ALPHA,
            threshold=THRESHOLD,
            count_resolutions=False,
        )
        assert new.tripped_at is None, (
            f'Expected the new statistic NOT to trip on a pure-drain window; '
            f'tripped at step {new.tripped_at} with peak {new.peak}'
        )

    def test_pure_drain_cannot_trip_by_construction(self) -> None:
        """A zero-submission window is strictly decreasing, so it cannot trip.

        Needs no tuned bound and no fixture: with the numerator at zero,
        update_ewa reduces to EWA(t+1) = (1 - alpha) * EWA(t), which for
        0 < alpha <= 1 is strictly decreasing from any positive start.

        Asserted in the two provable forms.  From a HEALTHY start the series
        never trips at all, for any step count — a strictly decreasing series
        cannot cross an upward threshold.  From the recorded 2026-08-20 trip
        value of 73.59 the series necessarily BEGINS above 24.6 (that is what
        "tripped" means), so the claim there is that it decays monotonically
        THROUGH the threshold and never returns: no step is tripped that
        follows an untripped one.  See esc-4559-6 — the plan's single
        "never trips from 73.59" form is unsatisfiable, since 0.7 * 73.59 =
        51.513 is still above 24.6.
        """
        # (a) From a healthy start: never trips, for any step count.
        for n_steps in (1, 5, 50, 500):
            series = digest.replay_ewa_series(
                [(0, 12, 0)] * n_steps,
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
        #     the threshold, and never crosses back up.
        series = digest.replay_ewa_series(
            [(0, 12, 0)] * 20,
            alpha=ALPHA,
            threshold=THRESHOLD,
            count_resolutions=False,
            prev_ewa=73.59,  # the recorded 2026-08-20 trip value
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

    def test_ordering_inversion_is_the_defect_and_the_fix_restores_it(self) -> None:
        """OLD ranks the drain above the burst; NEW ranks the burst above the drain.

        This inversion IS the "fails in both directions" defect made
        executable: under the old statistic a routine backlog cleanup outranks
        a genuine 297-escalation fault burst, so the breaker fires on the wrong
        one.  Excluding resolutions collapses the drain's numerator to zero
        while leaving the burst's intact, which restores the ordering.

        See the module docstring (and esc-4559-3) for why this ordering — not
        an absolute score for the burst — is the only true form of the
        miss-direction signal.
        """
        drain_old = digest.replay_ewa_series(
            DRAIN_2026_08_20, alpha=ALPHA, threshold=THRESHOLD, count_resolutions=True
        )
        burst_old = digest.replay_ewa_series(
            BURST_2026_08_16, alpha=ALPHA, threshold=THRESHOLD, count_resolutions=True
        )
        assert drain_old.peak > burst_old.peak, (
            f'Expected the OLD statistic to rank the drain ({drain_old.peak}) ABOVE '
            f'the burst ({burst_old.peak}) — that inversion is the defect'
        )

        drain_new = digest.replay_ewa_series(
            DRAIN_2026_08_20, alpha=ALPHA, threshold=THRESHOLD, count_resolutions=False
        )
        burst_new = digest.replay_ewa_series(
            BURST_2026_08_16, alpha=ALPHA, threshold=THRESHOLD, count_resolutions=False
        )
        assert burst_new.peak > drain_new.peak, (
            f'Expected the NEW statistic to rank the burst ({burst_new.peak}) ABOVE '
            f'the drain ({drain_new.peak}) — the fix restores the ordering'
        )

    def test_new_statistic_is_pointwise_le_old(self) -> None:
        """NEW peak <= OLD peak always, with equality iff the window has no resolutions.

        The monotonicity that makes the ordering assertion above exact rather
        than data-dependent: update_ewa is monotone non-decreasing in
        escalations_in_step, and dropping resolutions makes the numerator
        pointwise <= the old one over the identical window, so by induction the
        whole series is.  This is also why the burst can NOT be made to score
        higher than its measured 19.96 by this fix.
        """
        sequences: list[tuple[str, list[tuple[int, int, int]]]] = [
            ('drain', DRAIN_2026_08_20),
            ('burst', BURST_2026_08_16),
            ('mixed', [(5, 5, 2), (0, 20, 0), (12, 1, 9), (3, 30, 1)]),
            ('resolution-free', [(7, 0, 3), (11, 0, 0), (2, 0, 8)]),
            ('empty-numerators', [(0, 0, 4), (0, 0, 1)]),
        ]
        for label, steps in sequences:
            old = digest.replay_ewa_series(
                steps, alpha=ALPHA, threshold=THRESHOLD, count_resolutions=True
            )
            new = digest.replay_ewa_series(
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
        the per-step arithmetic rather than re-deriving the formula.  Its only
        branch is the numerator selection, which is precisely the production
        difference being demonstrated.
        """
        steps = [(4, 3, 2), (0, 9, 0), (11, 2, 5)]
        series = digest.replay_ewa_series(
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
        series = digest.replay_ewa_series(
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
