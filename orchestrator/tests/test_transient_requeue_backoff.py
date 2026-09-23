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

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import Scheduler, transient_requeue_cooldown

# ``fused_memory`` is a SIBLING workspace member, not an orchestrator
# dependency, so its importability here is an environment accident rather than
# a guarantee.  The verify lane runs ``cd orchestrator && uv run pytest``,
# which makes rootdir the SUBPROJECT — the repo-root conftest.py that
# sys.path-injects every subproject's src is never loaded (same reasoning as
# the git-isolation note in orchestrator/tests/conftest.py).  That leaves the
# uv workspace's editable-install .pth as the only route in, and .pth files
# are read ONCE, at interpreter startup, by whichever member uv last synced
# for.  A worker that starts before such a sync never sees it: that is how
# ``TestBoundaryRow4ViaGetSchedulerState`` below failed with
# ``ModuleNotFoundError: No module named 'fused_memory'`` across four xdist
# workers in a full-suite run while passing in isolation.
#
# APPEND, never insert(0, ...).  This directory contributes only the
# ``fused_memory`` package, so appending cannot shadow anything the conftest's
# src insertions already resolve.  When the editable install IS present,
# sys.path already holds this exact string and the guard no-ops.
#
# A bare path append is sufficient — no fused-memory third-party dependency
# needs to be installed: ``fused_memory.mcp_tools.scheduler_state`` imports
# stdlib only (json, logging, time, pathlib) and both parent ``__init__``s are
# bare docstrings.  Deliberately NOT ``pytest.importorskip`` (the idiom in
# test_reopen_sticks_e2e.py, which needs fused-memory's real backends): this
# class is the only gate driving boundary row 4 through the genuine product
# read path, and skipping it exactly when the venv is mid-sync would make that
# gate vacuous in the lane it is meant to protect.
_FUSED_MEMORY_SRC = Path(__file__).resolve().parents[2] / 'fused-memory' / 'src'
if _FUSED_MEMORY_SRC.is_dir() and str(_FUSED_MEMORY_SRC) not in sys.path:
    sys.path.append(str(_FUSED_MEMORY_SRC))

# Contract C3 defaults, and the closed-form envelope they produce.  Every
# assertion below is exact arithmetic — no numeric tolerance guesswork.
BASE = 30.0
CAP = 900.0
# n:                       1     2      3      4      5      6
ENVELOPES = [30.0, 60.0, 120.0, 240.0, 480.0, 900.0]


def _requeue(scheduler: Scheduler, task_id: str, **kw) -> int:
    """Call ``record_requeue`` with the boilerplate report fields filled in.

    Defaults to a TRANSIENT requeue (``reason`` carries the 5xx marker);
    pass ``reason='verify failed', api_error_status=None`` for a genuine one.
    """
    return scheduler.record_requeue(
        task_id,
        phase=kw.pop('phase', 'execute'),
        reason=kw.pop('reason', 'agent API error: HTTP 529'),
        detail=kw.pop('detail', ''),
        run_id=kw.pop('run_id', 'run-1'),
        cost_usd=kw.pop('cost_usd', 0.0),
        **kw,
    )


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


class TestPendingTransientCooldownStamp:
    """The consume-once stamp that tells ``release`` THIS requeue was transient.

    PRD open question 1.  ``record_requeue`` already runs strictly before
    ``Scheduler.release`` in ``Harness._run_slot``'s finally block, so the
    cumulative ``_transient_requeue_counts`` value is post-increment by the
    time arming happens — but the count alone cannot say whether *this*
    requeue was transient.  A task with prior transients that then requeues
    GENUINELY would read a non-zero count and wrongly get a backoff,
    violating boundary row 4's "genuine requeue stays flat 30s".
    ``_pending_transient_cooldown`` closes that gap: written only on the
    transient route, popped unconditionally by ``release``.
    """

    def _scheduler(self) -> Scheduler:
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        scheduler.finish_startup()
        return scheduler

    def test_fresh_scheduler_has_no_stamps(self):
        assert self._scheduler()._pending_transient_cooldown == {}

    def test_transient_requeue_stamps_the_post_increment_count(self):
        """The stamp equals ``transient_requeue_count`` AFTER the call.

        Pinned together so the two can never drift — the stamp IS the ``n``
        the backoff envelope is computed from.
        """
        scheduler = self._scheduler()
        _requeue(scheduler, 'T1', api_error_status=529)
        assert scheduler._pending_transient_cooldown['T1'] == 1
        assert scheduler.transient_requeue_count('T1') == 1

        _requeue(scheduler, 'T1', api_error_status=529)
        assert scheduler._pending_transient_cooldown['T1'] == 2
        assert scheduler.transient_requeue_count('T1') == 2

    def test_legacy_marker_route_also_stamps(self):
        """The stamp is keyed off the SAME is_transient_api_requeue decision.

        A reason-marker-only transient (no structured status yet — the phases
        that produce the field land in sibling PRD tasks) must stamp too, or
        the stamp would amount to a second, divergent classifier.
        """
        scheduler = self._scheduler()
        _requeue(
            scheduler, 'T1',
            reason='agent API error: HTTP 503', api_error_status=None,
        )
        assert scheduler._pending_transient_cooldown['T1'] == 1

    def test_genuine_requeue_leaves_no_stamp(self):
        scheduler = self._scheduler()
        _requeue(scheduler, 'G1', reason='verify failed', api_error_status=None)
        assert 'G1' not in scheduler._pending_transient_cooldown

    def test_non_counting_requeue_leaves_no_stamp(self):
        """Route 1 (counts_against_cap=False) precedence is preserved.

        A history-only requeue moves NEITHER counter, so it must not arm a
        backoff either — even when it also looks transient.
        """
        scheduler = self._scheduler()
        _requeue(
            scheduler, 'N1', api_error_status=529, counts_against_cap=False,
        )
        assert 'N1' not in scheduler._pending_transient_cooldown
        assert scheduler.transient_requeue_count('N1') == 0

    def test_release_requeued_true_consumes_the_stamp(self):
        scheduler = self._scheduler()
        _requeue(scheduler, 'T1', api_error_status=529)
        scheduler.release('T1', requeued=True)
        assert 'T1' not in scheduler._pending_transient_cooldown

    def test_release_requeued_false_also_consumes_the_stamp(self):
        """The cap-exhaust shape (``requeued=False``) must not leave residue.

        ``_apply_retry_cap`` records the requeue and THEN discovers the cap is
        exhausted, releasing with ``requeued=False``.  A stamp left behind
        would leak into whatever armed next.
        """
        scheduler = self._scheduler()
        _requeue(scheduler, 'T1', api_error_status=529)
        scheduler.release('T1', requeued=False)
        assert 'T1' not in scheduler._pending_transient_cooldown

    def test_clear_requeue_count_drops_the_stamp(self):
        """DONE / cap-exhaust reset clears the stamp alongside the counters."""
        scheduler = self._scheduler()
        _requeue(scheduler, 'T1', api_error_status=529)
        scheduler.clear_requeue_count('T1')
        assert 'T1' not in scheduler._pending_transient_cooldown
        assert scheduler.transient_requeue_count('T1') == 0

    def test_stamp_cannot_leak_into_a_later_genuine_arming(self):
        """Leak regression — the whole reason the stamp is consume-once.

        transient -> arm -> GENUINE -> arm.  At the second arming no stamp may
        be present, or the genuine requeue would inherit a backoff.
        """
        scheduler = self._scheduler()
        _requeue(scheduler, 'T1', api_error_status=529)
        scheduler.release('T1', requeued=True)

        _requeue(scheduler, 'T1', reason='verify failed', api_error_status=None)
        assert 'T1' not in scheduler._pending_transient_cooldown, (
            'a consumed stamp must not reappear for a genuine requeue'
        )

    def test_stamps_are_per_task(self):
        scheduler = self._scheduler()
        _requeue(scheduler, 'T1', api_error_status=529)
        _requeue(scheduler, 'T2', api_error_status=500)
        _requeue(scheduler, 'T2', api_error_status=500)
        assert scheduler._pending_transient_cooldown == {'T1': 1, 'T2': 2}
        scheduler.release('T1', requeued=True)
        assert scheduler._pending_transient_cooldown == {'T2': 2}


def _clocked_scheduler(
    clock: list[float],
    *,
    jitter_source=None,
    config: OrchestratorConfig | None = None,
) -> Scheduler:
    """A bare Scheduler on an injected monotonic clock (test_scheduler.py idiom)."""
    scheduler = Scheduler(
        config if config is not None else OrchestratorConfig(max_per_module=1),
        time_source=lambda: clock[0],
        jitter_source=jitter_source,
    )
    scheduler.finish_startup()
    return scheduler


def _arm_transient(scheduler: Scheduler, clock: list[float], task_id: str) -> float:
    """Drive one transient requeue+release and return the armed cooldown delta."""
    _requeue(scheduler, task_id, api_error_status=529)
    scheduler.release(task_id, requeued=True)
    return scheduler._requeue_until[task_id] - clock[0]


class TestReleaseArmsJitteredBackoff:
    """``release`` arms a growing cooldown for transient requeues only.

    Boundary row 4: armed cooldowns grow ~30 → 480s over the first five
    transient requeues (jittered, inside the envelope ``[d/2, d]``), while a
    genuine requeue stays flat at 30s.
    """

    def test_upper_edge_growth_is_exact(self):
        """With the max draw the deltas are exactly the envelopes."""
        clock = [1000.0]
        scheduler = _clocked_scheduler(clock, jitter_source=lambda lo, hi: hi)
        armed = [_arm_transient(scheduler, clock, 'T1') for _ in range(5)]
        assert armed == [30.0, 60.0, 120.0, 240.0, 480.0]

    def test_lower_edge_growth_is_exact(self):
        """With the min draw every delta is exactly envelope/2 — the floor."""
        clock = [1000.0]
        scheduler = _clocked_scheduler(clock, jitter_source=lambda lo, hi: lo)
        armed = [_arm_transient(scheduler, clock, 'T1') for _ in range(5)]
        assert armed == [15.0, 30.0, 60.0, 120.0, 240.0]

    def test_real_rng_stays_inside_the_envelope_and_is_nondecreasing(self):
        """No injected jitter: every arming lands in [envelope/2, envelope].

        The equal-jitter floor makes the schedule monotone-nondecreasing no
        matter how the draws land — a plain ``U(0, envelope)`` would not.
        """
        clock = [1000.0]
        scheduler = _clocked_scheduler(clock)
        armed = [_arm_transient(scheduler, clock, 'T1') for _ in range(5)]
        for i, delta in enumerate(armed):
            envelope = ENVELOPES[i]
            assert envelope / 2 <= delta <= envelope, (
                f'n={i + 1}: {delta} outside [{envelope / 2}, {envelope}]'
            )
        assert armed == sorted(armed), f'schedule must not shrink: {armed}'

    @pytest.mark.parametrize('n', [6, 7])
    def test_cap_clamps_the_arming(self, n):
        clock = [1000.0]
        scheduler = _clocked_scheduler(clock, jitter_source=lambda lo, hi: hi)
        armed = [_arm_transient(scheduler, clock, 'T1') for _ in range(n)]
        assert armed[-1] == 900.0

    def test_genuine_requeue_stays_flat_despite_prior_transients(self):
        """The cumulative-counter trap the stamp exists to close.

        Three prior transient requeues leave ``transient_requeue_count == 3``,
        but this requeue is GENUINE, so it must arm the flat 30s — not the
        240s an n=4 envelope would give.
        """
        clock = [1000.0]
        config = OrchestratorConfig(max_per_module=1)
        scheduler = _clocked_scheduler(
            clock, jitter_source=lambda lo, hi: hi, config=config,
        )
        for _ in range(3):
            _arm_transient(scheduler, clock, 'T1')
        assert scheduler.transient_requeue_count('T1') == 3

        _requeue(scheduler, 'T1', reason='verify failed', api_error_status=None)
        scheduler.release('T1', requeued=True)
        delta = scheduler._requeue_until['T1'] - clock[0]
        assert delta == config.requeue_cooldown_secs == 30.0

    def test_release_without_any_record_requeue_stays_flat(self):
        """The blast-radius and arm_requeue_cooldown shapes arm flat 30s.

        Neither is preceded by a ``record_requeue``, so neither leaves a
        stamp — the no-stamp path must fall back to ``requeue_cooldown_secs``.
        """
        clock = [1000.0]
        config = OrchestratorConfig(max_per_module=1)
        scheduler = _clocked_scheduler(
            clock, jitter_source=lambda lo, hi: hi, config=config,
        )
        scheduler.release('T2', requeued=True)
        assert scheduler._requeue_until['T2'] - clock[0] == 30.0

    def test_green_tier_retune_lands_on_the_next_arming(self):
        """An in-place config mutation (what apply_reload does) takes effect.

        ``release`` reads ``self.config.<knob>`` at ARM time, so no reload
        hook is needed: base 5 / cap 20 walks 5, 10, 20, 20.
        """
        clock = [1000.0]
        scheduler = _clocked_scheduler(clock, jitter_source=lambda lo, hi: hi)
        scheduler.config.transient_requeue_backoff_base_secs = 5.0
        scheduler.config.transient_requeue_backoff_cap_secs = 20.0
        armed = [_arm_transient(scheduler, clock, 'T1') for _ in range(4)]
        assert armed == [5.0, 10.0, 20.0, 20.0]

    def test_dispatch_eligibility_honours_the_longer_deadline(self):
        """The REAL ``_eligible_for_dispatch`` gate refuses for the full 120s.

        This is the product-visible point of the whole task: the longer
        cooldown must actually keep the task off the dispatch path.  So it
        drives the gate itself rather than re-reading ``_requeue_until`` —
        comparing the stored deadline against the clock that just set it is a
        tautology that would still pass if the gate at scheduler.py:5059 were
        deleted or inverted.

        n=3 at the upper edge arms 120s: still refused at +60s, where the old
        flat 30s would already have let the task back in; eligible again once
        the deadline passes.
        """
        clock = [1000.0]
        scheduler = _clocked_scheduler(clock, jitter_source=lambda lo, hi: hi)
        for _ in range(3):
            armed = _arm_transient(scheduler, clock, 'T1')
        assert armed == 120.0

        task = {'id': 'T1', 'status': 'pending', 'dependencies': []}
        status_map = {'T1': 'pending'}

        # Sanity: the gate is otherwise satisfied, so a False below can only
        # be the cooldown.  Proven by clearing the deadline and re-asking.
        without_cooldown = dict(scheduler._requeue_until)
        scheduler._requeue_until.clear()
        assert scheduler._eligible_for_dispatch(task, 'T1', status_map)[0] is True
        scheduler._requeue_until = without_cooldown

        clock[0] += 30.0
        assert scheduler._eligible_for_dispatch(task, 'T1', status_map)[0] is False, (
            'the old flat 30s cooldown would have expired here'
        )
        clock[0] += 30.0
        assert scheduler._eligible_for_dispatch(task, 'T1', status_map)[0] is False, (
            'still inside the 120s transient backoff at +60s'
        )

        clock[0] += 61.0
        assert scheduler._eligible_for_dispatch(task, 'T1', status_map)[0] is True, (
            'eligible again once the 120s deadline passes'
        )
        # The per-tick sweep then drops both the deadline and its meta.
        scheduler._gc_expired_cooldowns()
        assert 'T1' not in scheduler._requeue_until


class TestSnapshotExposesRequeueCooldowns:
    """``get_state_snapshot()['requeue_cooldowns']`` — the operator-visible read.

    Sourced from a meta dict stamped ONCE AT ARMING, never recomputed from
    "now": ``_build_snapshot_payload`` content-dedups the snapshot, so a
    naive ``remaining_secs`` would make every tick byte-different and defeat
    the write throttle.
    """

    def test_transient_arming_records_armed_envelope_and_count(self):
        clock = [1000.0]
        scheduler = _clocked_scheduler(clock, jitter_source=lambda lo, hi: hi)
        _arm_transient(scheduler, clock, 'T1')
        entry = scheduler.get_state_snapshot()['requeue_cooldowns']['T1']
        assert entry['armed_secs'] == 30.0
        assert entry['envelope_secs'] == 30.0
        assert entry['transient'] is True
        assert entry['transient_count'] == 1

        for _ in range(2):
            _arm_transient(scheduler, clock, 'T1')
        entry = scheduler.get_state_snapshot()['requeue_cooldowns']['T1']
        assert entry['armed_secs'] == entry['envelope_secs'] == 120.0
        assert entry['transient_count'] == 3

    def test_real_rng_armed_value_sits_inside_the_recorded_envelope(self):
        clock = [1000.0]
        scheduler = _clocked_scheduler(clock)
        for _ in range(4):
            _arm_transient(scheduler, clock, 'T1')
        entry = scheduler.get_state_snapshot()['requeue_cooldowns']['T1']
        assert entry['envelope_secs'] == 240.0
        assert entry['envelope_secs'] / 2 <= entry['armed_secs'] <= entry['envelope_secs']

    def test_genuine_requeue_records_a_flat_non_transient_entry(self):
        clock = [1000.0]
        scheduler = _clocked_scheduler(clock, jitter_source=lambda lo, hi: hi)
        _requeue(scheduler, 'G1', reason='verify failed', api_error_status=None)
        scheduler.release('G1', requeued=True)
        entry = scheduler.get_state_snapshot()['requeue_cooldowns']['G1']
        assert entry['armed_secs'] == 30.0
        assert entry['envelope_secs'] == 30.0
        assert entry['transient'] is False
        assert entry['transient_count'] == 0

    def test_armed_secs_grow_across_successive_transient_requeues(self):
        """Boundary row 4's product-visible signal: the deltas GROW."""
        clock = [1000.0]
        scheduler = _clocked_scheduler(clock, jitter_source=lambda lo, hi: hi)
        seen = []
        for _ in range(5):
            _arm_transient(scheduler, clock, 'T1')
            seen.append(
                scheduler.get_state_snapshot()['requeue_cooldowns']['T1']['armed_secs']
            )
        assert seen == [30.0, 60.0, 120.0, 240.0, 480.0]

    def test_gc_drops_the_meta_in_lockstep_with_the_deadline(self):
        """An entry must never outlive its deadline."""
        clock = [1000.0]
        scheduler = _clocked_scheduler(clock, jitter_source=lambda lo, hi: hi)
        _arm_transient(scheduler, clock, 'T1')
        assert 'T1' in scheduler.get_state_snapshot()['requeue_cooldowns']

        clock[0] += 31.0
        scheduler._gc_expired_cooldowns()
        assert 'T1' not in scheduler._requeue_until
        assert 'T1' not in scheduler.get_state_snapshot()['requeue_cooldowns']

    def test_gc_tolerates_a_directly_injected_deadline_with_no_meta(self):
        """Tests inject ``_requeue_until`` directly (test_scheduler_tick_phases.py).

        The meta pop must be a no-op when no entry exists, not a KeyError.
        """
        clock = [1000.0]
        scheduler = _clocked_scheduler(clock)
        scheduler._requeue_until['ORPHAN'] = 500.0
        scheduler._gc_expired_cooldowns()
        assert 'ORPHAN' not in scheduler._requeue_until

    def test_payload_is_dedup_stable_as_the_clock_advances(self):
        """Regression guard for the content-dedup write throttle.

        No cooldown field may be recomputed from "now" — with one cooldown
        armed and no re-arming, two payloads five seconds apart must be
        byte-identical, or every tick would rewrite the snapshot file.
        """
        clock = [1000.0]
        scheduler = _clocked_scheduler(clock, jitter_source=lambda lo, hi: hi)
        _arm_transient(scheduler, clock, 'T1')
        first = scheduler._build_snapshot_payload()
        clock[0] += 5.0
        assert scheduler._build_snapshot_payload() == first

    def test_returned_cooldowns_are_deep_copied(self):
        """Mutating the returned dict must not corrupt scheduler state."""
        clock = [1000.0]
        scheduler = _clocked_scheduler(clock, jitter_source=lambda lo, hi: hi)
        _arm_transient(scheduler, clock, 'T1')
        snap = scheduler.get_state_snapshot()
        snap['requeue_cooldowns']['T1']['armed_secs'] = 9999.0
        snap['requeue_cooldowns']['INJECTED'] = {}
        fresh = scheduler.get_state_snapshot()['requeue_cooldowns']
        assert fresh['T1']['armed_secs'] == 30.0
        assert 'INJECTED' not in fresh


class TestBoundaryRow4ViaGetSchedulerState:
    """Boundary row 4, read through the REAL product path.

    ``scheduler._write_state_snapshot_raw`` -> the on-disk
    ``<root>/data/orchestrator/scheduler_state.json`` ->
    ``fused_memory.mcp_tools.scheduler_state.read_scheduler_state``, which is
    the exact function the ``get_scheduler_state`` MCP tool delegates to.
    """

    def _drive(self, tmp_path):
        """Arm n=1..5 transient cooldowns, snapshotting through disk each time."""
        from fused_memory.mcp_tools.scheduler_state import read_scheduler_state

        clock = [1000.0]
        wall = datetime(2026, 7, 29, 12, 0, 0, tzinfo=UTC)
        scheduler = Scheduler(
            OrchestratorConfig(max_per_module=1),
            time_source=lambda: clock[0],
            wall_time_source=lambda: wall,
        )
        scheduler.finish_startup()
        snapshot_path = tmp_path / 'data' / 'orchestrator' / 'scheduler_state.json'

        states = []
        for _ in range(5):
            _arm_transient(scheduler, clock, 'T1')
            scheduler._write_state_snapshot_raw(snapshot_path)
            states.append(read_scheduler_state(tmp_path))
        return scheduler, clock, states, snapshot_path

    def test_envelopes_grow_thirty_to_four_eighty_through_the_product_path(self, tmp_path):
        """Row 4: armed cooldowns grow ~30 -> 480s, jittered inside [d/2, d]."""
        _scheduler, _clock, states, _path = self._drive(tmp_path)
        entries = [s['requeue_cooldowns']['T1'] for s in states]
        assert [e['envelope_secs'] for e in entries] == [30.0, 60.0, 120.0, 240.0, 480.0]
        for entry in entries:
            envelope = entry['envelope_secs']
            assert envelope / 2 <= entry['armed_secs'] <= envelope
            assert entry['transient'] is True

    def test_genuine_requeue_stays_flat_alongside_in_the_same_snapshot(self, tmp_path):
        """Row 4's other half — and the two tasks' entries do not interfere."""
        from fused_memory.mcp_tools.scheduler_state import read_scheduler_state

        scheduler, _clock, _states, path = self._drive(tmp_path)
        _requeue(scheduler, 'G1', reason='verify failed', api_error_status=None)
        scheduler.release('G1', requeued=True)
        scheduler._write_state_snapshot_raw(path)
        cooldowns = read_scheduler_state(tmp_path)['requeue_cooldowns']

        assert cooldowns['G1']['armed_secs'] == 30.0
        assert cooldowns['G1']['transient'] is False
        assert cooldowns['T1']['envelope_secs'] == 480.0
        assert cooldowns['T1']['transient'] is True

    def test_wall_clock_fields_are_parseable_and_consistent(self, tmp_path):
        """``armed_at``/``expires_at`` answer "when does this task come back".

        A monotonic deadline cannot: it has no epoch relation and does not
        survive a restart.  ``expires_at - armed_at`` must equal
        ``armed_secs`` (to millisecond tolerance — ``armed_secs`` is rounded
        to 3 dp), which requires deriving both from ONE wall read.
        """
        _scheduler, _clock, states, _path = self._drive(tmp_path)
        for state in states:
            entry = state['requeue_cooldowns']['T1']
            armed_at = datetime.fromisoformat(entry['armed_at'])
            expires_at = datetime.fromisoformat(entry['expires_at'])
            delta = (expires_at - armed_at).total_seconds()
            assert abs(delta - entry['armed_secs']) < 0.001

    def test_snapshot_is_strictly_json_native(self, tmp_path):
        """No datetime object may leak into the snapshot.

        The production writer serialises with ``json.dumps(state, default=str)``,
        which would silently stringify a datetime into a shape
        ``read_scheduler_state`` cannot round-trip into a comparable value.
        Serialising with NO ``default=`` fallback is what proves it.
        """
        scheduler, _clock, _states, _path = self._drive(tmp_path)
        json.dumps(scheduler.get_state_snapshot())  # no default= — must not raise
        for entry in scheduler.get_state_snapshot()['requeue_cooldowns'].values():
            for key, value in entry.items():
                assert isinstance(value, (str, int, float, bool)), (
                    f'{key}={value!r} ({type(value).__name__}) is not JSON-native'
                )

    def test_dedup_stability_survives_the_wall_clock_fields(self, tmp_path):
        """The new fields are arming-time constants, so the throttle still holds."""
        scheduler, clock, _states, _path = self._drive(tmp_path)
        first = scheduler._build_snapshot_payload()
        clock[0] += 5.0
        assert scheduler._build_snapshot_payload() == first
