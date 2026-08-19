"""Mechanism 2 (task 2408): Scheduler._phase_redispatch_stranded_blocked.

A periodic tick phase that flips genuinely-stranded BLOCKED tasks (no live
claimant, deps resolved, not deliberately parked) back to ``pending`` — the
safety net complementing mechanism 1's dispatch-time refusal (see
``TestClaimantLivenessDispatchGate`` in test_scheduler.py).

These construct a ``Scheduler`` + hand-build a ``TickContext`` and call
``_phase_redispatch_stranded_blocked`` directly, mirroring the isolation-test
pattern in test_scheduler_tick_phases.py (no full-tick ``acquire_next``
orchestration is driven here).

This module covers the CORE gates (step-05/06): claimant-liveness,
deps-satisfied, and the no-open-escalation park-protection guard, plus the
config kill-switch and the escalation-queue fail-safe. The additional
park-protection carve-outs (deterministic task_kind boundary, cooldown,
workflow_cancel_recent) are covered separately (step-07/08).
"""

import asyncio
import contextlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.models import Escalation

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.harness import Harness
from orchestrator.scheduler import _CONTINUE, Scheduler, TickContext

FIXED_DT = datetime(2026, 7, 10, 12, 0, 0, tzinfo=UTC)


class _FakeEscalationQueue:
    """Minimal ``get_by_task`` stand-in — no filesystem, no archive scan.

    ``by_task`` maps task_id -> the list ``get_by_task`` should return for
    that id (defaults to ``[]``, i.e. no open escalation).
    """

    def __init__(self, by_task: dict[str, list] | None = None):
        self._by_task = by_task or {}
        self.calls: list[tuple[str, str | None]] = []

    def get_by_task(self, task_id: str, status: str | None = None) -> list:
        self.calls.append((task_id, status))
        return self._by_task.get(task_id, [])


def _make_scheduler(*, time_source=None, **config_overrides) -> Scheduler:
    config = OrchestratorConfig(max_per_module=1, **config_overrides)
    kwargs: dict = {'wall_time_source': lambda: FIXED_DT}
    if time_source is not None:
        kwargs['time_source'] = time_source
    scheduler = Scheduler(config, **kwargs)
    scheduler.set_task_status = AsyncMock()  # type: ignore[method-assign]
    scheduler.set_task_claimant = AsyncMock()  # type: ignore[method-assign]
    return scheduler


def _deterministic_blocked_task(task_id: str, *, dep_id: str | None = '9') -> dict:
    """A blocked deterministic born-at-L2 gate task (mirrors the
    tasks-2407/2273 shape: always_escalates=True, before_done_ran_at set,
    null claimant) — the DESIGN-GAP park-protection boundary."""
    return {
        'id': task_id,
        'status': 'blocked',
        'dependencies': [dep_id] if dep_id is not None else [],
        'metadata': {
            'task_kind': 'deterministic',
            'always_escalates': True,
            'before_done_ran_at': '2026-07-01T00:00:00+00:00',
        },
        'claimant_run_id': None,
        'heartbeat_at': None,
    }


def _blocked_task(
    task_id: str,
    *,
    dep_id: str | None = '9',
    claimant_run_id: str | None = None,
    heartbeat_at: str | None = None,
    metadata: dict | None = None,
) -> dict:
    return {
        'id': task_id,
        'status': 'blocked',
        'dependencies': [dep_id] if dep_id is not None else [],
        'metadata': metadata if metadata is not None else {},
        'claimant_run_id': claimant_run_id,
        'heartbeat_at': heartbeat_at,
    }


def _ctx_with_done_dep(task: dict, *, dep_status: str = 'done') -> TickContext:
    dep = {'id': '9', 'status': dep_status}
    return TickContext(
        tasks=[dep, task],
        status_map={'9': dep_status},
        tasks_by_id={'9': dep, str(task['id']): task},
    )


class TestRedispatchStrandedBlockedCore:
    """CORE gates: claimant-liveness + deps-satisfied + no-open-escalation."""

    @pytest.mark.asyncio
    async def test_crash_stranded_blocked_task_flips_to_pending(self):
        """(a) No claimant, no heartbeat, dep done, no open escalation ->
        genuinely stranded -> flipped to pending exactly once."""
        scheduler = _make_scheduler()
        scheduler.escalation_queue = _FakeEscalationQueue()  # type: ignore[assignment]

        task = _blocked_task('T1')
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            'T1', 'pending'
        )
        # The stale (already-null) claimant is explicitly cleared alongside
        # the flip, mirroring the slot-release NULL-claimant convention
        # (harness.py:5693-5696) so a redispatched task never carries a
        # claimant that no longer owns it.
        scheduler.set_task_claimant.assert_awaited_once_with(  # type: ignore[attr-defined]
            'T1', claimant_run_id=None, heartbeat_at=None,
        )

    @pytest.mark.asyncio
    async def test_stranded_blocked_task_with_unresolved_external_dep_still_flips(self):
        """External deps are intentionally NOT checked by this sweep's
        ``_deps_satisfied`` call (only LOCAL deps gate the flip — see the
        docstring): ``ctx.external_cache`` is still empty when this phase
        runs (it runs before ``_phase_external_dep_policy``), so forwarding
        it would make external deps look permanently unsatisfied. The task
        still flips to pending; the REAL external-dep status is re-checked
        by ``_phase_external_dep_policy`` + ``_eligible_for_dispatch`` before
        it is ever actually dispatched."""
        scheduler = _make_scheduler()
        scheduler.escalation_queue = _FakeEscalationQueue()  # type: ignore[assignment]

        task = _blocked_task('T1', metadata={'external_deps': ['other_project:99']})
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            'T1', 'pending'
        )

    @pytest.mark.asyncio
    async def test_open_escalation_blocks_flip(self):
        """(b) Deliberately-parked /unblock: same null-claimant signature,
        but an open escalation exists -> must NOT be flipped."""
        scheduler = _make_scheduler()
        scheduler.escalation_queue = _FakeEscalationQueue(  # type: ignore[assignment]
            by_task={'T1': [{'id': 'esc-1'}]}
        )

        task = _blocked_task('T1')
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        # No flip -> the claimant clear (which is coupled to the flip) must
        # not fire either.
        scheduler.set_task_claimant.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_deps_not_satisfied_blocks_flip(self):
        """(c) Dependency still in-progress -> not genuinely stranded (still
        legitimately waiting) -> must NOT be flipped."""
        scheduler = _make_scheduler()
        scheduler.escalation_queue = _FakeEscalationQueue()  # type: ignore[assignment]

        task = _blocked_task('T1')
        ctx = _ctx_with_done_dep(task, dep_status='in-progress')

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        # No flip -> the claimant clear (which is coupled to the flip) must
        # not fire either.
        scheduler.set_task_claimant.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_live_claimant_blocks_flip(self):
        """(d) Blocked but LIVE (claimant + fresh heartbeat, ttl=300s default)
        -> is_stranded_blocked is False -> must NOT be flipped."""
        scheduler = _make_scheduler()
        scheduler.escalation_queue = _FakeEscalationQueue()  # type: ignore[assignment]

        fresh_heartbeat = (FIXED_DT - timedelta(minutes=1)).isoformat()
        task = _blocked_task(
            'T1',
            claimant_run_id='run-x/session-y/pid=1',
            heartbeat_at=fresh_heartbeat,
        )
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        # No flip -> the claimant clear (which is coupled to the flip) must
        # not fire either.
        scheduler.set_task_claimant.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_no_escalation_queue_fails_safe(self):
        """(e) scheduler.escalation_queue is None (not wired, e.g. pre-harness
        wiring or a test that never sets it) -> fail-safe -> never flip,
        since the no-open-escalation guard cannot be verified."""
        scheduler = _make_scheduler()
        assert scheduler.escalation_queue is None

        task = _blocked_task('T1')
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        # No flip -> the claimant clear (which is coupled to the flip) must
        # not fire either.
        scheduler.set_task_claimant.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_kill_switch_disables_sweep(self):
        """(f) config.stranded_blocked_redispatch_enabled=False -> sweep is
        entirely dormant, even for an otherwise-genuine crash-strand."""
        scheduler = _make_scheduler(stranded_blocked_redispatch_enabled=False)
        scheduler.escalation_queue = _FakeEscalationQueue()  # type: ignore[assignment]

        task = _blocked_task('T1')
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        # No flip -> the claimant clear (which is coupled to the flip) must
        # not fire either.
        scheduler.set_task_claimant.assert_not_awaited()  # type: ignore[attr-defined]


class TestRedispatchStrandedBlockedParkProtection:
    """Park-protection carve-outs (task 2408 step-07/08): a crash-strand and
    a deliberate park present IDENTICALLY at the claimant layer (both
    null/stale), so mechanism 2 needs guards beyond claimant-liveness alone.
    """

    # -- (a) DESIGN-GAP deterministic boundary --------------------------------

    @pytest.mark.asyncio
    async def test_deterministic_gate_task_not_flipped_with_empty_queue(self):
        """A deterministic born-at-L2 gate task (tasks 2407/2273 shape) is
        owned exclusively by the deterministic gate flow / human
        resolve_issue — redispatching it to pending would race/duplicate
        that flow. Unconditional exclusion, even with an empty escalation
        queue (which would otherwise pass the generic no-open-escalation
        gate)."""
        scheduler = _make_scheduler()
        scheduler.escalation_queue = _FakeEscalationQueue()  # type: ignore[assignment]

        task = _deterministic_blocked_task('T1')
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        # No flip -> the claimant clear (which is coupled to the flip) must
        # not fire either.
        scheduler.set_task_claimant.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_deterministic_gate_task_not_flipped_with_open_l2(self):
        """Companion variant: the exclusion holds regardless of escalation
        level — even WITH an open L2 escalation present, the deterministic
        carve-out (not the generic open-escalation gate) is what protects
        it."""
        scheduler = _make_scheduler()
        scheduler.escalation_queue = _FakeEscalationQueue(  # type: ignore[assignment]
            by_task={'T1': [{'id': 'esc-l2', 'level': 2}]}
        )

        task = _deterministic_blocked_task('T1')
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        # No flip -> the claimant clear (which is coupled to the flip) must
        # not fire either.
        scheduler.set_task_claimant.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_non_deterministic_control_still_flips(self):
        """Control: an otherwise-identical NON-deterministic blocked task
        (same null-claimant/done-dep/empty-queue shape) IS flipped — guards
        against the deterministic exclusion being too broad."""
        scheduler = _make_scheduler()
        scheduler.escalation_queue = _FakeEscalationQueue()  # type: ignore[assignment]

        task = _blocked_task('T1')
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            'T1', 'pending'
        )
        scheduler.set_task_claimant.assert_awaited_once_with(  # type: ignore[attr-defined]
            'T1', claimant_run_id=None, heartbeat_at=None,
        )

    # -- (b) actively dispatched ----------------------------------------------

    @pytest.mark.asyncio
    async def test_actively_dispatched_task_not_flipped(self):
        scheduler = _make_scheduler()
        scheduler.escalation_queue = _FakeEscalationQueue()  # type: ignore[assignment]
        scheduler._dispatched.add('T1')

        task = _blocked_task('T1')
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        # No flip -> the claimant clear (which is coupled to the flip) must
        # not fire either.
        scheduler.set_task_claimant.assert_not_awaited()  # type: ignore[attr-defined]

    # -- (c) within dispatch/requeue cooldown window --------------------------

    @pytest.mark.asyncio
    async def test_within_cooldown_window_not_flipped(self):
        fixed_monotonic = 1_000.0
        scheduler = _make_scheduler(time_source=lambda: fixed_monotonic)
        scheduler.escalation_queue = _FakeEscalationQueue()  # type: ignore[assignment]
        scheduler._requeue_until['T1'] = fixed_monotonic + 100.0

        task = _blocked_task('T1')
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        # No flip -> the claimant clear (which is coupled to the flip) must
        # not fire either.
        scheduler.set_task_claimant.assert_not_awaited()  # type: ignore[attr-defined]

    # -- (d) workflow just cancelled/parked (finally-block grace window) -----

    @pytest.mark.asyncio
    async def test_workflow_cancel_recent_not_flipped(self):
        scheduler = _make_scheduler()
        scheduler.escalation_queue = _FakeEscalationQueue()  # type: ignore[assignment]
        scheduler.note_workflow_cancelled('T1')
        assert scheduler.workflow_cancel_recent('T1') is True

        task = _blocked_task('T1')
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        # No flip -> the claimant clear (which is coupled to the flip) must
        # not fire either.
        scheduler.set_task_claimant.assert_not_awaited()  # type: ignore[attr-defined]


class TestRedispatchStrandedBlockedExceptionIsolation:
    """Task 2849: per-task exception isolation in the sweep loop.

    The un-guarded ``for task in ctx.tasks:`` loop previously let ONE
    task's write/predicate raise propagate out of the whole phase,
    collateral-stranding every genuinely-stranded sibling that sorts AFTER
    the raising task in ``ctx.tasks`` for that tick (and every tick after,
    if the failure recurs). These tests pin the fix: a per-task failure
    must be isolated (logged loudly, then skipped) so it can never abort
    the batch.
    """

    @pytest.mark.asyncio
    async def test_earlier_task_write_raises_does_not_strand_later_sibling(
        self, caplog,
    ):
        """An earlier-iterated sibling's ``set_task_status`` raising must not
        prevent a later-iterated, genuinely-stranded sibling from still
        being flipped to pending — and the failure must be logged loudly
        (WARNING naming the failing task), not swallowed silently."""
        import logging

        scheduler = _make_scheduler()
        scheduler.escalation_queue = _FakeEscalationQueue()  # type: ignore[assignment]

        dep_done = {'id': '9', 'status': 'done'}
        t_fail = _blocked_task('5234')
        t_ok = _blocked_task('5236')
        ctx = TickContext(
            tasks=[dep_done, t_fail, t_ok],
            status_map={'9': 'done'},
            tasks_by_id={'9': dep_done, '5234': t_fail, '5236': t_ok},
        )

        async def _raise_for_5234(task_id, status):
            if task_id == '5234':
                raise RuntimeError('transient backend failure')
            return None

        scheduler.set_task_status = AsyncMock(  # type: ignore[method-assign]
            side_effect=_raise_for_5234
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduler'):
            result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_any_await(  # type: ignore[attr-defined]
            '5236', 'pending'
        )
        warning_text = ' '.join(
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        )
        assert '5234' in warning_text, (
            f'Expected a WARNING naming the failing task 5234; got: {warning_text!r}'
        )

    @pytest.mark.asyncio
    async def test_reify_shape_self_failing_task_does_not_collateral_strand_later_sibling(
        self,
    ):
        """Faithful reify-incident shape: three sibling blocked tasks share
        identical satisfied deps (mirrors the observed 5233/5234/5235
        cluster). The MIDDLE task's status flip raises; the EARLIER sibling
        (5233) still flips, and — the actual defect this task fixes — the
        LATER sibling (5235) must NOT be collateral-stranded by 5234's
        raise."""
        scheduler = _make_scheduler()
        scheduler.escalation_queue = _FakeEscalationQueue()  # type: ignore[assignment]

        dep_done = {'id': '9', 'status': 'done'}
        t5233 = _blocked_task('5233')
        t5234 = _blocked_task('5234')
        t5235 = _blocked_task('5235')
        ctx = TickContext(
            tasks=[dep_done, t5233, t5234, t5235],
            status_map={'9': 'done'},
            tasks_by_id={
                '9': dep_done, '5233': t5233, '5234': t5234, '5235': t5235,
            },
        )

        async def _raise_for_5234(task_id, status):
            if task_id == '5234':
                raise RuntimeError('transient backend failure')
            return None

        scheduler.set_task_status = AsyncMock(  # type: ignore[method-assign]
            side_effect=_raise_for_5234
        )

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_any_await(  # type: ignore[attr-defined]
            '5233', 'pending'
        )
        scheduler.set_task_status.assert_any_await(  # type: ignore[attr-defined]
            '5235', 'pending'
        )


# ---------------------------------------------------------------------------
# Harness wiring (step-11/12): the escalation queue injected into the
# scheduler is the fail-safe consumer for mechanism 2's no-open-escalation
# guard above (TestRedispatchStrandedBlockedCore.test_no_escalation_queue_
# fails_safe) — without this wiring scheduler.escalation_queue stays None
# forever in production and the sweep never fires.
# ---------------------------------------------------------------------------


def _build_harness(mock_orch_config) -> Harness:
    """Construct a Harness with heavy constructors patched out (mirrors
    test_harness_merge_registry_wiring.py's _build_harness)."""
    mock_orch_config.max_concurrent_tasks = 2
    mock_orch_config.fused_memory.project_id = 'test'

    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        return Harness(mock_orch_config)


@pytest.mark.asyncio
class TestStartEscalationServerInjectsSchedulerEscalationQueue:
    """_start_escalation_server wires the SAME escalation-queue instance into
    scheduler.escalation_queue, mirroring the review_checkpoint.escalation_queue
    injection immediately above it in harness.py (the Scheduler is built long
    before self._escalation_queue exists, so constructor injection is
    impossible — this is attribute injection at server-start time).

    RED until step-12 adds the wiring line in harness.py.
    """

    async def test_scheduler_escalation_queue_is_harness_escalation_queue(
        self, tmp_path: Path, mock_orch_config,
    ):
        h = _build_harness(mock_orch_config)

        # Set up attributes _start_escalation_server reads from config
        mock_orch_config.escalation.queue_dir = str(tmp_path / 'esc')
        mock_orch_config.escalation.host = '127.0.0.1'
        mock_orch_config.escalation.port = 19998
        # _start_escalation_server wires review_checkpoint if not None
        h.review_checkpoint = None
        h.event_store = None

        with patch('orchestrator.harness.create_server', return_value=MagicMock()), \
             patch('uvicorn.Server') as mock_uv_cls:
            # Make the uvicorn serve() coroutine return immediately so the
            # background task finishes and doesn't leave a dangling Task.
            mock_uv_cls.return_value.serve = AsyncMock()
            await h._start_escalation_server()

        assert h._escalation_queue is not None, 'sanity: the queue must be constructed'
        assert h.scheduler.escalation_queue is h._escalation_queue, (
            'Expected the SAME EscalationQueue instance as h._escalation_queue'
        )

        # Clean up the escalation background task if it's still running
        task = getattr(h, '_escalation_task', None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


# ---------------------------------------------------------------------------
# Task 3535 (PRD task beta, D5) — STRUCTURED EMISSION at this veto site.
#
# Until now the open-escalation veto below was a BARE ``continue``: a blocked
# task could sit un-redispatched for weeks and the journal said nothing about
# which record was holding it.  These tests pin the emission half.
#
# The whole point of this task is EMISSION BEFORE BEHAVIOR, so every class
# here carries a disposition assertion alongside its event assertion: the set
# of blocked -> pending flips must be byte-identical to what it was before a
# single event row existed.  The canonical WHY lives in
# ``orchestrator.recovery_emission``'s module docstring; it is not restated
# here or at the call site.
# ---------------------------------------------------------------------------

def _esc(
    esc_id: str,
    *,
    task_id: str = 'T1',
    severity: str = 'blocking',
    level: int = 0,
    secs_ago: float = 120.0,
) -> Escalation:
    """A REAL ``Escalation`` row — the shape ``get_by_task`` actually returns.

    Real rather than a stub so ``classify_pins`` reads the genuine
    severity/level/filing-identity surface, and so the age helper is exercised
    against the real ``timestamp`` field name (``Escalation`` carries
    ``timestamp``; the emitter reads ``created_at``, so the scheduler must
    normalise — a mismatch there would silently null every age).
    """
    return Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role='implementer',
        severity=severity,
        category='infra_issue',
        summary=f'{esc_id} summary',
        level=level,
        timestamp=(FIXED_DT - timedelta(seconds=secs_ago)).isoformat(),
    )


class _RaisingEscalationQueue:
    """``get_by_task`` RAISES — the store-unreadable third state.

    Distinct from an empty return: collapsing the two would route a
    genuinely-pinned strand into the flip branch (the esc-3163 lesson).
    """

    def __init__(self, exc: Exception | None = None):
        self.exc = exc or OSError('escalation store is unreadable')
        self.calls: list[str] = []

    def get_by_task(self, task_id: str, status: str | None = None) -> list:
        self.calls.append(task_id)
        raise self.exc


def _emitting_scheduler(tmp_path: Path, **config_overrides) -> Scheduler:
    """``_make_scheduler`` plus a REAL EventStore on tmp_path.

    Real (not a mock) so the payload's JSON round-trip through the store is
    exercised — an unserialisable member would drop the whole row in
    production and a MagicMock would happily accept it.
    """
    scheduler = _make_scheduler(**config_overrides)
    scheduler.event_store = EventStore(tmp_path / 'runs.db', 'run-test')
    return scheduler


def _recovery_rows(scheduler: Scheduler) -> list[dict]:
    """Every recovery event row, oldest first, ``data`` JSON-decoded."""
    conn = sqlite3.connect(str(scheduler.event_store.db_path))  # type: ignore[union-attr]
    try:
        return [
            {'task_id': tid, 'event_type': et, 'data': json.loads(raw or '{}')}
            for tid, et, raw in conn.execute(
                'SELECT task_id, event_type, data FROM events '
                'WHERE event_type IN (?, ?) ORDER BY id',
                (EventType.recovery_vetoed.value, EventType.recovery_left.value),
            )
        ]
    finally:
        conn.close()


@pytest.mark.asyncio
class TestBlockedRedispatchVetoIsDescribed:
    """(1) The open-escalation veto emits ``recovery_vetoed`` on first sight."""

    async def test_veto_still_holds_the_task(self, tmp_path: Path):
        """The disposition half, asserted FIRST: emission changes nothing."""
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = _FakeEscalationQueue(  # type: ignore[assignment]
            {'T1': [_esc('esc-1')]},
        )

        result = await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_blocked_task('T1')),
        )

        assert result is _CONTINUE
        assert scheduler.set_task_status.await_count == 0, (  # type: ignore[attr-defined]
            'an open escalation must still VETO the flip — emission is not a decision'
        )

    async def test_veto_emits_exactly_one_recovery_vetoed_row(self, tmp_path: Path):
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = _FakeEscalationQueue(  # type: ignore[assignment]
            {'T1': [_esc('esc-1')]},
        )

        await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_blocked_task('T1')),
        )

        rows = _recovery_rows(scheduler)
        assert len(rows) == 1, f'expected exactly one row, got {rows}'
        assert rows[0]['event_type'] == EventType.recovery_vetoed.value
        assert rows[0]['task_id'] == 'T1', (
            'task_id must be a first-class COLUMN so these rows stay joinable'
        )

    async def test_payload_names_the_site_and_the_reason(self, tmp_path: Path):
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = _FakeEscalationQueue(  # type: ignore[assignment]
            {'T1': [_esc('esc-1')]},
        )

        await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_blocked_task('T1')),
        )

        data = _recovery_rows(scheduler)[0]['data']
        assert data['site'] == 'scheduler_blocked_redispatch'
        assert data['reason'] == 'escalation_pinned'
        assert data['store_unavailable'] is False
        assert data['task_id'] == 'T1'
        assert set(data) == {
            'task_id', 'site', 'shape', 'reason', 'escalation_ids',
            'ages_secs', 'measured_at', 'store_unavailable', 'streak',
        }

    async def test_payload_names_the_pinning_ids(self, tmp_path: Path):
        """Both a pinning L1 and a dead L0 are NAMED — the bucketing is
        descriptive here, never the veto answer (that stays ``bool(rows)``)."""
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = _FakeEscalationQueue(  # type: ignore[assignment]
            {'T1': [_esc('esc-1', level=1), _esc('esc-2', level=0)]},
        )

        await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_blocked_task('T1')),
        )

        buckets = _recovery_rows(scheduler)[0]['data']['escalation_ids']
        assert buckets['queue_handoff'] == ['esc-1']
        assert buckets['dead_l0'] == ['esc-2'], (
            'no live claimant, so an L0 filed by a dead incarnation buckets dead_l0'
        )
        assert set(buckets) == {'dead_l0', 'queue_handoff', 'non_pinning'}

    async def test_payload_ages_each_pinning_id_by_id(self, tmp_path: Path):
        """``ages_secs`` is a MAPPING keyed by id — a parallel list would
        mis-attribute an age the moment a consumer re-sorted the ids."""
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = _FakeEscalationQueue(  # type: ignore[assignment]
            {'T1': [_esc('esc-1', secs_ago=120.0), _esc('esc-2', secs_ago=600.0)]},
        )

        await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_blocked_task('T1')),
        )

        ages = _recovery_rows(scheduler)[0]['data']['ages_secs']
        assert set(ages) == {'esc-1', 'esc-2'}
        assert ages['esc-1'] is not None and ages['esc-1'] >= 100.0
        assert ages['esc-2'] > ages['esc-1'], (
            'the older record must age larger — the ages come from the real '
            'row timestamps, not from position'
        )

    async def test_payload_names_the_shape_this_site_can_resolve(self, tmp_path: Path):
        """The elements this phase KNOWS are stated; the branch state it does
        not resolve renders ``unknown`` rather than being guessed."""
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = _FakeEscalationQueue(  # type: ignore[assignment]
            {'T1': [_esc('esc-1')]},
        )

        await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_blocked_task('T1')),
        )

        assert _recovery_rows(scheduler)[0]['data']['shape'] == (
            'blocked|false|unknown|true|-'
        )

    async def test_payload_stamps_an_iso_utc_measured_at(self, tmp_path: Path):
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = _FakeEscalationQueue(  # type: ignore[assignment]
            {'T1': [_esc('esc-1')]},
        )

        await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_blocked_task('T1')),
        )

        measured_at = datetime.fromisoformat(
            _recovery_rows(scheduler)[0]['data']['measured_at'],
        )
        assert measured_at.tzinfo is not None

    async def test_a_healthy_flip_emits_nothing(self, tmp_path: Path):
        """A task that is actually redispatched is not a hold — no row."""
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = _FakeEscalationQueue()  # type: ignore[assignment]

        await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_blocked_task('T1')),
        )

        scheduler.set_task_status.assert_awaited_once_with('T1', 'pending')  # type: ignore[attr-defined]
        assert _recovery_rows(scheduler) == [], (
            'an ACTION is not a hold — emitting here would bury the strands'
        )

    async def test_a_task_skipped_before_the_veto_emits_nothing(self, tmp_path: Path):
        """A live-claimant / deterministic / cooldown skip never reaches the
        veto, so it stays silent — those are the healthy majority."""
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = _FakeEscalationQueue(  # type: ignore[assignment]
            {'T1': [_esc('esc-1')]},
        )

        await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_deterministic_blocked_task('T1')),
        )

        assert _recovery_rows(scheduler) == []


@pytest.mark.asyncio
class TestBlockedRedispatchTickFrequencyGuard:
    """(2) This phase runs per dispatch TICK, not per sweep.

    Unconditional emission would append one SQLite row per tick per pinned
    task, forever (INV-4).  Emission is signature-transition-gated instead.
    """

    async def test_repeat_ticks_with_an_identical_signature_emit_once(
        self, tmp_path: Path,
    ):
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = _FakeEscalationQueue(  # type: ignore[assignment]
            {'T1': [_esc('esc-1')]},
        )

        for _ in range(2):
            await scheduler._phase_redispatch_stranded_blocked(
                _ctx_with_done_dep(_blocked_task('T1')),
            )

        assert len(_recovery_rows(scheduler)) == 1, (
            'a quiet repeat of a hold already on record must not re-state it'
        )

    async def test_many_ticks_stay_bounded_not_one_row_per_tick(
        self, tmp_path: Path,
    ):
        """Ten ticks produce TWO rows, not ten: the transition, plus the one
        deliberate re-statement at the streak threshold crossing (the moment
        a sweep-frequency site would have alarmed)."""
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = _FakeEscalationQueue(  # type: ignore[assignment]
            {'T1': [_esc('esc-1')]},
        )

        for _ in range(10):
            await scheduler._phase_redispatch_stranded_blocked(
                _ctx_with_done_dep(_blocked_task('T1')),
            )

        rows = _recovery_rows(scheduler)
        assert len(rows) == 2, f'expected transition + threshold crossing, got {rows}'
        assert [r['data']['streak'] for r in rows] == [
            1, scheduler.config.recovery_emission.veto_streak_threshold,
        ]

    async def test_a_changed_pinning_id_set_re_emits(self, tmp_path: Path):
        """A DIFFERENT record holding the task is a new fact, not a repeat."""
        scheduler = _emitting_scheduler(tmp_path)
        queue = _FakeEscalationQueue({'T1': [_esc('esc-1')]})
        scheduler.escalation_queue = queue  # type: ignore[assignment]

        await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_blocked_task('T1')),
        )
        queue._by_task['T1'] = [_esc('esc-1'), _esc('esc-2')]
        await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_blocked_task('T1')),
        )

        rows = _recovery_rows(scheduler)
        assert len(rows) == 2
        assert [r['data']['streak'] for r in rows] == [1, 1], (
            'a changed signature RESTARTS the streak, so an alarm only ever '
            'describes N genuinely identical holds'
        )

    async def test_two_pinned_tasks_keep_independent_streaks(self, tmp_path: Path):
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = _FakeEscalationQueue(  # type: ignore[assignment]
            {'T1': [_esc('esc-1')], 'T2': [_esc('esc-2', task_id='T2')]},
        )
        dep = {'id': '9', 'status': 'done'}
        t1, t2 = _blocked_task('T1'), _blocked_task('T2')
        ctx = TickContext(
            tasks=[dep, t1, t2],
            status_map={'9': 'done'},
            tasks_by_id={'9': dep, 'T1': t1, 'T2': t2},
        )

        await scheduler._phase_redispatch_stranded_blocked(ctx)

        rows = _recovery_rows(scheduler)
        assert {r['task_id'] for r in rows} == {'T1', 'T2'}
        assert all(r['data']['streak'] == 1 for r in rows)


@pytest.mark.asyncio
class TestBlockedRedispatchStoreUnreadable:
    """(3) A get_by_task that RAISES is the store-unreadable third state."""

    async def test_read_failure_still_skips_the_task(self, tmp_path: Path):
        """The disposition is IDENTICAL to today's broad ``except: continue``
        — an unreadable store must never be read as "nothing holds it"."""
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = _RaisingEscalationQueue()  # type: ignore[assignment]

        result = await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_blocked_task('T1')),
        )

        assert result is _CONTINUE
        assert scheduler.set_task_status.await_count == 0, (  # type: ignore[attr-defined]
            'a task must NEVER be flipped to pending on an unreadable store'
        )

    async def test_read_failure_emits_recovery_left(self, tmp_path: Path):
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = _RaisingEscalationQueue()  # type: ignore[assignment]

        await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_blocked_task('T1')),
        )

        rows = _recovery_rows(scheduler)
        assert len(rows) == 1, f'expected exactly one row, got {rows}'
        assert rows[0]['event_type'] == EventType.recovery_left.value, (
            'nothing was held back by a RECORD — the site could not read the store'
        )
        assert rows[0]['task_id'] == 'T1'
        data = rows[0]['data']
        assert data['reason'] == 'escalation_store_unavailable'
        assert data['store_unavailable'] is True
        assert data['site'] == 'scheduler_blocked_redispatch'
        assert data['escalation_ids'] == {
            'dead_l0': [], 'queue_handoff': [], 'non_pinning': [],
        }

    async def test_read_failure_does_not_collateral_strand_a_sibling(
        self, tmp_path: Path,
    ):
        """The narrow read guard replaces the broad handler for this arm only
        — a sibling sorted after the failing task is still swept."""
        scheduler = _emitting_scheduler(tmp_path)

        class _OneBadRead:
            def get_by_task(self, task_id: str, status: str | None = None) -> list:
                if task_id == 'T1':
                    raise OSError('unreadable')
                return []

        scheduler.escalation_queue = _OneBadRead()  # type: ignore[assignment]
        dep = {'id': '9', 'status': 'done'}
        t1, t2 = _blocked_task('T1'), _blocked_task('T2')
        ctx = TickContext(
            tasks=[dep, t1, t2],
            status_map={'9': 'done'},
            tasks_by_id={'9': dep, 'T1': t1, 'T2': t2},
        )

        await scheduler._phase_redispatch_stranded_blocked(ctx)

        scheduler.set_task_status.assert_awaited_once_with('T2', 'pending')  # type: ignore[attr-defined]

    async def test_repeat_read_failures_stay_bounded(self, tmp_path: Path):
        """A multi-day store outage must not become one row per tick either."""
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = _RaisingEscalationQueue()  # type: ignore[assignment]

        for _ in range(2):
            await scheduler._phase_redispatch_stranded_blocked(
                _ctx_with_done_dep(_blocked_task('T1')),
            )

        assert len(_recovery_rows(scheduler)) == 1


@pytest.mark.asyncio
class TestBlockedRedispatchQueueAbsent:
    """(4) No queue at all is a PROCESS-scoped fact, not a per-task one."""

    async def test_queue_absent_emits_one_phase_scoped_row(self, tmp_path: Path):
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = None

        result = await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_blocked_task('T1')),
        )

        assert result is _CONTINUE
        assert scheduler.set_task_status.await_count == 0, (  # type: ignore[attr-defined]
            'without the queue this phase cannot verify "no open escalation", '
            'so it must never flip'
        )
        rows = _recovery_rows(scheduler)
        assert len(rows) == 1
        assert rows[0]['event_type'] == EventType.recovery_left.value
        assert rows[0]['task_id'] is None, (
            'no single subject — the whole PHASE is degraded, not one task'
        )
        data = rows[0]['data']
        assert data['task_id'] is None
        assert data['reason'] == 'escalation_store_unavailable'
        assert data['store_unavailable'] is True
        assert data['site'] == 'scheduler_blocked_redispatch'

    async def test_queue_absent_emits_once_per_process_not_per_tick(
        self, tmp_path: Path,
    ):
        """A queue that is never wired would otherwise emit forever, on every
        tick, for the life of the orchestrator."""
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = None

        for _ in range(5):
            await scheduler._phase_redispatch_stranded_blocked(
                _ctx_with_done_dep(_blocked_task('T1')),
            )

        assert len(_recovery_rows(scheduler)) == 1

    async def test_queue_absent_notice_re_arms_when_the_queue_comes_and_goes(
        self, tmp_path: Path,
    ):
        """The Harness attribute-injects the queue AFTER the Scheduler is
        built, so "absent" is a state the process genuinely leaves and can
        re-enter — a latch that never re-arms would hide the second outage."""
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = None
        await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_blocked_task('T1')),
        )

        scheduler.escalation_queue = _FakeEscalationQueue()  # type: ignore[assignment]
        await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_blocked_task('T1')),
        )

        scheduler.escalation_queue = None
        await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_blocked_task('T2')),
        )

        assert len(_recovery_rows(scheduler)) == 2

    async def test_disabled_sweep_emits_nothing(self, tmp_path: Path):
        """The kill switch short-circuits BEFORE the queue check — an
        operator who turned the sweep off is not running a degraded store."""
        scheduler = _emitting_scheduler(
            tmp_path, stranded_blocked_redispatch_enabled=False,
        )
        scheduler.escalation_queue = None

        await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_blocked_task('T1')),
        )

        assert _recovery_rows(scheduler) == []


@pytest.mark.asyncio
class TestBlockedRedispatchZeroDispositionChange:
    """(5) The load-bearing contract: the flip SET is byte-identical.

    The veto predicate stays ``bool(rows)`` verbatim.  ``classify_pins`` is
    consulted only to bucket ids for the payload — swapping in
    ``PinReport.pins`` would stop an INFO record vetoing here, which is a real
    disposition change owned by task eta (3541), not by this task.
    """

    async def test_an_info_severity_record_still_vetoes(self, tmp_path: Path):
        """``PinReport.pins`` is False for an info-only record.  If a
        well-meaning refactor swaps the predicate for it, this trips."""
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = _FakeEscalationQueue(  # type: ignore[assignment]
            {'T1': [_esc('esc-1', severity='info')]},
        )

        await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_blocked_task('T1')),
        )

        assert scheduler.set_task_status.await_count == 0, (  # type: ignore[attr-defined]
            'the predicate is bool(rows); an info record STILL vetoes at this site'
        )
        data = _recovery_rows(scheduler)[0]['data']
        assert data['reason'] == 'escalation_pinned'
        assert data['escalation_ids']['non_pinning'] == ['esc-1'], (
            'the payload may DESCRIBE the record as non-pinning — that is the '
            'measurable gap task eta closes — while the veto still stands'
        )

    @pytest.mark.parametrize(
        ('rows_by_task', 'expect_flip'),
        [
            pytest.param({}, True, id='no-open-records-flips'),
            pytest.param({'T1': [_esc('esc-1')]}, False, id='blocking-l0-vetoes'),
            pytest.param(
                {'T1': [_esc('esc-1', level=1)]}, False, id='l1-handoff-vetoes',
            ),
            pytest.param(
                {'T1': [_esc('esc-1', severity='info')]}, False, id='info-vetoes',
            ),
        ],
    )
    async def test_flip_set_is_identical_with_emission_on_and_off(
        self, tmp_path: Path, rows_by_task, expect_flip,
    ):
        outcomes = []
        for enabled in (True, False):
            scheduler = _emitting_scheduler(
                tmp_path / f'on-{enabled}',
                recovery_emission={'enabled': enabled},
            )
            scheduler.escalation_queue = _FakeEscalationQueue(  # type: ignore[assignment]
                dict(rows_by_task),
            )
            await scheduler._phase_redispatch_stranded_blocked(
                _ctx_with_done_dep(_blocked_task('T1')),
            )
            outcomes.append([
                c.args for c in scheduler.set_task_status.await_args_list  # type: ignore[attr-defined]
            ])
            if not enabled:
                assert _recovery_rows(scheduler) == [], (
                    'the kill switch must silence the events'
                )

        assert outcomes[0] == outcomes[1], (
            'emission changed a disposition — that is the one thing this task '
            'may not do'
        )
        assert bool(outcomes[0]) is expect_flip

    async def test_emission_failure_never_collateral_strands_a_sibling(
        self, tmp_path: Path,
    ):
        """(6) The per-task isolation guard still wraps the emission path."""
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = _FakeEscalationQueue(  # type: ignore[assignment]
            {'T1': [_esc('esc-1')]},
        )
        scheduler._emit_recovery_disposition = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError('telemetry exploded'),
        )
        dep = {'id': '9', 'status': 'done'}
        t1, t2 = _blocked_task('T1'), _blocked_task('T2')
        ctx = TickContext(
            tasks=[dep, t1, t2],
            status_map={'9': 'done'},
            tasks_by_id={'9': dep, 'T1': t1, 'T2': t2},
        )

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_awaited_once_with('T2', 'pending')  # type: ignore[attr-defined]

    async def test_no_event_store_is_a_silent_no_op(self, tmp_path: Path):
        """Every disposition still lands with telemetry entirely absent."""
        scheduler = _make_scheduler()
        scheduler.event_store = None
        scheduler.escalation_queue = _FakeEscalationQueue(  # type: ignore[assignment]
            {'T1': [_esc('esc-1')]},
        )

        result = await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(_blocked_task('T1')),
        )

        assert result is _CONTINUE
        assert scheduler.set_task_status.await_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
class TestBlockedRedispatchReleasesItsVetoStreak:
    """The phase pops its tracker entries at end of tick (amendment, cycle 1).

    This site runs per dispatch TICK and charges no alarm, so its tracker
    entries are pure footprint — and without a release edge a task vetoed here
    once and then gone done would leave its entry behind for the life of the
    process.  That is exactly the growth
    ``RecoveryVetoStreakTracker.clear``'s docstring says the pop exists to
    prevent, and the per-tick sites are where the loss would actually happen.
    """

    def _tracked(self, scheduler: Scheduler) -> list[tuple[str, str]]:
        tracker = getattr(scheduler, '_recovery_veto_tracker', None)
        return [] if tracker is None else list(tracker.tracked())

    async def _tick(self, scheduler: Scheduler, task: dict) -> None:
        await scheduler._phase_redispatch_stranded_blocked(
            _ctx_with_done_dep(task),
        )

    async def test_a_hold_that_ends_pops_the_entry(self, tmp_path: Path):
        scheduler = _emitting_scheduler(tmp_path)
        queue = _FakeEscalationQueue({'T1': [_esc('esc-1')]})
        scheduler.escalation_queue = queue  # type: ignore[assignment]

        await self._tick(scheduler, _blocked_task('T1'))
        assert self._tracked(scheduler) == [
            ('scheduler_blocked_redispatch', 'T1'),
        ]

        # The escalation is resolved: the task is no longer held, and this
        # tick redispatches it instead.
        queue._by_task['T1'] = []
        await self._tick(scheduler, _blocked_task('T1'))

        assert self._tracked(scheduler) == [], (
            'a tick that described no hold for T1 must pop its streak'
        )

    async def test_a_task_that_leaves_the_candidate_set_pops_too(
        self, tmp_path: Path,
    ):
        """The case a per-task 'the hold ended' signal can never deliver: the
        task went done/cancelled, so this phase never looks at it again."""
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = _FakeEscalationQueue(  # type: ignore[assignment]
            {'T1': [_esc('esc-1')]},
        )

        await self._tick(scheduler, _blocked_task('T1'))
        assert self._tracked(scheduler)

        # T1 is gone from ctx.tasks entirely; only an unrelated task remains.
        await self._tick(scheduler, _blocked_task('T2'))

        assert self._tracked(scheduler) == []

    async def test_a_hold_that_returns_starts_a_fresh_streak(
        self, tmp_path: Path,
    ):
        """The pop is a RELEASE, not a mute: the same hold reappearing is a
        new fact and must announce itself again (streak back to 1)."""
        scheduler = _emitting_scheduler(tmp_path)
        queue = _FakeEscalationQueue({'T1': [_esc('esc-1')]})
        scheduler.escalation_queue = queue  # type: ignore[assignment]

        await self._tick(scheduler, _blocked_task('T1'))
        queue._by_task['T1'] = []
        await self._tick(scheduler, _blocked_task('T1'))
        queue._by_task['T1'] = [_esc('esc-1')]
        await self._tick(scheduler, _blocked_task('T1'))

        rows = _recovery_rows(scheduler)
        assert [r['data']['streak'] for r in rows] == [1, 1], (
            f'a released-and-returned hold must re-announce at streak 1: {rows}'
        )

    async def test_an_isolated_exception_does_not_pop_the_entry(
        self, tmp_path: Path,
    ):
        """The isolation guard means the phase never reached a verdict for
        that task — popping would read as 'the hold ended', which is the
        opposite of what an exception says."""
        scheduler = _emitting_scheduler(tmp_path)
        queue = _FakeEscalationQueue({'T1': [_esc('esc-1')]})
        scheduler.escalation_queue = queue  # type: ignore[assignment]

        await self._tick(scheduler, _blocked_task('T1'))
        assert self._tracked(scheduler)

        scheduler.escalation_queue = _RaisingEscalationQueue(  # type: ignore[assignment]
            RuntimeError('boom'),
        )
        scheduler._emit_recovery_disposition = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError('boom'),
        )
        await self._tick(scheduler, _blocked_task('T1'))

        assert self._tracked(scheduler) == [
            ('scheduler_blocked_redispatch', 'T1'),
        ], 'an unknown verdict must not be read as a released hold'

    async def test_the_release_never_breaks_the_tick(self, tmp_path: Path):
        """Bookkeeping is never load-bearing: a tracker that raises must not
        abort a tick phase."""
        scheduler = _emitting_scheduler(tmp_path)
        scheduler.escalation_queue = _FakeEscalationQueue()  # type: ignore[assignment]
        scheduler._recovery_veto_tracker = MagicMock()
        scheduler._recovery_veto_tracker.tracked = MagicMock(
            side_effect=RuntimeError('tracker is gone'),
        )

        await self._tick(scheduler, _blocked_task('T1'))

        flip: AsyncMock = scheduler.set_task_status  # type: ignore[assignment]
        flip.assert_awaited_once_with('T1', 'pending')
