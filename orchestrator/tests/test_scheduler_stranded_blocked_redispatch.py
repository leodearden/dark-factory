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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import OrchestratorConfig
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
        scheduler.escalation_queue = _FakeEscalationQueue()

        task = _blocked_task('T1')
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_awaited_once_with('T1', 'pending')

    @pytest.mark.asyncio
    async def test_open_escalation_blocks_flip(self):
        """(b) Deliberately-parked /unblock: same null-claimant signature,
        but an open escalation exists -> must NOT be flipped."""
        scheduler = _make_scheduler()
        scheduler.escalation_queue = _FakeEscalationQueue(
            by_task={'T1': [{'id': 'esc-1'}]}
        )

        task = _blocked_task('T1')
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_deps_not_satisfied_blocks_flip(self):
        """(c) Dependency still in-progress -> not genuinely stranded (still
        legitimately waiting) -> must NOT be flipped."""
        scheduler = _make_scheduler()
        scheduler.escalation_queue = _FakeEscalationQueue()

        task = _blocked_task('T1')
        ctx = _ctx_with_done_dep(task, dep_status='in-progress')

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_live_claimant_blocks_flip(self):
        """(d) Blocked but LIVE (claimant + fresh heartbeat, ttl=300s default)
        -> is_stranded_blocked is False -> must NOT be flipped."""
        scheduler = _make_scheduler()
        scheduler.escalation_queue = _FakeEscalationQueue()

        fresh_heartbeat = (FIXED_DT - timedelta(minutes=1)).isoformat()
        task = _blocked_task(
            'T1',
            claimant_run_id='run-x/session-y/pid=1',
            heartbeat_at=fresh_heartbeat,
        )
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_not_awaited()

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
        scheduler.set_task_status.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_kill_switch_disables_sweep(self):
        """(f) config.stranded_blocked_redispatch_enabled=False -> sweep is
        entirely dormant, even for an otherwise-genuine crash-strand."""
        scheduler = _make_scheduler(stranded_blocked_redispatch_enabled=False)
        scheduler.escalation_queue = _FakeEscalationQueue()

        task = _blocked_task('T1')
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_not_awaited()


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
        scheduler.escalation_queue = _FakeEscalationQueue()

        task = _deterministic_blocked_task('T1')
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_deterministic_gate_task_not_flipped_with_open_l2(self):
        """Companion variant: the exclusion holds regardless of escalation
        level — even WITH an open L2 escalation present, the deterministic
        carve-out (not the generic open-escalation gate) is what protects
        it."""
        scheduler = _make_scheduler()
        scheduler.escalation_queue = _FakeEscalationQueue(
            by_task={'T1': [{'id': 'esc-l2', 'level': 2}]}
        )

        task = _deterministic_blocked_task('T1')
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_deterministic_control_still_flips(self):
        """Control: an otherwise-identical NON-deterministic blocked task
        (same null-claimant/done-dep/empty-queue shape) IS flipped — guards
        against the deterministic exclusion being too broad."""
        scheduler = _make_scheduler()
        scheduler.escalation_queue = _FakeEscalationQueue()

        task = _blocked_task('T1')
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_awaited_once_with('T1', 'pending')

    # -- (b) actively dispatched ----------------------------------------------

    @pytest.mark.asyncio
    async def test_actively_dispatched_task_not_flipped(self):
        scheduler = _make_scheduler()
        scheduler.escalation_queue = _FakeEscalationQueue()
        scheduler._dispatched.add('T1')

        task = _blocked_task('T1')
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_not_awaited()

    # -- (c) within dispatch/requeue cooldown window --------------------------

    @pytest.mark.asyncio
    async def test_within_cooldown_window_not_flipped(self):
        fixed_monotonic = 1_000.0
        scheduler = _make_scheduler(time_source=lambda: fixed_monotonic)
        scheduler.escalation_queue = _FakeEscalationQueue()
        scheduler._requeue_until['T1'] = fixed_monotonic + 100.0

        task = _blocked_task('T1')
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_not_awaited()

    # -- (d) workflow just cancelled/parked (finally-block grace window) -----

    @pytest.mark.asyncio
    async def test_workflow_cancel_recent_not_flipped(self):
        scheduler = _make_scheduler()
        scheduler.escalation_queue = _FakeEscalationQueue()
        scheduler.note_workflow_cancelled('T1')
        assert scheduler.workflow_cancel_recent('T1') is True

        task = _blocked_task('T1')
        ctx = _ctx_with_done_dep(task)

        result = await scheduler._phase_redispatch_stranded_blocked(ctx)

        assert result is _CONTINUE
        scheduler.set_task_status.assert_not_awaited()


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
