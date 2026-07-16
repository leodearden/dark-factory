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

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from orchestrator.config import OrchestratorConfig
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


def _make_scheduler(**config_overrides) -> Scheduler:
    config = OrchestratorConfig(max_per_module=1, **config_overrides)
    scheduler = Scheduler(config, wall_time_source=lambda: FIXED_DT)
    scheduler.set_task_status = AsyncMock()  # type: ignore[method-assign]
    return scheduler


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
