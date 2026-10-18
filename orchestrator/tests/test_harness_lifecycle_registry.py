"""Tests for Harness's LifecycleRegistry wiring (task 2241, W10-η).

PRD ``plans/harness-supervision-prd.md`` §5.3 (LR-1/2/3): the eleven
background-loop/service lifecycles scattered across harness.py collapse
behind ONE reusable seam — ``orchestrator.background_service``'s
``LifecycleRegistry`` — so the recurring shutdown-hang class (survey 2.3;
tasks 108/161/162/169/875/1080) becomes structurally impossible.

Covers:
  step-15 RED — (a) LR-3/S2 registration-list literal: with all eleven
                enable-gates at their (True) default, harness._lifecycle's
                registered service names equal the canonical START order;
                a disabled sweep is conditionally absent.
            (b) Wiring: run() awaits start_all() during startup (before the
                recovery block) and stop_all() in the finally.

RED until step-16 GREEN adds ``Harness._build_lifecycle_registry`` /
``Harness._lifecycle``. This module accesses both defensively — inside test
bodies, never at module/class scope — so collection stays clean while they
do not yet exist.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _escalation_server_capture import create_server_kwargs
from escalation.queue import EscalationQueue

from orchestrator.background_service import BackgroundService
from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.flake_debt_drain import STALE_DUE_ESCALATION_SENTINEL
from orchestrator.flake_ledger import (
    DueItem,
    FlakeCallSite,
    enqueue_debt,
    ledger_db_path,
    read_due_items,
    record_failed_due_attempt,
)
from orchestrator.harness import Harness
from orchestrator.verify_admission_gate import AdmissionPlan, AdmissionPolicy, gate_for

# Canonical start-order registration list — task 2241 plan.json "HARNESS
# MIGRATION": escalation-server and merge-worker first (the recovery block
# depends on both being live); the remaining bespoke adapter (offline-lane)
# and the seven BackgroundService sweeps follow, then the always-on services
# that keep their cadence whatever the dispatch loop does (task 5344; the
# verify-admission watch, task 5628; the flake-debt drain, task 6934).
_CANONICAL_ORDER = [
    'escalation-server',
    'merge-worker',
    'offline-lane',
    'orphan-l0-reaper',
    'terminal-status-watcher',
    'watcher-supervisor',
    'stranded-reconcile',
    'main-tip-sweep',
    'no-landings-breaker',
    'deterministic-recon-sweep',
    'warm-lane-gc',
    'stale-service-restart',
    'merge-heartbeat',
    'verify-admission-watch',
    'flake-debt-drain',
]


# ---------------------------------------------------------------------------
# step-15a: LR-3/S2 registration-list literal
# ---------------------------------------------------------------------------


class TestLifecycleRegistrationOrder:
    """harness._build_lifecycle_registry() wires the canonical name order."""

    def test_all_enabled_matches_canonical_order(self, tmp_path: Path) -> None:
        """A real OrchestratorConfig defaults every enable-gate to True, so
        the freshly built registry's services are exactly the canonical START
        order (LR-3)."""
        config = OrchestratorConfig(project_root=tmp_path)
        harness = Harness(config)

        harness._build_lifecycle_registry()

        assert harness._lifecycle is not None
        assert [s.name for s in harness._lifecycle.services] == _CANONICAL_ORDER

    def test_disabled_sweep_is_conditionally_absent(self, tmp_path: Path) -> None:
        """A disabled BackgroundService-shaped sweep is absent from the
        registry, while the rest of the canonical order is undisturbed —
        conditional registration, not a new ``enabled`` field on
        BackgroundService itself."""
        config = OrchestratorConfig(project_root=tmp_path)
        config.warm_lane_gc_enabled = False
        harness = Harness(config)

        harness._build_lifecycle_registry()

        assert harness._lifecycle is not None
        names = [s.name for s in harness._lifecycle.services]
        assert 'warm-lane-gc' not in names
        assert names == [n for n in _CANONICAL_ORDER if n != 'warm-lane-gc']


# ---------------------------------------------------------------------------
# step-15b: run() wiring — start_all() at startup, stop_all() in the finally
# ---------------------------------------------------------------------------


def _make_run_wired_harness(tmp_path: Path) -> tuple[Harness, MagicMock]:
    """Real Harness with run()'s non-lifecycle side-effects stubbed.

    Mirrors test_harness_watcher_supervisor._make_run_wired_harness /
    test_singleton_lock._build_dirty_tree_startup_harness: patches
    McpLifecycle/Scheduler/BriefingAssembler at construction so no real
    servers start, and short-circuits the dispatch loop via a sentinel
    RuntimeError from scheduler.acquire_next so run() returns (via its
    finally:) almost immediately.
    """
    config = OrchestratorConfig(project_root=tmp_path)

    with patch('orchestrator.harness.McpLifecycle') as mock_mcp_cls, \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(config)

    mock_mcp = mock_mcp_cls.return_value
    mock_mcp.start = AsyncMock()
    mock_mcp.stop = AsyncMock()

    h.git_ops = MagicMock()
    h.git_ops.has_dirty_working_tree = AsyncMock(return_value=None)
    h.git_ops.worktree_base = tmp_path / '.worktrees'

    h._dismiss_stale_escalations = AsyncMock()
    h._tag_task_modules = AsyncMock()
    h._recover_crashed_tasks = AsyncMock()
    h._reconcile_lane_checkouts = AsyncMock()
    h._reconcile_stranded_in_progress = AsyncMock()
    h._tag_prd_metadata = AsyncMock()

    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[{'id': '1', 'status': 'pending'}])
    h.scheduler.get_statuses = AsyncMock(return_value=({'1': 'pending'}, None))
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.acquire_next = AsyncMock(side_effect=RuntimeError('__stop_sentinel__'))

    return h, mock_mcp


class TestLifecycleRegistryRunWiring:
    """run() awaits start_all() during startup and stop_all() in the finally.

    Pre-builds harness._lifecycle and monkeypatches its start_all/stop_all to
    fresh AsyncMocks BEFORE calling run(). _build_lifecycle_registry() is
    idempotent (a no-op once harness._lifecycle is already set), so run()'s
    own internal call does not clobber the pre-built (patched) registry —
    the SAME instance, with the patched methods, is what run() awaits.
    """

    @pytest.mark.asyncio
    async def test_start_all_awaited_during_startup(self, tmp_path: Path) -> None:
        h, _mock_mcp = _make_run_wired_harness(tmp_path)
        h._build_lifecycle_registry()
        assert h._lifecycle is not None
        h._lifecycle.start_all = AsyncMock()
        h._lifecycle.stop_all = AsyncMock()

        with pytest.raises(RuntimeError, match='__stop_sentinel__'):
            await h.run(prd_path=None)

        h._lifecycle.start_all.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_all_awaited_in_finally(self, tmp_path: Path) -> None:
        h, _mock_mcp = _make_run_wired_harness(tmp_path)
        h._build_lifecycle_registry()
        assert h._lifecycle is not None
        h._lifecycle.start_all = AsyncMock()
        h._lifecycle.stop_all = AsyncMock()

        with pytest.raises(RuntimeError, match='__stop_sentinel__'):
            await h.run(prd_path=None)

        h._lifecycle.stop_all.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_all_awaited_before_dismiss_stale_escalations(
        self, tmp_path: Path,
    ) -> None:
        """start_all() sits at the former escalation-server startup
        position — BEFORE the recovery block, whose first step is
        _dismiss_stale_escalations (task 2241 design decision: the
        startup-ritual collapse is safe because escalation-server +
        merge-worker are registered first-in-order and the seven sweeps are
        sleep-first)."""
        h, _mock_mcp = _make_run_wired_harness(tmp_path)
        h._build_lifecycle_registry()
        assert h._lifecycle is not None
        h._lifecycle.start_all = AsyncMock()
        h._lifecycle.stop_all = AsyncMock()

        parent = MagicMock(name='order')
        dismiss_mock = AsyncMock()
        h._dismiss_stale_escalations = dismiss_mock
        parent.attach_mock(h._lifecycle.start_all, 'start_all')
        parent.attach_mock(dismiss_mock, '_dismiss_stale_escalations')

        with pytest.raises(RuntimeError, match='__stop_sentinel__'):
            await h.run(prd_path=None)

        call_strs = [str(c) for c in parent.mock_calls]
        start_idx = next(i for i, s in enumerate(call_strs) if 'start_all' in s)
        dismiss_idx = next(
            i for i, s in enumerate(call_strs) if '_dismiss_stale_escalations' in s
        )
        assert start_idx < dismiss_idx, (
            f'start_all() must be awaited before _dismiss_stale_escalations(); '
            f'calls: {call_strs}'
        )


# ---------------------------------------------------------------------------
# task 5628: the verify-admission watch pass reads the harness's own gate
# ---------------------------------------------------------------------------


class TestVerifyAdmissionWatchPass:
    """``_run_verify_admission_watch_pass`` ticks the watch on the gate for the
    harness's configured slots dir, onto the harness's event store."""

    @staticmethod
    async def _with_one_queued_waiter(slots_dir: Path, run) -> None:
        gate = gate_for(slots_dir)
        policy = AdmissionPolicy(slots=1, order='arrival', promote_after_secs=7200.0)
        release = asyncio.Event()

        async def leg(task_id: str) -> None:
            request = AdmissionPlan.begin(role='task', task_id=task_id).request(
                module=task_id, scope_kind=None,
            )
            async with gate.admit(request, lambda: policy):
                await release.wait()

        legs = [asyncio.create_task(leg('H')), asyncio.create_task(leg('W'))]
        try:
            async with asyncio.timeout(5):
                while gate.snapshot().depth != 1:
                    await asyncio.sleep(0.005)
            await run()
        finally:
            release.set()
            async with asyncio.timeout(5):
                await asyncio.gather(*legs)

    @pytest.mark.asyncio
    async def test_pass_writes_a_heartbeat_for_a_queued_waiter(self, tmp_path: Path) -> None:
        slots_dir = tmp_path / 'slots'
        h = Harness(OrchestratorConfig(
            project_root=tmp_path, verify_admission_slots_dir=str(slots_dir),
        ))
        store = EventStore(tmp_path / 'runs.db', run_id='t')
        h.event_store = store

        await self._with_one_queued_waiter(slots_dir, h._run_verify_admission_watch_pass)

        [beat] = store.fetch_events_by_type(EventType.verify_admission_heartbeat)
        assert beat['data']['depth'] == 1
        assert beat['data']['oldest']['task_id'] == 'W'

    @pytest.mark.asyncio
    async def test_pass_without_an_event_store_leaves_the_alarm_for_a_later_one(
        self, tmp_path: Path,
    ) -> None:
        slots_dir = tmp_path / 'slots'
        alarm_secs = 0.01
        h = Harness(OrchestratorConfig(
            project_root=tmp_path, verify_admission_slots_dir=str(slots_dir),
            verify_admission_wait_alarm_secs=alarm_secs,
        ))
        store = EventStore(tmp_path / 'runs.db', run_id='t')

        async def pass_without_then_with_a_store() -> None:
            async with asyncio.timeout(5):
                while (gate_for(slots_dir).snapshot().oldest_wait_secs or 0.0) < alarm_secs:
                    await asyncio.sleep(0.005)
            h.event_store = None
            await h._run_verify_admission_watch_pass()
            h.event_store = store
            await h._run_verify_admission_watch_pass()

        await self._with_one_queued_waiter(slots_dir, pass_without_then_with_a_store)

        assert len(store.fetch_events_by_type(EventType.verify_admission_heartbeat)) == 1
        [aged] = store.fetch_events_by_type(EventType.verify_admission_wait_aged)
        assert aged['data']['waiter']['task_id'] == 'W'


# ---------------------------------------------------------------------------
# task 6934: the flake-debt drain pass reads the escalation queue the harness
# creates AFTER the registry is built
# ---------------------------------------------------------------------------


class TestFlakeDebtDrainPass:
    """The registered ``flake-debt-drain`` pass reaches the live escalation queue.

    ``_build_lifecycle_registry`` runs before ``_start_escalation_server`` creates
    the queue, so a queue captured at registration would be ``None`` for the
    drain's whole life and its stale-item escalation would degrade to a log line.
    """

    @pytest.mark.asyncio
    async def test_pass_files_the_stale_item_escalation_on_the_harness_queue(
        self, tmp_path: Path,
    ) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        config.escalation.host = '127.0.0.1'
        config.escalation.port = 0
        h = Harness(config)
        h.review_checkpoint = None
        h._build_lifecycle_registry()
        await create_server_kwargs(h)

        db_path = ledger_db_path(tmp_path)
        now = datetime.now(UTC)
        enqueue_debt(
            db_path,
            config.fused_memory.project_id,
            [DueItem(
                test_id='tests/test_flaky.py::test_x',
                observed_at=now.isoformat(),
                call_site=FlakeCallSite.merge_gate,
                head=None,
                observing_task_id=None,
            )],
            now=now - timedelta(minutes=31),
        )
        [item] = read_due_items(db_path, now=now + timedelta(days=1), limit=10) or []
        assert record_failed_due_attempt(
            db_path, item.id, reason='task_client_failed',
            next_attempt_at=now + timedelta(hours=1),
        )

        assert h._lifecycle is not None
        [drain] = [
            s for s in h._lifecycle.services
            if isinstance(s, BackgroundService) and s.name == 'flake-debt-drain'
        ]
        await drain.pass_fn()

        queue_dir = Path(config.escalation.queue_dir)
        fresh = EscalationQueue(queue_dir if queue_dir.is_absolute() else tmp_path / queue_dir)
        [record] = fresh.get_by_task(STALE_DUE_ESCALATION_SENTINEL, status='pending')
        assert record.level == 1
        assert record.category == 'infra_issue'
