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
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _escalation_server_capture import create_server_kwargs
from _git_fixtures import seed_repo
from _merge_lane_fakes import branch_merge_request, lane_scene_config
from escalation.queue import EscalationQueue
from shared.project_layout import escalation_queue_dir
from test_harness_startup_steps import _STUBBED_STEPS, _SYNC_STEPS, _UNSTUBBED_STEPS

from orchestrator.background_service import BackgroundService, ManagedService
from orchestrator.config import GitConfig, OrchestratorConfig
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
# MIGRATION": the loop-stall monitor first, so it watches every other
# service's whole life and stops last (task 7366); then escalation-server and
# merge-worker (the recovery block needs the escalation server live and the
# merge worker constructed, not running; task 8268); the remaining bespoke adapter (offline-lane)
# and the eight BackgroundService sweeps follow, then the always-on services
# that keep their cadence whatever the dispatch loop does (task 5344; the
# verify-admission watch, task 5628; the flake-debt drain, task 6934).
_CANONICAL_ORDER = [
    'loop-stall-monitor',
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
    'scheduled-deploy-settle',
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

    def test_scheduled_deploy_settle_follows_the_recon_sweep_gate(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        config.deterministic_recon_sweep_enabled = False
        harness = Harness(config)

        harness._build_lifecycle_registry()

        assert harness._lifecycle is not None
        gated = {'deterministic-recon-sweep', 'scheduled-deploy-settle'}
        names = [s.name for s in harness._lifecycle.services]
        assert names == [n for n in _CANONICAL_ORDER if n not in gated]

    @pytest.mark.asyncio
    async def test_a_scheduled_deploy_settle_pass_without_a_queue_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        harness = Harness(OrchestratorConfig(project_root=tmp_path))
        harness._build_lifecycle_registry()
        assert harness._lifecycle is not None
        [settle] = [s for s in harness._lifecycle.services if s.name == 'scheduled-deploy-settle']
        assert isinstance(settle, BackgroundService)

        with caplog.at_level('WARNING', logger='orchestrator.harness'):
            await settle.pass_fn()

        assert any(
            'scheduled-deploy settle' in r.getMessage() and 'no escalation queue' in r.getMessage()
            for r in caplog.records if r.levelname == 'WARNING'
        )

    @pytest.mark.asyncio
    async def test_the_first_service_is_a_live_loop_stall_monitor(
        self, tmp_path: Path,
    ) -> None:
        harness = Harness(OrchestratorConfig(project_root=tmp_path))
        harness._build_lifecycle_registry()
        assert harness._lifecycle is not None
        first = harness._lifecycle.services[0]
        assert isinstance(first, ManagedService)
        assert first.name == 'loop-stall-monitor'

        await first.start()
        try:
            assert len(_live_stall_watchdogs()) == 1
        finally:
            await first.stop()

        assert _live_stall_watchdogs() == []


def _live_stall_watchdogs() -> list[threading.Thread]:
    return [
        t for t in threading.enumerate()
        if t.name == 'loop-stall-watchdog' and t.is_alive()
    ]


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
        sleep-first). The recovery block needs the merge worker constructed,
        not running: its loop launches after finish_startup() (task 8268)."""
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

        fresh = EscalationQueue(escalation_queue_dir(tmp_path))
        [record] = fresh.get_by_task(STALE_DUE_ESCALATION_SENTINEL, status='pending')
        assert record.level == 1
        assert record.category == 'infra_issue'


# ---------------------------------------------------------------------------
# task 8268 (plans/startup-hardening-prd.md §3.4): the merge worker is
# constructed at 1b; its loop is launched only after finish_startup()
# ---------------------------------------------------------------------------

_SCENE_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', remote='origin',
    worktree_dir='.worktrees', push_after_advance=False,
)
#: The startup-table steps run for real here: the merge worker's construction
#: and the recovery that journals into its queue.
_REAL_STARTUP_STEPS = _UNSTUBBED_STEPS | {'1c0a'}


class _ProcessKillingFault(BaseException):
    """Escapes the startup-step runner, which catches only ``Exception``."""


@pytest.fixture
def make_merge_lane_harness(tmp_path: Path) -> Callable[[], Harness]:
    """Build Harnesses over ONE seeded repo, each running the real 1b and 1c0a.

    A later Harness sees the journal and runs.db an earlier one left.
    """
    repo = seed_repo(tmp_path / 'repo')

    def make() -> Harness:
        with patch('orchestrator.harness.McpLifecycle') as mock_mcp_cls, \
             patch('orchestrator.harness.Scheduler'), \
             patch('orchestrator.harness.BriefingAssembler'):
            h = Harness(lane_scene_config(repo, _SCENE_GIT))
        mock_mcp_cls.return_value.start = AsyncMock()
        mock_mcp_cls.return_value.stop = AsyncMock()

        async def start_escalation_server() -> None:
            h._escalation_queue = EscalationQueue(tmp_path / 'escalations')

        h._start_escalation_server = AsyncMock(side_effect=start_escalation_server)
        h._stop_escalation_server = AsyncMock()
        for step, attr in _STUBBED_STEPS:
            if step not in _REAL_STARTUP_STEPS:
                setattr(h, attr, MagicMock() if attr in _SYNC_STEPS else AsyncMock())
        h._tag_task_modules = AsyncMock()
        h._tag_prd_metadata = AsyncMock()

        h.scheduler = MagicMock()
        h.scheduler.get_statuses = AsyncMock(return_value=({'1': 'pending'}, None))
        h.scheduler.set_task_status = AsyncMock()
        return h

    return make


async def journal_merge_request(h: Harness, task_id: str) -> str:
    """Journal a real merge of task *task_id*'s new branch, as the live enqueue does; its id."""
    request = await branch_merge_request(h.git_ops, h.config, task_id, filename=f'{task_id}.txt')
    h._merge_store.record(request)
    return request.request_id


def _drain_queued(h: Harness) -> list:
    queued = []
    while not h._merge_queue.empty():
        queued.append(h._merge_queue.get_nowait())
    return queued


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records]


def _index_of(messages: list[str], fragment: str) -> int:
    return next(i for i, m in enumerate(messages) if fragment in m)


def _merge_worker_loops() -> list[asyncio.Task]:
    return [t for t in asyncio.all_tasks() if t.get_name() == 'merge-worker']


@pytest.mark.asyncio
class TestMergeWorkerLaunchedAfterStartup:
    async def test_a_crash_in_2c_resolves_no_journaled_merge(
        self, make_merge_lane_harness,
    ) -> None:
        h = make_merge_lane_harness()
        request_id = await journal_merge_request(h, '41')
        h._recover_crashed_tasks = AsyncMock(side_effect=_ProcessKillingFault('injected 2c'))

        with pytest.raises(_ProcessKillingFault):
            await h.run(prd_path=None)

        assert h.event_store is not None
        assert h.event_store.fetch_events_by_type(EventType.merge_finalized) == []
        [queued] = _drain_queued(h)
        assert queued.request_id == request_id
        assert not queued.result.done()

        h2 = make_merge_lane_harness()
        await h2.run(prd_path=None, dry_run=True)

        assert [r.request_id for r in _drain_queued(h2)] == [request_id]

    async def test_a_healthy_start_launches_the_loop_after_the_task_tree_is_populated(
        self, make_merge_lane_harness, caplog: pytest.LogCaptureFixture,
    ) -> None:
        h = make_merge_lane_harness()
        loops_during_2c: list[asyncio.Task] = []
        loops_at_first_acquire: list[asyncio.Task] = []

        async def recover_crashed_tasks() -> None:
            loops_during_2c.extend(_merge_worker_loops())

        h._recover_crashed_tasks = recover_crashed_tasks

        async def first_acquire() -> None:
            loops_at_first_acquire.extend(_merge_worker_loops())
            raise RuntimeError('__stop_sentinel__')

        h.scheduler.acquire_next = AsyncMock(side_effect=first_acquire)

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'), \
             pytest.raises(RuntimeError, match='__stop_sentinel__'):
            await h.run(prd_path=None)

        assert loops_during_2c == []
        assert len(loops_at_first_acquire) == 1
        messages = _messages(caplog)
        assert _index_of(messages, 'Task tree populated') < _index_of(
            messages, 'Speculative merge worker started',
        )

    async def test_a_dry_run_start_never_runs_the_loop(
        self, make_merge_lane_harness, caplog: pytest.LogCaptureFixture,
    ) -> None:
        h = make_merge_lane_harness()

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
            await h.run(prd_path=None, dry_run=True)

        assert _merge_worker_loops() == []
        assert not any('Speculative merge worker started' in m for m in _messages(caplog))
        h.scheduler.finish_startup.assert_called_once()

    async def test_stopping_a_constructed_but_unlaunched_merge_worker_resolves_nothing(
        self, make_merge_lane_harness,
    ) -> None:
        h = make_merge_lane_harness()
        h._build_lifecycle_registry()
        assert h._lifecycle is not None
        [service] = [s for s in h._lifecycle.services if s.name == 'merge-worker']
        await service.start()
        assert h._merge_worker is not None
        assert _merge_worker_loops() == []

        with patch.object(h._merge_worker, 'stop', AsyncMock()) as stop, \
             patch.object(h._drain_participant, 'on_shutdown', MagicMock()) as on_shutdown:
            await service.stop()

        stop.assert_not_called()
        on_shutdown.assert_not_called()
