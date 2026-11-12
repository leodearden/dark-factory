"""Startup-step fault injection through the real ``Harness.run``.

Every startup step runs through ``orchestrator.startup_steps.run_startup_step``:
a failing non-fatal step leaves startup running, emits one
``startup_step_failed`` event and files (or refreshes) one L1 on
``orchestrator-startup-<step>``; the fatal step 1b still refuses startup.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.artifacts import AgentSession
from orchestrator.config import TranscriptArchiveConfig
from orchestrator.event_store import EventType
from orchestrator.harness import Harness
from orchestrator.merge_lane.liveness import MergeLivenessConfigError
from orchestrator.startup_steps import (
    STARTUP_STEP_SENTINEL_PREFIX,
    EntryFailures,
    run_startup_step,
    sentinel_task_id,
)

#: The stubbed table steps in run() order, as (step name, Harness attribute).
_STUBBED_STEPS: tuple[tuple[str, str], ...] = (
    ('1c', '_dismiss_stale_escalations'),
    ('1c0-dirty', '_file_dirty_tree_escalation'),
    ('1c0-config-keys', '_file_config_unknown_keys_escalation'),
    ('1c0', '_rehydrate_merge_halt'),
    ('1c0a', '_recover_pending_merges'),
    ('1c0a2', '_reap_orphaned_merge_worktrees'),
    ('1c0a3', '_reconcile_landed_outbox'),
    ('1c0b', '_file_restored_pause_escalation'),
    ('1c1g', '_run_interactive_worktree_reaper_pass'),
    ('1c1h', '_run_leftover_verify_scope_reaper_pass'),
    ('2c', '_recover_crashed_tasks'),
    ('2c2', '_reconcile_lane_checkouts'),
    ('2c3-streak-alarms', '_inherit_pending_streak_alarms'),
    ('2d', '_reconcile_stranded_in_progress'),
    ('2e', '_reap_orphan_worktrees'),
    ('diff7', '_reconcile_terminal_lanes'),
    ('2g', '_run_scheduled_deploy_settle_pass'),
)
_SYNC_STEPS = frozenset({
    '_rehydrate_merge_halt', '_file_restored_pause_escalation',
    '_inherit_pending_streak_alarms',
})
#: Table steps the fixture runs for real against its mocked services.
_UNSTUBBED_STEPS = frozenset({'1b', '2f'})

_INJECTABLE = [
    ('2c', '_recover_crashed_tasks'),
    ('2c2', '_reconcile_lane_checkouts'),
    ('2d', '_reconcile_stranded_in_progress'),
    ('2e', '_reap_orphan_worktrees'),
    ('diff7', '_reconcile_terminal_lanes'),
    ('1c0a', '_recover_pending_merges'),
    ('1c', '_dismiss_stale_escalations'),
]


@dataclass(frozen=True)
class StartupHarness:
    """A Harness whose run() reaches finish_startup, with the mocks installed on it."""

    harness: Harness
    scheduler: MagicMock
    stubs: Mapping[str, AsyncMock]
    sync_stubs: Mapping[str, MagicMock]


def _stub_table_steps(h: Harness) -> tuple[dict[str, AsyncMock], dict[str, MagicMock]]:
    stubs = {attr: AsyncMock() for _step, attr in _STUBBED_STEPS if attr not in _SYNC_STEPS}
    stubs['_recover_pending_merges'] = AsyncMock(return_value={'requests': []})
    sync_stubs = {attr: MagicMock() for attr in _SYNC_STEPS}
    for attr, stub in {**stubs, **sync_stubs}.items():
        setattr(h, attr, stub)
    h._tag_task_modules = AsyncMock()
    h._tag_prd_metadata = AsyncMock()
    return stubs, sync_stubs


@pytest.fixture
def make_startup_harness(
    tmp_path: Path, mock_orch_config: MagicMock,
) -> Callable[[], StartupHarness]:
    """Build Harnesses whose run() reaches finish_startup over one shared tmp_path."""
    mock_orch_config.max_concurrent_tasks = 2
    mock_orch_config.fused_memory.project_id = 'test'

    def make() -> StartupHarness:
        with patch('orchestrator.harness.McpLifecycle') as mock_mcp_cls, \
             patch('orchestrator.harness.Scheduler'), \
             patch('orchestrator.harness.BriefingAssembler'):
            h = Harness(mock_orch_config)
        mock_mcp_cls.return_value.start = AsyncMock()
        mock_mcp_cls.return_value.stop = AsyncMock()

        h.git_ops = MagicMock()
        h.git_ops.worktree_base = tmp_path / '.worktrees'
        h.git_ops.has_dirty_working_tree = AsyncMock(return_value=None)
        h.git_ops.disable_shared_repo_auto_maintenance = AsyncMock()

        async def start_escalation_server() -> None:
            h._escalation_queue = EscalationQueue(tmp_path / 'escalations')

        h._start_escalation_server = AsyncMock(side_effect=start_escalation_server)
        h._stop_escalation_server = AsyncMock()
        h._start_merge_worker = AsyncMock()
        h._stop_merge_worker = AsyncMock()
        stubs, sync_stubs = _stub_table_steps(h)

        scheduler = MagicMock()
        scheduler.get_statuses = AsyncMock(return_value=({'1': 'pending'}, None))
        h.scheduler = scheduler
        return StartupHarness(h, scheduler, stubs, sync_stubs)

    return make


async def _run(h: Harness) -> None:
    await h.run(prd_path=None, dry_run=True)


def _queue(h: Harness) -> EscalationQueue:
    assert h._escalation_queue is not None
    return h._escalation_queue


def _startup_l1s(queue: EscalationQueue) -> list[Escalation]:
    return [
        esc for esc in queue.get_pending()
        if esc.level == 1 and esc.task_id.startswith(STARTUP_STEP_SENTINEL_PREFIX)
    ]


def _failed_events(h: Harness) -> list[dict]:
    store = h.event_store
    assert store is not None
    return store.fetch_events_by_type(EventType.startup_step_failed)


def _inject(handle: StartupHarness, attr: str, exc: Exception) -> AsyncMock:
    injected = AsyncMock(side_effect=exc)
    setattr(handle.harness, attr, injected)
    return injected


def _steps_after(step: str) -> list[str]:
    names = [name for name, _attr in _STUBBED_STEPS]
    return [attr for name, attr in _STUBBED_STEPS[names.index(step) + 1:]]


def _not_ok_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == 'orchestrator.harness' and r.levelno == logging.WARNING
        and 'startup steps not ok' in r.getMessage()
    ]


@pytest.mark.asyncio
class TestStubbedStepsMirrorTheTable:
    async def test_every_other_table_step_is_stubbed_in_run_order(
        self, make_startup_harness,
    ) -> None:
        handle = make_startup_harness()
        runner = AsyncMock(wraps=run_startup_step)

        with patch('orchestrator.harness.run_startup_step', runner):
            await _run(handle.harness)

        ran = [call.args[0].name for call in runner.await_args_list]
        assert [name for name in ran if name not in _UNSTUBBED_STEPS] == [
            name for name, _attr in _STUBBED_STEPS
        ]


@pytest.mark.asyncio
class TestNonFatalStepFailure:
    @pytest.mark.parametrize(('step', 'attr'), _INJECTABLE)
    async def test_failure_is_reported_and_startup_continues(
        self, make_startup_harness, step: str, attr: str,
    ) -> None:
        handle = make_startup_harness()
        _inject(handle, attr, RuntimeError(f'injected {step}'))

        await _run(handle.harness)

        handle.scheduler.finish_startup.assert_called_once()
        [event] = _failed_events(handle.harness)
        assert event['data']['step'] == step
        assert event['data']['status'] == 'failed'
        [esc] = _startup_l1s(_queue(handle.harness))
        assert esc.task_id == sentinel_task_id(step)
        installed = {**handle.stubs, **handle.sync_stubs}
        for later in _steps_after(step):
            assert installed[later].called, f'{later} must still run after {step} failed'

    @pytest.mark.parametrize(('step', 'attr'), _INJECTABLE)
    async def test_repeat_failure_on_the_next_start_refreshes_the_l1(
        self, make_startup_harness, step: str, attr: str,
    ) -> None:
        first = make_startup_harness()
        _inject(first, attr, RuntimeError(f'injected {step}'))
        await _run(first.harness)
        [filed] = _startup_l1s(_queue(first.harness))

        second = make_startup_harness()
        _inject(second, attr, RuntimeError(f'injected {step}'))
        await _run(second.harness)

        [refreshed] = _startup_l1s(_queue(second.harness))
        assert refreshed.id == filed.id
        assert refreshed.dedupe_count == 1


@pytest.mark.asyncio
class TestRecoveredMergesReachTheReaper:
    async def test_failed_recovery_still_runs_the_reaper_with_no_requests(
        self, make_startup_harness,
    ) -> None:
        handle = make_startup_harness()
        _inject(handle, '_recover_pending_merges', RuntimeError('injected 1c0a'))

        await _run(handle.harness)

        handle.stubs['_reap_orphaned_merge_worktrees'].assert_awaited_once_with([])

    async def test_recovered_requests_reach_the_reaper(self, make_startup_harness) -> None:
        handle = make_startup_harness()
        request = object()
        handle.harness._recover_pending_merges = AsyncMock(return_value={'requests': [request]})

        await _run(handle.harness)

        handle.stubs['_reap_orphaned_merge_worktrees'].assert_awaited_once_with([request])


@pytest.mark.asyncio
class TestPartialStep:
    async def test_entry_failures_are_reported_as_partial(self, make_startup_harness) -> None:
        handle = make_startup_harness()
        h = handle.harness
        failures = EntryFailures('Recovery')
        failures.record('41', OSError('boom'))
        h._recover_crashed_tasks = AsyncMock(return_value=failures)

        await _run(h)

        [event] = _failed_events(h)
        assert event['data']['status'] == 'partial'
        assert [e['entry'] for e in event['data']['failed_entries']] == ['41']
        [esc] = _startup_l1s(_queue(h))
        assert esc.task_id == sentinel_task_id('2c')
        assert '41' in esc.summary or '41' in esc.detail
        handle.scheduler.finish_startup.assert_called_once()


@pytest.mark.asyncio
class TestFatalStep:
    async def test_merge_liveness_refusal_still_refuses_startup(
        self, make_startup_harness,
    ) -> None:
        handle = make_startup_harness()
        h = handle.harness
        h._build_lifecycle_registry()
        assert h._lifecycle is not None
        refusal = MergeLivenessConfigError('over budget')
        h._lifecycle.start_all = AsyncMock(side_effect=refusal)
        h._lifecycle.stop_all = AsyncMock()

        with pytest.raises(MergeLivenessConfigError) as raised:
            await _run(h)

        assert raised.value is refusal
        handle.scheduler.finish_startup.assert_not_called()
        [event] = _failed_events(h)
        assert event['data']['step'] == '1b'


@pytest.mark.asyncio
class TestSummaryWarning:
    async def test_healthy_start_reports_nothing(self, make_startup_harness, caplog) -> None:
        h = make_startup_harness().harness

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await _run(h)

        assert _failed_events(h) == []
        assert _startup_l1s(_queue(h)) == []
        assert _not_ok_warnings(caplog) == []

    async def test_failing_start_logs_one_summary_naming_the_step(
        self, make_startup_harness, caplog,
    ) -> None:
        handle = make_startup_harness()
        _inject(handle, '_recover_crashed_tasks', RuntimeError('injected 2c'))

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await _run(handle.harness)

        [summary] = _not_ok_warnings(caplog)
        assert '2c=failed' in summary


@pytest.mark.asyncio
class TestStrandedSweepAtStartup:
    async def test_startup_runs_2d_with_a_collector(self, make_startup_harness) -> None:
        handle = make_startup_harness()

        await _run(handle.harness)

        [call] = handle.stubs['_reconcile_stranded_in_progress'].await_args_list
        assert isinstance(call.kwargs.get('failures'), EntryFailures)
        assert call.kwargs.get('mid_run', False) is False


def _seed_dead_owner_sidecar(worktree_base: Path, task_id: str) -> Path:
    sidecar = worktree_base / task_id / '.task' / 'agent_session.json'
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps(AgentSession(
        session_id=f'sess-{task_id}',
        role='implementer',
        started_at='2026-11-01T00:00:00+00:00',
        owner_pid=2**31 - 1,
        task_id=task_id,
    ).to_dict()))
    return sidecar


@pytest.mark.asyncio
class TestZombieSidecarReaperSiting:
    """The sidecar reaper reads pre-2d ground truth (plans/session-resume-eligibility-seam-prd.capability-manifest.md §F4)."""

    async def test_reaper_runs_in_2c_before_2d_re_pends_stranded_tasks(
        self, make_startup_harness,
    ) -> None:
        handle = make_startup_harness()
        h = handle.harness
        del h._recover_crashed_tasks
        h.config.transcript_archive = TranscriptArchiveConfig()
        h.git_ops.worktree_base.mkdir(parents=True)
        h.git_ops.warm_lane_pool = None
        h.git_ops.spec_warm_lane_pool = None
        h.git_ops.pool_in_use = MagicMock(return_value=False)
        h.git_ops.cleanup_worktree = AsyncMock()
        stranded = _seed_dead_owner_sidecar(h.git_ops.worktree_base, '7')
        zombie = _seed_dead_owner_sidecar(h.git_ops.worktree_base, '8')
        statuses = {'1': 'pending', '7': 'in-progress', '8': 'pending'}
        handle.scheduler.get_statuses = AsyncMock(
            side_effect=lambda ids=None: (
                {k: v for k, v in statuses.items() if ids is None or k in ids}, None,
            ),
        )
        handle.stubs['_reconcile_stranded_in_progress'].side_effect = (
            lambda *a, **k: statuses.update({'7': 'pending'})
        )

        await _run(h)

        assert stranded.exists()
        assert not zombie.exists()
        assert h.event_store is not None
        reaped = h.event_store.fetch_events_by_type(EventType.sidecar_reaped)
        assert [row['task_id'] for row in reaped] == ['8']
