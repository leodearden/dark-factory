"""Startup-step fault injection through the real ``Harness.run``.

Every startup step runs through ``orchestrator.startup_steps.run_startup_step``:
a failing non-fatal step leaves startup running, emits one
``startup_step_failed`` event and files (or refreshes) one L1 on
``orchestrator-startup-<step>``; the fatal step 1b still refuses startup.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.event_store import EventStore, EventType
from orchestrator.harness import Harness
from orchestrator.merge_lane.liveness import MergeLivenessConfigError
from orchestrator.startup_steps import (
    STARTUP_STEP_SENTINEL_PREFIX,
    EntryFailures,
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
    ('2d', '_reconcile_stranded_in_progress'),
    ('2e', '_reap_orphan_worktrees'),
    ('diff7', '_reconcile_terminal_lanes'),
)
_SYNC_STEPS = frozenset({'_rehydrate_merge_halt', '_file_restored_pause_escalation'})

_INJECTABLE = [
    ('2c', '_recover_crashed_tasks'),
    ('2c2', '_reconcile_lane_checkouts'),
    ('2d', '_reconcile_stranded_in_progress'),
    ('2e', '_reap_orphan_worktrees'),
    ('diff7', '_reconcile_terminal_lanes'),
    ('1c0a', '_recover_pending_merges'),
    ('1c', '_dismiss_stale_escalations'),
]


def _stub_table_steps(h: Harness) -> None:
    for _step, attr in _STUBBED_STEPS:
        setattr(h, attr, MagicMock() if attr in _SYNC_STEPS else AsyncMock())
    h._recover_pending_merges = AsyncMock(return_value={'requests': []})
    h._tag_task_modules = AsyncMock()
    h._tag_prd_metadata = AsyncMock()


@pytest.fixture
def make_startup_harness(tmp_path: Path, mock_orch_config: MagicMock) -> Callable[[], Harness]:
    """Build Harnesses whose run() reaches finish_startup over one shared tmp_path."""
    mock_orch_config.max_concurrent_tasks = 2
    mock_orch_config.fused_memory.project_id = 'test'

    def make() -> Harness:
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
        _stub_table_steps(h)

        h.scheduler = MagicMock()
        h.scheduler.get_statuses = AsyncMock(return_value=({'1': 'pending'}, None))
        return h

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
    return cast(EventStore, h.event_store).fetch_events_by_type(EventType.startup_step_failed)


def _inject(h: Harness, attr: str, exc: Exception) -> None:
    setattr(h, attr, AsyncMock(side_effect=exc))


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
class TestNonFatalStepFailure:
    @pytest.mark.parametrize(('step', 'attr'), _INJECTABLE)
    async def test_failure_is_reported_and_startup_continues(
        self, make_startup_harness, step: str, attr: str,
    ) -> None:
        h = make_startup_harness()
        _inject(h, attr, RuntimeError(f'injected {step}'))

        await _run(h)

        cast(MagicMock, h.scheduler.finish_startup).assert_called_once()
        [event] = _failed_events(h)
        assert event['data']['step'] == step
        assert event['data']['status'] == 'failed'
        [esc] = _startup_l1s(_queue(h))
        assert esc.task_id == sentinel_task_id(step)
        for later in _steps_after(step):
            assert getattr(h, later).called, f'{later} must still run after {step} failed'

    @pytest.mark.parametrize(('step', 'attr'), _INJECTABLE)
    async def test_repeat_failure_on_the_next_start_refreshes_the_l1(
        self, make_startup_harness, step: str, attr: str,
    ) -> None:
        first = make_startup_harness()
        _inject(first, attr, RuntimeError(f'injected {step}'))
        await _run(first)
        [filed] = _startup_l1s(_queue(first))

        second = make_startup_harness()
        _inject(second, attr, RuntimeError(f'injected {step}'))
        await _run(second)

        [refreshed] = _startup_l1s(_queue(second))
        assert refreshed.id == filed.id
        assert refreshed.dedupe_count == 1


@pytest.mark.asyncio
class TestRecoveredMergesReachTheReaper:
    async def test_failed_recovery_still_runs_the_reaper_with_no_requests(
        self, make_startup_harness,
    ) -> None:
        h = make_startup_harness()
        _inject(h, '_recover_pending_merges', RuntimeError('injected 1c0a'))

        await _run(h)

        cast(AsyncMock, h._reap_orphaned_merge_worktrees).assert_awaited_once_with([])

    async def test_recovered_requests_reach_the_reaper(self, make_startup_harness) -> None:
        h = make_startup_harness()
        request = object()
        h._recover_pending_merges = AsyncMock(return_value={'requests': [request]})

        await _run(h)

        cast(AsyncMock, h._reap_orphaned_merge_worktrees).assert_awaited_once_with([request])


@pytest.mark.asyncio
class TestPartialStep:
    async def test_entry_failures_are_reported_as_partial(self, make_startup_harness) -> None:
        h = make_startup_harness()
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
        cast(MagicMock, h.scheduler.finish_startup).assert_called_once()


@pytest.mark.asyncio
class TestFatalStep:
    async def test_merge_liveness_refusal_still_refuses_startup(
        self, make_startup_harness,
    ) -> None:
        h = make_startup_harness()
        h._build_lifecycle_registry()
        assert h._lifecycle is not None
        refusal = MergeLivenessConfigError('over budget')
        h._lifecycle.start_all = AsyncMock(side_effect=refusal)
        h._lifecycle.stop_all = AsyncMock()

        with pytest.raises(MergeLivenessConfigError) as raised:
            await _run(h)

        assert raised.value is refusal
        cast(MagicMock, h.scheduler.finish_startup).assert_not_called()
        [event] = _failed_events(h)
        assert event['data']['step'] == '1b'


@pytest.mark.asyncio
class TestSummaryWarning:
    async def test_healthy_start_reports_nothing(self, make_startup_harness, caplog) -> None:
        h = make_startup_harness()

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await _run(h)

        assert _failed_events(h) == []
        assert _startup_l1s(_queue(h)) == []
        assert _not_ok_warnings(caplog) == []

    async def test_failing_start_logs_one_summary_naming_the_step(
        self, make_startup_harness, caplog,
    ) -> None:
        h = make_startup_harness()
        _inject(h, '_recover_crashed_tasks', RuntimeError('injected 2c'))

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await _run(h)

        [summary] = _not_ok_warnings(caplog)
        assert '2c=failed' in summary


@pytest.mark.asyncio
class TestStrandedSweepAtStartup:
    async def test_startup_runs_2d_with_a_collector(self, make_startup_harness) -> None:
        h = make_startup_harness()

        await _run(h)

        reconcile = cast(AsyncMock, h._reconcile_stranded_in_progress)
        [call] = reconcile.await_args_list
        assert isinstance(call.kwargs.get('failures'), EntryFailures)
        assert call.kwargs.get('mid_run', False) is False
