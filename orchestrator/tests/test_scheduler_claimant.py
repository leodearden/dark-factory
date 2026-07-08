"""Tests for Scheduler claimant_run_id/heartbeat_at plumbing (task 2188).

PRD plans/task-status-authority-prd.md contract C4/D4 — omega1 threads the
first-class claimant fields through the orchestrator's only path to
tasks.db: Scheduler.dispatch_tool.

Covers:
  - step-9/10: set_task_status forwards claimant_run_id/heartbeat_at into the
    dispatch_tool arguments dict only when supplied (legacy calls unaffected).
  - step-11/12: new Scheduler.set_task_claimant dispatches the dedicated
    'set_task_claimant' tool, preserving the omitted-vs-explicit-None
    distinction, and is best-effort (never raises).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import Scheduler


def _arguments_from(call_args) -> dict:
    """Extract the 'arguments' dict from a dispatch_tool(name, arguments, ...) call."""
    if call_args.args and len(call_args.args) > 1:
        return call_args.args[1]
    return call_args.kwargs.get('arguments', {})


@pytest.fixture
def scheduler() -> Scheduler:
    config = OrchestratorConfig(max_per_module=1)
    return Scheduler(config)


# ---------------------------------------------------------------------------
# step-9/10: set_task_status forwards claimant kwargs
# ---------------------------------------------------------------------------


class TestSetTaskStatusClaimantForwarding:
    @pytest.mark.asyncio
    async def test_supplied_claimant_kwargs_forwarded(self, scheduler: Scheduler):
        """claimant_run_id/heartbeat_at land in the dispatch_tool arguments dict."""
        scheduler.dispatch_tool = AsyncMock(return_value={})  # empty dict -> success
        tid = 'task-claimant-1'

        await scheduler.set_task_status(
            tid, 'in-progress',
            claimant_run_id='X', heartbeat_at='Y',
        )

        scheduler.dispatch_tool.assert_called_once()
        arguments = _arguments_from(scheduler.dispatch_tool.call_args)
        assert arguments.get('id') == tid
        assert arguments.get('status') == 'in-progress'
        assert arguments.get('project_root') == scheduler._project_root
        assert arguments.get('claimant_run_id') == 'X'
        assert arguments.get('heartbeat_at') == 'Y'

    @pytest.mark.asyncio
    async def test_omitted_claimant_kwargs_absent_from_arguments(self, scheduler: Scheduler):
        """Legacy call (no claimant kwargs) must not carry those keys at all."""
        scheduler.dispatch_tool = AsyncMock(return_value={})
        tid = 'task-claimant-2'

        await scheduler.set_task_status(tid, 'in-progress')

        scheduler.dispatch_tool.assert_called_once()
        arguments = _arguments_from(scheduler.dispatch_tool.call_args)
        assert 'claimant_run_id' not in arguments
        assert 'heartbeat_at' not in arguments

    @pytest.mark.asyncio
    async def test_claimant_kwargs_resent_on_transient_retry(self, scheduler: Scheduler):
        """Injected before the retry loop -> every retry attempt resends them."""
        transient = {'error': 'TimeoutError(mcp down)', 'error_type': 'TimeoutError'}
        scheduler.dispatch_tool = AsyncMock(side_effect=[transient, {}])
        tid = 'task-claimant-retry'

        await scheduler.set_task_status(
            tid, 'in-progress',
            claimant_run_id='X', heartbeat_at='Y',
        )

        assert scheduler.dispatch_tool.call_count == 2
        for call in scheduler.dispatch_tool.call_args_list:
            arguments = _arguments_from(call)
            assert arguments.get('claimant_run_id') == 'X'
            assert arguments.get('heartbeat_at') == 'Y'


# ---------------------------------------------------------------------------
# step-11/12: new Scheduler.set_task_claimant
# ---------------------------------------------------------------------------


class TestSchedulerSetTaskClaimant:
    @pytest.mark.asyncio
    async def test_string_kwargs_dispatch_set_task_claimant_tool(self, scheduler: Scheduler):
        scheduler.dispatch_tool = AsyncMock(return_value={})
        tid = 'task-9'

        await scheduler.set_task_claimant(
            tid, claimant_run_id='run-x/session-y/pid=1', heartbeat_at='2026-07-08T00:00:00+00:00',
        )

        scheduler.dispatch_tool.assert_called_once()
        name = scheduler.dispatch_tool.call_args.args[0]
        assert name == 'set_task_claimant'
        arguments = _arguments_from(scheduler.dispatch_tool.call_args)
        assert arguments.get('id') == tid
        assert arguments.get('project_root') == scheduler._project_root
        assert arguments.get('claimant_run_id') == 'run-x/session-y/pid=1'
        assert arguments.get('heartbeat_at') == '2026-07-08T00:00:00+00:00'

    @pytest.mark.asyncio
    async def test_heartbeat_only_refresh_omits_claimant_run_id(self, scheduler: Scheduler):
        """A heartbeat-only refresh must not send claimant_run_id at all."""
        scheduler.dispatch_tool = AsyncMock(return_value={})
        tid = 'task-10'

        await scheduler.set_task_claimant(tid, heartbeat_at='2026-07-08T00:00:00+00:00')

        arguments = _arguments_from(scheduler.dispatch_tool.call_args)
        assert 'claimant_run_id' not in arguments
        assert arguments.get('heartbeat_at') == '2026-07-08T00:00:00+00:00'

    @pytest.mark.asyncio
    async def test_clear_forwards_explicit_none_for_both(self, scheduler: Scheduler):
        """Release clear: explicit None for both fields must be SENT, not omitted."""
        scheduler.dispatch_tool = AsyncMock(return_value={})
        tid = 'task-11'

        await scheduler.set_task_claimant(tid, claimant_run_id=None, heartbeat_at=None)

        arguments = _arguments_from(scheduler.dispatch_tool.call_args)
        assert 'claimant_run_id' in arguments
        assert arguments['claimant_run_id'] is None
        assert 'heartbeat_at' in arguments
        assert arguments['heartbeat_at'] is None

    @pytest.mark.asyncio
    async def test_no_kwargs_supplied_omits_both(self, scheduler: Scheduler):
        scheduler.dispatch_tool = AsyncMock(return_value={})
        tid = 'task-12'

        await scheduler.set_task_claimant(tid)

        arguments = _arguments_from(scheduler.dispatch_tool.call_args)
        assert 'claimant_run_id' not in arguments
        assert 'heartbeat_at' not in arguments

    @pytest.mark.asyncio
    async def test_best_effort_swallows_dispatch_tool_exception(self, scheduler: Scheduler):
        """A raised dispatch_tool error must NOT propagate — heartbeat/clear is best-effort."""
        scheduler.dispatch_tool = AsyncMock(side_effect=RuntimeError('mcp down'))
        tid = 'task-13'

        # Must not raise.
        await scheduler.set_task_claimant(tid, claimant_run_id=None, heartbeat_at=None)

        scheduler.dispatch_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_best_effort_swallows_rejection_response(self, scheduler: Scheduler):
        """An error-shaped dispatch_tool response must also not raise."""
        scheduler.dispatch_tool = AsyncMock(
            return_value={'error': 'boom', 'error_type': 'RuntimeError'},
        )
        tid = 'task-14'

        await scheduler.set_task_claimant(tid, heartbeat_at='2026-07-08T00:00:00+00:00')

        scheduler.dispatch_tool.assert_called_once()
