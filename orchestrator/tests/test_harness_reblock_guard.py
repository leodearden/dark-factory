"""Tests for the rapid-re-block guard: slot-setup failure arms requeue cooldown.

Regression test for the issue (reify #2928) where a task with all deps satisfied
was rapid-re-blocked 3 consecutive times in one session, each time with no
block_reason.  The loop was:
  1. Reconciliation resets task to pending.
  2. Orchestrator dispatches task → worktree-create race raises inside workflow.run().
  3. except Exception handler in _run_slot returns BLOCKED without arming cooldown.
  4. scheduler.release(requeued=False) → task immediately eligible for re-dispatch.
  5. Back to step 2.

The fix adds a local slot_setup_failed flag set in the except Exception handler
and ORs it into the requeued argument of the final scheduler.release call,
arming the existing 30 s requeue_cooldown_secs grace window.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _orch_helpers import _init_harness_state_for_test

import orchestrator.harness as harness_module
from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness
from orchestrator.scheduler import Scheduler, TaskAssignment
from orchestrator.workflow import WorkflowOutcome


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=tmp_path)


def _make_harness(config: OrchestratorConfig, scheduler: Scheduler) -> Harness:
    """Build a minimal Harness for driving _run_slot directly.

    Reuses the Harness.__new__ + _init_harness_state_for_test pattern from
    test_retry_cap.py:277-287, wired to a real Scheduler with an injected
    time_source so cooldown assertions can be deterministic.
    """
    harness = Harness.__new__(Harness)
    _init_harness_state_for_test(harness)
    harness.config = config
    harness.scheduler = scheduler
    harness._escalation_queue = None
    harness._escalation_events = {}
    harness._workflow_cancel_events = {}
    harness._workflow_cancel_at = {}
    harness._workflow_slot_tasks = {}
    harness._recovered_plans = {}
    harness._recovered_sessions = {}
    harness._preserved_worktrees = set()
    harness._terminal_cancel_counts = {}
    harness.git_ops = MagicMock()
    harness.briefing = MagicMock()
    harness.mcp = MagicMock()
    harness.usage_gate = MagicMock()
    harness._merge_queue = asyncio.Queue()
    harness._merge_worker = None
    harness.event_store = None
    harness.cost_store = None
    harness._run_id = 'run-test'
    return harness


class _RaisingWorkflow:
    """Stub whose run() raises RuntimeError — worktree-create-race surrogate."""

    def __init__(self, **kwargs):
        pass

    async def run(self):
        raise RuntimeError('worktree-create race surrogate')


class _CancelledWorkflow:
    """Stub whose run() raises CancelledError — hard-cancel surrogate."""

    def __init__(self, **kwargs):
        pass

    async def run(self):
        raise asyncio.CancelledError()


def _pending_task(task_id: str) -> dict:
    return {'id': task_id, 'title': 'test-task', 'status': 'pending', 'dependencies': []}


class TestSlotSetupFailureReblockGuard:
    @pytest.mark.asyncio
    async def test_slot_setup_failure_blocks_immediate_redispatch(
        self, config: OrchestratorConfig, monkeypatch
    ):
        """A slot-setup failure (RuntimeError from workflow.run) must arm the
        requeue cooldown so the task cannot be immediately re-dispatched.

        RED on the base branch (no cooldown armed → task immediately re-dispatchable).
        GREEN after the fix (slot_setup_failed flag ORed into scheduler.release).
        """
        clock = [1000.0]
        scheduler = Scheduler(config, time_source=lambda: clock[0])
        harness = _make_harness(config, scheduler)

        monkeypatch.setattr(harness_module, 'TaskWorkflow', _RaisingWorkflow)

        assignment = TaskAssignment(
            task_id='2928',
            task=_pending_task('2928'),
            modules=['backend'],
        )
        sem = asyncio.Semaphore(1)

        result = await harness._run_slot(assignment, sem)

        # _run_slot returns a synthetic BLOCKED report on slot-setup failure.
        assert result is not None
        assert result.outcome == WorkflowOutcome.BLOCKED

        # The requeue cooldown MUST be armed — this assertion fails on the base branch.
        assert '2928' in scheduler._requeue_until
        deadline = scheduler._requeue_until['2928']
        assert deadline == pytest.approx(1000.0 + config.requeue_cooldown_secs)

        # Task is NOT eligible for re-dispatch while within the grace window.
        task = _pending_task('2928')
        eligible_before, _ = scheduler._eligible_for_dispatch(task, '2928', {})
        assert eligible_before is False

        # Task IS eligible after the deadline elapses.
        clock[0] = deadline
        eligible_after, _ = scheduler._eligible_for_dispatch(task, '2928', {})
        assert eligible_after is True

    @pytest.mark.asyncio
    async def test_hard_cancel_does_not_arm_reblock_cooldown(
        self, config: OrchestratorConfig, monkeypatch
    ):
        """Hard-cancel (CancelledError) must NOT arm the requeue cooldown.

        A hard-cancel is an intentional terminal action, not a transient infra
        failure — arming the cooldown there would delay deliberate stops.
        This test passes on both the base branch and after the fix (scope check).
        """
        clock = [1000.0]
        scheduler = Scheduler(config, time_source=lambda: clock[0])
        harness = _make_harness(config, scheduler)

        monkeypatch.setattr(harness_module, 'TaskWorkflow', _CancelledWorkflow)

        assignment = TaskAssignment(
            task_id='2928',
            task=_pending_task('2928'),
            modules=['backend'],
        )
        sem = asyncio.Semaphore(1)

        result = await harness._run_slot(assignment, sem)

        # _run_slot returns a synthetic CANCELLED report on hard-cancel.
        assert result is not None
        assert result.outcome == WorkflowOutcome.CANCELLED

        # NO cooldown must be armed for a hard-cancel.
        assert '2928' not in scheduler._requeue_until
