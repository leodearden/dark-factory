"""Tests for the claimant clear in ``_run_slot``'s finally block.

PRD plans/task-status-authority-prd.md contract C4/D4 (task 2188, omega1).

step-17/18: at slot teardown (``_run_slot``'s finally, harness.py, just
before ``scheduler.release``), the harness clears claimant_run_id/
heartbeat_at to NULL via
``scheduler.set_task_claimant(task_id, claimant_run_id=None, heartbeat_at=None)``
— unconditionally, for every outcome (terminal DONE/CANCELLED and
non-terminal BLOCKED/REQUEUED alike) — and the clear is best-effort: a raise
from ``set_task_claimant`` must not prevent ``scheduler.release``/
``sem.release`` from still running (slot teardown unaffected).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import _init_harness_state_for_test

import orchestrator.harness as harness_module
from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness
from orchestrator.merge_queue import InFlightMergeRegistry
from orchestrator.scheduler import Scheduler, TaskAssignment
from orchestrator.workflow import TerminalReport, WorkflowOutcome, WorkflowState


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=tmp_path)


def _make_harness(config: OrchestratorConfig, scheduler: Scheduler) -> Harness:
    """Build a minimal Harness for driving _run_slot directly.

    Mirrors test_harness_reblock_guard.py's ``_make_harness`` (Harness.__new__
    + _init_harness_state_for_test, wired to a real Scheduler).
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
    harness.git_ops.release_lane_for_terminal_task = AsyncMock()
    harness.briefing = MagicMock()
    harness.mcp = MagicMock()
    harness.usage_gate = MagicMock()
    harness._merge_queue = asyncio.Queue()
    harness._merge_worker = None
    harness._merge_inflight_registry = InFlightMergeRegistry()
    harness.event_store = None
    harness.cost_store = None
    harness._run_id = 'run-test'
    return harness


_OUTCOME_TO_PHASE = {
    WorkflowOutcome.DONE: WorkflowState.DONE,
    WorkflowOutcome.CANCELLED: WorkflowState.CANCELLED,
    WorkflowOutcome.BLOCKED: WorkflowState.BLOCKED,
    WorkflowOutcome.REQUEUED: WorkflowState.PLAN,
}


def _make_stub_workflow_class(outcome: WorkflowOutcome) -> type:
    """Build a stub TaskWorkflow replacement whose run() returns ``outcome``.

    Stands in for the real TaskWorkflow (patched onto orchestrator.harness)
    so this file exercises only the harness-level clear, independent of the
    workflow-level stamp/heartbeat covered by test_workflow_claimant.py.
    """

    class _StubWorkflow:
        def __init__(self, **kwargs):
            pass

        async def run(self) -> TerminalReport:
            return TerminalReport(
                outcome=outcome, reason='',
                phase=_OUTCOME_TO_PHASE.get(outcome, WorkflowState.PLAN),
                detail='', category=None,
            )

        _steward = None
        metrics = MagicMock(
            total_cost_usd=0.0,
            total_duration_ms=0,
            agent_invocations=0,
            execute_iterations=0,
            verify_attempts=0,
            review_cycles=0,
        )

    return _StubWorkflow


def _pending_task(task_id: str) -> dict:
    return {'id': task_id, 'title': 'test-task', 'status': 'pending', 'dependencies': []}


def _make_scheduler(config: OrchestratorConfig) -> Scheduler:
    """A real Scheduler (like test_harness_reblock_guard.py) so the retry-cap
    / release bookkeeping _run_slot's finally exercises works without hand
    -stubbing every internal method."""
    return Scheduler(config, time_source=lambda: 1000.0)


class TestClaimantClearedOnEveryOutcome:
    """scheduler.set_task_claimant(task_id, claimant_run_id=None, heartbeat_at=None)
    fires exactly once from _run_slot's finally, regardless of outcome."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'outcome',
        [
            WorkflowOutcome.DONE,
            WorkflowOutcome.CANCELLED,
            WorkflowOutcome.BLOCKED,
            WorkflowOutcome.REQUEUED,
        ],
        ids=['done', 'cancelled', 'blocked', 'requeued'],
    )
    async def test_claimant_cleared_regardless_of_outcome(
        self, config: OrchestratorConfig, monkeypatch, outcome: WorkflowOutcome,
    ):
        scheduler = _make_scheduler(config)
        scheduler.set_task_claimant = AsyncMock(return_value=None)
        harness = _make_harness(config, scheduler)

        monkeypatch.setattr(
            harness_module, 'TaskWorkflow', _make_stub_workflow_class(outcome),
        )

        task_id = f'claimant-clear-{outcome.value}'
        assignment = TaskAssignment(
            task_id=task_id, task=_pending_task(task_id), modules=['backend'],
        )
        sem = asyncio.Semaphore(1)

        report = await harness._run_slot(assignment, sem)

        assert report is not None
        assert report.outcome == outcome
        scheduler.set_task_claimant.assert_awaited_once_with(
            task_id, claimant_run_id=None, heartbeat_at=None,
        )


class TestClaimantClearIsBestEffort:
    """A raise from set_task_claimant must not break slot teardown."""

    @pytest.mark.asyncio
    async def test_claimant_clear_failure_does_not_block_release(
        self, config: OrchestratorConfig, monkeypatch,
    ):
        scheduler = _make_scheduler(config)
        scheduler.set_task_claimant = AsyncMock(
            side_effect=RuntimeError('fused-memory unreachable'),
        )
        # Spy (not replace) so the real release bookkeeping still runs while
        # we can assert it was actually invoked.
        scheduler.release = MagicMock(wraps=scheduler.release)
        harness = _make_harness(config, scheduler)

        monkeypatch.setattr(
            harness_module, 'TaskWorkflow',
            _make_stub_workflow_class(WorkflowOutcome.DONE),
        )

        task_id = 'claimant-clear-raises'
        assignment = TaskAssignment(
            task_id=task_id, task=_pending_task(task_id), modules=['backend'],
        )
        sem = MagicMock()
        sem.release = MagicMock()

        # Must not raise — the harness-level suppress must swallow this.
        report = await harness._run_slot(assignment, sem)

        assert report is not None
        assert report.outcome == WorkflowOutcome.DONE
        scheduler.set_task_claimant.assert_awaited_once_with(
            task_id, claimant_run_id=None, heartbeat_at=None,
        )
        scheduler.release.assert_called_once()
        sem.release.assert_called_once()
