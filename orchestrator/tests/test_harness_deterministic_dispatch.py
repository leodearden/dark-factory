"""Tests for Harness deterministic dispatch (β).

Step-9:  RED — _run_slot routes deterministic tasks to DeterministicRunner
         (not TaskWorkflow); e2e resume→done; dep-satisfied after gate done.
Step-11: RED — restart action clears gate stamps; park/abandon do NOT clear.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.harness import Harness
from orchestrator.scheduler import Scheduler, TaskAssignment
from orchestrator.workflow import WorkflowOutcome


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def harness(tmp_path: Path, mock_orch_config) -> Harness:
    """Harness with mocked internals for deterministic-dispatch unit testing.

    Reuses the same fixture pattern as test_harness_action_dispatch.py:
    patch McpLifecycle/OverrideStore/Scheduler/BriefingAssembler, then
    replace scheduler with an AsyncMock-wired MagicMock.
    """
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    # Async-mock scheduler
    h.scheduler = MagicMock()
    h.scheduler.is_deterministic = MagicMock(return_value=False)  # overridden per test
    h.scheduler.get_status = AsyncMock(return_value='blocked')
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.get_task = AsyncMock(return_value={'id': 'gate-1', 'metadata': {}})
    h.scheduler.update_task = AsyncMock(return_value=True)
    h.scheduler.release = MagicMock()

    # _merge_worker stays None — unhalt branch skipped
    return h


def _det_task(
    task_id: str = 'gate-1',
    title: str = 'Ship gate',
    gate_escalated_at: str | None = None,
) -> dict:
    """Build a minimal deterministic gate task dict."""
    metadata: dict = {
        'task_kind': 'deterministic',
        'always_escalates': True,
        'before_done': None,
    }
    if gate_escalated_at is not None:
        metadata['gate_escalated_at'] = gate_escalated_at
    return {'id': task_id, 'title': title, 'description': 'Gate desc', 'metadata': metadata}


def _make_assignment(task: dict) -> TaskAssignment:
    """Build a TaskAssignment for a deterministic (no module locks) task."""
    return TaskAssignment(task_id=str(task['id']), task=task, modules=[])


def _make_sem() -> asyncio.Semaphore:
    return asyncio.Semaphore(1)


# ---------------------------------------------------------------------------
# Step-9: _run_slot routing (B2/I4)
# (RED until step-10 adds _run_slot branch + _run_deterministic_slot)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRunSlotRouting:
    """_run_slot routes deterministic tasks to DeterministicRunner, not TaskWorkflow."""

    async def test_taskworkflow_not_instantiated_for_det_task(
        self, harness: Harness, tmp_path: Path
    ):
        """_run_slot must NOT create a TaskWorkflow for a deterministic task (I4/B2)."""
        task = _det_task()
        assignment = _make_assignment(task)

        # Make scheduler.is_deterministic return True for this task
        harness.scheduler.is_deterministic = MagicMock(return_value=True)
        harness._escalation_queue = EscalationQueue(tmp_path)

        with (
            patch('orchestrator.harness.TaskWorkflow') as mock_tw,
            patch('orchestrator.harness.DeterministicRunner') as mock_dr,
        ):
            # DeterministicRunner.run() returns BLOCKED
            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
            mock_dr.return_value = mock_runner

            report = await harness._run_slot(assignment, _make_sem())

        # TaskWorkflow must never have been instantiated
        mock_tw.assert_not_called()

    async def test_run_slot_det_task_returns_blocked_report(
        self, harness: Harness, tmp_path: Path
    ):
        """_run_slot for a deterministic task must return a TaskReport with outcome=BLOCKED."""
        task = _det_task()
        assignment = _make_assignment(task)

        harness.scheduler.is_deterministic = MagicMock(return_value=True)
        harness._escalation_queue = EscalationQueue(tmp_path)

        with patch('orchestrator.harness.DeterministicRunner') as mock_dr:
            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
            mock_dr.return_value = mock_runner

            report = await harness._run_slot(assignment, _make_sem())

        assert report is not None
        assert report.outcome == WorkflowOutcome.BLOCKED

    async def test_run_slot_det_task_blocked_report_reason(
        self, harness: Harness, tmp_path: Path
    ):
        """BLOCKED report from a deterministic slot must have block_reason='deterministic_gate'."""
        task = _det_task()
        assignment = _make_assignment(task)

        harness.scheduler.is_deterministic = MagicMock(return_value=True)
        harness._escalation_queue = EscalationQueue(tmp_path)

        with patch('orchestrator.harness.DeterministicRunner') as mock_dr:
            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
            mock_dr.return_value = mock_runner

            report = await harness._run_slot(assignment, _make_sem())

        assert report.block_reason == 'deterministic_gate'

    async def test_run_slot_det_task_done_report_on_resume(
        self, harness: Harness, tmp_path: Path
    ):
        """Second dispatch with gate_escalated_at set and no open esc → DONE report (B4)."""
        task = _det_task(gate_escalated_at='2026-06-23T12:00:00+00:00')
        assignment = _make_assignment(task)

        harness.scheduler.is_deterministic = MagicMock(return_value=True)
        harness._escalation_queue = EscalationQueue(tmp_path)  # empty — no pending esc

        with patch('orchestrator.harness.DeterministicRunner') as mock_dr:
            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value=WorkflowOutcome.DONE)
            mock_dr.return_value = mock_runner

            report = await harness._run_slot(assignment, _make_sem())

        assert report is not None
        assert report.outcome == WorkflowOutcome.DONE

    async def test_run_slot_normal_task_still_uses_taskworkflow(
        self, harness: Harness, tmp_path: Path
    ):
        """Non-deterministic tasks still go through the normal TaskWorkflow path."""
        task = {'id': 'normal-1', 'title': 'Normal task', 'metadata': {'task_kind': 'normal'}}
        assignment = TaskAssignment(task_id='normal-1', task=task, modules=['some/module'])

        # is_deterministic returns False → normal path
        harness.scheduler.is_deterministic = MagicMock(return_value=False)
        harness._escalation_queue = EscalationQueue(tmp_path)
        harness.scheduler.carries_substrate_probe = MagicMock(return_value=False)

        with (
            patch('orchestrator.harness.TaskWorkflow') as mock_tw,
            patch('orchestrator.harness.DeterministicRunner') as mock_dr,
        ):
            mock_wf = MagicMock()
            mock_wf.run = AsyncMock(return_value=WorkflowOutcome.DONE)
            mock_wf.metrics = MagicMock(
                total_cost_usd=0.0, total_duration_ms=0,
                agent_invocations=0, execute_iterations=0,
                verify_attempts=0, review_cycles=0,
            )
            mock_wf._steward = None
            mock_wf._last_block_reason = ''
            mock_wf._last_block_detail = ''
            mock_wf._last_block_phase = ''
            mock_tw.return_value = mock_wf

            report = await harness._run_slot(assignment, _make_sem())

        # DeterministicRunner must NOT have been called
        mock_dr.assert_not_called()
        # TaskWorkflow must have been called
        mock_tw.assert_called_once()


# ---------------------------------------------------------------------------
# Step-9 (continued): e2e resume → done + dep-satisfied check
# ---------------------------------------------------------------------------

class TestDepSatisfiedAfterGateDone:
    """B4: after gate transitions to done, a dependent task is dispatch-eligible."""

    def test_deps_satisfied_when_gate_is_done(self, tmp_path: Path):
        """Dep task with dependencies=[gate_id] is deps-satisfied when gate is done."""
        from orchestrator.config import OrchestratorConfig
        config = OrchestratorConfig(project_root=tmp_path)
        scheduler = Scheduler(config)

        dep_task = {
            'id': 'worker-1',
            'title': 'Worker task',
            'metadata': {'task_kind': 'normal'},
            'dependencies': [{'id': 'gate-1'}],
        }
        status_map = {'gate-1': 'done'}
        assert scheduler._deps_satisfied(dep_task, status_map) is True

    def test_deps_not_satisfied_when_gate_is_blocked(self, tmp_path: Path):
        """Dep task is NOT deps-satisfied when gate is still blocked."""
        from orchestrator.config import OrchestratorConfig
        config = OrchestratorConfig(project_root=tmp_path)
        scheduler = Scheduler(config)

        dep_task = {
            'id': 'worker-1',
            'title': 'Worker task',
            'metadata': {'task_kind': 'normal'},
            'dependencies': [{'id': 'gate-1'}],
        }
        status_map = {'gate-1': 'blocked'}
        assert scheduler._deps_satisfied(dep_task, status_map) is False


# ---------------------------------------------------------------------------
# Step-11: restart clears stamps; park/abandon do not (RED until step-12)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRestartStampClear:
    """_action_teardown_and_set_status restart: clears gate stamps for det tasks."""

    async def test_restart_clears_gate_escalated_at_stamp(self, harness: Harness):
        """restart action calls update_task(clear stamps) for a deterministic task."""
        task_id = 'gate-1'
        det_task = _det_task(task_id=task_id, gate_escalated_at='2026-06-23T12:00:00+00:00')
        det_task['metadata']['before_done_ran_at'] = '2026-06-23T12:01:00+00:00'

        # get_task returns the stamped deterministic task
        harness.scheduler.get_task = AsyncMock(return_value=det_task)
        # is_deterministic returns True for this task
        harness.scheduler.is_deterministic = MagicMock(return_value=True)
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        await harness._action_teardown_and_set_status(task_id, 'pending', 'restart')

        # update_task must have been called to clear the stamps
        harness.scheduler.update_task.assert_awaited()
        # Find the stamp-clearing call (should contain None values)
        stamp_clear_calls = [
            c for c in harness.scheduler.update_task.call_args_list
            if c.args and isinstance(c.args[1], dict)
            and c.args[1].get('gate_escalated_at') is None
        ]
        assert stamp_clear_calls, (
            'update_task must be called to clear gate_escalated_at=None and '
            'before_done_ran_at=None on restart for a deterministic task'
        )

    async def test_restart_clears_before_done_ran_at_stamp(self, harness: Harness):
        """restart action also clears before_done_ran_at stamp."""
        task_id = 'gate-2'
        det_task = _det_task(task_id=task_id, gate_escalated_at='2026-06-23T12:00:00+00:00')
        det_task['metadata']['before_done_ran_at'] = '2026-06-23T12:01:00+00:00'

        harness.scheduler.get_task = AsyncMock(return_value=det_task)
        harness.scheduler.is_deterministic = MagicMock(return_value=True)
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        await harness._action_teardown_and_set_status(task_id, 'pending', 'restart')

        # Check before_done_ran_at is also cleared
        stamp_clear_calls = [
            c for c in harness.scheduler.update_task.call_args_list
            if c.args and isinstance(c.args[1], dict)
            and c.args[1].get('before_done_ran_at') is None
        ]
        assert stamp_clear_calls, 'before_done_ran_at must also be cleared on restart'

    async def test_park_does_not_clear_stamps(self, harness: Harness):
        """park action (target=blocked) must NOT call update_task for stamp clearing."""
        task_id = 'gate-3'
        det_task = _det_task(task_id=task_id, gate_escalated_at='2026-06-23T12:00:00+00:00')

        harness.scheduler.get_task = AsyncMock(return_value=det_task)
        harness.scheduler.is_deterministic = MagicMock(return_value=True)
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        await harness._action_teardown_and_set_status(task_id, 'blocked', 'park')

        # update_task must NOT be called for stamp clearing on park
        stamp_clear_calls = [
            c for c in harness.scheduler.update_task.call_args_list
            if c.args and isinstance(c.args[1], dict)
            and c.args[1].get('gate_escalated_at') is None
        ]
        assert not stamp_clear_calls, (
            'park must NOT clear stamps — stamps preserved so resume sees gate as escalated'
        )

    async def test_abandon_does_not_clear_stamps(self, harness: Harness):
        """abandon action (target=cancelled) must NOT call update_task for stamp clearing."""
        task_id = 'gate-4'
        det_task = _det_task(task_id=task_id, gate_escalated_at='2026-06-23T12:00:00+00:00')

        harness.scheduler.get_task = AsyncMock(return_value=det_task)
        harness.scheduler.is_deterministic = MagicMock(return_value=True)
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        await harness._action_teardown_and_set_status(task_id, 'cancelled', 'abandon')

        stamp_clear_calls = [
            c for c in harness.scheduler.update_task.call_args_list
            if c.args and isinstance(c.args[1], dict)
            and c.args[1].get('gate_escalated_at') is None
        ]
        assert not stamp_clear_calls, 'abandon must NOT clear stamps'
