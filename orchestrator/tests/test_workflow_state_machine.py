"""Tests for ``orchestrator.workflow_types``: the WorkflowState/WorkflowOutcome
relocation, the ``orchestrator.workflow`` re-export shim, and (from step-3
onward) ``WorkflowStateMachine``.

W9-β (PRD ``plans/workflow-state-machine-prd.md`` §10 β; Contract §8.1/§8.2 SM-1).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec
from shared.task_statuses import TaskStatus
from shared.task_transitions import ActorClass
from shared.task_transitions import is_legal_transition as shared_is_legal_transition
from shared.task_transitions import outcome_allows_status as shared_outcome_allows_status

import orchestrator.workflow as workflow_mod
import orchestrator.workflow_types as workflow_types_mod
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow
from orchestrator.workflow_types import (
    IllegalTransition,
    WorkflowOutcome,
    WorkflowState,
    WorkflowStateMachine,
)

# Mirrors the WorkflowState -> TaskStatus projection WorkflowStateMachine.transition
# consumes (step-4's ``_STATE_TO_STATUS``). Defined independently here so this
# module's "iff" proof cross-checks the real implementation against the shared
# table rather than against a copy of the implementation's own table.
_STATE_TO_STATUS: dict[WorkflowState, TaskStatus] = {
    WorkflowState.PLAN: TaskStatus.IN_PROGRESS,
    WorkflowState.EXECUTE: TaskStatus.IN_PROGRESS,
    WorkflowState.VERIFY: TaskStatus.IN_PROGRESS,
    WorkflowState.REVIEW: TaskStatus.IN_PROGRESS,
    WorkflowState.MERGE: TaskStatus.IN_PROGRESS,
    WorkflowState.MERGE_DEFERRED: TaskStatus.MERGE_DEFERRED,
    WorkflowState.DONE: TaskStatus.DONE,
    WorkflowState.BLOCKED: TaskStatus.BLOCKED,
    WorkflowState.ESCALATED: TaskStatus.BLOCKED,
    WorkflowState.CANCELLED: TaskStatus.CANCELLED,
}


class TestEnumRelocationShim:
    """workflow_types.py is the new home for WorkflowState/WorkflowOutcome;
    workflow.py re-exports them so existing importers keep working unchanged.
    """

    def test_workflow_state_shim_is_identical_object(self):
        assert workflow_mod.WorkflowState is workflow_types_mod.WorkflowState

    def test_workflow_outcome_shim_is_identical_object(self):
        assert workflow_mod.WorkflowOutcome is workflow_types_mod.WorkflowOutcome

    def test_workflow_state_members_intact(self):
        assert [(m.name, m.value) for m in WorkflowState] == [
            ('PLAN', 'plan'),
            ('EXECUTE', 'execute'),
            ('VERIFY', 'verify'),
            ('REVIEW', 'review'),
            ('MERGE', 'merge'),
            ('MERGE_DEFERRED', 'merge-deferred'),
            ('DONE', 'done'),
            ('BLOCKED', 'blocked'),
            ('ESCALATED', 'escalated'),
            ('CANCELLED', 'cancelled'),
        ]

    def test_workflow_outcome_members_intact(self):
        assert [(m.name, m.value) for m in WorkflowOutcome] == [
            ('DONE', 'done'),
            ('PLANNED', 'planned'),
            ('BLOCKED', 'blocked'),
            ('REQUEUED', 'requeued'),
            ('ESCALATED', 'escalated'),
            ('CANCELLED', 'cancelled'),
            ('MERGE_DEFERRED', 'merge-deferred'),
            ('SOFT_CANCELLED', 'soft-cancelled'),
        ]

    def test_illegal_transition_is_exception_subclass(self):
        assert issubclass(IllegalTransition, Exception)


class TestWorkflowStateMachine:
    """Pure-unit tests for ``WorkflowStateMachine`` — no ``TaskWorkflow``."""

    def test_legal_linear_sequence_reaches_done(self):
        machine = WorkflowStateMachine()
        for state in (
            WorkflowState.EXECUTE,
            WorkflowState.VERIFY,
            WorkflowState.REVIEW,
            WorkflowState.MERGE,
            WorkflowState.DONE,
        ):
            machine.transition(state)
        assert machine.state == WorkflowState.DONE

    def test_done_to_blocked_raises_boundary_row_5(self):
        """The 7744 case: a late block after DONE must not silently re-block."""
        machine = WorkflowStateMachine(WorkflowState.DONE)
        with pytest.raises(IllegalTransition):
            machine.transition(WorkflowState.BLOCKED)
        assert machine.state == WorkflowState.DONE

    @pytest.mark.parametrize('to', [
        WorkflowState.BLOCKED, WorkflowState.CANCELLED, WorkflowState.PLAN,
    ])
    def test_done_out_transitions_raise(self, to):
        machine = WorkflowStateMachine(WorkflowState.DONE)
        with pytest.raises(IllegalTransition):
            machine.transition(to)
        assert machine.state == WorkflowState.DONE

    def test_done_to_done_is_a_noop(self):
        machine = WorkflowStateMachine(WorkflowState.DONE)
        machine.transition(WorkflowState.DONE)
        assert machine.state == WorkflowState.DONE

    @pytest.mark.parametrize('to', [
        WorkflowState.BLOCKED, WorkflowState.DONE, WorkflowState.PLAN,
    ])
    def test_cancelled_out_transitions_raise(self, to):
        machine = WorkflowStateMachine(WorkflowState.CANCELLED)
        with pytest.raises(IllegalTransition):
            machine.transition(to)
        assert machine.state == WorkflowState.CANCELLED

    def test_cancelled_to_cancelled_is_a_noop(self):
        machine = WorkflowStateMachine(WorkflowState.CANCELLED)
        machine.transition(WorkflowState.CANCELLED)
        assert machine.state == WorkflowState.CANCELLED

    @pytest.mark.parametrize(('state', 'expected'), [
        (WorkflowState.PLAN, False),
        (WorkflowState.EXECUTE, False),
        (WorkflowState.VERIFY, False),
        (WorkflowState.REVIEW, False),
        (WorkflowState.MERGE, False),
        (WorkflowState.BLOCKED, False),
        (WorkflowState.ESCALATED, False),
        (WorkflowState.MERGE_DEFERRED, False),
        (WorkflowState.DONE, True),
        (WorkflowState.CANCELLED, True),
    ])
    def test_is_terminal(self, state, expected):
        assert WorkflowStateMachine(state).is_terminal() is expected


class TestNeverAFourthTable:
    """Behavioral + identity proof that ``WorkflowStateMachine`` delegates
    legality entirely to ``shared.task_transitions`` — it defines no
    transition table of its own (G4 decision #1: escalation server, the fm
    interceptor, and this machine all consume the SAME table).
    """

    def test_is_legal_transition_is_the_shared_object(self):
        assert workflow_types_mod.is_legal_transition is shared_is_legal_transition

    def test_outcome_allows_status_is_the_shared_object(self):
        assert workflow_types_mod.outcome_allows_status is shared_outcome_allows_status

    @pytest.mark.parametrize(('frm', 'to'), [
        # Linear phase advances (same-projected-status, always legal).
        (WorkflowState.PLAN, WorkflowState.EXECUTE),
        (WorkflowState.EXECUTE, WorkflowState.VERIFY),
        (WorkflowState.REVIEW, WorkflowState.MERGE),
        # Completion.
        (WorkflowState.MERGE, WorkflowState.DONE),
        # Block / unblock / cancel from a working phase.
        (WorkflowState.PLAN, WorkflowState.BLOCKED),
        (WorkflowState.BLOCKED, WorkflowState.PLAN),
        (WorkflowState.BLOCKED, WorkflowState.CANCELLED),
        (WorkflowState.ESCALATED, WorkflowState.PLAN),
        # Merge-deferred park transitions.
        (WorkflowState.MERGE_DEFERRED, WorkflowState.DONE),
        (WorkflowState.MERGE_DEFERRED, WorkflowState.BLOCKED),
        # Terminal absorption — boundary row 5 / the 7744 case.
        (WorkflowState.DONE, WorkflowState.BLOCKED),
        (WorkflowState.DONE, WorkflowState.PLAN),
        (WorkflowState.CANCELLED, WorkflowState.PLAN),
        (WorkflowState.CANCELLED, WorkflowState.DONE),
        # Non-terminal but genuinely absent from the shared union.
        (WorkflowState.MERGE_DEFERRED, WorkflowState.PLAN),
        (WorkflowState.MERGE_DEFERRED, WorkflowState.EXECUTE),
    ])
    def test_transition_raises_iff_shared_table_says_illegal(self, frm, to):
        expected_legal = shared_is_legal_transition(
            _STATE_TO_STATUS[frm], _STATE_TO_STATUS[to], ActorClass.ORCHESTRATOR,
        )
        machine = WorkflowStateMachine(frm)
        if expected_legal:
            machine.transition(to)
            assert machine.state == to
        else:
            with pytest.raises(IllegalTransition):
                machine.transition(to)
            assert machine.state == frm


def _make_workflow(*, task_id: str = '77') -> tuple[TaskWorkflow, MagicMock]:
    """Minimal ``TaskWorkflow`` construction, mirroring the ``_make`` builder in
    ``test_workflow_already_done.py`` but trimmed to the state-machine wiring
    surface — no worktree/artifacts are needed for these tests.

    Returns ``(workflow, scheduler)`` so callers can assert on scheduler calls
    (e.g. the ``_mark_blocked`` terminal-absorption guard).
    """
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = []

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))

    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=MagicMock(),
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    return wf, scheduler


class TestTaskWorkflowStateMachineWiring:
    """Integration: ``TaskWorkflow.state``/``machine`` wiring (step-6)."""

    def test_construction_starts_machine_at_plan(self):
        wf, _ = _make_workflow()
        assert isinstance(wf.machine, WorkflowStateMachine)
        assert wf.machine.state == WorkflowState.PLAN
        assert wf.state == WorkflowState.PLAN

    def test_state_setter_delegates_to_machine_force_set(self):
        wf, _ = _make_workflow()
        wf.state = WorkflowState.MERGE
        assert wf.machine.state == WorkflowState.MERGE
        assert wf.state == WorkflowState.MERGE

    def test_enter_phase_advances_both_state_and_machine(self):
        wf, _ = _make_workflow()
        wf._enter_phase(WorkflowState.EXECUTE)
        assert wf.state == WorkflowState.EXECUTE
        assert wf.machine.state == WorkflowState.EXECUTE

    def test_enter_phase_after_terminal_raises_and_state_unchanged(self):
        wf, _ = _make_workflow()
        wf.state = WorkflowState.DONE
        with pytest.raises(IllegalTransition):
            wf._enter_phase(WorkflowState.BLOCKED)
        assert wf.state == WorkflowState.DONE


@pytest.mark.asyncio
class TestMarkBlockedTerminalAbsorption:
    """``_mark_blocked``'s terminal-absorption guard must be a no-op for
    ANY terminal ``WorkflowState`` (sourced from ``machine.is_terminal()``),
    not just DONE.
    """

    async def test_mark_blocked_after_cancelled_returns_cancelled(self):
        """New generalization — CANCELLED is absorbing too, not just DONE.

        Pre-step-8, the guard only checked ``WorkflowState.DONE``, so this
        fell through to ``_enter_phase(WorkflowState.BLOCKED)``, which
        raised ``IllegalTransition`` (CANCELLED is absorbing per the
        machine). The guard now sources absorption from
        ``machine.is_terminal()``, which covers both.
        """
        wf, scheduler = _make_workflow()
        wf.state = WorkflowState.CANCELLED

        outcome = await wf._mark_blocked('late block after cancel')

        assert outcome == WorkflowOutcome.CANCELLED
        scheduler.set_task_status.assert_not_called()

    async def test_mark_blocked_after_done_returns_done(self):
        """DONE regression — mirrors test_workflow_e2e.py TestDoneIsTerminal."""
        wf, scheduler = _make_workflow()
        wf.state = WorkflowState.DONE

        outcome = await wf._mark_blocked('late block')

        assert outcome == WorkflowOutcome.DONE
        scheduler.set_task_status.assert_not_called()
