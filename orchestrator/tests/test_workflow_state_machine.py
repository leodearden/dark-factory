"""Tests for ``orchestrator.workflow_types``: the WorkflowState/WorkflowOutcome
relocation, the ``orchestrator.workflow`` re-export shim, and (from step-3
onward) ``WorkflowStateMachine``.

W9-β (PRD ``plans/workflow-state-machine-prd.md`` §10 β; Contract §8.1/§8.2 SM-1).
"""

from __future__ import annotations

import orchestrator.workflow as workflow_mod
import orchestrator.workflow_types as workflow_types_mod
from orchestrator.workflow_types import IllegalTransition, WorkflowOutcome, WorkflowState


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
