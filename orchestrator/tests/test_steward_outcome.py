"""Tests for the ``StewardOutcome`` sum type (W9-δ).

Replaces the workflow's forensic re-read of escalation-queue files with a
typed, in-process channel carrying one of five frozen-dataclass variants.
Mirrors the ``TerminalReport`` frozen-dataclass precedent (task W9-γ / 2247,
see ``test_workflow_terminal_report.py``): each variant is frozen, carries a
typed payload, and is re-exported through the ``orchestrator.workflow`` shim.

PRD ``plans/workflow-state-machine-prd.md`` §10 / Contract §8.1-8.2 SO-1 /
Resolved decision D6.
"""

from __future__ import annotations

import dataclasses

import pytest
from shared.task_statuses import TaskStatus

from orchestrator import workflow_types
from orchestrator.workflow_types import (
    StewardBudgetExhausted,
    StewardInterrupted,
    StewardOutcome,
    StewardReescalatedL1,
    StewardResolved,
    StewardTerminalDecision,
)


class TestStewardOutcomeVariantsAreFrozen:
    """Each variant is a frozen dataclass — mutation raises FrozenInstanceError."""

    def test_steward_resolved_is_frozen(self):
        outcome = StewardResolved(resolution_text='fixed it')
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.resolution_text = 'x'  # type: ignore[misc]

    def test_steward_reescalated_l1_is_frozen(self):
        outcome = StewardReescalatedL1(esc_id='esc-42-2')
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.esc_id = 'x'  # type: ignore[misc]

    def test_steward_terminal_decision_is_frozen(self):
        outcome = StewardTerminalDecision(new_status=TaskStatus.DEFERRED)
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.new_status = TaskStatus.DONE  # type: ignore[misc]

    def test_steward_interrupted_is_frozen(self):
        outcome = StewardInterrupted(reason='timeout', wip_commits_present=True)
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.reason = 'attempt_cap'  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.wip_commits_present = False  # type: ignore[misc]

    def test_steward_budget_exhausted_rejects_extra_attribute(self):
        """No declared fields — frozen still blocks arbitrary attribute assignment."""
        outcome = StewardBudgetExhausted()
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.anything = 1  # type: ignore[misc, attr-defined]


class TestStewardOutcomePayloads:
    """Each variant carries its documented typed payload and round-trips it."""

    def test_steward_resolved_carries_resolution_text(self):
        outcome = StewardResolved(resolution_text='the fix')
        assert outcome.resolution_text == 'the fix'

    def test_steward_reescalated_l1_carries_esc_id(self):
        outcome = StewardReescalatedL1(esc_id='esc-42-7')
        assert outcome.esc_id == 'esc-42-7'

    def test_steward_terminal_decision_carries_task_status(self):
        outcome = StewardTerminalDecision(new_status=TaskStatus.DEFERRED)
        assert outcome.new_status == TaskStatus.DEFERRED
        assert isinstance(outcome.new_status, TaskStatus)

    @pytest.mark.parametrize('reason', ['timeout', 'attempt_cap'])
    def test_steward_interrupted_accepts_both_reasons(self, reason):
        outcome = StewardInterrupted(reason=reason, wip_commits_present=False)
        assert outcome.reason == reason

    @pytest.mark.parametrize('wip', [True, False])
    def test_steward_interrupted_carries_wip_commits_present(self, wip):
        outcome = StewardInterrupted(reason='timeout', wip_commits_present=wip)
        assert outcome.wip_commits_present is wip

    def test_steward_budget_exhausted_constructs_with_no_args(self):
        outcome = StewardBudgetExhausted()
        assert isinstance(outcome, StewardBudgetExhausted)


class TestStewardOutcomeExports:
    """Every variant + the StewardOutcome alias is public and shim-reexported."""

    @pytest.mark.parametrize('name', [
        'StewardResolved',
        'StewardReescalatedL1',
        'StewardTerminalDecision',
        'StewardInterrupted',
        'StewardBudgetExhausted',
        'StewardOutcome',
    ])
    def test_name_in_workflow_types_all(self, name):
        assert name in workflow_types.__all__

    def test_reexported_from_workflow_module(self):
        from orchestrator.workflow import (
            StewardBudgetExhausted as ShimBudgetExhausted,
        )
        from orchestrator.workflow import (
            StewardInterrupted as ShimInterrupted,
        )
        from orchestrator.workflow import (
            StewardOutcome as ShimOutcome,
        )
        from orchestrator.workflow import (
            StewardReescalatedL1 as ShimReescalatedL1,
        )
        from orchestrator.workflow import (
            StewardResolved as ShimResolved,
        )
        from orchestrator.workflow import (
            StewardTerminalDecision as ShimTerminalDecision,
        )

        assert ShimResolved is StewardResolved
        assert ShimReescalatedL1 is StewardReescalatedL1
        assert ShimTerminalDecision is StewardTerminalDecision
        assert ShimInterrupted is StewardInterrupted
        assert ShimBudgetExhausted is StewardBudgetExhausted
        assert ShimOutcome is StewardOutcome

    def test_steward_outcome_alias_covers_every_variant(self):
        """StewardOutcome is the PEP-604 union of all five variants."""
        import types

        assert isinstance(StewardOutcome, types.UnionType)
        members = set(StewardOutcome.__args__)
        assert members == {
            StewardResolved,
            StewardReescalatedL1,
            StewardTerminalDecision,
            StewardInterrupted,
            StewardBudgetExhausted,
        }
