"""Tests for the escalation_queue missing warning in TaskWorkflow.

Covers ``TaskWorkflow._maybe_warn_missing_escalation`` — the helper that emits
a single WARNING per workflow instance when an escalation-capable agent role is
first invoked while ``self.escalation_queue`` is None.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import ARCHITECT, ROLES
from orchestrator.workflow import _ESCALATION_CAPABLE_ROLES, TaskWorkflow, WorkflowOutcome, WorkflowState

# ---------------------------------------------------------------------------
# E2E fixtures and helpers — imported so pytest discovers them in this module.
# Re-exported from test_workflow_e2e (relative import within the tests package).
# See plan.json design decision: placed here per task instructions; fixtures are
# duplicated/re-imported to avoid conftest pollution.
# ---------------------------------------------------------------------------
from .test_workflow_e2e import (  # noqa: E402
    AgentStub,
    _build_workflow_with_escalation,
    config,        # fixture: OrchestratorConfig backed by a real git repo
    git_ops,       # fixture: GitOps from config
    git_repo,      # fixture: bare git repo in tmp_path
    task_assignment,  # fixture: TaskAssignment for task '42'
)


def _make_workflow(*, escalation_queue=None) -> TaskWorkflow:
    """Minimal TaskWorkflow instance for escalation-warning tests.

    Mirrors the pattern from ``test_workflow_amendment.py:23-50``: uses
    MagicMock for all dependencies, only overrides the field under test.
    """
    assignment = MagicMock()
    assignment.task_id = '42'
    assignment.task = {'id': '42', 'title': 'Test Task', 'description': 'd'}
    assignment.modules = ['some/module']

    config = MagicMock()
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0

    return TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=MagicMock(),
        scheduler=MagicMock(),
        briefing=MagicMock(),
        mcp=MagicMock(),
        escalation_queue=escalation_queue,
    )


class TestMaybeWarnMissingEscalation:
    """``_maybe_warn_missing_escalation`` — the per-workflow escalation warning."""

    def test_warns_when_escalation_queue_none_on_escalation_capable_role(self, caplog):
        """WARNING is emitted for an escalation-capable role when queue is None."""
        wf = _make_workflow(escalation_queue=None)
        with caplog.at_level(logging.WARNING):
            wf._maybe_warn_missing_escalation('architect')
        assert any(
            'escalation' in rec.message.lower() and 'architect' in rec.message
            for rec in caplog.records
            if rec.levelno >= logging.WARNING
        ), 'expected a WARNING containing "escalation" and "architect"'

    def test_no_warning_when_escalation_queue_present(self, caplog):
        """No WARNING is emitted when the escalation_queue is non-None."""
        wf = _make_workflow(escalation_queue=MagicMock())
        with caplog.at_level(logging.WARNING):
            wf._maybe_warn_missing_escalation('architect')
        assert not any(
            'escalation_queue is unavailable' in rec.message
            for rec in caplog.records
            if rec.levelno >= logging.WARNING
        ), 'unexpected WARNING about missing escalation_queue when queue is present'

    @pytest.mark.parametrize('role_name', ['judge', 'reviewer_comprehensive', 'steward', 'deep_reviewer'])
    def test_no_warning_for_non_escalation_capable_role(self, caplog, role_name):
        """No WARNING is emitted for roles that do not use escalation tools."""
        wf = _make_workflow(escalation_queue=None)
        with caplog.at_level(logging.WARNING):
            wf._maybe_warn_missing_escalation(role_name)
        assert not any(
            'escalation_queue is unavailable' in rec.message
            for rec in caplog.records
            if rec.levelno >= logging.WARNING
        ), f'unexpected WARNING for non-escalation-capable role {role_name!r}'

    def test_single_shot_across_multiple_invocations(self, caplog):
        """WARNING fires exactly once even when called multiple times with different roles."""
        wf = _make_workflow(escalation_queue=None)
        with caplog.at_level(logging.WARNING):
            wf._maybe_warn_missing_escalation('architect')
            wf._maybe_warn_missing_escalation('implementer')
            wf._maybe_warn_missing_escalation('merger')
        matching = [
            rec for rec in caplog.records
            if rec.levelno >= logging.WARNING
            and 'escalation_queue is unavailable' in rec.message
        ]
        assert len(matching) == 1, (
            f'expected exactly 1 WARNING but got {len(matching)}: {[r.message for r in matching]}'
        )
        assert wf._escalation_missing_warned is True


class TestEscalationCapableRolesDerivation:
    """``_ESCALATION_CAPABLE_ROLES`` is derived correctly from ``ROLES``."""

    def test_escalation_capable_roles_derived_from_roles(self):
        """_ESCALATION_CAPABLE_ROLES must equal the concrete expected set.

        Hard-coded so any formula change, ROLES edit, or tool-list modification
        requires a deliberate update here — a re-derived expected would be a
        tautology that can never fail.

        'steward' and 'deep_reviewer' are excluded by the dispatcher carve-out
        even though they carry _ESCALATION_TOOLS; every member must also be a
        valid ROLES entry.
        """
        expected = frozenset({'architect', 'implementer', 'debugger', 'merger'})
        assert expected == _ESCALATION_CAPABLE_ROLES
        assert 'steward' not in _ESCALATION_CAPABLE_ROLES, (
            "'steward' must not be in _ESCALATION_CAPABLE_ROLES (TaskSteward dispatcher carve-out)"
        )
        assert 'deep_reviewer' not in _ESCALATION_CAPABLE_ROLES, (
            "'deep_reviewer' must not be in _ESCALATION_CAPABLE_ROLES (ReviewCheckpoint dispatcher carve-out)"
        )
        unknown = _ESCALATION_CAPABLE_ROLES - set(ROLES)
        assert not unknown, (
            f'All members of _ESCALATION_CAPABLE_ROLES must be valid ROLES entries; '
            f'unknown: {unknown!r}'
        )


class TestInvokeWiresWarning:
    """Integration: ``_invoke`` calls ``_maybe_warn_missing_escalation``."""

    @pytest.mark.asyncio
    async def test_invoke_calls_maybe_warn_missing_escalation_for_architect(self):
        """_invoke calls _maybe_warn_missing_escalation with the architect role name."""
        stub_result = AgentResult(success=True, output='', cost_usd=0.0)
        wf = _make_workflow(escalation_queue=None)

        with patch('orchestrator.workflow.invoke_with_cap_retry', new=AsyncMock(return_value=stub_result)), \
             patch.object(wf, '_maybe_warn_missing_escalation') as mock_warn:
            await wf._invoke(ARCHITECT, prompt='x', cwd=Path('/tmp'))

        mock_warn.assert_called_once_with(ARCHITECT.name)


# ---------------------------------------------------------------------------
# E2E: first-invocation budget-exhaustion path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFirstInvocationBudgetExhaustion:
    """When SessionBudgetExhausted fires before any role completes, the escalation
    uses the 'last_completed_role' label and resolves to 'n/a' (None fallback).

    This test drives workflow.run() through the _SessionBudgetExhausted handler
    on the FIRST invoke_agent call — before the architect writes plan.json, so
    _last_completed_role stays None throughout.

    RED condition: current code uses the label 'last_role' (the old name), so both
    assertions below fail until the rename in step-2 impl lands.
    """

    async def test_label_is_last_completed_role_na_when_no_role_completed(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path
    ):
        """Before any role completes, _last_completed_role is None → label resolves to n/a.

        Asserts:
          (i)   outcome is BLOCKED
          (ii)  exactly one escalation in the queue
          (iii) detail contains 'last_completed_role=n/a'  (new label + default fallback)
          (iv)  summary contains 'last completed role: n/a'  (new summary wording)
        """
        from orchestrator.usage_gate import SessionBudgetExhausted

        stub = AgentStub()
        config.usage_cap.session_budget_usd = 0.10
        workflow, _, queue = _build_workflow_with_escalation(
            config, git_ops, task_assignment, stub, tmp_path,
        )

        # Raise SessionBudgetExhausted on the very first invoke_agent call —
        # the architect never writes plan.json, so _last_completed_role stays None.
        async def always_raise(*args, **kwargs):
            raise SessionBudgetExhausted(cumulative_cost=0.50)

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', always_raise)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(side_effect=AssertionError('run_scoped_verification must not be called')),
        )

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.BLOCKED, (
            f'Expected BLOCKED outcome, got: {outcome!r}'
        )

        escalations = queue.get_by_task(task_assignment.task_id)
        assert len(escalations) == 1, (
            f'Expected exactly 1 escalation, got {len(escalations)}'
        )
        esc = escalations[0]
        detail = esc.detail
        summary = esc.summary

        # (iii) detail label — fails on current code (uses 'last_role=n/a')
        assert 'last_completed_role=n/a' in detail, (
            f'Expected "last_completed_role=n/a" in detail, got: {detail!r}'
        )
        # (iv) summary wording — fails on current code (uses 'last role: n/a')
        assert 'last completed role: n/a' in summary, (
            f'Expected "last completed role: n/a" in summary, got: {summary!r}'
        )
