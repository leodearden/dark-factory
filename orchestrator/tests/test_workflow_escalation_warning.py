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
from orchestrator.workflow import _ESCALATION_CAPABLE_ROLES, TaskWorkflow


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
