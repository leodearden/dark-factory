"""Tests for the escalation_queue missing warning in TaskWorkflow.

Covers ``TaskWorkflow._maybe_warn_missing_escalation`` — the helper that emits
a single WARNING per workflow instance when an escalation-capable agent role is
first invoked while ``self.escalation_queue`` is None.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from orchestrator.workflow import TaskWorkflow


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
