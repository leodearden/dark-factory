"""Tests for review suggestion escalation and steward triage grace period."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — lightweight fakes for the workflow's escalation integration
# ---------------------------------------------------------------------------

def _make_workflow(*, escalation_queue=None, escalation_event=None):
    """Build a minimal TaskWorkflow-like object with the methods under test."""
    from orchestrator.workflow import TaskWorkflow

    assignment = MagicMock()
    assignment.task_id = '42'
    assignment.task = {'id': '42', 'title': 'Test Task', 'description': 'desc'}

    config = MagicMock()
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2

    git_ops = MagicMock()
    scheduler = MagicMock()

    mcp = MagicMock()
    mcp.url = 'http://localhost:8002'

    briefing = MagicMock()

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=briefing,
        mcp=mcp,
        escalation_queue=escalation_queue,
        escalation_event=escalation_event,
    )
    return wf


def _fake_reviews(suggestions=None):
    """Return a mock ReviewAggregation with given suggestions."""
    reviews = MagicMock()
    reviews.suggestions = suggestions or []
    reviews.has_blocking_issues = False
    return reviews


def _make_escalation(**overrides):
    from escalation.models import Escalation

    defaults = dict(
        id='esc-42-0',
        task_id='42',
        agent_role='orchestrator',
        severity='info',
        category='review_suggestions',
        summary='3 review suggestion(s) for triage',
        detail='[]',
    )
    defaults.update(overrides)
    return Escalation(**defaults)


# ---------------------------------------------------------------------------
# _escalate_suggestions
# ---------------------------------------------------------------------------

class TestEscalateSuggestions:
    def test_creates_escalation_with_correct_fields(self):
        queue = MagicMock()
        queue.make_id.return_value = 'esc-42-0'
        wf = _make_workflow(escalation_queue=queue)
        wf.state = MagicMock(value='review')

        suggestions = [
            {'reviewer': 'test_analyst', 'severity': 'suggestion',
             'location': 'src/foo.py:10', 'category': 'coverage',
             'description': 'Missing edge case', 'suggested_fix': 'Add test'},
            {'reviewer': 'reuse_auditor', 'severity': 'suggestion',
             'location': 'src/bar.py:20', 'category': 'duplication',
             'description': 'Duplicated logic', 'suggested_fix': 'Extract'},
        ]
        reviews = _fake_reviews(suggestions)

        wf._escalate_suggestions(reviews)

        queue.submit.assert_called_once()
        esc = queue.submit.call_args[0][0]
        assert esc.category == 'review_suggestions'
        assert esc.severity == 'info'
        assert esc.task_id == '42'
        assert json.loads(esc.detail) == suggestions

    def test_noop_without_queue(self):
        wf = _make_workflow(escalation_queue=None)
        reviews = _fake_reviews([{'description': 'something'}])
        # Should not raise
        wf._escalate_suggestions(reviews)

    def test_noop_without_suggestions(self):
        queue = MagicMock()
        wf = _make_workflow(escalation_queue=queue)
        reviews = _fake_reviews([])
        wf._escalate_suggestions(reviews)
        queue.submit.assert_not_called()


# ---------------------------------------------------------------------------
# _await_suggestion_triage
# ---------------------------------------------------------------------------

class TestAwaitSuggestionTriage:
    @pytest.mark.asyncio
    async def test_returns_immediately_if_no_queue(self):
        wf = _make_workflow(escalation_queue=None)
        await wf._await_suggestion_triage(timeout=1.0)

    @pytest.mark.asyncio
    async def test_returns_immediately_if_no_pending(self):
        queue = MagicMock()
        queue.get_by_task.return_value = []
        wf = _make_workflow(escalation_queue=queue)
        await wf._await_suggestion_triage(timeout=1.0)

    @pytest.mark.asyncio
    async def test_returns_when_resolved(self):
        """Should return as soon as the escalation is resolved."""
        esc = _make_escalation()
        queue = MagicMock()
        # First call: pending; second call (after event): resolved
        queue.get_by_task.side_effect = [[esc], []]

        event = asyncio.Event()
        wf = _make_workflow(escalation_queue=queue, escalation_event=event)

        async def _resolve_after_delay():
            await asyncio.sleep(0.05)
            event.set()

        task = asyncio.create_task(_resolve_after_delay())
        await wf._await_suggestion_triage(timeout=5.0)
        await task

        # Should not have created any level-1 re-escalation
        queue.submit.assert_not_called()
        queue.resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_re_escalates_on_timeout(self):
        """Should create level-1 escalation and dismiss level-0 on timeout."""
        detail = json.dumps([{'description': 'test suggestion'}])
        esc = _make_escalation(detail=detail)
        queue = MagicMock()
        queue.make_id.return_value = 'esc-42-1'
        # Always pending (never resolved)
        queue.get_by_task.return_value = [esc]

        wf = _make_workflow(escalation_queue=queue)

        await wf._await_suggestion_triage(timeout=0.1)

        # Should have submitted a level-1 re-escalation
        queue.submit.assert_called_once()
        reesc = queue.submit.call_args[0][0]
        assert reesc.level == 1
        assert reesc.category == 'review_suggestions'
        assert 'Triage timeout' in reesc.summary

        # Should have dismissed the original
        queue.resolve.assert_called_once_with(
            'esc-42-0',
            'Auto-dismissed: triage timeout, re-escalated to level 1',
            dismiss=True,
        )


# ---------------------------------------------------------------------------
# Integration: review loop routes to escalation
# ---------------------------------------------------------------------------

class TestReviewLoopRouting:
    def test_escalates_instead_of_memory_write_when_queue_available(self):
        queue = MagicMock()
        queue.make_id.return_value = 'esc-42-0'
        wf = _make_workflow(escalation_queue=queue)
        wf.state = MagicMock(value='review')

        suggestions = [{'description': 'something'}]
        reviews = _fake_reviews(suggestions)

        with patch.object(wf, '_write_suggestions_to_memory') as mock_write:
            wf._escalate_suggestions(reviews)
            mock_write.assert_not_called()
            queue.submit.assert_called_once()

    def test_falls_back_to_memory_write_without_queue(self):
        """When no escalation queue, _escalate_suggestions is a no-op and
        the caller should fall back to _write_suggestions_to_memory."""
        wf = _make_workflow(escalation_queue=None)
        suggestions = [{'description': 'something'}]
        reviews = _fake_reviews(suggestions)

        # _escalate_suggestions should be a safe no-op
        wf._escalate_suggestions(reviews)
        # The actual fallback happens at the call site in _execute_verify_review_loop
