"""Tests for review suggestion escalation and steward completion grace period."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch  # noqa: F401

import pytest

# ---------------------------------------------------------------------------
# Helpers
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
    config.steward_completion_timeout = 300.0

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


def _fake_reviews(suggestions=None, blocking_issues=None):
    """Return a mock ReviewAggregation."""
    reviews = MagicMock()
    reviews.suggestions = suggestions or []
    reviews.blocking_issues = blocking_issues or []
    reviews.has_blocking_issues = bool(blocking_issues)
    reviews.format_for_replan.return_value = 'formatted review feedback'
    return reviews


def _make_escalation(**overrides):
    from escalation.models import Escalation

    defaults: dict = dict(
        id='esc-42-0',
        task_id='42',
        agent_role='orchestrator',
        severity='info',
        category='review_suggestions',
        summary='3 review suggestion(s) for triage',
        detail='[]',
    )
    defaults.update(overrides)
    return Escalation(**defaults)  # type: ignore[arg-type]


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
        wf._escalate_suggestions(reviews)

    def test_noop_without_suggestions(self):
        queue = MagicMock()
        wf = _make_workflow(escalation_queue=queue)
        reviews = _fake_reviews([])
        wf._escalate_suggestions(reviews)
        queue.submit.assert_not_called()


# ---------------------------------------------------------------------------
# _escalate_review_issues
# ---------------------------------------------------------------------------

class TestEscalateReviewIssues:
    def test_creates_blocking_escalation(self):
        queue = MagicMock()
        queue.make_id.return_value = 'esc-42-5'
        wf = _make_workflow(escalation_queue=queue)
        wf.state = MagicMock(value='review')

        reviews = _fake_reviews(
            blocking_issues=[{'description': 'bug'}, {'description': 'crash'}],
            suggestions=[{'description': 'style'}],
        )

        wf._escalate_review_issues(reviews)

        queue.submit.assert_called_once()
        esc = queue.submit.call_args[0][0]
        assert esc.severity == 'blocking'
        assert esc.category == 'review_issues'
        assert '2 blocking issue(s)' in esc.summary
        assert '1 suggestion(s)' in esc.summary

    def test_noop_without_queue(self):
        wf = _make_workflow(escalation_queue=None)
        reviews = _fake_reviews(blocking_issues=[{'description': 'bug'}])
        wf._escalate_review_issues(reviews)  # Should not raise


# ---------------------------------------------------------------------------
# _await_steward_completion
# ---------------------------------------------------------------------------

class TestAwaitStewardCompletion:
    @pytest.mark.asyncio
    async def test_returns_immediately_if_no_queue(self):
        wf = _make_workflow(escalation_queue=None)
        await wf._await_steward_completion()

    @pytest.mark.asyncio
    async def test_returns_immediately_if_no_pending(self):
        queue = MagicMock()
        queue.get_by_task.return_value = []
        wf = _make_workflow(escalation_queue=queue)
        await wf._await_steward_completion()

    @pytest.mark.asyncio
    async def test_returns_when_resolved(self):
        esc = _make_escalation()
        queue = MagicMock()
        queue.get_by_task.side_effect = [[esc], []]

        event = asyncio.Event()
        wf = _make_workflow(escalation_queue=queue, escalation_event=event)
        wf._steward = MagicMock()  # steward must be running to wait

        async def _resolve_after_delay():
            await asyncio.sleep(0.05)
            event.set()

        task = asyncio.create_task(_resolve_after_delay())
        await wf._await_steward_completion()
        await task

        queue.submit.assert_not_called()
        queue.resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_re_escalates_on_timeout(self):
        detail = json.dumps([{'description': 'test suggestion'}])
        esc = _make_escalation(detail=detail)
        queue = MagicMock()
        queue.make_id.return_value = 'esc-42-1'
        queue.get_by_task.return_value = [esc]

        wf = _make_workflow(escalation_queue=queue)
        wf.config.steward_completion_timeout = 0.1
        wf._steward = MagicMock()  # steward must be running to wait

        await wf._await_steward_completion()

        queue.submit.assert_called_once()
        reesc = queue.submit.call_args[0][0]
        assert reesc.level == 1
        assert 'Steward timeout' in reesc.summary

        queue.resolve.assert_called_once()
        resolve_args = queue.resolve.call_args
        assert resolve_args[0][0] == 'esc-42-0'
        assert resolve_args[1].get('dismiss') is True


# ---------------------------------------------------------------------------
# Review loop routing
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
        wf = _make_workflow(escalation_queue=None)
        suggestions = [{'description': 'something'}]
        reviews = _fake_reviews(suggestions)
        wf._escalate_suggestions(reviews)
