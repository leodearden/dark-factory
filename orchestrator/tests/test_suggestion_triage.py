"""Tests for review suggestion escalation and steward completion grace period."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch  # noqa: F401

import pytest
from shared.cli_invoke import AllAccountsCappedException

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
        # Detail is prefixed with content fingerprint: #hash:<hex16>#<json>
        detail = esc.detail
        assert detail.startswith('#hash:')
        json_start = detail.index('#', 6) + 1
        assert json.loads(detail[json_start:]) == suggestions

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


# ---------------------------------------------------------------------------
# Pre-triage integration (steward)
# ---------------------------------------------------------------------------


def _make_steward(*, config_overrides=None, suggestion_count=15):
    """Build a minimal TaskSteward with mocked dependencies."""
    from pathlib import Path

    from orchestrator.steward import TaskSteward

    config = MagicMock()
    config.project_root = Path('/tmp/project')
    config.steward_lifetime_budget = 12.0
    config.steward_max_attempts = 3
    config.steward_max_timeouts_per_escalation = 3
    config.suggestion_triage_threshold = 10
    config.models.triage = 'sonnet'
    config.budgets.triage = 2.0
    config.max_turns.triage = 25
    config.effort.triage = 'medium'
    config.backends.triage = 'claude'
    config.models.steward = 'opus'
    config.budgets.steward = 5.0
    config.max_turns.steward = 100
    config.effort.steward = 'high'
    config.backends.steward = 'claude'
    config.escalation.host = '127.0.0.1'
    config.escalation.port = 8100
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(config, k, v)

    queue = MagicMock()
    queue.make_id.return_value = 'esc-42-1'
    queue.get_by_task.return_value = []

    briefing = MagicMock()
    briefing.build_steward_initial_prompt = AsyncMock(return_value='initial prompt')

    mcp = MagicMock()
    mcp.mcp_config_json.return_value = {}

    task = {'id': '42', 'title': 'Test Task', 'description': 'desc'}
    steward = TaskSteward(
        task_id='42',
        task=task,
        worktree=Path('/tmp/worktree'),
        config=config,
        mcp=mcp,
        escalation_queue=queue,
        briefing=briefing,
        usage_gate=None,
    )
    return steward


def _make_suggestions(n):
    """Generate n fake review suggestions."""
    return [
        {
            'reviewer': 'test_analyst',
            'severity': 'suggestion',
            'location': f'src/mod{i}.py:{i * 10}',
            'category': 'coverage',
            'description': f'Missing test for case {i}',
            'suggested_fix': f'Add test for case {i}',
        }
        for i in range(n)
    ]


def _fake_agent_result(*, cost=0.5, turns=3, structured_output=None):
    """Return a mock AgentResult."""
    result = MagicMock()
    result.cost_usd = cost
    result.duration_ms = 2000
    result.turns = turns
    result.success = True
    result.session_id = 'sess-123'
    result.stderr = ''
    result.output = ''
    result.structured_output = structured_output
    result.account_name = ''
    return result


class TestPreTriageSuggestions:
    @pytest.mark.asyncio
    async def test_pre_triage_invoked_above_threshold(self):
        steward = _make_steward()
        suggestions = _make_suggestions(15)

        triage_output = {
            'accepted': [
                {'index': i, 'suggestion': f'case {i}', 'reason': 'merit',
                 'files': [f'src/mod{i}.py'], 'proposed_task_title': f'Fix {i}'}
                for i in range(10)
            ],
            'skipped': [
                {'index': i, 'suggestion': f'case {i}', 'reason': 'noise'}
                for i in range(10, 15)
            ],
            'proposed_task_groups': [
                {'title': 'Add tests', 'description': 'Add missing tests',
                 'accepted_indices': list(range(10))},
            ],
        }

        esc = _make_escalation(detail=json.dumps(suggestions))

        with patch('orchestrator.steward.invoke_with_cap_retry',
                   return_value=_fake_agent_result(structured_output=triage_output)):
            result = await steward._pre_triage_suggestions(esc)

        assert '## Pre-Triaged Results' in result.detail
        assert '10 accepted' in result.summary
        assert '5 skipped' in result.summary

    @pytest.mark.asyncio
    async def test_pre_triage_not_invoked_below_threshold(self):
        """Small suggestion sets should skip pre-triage in _handle_escalation."""
        steward = _make_steward()
        suggestions = _make_suggestions(5)
        esc = _make_escalation(detail=json.dumps(suggestions))

        # Steward session mock — returns resolved escalation
        steward_result = _fake_agent_result(cost=2.0)
        cast(MagicMock, steward.escalation_queue).get.return_value = MagicMock(status='resolved')

        with patch('orchestrator.steward.invoke_agent', return_value=steward_result) as mock_invoke:
            await steward._handle_escalation(esc)

        # Only the steward session should be called — no triage invocation
        assert mock_invoke.call_count == 1
        call_kwargs = mock_invoke.call_args
        assert call_kwargs.kwargs.get('model') or 'opus' in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_pre_triage_failure_falls_back(self):
        steward = _make_steward()
        suggestions = _make_suggestions(15)
        esc = _make_escalation(detail=json.dumps(suggestions))

        # Triage returns no structured output
        bad_result = _fake_agent_result(structured_output=None)
        bad_result.success = False

        with patch('orchestrator.steward.invoke_with_cap_retry', return_value=bad_result):
            result = await steward._pre_triage_suggestions(esc)

        # Original escalation returned unchanged
        assert result.detail == esc.detail
        assert result.summary == esc.summary

    @pytest.mark.asyncio
    async def test_pre_triage_cost_tracked_in_metrics(self):
        steward = _make_steward()
        assert steward.metrics.total_cost_usd == 0.0

        suggestions = _make_suggestions(15)
        esc = _make_escalation(detail=json.dumps(suggestions))

        triage_output = {
            'accepted': [], 'skipped': [],
            'proposed_task_groups': [],
        }
        result = _fake_agent_result(cost=0.75, structured_output=triage_output)

        with patch('orchestrator.steward.invoke_with_cap_retry', return_value=result):
            await steward._pre_triage_suggestions(esc)

        assert steward.metrics.total_cost_usd == 0.75
        assert steward.metrics.invocations == 1

    @pytest.mark.asyncio
    async def test_pre_triage_replaces_escalation_detail(self):
        steward = _make_steward()
        suggestions = _make_suggestions(12)
        esc = _make_escalation(detail=json.dumps(suggestions))

        triage_output = {
            'accepted': [
                {'index': 0, 'suggestion': 'case 0', 'reason': 'merit',
                 'files': ['src/mod0.py'], 'proposed_task_title': 'Fix 0'},
            ],
            'skipped': [
                {'index': i, 'suggestion': f'case {i}', 'reason': 'noise'}
                for i in range(1, 12)
            ],
            'proposed_task_groups': [
                {'title': 'Fix case 0', 'description': 'Fix it',
                 'accepted_indices': [0]},
            ],
        }

        with patch('orchestrator.steward.invoke_with_cap_retry',
                   return_value=_fake_agent_result(structured_output=triage_output)):
            result = await steward._pre_triage_suggestions(esc)

        # Detail is replaced with pre-triaged markdown
        assert '## Pre-Triaged Results' in result.detail
        assert 'Fix case 0' in result.detail
        # Original suggestions are embedded as reference
        assert 'Original Suggestions' in result.detail


# ---------------------------------------------------------------------------
# Pre-triage cap handling
# ---------------------------------------------------------------------------


class TestPreTriageCapHandling:
    @pytest.mark.asyncio
    async def test_pre_triage_returns_original_escalation_on_cap(
        self, caplog
    ):
        """_pre_triage_suggestions must return the original escalation unchanged on cap.

        Before step-8 impl: AllAccountsCappedException propagates out of
        _pre_triage_suggestions, crashing the steward.
        After step-8 impl: exception is caught, original escalation returned.
        """
        steward = _make_steward()
        suggestions = _make_suggestions(15)
        escalation = _make_escalation(detail=json.dumps(suggestions))

        cap_exc = AllAccountsCappedException(
            retries=2, elapsed_secs=30.0, label='Steward for task 42 [pre-triage]'
        )

        with patch(
            'orchestrator.steward.invoke_with_cap_retry',
            AsyncMock(side_effect=cap_exc),
        ), caplog.at_level(logging.WARNING, logger='orchestrator.steward'):
            result = await steward._pre_triage_suggestions(escalation)

        # Must return the original escalation (identity check)
        assert result is escalation, (
            'Expected original escalation object to be returned unchanged'
        )

        # Must emit a warning containing 'all accounts capped'
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            'all accounts capped' in t.lower() for t in warning_texts
        ), f'Expected warning with "all accounts capped", got: {warning_texts}'
