"""Tests for reviewer trial runner — mock invoke_agent, verify structure."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.evals.reviewer_trial.corpus import CorpusDiff, CorpusManifest, GroundTruthIssue
from orchestrator.evals.reviewer_trial.runner import (
    REVIEW_SCHEMA,
    PanelRunResult,
    _build_reviewer_prompt,
    run_panel,
)
from orchestrator.evals.reviewer_trial.variants import (
    ReviewerSpec,
    VariantConfig,
)


def _make_review(name: str = 'test', verdict: str = 'PASS') -> dict:
    return {
        'reviewer': name,
        'verdict': verdict,
        'issues': [],
        'summary': 'All good.',
    }


def _make_result(output: str = '', structured: dict | None = None):
    """Build a mock AgentResult."""
    from types import SimpleNamespace
    return SimpleNamespace(
        success=True,
        output=output,
        cost_usd=0.50,
        duration_ms=5000,
        turns=5,
        session_id='mock-session',
        structured_output=structured,
        stderr='',
    )


class TestBuildReviewerPrompt:
    def test_contains_diff(self) -> None:
        prompt = _build_reviewer_prompt('--- a/f.py\n+++ b/f.py')
        assert '--- a/f.py' in prompt
        assert '```diff' in prompt

    def test_truncates_long_diff(self) -> None:
        long_diff = 'x' * 60_000
        prompt = _build_reviewer_prompt(long_diff)
        assert '... [diff truncated] ...' in prompt
        assert len(prompt) < 55_000

    def test_action_instruction(self) -> None:
        prompt = _build_reviewer_prompt('diff')
        assert 'Review the diff' in prompt


class TestReviewSchema:
    def test_required_fields(self) -> None:
        assert set(REVIEW_SCHEMA['required']) == {'reviewer', 'verdict', 'issues', 'summary'}

    def test_verdict_enum(self) -> None:
        verdict_props = REVIEW_SCHEMA['properties']['verdict']
        assert set(verdict_props['enum']) == {'PASS', 'ISSUES_FOUND'}


class TestPanelRunResult:
    def test_defaults(self) -> None:
        result = PanelRunResult(variant_name='test', diff_id='d1')
        assert result.reviews == {}
        assert result.total_cost_usd == 0.0
        assert result.errors == []


class TestRunPanel:
    @pytest.mark.asyncio
    async def test_single_reviewer_structured_output(self) -> None:
        review = _make_review('r1', 'PASS')
        mock_result = _make_result(structured=review)

        variant = VariantConfig(
            name='test_variant',
            description='Test',
            reviewers=[
                ReviewerSpec(name='r1', model='sonnet', specialization='Testing.'),
            ],
        )
        diff = CorpusDiff(
            diff_id='d1',
            language='python',
            source='synthetic',
            diff_text='--- a/f.py\n+++ b/f.py',
            description='Test',
            ground_truth=[],
        )

        with patch('orchestrator.evals.reviewer_trial.runner.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_result
            result = await run_panel(variant, diff, stagger_secs=0)

        assert result.variant_name == 'test_variant'
        assert result.diff_id == 'd1'
        assert 'r1' in result.reviews
        assert result.reviews['r1']['verdict'] == 'PASS'
        assert result.total_cost_usd == 0.50
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_multiple_reviewers(self) -> None:
        variant = VariantConfig(
            name='multi',
            description='Test',
            reviewers=[
                ReviewerSpec(name='r1', model='sonnet', specialization='S1.'),
                ReviewerSpec(name='r2', model='sonnet', specialization='S2.'),
            ],
        )
        diff = CorpusDiff(
            diff_id='d1',
            language='python',
            source='synthetic',
            diff_text='diff',
            description='Test',
            ground_truth=[],
        )

        call_count = 0

        async def mock_invoke(**kwargs):
            nonlocal call_count
            call_count += 1
            name = f'r{call_count}'
            return _make_result(structured=_make_review(name))

        with patch('orchestrator.evals.reviewer_trial.runner.invoke_agent', side_effect=mock_invoke):
            result = await run_panel(variant, diff, stagger_secs=0)

        assert len(result.reviews) == 2
        assert result.total_cost_usd == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_reviewer_error_recorded(self) -> None:
        variant = VariantConfig(
            name='error_test',
            description='Test',
            reviewers=[
                ReviewerSpec(name='r1', model='sonnet', specialization='S.'),
            ],
        )
        diff = CorpusDiff(
            diff_id='d1',
            language='python',
            source='synthetic',
            diff_text='diff',
            description='Test',
            ground_truth=[],
        )

        with patch(
            'orchestrator.evals.reviewer_trial.runner.invoke_agent',
            new_callable=AsyncMock,
            side_effect=RuntimeError('API error'),
        ):
            result = await run_panel(variant, diff, stagger_secs=0, max_retries=1)

        assert len(result.errors) == 1
        assert 'r1' not in result.reviews

    @pytest.mark.asyncio
    async def test_json_fallback_when_no_structured_output(self) -> None:
        review = _make_review('r1', 'ISSUES_FOUND')
        mock_result = _make_result(output=json.dumps(review))

        variant = VariantConfig(
            name='json_test',
            description='Test',
            reviewers=[
                ReviewerSpec(name='r1', model='sonnet', specialization='S.'),
            ],
        )
        diff = CorpusDiff(
            diff_id='d1',
            language='python',
            source='synthetic',
            diff_text='diff',
            description='Test',
            ground_truth=[],
        )

        with patch('orchestrator.evals.reviewer_trial.runner.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_result
            result = await run_panel(variant, diff, stagger_secs=0)

        assert 'r1' in result.reviews
        assert result.reviews['r1']['verdict'] == 'ISSUES_FOUND'
