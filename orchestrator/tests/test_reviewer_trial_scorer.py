"""Tests for reviewer trial scorer — metric computation with canned data (no LLM)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.evals.reviewer_trial.corpus import CorpusDiff, GroundTruthIssue
from orchestrator.evals.reviewer_trial.runner import PanelRunResult
from orchestrator.evals.reviewer_trial.scorer import (
    IssueMatch,
    ScoringResult,
    _deduplicate_issues,
    _normalize_location,
    match_issues,
    score_panel_run,
)


class TestNormalizeLocation:
    def test_strips_line_number(self) -> None:
        assert _normalize_location('src/foo.py:42') == 'src/foo.py'

    def test_strips_line_range(self) -> None:
        assert _normalize_location('src/foo.py:42-50') == 'src/foo.py'

    def test_no_line_number(self) -> None:
        assert _normalize_location('src/foo.py') == 'src/foo.py'

    def test_normalizes_backslash(self) -> None:
        assert _normalize_location('src\\foo.py:10') == 'src/foo.py'

    def test_strips_whitespace(self) -> None:
        assert _normalize_location('  src/foo.py:10  ') == 'src/foo.py'


class TestDeduplicateIssues:
    def test_no_duplicates(self) -> None:
        reviews = {
            'r1': {
                'issues': [
                    {'location': 'a.py:1', 'category': 'bug', 'description': 'Bug A'},
                ]
            },
            'r2': {
                'issues': [
                    {'location': 'b.py:2', 'category': 'style', 'description': 'Style B'},
                ]
            },
        }
        deduped = _deduplicate_issues(reviews)
        assert len(deduped) == 2

    def test_same_location_category_deduped(self) -> None:
        reviews = {
            'r1': {
                'issues': [
                    {'location': 'a.py:1', 'category': 'bug', 'description': 'Short.'},
                ]
            },
            'r2': {
                'issues': [
                    {'location': 'a.py:5', 'category': 'bug', 'description': 'Longer description here.'},
                ]
            },
        }
        deduped = _deduplicate_issues(reviews)
        assert len(deduped) == 1
        # Keeps the longer description
        assert 'Longer' in deduped[0]['description']

    def test_same_location_different_category_not_deduped(self) -> None:
        reviews = {
            'r1': {
                'issues': [
                    {'location': 'a.py:1', 'category': 'bug', 'description': 'Bug.'},
                    {'location': 'a.py:1', 'category': 'style', 'description': 'Style.'},
                ]
            },
        }
        deduped = _deduplicate_issues(reviews)
        assert len(deduped) == 2

    def test_empty_reviews(self) -> None:
        assert _deduplicate_issues({}) == []

    def test_reviews_with_no_issues(self) -> None:
        reviews = {'r1': {'issues': []}}
        assert _deduplicate_issues(reviews) == []

    def test_reviewer_name_preserved(self) -> None:
        reviews = {
            'test_analyst': {
                'issues': [
                    {'location': 'a.py:1', 'category': 'bug', 'description': 'X'},
                ]
            },
        }
        deduped = _deduplicate_issues(reviews)
        assert deduped[0]['reviewer'] == 'test_analyst'


class TestScoringMetrics:
    """Test metric computation with manually constructed ScoringResult data."""

    def test_perfect_recall(self) -> None:
        # 3/3 ground truth matched
        result = ScoringResult(
            variant_name='test',
            diff_id='d1',
            matches=[
                IssueMatch(reviewer_issue={}, ground_truth_id='gt1', match_confidence=0.9, match_reasoning=''),
                IssueMatch(reviewer_issue={}, ground_truth_id='gt2', match_confidence=0.8, match_reasoning=''),
                IssueMatch(reviewer_issue={}, ground_truth_id='gt3', match_confidence=0.7, match_reasoning=''),
            ],
            unmatched_gt=[],
            false_positives=[],
            recall=1.0,
            precision=1.0,
            f1=1.0,
            blocking_recall=1.0,
        )
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_zero_recall(self) -> None:
        result = ScoringResult(
            variant_name='test',
            diff_id='d1',
            matches=[],
            unmatched_gt=['gt1', 'gt2'],
            false_positives=[{'description': 'FP'}],
            recall=0.0,
            precision=0.0,
            f1=0.0,
            blocking_recall=0.0,
        )
        assert result.recall == 0.0
        assert result.f1 == 0.0

    def test_partial_metrics(self) -> None:
        # 2/4 GT matched, 2/3 findings are true positives
        recall = 2 / 4
        precision = 2 / 3
        f1 = 2 * precision * recall / (precision + recall)
        result = ScoringResult(
            variant_name='test',
            diff_id='d1',
            matches=[
                IssueMatch(reviewer_issue={}, ground_truth_id='gt1', match_confidence=0.9, match_reasoning=''),
                IssueMatch(reviewer_issue={}, ground_truth_id='gt2', match_confidence=0.8, match_reasoning=''),
            ],
            unmatched_gt=['gt3', 'gt4'],
            false_positives=[{'description': 'FP'}],
            recall=round(recall, 4),
            precision=round(precision, 4),
            f1=round(f1, 4),
            blocking_recall=0.5,
        )
        assert result.recall == pytest.approx(0.5)
        assert result.precision == pytest.approx(0.6667, abs=0.001)
        assert result.f1 > 0


def _make_matcher_result(cost_usd: float = 0.42, structured: dict | None = None, output: str = '') -> SimpleNamespace:
    """Build a fake AgentResult for invoke_agent in match_issues."""
    return SimpleNamespace(
        success=True,
        output=output,
        cost_usd=cost_usd,
        duration_ms=1000,
        turns=1,
        session_id='test-session',
        structured_output=structured if structured is not None else {'matches': []},
        stderr='',
    )


def _make_gt(gt_id: str = 'gt1') -> GroundTruthIssue:
    return GroundTruthIssue(
        id=gt_id,
        location='a.py:1',
        category='bug',
        severity='blocking',
        description='A bug',
        mutation_type='logic',
    )


class TestMatchIssuesCost:
    """Tests that match_issues returns a 3-tuple (matches, unmatched, cost_usd)."""

    @pytest.mark.asyncio
    async def test_returns_three_tuple_with_cost(self) -> None:
        """match_issues returns (matches, unmatched, cost_usd) capturing the LLM call cost."""
        fake_result = _make_matcher_result(cost_usd=0.42, structured={'matches': []})
        reviewer_issues = [{'location': 'a.py:1', 'category': 'bug', 'description': 'x'}]
        ground_truth = [_make_gt('gt1')]

        with patch('orchestrator.evals.reviewer_trial.scorer.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = fake_result
            result = await match_issues(
                reviewer_issues=reviewer_issues,
                ground_truth=ground_truth,
                diff_text='diff',
            )

        assert isinstance(result, tuple)
        assert len(result) == 3
        matches, unmatched, cost = result
        assert isinstance(matches, list)
        assert isinstance(unmatched, list)
        assert cost == pytest.approx(0.42)

    @pytest.mark.asyncio
    async def test_empty_inputs_cost_is_zero(self) -> None:
        """match_issues returns 0.0 cost and does NOT call invoke_agent for empty inputs."""
        with patch('orchestrator.evals.reviewer_trial.scorer.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            result_empty_reviewers = await match_issues(
                reviewer_issues=[],
                ground_truth=[_make_gt('gt1')],
                diff_text='diff',
            )
            result_empty_gt = await match_issues(
                reviewer_issues=[{'location': 'a.py:1', 'category': 'bug', 'description': 'x'}],
                ground_truth=[],
                diff_text='diff',
            )
            mock_invoke.assert_not_called()

        assert result_empty_reviewers[2] == 0.0
        assert result_empty_gt[2] == 0.0

    @pytest.mark.asyncio
    async def test_unparseable_output_still_reports_cost(self) -> None:
        """match_issues reports incurred cost even when LLM output is unparseable."""
        fake_result = _make_matcher_result(cost_usd=0.42, structured=None, output='not json at all')
        # Override structured_output to be None explicitly
        fake_result.structured_output = None

        reviewer_issues = [{'location': 'a.py:1', 'category': 'bug', 'description': 'x'}]
        ground_truth = [_make_gt('gt1')]

        with patch('orchestrator.evals.reviewer_trial.scorer.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = fake_result
            result = await match_issues(
                reviewer_issues=reviewer_issues,
                ground_truth=ground_truth,
                diff_text='diff',
            )

        matches, unmatched, cost = result
        assert matches == []
        assert unmatched == reviewer_issues
        assert cost == pytest.approx(0.42)


class TestScoringResultMatchCost:
    """ScoringResult should carry a match_cost_usd field with default 0.0."""

    def test_default_match_cost_is_zero(self) -> None:
        result = ScoringResult(variant_name='v', diff_id='d')
        assert result.match_cost_usd == 0.0
        assert result.cost_usd == 0.0


class TestScorePanelRunCost:
    """score_panel_run should populate match_cost_usd from the matcher."""

    @pytest.mark.asyncio
    async def test_match_cost_populated_from_matcher(self) -> None:
        """score_panel_run sets cost_usd=panel cost and match_cost_usd=matcher cost."""
        diff = CorpusDiff(
            diff_id='d1',
            language='python',
            source='synthetic',
            diff_text='--- a/f.py\n+++ b/f.py',
            description='Test diff',
            ground_truth=[],
        )
        run = PanelRunResult(
            variant_name='v1',
            diff_id='d1',
            total_cost_usd=1.25,
            reviews={},
            wall_clock_ms=100,
        )

        with patch('orchestrator.evals.reviewer_trial.scorer.match_issues', new_callable=AsyncMock) as mock_match:
            mock_match.return_value = ([], [], 0.37)
            score = await score_panel_run(run, diff)

        assert score.cost_usd == pytest.approx(1.25)
        assert score.match_cost_usd == pytest.approx(0.37)
