"""Tests for reviewer trial scorer — metric computation with canned data (no LLM)."""

from __future__ import annotations

import pytest

from orchestrator.evals.reviewer_trial.scorer import (
    IssueMatch,
    ScoringResult,
    _deduplicate_issues,
    _normalize_location,
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
