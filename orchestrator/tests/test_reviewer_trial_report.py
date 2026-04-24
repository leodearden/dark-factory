"""Tests for reviewer trial report generation."""

from __future__ import annotations

import pytest

from orchestrator.evals.reviewer_trial.corpus import CorpusDiff, CorpusManifest
from orchestrator.evals.reviewer_trial.report import build_trial_report
from orchestrator.evals.reviewer_trial.scorer import ScoringResult
from orchestrator.evals.reviewer_trial.variants import ReviewerSpec, VariantConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_corpus(*diff_ids: str) -> CorpusManifest:
    """Build a CorpusManifest with synthetic diffs and no ground truth."""
    diffs = [
        CorpusDiff(
            diff_id=did,
            language='python',
            source='synthetic',
            diff_text='--- a/f.py\n+++ b/f.py',
            description=f'Test diff {did}',
            ground_truth=[],
        )
        for did in diff_ids
    ]
    return CorpusManifest(diffs=diffs)


def _make_variant(name: str, n_reviewers: int = 1) -> VariantConfig:
    return VariantConfig(
        name=name,
        description=f'Variant {name}',
        reviewers=[
            ReviewerSpec(name=f'r{i}', model='sonnet', specialization='Testing.')
            for i in range(n_reviewers)
        ],
    )


def _make_score(
    variant_name: str,
    diff_id: str,
    cost_usd: float = 0.0,
    match_cost_usd: float = 0.0,
    f1: float = 0.0,
    blocking_recall: float = 0.0,
) -> ScoringResult:
    return ScoringResult(
        variant_name=variant_name,
        diff_id=diff_id,
        cost_usd=cost_usd,
        match_cost_usd=match_cost_usd,
        f1=f1,
        blocking_recall=blocking_recall,
    )


# ---------------------------------------------------------------------------
# Tests: TrialReport.total_cost_usd
# ---------------------------------------------------------------------------

class TestTrialReportTotalCost:
    """TrialReport.total_cost_usd must sum panel AND matcher costs."""

    def test_total_cost_includes_match_cost(self) -> None:
        """total_cost_usd = sum of (cost_usd + match_cost_usd) across all scores."""
        corpus = _make_corpus('d1', 'd2')
        variants = [_make_variant('v1')]

        # Three scores: panel=[1.0, 2.0, 0.5], match=[0.3, 0.4, 0.2]
        # Combined: 1.3 + 2.4 + 0.7 = 4.4
        # Panel only: 3.5  ← current (broken) behaviour
        scores = [
            _make_score('v1', 'd1', cost_usd=1.0, match_cost_usd=0.3),
            _make_score('v1', 'd2', cost_usd=2.0, match_cost_usd=0.4),
            _make_score('v1', 'd1', cost_usd=0.5, match_cost_usd=0.2),
        ]

        report = build_trial_report(scores, variants, corpus)

        assert report.total_cost_usd == pytest.approx(4.4, abs=0.01)


# ---------------------------------------------------------------------------
# Tests: VariantSummary cost and per-dollar metrics
# ---------------------------------------------------------------------------

class TestVariantSummaryCost:
    """VariantSummary.total_cost_usd and f1_per_dollar must use combined costs."""

    def test_variant_total_cost_includes_match_and_f1_per_dollar_uses_combined_denominator(self) -> None:
        """summary.total_cost_usd = panel+match; f1_per_dollar uses combined denominator."""
        corpus = _make_corpus('d1', 'd2')
        variants = [_make_variant('v1')]

        # 2 diffs, 1 variant
        # cost_usd=[1.0, 1.0], match_cost_usd=[0.5, 0.5] => combined=3.0
        # f1=[0.8, 0.6] => mean_f1=0.7
        # f1_per_dollar = 0.7 / 3.0 = 0.2333...
        scores = [
            _make_score('v1', 'd1', cost_usd=1.0, match_cost_usd=0.5, f1=0.8),
            _make_score('v1', 'd2', cost_usd=1.0, match_cost_usd=0.5, f1=0.6),
        ]

        report = build_trial_report(scores, variants, corpus)
        assert len(report.summaries) == 1
        summary = report.summaries[0]

        expected_total = 3.0   # panel 2.0 + match 1.0
        expected_f1_per_dollar = round(0.7 / 3.0, 4)

        assert summary.total_cost_usd == pytest.approx(expected_total, abs=0.01)
        assert summary.f1_per_dollar == pytest.approx(expected_f1_per_dollar, abs=0.0001)
