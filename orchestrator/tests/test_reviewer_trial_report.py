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
