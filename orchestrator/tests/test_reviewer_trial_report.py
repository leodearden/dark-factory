"""Tests for reviewer trial report generation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from orchestrator.evals.measurement import MeasurementStage, Unmeasured, UnmeasuredCause
from orchestrator.evals.reviewer_trial.corpus import CorpusDiff, CorpusManifest
from orchestrator.evals.reviewer_trial.report import build_trial_report, format_markdown
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


# ---------------------------------------------------------------------------
# Tests: an unmeasured score is excluded from every figure and counted
# ---------------------------------------------------------------------------

def _unmeasured_score(
    variant_name: str, diff_id: str, cost_usd: float = 1.0, match_cost_usd: float = 0.31,
) -> ScoringResult:
    return ScoringResult(
        variant_name=variant_name,
        diff_id=diff_id,
        cost_usd=cost_usd,
        match_cost_usd=match_cost_usd,
        unmeasured=Unmeasured(MeasurementStage.MATCHER, UnmeasuredCause.CAP_HIT, 'r', cost_usd=0.31),
    )


def _measured(variant_name: str, diff_id: str, f1: float) -> ScoringResult:
    return ScoringResult(
        variant_name=variant_name, diff_id=diff_id, recall=f1, precision=f1, f1=f1,
        blocking_recall=f1, cost_usd=1.0, match_cost_usd=0.1, wall_clock_ms=500,
    )


class TestUnmeasuredScores:
    def test_an_unmeasured_score_changes_no_variant_mean_but_its_counts(self) -> None:
        corpus = _make_corpus('d1', 'd2', 'd3')
        variants = [_make_variant('v1'), _make_variant('v2')]
        measured = [
            _measured('v1', 'd1', 0.8), _measured('v1', 'd2', 0.6),
            _measured('v2', 'd1', 0.4), _measured('v2', 'd2', 0.5),
        ]

        clean = build_trial_report(measured, variants, corpus)
        counted = build_trial_report([*measured, _unmeasured_score('v1', 'd3')], variants, corpus)

        clean_v1 = next(s for s in clean.summaries if s.name == 'v1')
        counted_v1 = next(s for s in counted.summaries if s.name == 'v1')
        assert replace(counted_v1, unmeasured=0, unmeasured_cost_usd=0.0) == clean_v1
        assert counted_v1.unmeasured == 1
        assert counted_v1.unmeasured_cost_usd == pytest.approx(1.31)
        assert counted.total_cost_usd == pytest.approx(clean.total_cost_usd + 1.31, abs=0.01)

    def test_a_variant_with_nothing_measured_reports_none_and_sorts_last(self) -> None:
        corpus = _make_corpus('d1', 'd2')
        variants = [_make_variant('v2'), _make_variant('v1')]
        scores = [
            _measured('v1', 'd1', 0.0), _measured('v1', 'd2', 0.0),
            _unmeasured_score('v2', 'd1'), _unmeasured_score('v2', 'd2'),
        ]

        report = build_trial_report(scores, variants, corpus)

        v2 = report.summaries[-1]
        assert v2.name == 'v2'
        assert v2.mean_recall is None
        assert v2.mean_precision is None
        assert v2.mean_f1 is None
        assert v2.mean_blocking_recall is None
        assert v2.mean_wall_clock_ms is None
        assert v2.f1_per_dollar is None
        assert v2.blocking_recall_per_dollar is None
        assert v2.unmeasured == 2
        assert 'n/a' in format_markdown(report)

    def test_head_to_head_skips_a_diff_either_side_did_not_measure(self) -> None:
        corpus = _make_corpus('d1', 'd2')
        variants = [_make_variant('v1'), _make_variant('v2')]
        scores = [
            _measured('v1', 'd1', 0.2), _unmeasured_score('v2', 'd1'),
            _measured('v1', 'd2', 0.2), _measured('v2', 'd2', 0.6),
        ]

        h2h = build_trial_report(scores, variants, corpus).head_to_heads[0]

        assert 'd1' not in h2h.a_wins + h2h.b_wins + h2h.ties
        assert h2h.unmeasured == ['d1']
        assert h2h.b_wins == ['d2']

    def test_per_diff_scores_mark_an_unmeasured_cell_none(self) -> None:
        corpus = _make_corpus('d1')
        variants = [_make_variant('v1'), _make_variant('v2')]
        scores = [_measured('v1', 'd1', 0.2), _unmeasured_score('v2', 'd1')]

        report = build_trial_report(scores, variants, corpus)

        assert report.per_diff_scores['d1']['v1'] == pytest.approx(0.2)
        assert report.per_diff_scores['d1']['v2'] is None
        assert '| d1 | 0.200 | n/a |' in format_markdown(report)

    def test_leaderboard_shows_an_unmeasured_column(self) -> None:
        corpus = _make_corpus('d1', 'd2')
        variants = [_make_variant('v1')]
        scores = [_measured('v1', 'd1', 0.5), _unmeasured_score('v1', 'd2')]

        md = format_markdown(build_trial_report(scores, variants, corpus)).splitlines()

        header = next(line for line in md if line.startswith('| Rank |'))
        row = next(line for line in md if line.startswith('| 1 | v1 |'))
        columns = [c.strip() for c in header.strip('|').split('|')]
        cells = [c.strip() for c in row.strip('|').split('|')]
        assert cells[columns.index('Unmeasured')] == '1'
