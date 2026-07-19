"""Tests for evals/report.py — the C4 composite report surface (task 2477 λ).

Covers the statistics substrate (mean_ci95, _ratio_score), the per-config
price table, the composite report over the UNION of configs (retiring the
all-tasks-intersection collapse), its deterministic renderer, and the
union-aggregation of Elo ratings.
"""

from __future__ import annotations

from math import sqrt

import pytest

# ---------------------------------------------------------------------------
# Task 2477 step-05: mean_ci95 (Student-t small-sample CI) + _ratio_score
# ---------------------------------------------------------------------------

class TestMeanCI95:
    """Small-sample Student-t two-sided 95% CI, stdlib-only (no scipy/numpy)."""

    def test_three_values_student_t_interval(self):
        from orchestrator.evals.report import mean_ci95

        out = mean_ci95([0.4, 0.5, 0.6])
        assert out['mean'] == pytest.approx(0.5)
        assert out['n'] == 3
        assert out['sufficient'] is True

        # Hand-computed: stdev(ddof=1) == 0.1; t(df=2, 0.975) == 4.303;
        # half-width == 4.303 * 0.1 / sqrt(3) == 0.24843…
        half = 4.303 * 0.1 / sqrt(3)
        assert out['lo'] == pytest.approx(0.5 - half, abs=1e-3)
        assert out['hi'] == pytest.approx(0.5 + half, abs=1e-3)
        # Cross-check the plan's stated endpoints.
        assert out['lo'] == pytest.approx(0.2515, abs=1e-3)
        assert out['hi'] == pytest.approx(0.7485, abs=1e-3)

    def test_single_value_has_no_interval(self):
        from orchestrator.evals.report import mean_ci95

        out = mean_ci95([0.5])
        assert out['mean'] == pytest.approx(0.5)
        assert out['n'] == 1
        assert out['sufficient'] is False
        # No CI for n<2 → the interval collapses to the point estimate.
        assert out['lo'] == pytest.approx(0.5)
        assert out['hi'] == pytest.approx(0.5)

    def test_empty_is_zero_and_insufficient(self):
        from orchestrator.evals.report import mean_ci95

        out = mean_ci95([])
        assert out['mean'] == 0.0
        assert out['n'] == 0
        assert out['sufficient'] is False

    def test_two_values_has_interval_but_insufficient(self):
        """n==2 → a CI IS computed, but 'sufficient' stays False (decision 10)."""
        from orchestrator.evals.report import mean_ci95

        out = mean_ci95([0.4, 0.6])
        assert out['n'] == 2
        assert out['sufficient'] is False
        assert out['lo'] < out['mean'] < out['hi']


class TestRatioScore:
    """best/value normalization, single-config-safe, clamped to [0, 1]."""

    def test_worse_value_scores_below_one(self):
        from orchestrator.evals.report import _ratio_score

        # value twice the best → 0.5
        assert _ratio_score(2.0, 1.0) == pytest.approx(0.5)

    def test_best_equals_value_scores_one(self):
        from orchestrator.evals.report import _ratio_score

        assert _ratio_score(1.0, 1.0) == pytest.approx(1.0)

    def test_nonpositive_best_or_value_is_neutral_one(self):
        """Undefined normalization (single-config / zero denominator) → 1.0."""
        from orchestrator.evals.report import _ratio_score

        assert _ratio_score(1.0, 0.0) == pytest.approx(1.0)
        assert _ratio_score(1.0, -3.0) == pytest.approx(1.0)
        assert _ratio_score(0.0, 1.0) == pytest.approx(1.0)
        assert _ratio_score(-2.0, 1.0) == pytest.approx(1.0)

    def test_result_clamped_to_unit_interval(self):
        from orchestrator.evals.report import _ratio_score

        # A value BETTER than 'best' (best/value > 1) clamps down to 1.0.
        assert _ratio_score(0.5, 1.0) == 1.0
