from math import sqrt

import pytest

from orchestrator.evals.stats import mean_ci95


class TestMeanCI95:
    """Small-sample Student-t two-sided 95% CI, stdlib-only (no scipy/numpy)."""

    def test_three_values_student_t_interval(self):
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
        out = mean_ci95([0.5])
        assert out['mean'] == pytest.approx(0.5)
        assert out['n'] == 1
        assert out['sufficient'] is False
        # No CI for n<2 → the interval collapses to the point estimate.
        assert out['lo'] == pytest.approx(0.5)
        assert out['hi'] == pytest.approx(0.5)

    def test_empty_is_zero_and_insufficient(self):
        out = mean_ci95([])
        assert out['mean'] == 0.0
        assert out['n'] == 0
        assert out['sufficient'] is False

    def test_two_values_has_interval_but_insufficient(self):
        """n==2 → a CI IS computed, but 'sufficient' stays False (decision 10)."""
        out = mean_ci95([0.4, 0.6])
        assert out['n'] == 2
        assert out['sufficient'] is False
        assert out['lo'] < out['mean'] < out['hi']
