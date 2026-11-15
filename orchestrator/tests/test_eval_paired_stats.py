"""The seeded percentile bootstrap of a paired mean difference (evals/paired_stats.py)."""

from __future__ import annotations

import math

import pytest

from orchestrator.evals.paired_stats import PairedDifference, bootstrap_mean_ci


def test_no_pairs_gives_no_mean_and_no_interval() -> None:
    assert bootstrap_mean_ci([]) == PairedDifference(n=0, mean=None, ci_low=None, ci_high=None)


def test_one_pair_gives_a_mean_but_no_interval() -> None:
    result = bootstrap_mean_ci([-1.0])
    assert (result.n, result.mean, result.ci_low, result.ci_high) == (1, -1.0, None, None)


@pytest.mark.parametrize('constant', [0.0, -1.0])
def test_constant_differences_give_a_degenerate_interval(constant: float) -> None:
    result = bootstrap_mean_ci([constant] * 8)
    assert (result.n, result.mean, result.ci_low, result.ci_high) == (8, constant, constant, constant)


def test_the_same_input_always_gives_the_same_interval() -> None:
    differences = [0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0]
    assert bootstrap_mean_ci(differences) == bootstrap_mean_ci(differences)


def test_the_mean_lies_inside_its_interval() -> None:
    result = bootstrap_mean_ci([0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    assert result.ci_low is not None and result.ci_high is not None and result.mean is not None
    assert result.ci_low <= result.mean <= result.ci_high


def test_a_balanced_binary_sample_brackets_its_mean_at_the_binomial_width() -> None:
    result = bootstrap_mean_ci([0.0, -1.0] * 50)
    assert result.n == 100
    assert result.mean == -0.5
    assert result.ci_low is not None and result.ci_high is not None
    assert -0.65 <= result.ci_low <= -0.55
    assert -0.45 <= result.ci_high <= -0.35


@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf])
def test_a_non_finite_difference_is_refused_by_value(bad: float) -> None:
    with pytest.raises(ValueError, match=repr(bad)):
        bootstrap_mean_ci([0.0, bad, -1.0])
