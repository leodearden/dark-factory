"""shared.quality_series.stats: the pure statistics behind the quality series (S6 worked numbers)."""

from __future__ import annotations

import hashlib
import statistics

import pytest

from shared.quality_series.stats import (
    bootstrap_interval,
    content_seed,
    newcombe_difference,
    wilson_interval,
)


class TestWilsonInterval:
    @pytest.mark.parametrize(
        ('k', 'n', 'lo', 'hi'),
        [(10, 100, 0.0552, 0.1744), (3, 30, 0.0346, 0.2562)],
    )
    def test_matches_the_published_values(self, k: int, n: int, lo: float, hi: float) -> None:
        assert wilson_interval(k, n) == pytest.approx((lo, hi), abs=5e-4)

    def test_zero_successes_has_a_zero_floor_and_an_honest_ceiling(self) -> None:
        lo, hi = wilson_interval(0, 100)
        assert lo == 0.0
        assert hi == pytest.approx(0.0370, abs=5e-4)

    @pytest.mark.parametrize(('k', 'n'), [(0, 0), (-1, 10), (11, 10)])
    def test_impossible_counts_are_refused_naming_k_and_n(self, k: int, n: int) -> None:
        with pytest.raises(ValueError) as excinfo:
            wilson_interval(k, n)
        assert f'k={k}' in str(excinfo.value)
        assert f'n={n}' in str(excinfo.value)


class TestNewcombeDifference:
    def test_a_halving_on_100_per_side_is_not_a_finding(self) -> None:
        assert newcombe_difference(10, 100, 5, 100) == pytest.approx(
            (-0.05, -0.1296, 0.0263), abs=5e-4
        )

    def test_a_halving_on_300_per_side_is_a_finding(self) -> None:
        assert newcombe_difference(30, 300, 15, 300) == pytest.approx(
            (-0.05, -0.0937, -0.0076), abs=5e-4
        )


class TestContentSeed:
    def test_is_insensitive_to_order(self) -> None:
        assert content_seed(['3', '1', '2']) == content_seed(['2', '3', '1'])

    def test_differs_for_different_id_sets(self) -> None:
        assert content_seed(['1', '2', '3']) != content_seed(['1', '2', '4'])

    def test_is_the_first_eight_sha256_bytes_of_the_sorted_joined_ids(self) -> None:
        expected = int.from_bytes(hashlib.sha256(b'1,2,3').digest()[:8], 'big')
        assert content_seed({1, 2, 3}) == expected


class TestBootstrapInterval:
    _SYMMETRIC = [float(x) for x in range(1, 42)]

    def test_is_deterministic_for_equal_inputs(self) -> None:
        first = bootstrap_interval(self._SYMMETRIC, statistics.median, seed=7, B=1000)
        second = bootstrap_interval(self._SYMMETRIC, statistics.median, seed=7, B=1000)
        assert first == second

    def test_brackets_the_sample_statistic_of_a_symmetric_sample(self) -> None:
        lo, hi = bootstrap_interval(self._SYMMETRIC, statistics.median, seed=7, B=1000)
        assert lo < statistics.median(self._SYMMETRIC) < hi

    def test_a_constant_sample_has_a_degenerate_interval(self) -> None:
        assert bootstrap_interval([2.5] * 20, statistics.median, seed=1, B=200) == (2.5, 2.5)

    def test_an_empty_sample_is_refused(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_interval([], statistics.median, seed=1)
