"""Tests for stats_utils statistical utility functions."""

from __future__ import annotations

import pytest

from dashboard.data.stats_utils import percentile


class TestPercentile:
    """Tests for the percentile() function."""

    def test_empty_list_returns_zero(self):
        """Empty input returns 0.0."""
        assert percentile([], 50) == 0.0

    def test_single_value_p50(self):
        """Single-element list returns that value for p=50."""
        assert percentile([42.0], 50) == 42.0

    def test_single_value_p0(self):
        """Single-element list returns that value for p=0."""
        assert percentile([7.0], 0) == 7.0

    def test_single_value_p100(self):
        """Single-element list returns that value for p=100."""
        assert percentile([7.0], 100) == 7.0

    def test_two_values_p50_interpolates_midpoint(self):
        """Two-value list at p=50 returns the midpoint."""
        assert percentile([10.0, 20.0], 50) == 15.0

    def test_five_values_p50_exact_middle(self):
        """[1,2,3,4,5] at p=50 lands on index 2 exactly → 3.0."""
        assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0

    def test_four_values_p50_interpolated(self):
        """[1,2,3,4] at p=50 interpolates between indices 1 and 2 → 2.5."""
        assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5

    def test_ten_values_p50(self):
        """[100,200,...,1000] at p=50 interpolates to 550.0."""
        values = [float(i * 100) for i in range(1, 11)]  # [100,200,...,1000]
        assert percentile(values, 50) == 550.0

    def test_ten_values_p95_greater_than_p50(self):
        """p=95 result is greater than p=50 result for a non-trivial list."""
        values = [float(i * 100) for i in range(1, 11)]
        assert percentile(values, 95) > percentile(values, 50)

    def test_p0_returns_first_element(self):
        """p=0 always returns the first element of a non-empty list."""
        assert percentile([5.0, 10.0, 15.0], 0) == 5.0

    def test_p100_returns_last_element(self):
        """p=100 always returns the last element of a non-empty list."""
        assert percentile([5.0, 10.0, 15.0], 100) == 15.0
