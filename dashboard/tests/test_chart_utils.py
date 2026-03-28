"""Tests for chart_utils utility functions."""

from __future__ import annotations

import typing

import pytest

from dashboard.data.chart_utils import ChartData, group_top_n, separate_label


class TestChartDataType:
    """Tests for ChartData TypedDict definition."""

    def test_chartdata_importable(self):
        """ChartData can be imported from chart_utils."""
        assert ChartData is not None

    def test_chartdata_has_labels_key(self):
        """ChartData has a 'labels' annotation."""
        hints = typing.get_type_hints(ChartData)
        assert 'labels' in hints

    def test_chartdata_has_values_key(self):
        """ChartData has a 'values' annotation."""
        hints = typing.get_type_hints(ChartData)
        assert 'values' in hints

    def test_chartdata_labels_is_list_of_str(self):
        """ChartData 'labels' annotation is list[str]."""
        hints = typing.get_type_hints(ChartData)
        assert hints['labels'] == list[str]

    def test_chartdata_values_is_list_of_int_or_float(self):
        """ChartData 'values' annotation is list[int | float]."""
        hints = typing.get_type_hints(ChartData)
        assert hints['values'] == list[int | float]


class TestGroupTopNSorting:
    """Tests for group_top_n sorting unsorted input descending by values."""

    def test_unsorted_input_produces_correct_top_n_and_other(self):
        """Unsorted input with len > n produces correct top-N + Other after sorting."""
        data: ChartData = {
            'labels': ['c', 'a', 'e', 'b', 'd', 'f'],
            'values': [30, 60, 10, 50, 20, 40],
        }
        result = group_top_n(data, n=3)
        # After descending sort: a=60, b=50, f=40, c=30, d=20, e=10
        # Top 3: a, b, f; Other = 30+20+10 = 60
        assert result['labels'][:3] == ['a', 'b', 'f']
        assert result['values'][:3] == [60, 50, 40]
        assert result['labels'][-1] == 'Other'
        assert result['values'][-1] == 60

    def test_unsorted_input_within_n_returns_descending_order(self):
        """Unsorted input with len <= n returns values in descending order."""
        data: ChartData = {
            'labels': ['c', 'a', 'b'],
            'values': [30, 60, 50],
        }
        result = group_top_n(data, n=5)
        assert result['labels'] == ['a', 'b', 'c']
        assert result['values'] == [60, 50, 30]

    def test_already_sorted_input_is_unchanged(self):
        """Already-sorted descending input is returned in same order."""
        data: ChartData = {
            'labels': ['a', 'b', 'c', 'd', 'e', 'f'],
            'values': [60, 50, 40, 30, 20, 10],
        }
        result = group_top_n(data, n=5)
        assert result['labels'][:5] == ['a', 'b', 'c', 'd', 'e']
        assert result['values'][:5] == [60, 50, 40, 30, 20]
        assert result['labels'][-1] == 'Other'
        assert result['values'][-1] == 10


class TestGroupTopNValidation:
    """Tests for length validation in group_top_n."""

    def test_raises_when_labels_longer_than_values(self):
        """ValueError is raised when labels list is longer than values list."""
        data: ChartData = {'labels': ['a', 'b', 'c'], 'values': [10, 20]}
        with pytest.raises(ValueError, match="same length"):
            group_top_n(data)

    def test_raises_when_values_longer_than_labels(self):
        """ValueError is raised when values list is longer than labels list."""
        data: ChartData = {'labels': ['a', 'b'], 'values': [10, 20, 30]}
        with pytest.raises(ValueError, match="same length"):
            group_top_n(data)

    def test_no_error_when_lengths_match(self):
        """No error is raised when labels and values have the same length."""
        data: ChartData = {'labels': ['a', 'b', 'c'], 'values': [30, 20, 10]}
        result = group_top_n(data, n=5)
        assert result['labels'] == ['a', 'b', 'c']
        assert result['values'] == [30, 20, 10]


class TestGroupTopN:
    """Tests for group_top_n(data, n) utility."""

    def test_passthrough_when_at_n_items(self):
        """When input has exactly N items, return unchanged."""
        data = {'labels': ['a', 'b', 'c', 'd', 'e'], 'values': [5, 4, 3, 2, 1]}
        result = group_top_n(data, n=5)
        assert result['labels'] == ['a', 'b', 'c', 'd', 'e']
        assert result['values'] == [5, 4, 3, 2, 1]
        assert 'Other' not in result['labels']

    def test_passthrough_when_below_n_items(self):
        """When input has fewer than N items, return unchanged."""
        data = {'labels': ['search', 'add_memory'], 'values': [100, 50]}
        result = group_top_n(data, n=5)
        assert result['labels'] == ['search', 'add_memory']
        assert result['values'] == [100, 50]
        assert 'Other' not in result['labels']

    def test_groups_tail_into_other_when_exceeds_n(self):
        """When input has more than N items, tail entries become 'Other'."""
        data = {
            'labels': ['a', 'b', 'c', 'd', 'e', 'f'],
            'values': [60, 50, 40, 30, 20, 10],
        }
        result = group_top_n(data, n=5)
        assert len(result['labels']) == 6  # top 5 + 'Other'
        assert result['labels'][-1] == 'Other'
        assert result['values'][-1] == 10

    def test_other_sums_correctly(self):
        """'Other' value is the sum of all tail items."""
        data = {
            'labels': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
            'values': [70, 60, 50, 40, 30, 20, 10],
        }
        result = group_top_n(data, n=5)
        assert result['labels'][-1] == 'Other'
        assert result['values'][-1] == 30  # 20 + 10

    def test_handles_empty_input(self):
        """Empty input returns empty output without error."""
        data = {'labels': [], 'values': []}
        result = group_top_n(data, n=5)
        assert result['labels'] == []
        assert result['values'] == []

    def test_preserves_descending_sort_order(self):
        """Top N items appear in their original order (already descending by convention)."""
        data = {
            'labels': ['a', 'b', 'c', 'd', 'e', 'f'],
            'values': [60, 50, 40, 30, 20, 10],
        }
        result = group_top_n(data, n=5)
        assert result['labels'][:5] == ['a', 'b', 'c', 'd', 'e']
        assert result['values'][:5] == [60, 50, 40, 30, 20]

    def test_custom_n_parameter(self):
        """Custom N grouping threshold is respected."""
        data = {
            'labels': ['a', 'b', 'c'],
            'values': [30, 20, 10],
        }
        result = group_top_n(data, n=2)
        assert len(result['labels']) == 3  # top 2 + 'Other'
        assert result['labels'] == ['a', 'b', 'Other']
        assert result['values'] == [30, 20, 10]

    def test_default_n_is_five(self):
        """Default N=5 groups items beyond 5 into 'Other'."""
        data = {
            'labels': ['a', 'b', 'c', 'd', 'e', 'f'],
            'values': [60, 50, 40, 30, 20, 10],
        }
        result = group_top_n(data)
        assert result['labels'][-1] == 'Other'
        assert len(result['labels']) == 6


class TestSeparateLabelBasic:
    """Tests for separate_label basic extraction behavior."""

    def test_returns_remaining_data_and_extracted_value(self):
        """separate_label removes target label and returns (remaining, value)."""
        data: ChartData = {'labels': ['a', 'b', 'c'], 'values': [30, 20, 10]}
        remaining, value = separate_label(data, 'b')
        assert value == 20
        assert 'b' not in remaining['labels']
        assert remaining['labels'] == ['a', 'c']
        assert remaining['values'] == [30, 10]

    def test_does_not_mutate_original_data(self):
        """separate_label does not modify the original data dict."""
        data: ChartData = {'labels': ['a', 'b', 'c'], 'values': [30, 20, 10]}
        original_labels = list(data['labels'])
        original_values = list(data['values'])
        separate_label(data, 'b')
        assert data['labels'] == original_labels
        assert data['values'] == original_values

    def test_extracts_first_label(self):
        """separate_label works when target is the first label."""
        data: ChartData = {'labels': ['a', 'b', 'c'], 'values': [30, 20, 10]}
        remaining, value = separate_label(data, 'a')
        assert value == 30
        assert remaining['labels'] == ['b', 'c']
        assert remaining['values'] == [20, 10]

    def test_extracts_last_label(self):
        """separate_label works when target is the last label."""
        data: ChartData = {'labels': ['a', 'b', 'c'], 'values': [30, 20, 10]}
        remaining, value = separate_label(data, 'c')
        assert value == 10
        assert remaining['labels'] == ['a', 'b']
        assert remaining['values'] == [30, 20]


class TestSeparateLabelNotFound:
    """Tests for separate_label when target label is not in the data."""

    def test_returns_data_copy_when_label_not_found(self):
        """Returns a copy of the data (not original) when label is not found."""
        data: ChartData = {'labels': ['a', 'b', 'c'], 'values': [30, 20, 10]}
        remaining, value = separate_label(data, 'z')
        assert remaining['labels'] == ['a', 'b', 'c']
        assert remaining['values'] == [30, 20, 10]

    def test_returns_zero_when_label_not_found(self):
        """Returns 0 as the extracted value when label is not found."""
        data: ChartData = {'labels': ['a', 'b', 'c'], 'values': [30, 20, 10]}
        _, value = separate_label(data, 'missing')
        assert value == 0

    def test_does_not_raise_when_label_not_found(self):
        """No exception is raised when the label does not exist."""
        data: ChartData = {'labels': ['a', 'b'], 'values': [10, 20]}
        remaining, value = separate_label(data, 'nonexistent')
        assert value == 0

    def test_not_found_on_empty_data(self):
        """Returns empty data and 0 for empty input when label not found."""
        data: ChartData = {'labels': [], 'values': []}
        remaining, value = separate_label(data, 'anything')
        assert remaining['labels'] == []
        assert remaining['values'] == []
        assert value == 0
