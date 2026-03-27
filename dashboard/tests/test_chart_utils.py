"""Tests for chart_utils utility functions."""

from __future__ import annotations

import pytest

from dashboard.data.chart_utils import group_top_n


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
