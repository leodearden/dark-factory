"""Utility functions for chart data preparation."""

from __future__ import annotations

from typing import TypedDict


class ChartData(TypedDict):
    """Typed shape for chart label/value pairs used across the dashboard."""

    labels: list[str]
    values: list[int | float]


def group_top_n(data: dict, n: int = 5) -> dict:
    """Group chart data into top N entries plus an 'Other' aggregate.

    Takes a dict with 'labels' and 'values' lists (already sorted descending
    by convention) and returns a new dict. When the number of items exceeds N,
    the tail items are summed into a single 'Other' entry appended at the end.
    When items <= N, the data is returned unchanged.

    Args:
        data: Dict with 'labels' (list[str]) and 'values' (list[int|float]).
        n: Maximum number of top items to keep before grouping. Default 5.

    Returns:
        Dict with 'labels' and 'values', with at most N+1 entries.
    """
    labels: list = data.get('labels', [])
    values: list = data.get('values', [])

    if len(labels) <= n:
        return {'labels': list(labels), 'values': list(values)}

    top_labels = list(labels[:n])
    top_values = list(values[:n])
    other_sum = sum(values[n:])

    return {
        'labels': top_labels + ['Other'],
        'values': top_values + [other_sum],
    }
