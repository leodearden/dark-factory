"""Utility functions for chart data preparation."""

from __future__ import annotations

from typing import TypedDict


class ChartData(TypedDict):
    """Typed shape for chart label/value pairs used across the dashboard."""

    labels: list[str]
    values: list[int | float]


def group_top_n(data: ChartData, n: int = 5) -> ChartData:
    """Group chart data into top N entries plus an 'Other' aggregate.

    Takes a ChartData dict with 'labels' and 'values' lists and returns a new
    ChartData. When the number of items exceeds N, the tail items are summed
    into a single 'Other' entry appended at the end. When items <= N, the data
    is returned with values sorted descending.

    Args:
        data: ChartData with 'labels' (list[str]) and 'values' (list[int|float]).
        n: Maximum number of top items to keep before grouping. Default 5.

    Returns:
        ChartData with 'labels' and 'values', with at most N+1 entries.

    Raises:
        ValueError: If labels and values have different lengths.
    """
    labels: list = data.get('labels', [])
    values: list = data.get('values', [])

    if len(labels) != len(values):
        raise ValueError(
            f"labels and values must have the same length "
            f"(got {len(labels)} labels and {len(values)} values)"
        )

    if len(labels) <= n:
        return {'labels': list(labels), 'values': list(values)}

    top_labels = list(labels[:n])
    top_values = list(values[:n])
    other_sum = sum(values[n:])

    return {
        'labels': top_labels + ['Other'],
        'values': top_values + [other_sum],
    }
