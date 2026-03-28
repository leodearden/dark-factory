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

    # Sort descending by value so top-N selection is deterministic regardless
    # of caller ordering. Timsort is O(n) on already-sorted input.
    sorted_pairs = sorted(zip(labels, values, strict=True), key=lambda pair: pair[1], reverse=True)
    labels = [pair[0] for pair in sorted_pairs]
    values = [pair[1] for pair in sorted_pairs]

    if len(labels) <= n:
        return {'labels': list(labels), 'values': list(values)}

    top_labels = list(labels[:n])
    top_values = list(values[:n])
    other_sum = sum(values[n:])

    return {
        'labels': top_labels + ['Other'],
        'values': top_values + [other_sum],
    }


def separate_label(
    data: ChartData, label: str
) -> tuple[ChartData, int | float]:
    """Extract a single label/value pair from ChartData by label name.

    Finds the first occurrence of label in data['labels'], removes it from
    both lists, and returns the remaining data along with the extracted value.
    The original data dict is not mutated.

    Args:
        data: ChartData with 'labels' and 'values' lists.
        label: The label string to find and extract.

    Returns:
        A tuple of (remaining_data, extracted_value). If label is not found,
        returns a shallow copy of data and 0.

    Raises:
        ValueError: If labels and values have different lengths.
    """
    labels: list[str] = list(data.get('labels', []))
    values: list[int | float] = list(data.get('values', []))

    if len(labels) != len(values):
        raise ValueError(
            f"labels and values must have the same length "
            f"(got {len(labels)} labels and {len(values)} values)"
        )

    if label not in labels:
        return {'labels': labels, 'values': values}, 0

    idx = labels.index(label)

    extracted_value = values[idx]
    remaining_labels = labels[:idx] + labels[idx + 1:]
    remaining_values = values[:idx] + values[idx + 1:]

    return {'labels': remaining_labels, 'values': remaining_values}, extracted_value
