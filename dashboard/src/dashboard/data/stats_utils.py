"""Statistical utilities for the dashboard data layer."""

from __future__ import annotations

import math
from collections.abc import Sequence


def percentile(sorted_values: Sequence[float], p: float) -> float:
    """Compute the p-th percentile from a pre-sorted list.

    Args:
        sorted_values: A list of floats sorted in ascending order.
        p: The percentile to compute, in the range [0, 100].

    Returns:
        The interpolated p-th percentile value, or 0.0 for an empty list.

    Note:
        The caller is responsible for sorting the input list before calling
        this function.  ``get_time_centiles`` (performance.py) pre-sorts its
        input.  ``_compute_latency_stats`` (merge_queue.py) receives data
        already sorted via SQL ``ORDER BY duration_ms`` from ``_get_durations``
        and passes it through unsorted.
    """
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)
