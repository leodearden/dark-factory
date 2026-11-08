"""Pure statistics over per-landing series: intervals, control charts, change points, cost fits.

Every function is deterministic for equal inputs and keeps no module-level state. The formulas and
their worked numbers are plans/quality-dashboard-prd.md §Contract "Pure statistics".
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Iterable, Sequence

__all__ = [
    'bootstrap_interval',
    'content_seed',
    'newcombe_difference',
    'wilson_interval',
]

Z_95 = 1.96


def _require(condition: bool, invariant: str, **values: object) -> None:
    if not condition:
        shown = ', '.join(f'{name}={value!r}' for name, value in values.items())
        raise ValueError(f'{invariant} ({shown})')


def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """The Wilson score interval for k successes in n trials, clamped to [0, 1]."""
    _require(n > 0 and 0 <= k <= n, 'a proportion needs 0 <= k <= n and n > 0', k=k, n=n)
    p = k / n
    z2 = z * z
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_difference(
    k_before: int, n_before: int, k_after: int, n_after: int, z: float = Z_95
) -> tuple[float, float, float]:
    """(d, lo, hi) for d = after rate − before rate, by Newcombe's hybrid Wilson interval."""
    lo_before, hi_before = wilson_interval(k_before, n_before, z)
    lo_after, hi_after = wilson_interval(k_after, n_after, z)
    p_before, p_after = k_before / n_before, k_after / n_after
    d = p_after - p_before
    lo = d - math.sqrt((p_after - lo_after) ** 2 + (hi_before - p_before) ** 2)
    hi = d + math.sqrt((hi_after - p_after) ** 2 + (p_before - lo_before) ** 2)
    return d, lo, hi


def content_seed(task_ids: Iterable[object]) -> int:
    """A bootstrap seed fixed by the window's content: equal id sets give equal bands."""
    joined = ','.join(sorted(str(task_id) for task_id in task_ids))
    return int.from_bytes(hashlib.sha256(joined.encode()).digest()[:8], 'big')


def _nearest_rank(ordered: Sequence[float], quantile: float) -> float:
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def bootstrap_interval(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float],
    seed: int,
    B: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """The percentile bootstrap interval of ``stat`` over B resamples drawn with Random(seed)."""
    _require(len(values) > 0, 'a bootstrap needs a non-empty sample', n=len(values))
    _require(B > 0 and 0 < alpha < 1, 'a bootstrap needs B > 0 and 0 < alpha < 1', B=B, alpha=alpha)
    rng = random.Random(seed)
    resampled = sorted(stat(rng.choices(values, k=len(values))) for _ in range(B))
    return _nearest_rank(resampled, alpha / 2), _nearest_rank(resampled, 1 - alpha / 2)
