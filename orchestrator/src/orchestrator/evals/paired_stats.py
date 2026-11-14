"""A seeded percentile bootstrap of the mean of paired differences, for the shadow-eval report.

The unit of replication is one pair, and one pair is one production task:
``shadow_cells``' unique ``(project_id, task_id, shape, candidate)`` index allows
at most one cell per task on a report line, so resampling pairs resamples tasks.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    'BOOTSTRAP_RESAMPLES',
    'BOOTSTRAP_SEED',
    'PairedDifference',
    'bootstrap_mean_ci',
]

BOOTSTRAP_RESAMPLES = 10_000
# Fixed so the same rows always print the same interval. Deliberately not
# ShadowEvalConfig.seed, which seeds cell sampling: a different purpose.
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class PairedDifference:
    """The mean paired difference over ``n`` pairs and its interval (none below two pairs)."""

    n: int
    mean: float | None
    ci_low: float | None
    ci_high: float | None

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError(f'PairedDifference: n must be >= 0 (got {self.n!r})')
        if (self.mean is None) != (self.n == 0):
            raise ValueError(
                f'PairedDifference: a mean exists exactly when there are pairs '
                f'(got n={self.n!r}, mean={self.mean!r})'
            )
        has_interval = (self.ci_low is not None, self.ci_high is not None)
        if has_interval != (self.n >= 2,) * 2:
            raise ValueError(
                f'PairedDifference: an interval exists exactly when there are two or more '
                f'pairs (got n={self.n!r}, ci=({self.ci_low!r}, {self.ci_high!r}))'
            )


def bootstrap_mean_ci(
    differences: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> PairedDifference:
    """The mean of *differences* with a percentile-bootstrap interval over pairs."""
    diffs = [_finite(value) for value in differences]
    slices = _quantile_slices(confidence)
    n = len(diffs)
    if n == 0:
        return PairedDifference(0, None, None, None)
    mean = statistics.fmean(diffs)
    if n == 1:
        return PairedDifference(1, mean, None, None)
    rng = random.Random(seed)
    resample_means = [statistics.fmean(rng.choices(diffs, k=n)) for _ in range(resamples)]
    cuts = statistics.quantiles(resample_means, n=slices, method='inclusive')
    return PairedDifference(n, mean, cuts[0], cuts[-1])


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f'bootstrap_mean_ci: every difference must be finite (got {value!r})')
    return value


def _quantile_slices(confidence: float) -> int:
    """How many equal quantile slices put one whole slice in each tail of *confidence*.

    The interval's endpoints are then the first and last cut points, so 0.95
    maps to 40 slices (2.5% per tail); a confidence whose tail is not 1/k of
    the distribution for a whole k is refused rather than rounded.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f'bootstrap_mean_ci: confidence must lie in (0, 1) (got {confidence!r})')
    slices = round(2.0 / (1.0 - confidence))
    if slices < 2 or not math.isclose(slices * (1.0 - confidence), 2.0):
        raise ValueError(
            f'bootstrap_mean_ci: confidence {confidence!r} leaves a tail of '
            f'{(1.0 - confidence) / 2!r}, which is not 1/k of the distribution for a whole k'
        )
    return slices
