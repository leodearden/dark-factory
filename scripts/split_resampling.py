"""Resampling draws over a routing split's tasks, and the cuts read off them.

Arms are lists of task indices. :func:`permuted_arms` relabels the pooled
tasks at the observed arm sizes (the A/A null); :func:`bootstrap_arms` resamples
each arm from its own members (an interval). Both draw from a caller's
``random.Random``, so one seed reproduces a whole report. Stdlib only.
"""
from __future__ import annotations

import itertools
import math
import random
import statistics
from collections.abc import Iterator, Sequence
from fractions import Fraction

__all__ = ['bootstrap_arms', 'null_cuts', 'percentile_interval', 'permuted_arms']


def permuted_arms(
    sizes: Sequence[int], resamples: int, rng: random.Random,
) -> Iterator[list[list[int]]]:
    """Per resample, every arm but the last, drawn from ``range(sum(sizes))`` at the observed sizes.

    Each resample is one random ordering of all but the last arm's worth of
    tasks, sliced into consecutive arms; the last arm is the complement, left
    to the caller. That is a permutation of the observed labels at half a
    shuffle's cost.
    """
    population = range(sum(sizes))
    bounds = list(itertools.accumulate(sizes[:-1], initial=0))
    for _ in range(resamples):
        order = rng.sample(population, bounds[-1])
        yield [order[low:high] for low, high in itertools.pairwise(bounds)]


def bootstrap_arms(
    members: Sequence[Sequence[int]], resamples: int, rng: random.Random,
) -> Iterator[list[list[int]]]:
    """Per resample, each arm redrawn with replacement from its own members, at its own size."""
    for _ in range(resamples):
        yield [rng.choices(arm, k=len(arm)) for arm in members]


def _refuse_too_few(values: Sequence[float]) -> None:
    if len(values) < 2:
        raise ValueError(f'a cut needs at least 2 values, got {len(values)}')


def null_cuts(values: Sequence[float]) -> tuple[float, float]:
    """The p95 and p99 of *values*, by inclusive (type-7) quantiles."""
    _refuse_too_few(values)
    cuts = statistics.quantiles(values, n=100, method='inclusive')
    return cuts[94], cuts[98]


def _type_7(ordered: Sequence[float], q: Fraction) -> float:
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + float(position - low) * (ordered[high] - ordered[low]))


def percentile_interval(values: Sequence[float], level: float) -> tuple[float, float]:
    """The two-sided percentile interval at *level*: type-7 quantiles at (1 - level)/2 and (1 + level)/2.

    *level* is read as the decimal it prints as, so 0.90 cuts exactly at the 5th
    and 95th percentiles rather than a binary rounding below the 5th.
    """
    if not 0 < level < 1:
        raise ValueError(f'an interval level lies strictly between 0 and 1, got {level!r}')
    _refuse_too_few(values)
    ordered = sorted(values)
    exact = Fraction(repr(level))
    return _type_7(ordered, (1 - exact) / 2), _type_7(ordered, (1 + exact) / 2)
