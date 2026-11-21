"""The fused-memory scripts' shared home for interval statistics.

A sample rate, Wilson score bounds for a sample proportion and their 95%
interval, nearest-rank percentiles and their 95% interval, Cohen's kappa
between two raters, and a stratum-weighted ratio with a seeded bootstrap
interval that resamples within each stratum. Every reported figure is rounded
to four places, so a report built from them is byte-stable across runs; only
:func:`wilson_bounds` is unrounded, for callers with their own presentation.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass

_WILSON_Z = 1.96
_PLACES = 4


@dataclass(frozen=True)
class StratumUnit:
    """One sampled unit: its stratum and its contribution to a ratio's two sums."""

    stratum: str
    numerator: float
    denominator: float


def rate(hits: int, total: int) -> float | None:
    """*hits*/*total*, or None when nothing was measured; never 0.0 for an empty sample."""
    return round(hits / total, _PLACES) if total else None


def wilson_bounds(share: float, n: float, z: float = _WILSON_Z) -> tuple[float, float]:
    """Unrounded Wilson score bounds for an observed *share* over *n* trials, *n* possibly a non-integer effective n."""
    if n <= 0:
        raise ValueError(f'wilson_bounds needs a positive n; got n={n}')
    z2 = z * z
    denominator = 1 + z2 / n
    centre = (share + z2 / (2 * n)) / denominator
    half = z * math.sqrt(share * (1 - share) / n + z2 / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def wilson95(hits: int, total: int) -> list[float] | None:
    """Wilson score 95% interval for *hits*/*total*; None for an empty sample."""
    if not total:
        return None
    low, high = wilson_bounds(hits / total, total)
    return [round(low, _PLACES), round(high, _PLACES)]


def nearest_rank(sorted_values: Sequence[float], percent: float) -> float:
    """The smallest observed value with at least *percent*% of values at or below it."""
    if not sorted_values:
        raise ValueError('nearest_rank needs at least one value')
    rank = max(1, math.ceil(percent * len(sorted_values) / 100))
    return sorted_values[rank - 1]


def nearest_rank95(sorted_values: Sequence[float]) -> list[float]:
    """The 2.5th and 97.5th nearest-rank percentiles of *sorted_values*, to four places."""
    return [round(nearest_rank(sorted_values, 2.5), _PLACES), round(nearest_rank(sorted_values, 97.5), _PLACES)]


def cohen_kappa(pairs: Iterable[tuple[Hashable, Hashable]]) -> float | None:
    """Cohen's kappa over (first rater, second rater) label pairs.

    None when there are no pairs, or when chance agreement is certain (both
    raters gave one and the same label throughout), where kappa is undefined.
    """
    pairs = list(pairs)
    if not pairs:
        return None
    observed = sum(first == second for first, second in pairs) / len(pairs)
    firsts = Counter(first for first, _ in pairs)
    seconds = Counter(second for _, second in pairs)
    chance = sum(firsts[label] * seconds[label] for label in firsts) / len(pairs) ** 2
    if chance == 1:
        return None
    return round((observed - chance) / (1 - chance), _PLACES)


def stratified_ratio(
    units: Iterable[StratumUnit], weights: Mapping[str, float],
) -> float | None:
    """sum(w * numerator) / sum(w * denominator), w the unit's stratum weight.

    None when there are no units or the weighted denominator is 0. A unit
    whose stratum has no weight is refused, naming the stratum.
    """
    units = _weighted_units(units, weights)
    ratio = _weighted_ratio(units, weights)
    return None if ratio is None else round(ratio, _PLACES)


def stratified_bootstrap95(
    units: Iterable[StratumUnit], weights: Mapping[str, float], *, resamples: int, seed: int,
) -> list[float] | None:
    """Nearest-rank 95% interval of :func:`stratified_ratio` over seeded resamples.

    Each draw resamples every stratum's units with replacement, keeping each
    stratum's size, and recomputes the weighted ratio. The strata are drawn in
    name order from one ``random.Random(seed)``, so a seed always gives the
    same draws. Draws whose weighted denominator is 0 are dropped; None when no
    units are given or every draw is dropped.
    """
    units = _weighted_units(units, weights)
    if not units:
        return None
    by_stratum: dict[str, list[StratumUnit]] = defaultdict(list)
    for unit in units:
        by_stratum[unit.stratum].append(unit)
    names = sorted(by_stratum)
    numerators = [tuple(weights[name] * unit.numerator for unit in by_stratum[name]) for name in names]
    denominators = [tuple(weights[name] * unit.denominator for unit in by_stratum[name]) for name in names]
    draws = random.Random(seed)
    ratios: list[float] = []
    for _ in range(resamples):
        picks = [draws.choices(range(len(terms)), k=len(terms)) for terms in denominators]
        denominator = _resampled_sum(denominators, picks)
        if denominator:
            ratios.append(_resampled_sum(numerators, picks) / denominator)
    if not ratios:
        return None
    ratios.sort()
    return nearest_rank95(ratios)


def _resampled_sum(terms_by_stratum: Sequence[Sequence[float]], picks: Sequence[Sequence[int]]) -> float:
    """Each stratum's weighted terms at its picked indices, summed in stratum order."""
    return sum([
        terms[index] for terms, picked in zip(terms_by_stratum, picks, strict=True) for index in picked
    ])


def _weighted_units(
    units: Iterable[StratumUnit], weights: Mapping[str, float],
) -> list[StratumUnit]:
    units = list(units)
    missing = sorted({unit.stratum for unit in units} - set(weights))
    if missing:
        raise ValueError(f'no weight for strata {missing}; weighted strata: {sorted(weights)}')
    return units


def _weighted_ratio(units: Sequence[StratumUnit], weights: Mapping[str, float]) -> float | None:
    denominator = sum(weights[unit.stratum] * unit.denominator for unit in units)
    if not denominator:
        return None
    return sum(weights[unit.stratum] * unit.numerator for unit in units) / denominator
