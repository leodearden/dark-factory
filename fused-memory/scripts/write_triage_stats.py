"""Interval statistics for the write-triage reader scorer (rubric v2).

A sample rate, Wilson 95% intervals for a sample proportion, nearest-rank
percentiles, Cohen's kappa between two raters, and a stratum-weighted ratio
with a seeded bootstrap interval that resamples within each stratum. Every
figure is rounded to four places, so a report built from them is byte-stable
across runs.
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


def wilson95(hits: int, total: int) -> list[float] | None:
    """Wilson score 95% interval for *hits*/*total*; None for an empty sample."""
    if not total:
        return None
    share = hits / total
    spread = _WILSON_Z * _WILSON_Z / total
    centre = (share + spread / 2) / (1 + spread)
    half = _WILSON_Z * math.sqrt(share * (1 - share) / total + spread / (4 * total)) / (1 + spread)
    return [round(max(0.0, centre - half), _PLACES), round(min(1.0, centre + half), _PLACES)]


def nearest_rank(sorted_values: Sequence[float], percent: float) -> float:
    """The smallest observed value with at least *percent*% of values at or below it."""
    if not sorted_values:
        raise ValueError('nearest_rank needs at least one value')
    rank = max(1, math.ceil(percent * len(sorted_values) / 100))
    return sorted_values[rank - 1]


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
    return [round(nearest_rank(ratios, 2.5), _PLACES), round(nearest_rank(ratios, 97.5), _PLACES)]


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
