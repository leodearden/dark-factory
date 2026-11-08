"""Pure statistics over per-landing series: intervals, control charts, change points, cost fits.

Every function is deterministic for equal inputs and keeps no module-level state. The formulas and
their worked numbers are plans/quality-dashboard-prd.md §Contract "Pure statistics".
"""

from __future__ import annotations

import bisect
import hashlib
import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

__all__ = [
    'CUSUM_H',
    'CusumDesign',
    'EwmaLimits',
    'RegimeEvent',
    'bootstrap_interval',
    'content_seed',
    'cusum_alarms',
    'cusum_design',
    'ewma_limits',
    'ewma_series',
    'explain_alarm',
    'newcombe_difference',
    'wilson_interval',
]

Z_95 = 1.96
CUSUM_H = 5.0


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


def ewma_series(xs: Sequence[float], lam: float, z0: float) -> list[float]:
    """z_t = lam * x_t + (1 - lam) * z_{t-1}, starting from z0."""
    _require(0 < lam <= 1, 'an EWMA weight lies in (0, 1]', lam=lam)
    out: list[float] = []
    z = z0
    for x in xs:
        z = lam * x + (1 - lam) * z
        out.append(z)
    return out


@dataclass(frozen=True)
class EwmaLimits:
    """Control limits at one landing; lcl is None when the lower limit is not above zero."""

    lcl: float | None
    ucl: float


def ewma_limits(p0: float, lam: float, L: float, t: int) -> EwmaLimits:
    """p0 ± L·σ_t with the exact variance σ_t² = p0(1−p0)·λ/(2−λ)·(1−(1−λ)^{2t})."""
    _require(0 < p0 < 1, 'a baseline rate lies in (0, 1)', p0=p0)
    _require(0 < lam <= 1 and L > 0 and t >= 1, 'EWMA limits need 0 < lam <= 1, L > 0, t >= 1',
             lam=lam, L=L, t=t)
    variance = p0 * (1 - p0) * lam / (2 - lam) * (1 - (1 - lam) ** (2 * t))
    spread = L * math.sqrt(variance)
    lower = p0 - spread
    return EwmaLimits(lcl=lower if lower > 0 else None, ucl=p0 + spread)


def _require_probability_pair(p0: float, p1: float) -> None:
    _require(0 < p0 < 1 and 0 < p1 < 1 and p0 != p1,
             'a CUSUM shift needs p0 and p1 in (0, 1) and p1 != p0', p0=p0, p1=p1)


def cusum_alarms(xs: Sequence[int], p0: float, p1: float, h: float) -> list[tuple[int, int]]:
    """One-sided Bernoulli CUSUM for p0 -> p1: (alarm index, onset index) per alarm.

    S_t = max(0, S_{t-1} + llr_t); an alarm fires when S_t > h and resets S to 0. The onset is the
    first index of the excursion that crossed h, i.e. the observation after the last zero of S.
    """
    _require_probability_pair(p0, p1)
    _require(h > 0, 'a CUSUM threshold is positive', h=h)
    llr_one = math.log(p1 / p0)
    llr_zero = math.log((1 - p1) / (1 - p0))
    alarms: list[tuple[int, int]] = []
    s = 0.0
    onset = 0
    for index, x in enumerate(xs):
        if s == 0.0:
            onset = index
        s = max(0.0, s + (llr_one if x else llr_zero))
        if s > h:
            alarms.append((index, onset))
            s = 0.0
    return alarms


@dataclass(frozen=True)
class CusumDesign:
    p1_up: float
    p1_down: float
    h: float


def cusum_design(p0: float) -> CusumDesign:
    """A doubling (capped so it stays a probability above p0 = 1/3) and a halving, at h = CUSUM_H."""
    _require(0 < p0 < 1, 'a baseline rate lies in (0, 1)', p0=p0)
    return CusumDesign(p1_up=min(2 * p0, (1 + p0) / 2), p1_down=p0 / 2, h=CUSUM_H)


@dataclass(frozen=True)
class RegimeEvent:
    """A dated change: a regime boundary, or (artefact=True) a range [at, until] to exclude.

    series None means the event applies to every series.
    """

    event_id: str
    at: datetime
    until: datetime | None = None
    series: tuple[str, ...] | None = None
    artefact: bool = False


def _landing_index(landed_at: Sequence[datetime], at: datetime) -> int:
    return bisect.bisect_left(landed_at, at)


def explain_alarm(
    onset_idx: int,
    onset_at: datetime,
    events: Iterable[RegimeEvent],
    landed_at: Sequence[datetime],
    k: int = 25,
    days: float = 3,
) -> str | None:
    """The event within ±k landings or ±days of the onset that explains an alarm, else None.

    An event's landing index is the first landing at or after its ``at``. Among qualifying events
    the nearest in landings wins, then the nearest in time, then the smallest event_id.
    """
    window = timedelta(days=days)
    candidates = []
    for event in events:
        landings_apart = abs(_landing_index(landed_at, event.at) - onset_idx)
        time_apart = abs(event.at - onset_at)
        if landings_apart <= k or time_apart <= window:
            candidates.append((landings_apart, time_apart, event.event_id))
    return min(candidates)[2] if candidates else None
