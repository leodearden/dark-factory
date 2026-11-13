"""Pure statistics over per-landing series: intervals, control charts, change points, cost fits.

Every function is deterministic for equal inputs and keeps no module-level state. The formulas and
their worked numbers are plans/quality-dashboard-prd.md §Contract "Pure statistics".
"""

from __future__ import annotations

import bisect
import hashlib
import math
import random
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

__all__ = [
    'CUSUM_H',
    'BaselineState',
    'CostEffect',
    'CusumDesign',
    'Effect',
    'EwmaLimits',
    'NotYet',
    'OccurrenceCap',
    'RateDatum',
    'RateState',
    'RegimeEvent',
    'RegimeSegments',
    'Segment',
    'baseline_rate',
    'before_after_cost',
    'before_after_rate',
    'block_rate_datum',
    'bootstrap_interval',
    'cap_occurrences',
    'content_seed',
    'cost_fit',
    'cusum_alarms',
    'cusum_design',
    'ewma_limits',
    'ewma_series',
    'explain_alarm',
    'log_residuals',
    'newcombe_difference',
    'reference_cost',
    'regime_segments',
    'theil_sen',
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

    def __post_init__(self) -> None:
        if self.artefact:
            _require(self.until is not None and self.until >= self.at,
                     'an artefact is a range ending at or after its start',
                     event_id=self.event_id, at=self.at, until=self.until)

    def applies_to(self, series: str) -> bool:
        return self.series is None or series in self.series


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


def theil_sen(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """(a, b) for y = a + b·x: b the median pairwise slope over distinct-x pairs, a = median(y − b·x)."""
    _require(len(xs) == len(ys), 'xs and ys pair one to one', len_xs=len(xs), len_ys=len(ys))
    slopes = [
        (ys[j] - ys[i]) / (xs[j] - xs[i])
        for i in range(len(xs))
        for j in range(i + 1, len(xs))
        if xs[j] != xs[i]
    ]
    _require(len(slopes) > 0, 'a slope needs at least two distinct xs', distinct_xs=len(set(xs)))
    b = statistics.median(slopes)
    a = statistics.median(y - b * x for x, y in zip(xs, ys, strict=True))
    return a, b


def cost_fit(
    lines: Sequence[int], costs: Sequence[float], min_lines: int = 5
) -> tuple[float, float]:
    """Theil–Sen on (ln lines, ln cost) over points with lines >= min_lines and cost > 0."""
    points = _log_points(lines, costs, min_lines)
    return theil_sen([x for x, _ in points], [y for _, y in points])


def _log_points(
    lines: Sequence[int], costs: Sequence[float], min_lines: int
) -> list[tuple[float, float]]:
    _require(len(lines) == len(costs), 'lines and costs pair one to one',
             len_lines=len(lines), len_costs=len(costs))
    return [(math.log(n), math.log(c)) for n, c in zip(lines, costs, strict=True)
            if n >= min_lines and c > 0]


def reference_cost(
    lines: Sequence[int], costs: Sequence[float], ref: float = 100, min_lines: int = 5
) -> float:
    """What a ``ref``-line landing costs under this window's log–log fit."""
    a, b = cost_fit(lines, costs, min_lines)
    return math.exp(a + b * math.log(ref))


def log_residuals(
    lines: Sequence[int], costs: Sequence[float], a: float, b: float
) -> list[float | None]:
    """ln(cost) − (a + b·ln lines) per landing; None where lines or cost is not positive."""
    return [
        math.log(c) - (a + b * math.log(n)) if n > 0 and c > 0 else None
        for n, c in zip(lines, costs, strict=True)
    ]


@dataclass(frozen=True)
class OccurrenceCap:
    """Per-landing bearing flags, and the landings whose occurrence count exceeds ``cap``."""

    bearing: tuple[bool, ...]
    mass_event_indices: tuple[int, ...]
    cap: float


def cap_occurrences(
    counts: Sequence[int], baseline_median: float, floor: float = 20
) -> OccurrenceCap:
    """A landing bears the outcome when its count is positive; above max(floor, 10·median) it is a mass event."""
    cap = max(floor, 10 * baseline_median)
    return OccurrenceCap(
        bearing=tuple(count > 0 for count in counts),
        mass_event_indices=tuple(i for i, count in enumerate(counts) if count > cap),
        cap=cap,
    )


@dataclass(frozen=True)
class Segment:
    """Landings [start, end) since the regime event ``event_id`` (None: since the first landing)."""

    start: int
    end: int
    event_id: str | None


@dataclass(frozen=True)
class RegimeSegments:
    segments: tuple[Segment, ...]
    excluded: frozenset[int]


def regime_segments(
    landed_at: Sequence[datetime], events: Iterable[RegimeEvent], series: str
) -> RegimeSegments:
    """Split the landings at each regime event for ``series``; artefact ranges become ``excluded``.

    An event's boundary is the first landing at or after its ``at``; empty segments are dropped.
    """
    relevant = [event for event in events if event.applies_to(series)]
    boundaries = sorted(
        (_landing_index(landed_at, event.at), event.at, event.event_id)
        for event in relevant
        if not event.artefact
    )
    starts: list[tuple[int, str | None]] = [(0, None)]
    starts.extend((index, event_id) for index, _, event_id in boundaries)
    ends = [start for start, _ in starts[1:]] + [len(landed_at)]
    segments = tuple(
        Segment(start, end, event_id)
        for (start, event_id), end in zip(starts, ends, strict=True)
        if start < end
    )
    excluded: set[int] = set()
    for event in relevant:
        if event.artefact and event.until is not None:
            excluded.update(range(_landing_index(landed_at, event.at),
                                  bisect.bisect_right(landed_at, event.until)))
    return RegimeSegments(segments=segments, excluded=frozenset(excluded))


class BaselineState(StrEnum):
    FROZEN = 'frozen'
    PROVISIONAL = 'provisional'


def _measured(xs: Iterable[int | None]) -> list[int]:
    return [x for x in xs if x is not None]


def baseline_rate(
    segment_xs: Sequence[int | None], n_base: int = 200, n_min: int = 100
) -> tuple[float | None, BaselineState | None]:
    """p0 over the first n_base measured landings of a segment: frozen at n_base, provisional from n_min."""
    measured = _measured(segment_xs)
    if len(measured) < n_min:
        return None, None
    window = measured[:n_base]
    state = BaselineState.FROZEN if len(measured) >= n_base else BaselineState.PROVISIONAL
    return sum(window) / len(window), state


class RateState(StrEnum):
    RATE = 'rate'
    K_OF_N = 'k_of_n'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class RateDatum:
    """k of n over the block; rate, lo and hi are None and ``reason`` says why when UNKNOWN."""

    k: int
    n: int
    rate: float | None
    lo: float | None
    hi: float | None
    state: RateState
    reason: str | None
    window_task_ids: tuple[str, ...]


def block_rate_datum(
    xs: Sequence[int | None], task_ids: Sequence[str], n_block: int = 100, n_min: int = 30
) -> RateDatum:
    """The headline rate over the last n_block measured landings; unmeasured landings never count."""
    _require(len(xs) == len(task_ids), 'xs and task_ids pair one to one',
             len_xs=len(xs), len_task_ids=len(task_ids))
    measured = [(x, task_id) for x, task_id in zip(xs, task_ids, strict=True) if x is not None]
    window = measured[-n_block:]
    k = sum(x for x, _ in window)
    n = len(window)
    ids = tuple(task_id for _, task_id in window)
    if n < n_min:
        reason = f'n={n} measured landings; a rate needs >= {n_min}'
        return RateDatum(k, n, None, None, None, RateState.UNKNOWN, reason, ids)
    lo, hi = wilson_interval(k, n)
    state = RateState.RATE if n >= n_block else RateState.K_OF_N
    return RateDatum(k, n, k / n, lo, hi, state, None, ids)


COMPARISON_WINDOW = 300


@dataclass(frozen=True)
class Effect:
    """d = after rate − before rate with its Newcombe interval."""

    d: float
    lo: float
    hi: float
    n_before: int
    n_after: int


@dataclass(frozen=True)
class CostEffect:
    """after/before ratio of size-corrected cost with its bootstrap interval."""

    ratio: float
    lo: float
    hi: float
    n_before: int
    n_after: int


@dataclass(frozen=True)
class NotYet:
    """Too few landings after the event; min_detectable is the resolvable rate difference (rates only)."""

    n_before: int
    n_after: int
    eta_days: float | None
    min_detectable: float | None


def _eta_days(n_after: int, n_min: int, landings_per_day: float | None) -> float | None:
    if landings_per_day is None:
        return None
    _require(landings_per_day > 0, 'a landing velocity is positive', landings_per_day=landings_per_day)
    return (n_min - n_after) / landings_per_day


def _min_detectable_rate(p_before: float, n_min: int) -> float:
    k = round(p_before * n_min)
    _, lo, hi = newcombe_difference(k, n_min, k, n_min)
    return (hi - lo) / 2


def before_after_rate(
    xs_before: Sequence[int | None],
    xs_after: Sequence[int | None],
    n_min: int = 100,
    landings_per_day: float | None = None,
) -> Effect | NotYet:
    """Newcombe difference over the last 300 measured before and the first 300 measured after."""
    before = _measured(xs_before)[-COMPARISON_WINDOW:]
    after = _measured(xs_after)[:COMPARISON_WINDOW]
    _require(len(before) > 0, 'a comparison needs a measured landing before the event',
             n_before=len(before))
    if len(after) < n_min:
        return NotYet(
            n_before=len(before),
            n_after=len(after),
            eta_days=_eta_days(len(after), n_min, landings_per_day),
            min_detectable=_min_detectable_rate(sum(before) / len(before), n_min),
        )
    d, lo, hi = newcombe_difference(sum(before), len(before), sum(after), len(after))
    return Effect(d, lo, hi, len(before), len(after))


def before_after_cost(
    before_lines: Sequence[int],
    before_costs: Sequence[float],
    after_lines: Sequence[int],
    after_costs: Sequence[float],
    seed: int,
    B: int = 1000,
    n_min: int = 100,
    min_lines: int = 5,
    landings_per_day: float | None = None,
    alpha: float = 0.05,
) -> CostEffect | NotYet:
    """Ratio of size-corrected cost after/before at the before window's Theil–Sen slope.

    ratio = exp(median(after y − b·x) − median(before y − b·x)) on (ln lines, ln cost); the band
    resamples both windows B times with Random(seed). Windows are capped as in before_after_rate.
    """
    before = _log_points(before_lines, before_costs, min_lines)[-COMPARISON_WINDOW:]
    after = _log_points(after_lines, after_costs, min_lines)[:COMPARISON_WINDOW]
    if len(after) < n_min:
        return NotYet(len(before), len(after), _eta_days(len(after), n_min, landings_per_day), None)
    _, b = theil_sen([x for x, _ in before], [y for _, y in before])
    before_r = [y - b * x for x, y in before]
    after_r = [y - b * x for x, y in after]
    _require(B > 0 and 0 < alpha < 1, 'a bootstrap needs B > 0 and 0 < alpha < 1', B=B, alpha=alpha)
    rng = random.Random(seed)
    resampled = sorted(
        math.exp(
            statistics.median(rng.choices(after_r, k=len(after_r)))
            - statistics.median(rng.choices(before_r, k=len(before_r)))
        )
        for _ in range(B)
    )
    ratio = math.exp(statistics.median(after_r) - statistics.median(before_r))
    return CostEffect(
        ratio=ratio,
        lo=_nearest_rank(resampled, alpha / 2),
        hi=_nearest_rank(resampled, 1 - alpha / 2),
        n_before=len(before),
        n_after=len(after),
    )
