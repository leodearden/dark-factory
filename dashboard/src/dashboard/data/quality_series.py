"""One project's per-landing quality series: the axis, each metric's points, bands, baseline and alarms.

The one access path for per-landing quality facts: read_quality_series serves what the quality
sampler stored in quality.db, with the freshness its samples and runs.db's recorded landings
support, and never runs git or computes a row (plans/quality-dashboard-prd.md decisions 13, 14,
15 and 22). Every number comes from shared/src/shared/quality_series/stats.py (decision 21); this
module only chooses what each statistic is applied to and shapes the wire (§Contract "Wire
shape"). Statistics run over the project's full stored history; the window only slices what is
served.
"""

from __future__ import annotations

import asyncio
import enum
import sqlite3
import statistics
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import TypeVar

import aiosqlite
from shared.project_id import project_id_for_root
from shared.project_layout import runs_db
from shared.quality_series.landings import (
    GitCrossCheck,
    UnrecognisedTrainEvent,
    recorded_landing_keys,
)
from shared.quality_series.rows import (
    CharterClassSource,
    LandingRow,
    SourceEpochs,
    module_allocation,
    unit_heads,
)
from shared.quality_series.stats import (
    BaselineState,
    CostEffect,
    Effect,
    NotYet,
    RateState,
    RegimeEvent,
    baseline_rate,
    before_after_cost,
    before_after_rate,
    block_rate_datum,
    bootstrap_interval,
    content_seed,
    cost_fit,
    cusum_alarms,
    cusum_design,
    ewma_limits,
    ewma_series,
    explain_alarm,
    log_residuals,
    reference_cost,
    regime_segments,
    wilson_interval,
)
from shared.tasks_db_readonly import TaskDbUnreadable, connect_ro

from dashboard.data.datum import (
    Datum,
    DatumContractError,
    DatumInvariant,
    DatumState,
    aged_at,
    unknown_datum,
    validate_datum,
)
from dashboard.data.quality_sampler import QUALITY_SAMPLE_INTERVAL_SECONDS, describe_unreadable
from dashboard.data.quality_store import (
    QualitySample,
    SampleState,
    newest_samples,
    read_landing_rows,
)

__all__ = [
    'QUALITY_FRESHNESS_BOUND_SECONDS',
    'Alarm',
    'AlarmDirection',
    'AxisPoint',
    'Baseline',
    'BeforeAfter',
    'ComparisonState',
    'CostModel',
    'Ewma',
    'Headline',
    'HeadlineForm',
    'MetricKind',
    'MetricSeries',
    'MetricSource',
    'ModuleRow',
    'Provenance',
    'QualitySeries',
    'QualitySeriesContractError',
    'QualitySeriesInvariant',
    'Rolling',
    'Rule',
    'quality_series_view',
    'read_quality_series',
    'validate_quality_series',
]

T = TypeVar('T')

QUALITY_FRESHNESS_BOUND_SECONDS = 2 * QUALITY_SAMPLE_INTERVAL_SECONDS

HEADLINE_LANDINGS = 100
"""The headline block and the rolling band: the last this-many measured landings."""
MIN_MEASURED = 30
"""Below this many measured landings a headline is unknown."""
EWMA_LAMBDA = 0.05
EWMA_SIGMAS = 3.0
NO_PRODUCER = 'no producer has written here'
COST_METRIC = 'all_role_usd'
COST_BASELINE_UNITS = 200
COST_MIN_BASELINE_UNITS = 100
COST_MIN_LINES = 5
SIGN_P0 = 0.5
"""A residual's sign against the baseline's median residual is a fair coin while cost holds."""
RECENT = timedelta(days=28)
"""The module table's window and the landing velocity's, ending at the newest landing."""
MODULE_MIN_LINES = 1000
MODULE_MIN_TASKS = 3


def _iso(at: datetime | None) -> str | None:
    return None if at is None else at.astimezone(UTC).isoformat()


class MetricKind(StrEnum):
    RATE = 'rate'
    COUNT = 'count'
    COST = 'cost'


class MetricSource(StrEnum):
    """Where a metric is measured from: a runs.db source, or a store with no writer yet."""

    REJECTIONS = 'rejections'
    MERGE_VERIFY = 'merge_verify'
    REVIEW_COMPLETED = 'review_completed'
    FLAKE_OCCURRENCE = 'flake_occurrence'
    INVOCATION_TURNS = 'invocation_turns'
    INVOCATIONS = 'invocations'
    TASK_RESULTS = 'task_results'
    CHARTER_CLASS = 'charter_class'
    LATE_EVENTS = 'late_events'


_EPOCH: dict[MetricSource, Callable[[SourceEpochs], datetime | None]] = {
    MetricSource.REJECTIONS: lambda epochs: epochs.rejections,
    MetricSource.MERGE_VERIFY: lambda epochs: epochs.merge_verify,
    MetricSource.REVIEW_COMPLETED: lambda epochs: epochs.review_completed,
    MetricSource.FLAKE_OCCURRENCE: lambda epochs: epochs.flake_occurrence,
    MetricSource.INVOCATION_TURNS: lambda epochs: epochs.invocation_turns,
    MetricSource.INVOCATIONS: lambda epochs: epochs.invocations,
}

_RECORDED_IN: dict[MetricSource, str] = {
    MetricSource.REJECTIONS: 'runs.db merge_blocked',
    MetricSource.MERGE_VERIFY: 'runs.db merge_verify',
    MetricSource.REVIEW_COMPLETED: 'runs.db review_completed',
    MetricSource.FLAKE_OCCURRENCE: 'runs.db flake_occurrence',
    MetricSource.INVOCATION_TURNS: 'runs.db invocation_end turns',
    MetricSource.INVOCATIONS: 'runs.db invocations',
    MetricSource.TASK_RESULTS: 'runs.db task_results.execute_iterations',
    MetricSource.CHARTER_CLASS: 'runs.db task_results.charter_class',
    MetricSource.LATE_EVENTS: 'quality.db late_events',
}


class HeadlineForm(StrEnum):
    RATE = 'rate'
    K_OF_N = 'k_of_n'
    MEAN = 'mean'
    MEDIAN = 'median'


class AlarmDirection(StrEnum):
    UP = 'up'
    DOWN = 'down'


class ComparisonState(StrEnum):
    MEASURED = 'measured'
    NOT_YET = 'not_yet'


@dataclass(frozen=True)
class Provenance:
    """What the series and every measured headline inherit from the sample it is read at, as
    that sample reads at *served_at*."""

    as_of: datetime
    state: DatumState
    reason: str | None
    freshness_bound_seconds: int
    served_at: datetime

    def measured(self, value: T) -> Datum[T]:
        datum = Datum(value, self.as_of, self.state, self.reason, self.freshness_bound_seconds)
        return aged_at(datum, self.served_at)

    def unknown(self, reason: str) -> Datum[Headline]:
        return unknown_datum(reason, self.freshness_bound_seconds)


@dataclass(frozen=True)
class AxisPoint:
    ordinal: int
    merge_sha: str
    task_id: str
    landed_at: datetime
    unit_id: str
    git_first_parent: bool
    late_claim: bool

    @classmethod
    def of(cls, row: LandingRow) -> AxisPoint:
        landing = row.landing
        return cls(landing.ordinal, landing.merge_sha, landing.task_id, landing.landed_at,
                   landing.unit_id, landing.git_first_parent, landing.late_claim)

    def to_wire(self) -> dict[str, object]:
        return {
            'ordinal': self.ordinal, 'merge_sha': self.merge_sha, 'task_id': self.task_id,
            'landed_at': _iso(self.landed_at), 'unit_id': self.unit_id,
            'git_first_parent': self.git_first_parent, 'late_claim': self.late_claim,
        }


@dataclass(frozen=True)
class Rule:
    """What a metric counts, over which landings, since when, from which source (decision 22)."""

    counts: str
    excludes: str
    population: str
    window_landings: int
    measured_since: datetime | None
    source: MetricSource

    def to_wire(self) -> dict[str, object]:
        return {
            'counts': self.counts, 'excludes': self.excludes, 'population': self.population,
            'window_landings': self.window_landings, 'measured_since': _iso(self.measured_since),
            'source': self.source.value,
        }


@dataclass(frozen=True)
class Headline:
    """The tile's figure with its 95% band; k is the event count of a rate, None otherwise."""

    form: HeadlineForm
    n: int
    k: int | None
    value: float
    lo: float
    hi: float

    def to_wire(self) -> dict[str, object]:
        return {'form': self.form.value, 'n': self.n, 'k': self.k, 'value': self.value,
                'lo': self.lo, 'hi': self.hi}


@dataclass(frozen=True)
class Rolling:
    """Per axis point over the last measured landings up to it; None below MIN_MEASURED.

    A rate's band is its Wilson interval; a count or a cost serves its centre alone.
    """

    center: tuple[float | None, ...]
    lo: tuple[float | None, ...]
    hi: tuple[float | None, ...]
    n: tuple[int, ...]

    def sliced(self, start: int) -> Rolling:
        return Rolling(self.center[start:], self.lo[start:], self.hi[start:], self.n[start:])

    def to_wire(self) -> dict[str, object]:
        return {'center': list(self.center), 'lo': list(self.lo), 'hi': list(self.hi),
                'n': list(self.n)}


@dataclass(frozen=True)
class Ewma:
    line: tuple[float | None, ...]
    ucl: tuple[float | None, ...]

    def to_wire(self) -> dict[str, object]:
        return {'line': list(self.line), 'ucl': list(self.ucl)}


@dataclass(frozen=True)
class Alarm:
    ordinal: int
    onset_ordinal: int
    direction: AlarmDirection
    explained_by: str | None

    def to_wire(self) -> dict[str, object]:
        return {'ordinal': self.ordinal, 'onset_ordinal': self.onset_ordinal,
                'direction': self.direction.value, 'explained_by': self.explained_by}


@dataclass(frozen=True)
class Baseline:
    """The reference rate in force at the newest landing, and the ordinal it was measured from."""

    p0: float | None
    state: BaselineState | None
    since_ordinal: int | None

    def to_wire(self) -> dict[str, object]:
        return {'p0': self.p0, 'state': None if self.state is None else self.state.value,
                'since_ordinal': self.since_ordinal}


NO_BASELINE = Baseline(None, None, None)


@dataclass(frozen=True)
class CostModel:
    """The log–log fit of unit cost on unit lines in force at the newest unit; absent: all None."""

    a: float | None
    b: float | None
    reference_cost_100: float | None
    residual_median: float | None
    state: BaselineState | None

    def to_wire(self) -> dict[str, object]:
        return {'a': self.a, 'b': self.b, 'reference_cost_100': self.reference_cost_100,
                'residual_median': self.residual_median,
                'state': None if self.state is None else self.state.value}


NO_COST_MODEL = CostModel(None, None, None, None, None)


@dataclass(frozen=True)
class ModuleRow:
    """A module's share of the recent units' lines and cost; module None is the no-src bucket."""

    module: str | None
    landings: int
    tasks: int
    allocated_lines: float
    allocated_usd: float
    usd_per_100_lines: float | None
    ranked: bool

    def to_wire(self) -> dict[str, object]:
        return {'module': self.module, 'landings': self.landings, 'tasks': self.tasks,
                'allocated_lines': self.allocated_lines, 'allocated_usd': self.allocated_usd,
                'usd_per_100_lines': self.usd_per_100_lines, 'ranked': self.ranked}


@dataclass(frozen=True)
class BeforeAfter:
    """One marker's effect on one metric: a rate difference or a cost ratio, or not yet."""

    marker_id: str
    metric: str
    n_before: int
    n_after: int
    effect: float | None
    lo: float | None
    hi: float | None
    state: ComparisonState
    eta_days: float | None
    min_detectable: float | None

    @classmethod
    def of(cls, marker_id: str, metric: str, result: Effect | CostEffect | NotYet) -> BeforeAfter:
        if isinstance(result, NotYet):
            return cls(marker_id, metric, result.n_before, result.n_after, None, None, None,
                       ComparisonState.NOT_YET, result.eta_days, result.min_detectable)
        effect = result.d if isinstance(result, Effect) else result.ratio
        return cls(marker_id, metric, result.n_before, result.n_after, effect, result.lo,
                   result.hi, ComparisonState.MEASURED, None, None)

    def to_wire(self) -> dict[str, object]:
        return {'marker_id': self.marker_id, 'metric': self.metric, 'n_before': self.n_before,
                'n_after': self.n_after, 'effect': self.effect, 'lo': self.lo, 'hi': self.hi,
                'state': self.state.value, 'eta_days': self.eta_days,
                'min_detectable': self.min_detectable}


@dataclass(frozen=True)
class MetricSeries:
    metric_id: str
    kind: MetricKind
    rule: Rule
    unit: str
    points: tuple[float | None, ...]
    rolling: Rolling
    ewma: Ewma | None
    alarms: tuple[Alarm, ...]
    baseline: Baseline
    headline: Datum[Headline]

    def to_wire(self) -> dict[str, object]:
        return {
            'kind': self.kind.value, 'rule': self.rule.to_wire(), 'unit': self.unit,
            'points': list(self.points), 'rolling': self.rolling.to_wire(),
            'ewma': None if self.ewma is None else self.ewma.to_wire(),
            'alarms': [alarm.to_wire() for alarm in self.alarms],
            'baseline': self.baseline.to_wire(), 'headline': self.headline.to_wire(),
        }


def _git_check_wire(check: GitCrossCheck | None) -> dict[str, int] | None:
    if check is None:
        return None
    return {'off_first_parent': check.off_first_parent,
            'merges_without_landing': check.merges_without_landing,
            'late_claims': check.late_claims, 'disagreement': check.disagreement}


@dataclass(frozen=True)
class QualitySeries:
    axis: tuple[AxisPoint, ...]
    metrics: tuple[MetricSeries, ...]
    cost_model: CostModel
    modules: tuple[ModuleRow, ...]
    before_after: tuple[BeforeAfter, ...]
    git_check: GitCrossCheck | None

    def to_wire(self) -> dict[str, object]:
        return {
            'axis': [point.to_wire() for point in self.axis],
            'metrics': {metric.metric_id: metric.to_wire() for metric in self.metrics},
            'cost_model': self.cost_model.to_wire(),
            'modules': [row.to_wire() for row in self.modules],
            'before_after': [comparison.to_wire() for comparison in self.before_after],
            'git_check': _git_check_wire(self.git_check),
        }


Values = Callable[[Sequence[LandingRow]], list[float | None]]


@dataclass(frozen=True)
class Since:
    """When a metric's measurement starts over the stored rows, and why it has not when it has not."""

    start: Callable[[Sequence[LandingRow]], datetime | None]
    unmeasured: str


def _per_landing(value: Callable[[LandingRow], float | None]) -> Values:
    return lambda rows: [value(row) for row in rows]


def _once_per_unit(value: Callable[[LandingRow], float | None]) -> Values:
    """A unit value copied to every member, counted at the unit's head and 0 on the others."""

    def values(rows: Sequence[LandingRow]) -> list[float | None]:
        heads = {row.landing.ordinal for row in unit_heads(rows)}
        xs = [value(row) for row in rows]
        return [x if x is None or row.landing.ordinal in heads else 0
                for row, x in zip(rows, xs, strict=True)]

    return values


def _no_producer(source: MetricSource) -> str:
    return f'{NO_PRODUCER}: {_RECORDED_IN[source]} has recorded nothing for this project'


def _source_began(source: MetricSource) -> Since:
    """The source's first row in runs.db, as the newest row's rule vintage records it."""
    return Since(lambda rows: _EPOCH[source](rows[-1].rule_vintage) if rows else None,
                 _no_producer(source))


def _first_measured(values: Values, carried: str) -> Since:
    """The first landing the value is known for: a source with no runs.db epoch of its own."""

    def start(rows: Sequence[LandingRow]) -> datetime | None:
        known = (row.landing.landed_at for row, x in zip(rows, values(rows), strict=True)
                 if x is not None)
        return next(known, None)

    return Since(start, f'no stored landing of this project carries {carried}')


def _first_typed(rows: Sequence[LandingRow]) -> datetime | None:
    """Every landing, regex backfill included, once the typed charter_class has been written."""
    typed = any(row.cost.charter_class_source is CharterClassSource.CHARTER_CLASS for row in rows)
    return rows[0].landing.landed_at if typed else None


def _never_written(source: MetricSource) -> Since:
    return Since(lambda rows: None, _no_producer(source))


@dataclass(frozen=True)
class MetricSpec:
    """One served metric: what it reads from the landing rows and how its rule is labelled."""

    metric_id: str
    kind: MetricKind
    source: MetricSource
    unit: str
    counts: str
    excludes: str
    population: str
    values: Values
    measured_since: Since


def _flag(value: bool | None) -> int | None:
    return None if value is None else int(value)


def _reviewer_blocking(row: LandingRow) -> int | None:
    return _flag(row.outcomes.review_blocking) if row.outcomes.reviewed else None


def _review_rounds_ge2(row: LandingRow) -> int | None:
    rounds = row.cost.review_rounds
    return None if rounds is None else int(rounds >= 2)


def _turns_capped(row: LandingRow) -> int | None:
    return None if row.cost.turns is None else int(row.cost.turns_capped)


def _metric(
    metric_id: str, kind: MetricKind, source: MetricSource, values: Values, *, unit: str,
    counts: str, excludes: str, population: str = 'landings', since: Since | None = None,
) -> MetricSpec:
    return MetricSpec(metric_id, kind, source, unit, counts, excludes, population, values,
                      since if since is not None else _source_began(source))


def _rate(
    metric_id: str, source: MetricSource, value: Callable[[LandingRow], int | None], *,
    counts: str, excludes: str, population: str = 'landings', since: Since | None = None,
) -> MetricSpec:
    return _metric(metric_id, MetricKind.RATE, source, _per_landing(value), unit='share',
                   counts=counts, excludes=excludes, population=population, since=since)


ORCHESTRATED_ONLY = 'orchestrated landings only'

CATALOGUE: tuple[MetricSpec, ...] = (
    _rate('rejected', MetricSource.REJECTIONS, lambda row: _flag(row.outcomes.rejected),
          counts='landings whose verify unit was blocked by post-merge verify',
          excludes='landings before the first categorised merge_blocked'),
    _rate('reviewer_blocking', MetricSource.REVIEW_COMPLETED, _reviewer_blocking,
          counts='reviewed landings with a blocking review',
          excludes='unreviewed landings and landings before the first review_completed',
          population='reviewed landings'),
    _rate('flakes_confirmed', MetricSource.FLAKE_OCCURRENCE,
          lambda row: _flag(row.outcomes.flake_confirmed),
          counts='landings with a failure that passed in isolation',
          excludes='landings before the first flake_occurrence'),
    _rate('flakes_real_red', MetricSource.FLAKE_OCCURRENCE,
          lambda row: _flag(row.outcomes.flake_real_red),
          counts='landings with a failure that also failed in isolation',
          excludes='landings before the first flake_occurrence'),
    _rate('review_rounds_ge2', MetricSource.REVIEW_COMPLETED, _review_rounds_ge2,
          counts='landings reviewed in two or more rounds',
          excludes='landings with no review round count',
          population='landings with a review round count',
          since=_first_measured(_per_landing(_review_rounds_ge2), 'a review round count')),
    _rate('turns_capped', MetricSource.INVOCATION_TURNS, _turns_capped,
          counts='landings whose implementer-class invocations hit the turns cap',
          excludes='landings with no implementer-class turn count',
          population='landings with an implementer-class turn count'),
    _rate('charter_blocked', MetricSource.CHARTER_CLASS,
          lambda row: int(row.cost.charter_class is not None),
          counts='landings whose task was blocked by its charter, regex backfill included',
          excludes='every landing until the typed charter_class is first written',
          since=Since(_first_typed, _no_producer(MetricSource.CHARTER_CLASS))),
    _metric('rejections_per_landing', MetricKind.COUNT, MetricSource.REJECTIONS,
            _once_per_unit(lambda row: row.outcomes.rejection_events),
            unit='rejections per landing',
            counts='post-merge verify blocks, once per verify unit at its first landing',
            excludes='landings before the first categorised merge_blocked'),
    _metric('reverifies_per_landing', MetricKind.COUNT, MetricSource.MERGE_VERIFY,
            _once_per_unit(lambda row: row.outcomes.reverify_count),
            unit='re-verifies per landing',
            counts='merge_verify runs beyond the first, once per verify unit at its first landing',
            excludes='landings before the first merge_verify'),
    _metric('implementer_invocations', MetricKind.COUNT, MetricSource.INVOCATIONS,
            _per_landing(lambda row: row.cost.implementer_invocations), unit='invocations',
            counts='implementer-class invocations per landing',
            excludes=f'{ORCHESTRATED_ONLY}: landings with no invocation',
            population='orchestrated landings'),
    _metric('turns', MetricKind.COUNT, MetricSource.INVOCATION_TURNS,
            _per_landing(lambda row: row.cost.turns), unit='turns',
            counts='implementer-class turns per landing',
            excludes=f'{ORCHESTRATED_ONLY}: landings with no implementer-class turn count',
            population='orchestrated landings'),
    _metric('execute_iterations', MetricKind.COUNT, MetricSource.TASK_RESULTS,
            _per_landing(lambda row: row.cost.execute_iterations), unit='iterations',
            counts='execute iterations summed over the dispatches that produced the landing',
            excludes='landings with no dispatch row',
            population='landings with a dispatch row',
            since=_first_measured(_per_landing(lambda row: row.cost.execute_iterations),
                                  'an execute iteration count')),
    _metric('reverts', MetricKind.COUNT, MetricSource.LATE_EVENTS,
            lambda rows: [None] * len(rows), unit='reverts',
            counts='later first-parent commits reverting the landing',
            excludes='every landing until late_events has a writer',
            since=_never_written(MetricSource.LATE_EVENTS)),
    _metric(COST_METRIC, MetricKind.COST, MetricSource.INVOCATIONS,
            _per_landing(lambda row: row.cost.all_role_usd), unit='usd',
            counts='all-role invocation cost per landing',
            excludes=f'{ORCHESTRATED_ONLY}: landings with no invocation',
            population='orchestrated landings'),
)


def _gated(spec: MetricSpec, rows: Sequence[LandingRow], since: datetime | None) -> list[float | None]:
    """The metric per landing; None for a landing before its source began, never 0."""
    return [
        None if since is None or row.landing.landed_at < since else x
        for row, x in zip(rows, spec.values(rows), strict=True)
    ]


def _flags(xs: Sequence[float | None]) -> list[int | None]:
    return [None if x is None else int(x) for x in xs]


def _markers(events: Iterable[RegimeEvent], series_id: str) -> list[RegimeEvent]:
    return [event for event in events if event.applies_to(series_id) and not event.artefact]


def _rolling_wilson(flags: Sequence[int | None]) -> Rolling:
    block: deque[int] = deque(maxlen=HEADLINE_LANDINGS)
    k = 0
    center: list[float | None] = []
    lo: list[float | None] = []
    hi: list[float | None] = []
    n: list[int] = []
    for x in flags:
        if x is not None:
            if len(block) == HEADLINE_LANDINGS:
                k -= block[0]
            block.append(x)
            k += x
        n.append(len(block))
        if len(block) < MIN_MEASURED:
            center.append(None)
            lo.append(None)
            hi.append(None)
            continue
        band = wilson_interval(k, len(block))
        center.append(k / len(block))
        lo.append(band[0])
        hi.append(band[1])
    return Rolling(tuple(center), tuple(lo), tuple(hi), tuple(n))


Statistic = Callable[[Sequence[float]], float]


def _rolling_centre(xs: Sequence[float | None], stat: Statistic) -> Rolling:
    block: deque[float] = deque(maxlen=HEADLINE_LANDINGS)
    center: list[float | None] = []
    n: list[int] = []
    for x in xs:
        if x is not None:
            block.append(float(x))
        n.append(len(block))
        center.append(stat(list(block)) if len(block) >= MIN_MEASURED else None)
    holes: tuple[float | None, ...] = (None,) * len(xs)
    return Rolling(tuple(center), holes, holes, tuple(n))


@dataclass(frozen=True)
class _Chart:
    baseline: Baseline
    line: tuple[float | None, ...]
    ucl: tuple[float | None, ...]
    alarms: tuple[Alarm, ...]


def _excluding_artefacts(
    xs: Sequence[float | None], landed_at: Sequence[datetime], events: Sequence[RegimeEvent],
    series_id: str,
) -> list[float | None]:
    excluded = regime_segments(landed_at, events, series_id).excluded
    return [None if index in excluded else x for index, x in enumerate(xs)]


def _control_chart(
    series_id: str, charted: Sequence[int | None], view: _View,
) -> _Chart:
    """EWMA and two one-sided CUSUMs per regime segment, against the baseline in force.

    A segment's own baseline is in force once it has one; until then the previous segment's
    stays in force, so a shift just after a marker still alarms and the marker explains it.
    """
    rows, landed_at = view.rows, view.landed_at
    markers = _markers(view.events, series_id)
    line: list[float | None] = [None] * len(rows)
    ucl: list[float | None] = [None] * len(rows)
    alarms: list[Alarm] = []
    in_force = NO_BASELINE
    for segment in regime_segments(landed_at, view.events, series_id).segments:
        p0, state = baseline_rate(charted[segment.start:segment.end])
        if p0 is not None:
            in_force = Baseline(p0, state, rows[segment.start].landing.ordinal)
        if in_force.p0 is None or not 0 < in_force.p0 < 1:
            continue
        p0 = in_force.p0
        indices = [i for i in range(segment.start, segment.end) if charted[i] is not None]
        xs = [x for x in (charted[i] for i in indices) if x is not None]
        for t, (index, z) in enumerate(zip(indices, ewma_series(xs, EWMA_LAMBDA, p0), strict=True), 1):
            line[index] = z
            ucl[index] = ewma_limits(p0, EWMA_LAMBDA, EWMA_SIGMAS, t).ucl
        alarms.extend(_cusum(xs, p0, [rows[i].landing.ordinal for i in indices], indices,
                             markers, landed_at))
    alarms.sort(key=lambda alarm: (alarm.ordinal, alarm.direction.value))
    return _Chart(in_force, tuple(line), tuple(ucl), tuple(alarms))


def _cusum(
    xs: Sequence[int], p0: float, ordinals: Sequence[int], indices: Sequence[int],
    markers: Sequence[RegimeEvent], landed_at: Sequence[datetime],
) -> list[Alarm]:
    """Both one-sided CUSUMs over *xs*; *ordinals* and *indices* place each x on the axis."""
    design = cusum_design(p0)
    alarms: list[Alarm] = []
    for direction, p1 in ((AlarmDirection.UP, design.p1_up), (AlarmDirection.DOWN, design.p1_down)):
        for alarm_at, onset_at in cusum_alarms(xs, p0, p1, design.h):
            onset = indices[onset_at]
            alarms.append(Alarm(
                ordinals[alarm_at], ordinals[onset_at], direction,
                explain_alarm(onset, landed_at[onset], markers, landed_at),
            ))
    return alarms


def _rate_comparisons(
    metric_id: str, charted: Sequence[int | None], view: _View,
) -> tuple[BeforeAfter, ...]:
    """Each marker's Newcombe difference: the segment before it against the segment it opens."""
    segments = regime_segments(view.landed_at, view.events, metric_id).segments
    comparisons: list[BeforeAfter] = []
    for before, after in pairwise(segments):
        xs_before = charted[before.start:before.end]
        if after.event_id is None or all(x is None for x in xs_before):
            continue
        comparisons.append(BeforeAfter.of(after.event_id, metric_id, before_after_rate(
            xs_before, charted[after.start:after.end], landings_per_day=view.landings_per_day,
        )))
    return tuple(comparisons)


def _unmeasured_headline(
    spec: MetricSpec, rows: Sequence[LandingRow], provenance: Provenance,
) -> Datum[Headline]:
    """The headline of a metric with no measurement start: no landing stored, or never measured."""
    if not rows:
        return provenance.unknown('no landing is stored for this project')
    return provenance.unknown(spec.measured_since.unmeasured)


def _since_note(spec: MetricSpec, since: datetime) -> str:
    return f'{spec.source.value} measured since {since.astimezone(UTC).date().isoformat()}'


def _rate_headline(
    spec: MetricSpec, flags: Sequence[int | None], view: _View, since: datetime,
) -> Datum[Headline]:
    block = block_rate_datum(flags, [row.landing.task_id for row in view.rows],
                             n_block=HEADLINE_LANDINGS, n_min=MIN_MEASURED)
    if block.rate is None or block.lo is None or block.hi is None:
        return view.provenance.unknown(f'{block.reason}; {_since_note(spec, since)}')
    form = HeadlineForm.RATE if block.state is RateState.RATE else HeadlineForm.K_OF_N
    return view.provenance.measured(Headline(form, block.n, block.k, block.rate, block.lo, block.hi))


def _bootstrap_headline(
    spec: MetricSpec, xs: Sequence[float | None], view: _View, since: datetime,
    form: HeadlineForm, stat: Statistic,
) -> Datum[Headline]:
    """*stat* over the last measured landings with a band seeded by their task ids."""
    block = [(float(x), row.landing.task_id) for x, row in zip(xs, view.rows, strict=True)
             if x is not None][-HEADLINE_LANDINGS:]
    if len(block) < MIN_MEASURED:
        return view.provenance.unknown(
            f'n={len(block)} measured landings; a {form.value} needs >= {MIN_MEASURED}; '
            f'{_since_note(spec, since)}',
        )
    values = [x for x, _ in block]
    lo, hi = bootstrap_interval(values, stat, content_seed(task_id for _, task_id in block))
    return view.provenance.measured(Headline(form, len(block), None, stat(values), lo, hi))


@dataclass(frozen=True)
class _Unit:
    """A verify unit whose lines and every member's cost are known: one cost-vs-lines point."""

    head: int
    ordinal: int
    landed_at: datetime
    lines: int
    usd: float
    modules: tuple[tuple[str, float], ...]
    task_ids: tuple[str, ...]


def _costed_units(rows: Sequence[LandingRow], costs: Sequence[float | None]) -> list[_Unit]:
    """The units whose head's lines and every member's cost (*costs*, one per row) are known."""
    members: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        members.setdefault(row.landing.unit_id, []).append(index)
    units: list[_Unit] = []
    for indices in members.values():
        head = rows[indices[0]]
        unit_costs = [costs[i] for i in indices]
        if head.cost.code_lines is None or any(usd is None for usd in unit_costs):
            continue
        units.append(_Unit(
            indices[0], head.landing.ordinal, head.landing.landed_at, head.cost.code_lines,
            sum(usd for usd in unit_costs if usd is not None), module_allocation(head),
            tuple(rows[i].landing.task_id for i in indices),
        ))
    return units


def _fittable(units: Sequence[_Unit]) -> bool:
    return len({unit.lines for unit in units}) >= 2


def _cost_model(baseline: Sequence[_Unit]) -> CostModel:
    """The fit over a segment's first units; absent below COST_MIN_BASELINE_UNITS."""
    if len(baseline) < COST_MIN_BASELINE_UNITS or not _fittable(baseline):
        return NO_COST_MODEL
    lines = [unit.lines for unit in baseline]
    costs = [unit.usd for unit in baseline]
    a, b = cost_fit(lines, costs, COST_MIN_LINES)
    residuals = [r for r in log_residuals(lines, costs, a, b) if r is not None]
    state = (BaselineState.FROZEN if len(baseline) >= COST_BASELINE_UNITS
             else BaselineState.PROVISIONAL)
    return CostModel(a, b, reference_cost(lines, costs, min_lines=COST_MIN_LINES),
                     statistics.median(residuals), state)


@dataclass(frozen=True)
class _CostAnalysis:
    """The cost model in force, the sign-of-residual alarms and the cost comparisons."""

    model: CostModel
    alarms: tuple[Alarm, ...]
    comparisons: tuple[BeforeAfter, ...]


def _cost_analysis(metric_id: str, costed: Sequence[_Unit], view: _View) -> _CostAnalysis:
    """Per unit, never per landing: a train's members share one merge's lines."""
    units = [unit for unit in costed if unit.lines >= COST_MIN_LINES and unit.usd > 0]
    regimes = regime_segments([unit.landed_at for unit in units], view.events, metric_id)
    segments = [
        [unit for i, unit in enumerate(units[segment.start:segment.end], segment.start)
         if i not in regimes.excluded]
        for segment in regimes.segments
    ]
    markers = _markers(view.events, metric_id)
    model = NO_COST_MODEL
    alarms: list[Alarm] = []
    for segment in segments:
        fitted = _cost_model(segment[:COST_BASELINE_UNITS])
        model = fitted if fitted.state is not None else model
        if model.a is None or model.b is None or model.residual_median is None:
            continue
        residuals = log_residuals([u.lines for u in segment], [u.usd for u in segment],
                                  model.a, model.b)
        signs = [int(r is not None and r > model.residual_median) for r in residuals]
        alarms.extend(_cusum(signs, SIGN_P0, [u.ordinal for u in segment],
                             [u.head for u in segment], markers, view.landed_at))
    alarms.sort(key=lambda alarm: (alarm.ordinal, alarm.direction.value))
    comparisons = tuple(
        BeforeAfter.of(after_id, metric_id, before_after_cost(
            [u.lines for u in before], [u.usd for u in before],
            [u.lines for u in after], [u.usd for u in after],
            seed=content_seed(task_id for u in (*before, *after) for task_id in u.task_ids),
            min_lines=COST_MIN_LINES, landings_per_day=view.landings_per_day,
        ))
        for (before, _), (after, after_id) in pairwise(
            zip(segments, (segment.event_id for segment in regimes.segments), strict=True),
        )
        if after_id is not None and _fittable(before)
    )
    return _CostAnalysis(model, tuple(alarms), comparisons)


@dataclass
class _ModuleTally:
    landings: int = 0
    tasks: set[str] = field(default_factory=set)
    lines: float = 0.0
    usd: float = 0.0

    def add(self, unit: _Unit, share: float) -> None:
        self.landings += len(unit.task_ids)
        self.tasks.update(unit.task_ids)
        self.lines += share * unit.lines
        self.usd += share * unit.usd

    def row(self, module: str | None) -> ModuleRow:
        ranked = (module is not None and self.lines >= MODULE_MIN_LINES
                  and len(self.tasks) >= MODULE_MIN_TASKS)
        return ModuleRow(module, self.landings, len(self.tasks), self.lines, self.usd,
                         100 * self.usd / self.lines if ranked else None, ranked)


def _module_rows(rows: Sequence[LandingRow], costed: Sequence[_Unit]) -> tuple[ModuleRow, ...]:
    """The *costed* units of the RECENT window before the newest landing, allocated by src-line
    share.

    Ranked modules first, costliest per line first; then the unranked; the no-src bucket last.
    """
    if not rows:
        return ()
    since = rows[-1].landing.landed_at - RECENT
    tallies: dict[str | None, _ModuleTally] = {}
    for unit in costed:
        if unit.landed_at < since:
            continue
        for module, share in unit.modules or ((None, 1.0),):
            tallies.setdefault(module, _ModuleTally()).add(unit, share)
    served = [tally.row(module) for module, tally in tallies.items()]
    return tuple(sorted(served, key=lambda row: (
        not row.ranked, row.module is None, -(row.usd_per_100_lines or 0.0), row.module or '',
    )))


def _landings_per_day(rows: Sequence[LandingRow]) -> float | None:
    """The landing velocity over the RECENT window before the newest landing."""
    if len(rows) < 2:
        return None
    newest = rows[-1].landing.landed_at
    recent = [row for row in rows if row.landing.landed_at >= newest - RECENT]
    span_days = (newest - recent[0].landing.landed_at).total_seconds() / 86400
    return (len(recent) - 1) / span_days if span_days > 0 else None


@dataclass(frozen=True)
class _View:
    """The rows, the served slice's first index, and what every metric is computed against."""

    rows: Sequence[LandingRow]
    landed_at: tuple[datetime, ...]
    start: int
    events: Sequence[RegimeEvent]
    provenance: Provenance
    landings_per_day: float | None

    def served(self, values: Sequence[float | None]) -> tuple[float | None, ...]:
        return tuple(values[self.start:])

    def served_alarms(self, alarms: Iterable[Alarm]) -> tuple[Alarm, ...]:
        if self.start >= len(self.rows):
            return ()
        first = self.rows[self.start].landing.ordinal
        return tuple(alarm for alarm in alarms if alarm.ordinal >= first)

    def rule(self, spec: MetricSpec, since: datetime | None) -> Rule:
        return Rule(spec.counts, spec.excludes, spec.population, HEADLINE_LANDINGS, since,
                    spec.source)


@dataclass(frozen=True)
class _CostTables:
    """The cost metric's series-wide figures: its model in force and the recent module table."""

    model: CostModel
    modules: tuple[ModuleRow, ...]


@dataclass(frozen=True)
class _Served:
    """One metric as served, its marker comparisons, and the cost metric's tables (else None)."""

    series: MetricSeries
    comparisons: tuple[BeforeAfter, ...]
    cost_tables: _CostTables | None = None


def _rate_series(spec: MetricSpec, view: _View) -> _Served:
    since = spec.measured_since.start(view.rows)
    flags = _flags(_gated(spec, view.rows, since))
    charted = _flags(_excluding_artefacts(flags, view.landed_at, view.events, spec.metric_id))
    chart = _control_chart(spec.metric_id, charted, view)
    line = view.served(chart.line)
    series = MetricSeries(
        metric_id=spec.metric_id,
        kind=spec.kind,
        rule=view.rule(spec, since),
        unit=spec.unit,
        points=view.served(flags),
        rolling=_rolling_wilson(flags).sliced(view.start),
        ewma=Ewma(line, view.served(chart.ucl)) if any(z is not None for z in line) else None,
        alarms=view.served_alarms(chart.alarms),
        baseline=chart.baseline,
        headline=(
            _unmeasured_headline(spec, view.rows, view.provenance) if since is None
            else _rate_headline(spec, flags, view, since)
        ),
    )
    return _Served(series, _rate_comparisons(spec.metric_id, charted, view))


def _centred_series(
    spec: MetricSpec, view: _View, form: HeadlineForm, stat: Statistic,
    alarms: tuple[Alarm, ...],
) -> MetricSeries:
    """A count or a cost: points, a rolling centre and a bootstrapped headline, no control chart."""
    since = spec.measured_since.start(view.rows)
    xs = _gated(spec, view.rows, since)
    return MetricSeries(
        metric_id=spec.metric_id,
        kind=spec.kind,
        rule=view.rule(spec, since),
        unit=spec.unit,
        points=view.served(xs),
        rolling=_rolling_centre(xs, stat).sliced(view.start),
        ewma=None,
        alarms=view.served_alarms(alarms),
        baseline=NO_BASELINE,
        headline=(
            _unmeasured_headline(spec, view.rows, view.provenance) if since is None
            else _bootstrap_headline(spec, xs, view, since, form, stat)
        ),
    )


def _count_series(spec: MetricSpec, view: _View) -> _Served:
    return _Served(_centred_series(spec, view, HeadlineForm.MEAN, statistics.fmean, ()), ())


def _cost_series(spec: MetricSpec, view: _View) -> _Served:
    costed = _costed_units(view.rows, spec.values(view.rows))
    analysis = _cost_analysis(spec.metric_id, costed, view)
    series = _centred_series(spec, view, HeadlineForm.MEDIAN, statistics.median, analysis.alarms)
    return _Served(series, analysis.comparisons,
                   _CostTables(analysis.model, _module_rows(view.rows, costed)))


_SERIES_BY_KIND: dict[MetricKind, Callable[[MetricSpec, _View], _Served]] = {
    MetricKind.RATE: _rate_series,
    MetricKind.COUNT: _count_series,
    MetricKind.COST: _cost_series,
}


def _require_one_cost_metric() -> None:
    """The wire has one cost_model and one module table, so the catalogue has one cost metric."""
    costs = [spec.metric_id for spec in CATALOGUE if spec.kind is MetricKind.COST]
    if len(costs) != 1:
        raise RuntimeError(f'CATALOGUE must hold exactly one cost metric, not {costs}')


_require_one_cost_metric()


def quality_series_view(
    rows: Sequence[LandingRow],
    *,
    window_landings: int | None,
    provenance: Provenance,
    events: Sequence[RegimeEvent] = (),
    git_check: GitCrossCheck | None = None,
) -> QualitySeries:
    """The served series over *rows* (ordinal order): statistics over all of them, the newest
    *window_landings* served (all when None). Pure and deterministic for equal inputs."""
    start = 0 if window_landings is None else max(0, len(rows) - window_landings)
    view = _View(rows, tuple(row.landing.landed_at for row in rows), start, tuple(events),
                 provenance, _landings_per_day(rows))
    served = [_SERIES_BY_KIND[spec.kind](spec, view) for spec in CATALOGUE]
    [cost] = [entry.cost_tables for entry in served if entry.cost_tables is not None]
    return QualitySeries(
        axis=tuple(AxisPoint.of(row) for row in rows[start:]),
        metrics=tuple(entry.series for entry in served),
        cost_model=cost.model,
        modules=cost.modules,
        before_after=tuple(comparison for entry in served for comparison in entry.comparisons),
        git_check=git_check,
    )


class QualitySeriesInvariant(enum.StrEnum):
    """The rule a :class:`QualitySeriesContractError` reports."""

    AXIS_ORDER = 'axis_order'
    ARRAY_LENGTH = 'array_length'
    ALARM_ON_AXIS = 'alarm_on_axis'


class QualitySeriesContractError(DatumContractError):
    """A :class:`QualitySeries` broke one of its declared rules, named by :attr:`rule`."""

    def __init__(self, rule: QualitySeriesInvariant, message: str) -> None:
        super().__init__(DatumInvariant.VALUE_CONTRACT, f'{rule.value}: {message}')
        self.rule = rule


def _served_arrays(metric: MetricSeries) -> dict[str, Sequence[object]]:
    arrays: dict[str, Sequence[object]] = {
        'points': metric.points, 'rolling.center': metric.rolling.center,
        'rolling.lo': metric.rolling.lo, 'rolling.hi': metric.rolling.hi,
        'rolling.n': metric.rolling.n,
    }
    if metric.ewma is not None:
        arrays.update({'ewma.line': metric.ewma.line, 'ewma.ucl': metric.ewma.ucl})
    return arrays


def validate_quality_series(series: QualitySeries, served_at: datetime) -> None:
    """Raise :class:`QualitySeriesContractError` (or the envelope's own error for a headline)
    if *series* breaks a declared rule."""
    ordinals = [point.ordinal for point in series.axis]
    if any(earlier >= later for earlier, later in pairwise(ordinals)):
        raise QualitySeriesContractError(
            QualitySeriesInvariant.AXIS_ORDER, f'axis ordinals are not strictly increasing: {ordinals}',
        )
    on_axis = set(ordinals)
    for metric in series.metrics:
        for name, array in _served_arrays(metric).items():
            if len(array) != len(ordinals):
                raise QualitySeriesContractError(
                    QualitySeriesInvariant.ARRAY_LENGTH,
                    f'{metric.metric_id}.{name} serves {len(array)} entries for '
                    f'{len(ordinals)} axis points',
                )
        off_axis = [alarm.ordinal for alarm in metric.alarms if alarm.ordinal not in on_axis]
        if off_axis:
            raise QualitySeriesContractError(
                QualitySeriesInvariant.ALARM_ON_AXIS,
                f'{metric.metric_id} alarms at ordinals {off_axis} not on the served axis',
            )
        validate_datum(metric.headline, served_at)


def _never_measured(project_id: str, latest: QualitySample | None) -> str:
    if latest is None:
        return f'nothing has been sampled for project {project_id} yet'
    return latest.reason or f'no value sample has been written for project {project_id}'


def _cursor_landed_at(rows: Sequence[LandingRow], cursor_sha: str | None) -> datetime | None:
    return next((row.landing.landed_at for row in reversed(rows)
                 if row.landing.merge_sha == cursor_sha and not row.landing.late_claim), None)


def _unsampled(root: Path, rows: Sequence[LandingRow], cursor_sha: str | None) -> str | None:
    """Why runs.db says the stored series is behind: landings it records after the cursor that the
    store lacks, or that runs.db cannot be read, or holds a landing event the landing rule refuses,
    to confirm none; None when it is not behind."""
    path = runs_db(root)
    try:
        conn = connect_ro(path)
    except TaskDbUnreadable as exc:
        return f'freshness cannot be confirmed: {describe_unreadable(exc)}'
    except sqlite3.Error as exc:
        return f'freshness cannot be confirmed: {path}: {type(exc).__name__}: {exc}'
    try:
        recorded = recorded_landing_keys(
            conn, recorded_after=_cursor_landed_at(rows, cursor_sha) if cursor_sha else None,
        )
    except (sqlite3.Error, UnrecognisedTrainEvent, ValueError) as exc:
        return f'freshness cannot be confirmed: {path}: {type(exc).__name__}: {exc}'
    finally:
        conn.close()
    pending = len(recorded - {(row.landing.merge_sha, row.landing.task_id) for row in rows})
    if pending == 0:
        return None
    return f'{pending} landing{"" if pending == 1 else "s"} not yet sampled'


def _read_at(
    value: QualitySample, latest: QualitySample, unsampled: str | None, now: datetime,
) -> Provenance:
    """A gap after the newest value names why the series stopped; else runs.db's verdict; then age."""
    reason = latest.reason if latest.state is SampleState.GAP else unsampled
    state = DatumState.FRESH if reason is None else DatumState.STALE
    return Provenance(value.ts, state, reason, QUALITY_FRESHNESS_BOUND_SECONDS, served_at=now)


def _touching(rows: Sequence[LandingRow], module: str | None) -> list[LandingRow]:
    """The *rows* whose landed lines touch *module* (every row when None); a train's members share
    their unit's modules, so a unit is kept or dropped whole."""
    if module is None:
        return list(rows)
    return [row for row in rows if any(lines.module == module for lines in row.modules)]


async def read_quality_series(
    db: aiosqlite.Connection | None,
    root: Path,
    window_landings: int | None,
    *,
    now: datetime,
    events: Sequence[RegimeEvent] = (),
    module: str | None = None,
) -> Datum[QualitySeries]:
    """*root*'s stored series as read at *now*: unknown until a value sample exists, stale while
    a newer gap, an unsampled recorded landing or the bound says so, otherwise fresh.

    *module* serves only the landings touching it; freshness is still read over every stored
    landing of the project."""
    if db is None:
        return unknown_datum('quality.db does not exist yet: the quality sampler has not written it',
                             QUALITY_FRESHNESS_BOUND_SECONDS)
    project_id = project_id_for_root(root)
    newest = await newest_samples(db, project_id)
    if newest.latest is None or newest.latest_value is None:
        return unknown_datum(_never_measured(project_id, newest.latest),
                             QUALITY_FRESHNESS_BOUND_SECONDS)
    value = newest.latest_value
    rows = await read_landing_rows(db, project_id)
    unsampled = await asyncio.to_thread(_unsampled, root, rows, value.cursor_sha)
    provenance = _read_at(value, newest.latest, unsampled, now)
    served = _touching(rows, module)
    series = await asyncio.to_thread(
        lambda: quality_series_view(served, window_landings=window_landings, provenance=provenance,
                                    events=events, git_check=value.git_check),
    )
    datum = provenance.measured(series)
    validate_datum(datum, now)
    validate_quality_series(series, now)
    return datum
