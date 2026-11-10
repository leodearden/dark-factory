"""One project's per-landing quality series: the axis, each metric's points, bands, baseline and alarms.

Every number comes from shared/src/shared/quality_series/stats.py (plans/quality-dashboard-prd.md
decision 21); this module only chooses what each statistic is applied to and shapes the wire
(decisions 13, 15 and 22, §Contract "Wire shape"). Statistics run over the project's full stored
history; the window only slices what is served.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from shared.quality_series.landings import GitCrossCheck
from shared.quality_series.rows import LandingRow, SourceEpochs
from shared.quality_series.stats import (
    BaselineState,
    RateState,
    RegimeEvent,
    baseline_rate,
    block_rate_datum,
    cusum_alarms,
    cusum_design,
    ewma_limits,
    ewma_series,
    explain_alarm,
    regime_segments,
    wilson_interval,
)

from dashboard.data.datum import Datum, DatumState, unknown_datum

__all__ = [
    'Alarm',
    'AlarmDirection',
    'AxisPoint',
    'Baseline',
    'Ewma',
    'Headline',
    'HeadlineForm',
    'MetricKind',
    'MetricSeries',
    'MetricSource',
    'Provenance',
    'QualitySeries',
    'Rolling',
    'Rule',
    'quality_series_view',
]

HEADLINE_LANDINGS = 100
"""The headline block and the rolling band: the last this-many measured landings."""
MIN_MEASURED = 30
"""Below this many measured landings a rate is unknown."""
EWMA_LAMBDA = 0.05
EWMA_SIGMAS = 3.0
NO_PRODUCER = 'no producer has written here'


def _iso(at: datetime | None) -> str | None:
    return None if at is None else at.astimezone(UTC).isoformat()


class MetricKind(StrEnum):
    RATE = 'rate'
    COUNT = 'count'
    COST = 'cost'


class MetricSource(StrEnum):
    """The runs.db source a metric is measured from; each names a SourceEpochs field."""

    REJECTIONS = 'rejections'
    MERGE_VERIFY = 'merge_verify'
    REVIEW_COMPLETED = 'review_completed'
    FLAKE_OCCURRENCE = 'flake_occurrence'
    INVOCATION_TURNS = 'invocation_turns'
    INVOCATIONS = 'invocations'


_EPOCH: dict[MetricSource, Callable[[SourceEpochs], datetime | None]] = {
    MetricSource.REJECTIONS: lambda epochs: epochs.rejections,
    MetricSource.MERGE_VERIFY: lambda epochs: epochs.merge_verify,
    MetricSource.REVIEW_COMPLETED: lambda epochs: epochs.review_completed,
    MetricSource.FLAKE_OCCURRENCE: lambda epochs: epochs.flake_occurrence,
    MetricSource.INVOCATION_TURNS: lambda epochs: epochs.invocation_turns,
    MetricSource.INVOCATIONS: lambda epochs: epochs.invocations,
}


class HeadlineForm(StrEnum):
    RATE = 'rate'
    K_OF_N = 'k_of_n'
    MEAN = 'mean'
    MEDIAN = 'median'


class AlarmDirection(StrEnum):
    UP = 'up'
    DOWN = 'down'


@dataclass(frozen=True)
class Provenance:
    """What every measured headline inherits from the sample the series is read at."""

    as_of: datetime
    state: DatumState
    reason: str | None
    freshness_bound_seconds: int

    def measured(self, headline: Headline) -> Datum[Headline]:
        return Datum(headline, self.as_of, self.state, self.reason, self.freshness_bound_seconds)

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
    """Per axis point: the band over the last measured landings up to it, None below MIN_MEASURED."""

    center: tuple[float | None, ...]
    lo: tuple[float | None, ...]
    hi: tuple[float | None, ...]
    n: tuple[int, ...]

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
    git_check: GitCrossCheck | None

    def to_wire(self) -> dict[str, object]:
        return {
            'axis': [point.to_wire() for point in self.axis],
            'metrics': {metric.metric_id: metric.to_wire() for metric in self.metrics},
            'cost_model': None,
            'modules': [],
            'before_after': [],
            'git_check': _git_check_wire(self.git_check),
        }


Since = Callable[[Sequence[LandingRow]], datetime | None]


def _source_began(source: MetricSource) -> Since:
    """The source's first row in runs.db, as the newest row's rule vintage records it."""
    return lambda rows: _EPOCH[source](rows[-1].rule_vintage) if rows else None


def _first_measured(value: Callable[[LandingRow], float | None]) -> Since:
    """The first landing the value is known for: a source backfilled before its runs.db epoch."""
    return lambda rows: next(
        (row.landing.landed_at for row in rows if value(row) is not None), None,
    )


@dataclass(frozen=True)
class MetricSpec:
    """One served metric: what it reads from a landing row and how its rule is labelled."""

    metric_id: str
    kind: MetricKind
    source: MetricSource
    unit: str
    counts: str
    excludes: str
    population: str
    value: Callable[[LandingRow], float | None]
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


def _rate(
    metric_id: str, source: MetricSource, value: Callable[[LandingRow], int | None], *,
    counts: str, excludes: str, population: str = 'landings', since: Since | None = None,
) -> MetricSpec:
    return MetricSpec(metric_id, MetricKind.RATE, source, 'share', counts, excludes, population,
                      value, since if since is not None else _source_began(source))


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
          since=_first_measured(_review_rounds_ge2)),
    _rate('turns_capped', MetricSource.INVOCATION_TURNS, _turns_capped,
          counts='landings whose implementer-class invocations hit the turns cap',
          excludes='landings with no implementer-class turn count',
          population='landings with an implementer-class turn count'),
)


def _measured(spec: MetricSpec, rows: Sequence[LandingRow], since: datetime | None) -> list[float | None]:
    """The metric per landing; None for a landing before its source began, never 0."""
    return [
        None if since is None or row.landing.landed_at < since else spec.value(row)
        for row in rows
    ]


def _flags(xs: Sequence[float | None]) -> list[int | None]:
    return [None if x is None else int(x) for x in xs]


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


@dataclass(frozen=True)
class _Chart:
    baseline: Baseline
    line: tuple[float | None, ...]
    ucl: tuple[float | None, ...]
    alarms: tuple[Alarm, ...]


def _control_chart(
    series_id: str, flags: Sequence[int | None], rows: Sequence[LandingRow],
    events: Sequence[RegimeEvent],
) -> _Chart:
    """EWMA and two one-sided CUSUMs per regime segment, against the baseline in force.

    A segment's own baseline is in force once it has one; until then the previous segment's
    stays in force, so a shift just after a marker still alarms and the marker explains it.
    """
    landed_at = [row.landing.landed_at for row in rows]
    regimes = regime_segments(landed_at, events, series_id)
    charted = [None if index in regimes.excluded else x for index, x in enumerate(flags)]
    markers = [event for event in events if event.applies_to(series_id) and not event.artefact]
    line: list[float | None] = [None] * len(rows)
    ucl: list[float | None] = [None] * len(rows)
    alarms: list[Alarm] = []
    in_force = NO_BASELINE
    for segment in regimes.segments:
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
        design = cusum_design(p0)
        for direction, p1 in ((AlarmDirection.UP, design.p1_up), (AlarmDirection.DOWN, design.p1_down)):
            for alarm_at, onset_at in cusum_alarms(xs, p0, p1, design.h):
                alarm, onset = indices[alarm_at], indices[onset_at]
                alarms.append(Alarm(
                    rows[alarm].landing.ordinal, rows[onset].landing.ordinal, direction,
                    explain_alarm(onset, landed_at[onset], markers, landed_at),
                ))
    alarms.sort(key=lambda alarm: (alarm.ordinal, alarm.direction.value))
    return _Chart(in_force, tuple(line), tuple(ucl), tuple(alarms))


def _unmeasured_headline(
    spec: MetricSpec, rows: Sequence[LandingRow], provenance: Provenance,
) -> Datum[Headline]:
    """The headline of a metric with no measurement start: no landing stored, or no producer."""
    if not rows:
        return provenance.unknown('no landing is stored for this project')
    return provenance.unknown(
        f'{NO_PRODUCER}: {spec.source.value} has recorded nothing for this project',
    )


def _since_note(spec: MetricSpec, since: datetime) -> str:
    return f'{spec.source.value} measured since {since.astimezone(UTC).date().isoformat()}'


def _rate_headline(
    spec: MetricSpec, flags: Sequence[int | None], rows: Sequence[LandingRow],
    since: datetime, provenance: Provenance,
) -> Datum[Headline]:
    block = block_rate_datum(flags, [row.landing.task_id for row in rows],
                             n_block=HEADLINE_LANDINGS, n_min=MIN_MEASURED)
    if block.rate is None or block.lo is None or block.hi is None:
        return provenance.unknown(f'{block.reason}; {_since_note(spec, since)}')
    form = HeadlineForm.RATE if block.state is RateState.RATE else HeadlineForm.K_OF_N
    return provenance.measured(Headline(form, block.n, block.k, block.rate, block.lo, block.hi))


@dataclass(frozen=True)
class _View:
    """The rows, the served slice's first index, and what every metric is computed against."""

    rows: Sequence[LandingRow]
    start: int
    events: Sequence[RegimeEvent]
    provenance: Provenance

    def served(self, values: Sequence[float | None]) -> tuple[float | None, ...]:
        return tuple(values[self.start:])

    def served_alarms(self, alarms: Iterable[Alarm]) -> tuple[Alarm, ...]:
        if self.start >= len(self.rows):
            return ()
        first = self.rows[self.start].landing.ordinal
        return tuple(alarm for alarm in alarms if alarm.ordinal >= first)


def _rate_series(spec: MetricSpec, view: _View) -> MetricSeries:
    since = spec.measured_since(view.rows)
    flags = _flags(_measured(spec, view.rows, since))
    chart = _control_chart(spec.metric_id, flags, view.rows, view.events)
    rolling = _rolling_wilson(flags)
    line = view.served(chart.line)
    ewma = Ewma(line, view.served(chart.ucl)) if any(z is not None for z in line) else None
    return MetricSeries(
        metric_id=spec.metric_id,
        kind=spec.kind,
        rule=Rule(spec.counts, spec.excludes, spec.population, HEADLINE_LANDINGS, since,
                  spec.source),
        unit=spec.unit,
        points=view.served(flags),
        rolling=Rolling(view.served(rolling.center), view.served(rolling.lo),
                        view.served(rolling.hi), rolling.n[view.start:]),
        ewma=ewma,
        alarms=view.served_alarms(chart.alarms),
        baseline=chart.baseline,
        headline=(
            _unmeasured_headline(spec, view.rows, view.provenance) if since is None
            else _rate_headline(spec, flags, view.rows, since, view.provenance)
        ),
    )


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
    view = _View(rows, start, tuple(events), provenance)
    return QualitySeries(
        axis=tuple(AxisPoint.of(row) for row in rows[start:]),
        metrics=tuple(_rate_series(spec, view) for spec in CATALOGUE),
        git_check=git_check,
    )
