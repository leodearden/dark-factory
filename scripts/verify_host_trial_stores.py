"""Read each project's runs.db and the load samples store, read-only, into typed rows.

The verify-host trial report's store boundary. The event types, the cap_hit
account event and the lock-conflict requeue reasons are spelled once here; no
enum elsewhere holds them. Each source declares its table and the columns it
needs in SOURCES. Those columns are asked of the store, never assumed: a
missing table or column makes that one source unavailable. A source whose
producer never wrote a row is never_written, which is distinct from a source
read with no row in reach. The CPU pressure metrics come from the gate-arm
table sampler.metrics.ARM_METRICS, so no metric name is spelled here.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar

import _checkout as _checkout
from sampler.metrics import ARM_METRICS, MetricSpec
from shared.project_layout import runs_db

__all__ = [
    'CPU_ARM_SPECS',
    'LOCK_CONFLICT_REASONS',
    'SOURCES',
    'AdmissionPlan',
    'CapHit',
    'CpuRead',
    'CpuSample',
    'Deferral',
    'EventType',
    'Invocation',
    'Landing',
    'LegHold',
    'LifecycleRow',
    'MergeVerify',
    'ProjectRead',
    'ProjectStatus',
    'SchedulerTick',
    'Source',
    'SourceRead',
    'SourceSpec',
    'SourceState',
    'TaskCompletion',
    'TaskStart',
    'columns',
    'connect_read_only',
    'read_cpu_pressure',
    'read_project',
]


class EventType(StrEnum):
    MERGE_VERIFY = 'merge_verify'
    MERGE_FINALIZED = 'merge_finalized'
    TASK_STARTED = 'task_started'
    TASK_COMPLETED = 'task_completed'
    SCHEDULER_TICK = 'scheduler_tick'
    DISPATCH_DEFERRED = 'dispatch_deferred'
    VERIFY_ADMISSION_PLAN = 'verify_admission_plan'
    VERIFY_HOST_ADMITTED = 'verify_host_admitted'
    VERIFY_HOST_RETIRED = 'verify_host_retired'


CAP_HIT = 'cap_hit'
"""The account_events event_type of an account reaching its weekly cap."""

LOCK_CONFLICT_REASONS = frozenset({'plan_blast_radius_lock_conflict', 'scope_grant_lock_conflict'})
"""task_completed reasons of a requeue on a lock conflict (workflow.py::TaskWorkflow._requeue_on_lock_conflict)."""

_DONE = 'done'
_TASK_ROLE = 'task'


# ---------------------------------------------------------------------------
# Sources: what each reads, from which columns, over what reach
# ---------------------------------------------------------------------------


class Source(StrEnum):
    MERGE_VERIFY = 'merge_verify'
    LIFECYCLE = 'host_lifecycle'
    MERGE_FINALIZED = 'merge_finalized'
    TASK_RESULTS = 'task_results'
    TASK_STARTED = 'task_started'
    TASK_COMPLETED = 'task_completed'
    SCHEDULER_TICK = 'scheduler_tick'
    DISPATCH_DEFERRED = 'dispatch_deferred'
    ADMISSION_PLAN = 'verify_admission_plan'
    INVOCATIONS = 'invocations'
    CAP_HIT = 'cap_hit'


@dataclass(frozen=True)
class Reach:
    """How far before since and after until a source's rows are read; None reads the whole store."""

    before: timedelta | None
    after: timedelta | None


_WHOLE_STORE = Reach(None, None)
_PAIRING = timedelta(days=1)
"""Lookaround for rows paired across a window edge: a task run, a tick span, an admission plan."""
_CYCLE = timedelta(days=7)
"""One account allowance cycle, the spend the allowance estimate needs before since."""


@dataclass(frozen=True)
class SourceSpec:
    """A source's table, the columns it selects, and the rows it reads (*kind_column* IN *kinds*)."""

    table: str
    columns: tuple[str, ...]
    time_column: str
    reach: Reach
    kind_column: str | None = None
    kinds: tuple[str, ...] = ()

    @property
    def required(self) -> frozenset[str]:
        return frozenset({*self.columns, self.time_column, *filter(None, [self.kind_column])})

    @property
    def producer(self) -> str:
        """The rows the source reads, as the report names them: event types, or the table and its kind."""
        if self.kind_column == 'event_type':
            return ' or '.join(self.kinds)
        return ' '.join([self.table, *self.kinds])


_EVENT_COLUMNS = ('timestamp', 'run_id', 'task_id', 'event_type', 'data')


def _events(reach: Reach, *kinds: EventType) -> SourceSpec:
    return SourceSpec('events', _EVENT_COLUMNS, 'timestamp', reach, 'event_type', tuple(kinds))


SOURCES: Mapping[Source, SourceSpec] = MappingProxyType({
    Source.MERGE_VERIFY: _events(_WHOLE_STORE, EventType.MERGE_VERIFY),
    Source.LIFECYCLE: _events(_WHOLE_STORE, EventType.VERIFY_HOST_ADMITTED, EventType.VERIFY_HOST_RETIRED),
    Source.MERGE_FINALIZED: _events(_WHOLE_STORE, EventType.MERGE_FINALIZED),
    Source.TASK_RESULTS: SourceSpec(
        'task_results', ('task_id', 'outcome', 'completed_at'), 'completed_at', _WHOLE_STORE, 'outcome', (_DONE,)
    ),
    Source.TASK_STARTED: _events(Reach(_PAIRING, timedelta(0)), EventType.TASK_STARTED),
    Source.TASK_COMPLETED: _events(Reach(_PAIRING, _PAIRING), EventType.TASK_COMPLETED),
    Source.SCHEDULER_TICK: _events(Reach(_PAIRING, _PAIRING), EventType.SCHEDULER_TICK),
    Source.DISPATCH_DEFERRED: _events(Reach(_PAIRING, timedelta(0)), EventType.DISPATCH_DEFERRED),
    Source.ADMISSION_PLAN: _events(Reach(timedelta(0), _PAIRING), EventType.VERIFY_ADMISSION_PLAN),
    Source.INVOCATIONS: SourceSpec(
        'invocations', ('account_name', 'cost_usd', 'completed_at'), 'completed_at', Reach(_CYCLE, timedelta(0))
    ),
    Source.CAP_HIT: SourceSpec(
        'account_events', ('account_name', 'details', 'created_at'), 'created_at', _WHOLE_STORE, 'event_type', (CAP_HIT,)
    ),
})


# ---------------------------------------------------------------------------
# Typed rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeVerify:
    start: datetime
    end: datetime
    runner: str
    task_id: str | None


@dataclass(frozen=True)
class LifecycleRow:
    at: datetime
    event_type: EventType


@dataclass(frozen=True)
class Landing:
    task_id: str
    at: datetime


@dataclass(frozen=True)
class TaskStart:
    at: datetime
    run_id: str
    task_id: str


@dataclass(frozen=True)
class TaskCompletion:
    at: datetime
    run_id: str
    task_id: str
    outcome: str
    reason: str


@dataclass(frozen=True)
class SchedulerTick:
    at: datetime
    run_id: str


@dataclass(frozen=True)
class Deferral:
    at: datetime
    run_id: str
    metric: str
    """The shared.psi gate arm that held the dispatch."""


@dataclass(frozen=True)
class LegHold:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class AdmissionPlan:
    task_id: str
    arrival: datetime
    completed: datetime
    legs: tuple[LegHold, ...]


@dataclass(frozen=True)
class Invocation:
    completed_at: datetime
    account: str
    cost_usd: float


@dataclass(frozen=True)
class CapHit:
    account: str
    at: datetime
    resets_at: datetime | None


class SourceState(StrEnum):
    READ = 'read'
    NEVER_WRITTEN = 'never_written'
    UNAVAILABLE = 'unavailable'


@dataclass(frozen=True)
class SourceRead:
    state: SourceState
    reason: str | None
    """Why an unavailable source could not be read."""
    skipped: int
    """Rows in reach that did not parse."""


class ProjectStatus(StrEnum):
    READ = 'read'
    MISSING = 'missing'
    UNREADABLE = 'unreadable'


def _no_sources() -> Mapping[Source, SourceRead]:
    return MappingProxyType({})


@dataclass(frozen=True)
class ProjectRead:
    """One project's runs.db, read for one host and window."""

    root: Path
    runs_db: Path
    status: ProjectStatus
    detail: str | None
    sources: Mapping[Source, SourceRead] = field(default_factory=_no_sources)
    merge_verifies: tuple[MergeVerify, ...] = ()
    """Every runner's verifies whose start or end falls in the window."""
    first_host_verify_at: datetime | None = None
    """The host's earliest merge_verify start over the whole store."""
    lifecycle: tuple[LifecycleRow, ...] = ()
    """The host's lifecycle rows over the whole store."""
    landings: tuple[Landing, ...] = ()
    """Each task's first done row over the whole store."""
    task_starts: tuple[TaskStart, ...] = ()
    task_completions: tuple[TaskCompletion, ...] = ()
    ticks: tuple[SchedulerTick, ...] = ()
    deferrals: tuple[Deferral, ...] = ()
    admission_plans: tuple[AdmissionPlan, ...] = ()
    """Task-role plans only."""
    invocations: tuple[Invocation, ...] = ()
    cap_hits: tuple[CapHit, ...] = ()

    @property
    def lifecycle_ever_written(self) -> bool:
        """Whether this store holds a lifecycle row for any host."""
        source = self.sources.get(Source.LIFECYCLE)
        return source is not None and source.state is SourceState.READ

    @property
    def skipped(self) -> Mapping[Source, int]:
        return MappingProxyType({source: read.skipped for source, read in self.sources.items()})


# ---------------------------------------------------------------------------
# Reading one source
# ---------------------------------------------------------------------------


def connect_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f'{path.absolute().as_uri()}?mode=ro', uri=True)


def columns(conn: sqlite3.Connection, table: str) -> frozenset[str]:
    """*table*'s column names, empty when there is no such table."""
    return frozenset(row[1] for row in conn.execute(f'PRAGMA table_info({table})'))


@dataclass(frozen=True)
class _Window:
    host: str
    since: datetime
    until: datetime

    def bounds(self, reach: Reach) -> tuple[datetime | None, datetime | None]:
        return (
            None if reach.before is None else self.since - reach.before,
            None if reach.after is None else self.until + reach.after,
        )


def _unavailable_reason(conn: sqlite3.Connection, spec: SourceSpec) -> str | None:
    present = columns(conn, spec.table)
    if not present:
        return f'no {spec.table} table'
    missing = sorted(spec.required - present)
    return f'{spec.table} lacks column {", ".join(missing)}' if missing else None


def _kind_filter(spec: SourceSpec) -> tuple[str, list[Any]]:
    if spec.kind_column is None:
        return '', []
    return f' AND {spec.kind_column} IN ({", ".join("?" for _ in spec.kinds)})', list(spec.kinds)


def _ever_written(conn: sqlite3.Connection, spec: SourceSpec) -> bool:
    kind_sql, params = _kind_filter(spec)
    return conn.execute(f'SELECT 1 FROM {spec.table} WHERE 1{kind_sql} LIMIT 1', params).fetchone() is not None


def _select(conn: sqlite3.Connection, spec: SourceSpec, window: _Window) -> Iterator[Mapping[str, Any]]:
    """The rows of *spec* within its reach, by a text bound padded one day either way."""
    kind_sql, params = _kind_filter(spec)
    lower, upper = window.bounds(spec.reach)
    if lower is not None:
        kind_sql += f' AND {spec.time_column} >= ?'
        params.append((lower - timedelta(days=1)).isoformat())
    if upper is not None:
        kind_sql += f' AND {spec.time_column} < ?'
        params.append((upper + timedelta(days=1)).isoformat())
    cursor = conn.execute(f'SELECT {", ".join(spec.columns)} FROM {spec.table} WHERE 1{kind_sql} ORDER BY rowid', params)
    names = spec.columns
    for row in cursor:
        yield dict(zip(names, row, strict=True))


def _at(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f'timestamp {value!r}')
    parsed = datetime.fromisoformat(value)
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).astimezone(UTC)


def _in_reach(at: datetime, bounds: tuple[datetime | None, datetime | None]) -> bool:
    lower, upper = bounds
    return (lower is None or at >= lower) and (upper is None or at < upper)


_Row = TypeVar('_Row')
_Parse = Callable[[Mapping[str, Any], datetime, _Window], _Row | None]
_UNPARSEABLE = (ValueError, TypeError, KeyError)


def _read(
    conn: sqlite3.Connection, source: Source, window: _Window, parse: _Parse[_Row]
) -> tuple[SourceRead, tuple[_Row, ...]]:
    """*source*'s rows in reach parsed by *parse*, which returns None for a row it does not want."""
    spec = SOURCES[source]
    reason = _unavailable_reason(conn, spec)
    if reason is not None:
        return SourceRead(SourceState.UNAVAILABLE, reason, 0), ()
    if not _ever_written(conn, spec):
        return SourceRead(SourceState.NEVER_WRITTEN, None, 0), ()
    bounds = window.bounds(spec.reach)
    rows: list[_Row] = []
    skipped = 0
    for record in _select(conn, spec, window):
        try:
            at = _at(record[spec.time_column])
            row = parse(record, at, window) if _in_reach(at, bounds) else None
        except _UNPARSEABLE:
            skipped += 1
            continue
        if row is not None:
            rows.append(row)
    return SourceRead(SourceState.READ, None, skipped), tuple(rows)


# ---------------------------------------------------------------------------
# One parser per source; each raises one of _UNPARSEABLE for a malformed row
# ---------------------------------------------------------------------------


def _object(text: object) -> Mapping[str, Any]:
    payload = json.loads(text) if isinstance(text, str) else None
    if not isinstance(payload, dict):
        raise ValueError('payload is not a JSON object')
    return payload


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f'expected text, got {value!r}')
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f'expected a number, got {value!r}')
    return float(value)


def _merge_verify(record: Mapping[str, Any], at: datetime, _window: _Window) -> MergeVerify:
    data = _object(record['data'])
    duration_ms = _number(data['duration_ms'])
    if duration_ms < 0:
        raise ValueError(f'negative duration_ms {duration_ms}')
    return MergeVerify(at - timedelta(milliseconds=duration_ms), at, _text(data['runner']), record['task_id'])


def _lifecycle(record: Mapping[str, Any], at: datetime, window: _Window) -> LifecycleRow | None:
    host = _text(_object(record['data'])['host'])
    return LifecycleRow(at, EventType(record['event_type'])) if host == window.host else None


def _finalized_landing(record: Mapping[str, Any], at: datetime, _window: _Window) -> Landing | None:
    done = _object(record['data']).get('state') == _DONE
    return Landing(_text(record['task_id']), at) if done else None


def _result_landing(record: Mapping[str, Any], at: datetime, _window: _Window) -> Landing:
    return Landing(_text(record['task_id']), at)


def _task_start(record: Mapping[str, Any], at: datetime, _window: _Window) -> TaskStart:
    return TaskStart(at, _text(record['run_id']), _text(record['task_id']))


def _task_completion(record: Mapping[str, Any], at: datetime, _window: _Window) -> TaskCompletion:
    data = _object(record['data'])
    reason = data.get('reason')
    return TaskCompletion(
        at, _text(record['run_id']), _text(record['task_id']), _text(data['outcome']), '' if reason is None else _text(reason)
    )


def _tick(record: Mapping[str, Any], at: datetime, _window: _Window) -> SchedulerTick:
    return SchedulerTick(at, _text(record['run_id']))


def _deferral(record: Mapping[str, Any], at: datetime, _window: _Window) -> Deferral:
    return Deferral(at, _text(record['run_id']), _text(_object(record['data'])['metric']))


def _leg_hold(leg: object) -> LegHold | None:
    """The slot a leg held; None for a leg never granted one."""
    if not isinstance(leg, dict):
        raise ValueError(f'leg {leg!r}')
    if leg.get('granted_at') is None:
        return None
    granted = _at(leg['granted_at'])
    return LegHold(granted, granted + timedelta(seconds=_number(leg['held_secs'])))


def _admission_plan(record: Mapping[str, Any], _at_column: datetime, _window: _Window) -> AdmissionPlan | None:
    data = _object(record['data'])
    if data.get('role') != _TASK_ROLE:
        return None
    legs = data['legs']
    if not isinstance(legs, list):
        raise ValueError(f'legs {legs!r}')
    holds = tuple(hold for hold in map(_leg_hold, legs) if hold is not None)
    return AdmissionPlan(_text(data['task_id']), _at(data['plan_arrival_at']), _at(data['completed_at']), holds)


def _invocation(record: Mapping[str, Any], at: datetime, _window: _Window) -> Invocation:
    return Invocation(at, _text(record['account_name']), _number(record['cost_usd']))


def _cap_hit(record: Mapping[str, Any], at: datetime, _window: _Window) -> CapHit:
    resets_at = _object(record['details']).get('resets_at')
    return CapHit(_text(record['account_name']), at, None if resets_at is None else _at(resets_at))


# ---------------------------------------------------------------------------
# One project
# ---------------------------------------------------------------------------


def _first_landings(landings: Iterable[Landing]) -> tuple[Landing, ...]:
    first: dict[str, Landing] = {}
    for landing in landings:
        if landing.task_id not in first or landing.at < first[landing.task_id].at:
            first[landing.task_id] = landing
    return tuple(sorted(first.values(), key=lambda landing: (landing.at, landing.task_id)))


def _touches(verify: MergeVerify, window: _Window) -> bool:
    return verify.start < window.until and verify.end >= window.since


def _read_store(root: Path, path: Path, conn: sqlite3.Connection, window: _Window) -> ProjectRead:
    sources: dict[Source, SourceRead] = {}

    def read(source: Source, parse: _Parse[_Row]) -> tuple[_Row, ...]:
        sources[source], rows = _read(conn, source, window, parse)
        return rows

    verifies = read(Source.MERGE_VERIFY, _merge_verify)
    host_starts = [verify.start for verify in verifies if verify.runner == window.host]
    landings = (*read(Source.MERGE_FINALIZED, _finalized_landing), *read(Source.TASK_RESULTS, _result_landing))
    return ProjectRead(
        root=root,
        runs_db=path,
        status=ProjectStatus.READ,
        detail=None,
        merge_verifies=tuple(verify for verify in verifies if _touches(verify, window)),
        first_host_verify_at=min(host_starts, default=None),
        lifecycle=tuple(sorted(read(Source.LIFECYCLE, _lifecycle), key=lambda row: row.at)),
        landings=_first_landings(landings),
        task_starts=read(Source.TASK_STARTED, _task_start),
        task_completions=read(Source.TASK_COMPLETED, _task_completion),
        ticks=read(Source.SCHEDULER_TICK, _tick),
        deferrals=read(Source.DISPATCH_DEFERRED, _deferral),
        admission_plans=read(Source.ADMISSION_PLAN, _admission_plan),
        invocations=read(Source.INVOCATIONS, _invocation),
        cap_hits=read(Source.CAP_HIT, _cap_hit),
        sources=MappingProxyType(sources),
    )


def read_project(root: Path, host: str, since: datetime, until: datetime) -> ProjectRead:
    """*root*'s runs.db read for *host* over [since, until); a missing or unreadable store says so."""
    path = runs_db(root)
    if not path.is_file():
        return ProjectRead(root, path, ProjectStatus.MISSING, None)
    try:
        conn = connect_read_only(path)
        try:
            return _read_store(root, path, conn, _Window(host, since, until))
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return ProjectRead(root, path, ProjectStatus.UNREADABLE, f'{type(exc).__name__}: {exc}')


# ---------------------------------------------------------------------------
# CPU pressure, from the load samples store
# ---------------------------------------------------------------------------


def _arm_specs(arms: Iterable[str]) -> tuple[MetricSpec, ...]:
    arms = tuple(arms)
    absent = [arm for arm in arms if arm not in ARM_METRICS]
    if absent:
        raise ImportError(f'CPU arms {absent} are not gate arms in sampler.metrics.ARM_METRICS')
    per_leaf = [arm for arm in arms if ARM_METRICS[arm].per_leaf]
    if per_leaf:
        raise ImportError(f'CPU arms {per_leaf} are per-leaf; workstation CPU pressure is host-level')
    return tuple(ARM_METRICS[arm] for arm in arms)


_CPU_ARMS = ('cpu_some_avg10', 'runqueue_ratio')
CPU_ARM_SPECS: tuple[MetricSpec, ...] = _arm_specs(_CPU_ARMS)
"""The host-level metrics recording the CPU gate arms."""


@dataclass(frozen=True)
class CpuSample:
    at: datetime
    value: float


@dataclass(frozen=True)
class CpuRead:
    samples_db: Path
    reason: str | None
    """Why the samples store could not be read."""
    samples: Mapping[MetricSpec, tuple[CpuSample, ...]]
    unreadable_dropped: int
    """Samples dropped because their readability metric read 0 at the same ts."""


_CPU_SQL = (
    'SELECT s.ts, s.value, r.value FROM samples s'
    ' LEFT JOIN samples r ON r.metric = ? AND r.ts = s.ts AND r.leaf IS NULL'
    ' WHERE s.metric = ? AND s.leaf IS NULL AND s.ts >= ? AND s.ts < ? ORDER BY s.ts'
)


def _cpu_unavailable(path: Path, reason: str) -> CpuRead:
    return CpuRead(path, reason, MappingProxyType({spec: () for spec in CPU_ARM_SPECS}), 0)


def read_cpu_pressure(samples_db: Path, since: datetime, until: datetime) -> CpuRead:
    """Each CPU arm's host-level samples in [since, until), less those whose readability read 0."""
    if not samples_db.is_file():
        return _cpu_unavailable(samples_db, f'no samples store at {samples_db}')
    samples: dict[MetricSpec, tuple[CpuSample, ...]] = {}
    dropped = 0
    try:
        conn = connect_read_only(samples_db)
        try:
            for spec in CPU_ARM_SPECS:
                readability = None if spec.readability is None else spec.readability.name
                rows = conn.execute(_CPU_SQL, (readability, spec.name, since.timestamp(), until.timestamp())).fetchall()
                kept = [CpuSample(datetime.fromtimestamp(ts, UTC), float(value)) for ts, value, ok in rows if ok != 0]
                dropped += len(rows) - len(kept)
                samples[spec] = tuple(kept)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return _cpu_unavailable(samples_db, f'{samples_db}: {type(exc).__name__}: {exc}')
    return CpuRead(samples_db, None, MappingProxyType(samples), dropped)
