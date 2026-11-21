"""The ``holds`` report: which consumers grew and which paid before each memory hold.

PRD ``plans/per-consumer-resource-recording-prd.md`` leaf 2, contract C2. A
memory hold is a ``dispatch_deferred`` event whose gate arm is one of
``MEMORY_ARMS``, read from each named project's runs.db. The consumer figures
come from the load recorder's store, ``data/load-samples.db``. Both stores are
opened read-only.

Every metric name is a spec's ``name`` from ``sampler.metrics``, and every
consumer name is produced and parsed by ``sampler.consumers``; this module
spells neither (sampler/tests/test_vocabulary_one_table.py scans it).
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import _checkout as _checkout
from sampler.consumers import Consumer, Figure, Role, classify_member, encode_consumer
from sampler.metrics import FIGURE_METRICS, MetricSpec
from shared.project_layout import runs_db
from shared.psi import ARM_FIELDS, PsiSample
from shared.verify_scope_tag import scope_tag_for

__all__ = [
    'MEMORY_ARMS',
    'Hold',
    'HoldConsumers',
    'HoldReading',
    'HoldState',
    'HoldsReport',
    'RootRead',
    'RootStatus',
    'SamplesUnavailable',
    'Window',
    'format_utc',
    'hold_consumers',
    'parse_utc',
    'read_holds_report',
    'read_root',
    'render_text',
    'report_json',
]


def _memory_arms(sample_fields: Mapping[str, str]) -> Mapping[str, str]:
    strays = sorted(set(sample_fields) - ARM_FIELDS)
    unknown = sorted(set(sample_fields.values()) - {field.name for field in fields(PsiSample)})
    if strays or unknown:
        raise ImportError(
            f'memory arms out of step with shared.psi: arms {strays} are not in ARM_FIELDS, '
            f'fields {unknown} are not on PsiSample'
        )
    return MappingProxyType(dict(sample_fields))


_ARM_SAMPLE_FIELDS: Mapping[str, str] = _memory_arms(
    {'mem_full_avg10': 'mem_full10', 'mem_some_avg10': 'mem_some10'}
)
"""Each memory arm, and the PsiSample field carrying the avg10 value it held on."""

MEMORY_ARMS: frozenset[str] = frozenset(_ARM_SAMPLE_FIELDS)
"""The gate arms whose holds are memory holds."""


def parse_utc(text: str) -> datetime:
    """An ISO date or datetime as an aware UTC datetime; a date is its midnight, a naive value UTC."""
    parsed = datetime.fromisoformat(text)
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).astimezone(UTC)


def format_utc(moment: datetime) -> str:
    """``YYYY-MM-DDTHH:MM:SSZ``: the one rendering of a time in the report."""
    return moment.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def _utc(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, UTC)


def _read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f'{path.absolute().as_uri()}?mode=ro', uri=True)


# ---------------------------------------------------------------------------
# Memory holds, per project root
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hold:
    project_root: Path
    event_id: int
    at: datetime
    arm: str
    task_id: str | None
    own_cgroup: str | None
    """The daemon's kernel path, '' when it was unresolved; None when the event carries none."""
    arm_value: float | None


class RootStatus(StrEnum):
    READ = 'read'
    MISSING = 'missing'
    UNREADABLE = 'unreadable'


@dataclass(frozen=True)
class RootRead:
    """One root's runs.db: whether it was read, and its memory holds since the report's start.

    ``skipped`` counts the hold events that could not be parsed.
    """

    root: Path
    runs_db: Path
    status: RootStatus
    detail: str | None
    holds: tuple[Hold, ...]
    skipped: int


_HOLD_EVENTS_SQL = (
    "SELECT id, timestamp, task_id, data FROM events WHERE event_type = 'dispatch_deferred'"
    ' ORDER BY id'
)


def _memory_hold(root: Path, row: tuple[Any, ...], since: datetime) -> Hold | None:
    """The memory hold *row* records at or after *since*; None for an earlier one or another arm's.

    Raises ValueError when the event does not parse. An event the gate wrote
    before its payload carried the PSI sample (``{metric, value, in_flight,
    floor}``) is still a hold, with no own cgroup.
    """
    event_id, timestamp, task_id, data = row
    if not isinstance(timestamp, str):
        raise ValueError(f'event {event_id}: timestamp {timestamp!r}')
    at = parse_utc(timestamp)
    if at < since:
        return None
    payload = json.loads(data) if isinstance(data, str) else None
    if not isinstance(payload, dict):
        raise ValueError(f'event {event_id}: data is not a JSON object')
    arm = payload.get('metric')
    if arm not in MEMORY_ARMS:
        return None
    sample = payload.get('sample')
    if sample is not None and not isinstance(sample, dict):
        raise ValueError(f'event {event_id}: sample is not a JSON object')
    own_cgroup = (sample or {}).get('own_cgroup')
    if own_cgroup is not None and not isinstance(own_cgroup, str):
        raise ValueError(f'event {event_id}: own_cgroup {own_cgroup!r}')
    value = payload.get('value') if sample is None else sample.get(_ARM_SAMPLE_FIELDS[arm])
    return Hold(root, event_id, at, arm, task_id, own_cgroup, _number(value))


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def read_root(root: Path, since: datetime) -> RootRead:
    """*root*'s memory holds at or after *since*; a missing or unreadable runs.db says so."""
    path = runs_db(root)
    if not path.is_file():
        return RootRead(root, path, RootStatus.MISSING, None, (), 0)
    try:
        conn = _read_only(path)
        try:
            rows = conn.execute(_HOLD_EVENTS_SQL).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return RootRead(root, path, RootStatus.UNREADABLE, f'{type(exc).__name__}: {exc}', (), 0)
    holds: list[Hold] = []
    skipped = 0
    for row in rows:
        try:
            hold = _memory_hold(root, row, since)
        except ValueError:
            skipped += 1
            continue
        if hold is not None:
            holds.append(hold)
    return RootRead(root, path, RootStatus.READ, None, tuple(holds), skipped)


# ---------------------------------------------------------------------------
# The holding project's consumers (C2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldConsumers:
    """The consumers of the project whose runs.db holds the hold.

    ``unmapped_reason`` says why service, agents and project are all None.
    """

    service: Consumer | None
    agents: Consumer | None
    project: Consumer | None
    verify: Consumer
    unmapped_reason: str | None


_DAEMON_PLACEMENTS = frozenset({Role.SERVICE, Role.PROJECT})
"""What the recorder's own rule makes of a daemon's last segment: its unit, or its project slice."""


def hold_consumers(own_cgroup: str | None, project_root: Path) -> HoldConsumers:
    """Map a hold to its project's consumers by its own cgroup's last segment and its root.

    The segment's shape is told by ``sampler.consumers.classify_member``, D2's
    own rule: a unit keys service and agents, and a project slice keys project
    too. Verify is keyed by the root's scope tag. Nothing is derived from the
    project id, so the unit stem, the slug and the id are never assumed alike.
    """
    verify = Consumer(Role.VERIFY, scope_tag_for(project_root))

    def unmapped(reason: str) -> HoldConsumers:
        return HoldConsumers(None, None, None, verify, reason)

    if own_cgroup is None:
        return unmapped('the hold event carries no own cgroup')
    segment = next((name for name in reversed(own_cgroup.split('/')) if name), None)
    if segment is None:
        return unmapped('own cgroup unresolved')
    placement = classify_member((segment,))
    if placement.role not in _DAEMON_PLACEMENTS:
        return unmapped(f'own cgroup {segment} is neither an orchestrator unit nor a project slice')
    project = Consumer(Role.PROJECT, segment) if placement.role is Role.PROJECT else None
    return HoldConsumers(
        Consumer(Role.SERVICE, segment), Consumer(Role.AGENTS, segment), project, verify, None
    )


# ---------------------------------------------------------------------------
# The samples store
# ---------------------------------------------------------------------------


class SamplesUnavailable(Exception):
    """The samples store cannot be read, so no hold can be classified truthfully."""


def _readability(spec: MetricSpec) -> MetricSpec:
    if spec.readability is None:
        raise ImportError(f'{spec.name} declares no readability metric: no collection clock')
    return spec.readability


_READ_OK = _readability(FIGURE_METRICS[Figure.ANON].spec)
"""Written for every consumer at every collection, the kernel included: the collection clock."""

_SAMPLES_TABLE_SQL = "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'samples'"
_FIRST_SQL = 'SELECT MIN(ts) FROM samples WHERE metric = ?'
_LAST_SQL = 'SELECT MAX(ts) FROM samples WHERE metric = ?'
_LATEST_SQL = 'SELECT MAX(ts) FROM samples WHERE metric = ? AND ts <= ?'
_WINDOW_SQL = 'SELECT ts, metric, leaf, value FROM samples WHERE metric IN ({}) AND ts BETWEEN ? AND ?'

_Row = tuple[int, str, str | None, float]


def _open_samples(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise SamplesUnavailable(f'samples store missing: {path}')
    try:
        conn = _read_only(path)
        if conn.execute(_SAMPLES_TABLE_SQL).fetchone() is None:
            conn.close()
            raise SamplesUnavailable(f'samples store has no samples table: {path}')
    except sqlite3.Error as exc:
        raise SamplesUnavailable(f'samples store unreadable: {path} ({exc})') from exc
    return conn


class _Samples:
    """The consumer collections, read through the (metric, ts, leaf) index on one connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def _ts(self, sql: str, *args: object) -> int | None:
        (ts,) = self._conn.execute(sql, args).fetchone()
        return ts

    def first(self) -> int | None:
        return self._ts(_FIRST_SQL, _READ_OK.name)

    def last(self) -> int | None:
        return self._ts(_LAST_SQL, _READ_OK.name)

    def latest_at_or_before(self, moment: float) -> int | None:
        return self._ts(_LATEST_SQL, _READ_OK.name, moment)

    def rows(self, specs: Sequence[MetricSpec], start: int, end: int) -> list[_Row]:
        sql = _WINDOW_SQL.format(', '.join('?' * len(specs)))
        return self._conn.execute(sql, (*(spec.name for spec in specs), start, end)).fetchall()


# ---------------------------------------------------------------------------
# The window before a hold
# ---------------------------------------------------------------------------


class HoldState(StrEnum):
    RANKED = 'ranked'
    NOT_LIVE = 'recorder_not_live'
    NO_SAMPLES = 'no_samples'


_STATE_TEXT: Mapping[HoldState, str] = MappingProxyType({
    HoldState.RANKED: 'ranked over the window',
    HoldState.NOT_LIVE: 'recorder not live at this hold',
    HoldState.NO_SAMPLES: 'no samples: under two consumer collections in the window',
})


@dataclass(frozen=True)
class Window:
    """The collections a hold is read over: gauges at start and end, increments in (start, end]."""

    start: datetime
    end: datetime
    collections: int


@dataclass(frozen=True)
class HoldReading:
    hold: Hold
    mapping: HoldConsumers
    state: HoldState
    window: Window | None
    last_collection: datetime | None
    """The latest collection at or before the hold."""


def _read_hold(samples: _Samples, hold: Hold, window_seconds: int) -> HoldReading:
    """Read *hold* over the window ending at the latest collection at or before it.

    The window starts at the latest collection at or before ``hold - window``,
    or at the first collection when the recorder went live inside the window.
    """
    mapping = hold_consumers(hold.own_cgroup, hold.project_root)
    moment = hold.at.timestamp()
    end = samples.latest_at_or_before(moment)
    if end is None:
        return HoldReading(hold, mapping, HoldState.NOT_LIVE, None, None)
    opens = moment - window_seconds
    start = samples.latest_at_or_before(opens)
    if start is None:
        start = samples.first()
    if start is None or end < opens or start == end:
        return HoldReading(hold, mapping, HoldState.NO_SAMPLES, None, _utc(end))
    collections = len({row[0] for row in samples.rows((_READ_OK,), start, end)})
    return HoldReading(
        hold, mapping, HoldState.RANKED, Window(_utc(start), _utc(end), collections), _utc(end)
    )


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldsReport:
    roots: tuple[RootRead, ...]
    samples_db: Path
    first_collection: datetime | None
    last_collection: datetime | None
    since: datetime
    window_seconds: int
    readings: tuple[HoldReading, ...]
    """In hold order."""


def _hold_order(hold: Hold) -> tuple[datetime, str, int]:
    return hold.at, str(hold.project_root), hold.event_id


def read_holds_report(
    roots: Iterable[Path], samples_db: Path, *, since: datetime, window_seconds: int
) -> HoldsReport:
    """Every memory hold the roots' runs.db files hold since *since*, read against *samples_db*."""
    conn = _open_samples(samples_db)
    try:
        reads = tuple(read_root(root, since) for root in roots)
        samples = _Samples(conn)
        readings = tuple(
            _read_hold(samples, hold, window_seconds)
            for hold in sorted((hold for read in reads for hold in read.holds), key=_hold_order)
        )
        first, last = samples.first(), samples.last()
    except sqlite3.Error as exc:
        raise SamplesUnavailable(f'samples store unreadable: {samples_db} ({exc})') from exc
    finally:
        conn.close()
    return HoldsReport(
        roots=reads,
        samples_db=samples_db,
        first_collection=None if first is None else _utc(first),
        last_collection=None if last is None else _utc(last),
        since=since,
        window_seconds=window_seconds,
        readings=readings,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _root_line(read: RootRead) -> str:
    if read.status is RootStatus.MISSING:
        found = f'runs.db missing: {read.runs_db}'
    elif read.status is RootStatus.UNREADABLE:
        found = f'runs.db unreadable: {read.runs_db} ({read.detail})'
    else:
        found = f'runs.db {read.runs_db}'
    skipped = f'; {read.skipped} hold events skipped, unparseable' if read.skipped else ''
    return f'  {read.root}: {found}{skipped}'


def _samples_line(report: HoldsReport) -> str:
    first, last = report.first_collection, report.last_collection
    if first is None or last is None:
        span = 'no consumer collections'
    else:
        span = f'consumer collections {format_utc(first)} to {format_utc(last)}'
    return f'samples db: {report.samples_db} ({span})'


def _hold_count_line(report: HoldsReport) -> str:
    count = len(report.readings)
    noun = 'memory hold' if count == 1 else 'memory holds'
    return f'{count} {noun} since {format_utc(report.since)}'


def _header(report: HoldsReport) -> list[str]:
    return [
        f'roots ({len(report.roots)}):',
        *(_root_line(read) for read in report.roots),
        _samples_line(report),
        f'window: {report.window_seconds} s before each hold',
        _hold_count_line(report),
    ]


def _heading(hold: Hold) -> str:
    value = 'n/a' if hold.arm_value is None else f'{hold.arm_value:g}'
    return f'hold {format_utc(hold.at)}  {hold.project_root}  task {hold.task_id}  {hold.arm} {value}'


def _window_lines(window: Window) -> list[str]:
    return [
        f'  window {format_utc(window.start)} to {format_utc(window.end)}, '
        f'{window.collections} collections',
        '  who grew: no consumer grew',
        '  who paid: no consumer paid',
    ]


def _hold_block(reading: HoldReading) -> list[str]:
    if reading.window is not None:
        return [_heading(reading.hold), *_window_lines(reading.window)]
    last = reading.last_collection
    after = '' if last is None else f'; the last collection was {format_utc(last)}'
    return [_heading(reading.hold), f'  {_STATE_TEXT[reading.state]}{after}']


def render_text(report: HoldsReport, *, summary: bool = False) -> str:
    sections = [_header(report), *(_hold_block(reading) for reading in report.readings)]
    return '\n\n'.join('\n'.join(lines) for lines in sections)


def _time_json(moment: datetime | None) -> str | None:
    return None if moment is None else format_utc(moment)


def _consumer_json(consumer: Consumer | None) -> str | None:
    return None if consumer is None else encode_consumer(consumer)


def _root_json(read: RootRead) -> dict[str, Any]:
    return {
        'root': str(read.root),
        'runs_db': str(read.runs_db),
        'status': read.status.value,
        'detail': read.detail,
        'skipped': read.skipped,
    }


def _mapping_json(mapping: HoldConsumers) -> dict[str, Any]:
    return {
        'service': _consumer_json(mapping.service),
        'agents': _consumer_json(mapping.agents),
        'project': _consumer_json(mapping.project),
        'verify': encode_consumer(mapping.verify),
        'unmapped_reason': mapping.unmapped_reason,
    }


def _window_json(window: Window) -> dict[str, Any]:
    return {
        'window': {
            'start': format_utc(window.start),
            'end': format_utc(window.end),
            'collections': window.collections,
        },
        'who_grew': [],
        'who_paid': [],
    }


def _reading_json(reading: HoldReading) -> dict[str, Any]:
    hold = reading.hold
    entry: dict[str, Any] = {
        'at': format_utc(hold.at),
        'project_root': str(hold.project_root),
        'event_id': hold.event_id,
        'task_id': hold.task_id,
        'arm': hold.arm,
        'arm_value': hold.arm_value,
        'own_cgroup': hold.own_cgroup,
        'mapping': _mapping_json(reading.mapping),
        'state': reading.state.value,
        'last_collection': _time_json(reading.last_collection),
        'window': None,
    }
    if reading.window is not None:
        entry.update(_window_json(reading.window))
    return entry


def report_json(report: HoldsReport, *, summary: bool = False) -> dict[str, Any]:
    return {
        'roots': [_root_json(read) for read in report.roots],
        'samples_db': str(report.samples_db),
        'first_collection': _time_json(report.first_collection),
        'last_collection': _time_json(report.last_collection),
        'since': format_utc(report.since),
        'window_seconds': report.window_seconds,
        'holds': [_reading_json(reading) for reading in report.readings],
    }
