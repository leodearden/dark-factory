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
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any

import _checkout as _checkout
from sampler.consumers import (
    Consumer,
    Figure,
    FigureKind,
    Role,
    classify_member,
    encode_consumer,
    parse_consumer,
)
from sampler.increments import is_gap
from sampler.metrics import FIGURE_METRICS, METRIC_GROUPS, RECLAIM_METRICS, MetricSpec
from shared.project_layout import runs_db
from shared.psi import ARM_FIELDS, PsiSample
from shared.verify_scope_tag import scope_tag_for

__all__ = [
    'ASIDE_ROLES',
    'MEMORY_ARMS',
    'Aside',
    'ConsumerWindow',
    'Hold',
    'HoldConsumers',
    'HoldReading',
    'HoldState',
    'HoldsReport',
    'ProjectConsumer',
    'ProjectStatus',
    'RANKINGS',
    'RootRead',
    'RootStatus',
    'SamplerGap',
    'SamplesUnavailable',
    'Summary',
    'Totals',
    'Window',
    'aside',
    'format_utc',
    'hold_consumers',
    'holding_project',
    'incomplete',
    'parse_utc',
    'reclaim_threads_share',
    'read_holds_report',
    'read_root',
    'render_text',
    'report_json',
    'summarise',
    'total_positive_growth',
    'total_reclaimed',
    'who_grew',
    'who_paid',
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

    def by_role(self) -> Mapping[Role, Consumer | None]:
        return {
            Role.SERVICE: self.service,
            Role.AGENTS: self.agents,
            Role.VERIFY: self.verify,
            Role.PROJECT: self.project,
        }


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


def _cadence(spec: MetricSpec) -> int:
    (group,) = (group for group in METRIC_GROUPS if spec in group.metrics)
    if group.due_seconds is None:
        raise ImportError(f'{spec.name} is collected every tick: no collection interval')
    return group.due_seconds


_COLLECTION_INTERVAL = _cadence(_READ_OK)
"""The consumer group's nominal interval, which D3's gap rule (``sampler.increments.is_gap``) reads."""

_SAMPLES_TABLE_SQL = "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'samples'"
_FIRST_SQL = 'SELECT MIN(ts) FROM samples WHERE metric = ?'
_LAST_SQL = 'SELECT MAX(ts) FROM samples WHERE metric = ?'
_LATEST_SQL = 'SELECT MAX(ts) FROM samples WHERE metric = ? AND ts <= ?'
_PREVIOUS_SQL = 'SELECT MAX(ts) FROM samples WHERE metric = ? AND ts < ?'
_CLOCK_SQL = 'SELECT DISTINCT ts FROM samples WHERE metric = ? AND ts BETWEEN ? AND ? ORDER BY ts'
_WINDOW_SQL = 'SELECT ts, metric, leaf, value FROM samples WHERE metric IN ({}) AND ts BETWEEN ? AND ?'

_Row = tuple[int, str, str | None, float]


def _require_samples_table(conn: sqlite3.Connection, path: Path) -> None:
    try:
        found = conn.execute(_SAMPLES_TABLE_SQL).fetchone() is not None
    except sqlite3.Error as exc:
        raise SamplesUnavailable(f'samples store unreadable: {path} ({exc})') from exc
    if not found:
        raise SamplesUnavailable(f'samples store has no samples table: {path}')


def _open_samples(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise SamplesUnavailable(f'samples store missing: {path}')
    try:
        conn = _read_only(path)
    except sqlite3.Error as exc:
        raise SamplesUnavailable(f'samples store unreadable: {path} ({exc})') from exc
    try:
        _require_samples_table(conn, path)
    except BaseException:
        conn.close()
        raise
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

    def previous(self, ts: int) -> int | None:
        return self._ts(_PREVIOUS_SQL, _READ_OK.name, ts)

    def collections(self, start: int, end: int) -> list[int]:
        """The collections in [start, end], in order."""
        return [ts for (ts,) in self._conn.execute(_CLOCK_SQL, (_READ_OK.name, start, end))]

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
    GAP = 'sampler_gap'


_STATE_TEXT: Mapping[HoldState, str] = MappingProxyType({
    HoldState.RANKED: 'ranked over the window',
    HoldState.NOT_LIVE: 'recorder not live at this hold',
    HoldState.NO_SAMPLES: (
        'no samples: under two consumer collections in the window, '
        'not counting one just after a gap'
    ),
    HoldState.GAP: 'sampler gap in the window, so it is not ranked',
})


@dataclass(frozen=True)
class ConsumerWindow:
    """One consumer's figures over a window; absent at a collection counts as none (C1)."""

    consumer: Consumer
    program_growth_bytes: float
    swap_growth_bytes: float
    reclaimed_pages: float
    memory_some_stall_seconds: float
    memory_full_stall_seconds: float
    cpu_usec: float
    complete: bool
    """False when the consumer was flagged unreadable at any collection of the window."""

    @property
    def grew(self) -> float:
        """Program memory plus swap growth: a swap-out moves memory, it does not shrink it."""
        return self.program_growth_bytes + self.swap_growth_bytes


@dataclass(frozen=True)
class Window:
    """The collections a hold is read over: gauges at start and end, increments in (start, end]."""

    start: datetime
    end: datetime
    collections: int
    consumers: Mapping[Consumer, ConsumerWindow]
    reclaim_threads_cpu_usec: float | None
    """The kernel's reclaim threads' CPU over (start, end]; None when no collection recorded it."""
    host_cpu_usec: float
    """Every consumer's CPU over (start, end], which D3 makes the root's."""


_RECLAIM_FIGURES = (Figure.RECLAIM_KSWAPD_PAGES, Figure.RECLAIM_DIRECT_PAGES)
_WINDOW_FIGURES = (
    Figure.ANON,
    Figure.SWAP,
    *_RECLAIM_FIGURES,
    Figure.MEMORY_SOME_STALL_USEC,
    Figure.MEMORY_FULL_STALL_USEC,
    Figure.CPU_USEC,
)
_FIGURE_NAMED: Mapping[str, Figure] = MappingProxyType(
    {FIGURE_METRICS[figure].spec.name: figure for figure in _WINDOW_FIGURES}
)
_RECLAIM_THREADS = frozenset(spec.name for spec in RECLAIM_METRICS.values())
_WINDOW_SPECS = (
    _READ_OK,
    *(FIGURE_METRICS[figure].spec for figure in _WINDOW_FIGURES),
    *RECLAIM_METRICS.values(),
)


class _WindowRows:
    """The window's rows, sorted into gauges at its two ends and increments inside it."""

    def __init__(self, start: int, end: int) -> None:
        self.start, self.end = start, end
        self.collections: set[int] = set()
        self.gauges: dict[tuple[Consumer, Figure, int], float] = {}
        self.increments: dict[Consumer, dict[Figure, float]] = {}
        """Every consumer with a row in the window, and its counters summed over (start, end]."""
        self.unreadable: set[Consumer] = set()
        self.reclaim_threads_usec: float | None = None

    def add(self, ts: int, metric: str, leaf: str | None, value: float) -> None:
        if metric in _RECLAIM_THREADS:
            if ts > self.start:
                self.reclaim_threads_usec = (self.reclaim_threads_usec or 0.0) + value
            return
        if leaf is None:
            raise ValueError(f'a consumer metric row without a consumer: {metric} at {ts}')
        consumer = parse_consumer(leaf)
        increments = self.increments.setdefault(consumer, {})
        if metric == _READ_OK.name:
            self.collections.add(ts)
            if not value:
                self.unreadable.add(consumer)
            return
        figure = _FIGURE_NAMED[metric]
        if figure.kind is FigureKind.GAUGE:
            self.gauges[(consumer, figure, ts)] = value
        elif ts > self.start:
            increments[figure] = increments.get(figure, 0.0) + value

    def growth(self, consumer: Consumer, figure: Figure) -> float:
        return self.gauges.get((consumer, figure, self.end), 0.0) - self.gauges.get(
            (consumer, figure, self.start), 0.0
        )

    def consumer_window(self, consumer: Consumer) -> ConsumerWindow:
        increments = self.increments[consumer]
        return ConsumerWindow(
            consumer=consumer,
            program_growth_bytes=self.growth(consumer, Figure.ANON),
            swap_growth_bytes=self.growth(consumer, Figure.SWAP),
            reclaimed_pages=sum(increments.get(figure, 0.0) for figure in _RECLAIM_FIGURES),
            memory_some_stall_seconds=increments.get(Figure.MEMORY_SOME_STALL_USEC, 0.0),
            memory_full_stall_seconds=increments.get(Figure.MEMORY_FULL_STALL_USEC, 0.0),
            cpu_usec=increments.get(Figure.CPU_USEC, 0.0),
            complete=consumer not in self.unreadable,
        )


def _window(samples: _Samples, start: int, end: int) -> Window:
    rows = _WindowRows(start, end)
    for row in samples.rows(_WINDOW_SPECS, start, end):
        rows.add(*row)
    consumers = {c: rows.consumer_window(c) for c in sorted(rows.increments)}
    return Window(
        start=_utc(start),
        end=_utc(end),
        collections=len(rows.collections),
        consumers=MappingProxyType(consumers),
        reclaim_threads_cpu_usec=rows.reclaim_threads_usec,
        host_cpu_usec=sum(c.cpu_usec for c in consumers.values()),
    )


@dataclass(frozen=True)
class SamplerGap:
    """Two consecutive collections D3 calls a gap: the later one carries no increments."""

    before: datetime
    after: datetime


@dataclass(frozen=True)
class HoldReading:
    hold: Hold
    mapping: HoldConsumers
    state: HoldState
    window: Window | None
    last_collection: datetime | None
    """The latest collection at or before the hold."""
    gap: SamplerGap | None
    """The sampler gap the window spans, for a GAP reading."""


def _is_gap_collection(previous: int | None, ts: int) -> bool:
    return is_gap(previous, ts, _COLLECTION_INTERVAL)


def _spanned_gap(clock: Sequence[int]) -> SamplerGap | None:
    return next(
        (
            SamplerGap(_utc(before), _utc(after))
            for before, after in pairwise(clock)
            if _is_gap_collection(before, after)
        ),
        None,
    )


def _read_hold(samples: _Samples, hold: Hold, window_seconds: int) -> HoldReading:
    """Read *hold* over the window ending at the latest collection at or before it.

    The window opens at the latest collection at or before ``hold - window``,
    or at the first collection when the recorder went live inside the window.
    A window spanning a sampler gap is not ranked, since the collection after
    the gap carries no increments. A window never starts at a collection just
    after a gap, the recorder's first included: D3 flags every consumer there,
    so the window starts at the next one.
    """
    mapping = hold_consumers(hold.own_cgroup, hold.project_root)
    moment = hold.at.timestamp()
    end = samples.latest_at_or_before(moment)
    if end is None:
        return HoldReading(hold, mapping, HoldState.NOT_LIVE, None, None, None)
    opens = moment - window_seconds
    start = samples.latest_at_or_before(opens)
    if start is None:
        start = samples.first()
    clock = [] if start is None or end < opens else samples.collections(start, end)
    gap = _spanned_gap(clock)
    if gap is not None:
        return HoldReading(hold, mapping, HoldState.GAP, None, _utc(end), gap)
    if clock and _is_gap_collection(samples.previous(clock[0]), clock[0]):
        clock = clock[1:]
    if len(clock) < 2:
        return HoldReading(hold, mapping, HoldState.NO_SAMPLES, None, _utc(end), None)
    window = _window(samples, clock[0], end)
    return HoldReading(hold, mapping, HoldState.RANKED, window, _utc(end), None)


# ---------------------------------------------------------------------------
# The two rankings, and what is reported beside them
# ---------------------------------------------------------------------------

ASIDE_ROLES: Mapping[Role, str] = MappingProxyType(
    {Role.KERNEL: 'kernel', Role.OTHER: 'unattributed', Role.EXITED: 'exited'}
)
"""The roles reported beside the rankings rather than in them, each with its label."""


def _ranked(window: Window) -> list[ConsumerWindow]:
    return [c for c in window.consumers.values() if c.consumer.role not in ASIDE_ROLES]


def who_grew(window: Window) -> tuple[ConsumerWindow, ...]:
    """The consumers whose program memory plus swap grew, most first."""
    grown = (c for c in _ranked(window) if c.grew > 0)
    return tuple(sorted(grown, key=lambda c: (-c.grew, c.consumer)))


def who_paid(window: Window) -> tuple[ConsumerWindow, ...]:
    """The consumers pages were reclaimed from, most first; then their own memory stall."""
    paid = (
        c
        for c in _ranked(window)
        if c.reclaimed_pages > 0 or c.memory_full_stall_seconds > 0 or c.memory_some_stall_seconds > 0
    )
    return tuple(
        sorted(
            paid,
            key=lambda c: (
                -c.reclaimed_pages,
                -c.memory_full_stall_seconds,
                -c.memory_some_stall_seconds,
                c.consumer,
            ),
        )
    )


def _share(part: float, whole: float) -> float | None:
    return None if whole == 0 else part / whole


@dataclass(frozen=True)
class Aside:
    """One aside role's figures, and its shares of all consumers' positive growth and pages."""

    role: Role
    label: str
    positive_growth: float
    """The members' positive growth, on the same basis as ``total_positive_growth``."""
    net_grew: float
    """The members' growth net of any member that shrank."""
    reclaimed_pages: float
    growth_share: float | None
    reclaim_share: float | None


def total_positive_growth(window: Window) -> float:
    return sum(max(c.grew, 0.0) for c in window.consumers.values())


def total_reclaimed(window: Window) -> float:
    return sum(c.reclaimed_pages for c in window.consumers.values())


def aside(window: Window) -> tuple[Aside, ...]:
    """The kernel, unattributed and exited figures, in ``ASIDE_ROLES`` order."""
    growth, pages = total_positive_growth(window), total_reclaimed(window)
    rows = []
    for role, label in ASIDE_ROLES.items():
        members = [c for c in window.consumers.values() if c.consumer.role is role]
        positive_growth = sum(max(c.grew, 0.0) for c in members)
        net_grew = sum(c.grew for c in members)
        reclaimed = sum(c.reclaimed_pages for c in members)
        rows.append(
            Aside(
                role, label, positive_growth, net_grew, reclaimed,
                _share(positive_growth, growth), _share(reclaimed, pages),
            )
        )
    return tuple(rows)


RANKINGS: Mapping[str, Callable[[Window], tuple[ConsumerWindow, ...]]] = MappingProxyType(
    {'who_grew': who_grew, 'who_paid': who_paid}
)
"""The two rankings, by the name the report gives each."""


def reclaim_threads_share(window: Window) -> float | None:
    """The reclaim threads' share of host CPU; None when either is unavailable or CPU is 0."""
    if window.reclaim_threads_cpu_usec is None:
        return None
    return _share(window.reclaim_threads_cpu_usec, window.host_cpu_usec)


def incomplete(window: Window) -> tuple[Consumer, ...]:
    return tuple(c.consumer for c in window.consumers.values() if not c.complete)


class ProjectStatus(StrEnum):
    PRESENT = 'present'
    ABSENT = 'absent'
    NOT_MAPPED = 'not_mapped'


@dataclass(frozen=True)
class ProjectConsumer:
    """One of the holding project's consumers, as the window saw it; a rank is 1-based."""

    role: Role
    consumer: Consumer | None
    status: ProjectStatus
    reason: str | None
    figures: ConsumerWindow | None
    who_grew_rank: int | None
    who_paid_rank: int | None


_PROJECT_NOT_KEYED = (
    "the daemon's own cgroup is its unit, not a project slice, so this report cannot key "
    "the project's slice (it may still appear in the rankings)"
)


def _rank(consumer: Consumer, ranking: Sequence[ConsumerWindow]) -> int | None:
    return next((i for i, row in enumerate(ranking, start=1) if row.consumer == consumer), None)


def holding_project(mapping: HoldConsumers, window: Window) -> tuple[ProjectConsumer, ...]:
    """The holding project's service, agents, verify and project consumers over *window*."""
    grew, paid = who_grew(window), who_paid(window)
    rows = []
    for role, consumer in mapping.by_role().items():
        if consumer is None:
            reason = mapping.unmapped_reason or _PROJECT_NOT_KEYED
            rows.append(ProjectConsumer(role, None, ProjectStatus.NOT_MAPPED, reason, None, None, None))
            continue
        figures = window.consumers.get(consumer)
        status = ProjectStatus.ABSENT if figures is None else ProjectStatus.PRESENT
        rows.append(
            ProjectConsumer(
                role, consumer, status, None, figures, _rank(consumer, grew), _rank(consumer, paid)
            )
        )
    return tuple(rows)


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
# The summary across holds (C2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Totals:
    """One consumer's figures summed over the ranked holds' windows it was complete in."""

    grew: float = 0.0
    reclaimed_pages: float = 0.0
    memory_some_stall_seconds: float = 0.0
    memory_full_stall_seconds: float = 0.0
    incomplete_holds: int = 0
    """The ranked holds left out of the sums, the consumer being incomplete in their windows."""

    def plus(self, c: ConsumerWindow) -> Totals:
        if not c.complete:
            return replace(self, incomplete_holds=self.incomplete_holds + 1)
        return Totals(
            self.grew + c.grew,
            self.reclaimed_pages + c.reclaimed_pages,
            self.memory_some_stall_seconds + c.memory_some_stall_seconds,
            self.memory_full_stall_seconds + c.memory_full_stall_seconds,
            self.incomplete_holds,
        )


@dataclass(frozen=True)
class Summary:
    """Across the report's holds; every share is None where nothing was measured."""

    holds_per_project: Mapping[Path, Mapping[HoldState, int]]
    tops: Mapping[str, Mapping[Consumer, int]]
    """Per ranking in ``RANKINGS``, how many ranked holds each consumer topped while complete."""
    incomplete_tops: Mapping[str, Mapping[Consumer, int]]
    """Per ranking, the ranked holds whose top consumer was incomplete, by that consumer."""
    totals: Mapping[Consumer, Totals]
    aside_shares: Mapping[str, tuple[float | None, float | None]]
    """Per aside label, its share of the summed positive growth and of the summed pages."""
    reclaim_thread_share: float | None
    """Summed reclaim-thread CPU over summed host CPU, where the reclaim figure is available."""


def _holds_per_project(report: HoldsReport) -> dict[Path, dict[HoldState, int]]:
    counts = {read.root: dict.fromkeys(HoldState, 0) for read in report.roots}
    for reading in report.readings:
        counts[reading.hold.project_root][reading.state] += 1
    return counts


def _aside_shares(windows: Sequence[Window]) -> dict[str, tuple[float | None, float | None]]:
    rows = [row for window in windows for row in aside(window)]
    growth = sum(total_positive_growth(window) for window in windows)
    pages = sum(total_reclaimed(window) for window in windows)
    return {
        label: (
            _share(sum(row.positive_growth for row in rows if row.label == label), growth),
            _share(sum(row.reclaimed_pages for row in rows if row.label == label), pages),
        )
        for label in ASIDE_ROLES.values()
    }


def _reclaim_thread_share(windows: Sequence[Window]) -> float | None:
    measured = [w for w in windows if w.reclaim_threads_cpu_usec is not None]
    return _share(
        sum(w.reclaim_threads_cpu_usec or 0.0 for w in measured),
        sum(w.host_cpu_usec for w in measured),
    )


def _tops(windows: Sequence[Window], *, complete: bool) -> dict[str, Counter[Consumer]]:
    return {
        name: Counter(
            ranked[0].consumer
            for window in windows
            if (ranked := rank(window)) and ranked[0].complete is complete
        )
        for name, rank in RANKINGS.items()
    }


def summarise(report: HoldsReport) -> Summary:
    """The C2 summary, built from the ranked readings' windows alone.

    A consumer incomplete in a window is left out of that window's tops and
    totals, and counted apart, so a partial figure never reads as a whole one.
    """
    windows = [reading.window for reading in report.readings if reading.window is not None]
    totals: dict[Consumer, Totals] = {}
    for window in windows:
        for c in window.consumers.values():
            totals[c.consumer] = totals.get(c.consumer, Totals()).plus(c)
    return Summary(
        holds_per_project=_holds_per_project(report),
        tops=_tops(windows, complete=True),
        incomplete_tops=_tops(windows, complete=False),
        totals=dict(sorted(totals.items(), key=lambda item: (-item[1].grew, item[0]))),
        aside_shares=_aside_shares(windows),
        reclaim_thread_share=_reclaim_thread_share(windows),
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


_TOP = 5
_USEC_PER_SECOND = 1_000_000
_KiB = 2**10
_MiB = 2**20
_GiB = 2**30


def _bytes(value: float) -> str:
    sign = '-' if value < 0 else '+'
    size = abs(value)
    if size >= _GiB:
        return f'{sign}{size / _GiB:.2f} GiB'
    if size >= _MiB:
        return f'{sign}{size / _MiB:.1f} MiB'
    if size >= _KiB:
        return f'{sign}{size / _KiB:.0f} KiB'
    return f'{sign}{size:.0f} B'


def _pages(value: float) -> str:
    return f'{value:.0f} pages'


def _percent(share: float | None) -> str:
    return 'n/a' if share is None else f'{share:.1%}'


def _growth(c: ConsumerWindow) -> str:
    return (
        f'{_bytes(c.grew)} (program {_bytes(c.program_growth_bytes)}, '
        f'swap {_bytes(c.swap_growth_bytes)})'
    )


def _paid(c: ConsumerWindow) -> str:
    return (
        f'{_pages(c.reclaimed_pages)}, stall some {c.memory_some_stall_seconds:.1f} s, '
        f'full {c.memory_full_stall_seconds:.1f} s'
    )


def _ranking_lines(
    title: str,
    empty: str,
    ranking: Sequence[ConsumerWindow],
    figures: Callable[[ConsumerWindow], str],
    marks: Mapping[Consumer, str],
) -> list[str]:
    if not ranking:
        return [f'  {empty}']
    return [f'  {title}:'] + [
        f'    {i}. {encode_consumer(c.consumer)}  {figures(c)}{marks.get(c.consumer, "")}'
        for i, c in enumerate(ranking[:_TOP], start=1)
    ]


def _reclaim_threads(window: Window) -> str:
    usec = window.reclaim_threads_cpu_usec
    if usec is None:
        return 'reclaim threads unavailable'
    share = _percent(reclaim_threads_share(window))
    return f'reclaim threads {usec / _USEC_PER_SECOND:.1f} s CPU ({share} of host CPU)'


def _aside_line(row: Aside, window: Window) -> str:
    line = (
        f'  {row.label}: grew {_bytes(row.positive_growth)} ({_percent(row.growth_share)} of growth, '
        f'net {_bytes(row.net_grew)}), '
        f'{_pages(row.reclaimed_pages)} reclaimed ({_percent(row.reclaim_share)} of pages)'
    )
    return f'{line}; {_reclaim_threads(window)}' if row.role is Role.KERNEL else line


_INCOMPLETE_MARK = '  [incomplete]'


def _completeness(c: ConsumerWindow) -> str:
    return '' if c.complete else _INCOMPLETE_MARK


def _ranks(row: ProjectConsumer) -> str:
    def rank(value: int | None) -> str:
        return 'not ranked' if value is None else str(value)

    return f'who grew: {rank(row.who_grew_rank)}; who paid: {rank(row.who_paid_rank)}'


def _project_line(row: ProjectConsumer) -> str:
    if row.consumer is None:
        return f'    {row.role.value}: not mapped ({row.reason})'
    name = encode_consumer(row.consumer)
    if row.figures is None:
        return f'    {row.role.value}: {name} absent, no rows in the window'
    return (
        f'    {row.role.value}: {name} grew {_bytes(row.figures.grew)}, '
        f'{_pages(row.figures.reclaimed_pages)}; {_ranks(row)}{_completeness(row.figures)}'
    )


def _marks(mapping: HoldConsumers, window: Window) -> dict[Consumer, str]:
    project = {
        consumer: f"  [this project's {role.value}]"
        for role, consumer in mapping.by_role().items()
        if consumer is not None
    }
    return {
        consumer: project.get(consumer, '') + _completeness(row)
        for consumer, row in window.consumers.items()
    }


def _window_lines(mapping: HoldConsumers, window: Window) -> list[str]:
    marks = _marks(mapping, window)
    unreadable = incomplete(window)
    return [
        f'  window {format_utc(window.start)} to {format_utc(window.end)}, '
        f'{window.collections} collections',
        *_ranking_lines(
            'who grew (program + swap)', 'who grew: no consumer grew', who_grew(window), _growth, marks
        ),
        *_ranking_lines(
            'who paid (pages reclaimed, own memory stall)',
            'who paid: no consumer paid',
            who_paid(window),
            _paid,
            marks,
        ),
        *(_aside_line(row, window) for row in aside(window)),
        '  this project:',
        *(_project_line(row) for row in holding_project(mapping, window)),
        *(
            [f'  incomplete (unreadable in the window): {", ".join(map(encode_consumer, unreadable))}']
            if unreadable
            else []
        ),
    ]


def _hold_block(reading: HoldReading) -> list[str]:
    if reading.window is not None:
        return [_heading(reading.hold), *_window_lines(reading.mapping, reading.window)]
    gap, last = reading.gap, reading.last_collection
    between = (
        '' if gap is None
        else f'; no collection between {format_utc(gap.before)} and {format_utc(gap.after)}'
    )
    after = '' if last is None else f'; the last collection was {format_utc(last)}'
    return [_heading(reading.hold), f'  {_STATE_TEXT[reading.state]}{between}{after}']


def _state_counts(counts: Mapping[HoldState, int]) -> str:
    return ', '.join(f'{state.value} {count}' for state, count in counts.items())


def _tops_line(title: str, tops: Mapping[Consumer, int]) -> str:
    topped = ', '.join(f'{encode_consumer(c)} {n}' for c, n in Counter(tops).most_common())
    return f'  {title}: {topped or "none"}'


def _totals_line(consumer: Consumer, totals: Totals) -> str:
    left_out = totals.incomplete_holds
    return (
        f'    {encode_consumer(consumer)}  grew {_bytes(totals.grew)}, '
        f'{_pages(totals.reclaimed_pages)}, stall some {totals.memory_some_stall_seconds:.1f} s, '
        f'full {totals.memory_full_stall_seconds:.1f} s'
        + (f"; {left_out} hold{'' if left_out == 1 else 's'} left out, incomplete" if left_out else '')
    )


def _summary_lines(report: HoldsReport) -> list[str]:
    summary = summarise(report)
    ranked = sum(reading.window is not None for reading in report.readings)
    return [
        f'summary across {len(report.readings)} memory holds ({ranked} ranked):',
        '  holds per project:',
        *(
            f'    {root}: {sum(counts.values())} ({_state_counts(counts)})'
            for root, counts in summary.holds_per_project.items()
        ),
        *(_tops_line(f'times topping {name}', tops) for name, tops in summary.tops.items()),
        *(
            _tops_line(f'times topping {name} while incomplete, left out above', tops)
            for name, tops in summary.incomplete_tops.items()
        ),
        '  totals per consumer over the ranked holds, by growth:',
        *(_totals_line(consumer, totals) for consumer, totals in summary.totals.items()),
        *(
            f'  {label}: {_percent(growth)} of growth, {_percent(pages)} of pages'
            for label, (growth, pages) in summary.aside_shares.items()
        ),
        f'  reclaim threads: {_percent(summary.reclaim_thread_share)} of host CPU',
    ]


def render_text(report: HoldsReport, *, summary: bool = False) -> str:
    sections = [_header(report), *(_hold_block(reading) for reading in report.readings)]
    if summary:
        sections.append(_summary_lines(report))
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


def _figures_json(c: ConsumerWindow) -> dict[str, Any]:
    return {
        'consumer': encode_consumer(c.consumer),
        'grew_bytes': c.grew,
        'program_growth_bytes': c.program_growth_bytes,
        'swap_growth_bytes': c.swap_growth_bytes,
        'reclaimed_pages': c.reclaimed_pages,
        'memory_some_stall_seconds': c.memory_some_stall_seconds,
        'memory_full_stall_seconds': c.memory_full_stall_seconds,
        'cpu_usec': c.cpu_usec,
        'complete': c.complete,
    }


def _aside_json(row: Aside) -> dict[str, Any]:
    return {
        'label': row.label,
        'positive_growth_bytes': row.positive_growth,
        'net_grew_bytes': row.net_grew,
        'reclaimed_pages': row.reclaimed_pages,
        'growth_share': row.growth_share,
        'reclaim_share': row.reclaim_share,
    }


def _project_json(row: ProjectConsumer) -> dict[str, Any]:
    return {
        'consumer': _consumer_json(row.consumer),
        'status': row.status.value,
        'reason': row.reason,
        'figures': None if row.figures is None else _figures_json(row.figures),
        'who_grew_rank': row.who_grew_rank,
        'who_paid_rank': row.who_paid_rank,
    }


def _window_json(mapping: HoldConsumers, window: Window) -> dict[str, Any]:
    return {
        'window': {
            'start': format_utc(window.start),
            'end': format_utc(window.end),
            'collections': window.collections,
        },
        **{name: [_figures_json(c) for c in rank(window)] for name, rank in RANKINGS.items()},
        'aside': [_aside_json(row) for row in aside(window)],
        'holding_project': {
            row.role.value: _project_json(row) for row in holding_project(mapping, window)
        },
        'incomplete': [encode_consumer(c) for c in incomplete(window)],
        'reclaim_threads_cpu_usec': window.reclaim_threads_cpu_usec,
        'host_cpu_usec': window.host_cpu_usec,
        'reclaim_threads_share': reclaim_threads_share(window),
    }


def _gap_json(gap: SamplerGap) -> dict[str, str]:
    return {'before': format_utc(gap.before), 'after': format_utc(gap.after)}


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
        'gap': None if reading.gap is None else _gap_json(reading.gap),
        'window': None,
    }
    if reading.window is not None:
        entry.update(_window_json(reading.mapping, reading.window))
    return entry


def _totals_json(totals: Totals) -> dict[str, Any]:
    return {
        'grew_bytes': totals.grew,
        'reclaimed_pages': totals.reclaimed_pages,
        'memory_some_stall_seconds': totals.memory_some_stall_seconds,
        'memory_full_stall_seconds': totals.memory_full_stall_seconds,
        'incomplete_holds': totals.incomplete_holds,
    }


def _tops_json(tops: Mapping[str, Mapping[Consumer, int]]) -> dict[str, dict[str, int]]:
    return {
        name: {encode_consumer(c): n for c, n in Counter(counts).most_common()}
        for name, counts in tops.items()
    }


def _summary_json(summary: Summary) -> dict[str, Any]:
    return {
        'holds_per_project': {
            str(root): {'total': sum(counts.values()), **{s.value: n for s, n in counts.items()}}
            for root, counts in summary.holds_per_project.items()
        },
        'tops': _tops_json(summary.tops),
        'incomplete_tops': _tops_json(summary.incomplete_tops),
        'totals': {encode_consumer(c): _totals_json(t) for c, t in summary.totals.items()},
        'aside_shares': {
            label: {'growth_share': growth, 'reclaim_share': pages}
            for label, (growth, pages) in summary.aside_shares.items()
        },
        'reclaim_thread_share': summary.reclaim_thread_share,
    }


def report_json(report: HoldsReport, *, summary: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        'roots': [_root_json(read) for read in report.roots],
        'samples_db': str(report.samples_db),
        'first_collection': _time_json(report.first_collection),
        'last_collection': _time_json(report.last_collection),
        'since': format_utc(report.since),
        'window_seconds': report.window_seconds,
        'holds': [_reading_json(reading) for reading in report.readings],
    }
    if summary:
        out['summary'] = _summary_json(summarise(report))
    return out
