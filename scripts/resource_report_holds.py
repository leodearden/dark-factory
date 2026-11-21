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
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import _checkout as _checkout
from sampler.consumers import Consumer, Role, classify_member
from shared.project_layout import runs_db
from shared.psi import ARM_FIELDS
from shared.verify_scope_tag import scope_tag_for

__all__ = [
    'MEMORY_ARMS',
    'Hold',
    'HoldConsumers',
    'HoldsReport',
    'RootRead',
    'RootStatus',
    'SamplesUnavailable',
    'format_utc',
    'hold_consumers',
    'parse_utc',
    'read_holds_report',
    'read_root',
    'render_text',
    'report_json',
]


def _gate_arms(arms: frozenset[str]) -> frozenset[str]:
    strays = sorted(arms - ARM_FIELDS)
    if strays:
        raise ImportError(f'not shared.psi gate arms: {strays}; ARM_FIELDS is {sorted(ARM_FIELDS)}')
    return arms


MEMORY_ARMS: frozenset[str] = _gate_arms(frozenset({'mem_full_avg10', 'mem_some_avg10'}))
"""The gate arms whose holds are memory holds."""


def parse_utc(text: str) -> datetime:
    """An ISO date or datetime as an aware UTC datetime; a date is its midnight, a naive value UTC."""
    parsed = datetime.fromisoformat(text)
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).astimezone(UTC)


def format_utc(moment: datetime) -> str:
    """``YYYY-MM-DDTHH:MM:SSZ``: the one rendering of a time in the report."""
    return moment.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


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
    return Hold(root, event_id, at, arm, task_id, own_cgroup)


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


_SAMPLES_TABLE_SQL = "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'samples'"


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


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldsReport:
    roots: tuple[RootRead, ...]
    samples_db: Path
    since: datetime
    window_seconds: int
    readings: tuple[Hold, ...]


def read_holds_report(
    roots: Iterable[Path], samples_db: Path, *, since: datetime, window_seconds: int
) -> HoldsReport:
    """Every memory hold the roots' runs.db files hold since *since*, read against *samples_db*."""
    conn = _open_samples(samples_db)
    conn.close()
    reads = tuple(read_root(root, since) for root in roots)
    holds = sorted((hold for read in reads for hold in read.holds), key=lambda h: (h.at, str(h.project_root)))
    return HoldsReport(reads, samples_db, since, window_seconds, tuple(holds))


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


def _hold_count_line(report: HoldsReport) -> str:
    count = len(report.readings)
    noun = 'memory hold' if count == 1 else 'memory holds'
    return f'{count} {noun} since {format_utc(report.since)}'


def _header(report: HoldsReport) -> list[str]:
    return [
        f'roots ({len(report.roots)}):',
        *(_root_line(read) for read in report.roots),
        f'samples db: {report.samples_db}',
        _hold_count_line(report),
    ]


def render_text(report: HoldsReport, *, summary: bool = False) -> str:
    return '\n'.join(_header(report))


def _root_json(read: RootRead) -> dict[str, Any]:
    return {
        'root': str(read.root),
        'runs_db': str(read.runs_db),
        'status': read.status.value,
        'detail': read.detail,
        'skipped': read.skipped,
    }


def _hold_json(hold: Hold) -> dict[str, Any]:
    return {
        'at': format_utc(hold.at),
        'project_root': str(hold.project_root),
        'event_id': hold.event_id,
        'task_id': hold.task_id,
        'arm': hold.arm,
        'own_cgroup': hold.own_cgroup,
    }


def report_json(report: HoldsReport, *, summary: bool = False) -> dict[str, Any]:
    return {
        'roots': [_root_json(read) for read in report.roots],
        'samples_db': str(report.samples_db),
        'since': format_utc(report.since),
        'window_seconds': report.window_seconds,
        'holds': [_hold_json(hold) for hold in report.readings],
    }
