"""A project's task activity, read from its orchestrator run ledger.

ACCESS (PRD ``plans/dashboard-prd-lanes-prd.md`` §4.6). It reads the project's
``runs.db`` through the dashboard's read-only aiosqlite connection, so the
queries run off the event loop. Every ``events`` query filters on
``event_type``, which ``idx_events_type`` bounds; the earliest event is
``MIN(timestamp)``, which ``idx_events_ts`` answers.

No clock read: the moving window's cutoff and the Datum's ``as_of`` are the
caller's *now*.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType

import aiosqlite
from shared.timestamps import parse_timestamp_or_warn

from dashboard.data.datum import Datum, DatumState, unknown_datum
from dashboard.data.prd_view import MOVING_WINDOW_DAYS, TaskActivity
from dashboard.data.served_read import ServedRead
from dashboard.data.task_ids import parse_task_id

RUNS_DB = Path('data') / 'orchestrator' / 'runs.db'
"""Where a project keeps its run ledger, relative to its root."""

ACTIVITY_TTL_SECONDS = 60.0
"""Age below which a project's activity is served without reading its runs.db again."""

ACTIVITY_FRESHNESS_BOUND_SECONDS = 180
"""The activity Datum's freshness bound, three TTLs."""

ACTIVITY_REQUEST_WAIT_SECONDS = 4.0
"""Longest a ``/prds`` request waits on an in-flight activity read before it is
served the last good activity (or unknown); the read still lands for the next poll."""

ACTIVITY_QUERIES: Mapping[str, str] = MappingProxyType({
    'started': (
        "SELECT task_id, MAX(timestamp) FROM events "
        "WHERE event_type = 'task_started' AND timestamp >= ? GROUP BY task_id"
    ),
    'ever_started': "SELECT DISTINCT task_id FROM events WHERE event_type = 'task_started'",
    'completed_events': (
        "SELECT task_id, MAX(timestamp) FROM events "
        "WHERE event_type = 'task_completed' "
        "AND CASE WHEN json_valid(data) THEN json_extract(data, '$.outcome') END = 'done' "
        "GROUP BY task_id"
    ),
    'first_event': 'SELECT MIN(timestamp) FROM events',
    'has_task_results': (
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'task_results'"
    ),
    'completed_results': (
        "SELECT task_id, MAX(completed_at) FROM task_results "
        "WHERE outcome = 'done' AND completed_at IS NOT NULL GROUP BY task_id"
    ),
})
"""Every SQL text the activity read runs, by name."""

_activity: ServedRead[str, TaskActivity] = ServedRead(
    ttl_seconds=lambda: ACTIVITY_TTL_SECONDS,
    freshness_bound_seconds=ACTIVITY_FRESHNESS_BOUND_SECONDS,
)


async def acquire_activity(
    conn: aiosqlite.Connection | None,
    project_root: str | os.PathLike[str],
    *,
    now: datetime,
    wait_seconds: float | None = None,
    absent_reason: str | None = None,
) -> Datum[TaskActivity]:
    """*project_root*'s task activity as it reads at *now*; *conn* is its runs.db, if any.

    No runs.db is unknown: it reads the same as an unmounted volume. A
    runs.db without ``task_results`` is a lower bound, its completions read
    from ``task_completed`` events alone. *absent_reason* says why *conn* is
    None when the caller knows better than a missing file.
    """
    store = Path(project_root) / RUNS_DB

    async def read() -> Datum[TaskActivity]:
        if conn is None:
            return unknown_datum(
                absent_reason or f'no runs.db at {store}', ACTIVITY_FRESHNESS_BOUND_SECONDS,
            )
        try:
            return await _measure(conn, store, now)
        except sqlite3.Error as exc:
            return unknown_datum(
                f'runs.db at {store} could not be read: {type(exc).__name__}: {exc}',
                ACTIVITY_FRESHNESS_BOUND_SECONDS,
            )

    return await _activity.serve(str(project_root), read, now=now, wait_seconds=wait_seconds)


def clear_activity_state() -> None:
    """Forget every project's activity (test/admin hook)."""
    _activity.clear()


async def _measure(conn: aiosqlite.Connection, store: Path, now: datetime) -> Datum[TaskActivity]:
    cutoff = (now - timedelta(days=MOVING_WINDOW_DAYS)).astimezone(UTC).isoformat()
    started = _latest(await _query(conn, 'started', (cutoff,)))
    completed = _latest(await _query(conn, 'completed_events'))
    has_task_results = bool(await _query(conn, 'has_task_results'))
    if has_task_results:
        for task_id, at in _latest(await _query(conn, 'completed_results')).items():
            if task_id not in completed or at > completed[task_id]:
                completed[task_id] = at
    first_rows = await _query(conn, 'first_event')
    activity = TaskActivity(
        started=MappingProxyType(started),
        completed=MappingProxyType(completed),
        ever_started=frozenset(
            task_id for row in await _query(conn, 'ever_started')
            if (task_id := parse_task_id(row[0])) is not None
        ),
        first_event_at=_instant(first_rows[0][0]) if first_rows else None,
    )
    if has_task_results:
        return Datum(activity, now, DatumState.FRESH, None, ACTIVITY_FRESHNESS_BOUND_SECONDS)
    return Datum(
        activity, now, DatumState.LOWER_BOUND,
        f'runs.db at {store} has no task_results table, so completions are read from '
        'task_completed events alone',
        ACTIVITY_FRESHNESS_BOUND_SECONDS,
    )


async def _query(
    conn: aiosqlite.Connection, name: str, parameters: Sequence[object] = (),
) -> list[Sequence[object]]:
    return list(await conn.execute_fetchall(ACTIVITY_QUERIES[name], parameters))


def _latest(rows: Iterable[Sequence[object]]) -> dict[int, datetime]:
    """``{task id: instant}`` from ``(task_id, timestamp)`` rows, skipping unnumbered tasks."""
    latest: dict[int, datetime] = {}
    for raw_id, raw_at in rows:
        task_id, at = parse_task_id(raw_id), _instant(raw_at)
        if task_id is not None and at is not None:
            latest[task_id] = at
    return latest


def _instant(raw: object) -> datetime | None:
    if raw is None:
        return None
    at, parsed = parse_timestamp_or_warn(raw, context='task_activity')
    return at if parsed else None
