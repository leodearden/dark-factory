"""Burndown snapshot collection, downsampling, and chart queries.

Periodically snapshots task status counts per project into a SQLite table.
The background collector (in app.py lifespan) writes via a dedicated writable
connection; route handlers read via DbPool (read-only).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite

from dashboard.config import DashboardConfig
from dashboard.data.orchestrator import (
    _read_project_root_from_config,
    _resolve_project_root,
    find_running_orchestrators,
    load_task_tree,
)

logger = logging.getLogger(__name__)

BURNDOWN_SCHEMA = """\
CREATE TABLE IF NOT EXISTS snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT    NOT NULL,
    ts          TEXT    NOT NULL,
    pending     INTEGER NOT NULL DEFAULT 0,
    in_progress INTEGER NOT NULL DEFAULT 0,
    blocked     INTEGER NOT NULL DEFAULT 0,
    deferred    INTEGER NOT NULL DEFAULT 0,
    cancelled   INTEGER NOT NULL DEFAULT 0,
    done        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_snapshots_project_ts ON snapshots(project_id, ts);
"""

_INSERT_SNAPSHOT_SQL = (
    'INSERT INTO snapshots (project_id, ts, pending, in_progress, blocked, deferred, cancelled, done) '
    'VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
)

# Maps raw task statuses to the 6 display zones.
_STATUS_MAP: dict[str, str] = {
    'pending': 'pending',
    'in-progress': 'in_progress',
    'review': 'in_progress',
    'blocked': 'blocked',
    'deferred': 'deferred',
    'cancelled': 'cancelled',
    'done': 'done',
}

_ZONE_KEYS = ('pending', 'in_progress', 'blocked', 'deferred', 'cancelled', 'done')


def _count_statuses(tasks: list[dict]) -> dict[str, int]:
    """Count tasks by mapped display zone."""
    counts: dict[str, int] = {k: 0 for k in _ZONE_KEYS}
    for task in tasks:
        raw = task.get('status', 'pending')
        zone = _STATUS_MAP.get(raw, 'pending')
        counts[zone] += 1
    return counts


async def _insert_snapshot_for_root(
    conn: aiosqlite.Connection,
    root_str: str,
    tasks_json: Path,
    now: str,
) -> None:
    """Load tasks from *tasks_json* and insert one snapshot row for *root_str*.

    The caller is responsible for committing the transaction.
    """
    tasks = await asyncio.to_thread(load_task_tree, tasks_json)
    counts = _count_statuses(tasks)
    await conn.execute(
        _INSERT_SNAPSHOT_SQL,
        (
            root_str,
            now,
            counts['pending'],
            counts['in_progress'],
            counts['blocked'],
            counts['deferred'],
            counts['cancelled'],
            counts['done'],
        ),
    )


async def collect_snapshot(
    conn: aiosqlite.Connection,
    config: DashboardConfig,
) -> None:
    """Discover projects and insert one snapshot row per project."""
    now = datetime.now(UTC).isoformat()

    # Always snapshot the main project.
    seen_roots: set[str] = {str(config.project_root)}
    await _insert_snapshot_for_root(conn, str(config.project_root), config.tasks_json, now)

    # Snapshot any additional projects discovered from running orchestrators.
    orchestrators = await asyncio.to_thread(find_running_orchestrators)
    for proc in orchestrators:
        if proc.get('prd'):
            project_root = _resolve_project_root(proc['prd'], config.project_root)
        elif proc.get('config_path'):
            resolved = _read_project_root_from_config(proc['config_path'])
            if resolved is None:
                continue
            project_root = resolved
        else:
            continue
        root_str = str(project_root)
        if root_str in seen_roots:
            continue
        seen_roots.add(root_str)
        await _insert_snapshot_for_root(
            conn, root_str, project_root / '.taskmaster' / 'tasks' / 'tasks.json', now
        )

    # Snapshot any additional projects from the configured known_project_roots list.
    # This handles projects whose orchestrators are not currently running.
    for known_root in config.known_project_roots:
        resolved = known_root.resolve()
        root_str = str(resolved)
        if root_str in seen_roots:
            continue
        seen_roots.add(root_str)
        await _insert_snapshot_for_root(
            conn, root_str, resolved / '.taskmaster' / 'tasks' / 'tasks.json', now
        )

    await conn.commit()


async def downsample(conn: aiosqlite.Connection) -> None:
    """Compact old snapshots: hourly after 7 days, expire after 90 days."""
    now = datetime.now(UTC)
    cutoff_7d = (now - timedelta(days=7)).isoformat()
    cutoff_90d = (now - timedelta(days=90)).isoformat()

    # Phase 1: For rows older than 7 days, keep only the last per (project_id, hour).
    await conn.execute(
        """
        DELETE FROM snapshots
        WHERE ts < ?
          AND id NOT IN (
              SELECT id FROM (
                  SELECT id, ROW_NUMBER() OVER (
                      PARTITION BY project_id, strftime('%%Y-%%m-%%dT%%H', ts)
                      ORDER BY ts DESC
                  ) AS rn
                  FROM snapshots
                  WHERE ts < ?
              )
              WHERE rn = 1
          )
        """,
        (cutoff_7d, cutoff_7d),
    )

    # Phase 2: Delete everything older than 90 days.
    await conn.execute('DELETE FROM snapshots WHERE ts < ?', (cutoff_90d,))

    await conn.commit()


# ---------------------------------------------------------------------------
# Read-side queries (used by route handlers via DbPool read-only connections)
# ---------------------------------------------------------------------------


async def get_burndown_projects(db: aiosqlite.Connection | None) -> list[str]:
    """Return distinct project IDs that have snapshot data."""
    if db is None:
        return []
    try:
        async with db.execute('SELECT DISTINCT project_id FROM snapshots ORDER BY project_id') as cur:
            rows = await cur.fetchall()
        return [row[0] for row in rows]
    except Exception:
        logger.warning('Error fetching burndown projects', exc_info=True)
        return []


async def get_burndown_series(
    db: aiosqlite.Connection | None,
    project_id: str,
    *,
    days: int = 7,
) -> dict:
    """Return time-series data for a project's burndown chart.

    Returns ``{labels: [...], done: [...], cancelled: [...], blocked: [...],
    deferred: [...], in_progress: [...], pending: [...]}``.
    """
    empty: dict = {
        'labels': [],
        'done': [],
        'cancelled': [],
        'blocked': [],
        'deferred': [],
        'in_progress': [],
        'pending': [],
    }
    if db is None:
        return empty
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    try:
        async with db.execute(
            'SELECT ts, done, cancelled, blocked, deferred, in_progress, pending '
            'FROM snapshots WHERE project_id = ? AND ts >= ? ORDER BY ts',
            (project_id, since),
        ) as cur:
            rows = await cur.fetchall()
    except Exception:
        logger.warning('Error fetching burndown series', exc_info=True)
        return empty

    labels = []
    done = []
    cancelled = []
    blocked = []
    deferred = []
    in_progress = []
    pending = []
    for row in rows:
        labels.append(row[0])
        done.append(row[1])
        cancelled.append(row[2])
        blocked.append(row[3])
        deferred.append(row[4])
        in_progress.append(row[5])
        pending.append(row[6])

    return {
        'labels': labels,
        'done': done,
        'cancelled': cancelled,
        'blocked': blocked,
        'deferred': deferred,
        'in_progress': in_progress,
        'pending': pending,
    }
