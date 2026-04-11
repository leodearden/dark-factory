"""Burndown snapshot collection, downsampling, and chart queries.

Periodically snapshots task status counts per project into a SQLite table.
The background collector (in app.py lifespan) writes via a dedicated writable
connection; route handlers read via DbPool (read-only).
"""

from __future__ import annotations

import asyncio
import contextlib
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


async def collect_snapshot(
    conn: aiosqlite.Connection,
    config: DashboardConfig,
) -> None:
    """Discover projects and insert one snapshot row per project."""
    try:
        await _collect_snapshot_impl(conn, config)
    except Exception:
        with contextlib.suppress(Exception):
            await conn.rollback()
        raise


async def _collect_snapshot_impl(
    conn: aiosqlite.Connection,
    config: DashboardConfig,
) -> None:
    """Internal implementation of collect_snapshot — called inside the outer try/except."""
    now = datetime.now(UTC).isoformat()
    # config.project_root is already resolved by DashboardConfig.__post_init__
    resolved_root = str(config.project_root)

    # Phase 1 — Discovery (sequential, in-memory):
    # Build the ordered list of (project_id_str, tasks_json_path) tuples to snapshot.
    # Main project is always first; seen_roots dedup is preserved exactly.
    # project_root is already canonical (symlink-resolved) so a symlinked
    # project_root deduplicates correctly against orchestrator /
    # known_project_roots entries that surface the real path.
    roots_to_snapshot: list[tuple[str, Path]] = []
    seen_roots: set[str] = {resolved_root}
    roots_to_snapshot.append((resolved_root, config.tasks_json))

    try:
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
            root_str = str(project_root.resolve())
            if root_str in seen_roots:
                continue
            seen_roots.add(root_str)
            roots_to_snapshot.append((root_str, project_root / '.taskmaster' / 'tasks' / 'tasks.json'))
    except Exception:
        logger.warning(
            'Orchestrator discovery failed; skipping orchestrator-discovered extras',
            exc_info=True,
        )

    for known_root in config.known_project_roots:
        # known_root is already resolved by DashboardConfig.__post_init__,
        # so .resolve() is unnecessary here.  Per-root error isolation for
        # load failures (PermissionError, etc.) is handled by Phase 2's
        # return_exceptions=True and Phase 3's exception check below.
        root_str = str(known_root)
        if root_str in seen_roots:
            continue
        seen_roots.add(root_str)
        roots_to_snapshot.append((root_str, known_root / '.taskmaster' / 'tasks' / 'tasks.json'))

    # Phase 2 — Parallel read:
    # All load_task_tree calls are independent (separate files), so run them concurrently.
    # load_task_tree catches OSError internally and returns [] for unreadable files,
    # but we pass return_exceptions=True as defense-in-depth so a single failing read
    # cannot sink the entire cycle and drop all snapshots before commit.
    all_tasks = await asyncio.gather(
        *(asyncio.to_thread(load_task_tree, tasks_json) for _, tasks_json in roots_to_snapshot),
        return_exceptions=True,
    )

    # Phase 3 — Sequential insert (per-project commit):
    # aiosqlite serialises writes on a single connection; keep inserts sequential.
    # Each project gets its own INSERT + commit so a DB failure on one project
    # cannot roll back rows that were already committed for earlier projects.
    # Main project is always roots_to_snapshot[0], so its commit fires first.
    # Skip any roots whose load raised — log and continue so other snapshots commit.
    for (root_str, _), tasks in zip(roots_to_snapshot, all_tasks, strict=True):
        if isinstance(tasks, BaseException):
            logger.warning('Failed to load tasks for %s', root_str, exc_info=tasks)
            continue
        counts = _count_statuses(tasks)
        try:
            await conn.execute(
                'INSERT INTO snapshots (project_id, ts, pending, in_progress, blocked, deferred, cancelled, done) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
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
            await conn.commit()
        except Exception:
            logger.warning('Failed to insert snapshot for %s', root_str, exc_info=True)
            with contextlib.suppress(Exception):
                await conn.rollback()


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
