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

_INSERT_SNAPSHOT_SQL = (
    'INSERT INTO snapshots (project_id, ts, pending, in_progress, blocked, deferred, cancelled, done) '
    'VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
)


def _tasks_json_for(root: Path) -> Path:
    """Return the canonical tasks.json path for *root*."""
    return root / '.taskmaster' / 'tasks' / 'tasks.json'


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
    """Discover projects and insert one snapshot row per project.

    Partial-failure semantics:

    - **Main project committed first**: the main project is always
      roots_to_snapshot[0], so its per-project INSERT + commit fires before
      any extra project is attempted.  A DB failure on an extra cannot roll
      back a row that is already committed.

    - **Orchestrator discovery is best-effort**: the block that calls
      find_running_orchestrators and iterates the results is wrapped in
      try/except.  If it raises, a WARNING is logged and the function
      degrades gracefully — the main project and known_project_roots are
      still snapshotted.

    - **Extra inserts are per-project / isolated**: each project in Phase 3
      gets its own INSERT + commit wrapped in try/except.  A disk-full or
      constraint error on one project is logged as a WARNING and the loop
      continues; other projects are unaffected.

    - **Explicit rollback on unexpected error** (defensive guard): if an
      exception escapes all inner guards (e.g., a future code change
      introduces a DB write before Phase 3), conn.rollback() is called
      before re-raising.  This is purely defensive — the realistic failure
      modes (per-project INSERT errors) are already handled by Phase 3's
      per-project try/except.  The guard ensures the long-lived writer
      connection in app.py is left in a clean state even for unexpected
      regressions.
    """
    try:
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
        except Exception:
            logger.warning(
                'Orchestrator discovery failed; skipping orchestrator-discovered extras',
                exc_info=True,
            )
        else:
            for proc in orchestrators:
                try:
                    if proc.get('prd'):
                        project_root = _resolve_project_root(proc['prd'], config.project_root)
                    elif proc.get('config_path'):
                        resolved = _read_project_root_from_config(proc['config_path'])
                        if resolved is None:
                            continue
                        project_root = resolved
                    else:
                        continue
                    # project_root is already resolved by _resolve_project_root / _read_project_root_from_config
                    root_str = str(project_root)
                    if root_str in seen_roots:
                        continue
                    seen_roots.add(root_str)
                    roots_to_snapshot.append((root_str, _tasks_json_for(project_root)))
                except OSError:
                    logger.warning(
                        'OSError while resolving orchestrator project root; skipping',
                        exc_info=True,
                    )
                except Exception:
                    logger.warning(
                        'Orchestrator entry processing failed; skipping entry',
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
            roots_to_snapshot.append((root_str, _tasks_json_for(known_root)))

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
        # O(N) commits (one fsync per project) are acceptable at current scale
        # (handful of projects every 10 minutes). If N grows significantly,
        # consider SAVEPOINT-based isolation for a single trailing fsync.
        for (root_str, _), tasks in zip(roots_to_snapshot, all_tasks, strict=True):
            if isinstance(tasks, BaseException):
                logger.warning('Failed to load tasks for %s', root_str, exc_info=tasks)
                continue
            try:
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
                await conn.commit()
            except Exception:
                logger.warning('Failed to insert snapshot for %s', root_str, exc_info=True)
                with contextlib.suppress(Exception):
                    await conn.rollback()
    except Exception:
        with contextlib.suppress(Exception):
            await conn.rollback()
        raise


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


async def aggregate_burndown_projects(
    dbs: list[aiosqlite.Connection | None],
) -> list[str]:
    """Return distinct project IDs across *all* burndown DBs, sorted.

    Calls :func:`get_burndown_projects` for each DB in *dbs* concurrently via
    ``asyncio.gather``, then unions the results and returns a sorted list.
    ``None`` entries are tolerated (``get_burndown_projects`` returns ``[]``
    for ``None``).
    """
    if not dbs:
        return []
    per_db: list[list[str]] = list(await asyncio.gather(
        *(get_burndown_projects(db) for db in dbs)
    ))
    seen: set[str] = set()
    for project_list in per_db:
        seen.update(project_list)
    return sorted(seen)


async def aggregate_burndown_series(
    dbs: list[aiosqlite.Connection | None],
    project_id: str,
    *,
    days: int = 7,
) -> dict:
    """Return merged time-series data for *project_id* across all burndown DBs.

    Calls :func:`get_burndown_series` for each DB in *dbs* concurrently, then
    merges by timestamp label using a last-writer-wins strategy (later DBs in
    the list overwrite earlier ones for the same timestamp).  The result is
    sorted by timestamp and returned in the same dict-of-lists shape as
    :func:`get_burndown_series`.

    Returns the empty-series default ``{labels: [], done: [], ...}`` when no
    rows are found across any DB.
    """
    _keys = ('done', 'cancelled', 'blocked', 'deferred', 'in_progress', 'pending')
    empty: dict = {'labels': [], **{k: [] for k in _keys}}

    if not dbs:
        return empty

    per_db: list[dict] = list(await asyncio.gather(
        *(get_burndown_series(db, project_id, days=days) for db in dbs)
    ))

    # Why last-writer-wins and not sum-of-counts (as used in performance.py /
    # costs.py)?  Burndown values are *snapshots* of task state at an instant,
    # not counts of independent events.  Summing two snapshots recorded at the
    # same timestamp would double-count every task state.  In the expected
    # deployment a single collector writes all projects to the main DB, so two
    # distinct burndown DBs should never share a (project_id, timestamp) pair.
    # If they do (misconfigured overlapping roots), the later DB's values win
    # and a warning is emitted so the misconfiguration is visible in logs.
    merged: dict[str, dict[str, int]] = {}
    collisions = 0
    first_colliding: list[str] = []
    for series in per_db:
        for i, label in enumerate(series['labels']):
            if label in merged:
                collisions += 1
                if len(first_colliding) < 3:
                    first_colliding.append(label)
            merged[label] = {k: series[k][i] for k in _keys}
    if collisions:
        logger.warning(
            'aggregate_burndown_series: %d timestamp collisions for project %r '
            '— last-writer-wins applied; check for overlapping project roots '
            '(first: %s)',
            collisions,
            project_id,
            ', '.join(first_colliding),
        )

    if not merged:
        return empty

    sorted_labels = sorted(merged)
    result: dict = {'labels': sorted_labels}
    for k in _keys:
        result[k] = [merged[label][k] for label in sorted_labels]
    return result


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


_VELOCITY_FLOOR = 0.1  # tasks/day; prevents forecast from going to infinity


def compute_forecast_confidence(series: dict) -> dict:
    """Return ``{forecast_low, forecast_high}`` days from a burndown series.

    Reads ``done`` and ``pending`` over the series' ``labels`` (each label
    is a snapshot timestamp).  Computes two velocities — one over the most
    recent 7 days of data, one over the full series — and returns the
    forecast clearance window.

    Returns ``{forecast_low: None, forecast_high: None}`` when the series
    has fewer than 7 distinct days of history (no synthesis on sparse
    history; the dashboard renders ``—``).
    """
    none = {'forecast_low': None, 'forecast_high': None}
    labels = series.get('labels') or []
    done = series.get('done') or []
    pending = series.get('pending') or []
    if not labels or not done or not pending:
        return none
    if len(labels) != len(done) or len(labels) != len(pending):
        return none

    # Distinct day count from ISO labels (date prefix).
    day_keys: list[str] = []
    seen: set[str] = set()
    for lbl in labels:
        day = (lbl[:10] if isinstance(lbl, str) and len(lbl) >= 10 else None)
        if day and day not in seen:
            seen.add(day)
            day_keys.append(day)
    if len(day_keys) < 7:
        return none

    last_pending = pending[-1] or 0
    if last_pending <= 0:
        return {'forecast_low': 0, 'forecast_high': 0}

    last_done = done[-1] or 0

    def _velocity_over(window_days: int) -> float:
        cutoff = day_keys[-window_days]
        # First label whose day >= cutoff.
        first_idx = 0
        for i, lbl in enumerate(labels):
            if isinstance(lbl, str) and len(lbl) >= 10 and lbl[:10] >= cutoff:
                first_idx = i
                break
        first_done = done[first_idx] or 0
        delta_done = max(0, last_done - first_done)
        # Duration in days, with a minimum of 1.0 to avoid div-by-zero on
        # narrow windows.
        return max(_VELOCITY_FLOOR, delta_done / max(1.0, window_days))

    v_recent = _velocity_over(7)
    v_lifetime = _velocity_over(len(day_keys))

    f_recent = last_pending / v_recent
    f_lifetime = last_pending / v_lifetime
    return {
        'forecast_low': round(min(f_recent, f_lifetime), 1),
        'forecast_high': round(max(f_recent, f_lifetime), 1),
    }


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
