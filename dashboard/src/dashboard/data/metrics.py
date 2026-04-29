"""Periodic metrics-snapshot collection and chart queries.

Records sparse-history signals (orchestrator running count, fused-memory
store sizes, write queue depth, reconciliation state, merge queue active
depth) into a dedicated SQLite database so they can be rendered as time
series even though the underlying sources are point-in-time only.

Sampling cadence and downsampling rules match
``dashboard/data/burndown.py`` so both files retain at most 90 days of
history (10-minute raw, hourly after 7 days). The collector runs on a
sibling lifespan task in ``app.py``; route handlers read via the shared
``DbPool`` (read-only).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite
import httpx

from dashboard.config import DashboardConfig
from dashboard.data.merge_queue import active_queued_merges
from dashboard.data.memory import get_memory_status, get_queue_stats
from dashboard.data.orchestrator import (
    _read_project_root_from_config,
    _resolve_project_root,
    find_running_orchestrators,
)
from dashboard.data.reconciliation import get_buffer_stats, get_burst_state, partition_burst_state

logger = logging.getLogger(__name__)

# 5s ceiling on any fused-memory HTTP call so a hung MCP cannot extend
# the 10-minute sampling cycle.
_HTTP_SAMPLER_TIMEOUT_SECONDS = 5.0

METRICS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS orchestrator_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    project_id    TEXT    NOT NULL,
    running_count INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orch_snap_pid_ts ON orchestrator_snapshots(project_id, ts);

CREATE TABLE IF NOT EXISTS memory_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    project_id      TEXT    NOT NULL,
    graphiti_nodes  INTEGER,
    mem0_memories   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_mem_snap_pid_ts ON memory_snapshots(project_id, ts);

CREATE TABLE IF NOT EXISTS queue_snapshots (
    ts      TEXT PRIMARY KEY,
    pending INTEGER,
    retry   INTEGER,
    dead    INTEGER
);

CREATE TABLE IF NOT EXISTS recon_snapshots (
    ts             TEXT PRIMARY KEY,
    buffered_count INTEGER,
    active_agents  INTEGER
);

CREATE TABLE IF NOT EXISTS merge_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT    NOT NULL,
    project_id   TEXT    NOT NULL,
    active_count INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_merge_snap_pid_ts ON merge_snapshots(project_id, ts);
"""


# ---------------------------------------------------------------------------
# Per-sampler helpers (return rows; never raise to the caller)
# ---------------------------------------------------------------------------


def _sample_orchestrators(config: DashboardConfig) -> list[tuple[str, int]]:
    """Return [(project_id, running_count)] for all detected orchestrators.

    Mirrors the discovery logic in burndown.collect_snapshot so per-project
    counts agree across files. Returns [] on failure (caller logs).
    """
    seen: dict[str, int] = {str(config.project_root): 0}
    procs = find_running_orchestrators()
    for proc in procs:
        try:
            if proc.get('prd'):
                root = _resolve_project_root(proc['prd'], config.project_root)
            elif proc.get('config_path'):
                resolved = _read_project_root_from_config(proc['config_path'])
                if resolved is None:
                    continue
                root = resolved
            else:
                root = config.project_root.resolve()
            key = str(root)
            seen[key] = seen.get(key, 0) + 1
        except (OSError, ValueError):
            logger.debug('Orchestrator entry sampling failed', exc_info=True)
    # Include known_project_roots even with zero count so the spark line
    # has continuity when an orchestrator stops running.
    for known in config.known_project_roots:
        seen.setdefault(str(known), 0)
    return list(seen.items())


def _split_status(status: dict) -> tuple[list[tuple[str, dict]], dict | None]:
    """Pull (project_id, project_dict) pairs out of a fused-memory status.

    Returns (project_pairs, queue_dict_or_none). On offline or malformed
    payloads, both pieces degrade to empty/None — never raises.
    """
    if not isinstance(status, dict) or status.get('offline'):
        return [], None
    projects = status.get('projects')
    pairs: list[tuple[str, dict]] = []
    if isinstance(projects, dict):
        for pid, payload in projects.items():
            if isinstance(pid, str) and isinstance(payload, dict):
                pairs.append((pid, payload))
    queue = status.get('queue') if isinstance(status.get('queue'), dict) else None
    return pairs, queue


def _split_queue_stats(qstats: dict) -> tuple[int | None, int | None, int | None]:
    """Extract pending/retry/dead from a get_queue_stats result.

    Returns (None, None, None) on offline / malformed payloads.
    """
    if not isinstance(qstats, dict) or qstats.get('offline'):
        return None, None, None
    counts = qstats.get('counts') if isinstance(qstats.get('counts'), dict) else {}
    return counts.get('pending'), counts.get('retry'), counts.get('dead')


# ---------------------------------------------------------------------------
# Snapshot driver — runs every 10 minutes from the lifespan task.
# ---------------------------------------------------------------------------


async def collect_metrics_snapshot(
    conn: aiosqlite.Connection,
    config: DashboardConfig,
    http_client: httpx.AsyncClient,
    recon_db: aiosqlite.Connection | None,
    merge_dbs: list[tuple[str, aiosqlite.Connection | None]],
) -> None:
    """Sample every metric source and insert one row per signal.

    Each sampler runs in its own try/except; a failure in one source (e.g.,
    fused-memory HTTP timeout) does not poison the others. Per-row inserts
    commit individually so a write error on one signal cannot drop earlier
    rows.
    """
    now = datetime.now(UTC).isoformat()

    # Orchestrators (synchronous; subprocess in to_thread).
    try:
        rows = await asyncio.to_thread(_sample_orchestrators, config)
        for pid, count in rows:
            await conn.execute(
                'INSERT INTO orchestrator_snapshots (ts, project_id, running_count) VALUES (?, ?, ?)',
                (now, pid, count),
            )
        await conn.commit()
    except Exception:
        logger.warning('orchestrator sampler failed', exc_info=True)
        with contextlib.suppress(Exception):
            await conn.rollback()

    # Memory (HTTP via fused-memory MCP — wrap in wait_for as belt-and-braces).
    try:
        status = await asyncio.wait_for(
            get_memory_status(http_client, config),
            timeout=_HTTP_SAMPLER_TIMEOUT_SECONDS,
        )
        pairs, _queue_inline = _split_status(status)
        for pid, payload in pairs:
            await conn.execute(
                'INSERT INTO memory_snapshots (ts, project_id, graphiti_nodes, mem0_memories) '
                'VALUES (?, ?, ?, ?)',
                (now, pid, payload.get('graphiti_nodes'), payload.get('mem0_memories')),
            )
        await conn.commit()
    except (TimeoutError, Exception):
        logger.warning('memory sampler failed', exc_info=True)
        with contextlib.suppress(Exception):
            await conn.rollback()

    # Write queue (HTTP, separate call; tolerate either source failing).
    try:
        qstats = await asyncio.wait_for(
            get_queue_stats(http_client, config),
            timeout=_HTTP_SAMPLER_TIMEOUT_SECONDS,
        )
        pending, retry, dead = _split_queue_stats(qstats)
        await conn.execute(
            'INSERT OR REPLACE INTO queue_snapshots (ts, pending, retry, dead) VALUES (?, ?, ?, ?)',
            (now, pending, retry, dead),
        )
        await conn.commit()
    except (TimeoutError, Exception):
        logger.warning('queue sampler failed', exc_info=True)
        with contextlib.suppress(Exception):
            await conn.rollback()

    # Reconciliation (read-only SQLite — get_buffer_stats + active-agent count).
    try:
        buf = await get_buffer_stats(recon_db)
        burst = await get_burst_state(recon_db)
        active, _idle = partition_burst_state(burst)
        await conn.execute(
            'INSERT OR REPLACE INTO recon_snapshots (ts, buffered_count, active_agents) '
            'VALUES (?, ?, ?)',
            (now, buf.get('buffered_count'), len(active)),
        )
        await conn.commit()
    except Exception:
        logger.warning('recon sampler failed', exc_info=True)
        with contextlib.suppress(Exception):
            await conn.rollback()

    # Merge queue (per-project active count from runs.db events).
    try:
        for pid, db in merge_dbs:
            try:
                merges = await active_queued_merges(db)
            except Exception:
                logger.debug('merge sampler failed for %s', pid, exc_info=True)
                continue
            await conn.execute(
                'INSERT INTO merge_snapshots (ts, project_id, active_count) VALUES (?, ?, ?)',
                (now, pid, len(merges)),
            )
        await conn.commit()
    except Exception:
        logger.warning('merge sampler driver failed', exc_info=True)
        with contextlib.suppress(Exception):
            await conn.rollback()


# ---------------------------------------------------------------------------
# Downsampling — same policy as burndown (raw 10min, hourly after 7d, drop 90d).
# ---------------------------------------------------------------------------


_DOWNSAMPLE_TABLES = (
    ('orchestrator_snapshots', 'project_id'),
    ('memory_snapshots', 'project_id'),
    ('queue_snapshots', None),
    ('recon_snapshots', None),
    ('merge_snapshots', 'project_id'),
)


async def downsample_metrics(conn: aiosqlite.Connection) -> None:
    """Compact old metrics rows: hourly after 7 days, drop after 90 days.

    For per-project tables, partition key is (project_id, hour); for
    system-wide tables, partition key is just hour.
    """
    now = datetime.now(UTC)
    cutoff_7d = (now - timedelta(days=7)).isoformat()
    cutoff_90d = (now - timedelta(days=90)).isoformat()

    for table, pid_col in _DOWNSAMPLE_TABLES:
        rowid_col = 'id' if pid_col is not None else 'rowid'
        partition = (
            f"{pid_col}, strftime('%%Y-%%m-%%dT%%H', ts)"
            if pid_col is not None
            else "strftime('%%Y-%%m-%%dT%%H', ts)"
        )
        await conn.execute(
            f"""
            DELETE FROM {table}
            WHERE ts < ?
              AND {rowid_col} NOT IN (
                  SELECT {rowid_col} FROM (
                      SELECT {rowid_col}, ROW_NUMBER() OVER (
                          PARTITION BY {partition}
                          ORDER BY ts DESC
                      ) AS rn
                      FROM {table}
                      WHERE ts < ?
                  )
                  WHERE rn = 1
              )
            """,
            (cutoff_7d, cutoff_7d),
        )
        await conn.execute(f'DELETE FROM {table} WHERE ts < ?', (cutoff_90d,))
    await conn.commit()


# ---------------------------------------------------------------------------
# Read-side queries (read-only DbPool connections from route handlers)
# ---------------------------------------------------------------------------


_EMPTY_SERIES: dict[str, list] = {'labels': [], 'values': []}


def _coerce_series(rows: list[tuple[str, Any]]) -> dict[str, list]:
    return {
        'labels': [r[0] for r in rows],
        'values': [r[1] for r in rows],
    }


async def get_orchestrators_running_series(
    db: aiosqlite.Connection | None,
    *,
    days: int = 1,
    project_id: str | None = None,
) -> dict[str, list]:
    """Return total running-orchestrator count over time.

    When *project_id* is None, returns the sum across all projects per
    timestamp; otherwise filters to a single project.
    """
    if db is None:
        return dict(_EMPTY_SERIES)
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    try:
        if project_id is None:
            sql = (
                'SELECT ts, SUM(running_count) FROM orchestrator_snapshots '
                'WHERE ts >= ? GROUP BY ts ORDER BY ts'
            )
            params: tuple = (since,)
        else:
            sql = (
                'SELECT ts, running_count FROM orchestrator_snapshots '
                'WHERE project_id = ? AND ts >= ? ORDER BY ts'
            )
            params = (project_id, since)
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
    except Exception:
        logger.debug('orchestrator series query failed', exc_info=True)
        return dict(_EMPTY_SERIES)
    return _coerce_series([(r[0], r[1]) for r in rows])


async def get_memory_sparks(
    db: aiosqlite.Connection | None,
    *,
    days: int = 1,
) -> dict[str, dict[str, list]]:
    """Return system-wide Graphiti node + Mem0 memory sparks.

    Sums per-project rows per timestamp so the spark reflects total store
    size even when multiple projects share the dashboard.
    """
    empty = {'graphiti_nodes': dict(_EMPTY_SERIES), 'mem0_memories': dict(_EMPTY_SERIES)}
    if db is None:
        return empty
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    try:
        async with db.execute(
            'SELECT ts, SUM(graphiti_nodes), SUM(mem0_memories) '
            'FROM memory_snapshots WHERE ts >= ? GROUP BY ts ORDER BY ts',
            (since,),
        ) as cur:
            rows = await cur.fetchall()
    except Exception:
        logger.debug('memory sparks query failed', exc_info=True)
        return empty
    return {
        'graphiti_nodes': _coerce_series([(r[0], r[1]) for r in rows]),
        'mem0_memories': _coerce_series([(r[0], r[2]) for r in rows]),
    }


_24H_TOLERANCE_SECONDS = 2 * 3600  # accept rows within ±2h of the target


async def get_memory_24h_ago(
    db: aiosqlite.Connection | None,
) -> dict[str, dict]:
    """Return {project_id: {graphiti_nodes, mem0_memories}} from ~24h ago.

    Picks the row whose ``ts`` is closest to (now - 24h) per project,
    but only if it falls within ±2 hours of the target. Projects whose
    only available rows are far from the 24h mark are absent from the
    result so the UI renders ``—`` rather than a misleading zero delta.
    """
    if db is None:
        return {}
    target = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
    try:
        async with db.execute(
            """
            SELECT project_id, graphiti_nodes, mem0_memories, ts FROM (
                SELECT project_id, graphiti_nodes, mem0_memories, ts,
                       ABS(strftime('%s', ts) - strftime('%s', ?)) AS d,
                       ROW_NUMBER() OVER (
                           PARTITION BY project_id
                           ORDER BY ABS(strftime('%s', ts) - strftime('%s', ?)) ASC
                       ) AS rn
                FROM memory_snapshots
            ) WHERE rn = 1 AND d <= ?
            """,
            (target, target, _24H_TOLERANCE_SECONDS),
        ) as cur:
            rows = await cur.fetchall()
    except Exception:
        logger.debug('memory 24h-ago query failed', exc_info=True)
        return {}
    return {
        r[0]: {'graphiti_nodes': r[1], 'mem0_memories': r[2]}
        for r in rows
    }


async def get_queue_pending_series(
    db: aiosqlite.Connection | None,
    *,
    days: int = 1,
) -> dict[str, list]:
    """Return write-queue pending depth over time."""
    if db is None:
        return dict(_EMPTY_SERIES)
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    try:
        async with db.execute(
            'SELECT ts, pending FROM queue_snapshots WHERE ts >= ? ORDER BY ts',
            (since,),
        ) as cur:
            rows = await cur.fetchall()
    except Exception:
        logger.debug('queue series query failed', exc_info=True)
        return dict(_EMPTY_SERIES)
    return _coerce_series([(r[0], r[1]) for r in rows])


async def get_recon_sparks(
    db: aiosqlite.Connection | None,
    *,
    days: int = 1,
) -> dict[str, dict[str, list]]:
    """Return reconciliation buffered_count and active_agents over time."""
    empty = {'buffered_count': dict(_EMPTY_SERIES), 'active_agents': dict(_EMPTY_SERIES)}
    if db is None:
        return empty
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    try:
        async with db.execute(
            'SELECT ts, buffered_count, active_agents FROM recon_snapshots '
            'WHERE ts >= ? ORDER BY ts',
            (since,),
        ) as cur:
            rows = await cur.fetchall()
    except Exception:
        logger.debug('recon sparks query failed', exc_info=True)
        return empty
    return {
        'buffered_count': _coerce_series([(r[0], r[1]) for r in rows]),
        'active_agents': _coerce_series([(r[0], r[2]) for r in rows]),
    }


async def get_merge_active_series(
    db: aiosqlite.Connection | None,
    *,
    project_id: str | None = None,
    days: int = 1,
) -> dict[str, list]:
    """Return active merge-queue depth over time.

    When *project_id* is None, returns the sum across all projects per
    timestamp; otherwise filters to a single project.
    """
    if db is None:
        return dict(_EMPTY_SERIES)
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    try:
        if project_id is None:
            sql = (
                'SELECT ts, SUM(active_count) FROM merge_snapshots '
                'WHERE ts >= ? GROUP BY ts ORDER BY ts'
            )
            params: tuple = (since,)
        else:
            sql = (
                'SELECT ts, active_count FROM merge_snapshots '
                'WHERE project_id = ? AND ts >= ? ORDER BY ts'
            )
            params = (project_id, since)
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
    except Exception:
        logger.debug('merge series query failed', exc_info=True)
        return dict(_EMPTY_SERIES)
    return _coerce_series([(r[0], r[1]) for r in rows])
