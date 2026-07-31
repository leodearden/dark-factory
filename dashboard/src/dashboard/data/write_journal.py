"""Async SQLite queries for write journal metrics (memory operation graphs).

Queries the write_journal.db for time-series and breakdown data used by
the memory graphs section of the dashboard.

**Why no multi-DB aggregation here (task 841):**
``write_journal.db`` is written exclusively by the fused-memory server, which
is a single-host singleton process.  There is exactly *one* write-journal DB
per host.  The ``write_ops`` table already carries a ``project_id`` column, so
the DB is multi-project by construction — every project's memory writes land in
the same DB.  The existing queries aggregate across all projects intentionally;
per-project filtering is intentionally out of scope.  Task 841 evaluated
whether multi-DB aggregation applied here and concluded it is moot.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import aiosqlite

from dashboard.data.db import with_db
from dashboard.data.utils import resolve_now

logger = logging.getLogger(__name__)


async def get_memory_timeseries(
    db: aiosqlite.Connection | None, *, hours: int = 24, now: datetime | None = None,
) -> dict:
    """Hourly read/write counts for the last *hours* hours.

    Returns ``{labels: ["HH:00", ...], reads: [int, ...], writes: [int, ...]}``.
    Labels cover every hour in the window, with zeros for gaps.

    *now* defaults to the live clock via :func:`dashboard.data.utils.resolve_now`;
    pass an explicit value for deterministic results.
    """
    now = resolve_now(now)
    since = now - timedelta(hours=hours)

    # Pre-fill all hour buckets
    buckets: dict[str, dict[str, int]] = {}
    for i in range(hours):
        dt = since + timedelta(hours=i + 1)
        key = dt.strftime('%Y-%m-%dT%H:00')
        buckets[key] = {'read': 0, 'write': 0}

    since_iso = since.isoformat()

    async def _query(db: aiosqlite.Connection) -> list:
        async with db.execute(
            "SELECT strftime('%Y-%m-%dT%H:00', created_at) AS hour,"
            ' kind, COUNT(*) AS cnt'
            ' FROM write_ops WHERE created_at >= ?'
            ' GROUP BY hour, kind',
            (since_iso,),
        ) as cursor:
            return list(await cursor.fetchall())

    rows = await with_db(db, _query, [])
    for hour, kind, cnt in rows:
        if hour in buckets and kind in ('read', 'write'):
            buckets[hour][kind] = cnt

    sorted_keys = sorted(buckets)
    return {
        'labels': [k[11:16] for k in sorted_keys],  # "HH:MM"
        'reads': [buckets[k]['read'] for k in sorted_keys],
        'writes': [buckets[k]['write'] for k in sorted_keys],
    }


async def get_operations_breakdown(
    db: aiosqlite.Connection | None, *, hours: int = 24, now: datetime | None = None,
) -> dict:
    """Operation type distribution for the last *hours* hours.

    Returns ``{labels: [str, ...], values: [int, ...]}``.

    *now* defaults to the live clock via :func:`dashboard.data.utils.resolve_now`;
    pass an explicit value for deterministic results.

    Performance — this query does NOT benefit from ``idx_wo_created`` (task 3304).
    With that index present it still plans as
    ``SCAN write_ops USING INDEX idx_wo_operation``: SQLite prefers walking
    idx_wo_operation to satisfy ``GROUP BY operation`` in order (avoiding a temp
    B-tree) over range-seeking the date index, and pays a per-row created_at test
    across all 16.6M rows instead. That is planner behaviour, not a tuning miss —
    unchanged at 0 rows, at 5 000 rows, after ANALYZE, and with a covering
    (created_at, operation) variant. Measured 2026-07-31 on a copy of the live
    7 GB journal, warm: 15.958 s before the index, 14.132-20.800 s after, versus
    31.644 s -> 0.399 s for :func:`get_agent_breakdown` and 2.570 s -> 1.019 s for
    :func:`get_memory_timeseries`, which both flip to
    ``SEARCH ... idx_wo_created (created_at>?)``.

    ``api_memory_graphs`` (dashboard/src/dashboard/app.py:587-598) serves exactly
    this query plus :func:`get_memory_timeseries`, so the endpoint's residual
    latency is now dominated by this one. Its remedy — query reshape, cache,
    materialised rollup, or dropping it from that endpoint — is explicitly NOT
    task 3304 and needs its own PRD row; see the "Note on α's corrected signal"
    amendment in plans/dashboard-availability-prd.md. Do not add indexes here
    chasing it.
    """
    since = (resolve_now(now) - timedelta(hours=hours)).isoformat()

    async def _query(db: aiosqlite.Connection) -> dict:
        async with db.execute(
            'SELECT operation, COUNT(*) AS cnt FROM write_ops'
            ' WHERE created_at >= ? GROUP BY operation ORDER BY cnt DESC',
            (since,),
        ) as cursor:
            rows = await cursor.fetchall()
        return {
            'labels': [r[0] or 'unknown' for r in rows],
            'values': [r[1] for r in rows],
        }

    return await with_db(db, _query, {'labels': [], 'values': []})


async def get_agent_breakdown(
    db: aiosqlite.Connection | None, *, hours: int = 24, now: datetime | None = None,
) -> dict:
    """Agent distribution for the last *hours* hours.

    Returns ``{labels: [str, ...], values: [int, ...]}``.

    *now* defaults to the live clock via :func:`dashboard.data.utils.resolve_now`;
    pass an explicit value for deterministic results.
    """
    since = (resolve_now(now) - timedelta(hours=hours)).isoformat()

    async def _query(db: aiosqlite.Connection) -> dict:
        async with db.execute(
            "SELECT COALESCE(agent_id, 'unknown') AS agent, COUNT(*) AS cnt"
            ' FROM write_ops WHERE created_at >= ?'
            ' GROUP BY agent ORDER BY cnt DESC',
            (since,),
        ) as cursor:
            rows = await cursor.fetchall()
        return {
            'labels': [r[0] for r in rows],
            'values': [r[1] for r in rows],
        }

    return await with_db(db, _query, {'labels': [], 'values': []})
