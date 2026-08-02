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

# Cross-package coupling (task 3519): this query HARD-depends on
# idx_wo_created, owned by fused-memory's SCHEMA_SQL constant in
# fused_memory/services/write_journal.py (cited by file + constant name, not
# line number — a numbered citation across these two packages has gone stale
# before). `INDEXED BY` makes the index a hard constraint: SQLite raises
# `no such index: idx_wo_created` rather than silently falling back to a scan
# if it is ever absent. `with_db` (dashboard/data/db.py) catches that
# OperationalError, so the actual observable of a missing index is a
# silently EMPTY operations chart plus a WARNING log — not an error surfaced
# to the user (pinned by TestOperationsBreakdownMissingIndexDegradation).
# The index cannot go missing in normal operation: fused-memory's
# initialize() re-runs the `IF NOT EXISTS` DDL on every start. The residual
# risk this comment guards against is a future SCHEMA EDIT on the
# fused-memory side (a rename or drop), which nothing else would catch
# before this endpoint silently went blank.
OPS_BREAKDOWN_SQL = (
    'SELECT operation, COUNT(*) AS cnt FROM write_ops INDEXED BY idx_wo_created'
    ' WHERE created_at >= ? GROUP BY operation ORDER BY cnt DESC'
)


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

    Performance — unhinted, the planner chooses ``SCAN write_ops USING INDEX
    idx_wo_operation`` to satisfy ``GROUP BY operation`` in order and avoid a
    temp B-tree, paying a per-row ``created_at`` test across every row instead
    of range-seeking the date index. That choice is stable across ``ANALYZE``,
    row counts, and a covering ``(created_at, operation)`` variant (task
    3304's finding — still true, unhinted). An explicit ``INDEXED BY`` (see
    :data:`OPS_BREAKDOWN_SQL`) overrides it: measured 2026-08-02 on the live
    DB, 22.01 s -> 0.64 s, 34x, both returning the same 19 groups (task 3519).
    ``api_memory_graphs`` in :mod:`dashboard.app` serves exactly this query
    plus :func:`get_memory_timeseries`, so this was the endpoint's dominant
    residual latency.

    Do NOT "fix" this by restructuring into a ``created_at`` subquery instead
    of the hint: SQLite flattens an un-hinted subquery pre-filter straight
    back to the same ``SCAN`` (measured 18.12 s) — the hint is load-bearing;
    a reshape alone does nothing.

    Full variant table and measurements are recorded, authoritatively, in the
    "Follow-on: α's residual is now owned (added 2026-08-02)" section of
    plans/dashboard-availability-prd.md.
    """
    since = (resolve_now(now) - timedelta(hours=hours)).isoformat()

    async def _query(db: aiosqlite.Connection) -> dict:
        async with db.execute(OPS_BREAKDOWN_SQL, (since,)) as cursor:
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

    Deliberately NOT hinted with ``INDEXED BY idx_wo_created`` (task 3519 —
    decided and recorded, not left to chance): (1) zero production callers
    today — ``api_memory_graphs`` in :mod:`dashboard.app` calls only
    :func:`get_memory_timeseries` and :func:`get_operations_breakdown` — so
    the oversubscription argument that justifies the hint on the breakdown
    query does not apply here; (2) once ``sqlite_stat1`` exists, the planner
    already reaches the same ``created_at`` range seek via a strictly better
    COVERING skip-scan on ``idx_wo_kind_time`` (``SEARCH ... COVERING INDEX
    idx_wo_kind_time (ANY(kind) AND created_at>?)``), which pinning
    ``idx_wo_created`` here would forbid; and (3) fused-memory's
    ``test_created_at_range_is_seekable_for_dashboard_queries`` already pins
    that seekability property at the schema owner. Hinting here would only
    add cross-package coupling cost for no benefit.
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
