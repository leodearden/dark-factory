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
import sqlite3
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

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
# if it is ever absent. `get_memory_ops` catches exactly that error and
# retries against `MEMORY_OPS_SQL_UNHINTED`, so the observable of a missing
# index is a correct-but-slow chart plus an ERROR log naming
# `idx_wo_created` (pinned by TestMemoryOpsMissingIndexFallback) — never a
# 500 and never a silently empty chart. The index cannot go missing in normal
# operation: fused-memory's initialize() re-runs the `IF NOT EXISTS` DDL on
# every start, and
# fused-memory/tests/test_write_journal.py::test_schema_creates_idx_wo_created
# already fails CI on a rename or a drop there. The residual risk this
# fallback guards against is narrower than "nothing else catches it": a
# schema edit that lands without that suite running, or a journal DB produced
# by an older fused-memory build.
MEMORY_OPS_SQL = (
    "SELECT strftime('%Y-%m-%dT%H:00', created_at) AS hour, kind, operation, COUNT(*)"
    ' FROM write_ops INDEXED BY idx_wo_created'
    ' WHERE created_at >= ? AND created_at < ?'
    ' GROUP BY hour, kind, operation'
)

# Fallback used by `get_memory_ops` when idx_wo_created is unexpectedly
# absent (see the coupling comment above). Derived from MEMORY_OPS_SQL by
# construction, rather than retyped, so the two strings cannot diverge.
MEMORY_OPS_SQL_UNHINTED = MEMORY_OPS_SQL.replace(' INDEXED BY idx_wo_created', '')

_HOUR_KEY_FORMAT = '%Y-%m-%dT%H:00'


@dataclass(frozen=True, slots=True)
class MemoryOps:
    """One window's memory operations, counted once and offered two views.

    ``reads``, ``writes`` and ``other`` (every kind outside read/write) are
    hourly counts aligned to ``labels`` ('HH:MM', oldest first).
    ``by_operation`` is ``(operation, count)`` pairs, count descending then
    operation ascending. Both views come from the same rows, so the series
    sum to exactly the by_operation total.
    """

    labels: tuple[str, ...]
    reads: tuple[int, ...]
    writes: tuple[int, ...]
    other: tuple[int, ...]
    by_operation: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        hours = len(self.labels)
        if not len(self.reads) == len(self.writes) == len(self.other) == hours:
            raise ValueError(f'MemoryOps series must each hold {hours} hourly buckets')
        series_total = sum(self.reads) + sum(self.writes) + sum(self.other)
        operations_total = sum(count for _, count in self.by_operation)
        if series_total != operations_total:
            raise ValueError(
                f'MemoryOps views disagree: series count {series_total} rows, '
                f'by_operation counts {operations_total}'
            )


def _hour_keys(newest_hour: datetime, hours: int) -> tuple[str, ...]:
    """The *hours* hour keys ending with *newest_hour*, oldest first."""
    oldest_hour = newest_hour - timedelta(hours=hours - 1)
    return tuple(
        (oldest_hour + timedelta(hours=i)).strftime(_HOUR_KEY_FORMAT) for i in range(hours)
    )


def _newest_hour(now: datetime | None) -> datetime:
    return resolve_now(now).replace(minute=0, second=0, microsecond=0)


def empty_memory_ops(*, hours: int = 24, now: datetime | None = None) -> MemoryOps:
    """The window :func:`get_memory_ops` reads, every bucket zero and no operations.

    What :func:`get_memory_ops` returns when the journal cannot be read.
    """
    return _reduce_memory_ops(_hour_keys(_newest_hour(now), hours), ())


def _reduce_memory_ops(
    hour_keys: Sequence[str], rows: Iterable[Sequence[Any]],
) -> MemoryOps:
    """Fold ``(hour, kind, operation, count)`` rows into both views at once.

    A row whose hour is not one of *hour_keys* is skipped from BOTH views, so
    they reconcile by construction.
    """
    position = {key: i for i, key in enumerate(hour_keys)}
    series: dict[str, list[int]] = {
        'read': [0] * len(hour_keys),
        'write': [0] * len(hour_keys),
        'other': [0] * len(hour_keys),
    }
    by_operation: Counter[str] = Counter()
    for hour, kind, operation, count in rows:
        bucket = position.get(hour)
        if bucket is None:
            continue
        series[kind if kind in ('read', 'write') else 'other'][bucket] += count
        by_operation[operation or 'unknown'] += count
    return MemoryOps(
        labels=tuple(key[11:16] for key in hour_keys),
        reads=tuple(series['read']),
        writes=tuple(series['write']),
        other=tuple(series['other']),
        by_operation=tuple(
            sorted(by_operation.items(), key=lambda item: (-item[1], item[0]))
        ),
    )


async def get_memory_ops(
    db: aiosqlite.Connection | None, *, hours: int = 24, now: datetime | None = None,
) -> MemoryOps:
    """Memory operations over the *hours* whole hours ending with now's hour.

    The window is hour-ALIGNED and bounded on both sides:
    ``[floor(now) - (hours-1)h, floor(now) + 1h)``, exactly the hour buckets
    the chart labels, so no row can count in one view and not the other — a
    row before the oldest bucket or dated in the future is in neither. (The
    two queries this replaced disagreed: the timeseries dropped the window's
    partial oldest hour and every kind outside read/write, while the
    breakdown counted both.) On a missing DB or a query error it returns
    :func:`empty_memory_ops` for the same window.

    *now* defaults to the live clock via :func:`dashboard.data.utils.resolve_now`;
    pass an explicit value for deterministic results.

    Performance — one ``GROUP BY hour, kind, operation`` over a
    ``SEARCH ... USING INDEX idx_wo_created`` range seek. Measured 2026-10-03
    read-only on the live 17 GB journal (935 554 rows in 24h): 3.93 s for 293
    groups, against 4.11 s + 0.92 s for the two queries it replaced, which
    aiosqlite serialised on one connection. The ``INDEXED BY`` hint (see
    :data:`MEMORY_OPS_SQL`) is inherited from the old operations breakdown,
    whose unhinted form chose ``SCAN write_ops USING INDEX idx_wo_operation``
    to satisfy ``GROUP BY operation`` in order — stable across ``ANALYZE``
    and row counts — and paid a per-row ``created_at`` test across every row:
    22.01 s unhinted vs 0.64 s hinted, measured 2026-08-02 (task 3519).

    Do NOT "fix" a slow plan by restructuring into a ``created_at`` subquery
    instead of the hint: SQLite flattens an un-hinted subquery pre-filter
    straight back to the same ``SCAN`` (measured 18.12 s) — the hint is
    load-bearing; a reshape alone does nothing.

    The hint is tuned for SHORT windows — it is measured only against the
    24h default. ``INDEXED BY`` is unconditional, so a much wider ``hours``
    forces the same range seek over a large fraction of the index, and the
    planner is no longer free to choose otherwise. No caller passes a
    non-default ``hours`` today.

    Full variant table for the 2026-08-02 measurements:
    the "Follow-on: α's residual is now owned (added 2026-08-02)" section of
    plans/dashboard-availability-prd.md.
    """
    newest_hour = _newest_hour(now)
    hour_keys = _hour_keys(newest_hour, hours)
    window = (
        (newest_hour - timedelta(hours=hours - 1)).isoformat(),
        (newest_hour + timedelta(hours=1)).isoformat(),
    )

    async def _query(db: aiosqlite.Connection) -> MemoryOps:
        try:
            async with db.execute(MEMORY_OPS_SQL, window) as cursor:
                rows = await cursor.fetchall()
        except sqlite3.OperationalError as exc:
            if 'no such index' not in str(exc):
                raise
            # idx_wo_created is missing — see the coupling comment at
            # MEMORY_OPS_SQL. Fall back to the unhinted query so the chart
            # stays correct (just slow), and log loudly enough to name the
            # index without needing the traceback.
            logger.error(
                'write_ops index idx_wo_created missing — memory ops '
                'falling back to unhinted scan',
                exc_info=True,
            )
            async with db.execute(MEMORY_OPS_SQL_UNHINTED, window) as cursor:
                rows = await cursor.fetchall()
        return _reduce_memory_ops(hour_keys, rows)

    return await with_db(db, _query, _reduce_memory_ops(hour_keys, ()))


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
    :func:`get_memory_ops` — so the oversubscription argument that justifies
    the hint on that query does not apply here; (2) once ``sqlite_stat1`` exists, the planner
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
