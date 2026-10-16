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

from dashboard.data.datum import Datum, DatumState, unknown_datum
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
_RAW_OPS_ARM = (
    "SELECT strftime('%Y-%m-%dT%H:00', created_at) AS hour, kind, operation, COUNT(*)"
    ' FROM write_ops INDEXED BY idx_wo_created'
    ' WHERE created_at >= :start AND created_at < :end'
    ' GROUP BY hour, kind, operation'
)

# Second schema dependency (task 5405): read_op_counts, owned by
# fused-memory/src/fused_memory/services/read_rollup.py::READ_OP_COUNTS_SCHEMA_SQL.
# A journal without it falls back to RAW_ONLY_MEMORY_OPS_SQL (see
# `_memory_ops_rows`).
_COUNTED_READS_ARM = (
    "SELECT strftime('%Y-%m-%dT%H:00', hour_start) AS hour, 'read' AS kind, operation,"
    ' SUM(count)'
    ' FROM read_op_counts'
    ' WHERE hour_start >= :start AND hour_start < :end'
    ' GROUP BY hour, operation'
)

# ONE statement, so ONE read snapshot: a migration chunk committing between
# two separate statements would count its rows twice or not at all.
MEMORY_OPS_SQL = _RAW_OPS_ARM + ' UNION ALL ' + _COUNTED_READS_ARM

# Fallback used by `get_memory_ops` when idx_wo_created is unexpectedly
# absent (see the coupling comment above). Derived from MEMORY_OPS_SQL by
# construction, rather than retyped, so the two strings cannot diverge.
MEMORY_OPS_SQL_UNHINTED = MEMORY_OPS_SQL.replace(' INDEXED BY idx_wo_created', '')

# Fallback for a journal without read_op_counts: a fused-memory build that
# predates it journals every read per row, so the raw arm alone is exact.
RAW_ONLY_MEMORY_OPS_SQL = _RAW_OPS_ARM

_MISSING_ROLLUP_TABLE_ERROR = 'no such table: read_op_counts'

_HOUR_KEY_FORMAT = '%Y-%m-%dT%H:00'

# A healthy read of the live journal takes ~4s (measured 2026-10-03, see
# get_memory_ops), so only a wedged read ages a reading past this.
MEMORY_OPS_FRESHNESS_BOUND_SECONDS = 60


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
) -> Datum[MemoryOps]:
    """Memory operations over the *hours* whole hours ending with now's hour.

    Every read is journalled exactly once (task 5405): as a ``write_ops`` row
    (search, failed reads, not-yet-migrated history) or as a ``read_op_counts``
    count, so the two sources ADD inside one statement. The rollup is defined
    in fused-memory/src/fused_memory/services/read_rollup.py.

    The window is hour-ALIGNED and bounded on both sides:
    ``[floor(now) - (hours-1)h, floor(now) + 1h)``, exactly the hour buckets
    the chart labels, so no row can count in one view and not the other — a
    row before the oldest bucket or dated in the future is in neither. (The
    two queries this replaced disagreed: the timeseries dropped the window's
    partial oldest hour and every kind outside read/write, while the
    breakdown counted both.)

    A read is a ``fresh`` Datum measured at *now*. A missing journal or a
    failed query is an ``unknown`` Datum whose reason names the cause, never
    a zero window: that would read as a quiet day. A readable journal with no
    rows in the window IS a quiet day, and is served as one.

    *now* defaults to the live clock via :func:`dashboard.data.utils.resolve_now`
    and is read once; pass an explicit value for deterministic results.

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
    measured_at = resolve_now(now)
    newest_hour = measured_at.replace(minute=0, second=0, microsecond=0)
    hour_keys = _hour_keys(newest_hour, hours)
    window = {
        'start': (newest_hour - timedelta(hours=hours - 1)).isoformat(),
        'end': (newest_hour + timedelta(hours=1)).isoformat(),
    }
    if db is None:
        return unknown_datum(
            'the write journal is missing or could not be opened',
            MEMORY_OPS_FRESHNESS_BOUND_SECONDS,
        )
    try:
        rows = await _memory_ops_rows(db, window)
    except (sqlite3.OperationalError, OSError) as exc:
        logger.warning('memory ops: write journal query failed', exc_info=True)
        return unknown_datum(
            f'the write journal query failed: {type(exc).__name__}: {exc}',
            MEMORY_OPS_FRESHNESS_BOUND_SECONDS,
        )
    return Datum(
        _reduce_memory_ops(hour_keys, rows),
        measured_at,
        DatumState.FRESH,
        None,
        MEMORY_OPS_FRESHNESS_BOUND_SECONDS,
    )


async def _memory_ops_rows(
    db: aiosqlite.Connection, window: dict[str, str],
) -> Iterable[Sequence[Any]]:
    """The ``(hour, kind, operation, count)`` rows of *window*, hinted when it can be."""
    try:
        async with db.execute(MEMORY_OPS_SQL, window) as cursor:
            return await cursor.fetchall()
    except sqlite3.OperationalError as exc:
        fallback_sql = _memory_ops_fallback(exc)
        async with db.execute(fallback_sql, window) as cursor:
            return await cursor.fetchall()


def _memory_ops_fallback(exc: sqlite3.OperationalError) -> str:
    """The query to retry after *exc*, logged; re-raises an unexpected error."""
    if 'no such index' in str(exc):
        # idx_wo_created is missing — see the coupling comment at
        # MEMORY_OPS_SQL. Fall back to the unhinted query so the chart
        # stays correct (just slow), and log loudly enough to name the
        # index without needing the traceback.
        logger.error(
            'write_ops index idx_wo_created missing — memory ops '
            'falling back to unhinted scan',
            exc_info=True,
        )
        return MEMORY_OPS_SQL_UNHINTED
    if str(exc) == _MISSING_ROLLUP_TABLE_ERROR:
        # An expected deploy transient: the dashboard restarts on merge,
        # fused-memory up to ~8 h later via its staleness backstop.
        logger.info(
            'write journal has no read_op_counts yet (older fused-memory build) '
            '— memory ops reading raw write_ops rows only, which is exact there'
        )
        return RAW_ONLY_MEMORY_OPS_SQL
    raise exc
