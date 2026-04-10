"""Async queries for merge queue operational metrics.

Reads from data/orchestrator/runs.db (events table written by MergeWorker via
EventStore) to produce merge queue statistics.

Note on queue_depth_timeseries approximation
--------------------------------------------
The events table only records completions: ``EventStore.emit()`` is called
synchronously *after* an attempt finishes.  We therefore approximate
"queue depth" as the count of ``merge_attempt`` events per 15-minute bucket
(throughput proxy), *not* true in-flight queue depth.  The MergeWorker's
in-flight queue state is in-memory and not persisted to the events table.

When multiple project roots are configured, the ``aggregate_*`` functions
query each project's runs.db in parallel and merge the results.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import aiosqlite

from dashboard.data.chart_utils import ChartData
from dashboard.data.db import with_db
from dashboard.data.performance import _percentile

logger = logging.getLogger(__name__)

_CANONICAL_OUTCOMES = ['done', 'conflict', 'blocked', 'already_merged']


def _cutoff_iso(hours: int) -> str:
    """Return ISO-format cutoff datetime for the given look-back window (hours)."""
    return (datetime.now(UTC) - timedelta(hours=hours)).isoformat()


# ---------------------------------------------------------------------------
# 1. Queue depth timeseries (15-min bins)
# ---------------------------------------------------------------------------

async def queue_depth_timeseries(
    db: aiosqlite.Connection | None,
    *,
    hours: int = 24,
) -> ChartData:
    """Approximate merge queue throughput as 15-min bucket counts.

    Returns ChartData with ISO bucket-start labels and integer counts.
    For a 24h window, this produces exactly ``hours * 4 = 96`` buckets.
    Buckets are aligned to 15-min boundaries starting from
    ``floor(now - hours, 15min)``.
    """
    if db is None:
        return {'labels': [], 'values': []}

    async def _query(conn: aiosqlite.Connection) -> ChartData:
        now = datetime.now(UTC)
        cutoff = now - timedelta(hours=hours)

        # Align cutoff to 15-min boundary (floor)
        cutoff_aligned = cutoff.replace(
            minute=(cutoff.minute // 15) * 15,
            second=0,
            microsecond=0,
        )

        # Generate exactly hours*4 buckets
        buckets = [
            cutoff_aligned + timedelta(minutes=15 * i)
            for i in range(hours * 4)
        ]

        # Fetch all merge_attempt events in the window
        rows = await conn.execute_fetchall(
            "SELECT timestamp FROM events "
            "WHERE event_type = 'merge_attempt' AND timestamp >= ?",
            (cutoff.isoformat(),),
        )

        # Build count map keyed by ISO bucket label
        counts: dict[str, int] = {b.isoformat(): 0 for b in buckets}
        for row in rows:
            ts_str = row['timestamp']
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                # Floor to 15-min bucket
                bucket = ts.replace(
                    minute=(ts.minute // 15) * 15,
                    second=0,
                    microsecond=0,
                )
                key = bucket.isoformat()
                if key in counts:
                    counts[key] += 1
            except (ValueError, TypeError):
                continue

        labels = [b.isoformat() for b in buckets]
        values = [counts[lbl] for lbl in labels]
        return {'labels': labels, 'values': values}

    return await with_db(db, _query, {'labels': [], 'values': []})


# ---------------------------------------------------------------------------
# 2. Outcome distribution
# ---------------------------------------------------------------------------

async def outcome_distribution(
    db: aiosqlite.Connection | None,
    *,
    hours: int = 24,
) -> ChartData:
    """Count merge_attempt events by outcome within the window.

    Returns ChartData with canonical outcomes first (done, conflict, blocked,
    already_merged), then any unknown outcomes sorted alphabetically.
    Missing canonical outcomes are omitted (count=0 entries are dropped).
    """
    if db is None:
        return {'labels': [], 'values': []}

    async def _query(conn: aiosqlite.Connection) -> ChartData:
        since = _cutoff_iso(hours)
        rows = await conn.execute_fetchall(
            "SELECT json_extract(data, '$.outcome') AS outcome, COUNT(*) AS cnt "
            "FROM events "
            "WHERE event_type = 'merge_attempt' AND timestamp >= ? "
            "GROUP BY outcome",
            (since,),
        )

        counts: dict[str, int] = {}
        for row in rows:
            outcome = row['outcome'] or 'unknown'
            counts[outcome] = row['cnt']

        if not counts:
            return {'labels': [], 'values': []}

        # Canonical outcomes first, then unknowns alphabetically
        labels: list[str] = []
        values: list[int] = []
        for outcome in _CANONICAL_OUTCOMES:
            if outcome in counts:
                labels.append(outcome)
                values.append(counts[outcome])

        unknowns = sorted(k for k in counts if k not in _CANONICAL_OUTCOMES)
        for outcome in unknowns:
            labels.append(outcome)
            values.append(counts[outcome])

        return {'labels': labels, 'values': values}

    return await with_db(db, _query, {'labels': [], 'values': []})


# ---------------------------------------------------------------------------
# 3. Latency stats
# ---------------------------------------------------------------------------

async def _get_durations(
    db: aiosqlite.Connection | None,
    *,
    hours: int = 24,
) -> list[float]:
    """Return sorted list of non-null merge_attempt duration_ms values."""
    if db is None:
        return []

    async def _query(conn: aiosqlite.Connection) -> list[float]:
        since = _cutoff_iso(hours)
        rows = await conn.execute_fetchall(
            "SELECT duration_ms FROM events "
            "WHERE event_type = 'merge_attempt' "
            "  AND timestamp >= ? "
            "  AND duration_ms IS NOT NULL "
            "  AND duration_ms > 0 "
            "ORDER BY duration_ms",
            (since,),
        )
        return [float(row['duration_ms']) for row in rows]

    return await with_db(db, _query, [])


def _compute_latency_stats(durations: list[float]) -> dict:
    """Compute latency stats dict from a sorted list of durations."""
    if not durations:
        return {'p50': 0, 'p95': 0, 'p99': 0, 'count': 0, 'mean_ms': 0.0}
    return {
        'p50': round(_percentile(durations, 50)),
        'p95': round(_percentile(durations, 95)),
        'p99': round(_percentile(durations, 99)),
        'count': len(durations),
        'mean_ms': sum(durations) / len(durations),
    }


async def latency_stats(
    db: aiosqlite.Connection | None,
    *,
    hours: int = 24,
) -> dict:
    """P50/P95/P99 latency and count for merge_attempt events.

    Returns {'p50': int, 'p95': int, 'p99': int, 'count': int,
             'mean_ms': float}.
    When no rows have non-null duration_ms, returns all zeros with count=0.
    """
    durations = await _get_durations(db, hours=hours)
    return _compute_latency_stats(sorted(durations))


# ---------------------------------------------------------------------------
# 4. Recent merges
# ---------------------------------------------------------------------------

async def recent_merges(
    db: aiosqlite.Connection | None,
    *,
    limit: int = 20,
) -> list[dict]:
    """Most recent merge_attempt events, newest first.

    Returns list of {'task_id', 'run_id', 'outcome', 'duration_ms',
                     'timestamp'} dicts.
    """
    if db is None:
        return []

    async def _query(conn: aiosqlite.Connection) -> list[dict]:
        rows = await conn.execute_fetchall(
            "SELECT task_id, run_id, "
            "       json_extract(data, '$.outcome') AS outcome, "
            "       duration_ms, timestamp "
            "FROM events "
            "WHERE event_type = 'merge_attempt' "
            "ORDER BY timestamp DESC "
            "LIMIT ?",
            (limit,),
        )
        return [
            {
                'task_id': row['task_id'],
                'run_id': row['run_id'],
                'outcome': row['outcome'],
                'duration_ms': row['duration_ms'],
                'timestamp': row['timestamp'],
            }
            for row in rows
        ]

    return await with_db(db, _query, [])


# ---------------------------------------------------------------------------
# 5. Speculative stats
# ---------------------------------------------------------------------------

async def speculative_stats(
    db: aiosqlite.Connection | None,
    *,
    hours: int = 24,
) -> dict:
    """Hit/discard counts and hit rate for speculative merge events.

    Returns {'hit_count': int, 'discard_count': int, 'total': int,
             'hit_rate': float}.
    """
    if db is None:
        return {'hit_count': 0, 'discard_count': 0, 'total': 0, 'hit_rate': 0.0}

    async def _query(conn: aiosqlite.Connection) -> dict:
        since = _cutoff_iso(hours)
        rows = await conn.execute_fetchall(
            "SELECT event_type, COUNT(*) AS cnt "
            "FROM events "
            "WHERE event_type IN ('speculative_merge', 'speculative_discard') "
            "  AND timestamp >= ? "
            "GROUP BY event_type",
            (since,),
        )
        hit_count = 0
        discard_count = 0
        for row in rows:
            if row['event_type'] == 'speculative_merge':
                hit_count = row['cnt']
            else:
                discard_count = row['cnt']
        total = hit_count + discard_count
        hit_rate = hit_count / total if total > 0 else 0.0
        return {
            'hit_count': hit_count,
            'discard_count': discard_count,
            'total': total,
            'hit_rate': hit_rate,
        }

    return await with_db(db, _query, {'hit_count': 0, 'discard_count': 0, 'total': 0, 'hit_rate': 0.0})


# ---------------------------------------------------------------------------
# 6. Multi-DB aggregation
# ---------------------------------------------------------------------------

async def aggregate_queue_depth_timeseries(
    dbs: list[aiosqlite.Connection | None],
    *,
    hours: int = 24,
) -> ChartData:
    """Aggregate queue depth timeseries across multiple project DBs.

    Counts per bucket are summed across all DBs.
    """
    results = await asyncio.gather(
        *[queue_depth_timeseries(db, hours=hours) for db in dbs],
        return_exceptions=True,
    )

    # Collect all valid results
    valid: list[ChartData] = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning('aggregate_queue_depth_timeseries: error from one DB: %s', r)
            continue
        valid.append(r)

    if not valid:
        return {'labels': [], 'values': []}

    # All results share the same labels (same hours window, same alignment)
    # Use the first non-empty label set
    labels: list[str] = []
    for v in valid:
        if v['labels']:
            labels = v['labels']
            break

    if not labels:
        return {'labels': [], 'values': []}

    # Sum counts per label
    total_counts: dict[str, int] = {lbl: 0 for lbl in labels}
    for v in valid:
        for lbl, cnt in zip(v['labels'], v['values']):
            if lbl in total_counts:
                total_counts[lbl] += cnt

    return {'labels': labels, 'values': [total_counts[lbl] for lbl in labels]}


async def aggregate_outcome_distribution(
    dbs: list[aiosqlite.Connection | None],
    *,
    hours: int = 24,
) -> ChartData:
    """Aggregate outcome distribution across multiple project DBs.

    Counts per outcome are summed; canonical ordering is preserved.
    """
    results = await asyncio.gather(
        *[outcome_distribution(db, hours=hours) for db in dbs],
        return_exceptions=True,
    )

    merged: dict[str, int] = {}
    for r in results:
        if isinstance(r, Exception):
            logger.warning('aggregate_outcome_distribution: error from one DB: %s', r)
            continue
        for lbl, cnt in zip(r['labels'], r['values']):
            merged[lbl] = merged.get(lbl, 0) + cnt

    if not merged:
        return {'labels': [], 'values': []}

    # Re-apply canonical ordering
    labels: list[str] = []
    values: list[int] = []
    for outcome in _CANONICAL_OUTCOMES:
        if outcome in merged:
            labels.append(outcome)
            values.append(merged[outcome])
    for outcome in sorted(k for k in merged if k not in _CANONICAL_OUTCOMES):
        labels.append(outcome)
        values.append(merged[outcome])

    return {'labels': labels, 'values': values}


async def aggregate_latency_stats(
    dbs: list[aiosqlite.Connection | None],
    *,
    hours: int = 24,
) -> dict:
    """Aggregate latency stats across multiple project DBs.

    Recomputes percentiles from the merged raw duration list.
    """
    all_durations: list[float] = []
    gather_results = await asyncio.gather(
        *[_get_durations(db, hours=hours) for db in dbs],
        return_exceptions=True,
    )
    for r in gather_results:
        if isinstance(r, Exception):
            logger.warning('aggregate_latency_stats: error from one DB: %s', r)
            continue
        all_durations.extend(r)

    return _compute_latency_stats(sorted(all_durations))


async def aggregate_recent_merges(
    dbs: list[aiosqlite.Connection | None],
    *,
    limit: int = 20,
) -> list[dict]:
    """Aggregate recent merges across multiple project DBs.

    Concatenates, re-sorts by timestamp DESC, and truncates to limit.
    """
    gather_results = await asyncio.gather(
        *[recent_merges(db, limit=limit) for db in dbs],
        return_exceptions=True,
    )
    merged: list[dict] = []
    for r in gather_results:
        if isinstance(r, Exception):
            logger.warning('aggregate_recent_merges: error from one DB: %s', r)
            continue
        merged.extend(r)

    merged.sort(key=lambda x: x.get('timestamp') or '', reverse=True)
    return merged[:limit]


async def aggregate_speculative_stats(
    dbs: list[aiosqlite.Connection | None],
    *,
    hours: int = 24,
) -> dict:
    """Aggregate speculative stats across multiple project DBs.

    Sums hit/discard counts and recomputes hit_rate.
    """
    gather_results = await asyncio.gather(
        *[speculative_stats(db, hours=hours) for db in dbs],
        return_exceptions=True,
    )
    hit_count = 0
    discard_count = 0
    for r in gather_results:
        if isinstance(r, Exception):
            logger.warning('aggregate_speculative_stats: error from one DB: %s', r)
            continue
        hit_count += r['hit_count']
        discard_count += r['discard_count']

    total = hit_count + discard_count
    hit_rate = hit_count / total if total > 0 else 0.0
    return {
        'hit_count': hit_count,
        'discard_count': discard_count,
        'total': total,
        'hit_rate': hit_rate,
    }
