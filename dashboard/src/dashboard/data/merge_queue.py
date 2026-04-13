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
import math
from datetime import UTC, datetime, timedelta

import aiosqlite

from dashboard.data.chart_utils import ChartData
from dashboard.data.db import with_db
from dashboard.data.stats_utils import percentile
from dashboard.data.utils import parse_utc

logger = logging.getLogger(__name__)


def _ts_sort_key(entry: dict) -> datetime:
    """Return a UTC-aware datetime sort key for a merge entry dict.

    Parses ``entry['timestamp']`` via :func:`parse_utc`.  Returns
    ``datetime.min`` (UTC-aware) on missing, None, or unparseable values so
    that malformed entries sort to the end of a descending sort.
    """
    try:
        return parse_utc(entry.get('timestamp'))
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=UTC)

_CANONICAL_OUTCOMES = ['done', 'conflict', 'blocked', 'already_merged']

# ---------------------------------------------------------------------------
# Adaptive bucket ladder: (max_hours | None, bucket_minutes)
# None as max_hours means "catch-all / no upper bound".
# ---------------------------------------------------------------------------
BUCKET_LADDER: tuple[tuple[int | None, int], ...] = (
    (24, 15),      # ≤ 24 h  → 15-min buckets  (≤  97 pts)
    (168, 60),     # ≤  7 d  → 60-min buckets  (≤ 169 pts)
    (720, 360),    # ≤ 30 d  →  6-h  buckets   (≤ 121 pts)
    (None, 1440),  # > 30 d  →  1-d  buckets   (≤ 3 651 pts, covers all=87 600 h)
)


def _cutoff_iso(hours: int, *, now: datetime | None = None) -> str:
    """Return ISO-format cutoff datetime for the given look-back window (hours).

    The returned string has a ``+00:00`` UTC offset.  All query functions
    compare stored timestamps against this string using SQLite's lexicographic
    ordering (``timestamp >= ?``).  This works correctly for timestamps stored
    in UTC format (as produced by ``datetime.now(UTC).isoformat()``), but may
    silently include or exclude rows whose timestamps carry a non-UTC offset
    such as ``+05:00``.  The correct long-term fix is to normalise timestamps
    to UTC at write time; this is a pre-existing pattern shared by all query
    functions in this module.

    Args:
        hours: Look-back window in hours.
        now: Reference timestamp. When None (the default), ``datetime.now(UTC)``
            is used. Pass an explicit value to get deterministic results or to
            share a single timestamp across concurrent per-DB calls.
    """
    effective_now = now if now is not None else datetime.now(UTC)
    return (effective_now - timedelta(hours=hours)).isoformat()


def _bucket_minutes_for_window(hours: int) -> int:
    """Return the adaptive bucket width in minutes for the given window length.

    Iterates ``BUCKET_LADDER`` and returns the bucket width for the first tier
    whose ``max_hours`` bound is not exceeded.  The ladder's final entry has
    ``max_hours=None`` (catch-all), so this function always returns a value.

    Ladder (from ``BUCKET_LADDER``):
      <=  24 h → 15 min  (≤ 97 buckets)
      <= 168 h → 60 min  (≤ 169 buckets)
      <= 720 h → 360 min (≤ 121 buckets)
      >  720 h → 1440 min (≤ 3 651 buckets, covers window=all / 87 600 h)
    """
    for max_hours, bucket_min in BUCKET_LADDER:
        if max_hours is None or hours <= max_hours:
            return bucket_min
    return 1440  # unreachable — BUCKET_LADDER always ends with (None, ...)


def _align_bucket(t: datetime, bucket_min: int) -> datetime:
    """Floor *t* to the nearest bucket boundary using epoch-based arithmetic.

    Uses 1970-01-01 00:00 UTC as the epoch, which naturally aligns on hour
    and day boundaries for all four supported bucket widths (15/60/360/1440).

    Uses ``math.floor`` (not ``int``) so that pre-epoch timestamps (negative
    total_seconds) are floored correctly rather than truncated toward zero.
    In practice merge events are always post-epoch, but the implementation is
    correct for all inputs.

    Args:
        t: A timezone-aware datetime (UTC assumed if no tzinfo).
        bucket_min: Bucket width in minutes (15, 60, 360, or 1440).

    Returns:
        A UTC-aware datetime at the start of the bucket containing *t*.
    """
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    bucket_sec = bucket_min * 60
    total_sec = math.floor((t - epoch).total_seconds())
    aligned_sec = (total_sec // bucket_sec) * bucket_sec
    return epoch + timedelta(seconds=aligned_sec)


# ---------------------------------------------------------------------------
# 1. Queue depth timeseries (15-min bins)
# ---------------------------------------------------------------------------

async def queue_depth_timeseries(
    db: aiosqlite.Connection | None,
    *,
    hours: int = 24,
    now: datetime | None = None,
) -> ChartData:
    """Approximate merge queue throughput as adaptive-width bucket counts.

    Returns ChartData with ISO bucket-start labels and integer counts.
    Bucket width is chosen adaptively via ``_bucket_minutes_for_window`` so
    the point count stays manageable for all window sizes:

    * ``hours ≤ 24``  → 15-min buckets  (≤ 97 points)
    * ``hours ≤ 168`` → 60-min buckets  (≤ 169 points)
    * ``hours ≤ 720`` → 360-min buckets (≤ 121 points)
    * ``hours > 720`` → 1440-min buckets (≤ 3 651 points, covers window=all)

    Buckets span ``[_align_bucket(now - hours, bm), _align_bucket(now, bm)]``
    inclusive.  The current bucket is always included.

    Args:
        db: aiosqlite connection, or None (returns empty ChartData).
        hours: Look-back window in hours (default 24).
        now: Reference timestamp for bucket alignment.  When None (the default),
            ``datetime.now(UTC)`` is used.  Pass an explicit value in tests to
            get deterministic bucket counts and eliminate boundary flakiness.
    """
    if db is None:
        return {'labels': [], 'values': []}

    async def _query(conn: aiosqlite.Connection) -> ChartData:
        effective_now = now if now is not None else datetime.now(UTC)
        cutoff = effective_now - timedelta(hours=hours)

        # Determine adaptive bucket width for this window
        bucket_min = _bucket_minutes_for_window(hours)

        # Align both ends to bucket boundaries using epoch-based flooring
        cutoff_aligned = _align_bucket(cutoff, bucket_min)
        now_aligned = _align_bucket(effective_now, bucket_min)

        # Generate buckets from cutoff_aligned through now_aligned inclusive
        num_buckets = int((now_aligned - cutoff_aligned) / timedelta(minutes=bucket_min)) + 1
        buckets = [
            cutoff_aligned + timedelta(minutes=bucket_min * i)
            for i in range(num_buckets)
        ]

        # Fetch all merge_attempt events in the window.
        # Upper bound is effective_now (not now_aligned) to avoid excluding
        # events in [now_aligned, effective_now) that belong to the last bucket.
        rows = await conn.execute_fetchall(
            "SELECT timestamp FROM events "
            "WHERE event_type = 'merge_attempt' AND timestamp >= ? AND timestamp <= ?",
            (cutoff_aligned.isoformat(), effective_now.isoformat()),
        )

        # Build count map keyed by ISO bucket label
        counts: dict[str, int] = {b.isoformat(): 0 for b in buckets}
        for row in rows:
            ts_str = row['timestamp']
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                # Floor to the adaptive bucket
                bucket = _align_bucket(ts, bucket_min)
                key = bucket.isoformat()
                if key in counts:
                    counts[key] += 1
            except (ValueError, TypeError):
                continue

        labels = [b.isoformat() for b in buckets]
        values: list[int | float] = [counts[lbl] for lbl in labels]
        return {'labels': labels, 'values': values}

    return await with_db(db, _query, {'labels': [], 'values': []})


# ---------------------------------------------------------------------------
# 2. Outcome distribution
# ---------------------------------------------------------------------------

async def outcome_distribution(
    db: aiosqlite.Connection | None,
    *,
    hours: int = 24,
    now: datetime | None = None,
) -> ChartData:
    """Count merge_attempt events by outcome within the window.

    Returns ChartData with canonical outcomes first (done, conflict, blocked,
    already_merged), then any unknown outcomes sorted alphabetically.
    Missing canonical outcomes are omitted (count=0 entries are dropped).

    Args:
        db: aiosqlite connection, or None (returns empty ChartData).
        hours: Look-back window in hours (default 24).
        now: Reference timestamp for the cutoff. When None, ``datetime.now(UTC)``
            is used. Pass an explicit value to get deterministic results or to
            share a single timestamp across concurrent per-DB calls.
    """
    if db is None:
        return {'labels': [], 'values': []}

    async def _query(conn: aiosqlite.Connection) -> ChartData:
        since = _cutoff_iso(hours, now=now)
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
        values: list[int | float] = []
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
    now: datetime | None = None,
) -> list[float]:
    """Return sorted list of non-null merge_attempt duration_ms values.

    Args:
        db: aiosqlite connection, or None (returns empty list).
        hours: Look-back window in hours (default 24).
        now: Reference timestamp for the cutoff. When None, ``datetime.now(UTC)``
            is used. Pass an explicit value to share a timestamp with sibling calls.
    """
    if db is None:
        return []

    async def _query(conn: aiosqlite.Connection) -> list[float]:
        since = _cutoff_iso(hours, now=now)
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
        'p50': round(percentile(durations, 50)),
        'p95': round(percentile(durations, 95)),
        'p99': round(percentile(durations, 99)),
        'count': len(durations),
        'mean_ms': sum(durations) / len(durations),
    }


async def latency_stats(
    db: aiosqlite.Connection | None,
    *,
    hours: int = 24,
    now: datetime | None = None,
) -> dict:
    """P50/P95/P99 latency and count for merge_attempt events.

    Returns {'p50': int, 'p95': int, 'p99': int, 'count': int,
             'mean_ms': float}.
    When no rows have non-null duration_ms, returns all zeros with count=0.

    Args:
        db: aiosqlite connection, or None (returns all-zeros dict).
        hours: Look-back window in hours (default 24).
        now: Reference timestamp for the cutoff. When None, ``datetime.now(UTC)``
            is used. Pass an explicit value to share a timestamp with sibling calls.
    """
    durations = await _get_durations(db, hours=hours, now=now)
    return _compute_latency_stats(durations)


# ---------------------------------------------------------------------------
# 4. Recent merges
# ---------------------------------------------------------------------------

async def recent_merges(
    db: aiosqlite.Connection | None,
    *,
    limit: int = 20,
    hours: int = 168,
) -> list[dict]:
    """Most recent merge_attempt events, newest first.

    Args:
        db: Async SQLite connection, or None (returns []).
        limit: Maximum number of rows to return.
        hours: Look-back window in hours (default 168 = 7 days).  Only
            events with ``timestamp >= now - hours`` are included.

    Returns list of {'task_id', 'run_id', 'outcome', 'duration_ms',
                     'timestamp'} dicts.
    """
    if db is None:
        return []

    async def _query(conn: aiosqlite.Connection) -> list[dict]:
        since = _cutoff_iso(hours)
        rows = await conn.execute_fetchall(
            "SELECT task_id, run_id, "
            "       json_extract(data, '$.outcome') AS outcome, "
            "       duration_ms, timestamp "
            "FROM events "
            "WHERE event_type = 'merge_attempt' "
            "  AND timestamp >= ? "
            "ORDER BY timestamp DESC "
            "LIMIT ?",
            (since, limit),
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
    now: datetime | None = None,
) -> dict:
    """Hit/discard counts and hit rate for speculative merge events.

    Returns {'hit_count': int, 'discard_count': int, 'total': int,
             'hit_rate': float}.

    Args:
        db: aiosqlite connection, or None (returns all-zeros dict).
        hours: Look-back window in hours (default 24).
        now: Reference timestamp for the cutoff. When None, ``datetime.now(UTC)``
            is used. Pass an explicit value to share a timestamp with sibling calls.
    """
    if db is None:
        return {'hit_count': 0, 'discard_count': 0, 'total': 0, 'hit_rate': 0.0}

    async def _query(conn: aiosqlite.Connection) -> dict:
        since = _cutoff_iso(hours, now=now)
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
    now: datetime | None = None,
) -> ChartData:
    """Aggregate queue depth timeseries across multiple project DBs.

    Counts per bucket are summed across all DBs.  Bucket width is adaptive
    to ``hours`` (see ``_bucket_minutes_for_window``).

    Args:
        dbs: List of aiosqlite connections (None entries are tolerated).
        hours: Look-back window in hours (default 24).
        now: Reference timestamp captured **once** for the entire aggregation
            call and threaded into every per-DB ``queue_depth_timeseries``
            query.  When None (the default), ``datetime.now(UTC)`` is resolved
            here so that all concurrent per-DB coroutines share the same
            alignment — eliminating the race where concurrent calls to
            ``datetime.now(UTC)`` inside each per-DB ``_query`` could straddle
            a bucket boundary and produce divergent label sets.  Pass an
            explicit value in tests for full determinism.
    """
    effective_now = now if now is not None else datetime.now(UTC)
    results = await asyncio.gather(
        *[queue_depth_timeseries(db, hours=hours, now=effective_now) for db in dbs],
        return_exceptions=True,
    )

    # Collect all valid results
    valid: list[ChartData] = []
    for r in results:
        if isinstance(r, BaseException):
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
    total_counts: dict[str, int | float] = {lbl: 0 for lbl in labels}
    for v in valid:
        for lbl, cnt in zip(v['labels'], v['values'], strict=False):
            if lbl in total_counts:
                total_counts[lbl] += cnt

    return {'labels': labels, 'values': [total_counts[lbl] for lbl in labels]}


async def aggregate_outcome_distribution(
    dbs: list[aiosqlite.Connection | None],
    *,
    hours: int = 24,
    now: datetime | None = None,
) -> ChartData:
    """Aggregate outcome distribution across multiple project DBs.

    Counts per outcome are summed; canonical ordering is preserved.

    Args:
        dbs: List of aiosqlite connections (None entries are tolerated).
        hours: Look-back window in hours (default 24).
        now: Reference timestamp captured **once** for the entire aggregation
            call and threaded into every per-DB ``outcome_distribution`` query.
            When None (the default), ``datetime.now(UTC)`` is resolved here so
            that all concurrent per-DB coroutines share the same cutoff window.
            Pass an explicit value in tests for full determinism.
    """
    effective_now = now if now is not None else datetime.now(UTC)
    results = await asyncio.gather(
        *[outcome_distribution(db, hours=hours, now=effective_now) for db in dbs],
        return_exceptions=True,
    )

    merged: dict[str, int | float] = {}
    for r in results:
        if isinstance(r, BaseException):
            logger.warning('aggregate_outcome_distribution: error from one DB: %s', r)
            continue
        for lbl, cnt in zip(r['labels'], r['values'], strict=False):
            merged[lbl] = merged.get(lbl, 0) + cnt

    if not merged:
        return {'labels': [], 'values': []}

    # Re-apply canonical ordering
    labels: list[str] = []
    values: list[int | float] = []
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
    now: datetime | None = None,
) -> dict:
    """Aggregate latency stats across multiple project DBs.

    Recomputes percentiles from the merged raw duration list.

    Args:
        dbs: List of aiosqlite connections (None entries are tolerated).
        hours: Look-back window in hours (default 24).
        now: Reference timestamp captured **once** for the entire aggregation
            call and threaded into every per-DB ``_get_durations`` query.
            When None (the default), ``datetime.now(UTC)`` is resolved here so
            that all concurrent per-DB coroutines share the same cutoff window.
            Pass an explicit value in tests for full determinism.
    """
    effective_now = now if now is not None else datetime.now(UTC)
    all_durations: list[float] = []
    gather_results = await asyncio.gather(
        *[_get_durations(db, hours=hours, now=effective_now) for db in dbs],
        return_exceptions=True,
    )
    for r in gather_results:
        if isinstance(r, BaseException):
            logger.warning('aggregate_latency_stats: error from one DB: %s', r)
            continue
        all_durations.extend(r)

    return _compute_latency_stats(sorted(all_durations))


async def aggregate_recent_merges(
    dbs: list[aiosqlite.Connection | None],
    *,
    limit: int = 20,
    hours: int = 168,
) -> list[dict]:
    """Aggregate recent merges across multiple project DBs.

    Concatenates, re-sorts by timestamp DESC, and truncates to limit.

    Args:
        dbs: List of async SQLite connections (None entries are skipped).
        limit: Maximum number of rows to return.
        hours: Look-back window in hours (default 168 = 7 days), passed
            through to each per-DB :func:`recent_merges` call.
    """
    gather_results = await asyncio.gather(
        *[recent_merges(db, limit=limit, hours=hours) for db in dbs],
        return_exceptions=True,
    )
    merged: list[dict] = []
    for r in gather_results:
        if isinstance(r, BaseException):
            logger.warning('aggregate_recent_merges: error from one DB: %s', r)
            continue
        merged.extend(r)

    merged.sort(key=_ts_sort_key, reverse=True)
    return merged[:limit]


async def aggregate_speculative_stats(
    dbs: list[aiosqlite.Connection | None],
    *,
    hours: int = 24,
    now: datetime | None = None,
) -> dict:
    """Aggregate speculative stats across multiple project DBs.

    Sums hit/discard counts and recomputes hit_rate.

    Args:
        dbs: List of aiosqlite connections (None entries are tolerated).
        hours: Look-back window in hours (default 24).
        now: Reference timestamp captured **once** for the entire aggregation
            call and threaded into every per-DB ``speculative_stats`` query.
            When None (the default), ``datetime.now(UTC)`` is resolved here so
            that all concurrent per-DB coroutines share the same cutoff window.
            Pass an explicit value in tests for full determinism.
    """
    effective_now = now if now is not None else datetime.now(UTC)
    gather_results = await asyncio.gather(
        *[speculative_stats(db, hours=hours, now=effective_now) for db in dbs],
        return_exceptions=True,
    )
    hit_count = 0
    discard_count = 0
    for r in gather_results:
        if isinstance(r, BaseException):
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
