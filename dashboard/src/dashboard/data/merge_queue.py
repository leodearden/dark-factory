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
"""

from __future__ import annotations

import asyncio
import functools
import logging
import math
import os
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite

from dashboard.data.chart_utils import ChartData
from dashboard.data.db import with_db
from dashboard.data.orchestrator import load_task_tree
from dashboard.data.stats_utils import percentile
from dashboard.data.utils import parse_utc, safe_gather_result

logger = logging.getLogger(__name__)


def _ts_sort_key(entry: dict) -> datetime:
    """Return a UTC-aware datetime sort key for a merge entry dict.

    Parses ``entry['timestamp']`` via :func:`parse_utc`.  Returns
    ``datetime.min`` (UTC-aware) on missing, None, or unparseable values so
    that malformed entries sort to the end of a descending sort.
    """
    try:
        return parse_utc(entry.get('timestamp')).astimezone(UTC)
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=UTC)

_CANONICAL_OUTCOMES = ['done', 'conflict', 'blocked', 'already_merged']

_TERMINAL_MERGE_OUTCOMES: frozenset[str] = frozenset({
    'done', 'already_merged', 'conflict', 'blocked',
    'dropped_plan_targets', 'cas_exhausted', 'abandoned_verify_timeouts',
})
_ACTIVE_EVENT_TYPES: tuple[str, ...] = ('merge_queued', 'merge_dequeued', 'merge_attempt')

# Soft operational-warning threshold: warn when recent_merges returns more rows
# than this while running unbounded (limit=None).  Signals a possible producer
# burst worth investigating before memory pressure is reached.
_RECENT_MERGES_BURST_WARN = 1_000

# Hard memory-safety bound for recent_merges(limit=None).  Enforced via a
# SQL LIMIT probe (_RECENT_MERGES_HARD_CAP + 1) so that a probe result of
# exactly hard_cap rows does NOT fire a false-positive WARN.
_RECENT_MERGES_HARD_CAP = 100_000

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
        num_buckets = (now_aligned - cutoff_aligned) // timedelta(minutes=bucket_min) + 1
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
    limit: int | None = 20,
    hours: int = 168,
    now: datetime | None = None,
) -> list[dict]:
    """Most recent merge_attempt events, newest first.

    Args:
        db: Async SQLite connection, or None (returns []).
        limit: Maximum number of rows to return.  When ``None``, rows are
            bounded by ``_RECENT_MERGES_HARD_CAP`` (enforced via a SQL probe)
            rather than returned without bound.  A WARNING is logged if the
            hard cap is reached.  The SQL WHERE window (``hours``) is the
            primary bound; the hard cap is a memory-safety backstop.  When an
            explicit ``limit`` exceeds ``_RECENT_MERGES_HARD_CAP`` it is
            clamped to the cap and a WARNING is logged on actual truncation
            (symmetric with the ``limit=None`` branch).
        hours: Look-back window in hours (default 168 = 7 days).  Only
            events with ``timestamp >= now - hours`` are included.
        now: Reference timestamp for the cutoff window (default:
            ``datetime.now(UTC)``).  Pass an explicit value in tests for
            full determinism.

    Returns list of {'task_id', 'run_id', 'outcome', 'duration_ms',
                     'timestamp'} dicts.
    """
    if db is None:
        return []

    async def _query(conn: aiosqlite.Connection) -> list[dict]:
        since = _cutoff_iso(hours, now=now)
        base_sql = (
            "SELECT task_id, run_id, "
            "       json_extract(data, '$.outcome') AS outcome, "
            "       duration_ms, timestamp "
            "FROM events "
            "WHERE event_type = 'merge_attempt' "
            "  AND timestamp >= ? "
            "ORDER BY timestamp DESC"
        )
        # cap_in_effect: hard-cap backstop applies when limit is None OR when
        # an explicit limit exceeds the hard cap (so the cap is truly hard).
        cap_in_effect = limit is None or limit > _RECENT_MERGES_HARD_CAP
        if cap_in_effect:
            # Probe with hard_cap + 1 rows so we can distinguish "exactly
            # hard_cap rows existed" from "window exceeded hard_cap".
            probe = _RECENT_MERGES_HARD_CAP + 1
            sql, params = base_sql + " LIMIT ?", (since, probe)
        else:
            sql, params = base_sql + " LIMIT ?", (since, limit)
        rows = list(await conn.execute_fetchall(sql, params))
        if cap_in_effect and len(rows) > _RECENT_MERGES_HARD_CAP:
            logger.warning(
                'recent_merges: hard cap %d reached; window contained more rows '
                'than the cap and was truncated to %d — consider adding a tighter '
                'time window or rate-limiting the producer',
                _RECENT_MERGES_HARD_CAP,
                _RECENT_MERGES_HARD_CAP,
            )
            rows = rows[:_RECENT_MERGES_HARD_CAP]
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
# 6. Active queued merges
# ---------------------------------------------------------------------------


async def active_queued_merges(
    db: aiosqlite.Connection | None,
    *,
    ttl_minutes: int = 30,
    now: datetime | None = None,
) -> list[dict]:
    """Return tasks whose latest merge-lifecycle event is not a terminal outcome.

    Queries events with event_type IN ('merge_queued', 'merge_dequeued',
    'merge_attempt'), picks the latest row per task_id within the TTL window,
    and excludes tasks whose latest event is a terminal merge_attempt outcome
    (done, already_merged, conflict, blocked, dropped_plan_targets,
    cas_exhausted, abandoned_verify_timeouts).

    Args:
        db:  Async SQLite connection, or None (returns []).
        ttl_minutes:  Drop tasks whose latest relevant event is older than
            this many minutes.  Acts as a safety net for crashed orchestrators
            that left dangling merge_queued rows.
        now:  Reference timestamp for the TTL cutoff.  Defaults to
            ``datetime.now(UTC)`` when None.

    Returns:
        List of dicts with keys: task_id, run_id, state, timestamp, branch,
        outcome.  ``state`` is 'queued' when latest event is merge_queued;
        'in_flight' for merge_dequeued or merge_attempt(cas_retry).
    """
    if db is None:
        return []

    effective_now = now if now is not None else datetime.now(UTC)
    cutoff = (effective_now - timedelta(minutes=ttl_minutes)).isoformat()
    et_placeholders = ','.join('?' * len(_ACTIVE_EVENT_TYPES))

    async def _query(conn: aiosqlite.Connection) -> list[dict]:
        sql = f"""
            SELECT task_id, run_id, event_type,
                   json_extract(data, '$.outcome') AS outcome,
                   json_extract(data, '$.branch') AS branch,
                   timestamp
            FROM events e
            WHERE event_type IN ({et_placeholders})
              AND timestamp >= ?
              AND timestamp = (
                  SELECT MAX(timestamp)
                  FROM events e2
                  WHERE e2.task_id = e.task_id
                    AND e2.event_type IN ({et_placeholders})
                    AND e2.timestamp >= ?
              )
        """
        params = (*_ACTIVE_EVENT_TYPES, cutoff, *_ACTIVE_EVENT_TYPES, cutoff)
        rows = await conn.execute_fetchall(sql, params)

        result = []
        for row in rows:
            et = row['event_type']
            outcome = row['outcome']
            # Exclude terminal merge_attempt rows
            if et == 'merge_attempt' and outcome in _TERMINAL_MERGE_OUTCOMES:
                continue
            # Derive state
            state = 'queued' if et == 'merge_queued' else 'in_flight'
            result.append({
                'task_id': row['task_id'],
                'run_id': row['run_id'],
                'state': state,
                'timestamp': row['timestamp'],
                'branch': row['branch'],
                'outcome': outcome,
            })
        return result

    return await with_db(db, _query, [])


# ---------------------------------------------------------------------------
# 7. Per-project helpers
# ---------------------------------------------------------------------------


def filter_merges_within(
    merges: list[dict],
    *,
    minutes: int,
    now: datetime | None = None,
) -> list[dict]:
    """Return only the rows whose timestamp falls within the last *minutes*.

    Args:
        merges: List of merge-row dicts.  Each dict must have a 'timestamp' key
            whose value is an ISO-8601 string parseable by :func:`parse_utc`.
        minutes: Sliding-window width in minutes.  Rows older than
            ``now - timedelta(minutes=minutes)`` are excluded.
        now: Reference timestamp.  Defaults to ``datetime.now(UTC)`` when None.

    Returns:
        A new list preserving the input order.  Rows with missing or malformed
        timestamps are silently dropped.
    """
    effective_now = now if now is not None else datetime.now(UTC)
    cutoff = effective_now - timedelta(minutes=minutes)
    result: list[dict] = []
    for row in merges:
        try:
            ts = parse_utc(row.get('timestamp')).astimezone(UTC)
            if ts >= cutoff:
                result.append(row)
        except (ValueError, TypeError, AttributeError):
            pass  # malformed timestamp or parse_utc returned None → drop row
    return result


def enrich_merges_with_titles(
    merges: list[dict],
    task_title_map: dict[str, str],
) -> list[dict]:
    """Return a new list of merge rows with a 'title' field added to each.

    For each row, the key ``str(row['task_id'])`` is looked up in
    *task_title_map*.  Rows with ``task_id=None`` or an unknown task_id get
    ``title=''``.  Input rows are NOT mutated (a shallow copy is made for
    each row).

    Args:
        merges: List of merge-row dicts (from :func:`recent_merges` or similar).
        task_title_map: Mapping of ``str(task_id) → title`` built by
            :func:`load_task_titles`.

    Returns:
        New list of dicts, each with an added 'title' key.
    """
    result: list[dict] = []
    for row in merges:
        raw_id = row.get('task_id')
        title = task_title_map.get(str(raw_id), '') if raw_id is not None else ''
        result.append({**row, 'title': title})
    return result


# 32 comfortably covers the expected number of concurrently enumerated projects.
# If the working set exceeds 32, the LRU evicts the oldest entry and the next access
# triggers one extra load_task_tree call (a single JSON re-parse) — there is NO
# correctness impact. Bumping maxsize or switching to maxsize=None is therefore
# rarely worthwhile and risks unbounded memory growth if callers ever supply many
# distinct tasks.json paths (e.g., a buggy loop).
@functools.lru_cache(maxsize=32)
def _load_task_titles_cached(path_str: str, mtime_ns: int) -> dict[str, str]:
    """Cache body for :func:`load_task_titles`, keyed on ``(path, mtime_ns)``.

    Only called when the file's ``st_mtime_ns`` differs from the last observed
    value; otherwise :func:`load_task_titles` returns the cached result without
    entering this function.

    The returned dict is shared across callers via the LRU cache — do NOT
    mutate it.  :func:`load_task_titles` returns a shallow copy to callers so
    they cannot reach this cached object.
    """
    return {str(t['id']): t['title'] for t in load_task_tree(Path(path_str)) if t.get('title')}


def load_task_titles(tasks_json_path: Path) -> dict[str, str]:
    """Return a {str(task_id): title} map from a Taskmaster tasks.json file.

    Wraps :func:`dashboard.data.orchestrator.load_task_tree` and builds the
    mapping needed by :func:`enrich_merges_with_titles`.  Tasks without a
    title (``title=None`` or missing) are omitted.

    Results are mtime-keyed: repeat calls where the file's ``st_mtime_ns`` has
    not changed return a cached ``dict`` in O(1) without re-reading the file.
    A missing or inaccessible file short-circuits to ``{}`` before the cache is
    consulted, preserving the original OSError-safe contract.

    The path is resolved to its real path before caching so that different
    spellings of the same file (symlinks, relative vs. absolute) map to the
    same cache entry.  A fresh shallow copy of the cached mapping is returned
    to callers so that downstream mutation cannot poison the shared cache entry.

    Args:
        tasks_json_path: Path to the ``.taskmaster/tasks/tasks.json`` file.

    Returns:
        Dict mapping ``str(task.id)`` to ``task.title``.  Returns ``{}`` on
        missing file, invalid JSON, or missing tasks structure.
    """
    real_path = os.path.realpath(tasks_json_path)
    try:
        mtime_ns = os.stat(real_path).st_mtime_ns
    except OSError:
        return {}
    return dict(_load_task_titles_cached(real_path, mtime_ns))


async def build_per_project_merge_queue(
    project_dbs: Sequence[tuple[str, aiosqlite.Connection | None]],
    *,
    hours: int,
    now: datetime,
    recent_window_minutes: int,
) -> dict[str, dict]:
    """Build per-project merge queue stats by querying each project's DB independently.

    For each ``(pid, db)`` pair, gathers the 5 per-DB stats concurrently and
    applies :func:`filter_merges_within` to the recent-merges list.  Pairs with
    ``db=None`` produce empty/default stats (the per-DB functions handle None
    gracefully by returning declared defaults).  All per-project gathers also
    run concurrently across projects via a single top-level :func:`asyncio.gather`.

    The recent-merges SQL query uses an hour-granular window derived from
    ``recent_window_minutes`` (``max(1, ceil(recent_window_minutes / 60))``
    hours) with ``limit=None`` so that bursts exceeding any fixed row cap are
    not silently truncated.  :func:`filter_merges_within` then tightens the
    result to the precise ``recent_window_minutes`` boundary.

    Args:
        project_dbs: List of ``(project_root_str, connection_or_None)`` tuples
            from :func:`_project_scoped_dbs_labeled`.
        hours: Look-back window in hours (forwarded to each per-DB function).
        now: Shared reference timestamp captured once per request.
        recent_window_minutes: Sliding window for recent-merges trimming.  The
            SQL WHERE uses ``max(1, ceil(recent_window_minutes / 60))`` hours;
            the Python post-filter tightens to the exact minute boundary.

    Returns:
        Dict ``{pid: {depth_timeseries, outcomes, latency, recent, speculative, active}}``.
    """
    _DEFAULT_DEPTH: ChartData = {'labels': [], 'values': []}
    _DEFAULT_OUTCOMES: ChartData = {'labels': [], 'values': []}
    _DEFAULT_LATENCY = {'p50': 0, 'p95': 0, 'p99': 0, 'count': 0, 'mean_ms': 0.0}
    _DEFAULT_SPEC = {'hit_count': 0, 'discard_count': 0, 'total': 0, 'hit_rate': 0.0}

    recent_hours = max(1, math.ceil(recent_window_minutes / 60))

    async def _one_project(pid: str, db: aiosqlite.Connection | None) -> tuple[str, dict]:
        try:
            depth_r, outcomes_r, latency_r, recent_r, spec_r, active_r = await asyncio.gather(
                queue_depth_timeseries(db, hours=hours, now=now),
                outcome_distribution(db, hours=hours, now=now),
                latency_stats(db, hours=hours, now=now),
                recent_merges(db, limit=None, hours=recent_hours, now=now),
                speculative_stats(db, hours=hours, now=now),
                active_queued_merges(db, ttl_minutes=30, now=now),
                return_exceptions=True,
            )
            depth = safe_gather_result(depth_r, _DEFAULT_DEPTH, f'{pid}/depth')
            outcomes = safe_gather_result(outcomes_r, _DEFAULT_OUTCOMES, f'{pid}/outcomes')
            latency = safe_gather_result(latency_r, _DEFAULT_LATENCY, f'{pid}/latency')
            recent_raw = safe_gather_result(recent_r, [], f'{pid}/recent')
            spec = safe_gather_result(spec_r, _DEFAULT_SPEC, f'{pid}/speculative')
            active_list = safe_gather_result(active_r, [], f'{pid}/active')
            if len(recent_raw) > _RECENT_MERGES_BURST_WARN:  # type: ignore[arg-type]
                logger.warning(
                    'build_per_project_merge_queue %s: recent_merges returned %d rows'
                    ' (limit=None, hours=%d) — possible runaway burst; consider'
                    ' rate-limiting the producer or adding capacity monitoring',
                    pid,
                    len(recent_raw),  # type: ignore[arg-type]
                    recent_hours,
                )

            recent_trimmed = filter_merges_within(
                recent_raw,  # type: ignore[arg-type]
                minutes=recent_window_minutes,
                now=now,
            )
            return pid, {
                'depth_timeseries': depth,
                'outcomes': outcomes,
                'latency': latency,
                'recent': recent_trimmed,
                'speculative': spec,
                'active': active_list,
            }
        except Exception as exc:
            logger.warning(
                'build_per_project_merge_queue %s: unexpected error (returning defaults): %s',
                pid,
                exc,
            )
            return pid, {
                'depth_timeseries': _DEFAULT_DEPTH,
                'outcomes': _DEFAULT_OUTCOMES,
                'latency': _DEFAULT_LATENCY,
                'recent': [],
                'speculative': _DEFAULT_SPEC,
                'active': [],
            }

    results = await asyncio.gather(*[_one_project(pid, db) for pid, db in project_dbs])
    return dict(results)

