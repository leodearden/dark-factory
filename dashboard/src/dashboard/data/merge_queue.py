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
import json
import logging
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, TypedDict

import aiosqlite
import httpx

from dashboard.data.chart_utils import ChartData
from dashboard.data.datum import Datum, DatumState, unknown_datum
from dashboard.data.db import with_db
from dashboard.data.memory import mcp_tool_call
from dashboard.data.stats_utils import percentile
from dashboard.data.task_lookup import FETCHED_ROW_FRESHNESS_BOUND_SECONDS, TaskRef
from dashboard.data.utils import parse_utc, resolve_now, safe_gather_result

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

# How many of a window's merge_attempt rows the Recent-merges table shows
# (PRD open question 5). Measured 19 / 228 / 982 / 4901 events at
# 24h / 7d / 30d / all, so the cap never bites at the default 24h and bites
# from 7d — which is when "showing N of M" carries information.
RECENT_MERGES_CAP = 200

LIVE_QUEUE_FRESHNESS_BOUND_SECONDS = 30
"""How long a live queue reading stays fresh.

Equal to ``task_snapshot.FRESHNESS_BOUND_SECONDS`` and for the same reason:
it must outlast the route's own worst-case fan-out — the probes, then the
task lookup's deadline — so the routine slow path does not serve its own
probe stale. Ageing in the browser is the client's age badge's job.
"""

RUNS_DB_READ_FRESHNESS_BOUND_SECONDS = LIVE_QUEUE_FRESHNESS_BOUND_SECONDS
"""How long a runs.db window read stays fresh.

A window read is stamped at the route's ``render_at``, the same instant the
live probe is stamped, so it must outlast the same fan-out — hence the same
number, held once.
"""

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
    effective_now = resolve_now(now)
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
        effective_now = resolve_now(now)
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
# 2. Merge attempts — one window read behind the outcome chart and the latency
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MergeAttempts:
    """Every merge_attempt of one window, read once.

    The outcome chart and the latency block are two projections of this one
    row set, so the chart's total always equals ``with_duration +
    without_duration`` — a headline can never count a subset of what the
    donut beside it counts (PRD dashboard-one-datum-one-path, sketch #9).

    Attributes:
        outcomes: Every attempt counted by outcome; an unrecorded outcome
            counts as ``'unknown'``.
        durations: The attempts' positive ``duration_ms`` values, sorted
            ascending. Attempts with a NULL or zero duration are counted in
            :attr:`outcomes` and absent here.
    """

    outcomes: Mapping[str, int] = field(default_factory=dict)
    durations: tuple[float, ...] = ()

    def outcome_chart(self) -> ChartData:
        """Outcomes by count descending, ties alphabetical, zero counts omitted."""
        ordered = sorted(
            ((label, count) for label, count in self.outcomes.items() if count),
            key=lambda item: (-item[1], item[0]),
        )
        return {
            'labels': [label for label, _ in ordered],
            'values': [count for _, count in ordered],
        }

    def latency(self) -> dict:
        """Centiles and mean over the timed attempts, and the timed/untimed split."""
        with_duration = len(self.durations)
        without_duration = sum(self.outcomes.values()) - with_duration
        if not self.durations:
            return {
                'p50': 0, 'p95': 0, 'p99': 0, 'mean_ms': 0.0,
                'with_duration': 0, 'without_duration': without_duration,
            }
        return {
            'p50': round(percentile(self.durations, 50)),
            'p95': round(percentile(self.durations, 95)),
            'p99': round(percentile(self.durations, 99)),
            'mean_ms': sum(self.durations) / with_duration,
            'with_duration': with_duration,
            'without_duration': without_duration,
        }


async def merge_attempts(
    db: aiosqlite.Connection | None,
    *,
    hours: int = 24,
    now: datetime | None = None,
) -> MergeAttempts:
    """Read every merge_attempt in the window with ONE query.

    Args:
        db: aiosqlite connection, or None (returns an empty record).
        hours: Look-back window in hours (default 24).
        now: Reference timestamp for the cutoff. When None, ``datetime.now(UTC)``
            is used. Pass an explicit value to share a timestamp with sibling calls.
    """
    if db is None:
        return MergeAttempts()

    async def _query(conn: aiosqlite.Connection) -> MergeAttempts:
        rows = await conn.execute_fetchall(
            "SELECT json_extract(data, '$.outcome') AS outcome, duration_ms "
            "FROM events "
            "WHERE event_type = 'merge_attempt' AND timestamp >= ?",
            (_cutoff_iso(hours, now=now),),
        )
        outcomes = Counter(row['outcome'] or 'unknown' for row in rows)
        durations = sorted(
            float(row['duration_ms']) for row in rows
            if row['duration_ms'] is not None and row['duration_ms'] > 0
        )
        return MergeAttempts(outcomes=dict(outcomes), durations=tuple(durations))

    return await with_db(db, _query, MergeAttempts())


# ---------------------------------------------------------------------------
# 4. Recent merges
# ---------------------------------------------------------------------------

class RecentMerges(TypedDict):
    """The newest merge_attempt rows of a window, and how many the window holds."""

    rows: list[dict]
    total: int


async def recent_merges(
    db: aiosqlite.Connection | None,
    *,
    limit: int,
    hours: int,
    now: datetime | None = None,
) -> RecentMerges:
    """The newest ``limit`` merge_attempt events of the window, plus the window's total.

    ``total`` comes from ``COUNT(*) OVER ()`` in the same statement as the
    rows. The window function is evaluated before ``LIMIT``, so it counts
    every event in the window, and ``len(rows) <= total`` holds by
    construction — no second query can see a different window.

    Args:
        db: Async SQLite connection, or None (returns no rows, total 0).
        limit: Maximum number of rows to return. Must be at least 1: with
            ``LIMIT 0`` no row carries the window total, so it would read 0.
        hours: Look-back window in hours. Only events with
            ``timestamp >= now - hours`` are included.
        now: Reference timestamp for the cutoff window (default:
            ``datetime.now(UTC)``).

    Returns ``{'rows': [...], 'total': int}``, each row a
    ``{'task_id', 'run_id', 'outcome', 'duration_ms', 'timestamp'}`` dict,
    newest first.

    Raises:
        ValueError: ``limit`` is less than 1.
    """
    if limit < 1:
        raise ValueError(f'recent_merges limit must be at least 1, got {limit}')
    empty: RecentMerges = {'rows': [], 'total': 0}
    if db is None:
        return empty

    async def _query(conn: aiosqlite.Connection) -> RecentMerges:
        rows = list(await conn.execute_fetchall(
            "SELECT task_id, run_id, "
            "       json_extract(data, '$.outcome') AS outcome, "
            "       duration_ms, timestamp, "
            "       COUNT(*) OVER () AS total "
            "FROM events "
            "WHERE event_type = 'merge_attempt' "
            "  AND timestamp >= ? "
            "ORDER BY timestamp DESC "
            "LIMIT ?",
            (_cutoff_iso(hours, now=now), limit),
        ))
        return {
            'rows': [
                {
                    'task_id': row['task_id'],
                    'run_id': row['run_id'],
                    'outcome': row['outcome'],
                    'duration_ms': row['duration_ms'],
                    'timestamp': row['timestamp'],
                }
                for row in rows
            ],
            'total': rows[0]['total'] if rows else 0,
        }

    return await with_db(db, _query, empty)


async def recent_train_events(
    db: aiosqlite.Connection | None,
    *,
    limit: int = 50,
    hours: int = 168,
    now: datetime | None = None,
) -> list[dict]:
    """Most recent train_* lifecycle events, newest first.

    Args:
        db: Async SQLite connection, or None (returns []).
        limit: Maximum number of rows to return.
        hours: Look-back window in hours (default 168 = 7 days).
        now: Reference timestamp for the cutoff window (default:
            ``datetime.now(UTC)``).

    Returns list of {'task_id', 'run_id', 'event_type', 'timestamp', 'data'} dicts
    where 'data' is a parsed dict.
    """
    if db is None:
        return []

    async def _query(conn: aiosqlite.Connection) -> list[dict]:
        since = _cutoff_iso(hours, now=now)
        sql = (
            "SELECT task_id, run_id, event_type, timestamp, data "
            "FROM events "
            "WHERE event_type IN ("
            "  'train_started', 'train_member_deferred', "
            "  'train_merged', 'train_derailed'"
            ") "
            "  AND timestamp >= ? "
            "ORDER BY timestamp DESC "
            "LIMIT ?"
        )
        rows = list(await conn.execute_fetchall(sql, (since, limit)))
        return [
            {
                'task_id': row['task_id'],
                'run_id': row['run_id'],
                'event_type': row['event_type'],
                'timestamp': row['timestamp'],
                'data': json.loads(row['data'] or '{}'),
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
) -> Datum[dict]:
    """Hit/discard counts and hit rate for speculative merge events.

    Args:
        db: aiosqlite connection, or None (no runs.db is open for the project).
        hours: Look-back window in hours (default 24).
        now: Reference timestamp for the cutoff. When None, ``datetime.now(UTC)``
            is used. Pass an explicit value to share a timestamp with sibling calls.

    Returns:
        One ``Datum`` over ``{'hit_count', 'discard_count', 'total',
        'hit_rate'}``, FRESH at *now* when the window was read. ``hit_rate`` is
        None for a zero-attempt window, which has no rate. With no runs.db, or
        when the read fails, the Datum is UNKNOWN with a reason saying which —
        never measured-looking zeros.
    """
    bound = RUNS_DB_READ_FRESHNESS_BOUND_SECONDS
    if db is None:
        return unknown_datum('no runs.db is open for this project', bound)

    async def _query(conn: aiosqlite.Connection) -> Datum[dict]:
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
        stats = {
            'hit_count': hit_count,
            'discard_count': discard_count,
            'total': total,
            'hit_rate': hit_count / total if total > 0 else None,
        }
        return Datum(stats, resolve_now(now), DatumState.FRESH, None, bound)

    return await with_db(
        db, _query,
        unknown_datum('the speculative-merge events could not be read from runs.db', bound),
    )


# ---------------------------------------------------------------------------
# 5b. Train throughput stats
# ---------------------------------------------------------------------------

# All-zeros default dict for train_throughput_stats (returned when db is None
# or when an exception occurs inside the query).
_TRAIN_THROUGHPUT_DEFAULT: dict = {
    'trains_landed': 0,
    'tasks_landed_via_trains': 0,
    'train_verifies_per_landed_task': 0.0,
    'baseline_solo_landed': 0,
    'baseline_verifies_per_landed_task': 0.0,
    'verifies_per_landed_task_delta': 0.0,
    'train_cas_retry_rate': 0.0,
    'baseline_cas_retry_rate': 0.0,
    'cas_retry_rate_delta': 0.0,
    'improved': False,
}


async def train_throughput_stats(
    db: aiosqlite.Connection | None,
    *,
    hours: int = 24,
    now: datetime | None = None,
) -> dict:
    """Throughput amortisation metrics derived from the events table.

    Reads train_merged + merge_attempt rows in the look-back window and computes:
    - trains_landed: count of train_merged events
    - tasks_landed_via_trains: Σ len(member_task_ids) over train_merged rows
    - train_verifies_per_landed_task: trains_landed / tasks_landed_via_trains
      (one union verify per landed train; 0.0 when no trains)
    - baseline_solo_landed: count of merge_attempt(outcome='done', train_id IS NULL)
    - baseline_verifies_per_landed_task: 1.0 when any solo landed, else 0.0
    - verifies_per_landed_task_delta: baseline − train (positive = amortisation win)
    - train_cas_retry_rate: train cas_retry count / tasks_landed_via_trains
    - baseline_cas_retry_rate: solo cas_retry count / baseline_solo_landed
    - cas_retry_rate_delta: baseline − train
    - improved: True when verifies_per_landed_task_delta > 0

    Returns the all-zeros default when db is None or on exception.

    Args:
        db: aiosqlite connection, or None (returns all-zeros dict).
        hours: Look-back window in hours (default 24).
        now: Reference timestamp. When None, uses datetime.now(UTC).
    """
    if db is None:
        return dict(_TRAIN_THROUGHPUT_DEFAULT)

    async def _query(conn: aiosqlite.Connection) -> dict:
        since = _cutoff_iso(hours, now=now)
        # --- train_merged rows ---
        train_merged_rows = list(await conn.execute_fetchall(
            "SELECT data, timestamp FROM events "
            "WHERE event_type = 'train_merged' AND timestamp >= ?",
            (since,),
        ))
        trains_landed = len(train_merged_rows)
        tasks_landed_via_trains = 0
        for row in train_merged_rows:
            try:
                data = json.loads(row['data'] or '{}')
                members = data.get('member_task_ids') or []
                if not isinstance(members, list):
                    logger.warning(
                        "train_throughput_stats: member_task_ids is not a list "
                        "(got %s) at timestamp=%s; skipping row",
                        type(members).__name__,
                        row['timestamp'],
                    )
                    continue
                tasks_landed_via_trains += len(members)
            except Exception as exc:
                logger.warning(
                    "train_throughput_stats: failed to parse train_merged data "
                    "at timestamp=%s: %s; skipping row",
                    row['timestamp'],
                    exc,
                )

        train_verifies_per_landed_task = (
            trains_landed / tasks_landed_via_trains
            if tasks_landed_via_trains > 0
            else 0.0
        )

        # --- baseline solo merge_attempt(outcome='done', train_id IS NULL) ---
        solo_rows = list(await conn.execute_fetchall(
            "SELECT COUNT(*) AS cnt FROM events "
            "WHERE event_type = 'merge_attempt' "
            "  AND json_extract(data, '$.outcome') = 'done' "
            "  AND json_extract(data, '$.train_id') IS NULL "
            "  AND timestamp >= ?",
            (since,),
        ))
        baseline_solo_landed = solo_rows[0]['cnt'] if solo_rows else 0
        baseline_verifies_per_landed_task = 1.0 if baseline_solo_landed > 0 else 0.0

        verifies_per_landed_task_delta = (
            baseline_verifies_per_landed_task - train_verifies_per_landed_task
        )

        # --- CAS-retry rates ---
        train_retry_rows = list(await conn.execute_fetchall(
            "SELECT COUNT(*) AS cnt FROM events "
            "WHERE event_type = 'merge_attempt' "
            "  AND json_extract(data, '$.outcome') = 'cas_retry' "
            "  AND json_extract(data, '$.train_id') IS NOT NULL "
            "  AND timestamp >= ?",
            (since,),
        ))
        train_retries = train_retry_rows[0]['cnt'] if train_retry_rows else 0
        train_cas_retry_rate = (
            train_retries / tasks_landed_via_trains
            if tasks_landed_via_trains > 0
            else 0.0
        )

        solo_retry_rows = list(await conn.execute_fetchall(
            "SELECT COUNT(*) AS cnt FROM events "
            "WHERE event_type = 'merge_attempt' "
            "  AND json_extract(data, '$.outcome') = 'cas_retry' "
            "  AND json_extract(data, '$.train_id') IS NULL "
            "  AND timestamp >= ?",
            (since,),
        ))
        solo_retries = solo_retry_rows[0]['cnt'] if solo_retry_rows else 0
        baseline_cas_retry_rate = (
            solo_retries / baseline_solo_landed
            if baseline_solo_landed > 0
            else 0.0
        )

        cas_retry_rate_delta = baseline_cas_retry_rate - train_cas_retry_rate
        # Only report improved=True when there is a real baseline to compare against.
        # A window with trains but no solo merges has no baseline and should not
        # report improved=False as a false regression signal.
        improved = (
            baseline_solo_landed > 0
            and trains_landed > 0
            and verifies_per_landed_task_delta > 0
        )

        return {
            'trains_landed': trains_landed,
            'tasks_landed_via_trains': tasks_landed_via_trains,
            'train_verifies_per_landed_task': train_verifies_per_landed_task,
            'baseline_solo_landed': baseline_solo_landed,
            'baseline_verifies_per_landed_task': baseline_verifies_per_landed_task,
            'verifies_per_landed_task_delta': verifies_per_landed_task_delta,
            'train_cas_retry_rate': train_cas_retry_rate,
            'baseline_cas_retry_rate': baseline_cas_retry_rate,
            'cas_retry_rate_delta': cas_retry_rate_delta,
            'improved': improved,
        }

    return await with_db(db, _query, dict(_TRAIN_THROUGHPUT_DEFAULT))


# ---------------------------------------------------------------------------
# 7. Per-project helpers
# ---------------------------------------------------------------------------


def merge_task_refs(project_root: str, rows: Iterable[Mapping[str, Any]]) -> set[TaskRef]:
    """The task each of *rows* names, for the ids that parse as one."""
    refs: set[TaskRef] = set()
    for row in rows:
        task_id = _task_id_of(row)
        if task_id is not None:
            refs.add(TaskRef(project_root, task_id))
    return refs


def _task_id_of(row: Mapping[str, Any]) -> int | None:
    raw = row.get('task_id')
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def enrich_merges_with_titles(
    rows: Sequence[Mapping[str, Any]],
    project_root: str,
    lookup: Mapping[TaskRef, Datum[dict]],
) -> list[dict]:
    """Copies of *rows*, each with its task's ``title`` as a ``Datum[str]``.

    The title carries its lookup's provenance unchanged: ``as_of``, state,
    reason and bound are the row datum's own. A row naming no task, or a task
    the lookup did not answer, gets an ``unknown`` title that says which.
    """
    titled: list[dict] = []
    for row in rows:
        task_id = _task_id_of(row)
        if task_id is None:
            title = _unknown_title(f'this merge row names no task id ({row.get("task_id")!r})')
        elif (found := lookup.get(TaskRef(project_root, task_id))) is None:
            title = _unknown_title(f'task {task_id} was not looked up')
        else:
            title = Datum(
                None if found.value is None else str(found.value.get('title') or ''),
                found.as_of, found.state, found.reason, found.freshness_bound_seconds,
            )
        titled.append({**row, 'title': title})
    return titled


def _unknown_title(reason: str) -> Datum[str]:
    return unknown_datum(reason, FETCHED_ROW_FRESHNESS_BOUND_SECONDS)


async def build_per_project_merge_queue(
    project_dbs: Sequence[tuple[str, aiosqlite.Connection | None]],
    *,
    hours: int,
    now: datetime,
) -> dict[str, dict]:
    """Build per-project merge queue stats by querying each project's DB independently.

    For each ``(pid, db)`` pair, gathers the per-DB stats concurrently. Pairs
    with ``db=None`` produce empty/default stats (the per-DB functions handle
    None gracefully by returning declared defaults). All per-project gathers
    also run concurrently across projects via a single top-level
    :func:`asyncio.gather`.

    ``recent`` is the newest :data:`RECENT_MERGES_CAP` merge_attempt rows in
    the same ``hours`` window every other leg uses, and ``recent_total`` is
    how many that window holds (see :func:`recent_merges`).

    Args:
        project_dbs: List of ``(project_root_str, connection_or_None)`` tuples
            from :func:`_project_scoped_dbs_labeled`.
        hours: Look-back window in hours (forwarded to each per-DB function).
        now: Shared reference timestamp captured once per request.

    Returns:
        Dict ``{pid: {depth_timeseries, outcomes, latency, recent, recent_total, speculative, train_events, train_throughput}}``.
    """
    _DEFAULT_DEPTH: ChartData = {'labels': [], 'values': []}
    _DEFAULT_SPEC = {'hit_count': 0, 'discard_count': 0, 'total': 0, 'hit_rate': 0.0}
    _DEFAULT_RECENT: RecentMerges = {'rows': [], 'total': 0}

    async def _one_project(pid: str, db: aiosqlite.Connection | None) -> tuple[str, dict]:
        try:
            depth_r, attempts_r, recent_r, spec_r, train_r, throughput_r = await asyncio.gather(
                queue_depth_timeseries(db, hours=hours, now=now),
                merge_attempts(db, hours=hours, now=now),
                recent_merges(db, limit=RECENT_MERGES_CAP, hours=hours, now=now),
                speculative_stats(db, hours=hours, now=now),
                recent_train_events(db, hours=hours, now=now),
                train_throughput_stats(db, hours=hours, now=now),
                return_exceptions=True,
            )
            depth = safe_gather_result(depth_r, _DEFAULT_DEPTH, f'{pid}/depth')
            attempts = safe_gather_result(attempts_r, MergeAttempts(), f'{pid}/attempts')
            recent = safe_gather_result(recent_r, _DEFAULT_RECENT, f'{pid}/recent')
            spec = safe_gather_result(spec_r, _DEFAULT_SPEC, f'{pid}/speculative')
            train_events_list = safe_gather_result(train_r, [], f'{pid}/train_events')
            train_throughput = safe_gather_result(throughput_r, dict(_TRAIN_THROUGHPUT_DEFAULT), f'{pid}/train_throughput')
            return pid, {
                'depth_timeseries': depth,
                'outcomes': attempts.outcome_chart(),
                'latency': attempts.latency(),
                'recent': recent['rows'],
                'recent_total': recent['total'],
                'speculative': spec,
                'train_events': train_events_list,
                'train_throughput': train_throughput,
            }
        except Exception as exc:
            logger.warning(
                'build_per_project_merge_queue %s: unexpected error (returning defaults): %s',
                pid,
                exc,
            )
            return pid, {
                'depth_timeseries': _DEFAULT_DEPTH,
                'outcomes': MergeAttempts().outcome_chart(),
                'latency': MergeAttempts().latency(),
                'recent': [],
                'recent_total': 0,
                'speculative': _DEFAULT_SPEC,
                'train_events': [],
                'train_throughput': dict(_TRAIN_THROUGHPUT_DEFAULT),
            }

    results = await asyncio.gather(*[_one_project(pid, db) for pid, db in project_dbs])
    return dict(results)


# ---------------------------------------------------------------------------
# 9. Live merge-queue fan-out (task-1606)
# ---------------------------------------------------------------------------

_LIVE_DEFAULT_PER_CALL_TIMEOUT = 2.0


async def _probe_live_one(
    client: httpx.AsyncClient,
    base_url: str,
    timeout: float,
) -> dict:
    """Probe one orchestrator's get_merge_queue tool; return a result dict.

    On transport/timeout failure (or when the tool result itself contains an
    'error' key, meaning the orchestrator/worker is not running), returns
    {entries: [], reachable: False, error: <message>}.  An authoritative empty
    queue (entries=[], no error) returns {entries: [], reachable: True}.

    *timeout* bounds the probe at two complementary layers: it is threaded
    into ``mcp_tool_call`` so it reaches ``client.post`` (bounding
    connect/read/write *and pool acquisition* on the shared client), while
    the enclosing ``asyncio.wait_for`` still bounds the operation as a whole
    — a cold session performs three posts, so the per-request layer alone
    would permit roughly 3x *timeout*.
    """
    try:
        result = await asyncio.wait_for(
            mcp_tool_call(client, base_url, 'get_merge_queue', {}, timeout=timeout),
            timeout=timeout,
        )
    except (TimeoutError, httpx.HTTPError, OSError, ValueError) as exc:
        logger.debug('get_merge_queue failed for %s: %s', base_url, exc)
        return {'entries': [], 'reachable': False, 'error': str(exc), 'metrics': None}
    # Guard against unexpected non-dict results (e.g. a list)
    if not isinstance(result, dict):
        return {'entries': [], 'reachable': False, 'error': 'unexpected result type', 'metrics': None}
    # Orchestrator/worker not running → result dict has an 'error' key
    if 'error' in result:
        return {'entries': [], 'reachable': False, 'error': result['error'], 'metrics': None}
    raw_entries = result.get('entries') or []
    return {
        'entries': [_normalize_entry(e) for e in raw_entries],
        'reachable': True,
        # ι=1894: carry through the live metrics emitted by SpeculativeMergeWorker
        'metrics': result.get('metrics'),
    }


async def fetch_live_merge_queues(
    client: httpx.AsyncClient,
    escalation_urls: dict[str, str],
    *,
    per_call_timeout: float = _LIVE_DEFAULT_PER_CALL_TIMEOUT,
) -> dict[str, dict]:
    """Fan out get_merge_queue to every escalation URL concurrently.

    Returns ``{project_label: {entries, reachable, [error]}}``.  Keys match
    the labels from ``config.escalation_urls`` (project basenames).  Failures
    (transport errors, timeouts, or orchestrator/worker down) produce
    ``{entries: [], reachable: False, error: ...}`` — an authoritative empty
    queue produces ``{entries: [], reachable: True}``.
    """
    if not escalation_urls:
        return {}
    labels = list(escalation_urls.keys())
    urls = [escalation_urls[lbl] for lbl in labels]
    base_urls = [u.removesuffix('/mcp').rstrip('/') for u in urls]
    results = await asyncio.gather(
        *(_probe_live_one(client, base, per_call_timeout) for base in base_urls),
        return_exceptions=False,
    )
    return dict(zip(labels, results, strict=True))


@dataclass(frozen=True, slots=True)
class ActiveQueue:
    """One project's "In queue now": the count as a Datum, and the live entries.

    ``entries`` is what the probe returned and nothing else — empty whenever
    the count is not a live reading. ``probe_configured`` is false for a
    project no live ``get_merge_queue`` probe exists for: it has no queue the
    dashboard can read, so it is outside every multi-project in-queue total
    rather than a permanent hole in one.
    """

    in_queue: Datum[int]
    entries: list[dict]
    probe_configured: bool


def resolve_active(
    label: str,
    live_map: Mapping[str, Mapping[str, Any]],
    history: Mapping[str, Sequence[Any]],
    *,
    now: datetime,
) -> ActiveQueue:
    """*label*'s queue: the live probe, else its own history's last sample.

    *live_map* holds one answer per configured probe
    (:func:`fetch_live_merge_queues`). A reachable probe is a ``fresh`` count
    at *now*, the probe instant. A failed probe serves the last
    ``merge_snapshots`` sample in *history* (``get_merge_active_series``,
    which records this same probe) ``stale`` at the sample's own instant, with
    the probe's error verbatim as the reason. With no parseable sample the
    count is ``unknown``. A label with no probe configured is ``unknown`` and
    reads no history: the sampler records only probes, so it has none.
    """
    live = live_map.get(label)
    if live is None:
        return ActiveQueue(
            unknown_datum(
                f'no live get_merge_queue probe is configured for {label}',
                LIVE_QUEUE_FRESHNESS_BOUND_SECONDS,
            ),
            [],
            probe_configured=False,
        )
    if live.get('reachable'):
        entries = list(live.get('entries') or [])
        return ActiveQueue(
            Datum(len(entries), now, DatumState.FRESH, None, LIVE_QUEUE_FRESHNESS_BOUND_SECONDS),
            entries,
            probe_configured=True,
        )
    why = f'the live get_merge_queue probe for {label} failed: {live.get("error")}'
    return ActiveQueue(_last_sample(history, why), [], probe_configured=True)


def _last_sample(history: Mapping[str, Sequence[Any]], why: str) -> Datum[int]:
    labels, values = history.get('labels') or (), history.get('values') or ()
    if not labels or not values:
        return unknown_datum(
            f'{why}; no sample in the history window', LIVE_QUEUE_FRESHNESS_BOUND_SECONDS,
        )
    try:
        sampled_at = parse_utc(labels[-1])
    except (TypeError, ValueError):
        return unknown_datum(
            f'{why}; the last sample has no readable instant ({labels[-1]!r})',
            LIVE_QUEUE_FRESHNESS_BOUND_SECONDS,
        )
    return Datum(
        int(values[-1]), sampled_at, DatumState.STALE,
        f'{why}; last sampled count shown', LIVE_QUEUE_FRESHNESS_BOUND_SECONDS,
    )


def _normalize_entry(raw: dict) -> dict:
    """Project a raw get_merge_queue snapshot entry to the display shape.

    Returns a dict with exactly six keys:
      task_id, branch, state, age_secs, position, waiter_alive.

    All fields are mandatory in the display contract; optional snapshot fields
    (worktree, pre_rebased, enqueued_at) are dropped.  Safe defaults prevent
    KeyError on partially-populated entries:
      - position  defaults to 0   (unknown position)
      - waiter_alive defaults to True  (assume waiter alive when unknown)
      - age_secs  defaults to 0.0
    """
    waiter_alive = raw.get('waiter_alive')
    position = raw.get('position')
    return {
        'task_id': raw.get('task_id'),
        'branch': raw.get('branch'),
        'state': raw.get('state'),
        'age_secs': float(raw.get('age_secs') or 0.0),
        'position': int(position) if position is not None else 0,
        'waiter_alive': bool(waiter_alive) if waiter_alive is not None else True,
    }

