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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TypedDict

import aiosqlite
import httpx

from dashboard.config import DashboardConfig
from dashboard.data.chart_utils import ChartData
from dashboard.data.db import with_db
from dashboard.data.mcp_fanout import TTLCache
from dashboard.data.memory import mcp_tool_call
from dashboard.data.stats_utils import percentile
from dashboard.data.tasks import DEFAULT_WHOLE_OPERATION_BUDGET, fetch_tasks
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

# _ACTIVE_ONLY mirrors the non-terminal members of orchestrator merge_types.OutcomeKind
# (_NON_TERMINAL_OUTCOMES in orchestrator/src/orchestrator/merge_types.py). The dashboard
# has NO dependency on the orchestrator package, so this is a hand-maintained mirror, not an
# import; the orchestrator-side frozen-contract test
# (tests/test_outcome_kind.py::TestOutcomeKindFrozenContract) is the drift tripwire. A latest
# merge_attempt event is TERMINAL (drops off the active panel) UNLESS its outcome is listed
# here — new/unknown terminal outcomes fail SAFE instead of phantoming for the full TTL.
# MAINTENANCE: no test or CI check enforces this mirror across the package boundary — if a
# change to orchestrator's _NON_TERMINAL_OUTCOMES lands, this frozenset must be updated to
# match by hand (see test_active_only_set_contents in test_merge_queue_data.py for the
# dashboard-side pin).
_ACTIVE_ONLY: frozenset[str] = frozenset({
    'cas_retry', 'gate_retry', 'post_merge_generation_chained', 'plan_files_narrowed',
})
_ACTIVE_EVENT_TYPES: tuple[str, ...] = ('merge_queued', 'merge_dequeued', 'merge_attempt')

# How many of a window's merge_attempt rows the Recent-merges table shows
# (PRD open question 5). Measured 19 / 228 / 982 / 4901 events at
# 24h / 7d / 30d / all, so the cap never bites at the default 24h and bites
# from 7d — which is when "showing N of M" carries information.
RECENT_MERGES_CAP = 200

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
    and excludes tasks whose latest event is a terminal merge_attempt outcome.
    A merge_attempt outcome is terminal UNLESS it is listed in _ACTIVE_ONLY
    (cas_retry, gate_retry, post_merge_generation_chained, plan_files_narrowed)
    — this fails safe: new or unrecognized outcomes drop off the active panel
    instead of phantoming as in_flight for the full TTL.

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

    effective_now = resolve_now(now)
    cutoff = (effective_now - timedelta(minutes=ttl_minutes)).isoformat()
    et_placeholders = ','.join('?' * len(_ACTIVE_EVENT_TYPES))

    async def _query(conn: aiosqlite.Connection) -> list[dict]:
        # Use ROW_NUMBER() window function for a single-pass plan that:
        # (a) avoids the O(N²) correlated-subquery scan, and
        # (b) deterministically picks one row per task_id when two events
        #     share an identical timestamp (ties broken by insertion order
        #     via id DESC, which is unique by AUTOINCREMENT).
        sql = f"""
            WITH ranked AS (
                SELECT task_id, run_id, event_type,
                       json_extract(data, '$.outcome') AS outcome,
                       json_extract(data, '$.branch') AS branch,
                       timestamp,
                       ROW_NUMBER() OVER (
                           PARTITION BY task_id
                           ORDER BY timestamp DESC, id DESC
                       ) AS rn
                FROM events
                WHERE event_type IN ({et_placeholders})
                  AND timestamp >= ?
            )
            SELECT task_id, run_id, event_type, outcome, branch, timestamp
            FROM ranked
            WHERE rn = 1
        """
        params = (*_ACTIVE_EVENT_TYPES, cutoff)
        rows = await conn.execute_fetchall(sql, params)

        result = []
        for row in rows:
            et = row['event_type']
            outcome = row['outcome']
            # Exclude terminal merge_attempt rows (terminal-unless-listed: fail safe)
            if et == 'merge_attempt' and outcome not in _ACTIVE_ONLY:
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


# Per-project TTL cache for load_task_titles.  Keyed on project_root_str.
# 10s window comfortably covers the dashboard's poll cadence (one refresh
# per few seconds) without introducing user-visible staleness on title
# lookups.  Cache is in-process; multi-worker deployments will each pay
# their own MCP roundtrip on first lookup.
_TASK_TITLES_TTL_SECONDS = 10.0

# Whole-operation bound for ``load_task_titles``, enforced with
# ``asyncio.wait_for``. Bound to the shared default rather than restating the
# literal, so the arithmetic lives in exactly one place; this site may later
# TIGHTEN its own constant (the structural test enforces it can never widen
# it). No whole-loop deadline is needed here: this is a single-root call whose
# fan-out happens at the CALLER via ``asyncio.gather``, so the handler cost is
# max-of-N rather than sum-of-N and one per-call budget already bounds the
# whole gather. (``discover_orchestrators`` used to be the contrasting case —
# a SEQUENTIAL per-root walk that needed a second, whole-loop bound. It reads
# no task tree since task 5587, so there is no longer a sibling to contrast
# with.)
_TASK_TITLES_BUDGET = DEFAULT_WHOLE_OPERATION_BUDGET

_task_titles_cache: TTLCache[dict[str, str] | None] = TTLCache(
    ttl_seconds=lambda: _TASK_TITLES_TTL_SECONDS
)


def _task_titles_cache_clear() -> None:
    """Clear the task-titles TTL cache (test/admin hook)."""
    _task_titles_cache.clear()


async def load_task_titles(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: str,
) -> dict[str, str]:
    """Return a ``{str(task_id): title}`` map for *project_root* via fused-memory MCP.

    Fetches the dashboard-shaped task list and projects out (id → title) for
    rows that have a non-empty title.  Results are cached per project_root
    for ``_TASK_TITLES_TTL_SECONDS`` (~10 s) so that the dashboard's per-poll
    enrichment doesn't hammer the MCP server.  An MCP failure returns ``{}``
    so the merge-queue tab still renders (titles fall back to empty strings).
    Concurrent cold callers for the same project_root collapse onto one
    in-flight fetch_tasks call (TTLCache single-flight).

    **Bounded as a whole.** The whole operation is bounded by
    ``_TASK_TITLES_BUDGET`` via ``asyncio.wait_for``. ``fetch_tasks``' own
    *timeout* is a PER-HTTP-REQUEST budget — it bounds connect/read/write and
    pool acquisition, never the operation as a whole — so without this layer a
    hang that opens no socket (a connection-pool lock, say) is unbounded, and
    that is exactly what wedged /merge-queue for 19.8 h. A timeout returns the
    SAME ``{}``, so titles degrade to empty strings rather than the tab 500ing
    or hanging, and nothing is written to the cache (the refresh never
    completed), so the next poll re-attempts and pays at most the budget
    again — a timeout can never pin an empty title map for the TTL window.

    The ``wait_for`` deliberately encloses ``get_or_refresh`` rather than the
    inner ``fetch_tasks``. ``TTLCache.get_or_refresh`` serializes cold callers
    for one key behind a per-key lock and runs the refresh WHILE HOLDING it,
    so an inner-only wrap would leave a QUEUED caller waiting unbounded for
    the holder's full budget before paying its own: the pair costs 2x and N
    waiters cost N x, and the dashboard's 3 s poll makes waiters routine.
    Enclosing the outer call bounds the lock wait too, and is safe —
    ``wait_for`` cancels the inner task, cancellation unwinds
    ``async with lock``, and ``__aexit__`` releases it rather than leaking it.

    The five-line ``wait_for``/``except TimeoutError``/warn/degrade construct
    below, and the lock-placement rationale above, are duplicated verbatim at
    the sibling call site (``app._load_task_cards``). That duplication is
    KNOWN and deliberate for now: the mechanism is a property of
    ``TTLCache`` — not of either call site — so the idiom belongs on
    ``dashboard/src/dashboard/data/mcp_fanout.py::TTLCache`` as a
    ``get_or_refresh_bounded`` that owns the timeout, the warning and the
    degraded return. That file is outside this change's lock set, so the
    extraction is left to the sibling TTLCache task referenced below.

    This bounds THIS caller only. It does not fix the general TTLCache
    queue-amplifier class across all of its call sites; that is the sibling
    task filed in the same batch.
    """

    async def _refresh() -> dict[str, str] | None:
        fetched = await fetch_tasks(client, config, project_root)
        if not isinstance(fetched, list):
            return None
        return {str(t['id']): t['title'] for t in fetched if t.get('title')}

    try:
        result = await asyncio.wait_for(
            _task_titles_cache.get_or_refresh(
                project_root, _refresh, cache_ok=lambda v: v is not None,
            ),
            timeout=_TASK_TITLES_BUDGET,
        )
    except TimeoutError:
        # Broader than the ``wait_for`` expiry, deliberately. On 3.11+
        # ``asyncio.TimeoutError`` IS the builtin, and ``socket.timeout`` is
        # too, so a ``TimeoutError`` raised INSIDE the refresh is folded into
        # this same budget path rather than 500ing the merge-queue tab. The
        # message below is therefore authoritative about the OUTCOME — the
        # titles are unknown for this poll — and not about the cause.
        logger.warning(
            'load_task_titles %s: exceeded the %.1fs whole-operation budget — '
            'merge rows render with empty titles for this poll (titles are '
            'UNKNOWN, not absent)',
            project_root, _TASK_TITLES_BUDGET,
        )
        return {}
    return dict(result) if isinstance(result, dict) else {}


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
        Dict ``{pid: {depth_timeseries, outcomes, latency, recent, recent_total, speculative, active, train_events, train_throughput}}``.
    """
    _DEFAULT_DEPTH: ChartData = {'labels': [], 'values': []}
    _DEFAULT_SPEC = {'hit_count': 0, 'discard_count': 0, 'total': 0, 'hit_rate': 0.0}
    _DEFAULT_RECENT: RecentMerges = {'rows': [], 'total': 0}

    async def _one_project(pid: str, db: aiosqlite.Connection | None) -> tuple[str, dict]:
        try:
            depth_r, attempts_r, recent_r, spec_r, active_r, train_r, throughput_r = await asyncio.gather(
                queue_depth_timeseries(db, hours=hours, now=now),
                merge_attempts(db, hours=hours, now=now),
                recent_merges(db, limit=RECENT_MERGES_CAP, hours=hours, now=now),
                speculative_stats(db, hours=hours, now=now),
                active_queued_merges(db, ttl_minutes=30, now=now),
                recent_train_events(db, hours=hours, now=now),
                train_throughput_stats(db, hours=hours, now=now),
                return_exceptions=True,
            )
            depth = safe_gather_result(depth_r, _DEFAULT_DEPTH, f'{pid}/depth')
            attempts = safe_gather_result(attempts_r, MergeAttempts(), f'{pid}/attempts')
            recent = safe_gather_result(recent_r, _DEFAULT_RECENT, f'{pid}/recent')
            spec = safe_gather_result(spec_r, _DEFAULT_SPEC, f'{pid}/speculative')
            active_list = safe_gather_result(active_r, [], f'{pid}/active')
            train_events_list = safe_gather_result(train_r, [], f'{pid}/train_events')
            train_throughput = safe_gather_result(throughput_r, dict(_TRAIN_THROUGHPUT_DEFAULT), f'{pid}/train_throughput')
            return pid, {
                'depth_timeseries': depth,
                'outcomes': attempts.outcome_chart(),
                'latency': attempts.latency(),
                'recent': recent['rows'],
                'recent_total': recent['total'],
                'speculative': spec,
                'active': active_list,
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
                'active': [],
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


def resolve_active(
    label: str,
    live_map: dict[str, dict],
    fallback_active: list[dict],
) -> dict:
    """Choose between the live queue snapshot and the event-derived fallback.

    Returns ``{entries, approximate}`` where:
      - ``entries``     is the chosen list of active-queue entries.
      - ``approximate`` is True when the entries come from the event-derived
                        fallback (orchestrator unreachable / not running).

    Selection logic:
      - If ``live_map[label]`` exists and ``reachable`` is True → use live
        entries (may be empty) with ``approximate=False``.
      - Otherwise (label absent or reachable=False) → use ``fallback_active``
        with ``approximate=True``.
    """
    live = live_map.get(label)
    if live is not None and live.get('reachable'):
        return {'entries': live['entries'], 'approximate': False}
    return {'entries': fallback_active, 'approximate': True}


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

