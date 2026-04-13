"""Tests for dashboard.data.merge_queue — merge queue query functions."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

# ---------------------------------------------------------------------------
# Schema — events table from orchestrator/src/orchestrator/event_store.py
# ---------------------------------------------------------------------------

MERGE_EVENTS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    run_id TEXT NOT NULL,
    task_id TEXT,
    event_type TEXT NOT NULL,
    phase TEXT,
    role TEXT,
    data TEXT DEFAULT '{}',
    cost_usd REAL,
    duration_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_task ON events(run_id, task_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_event(conn, *, event_type, timestamp, run_id='run-1', task_id=None,
                  phase='merge', data=None, duration_ms=None):
    """Insert a single event row into the events table."""
    if data is None:
        data = {}
    ts = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
    conn.execute(
        'INSERT INTO events (timestamp, run_id, task_id, event_type, phase, data, duration_ms) '
        'VALUES (?, ?, ?, ?, ?, ?, ?)',
        (ts, run_id, task_id, event_type, phase, json.dumps(data), duration_ms),
    )


def _bucket_start(t: datetime) -> datetime:
    """Return the start of the 15-min bucket containing t."""
    return t.replace(minute=(t.minute // 15) * 15, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def merge_events_db(tmp_path):
    """Empty-schema runs.db, ready to be populated per test."""
    db_path = tmp_path / 'runs.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(MERGE_EVENTS_SCHEMA)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def empty_merge_events_db(tmp_path):
    """Empty runs.db with schema only — no data."""
    db_path = tmp_path / 'empty_runs.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(MERGE_EVENTS_SCHEMA)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
async def empty_merge_events_conn(empty_merge_events_db):
    async with aiosqlite.connect(str(empty_merge_events_db)) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


# ---------------------------------------------------------------------------
# Imports under test (deferred so the test file fails gracefully before impl)
# ---------------------------------------------------------------------------

from dashboard.data.merge_queue import (  # noqa: E402
    _align_bucket,
    _bucket_minutes_for_window,
    _get_durations,
    aggregate_latency_stats,
    aggregate_outcome_distribution,
    aggregate_queue_depth_timeseries,
    aggregate_recent_merges,
    aggregate_speculative_stats,
    latency_stats,
    outcome_distribution,
    queue_depth_timeseries,
    recent_merges,
    speculative_stats,
)

# ---------------------------------------------------------------------------
# TestBucketMinutesForWindow
# ---------------------------------------------------------------------------

class TestBucketMinutesForWindow:
    @pytest.mark.parametrize('hours,expected', [
        (1, 15),
        (24, 15),
        (25, 60),
        (168, 60),
        (169, 360),
        (720, 360),
        (721, 1440),
        (87600, 1440),
    ])
    def test_ladder_tiers(self, hours, expected):
        assert _bucket_minutes_for_window(hours) == expected


# ---------------------------------------------------------------------------
# TestAlignBucket
# ---------------------------------------------------------------------------

class TestAlignBucket:
    def test_15min_alignment(self):
        """12:07:30 → 15-min bucket 12:00:00."""
        t = datetime(2026, 4, 11, 12, 7, 30, tzinfo=UTC)
        result = _align_bucket(t, 15)
        assert result == datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)

    def test_60min_alignment(self):
        """12:07:30 → 60-min bucket 12:00:00."""
        t = datetime(2026, 4, 11, 12, 7, 30, tzinfo=UTC)
        result = _align_bucket(t, 60)
        assert result == datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)

    def test_360min_alignment_at_12_07(self):
        """12:07:30 → 360-min (6h) bucket 12:00:00."""
        t = datetime(2026, 4, 11, 12, 7, 30, tzinfo=UTC)
        result = _align_bucket(t, 360)
        assert result == datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)

    def test_360min_alignment_at_14_07(self):
        """14:07:30 → 360-min (6h) bucket 12:00:00."""
        t = datetime(2026, 4, 11, 14, 7, 30, tzinfo=UTC)
        result = _align_bucket(t, 360)
        assert result == datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)

    def test_1440min_alignment_at_23_50(self):
        """23:50:00 → 1440-min (1d) bucket 00:00:00 same day."""
        t = datetime(2026, 4, 11, 23, 50, 0, tzinfo=UTC)
        result = _align_bucket(t, 1440)
        assert result == datetime(2026, 4, 11, 0, 0, 0, tzinfo=UTC)

    def test_1440min_alignment_at_00_02(self):
        """00:02:00 → 1440-min (1d) bucket 00:00:00 same day."""
        t = datetime(2026, 4, 11, 0, 2, 0, tzinfo=UTC)
        result = _align_bucket(t, 1440)
        assert result == datetime(2026, 4, 11, 0, 0, 0, tzinfo=UTC)

    def test_1440min_alignment_preserves_timezone(self):
        """Result is UTC-aware."""
        t = datetime(2026, 4, 11, 15, 30, 0, tzinfo=UTC)
        result = _align_bucket(t, 1440)
        assert result.tzinfo is not None
        assert result == datetime(2026, 4, 11, 0, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# TestQueueDepthTimeseries
# ---------------------------------------------------------------------------

class TestQueueDepthTimeseries:
    @pytest.mark.asyncio
    async def test_buckets_15min_over_24h(self, merge_events_db):
        """24h window produces 97 buckets; events fall in correct buckets."""
        now = datetime(2026, 4, 11, 12, 7, 30, tzinfo=UTC)
        cutoff = now - timedelta(hours=24)
        cutoff_aligned = _bucket_start(cutoff)

        # Two distinct bucket starts inside the 24h window
        bucket_a = cutoff_aligned + timedelta(hours=22)       # ~2h before now
        bucket_b = cutoff_aligned + timedelta(hours=21, minutes=45)  # different bucket

        conn_sync = sqlite3.connect(str(merge_events_db))
        for i in range(3):
            _insert_event(conn_sync, event_type='merge_attempt',
                          timestamp=bucket_a + timedelta(minutes=i + 1),
                          data={'outcome': 'done', 'attempt': 1})
        for i in range(2):
            _insert_event(conn_sync, event_type='merge_attempt',
                          timestamp=bucket_b + timedelta(minutes=i + 1),
                          data={'outcome': 'done', 'attempt': 1})
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await queue_depth_timeseries(db, hours=24, now=now)

        labels = result['labels']
        values = result['values']

        assert len(labels) == 97
        assert len(values) == 97
        assert all(v >= 0 for v in values)
        assert sum(values) == 5
        assert sum(1 for v in values if v > 0) == 2

        label_a = bucket_a.isoformat()
        label_b = bucket_b.isoformat()
        assert label_a in labels
        assert label_b in labels
        assert values[labels.index(label_a)] == 3
        assert values[labels.index(label_b)] == 2

    @pytest.mark.asyncio
    async def test_current_bucket_event_included(self, merge_events_db):
        """An event in the current 15-min bucket must appear in the output."""
        now = datetime(2026, 4, 11, 12, 7, 30, tzinfo=UTC)
        event_ts = datetime(2026, 4, 11, 12, 6, 0, tzinfo=UTC)
        expected_label = _bucket_start(now).isoformat()  # "2026-04-11T12:00:00+00:00"

        conn_sync = sqlite3.connect(str(merge_events_db))
        _insert_event(conn_sync, event_type='merge_attempt', timestamp=event_ts,
                      data={'outcome': 'done'})
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await queue_depth_timeseries(db, hours=24, now=now)

        assert expected_label in result['labels']
        idx = result['labels'].index(expected_label)
        assert result['values'][idx] == 1

    @pytest.mark.asyncio
    async def test_none_db(self):
        """None DB returns empty ChartData."""
        result = await queue_depth_timeseries(None, hours=24)
        assert result == {'labels': [], 'values': []}

    @pytest.mark.asyncio
    async def test_empty_db(self, empty_merge_events_conn):
        """No events → 97 buckets all with count 0."""
        now = datetime(2026, 4, 11, 12, 7, 30, tzinfo=UTC)
        result = await queue_depth_timeseries(empty_merge_events_conn, hours=24, now=now)
        assert len(result['labels']) == 97
        assert len(result['values']) == 97
        assert all(v == 0 for v in result['values'])

    @pytest.mark.asyncio
    async def test_7d_uses_1h_buckets(self, merge_events_db):
        """7d window (168h) produces 169 buckets spaced exactly 1 hour apart."""
        now = datetime(2026, 4, 11, 12, 7, 30, tzinfo=UTC)

        # Insert two events 12 hours before now (in the same 1h bucket)
        event_time = now - timedelta(hours=12)
        conn_sync = sqlite3.connect(str(merge_events_db))
        for i in range(2):
            _insert_event(conn_sync, event_type='merge_attempt',
                          timestamp=event_time + timedelta(minutes=i * 15),
                          data={'outcome': 'done'})
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await queue_depth_timeseries(db, hours=168, now=now)

        labels = result['labels']
        values = result['values']

        assert len(labels) == 169
        assert len(values) == 169

        # Consecutive labels must be exactly 1 hour apart
        for i in range(1, len(labels)):
            prev = datetime.fromisoformat(labels[i - 1])
            curr = datetime.fromisoformat(labels[i])
            assert curr - prev == timedelta(hours=1)

        # Both events fall in the same 1h bucket (floor of event_time to hour)
        bucket_label = _align_bucket(event_time, 60).isoformat()
        assert bucket_label in labels
        idx = labels.index(bucket_label)
        assert values[idx] == 2

    @pytest.mark.asyncio
    async def test_30d_uses_6h_buckets(self, merge_events_db):
        """30d window (720h) produces 121 buckets spaced exactly 6 hours apart."""
        now = datetime(2026, 4, 11, 12, 7, 30, tzinfo=UTC)

        # Insert one event ~5 days before now (inside a specific 6h bucket)
        event_time = now - timedelta(days=5, hours=2)
        conn_sync = sqlite3.connect(str(merge_events_db))
        _insert_event(conn_sync, event_type='merge_attempt',
                      timestamp=event_time,
                      data={'outcome': 'done'})
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await queue_depth_timeseries(db, hours=720, now=now)

        labels = result['labels']
        values = result['values']

        assert len(labels) == 121

        # Consecutive labels must be exactly 6 hours apart
        for i in range(1, len(labels)):
            prev = datetime.fromisoformat(labels[i - 1])
            curr = datetime.fromisoformat(labels[i])
            assert curr - prev == timedelta(hours=6)

        # Event falls in its 6h bucket
        bucket_label = _align_bucket(event_time, 360).isoformat()
        assert bucket_label in labels
        idx = labels.index(bucket_label)
        assert values[idx] == 1

    @pytest.mark.asyncio
    async def test_all_window_is_bounded(self, empty_merge_events_conn):
        """87600h window (window=all) produces exactly 3651 daily buckets — not 350k.

        The exact count is deterministic given the fixed ``now`` value:
        - now_aligned   = 2026-04-11T00:00:00 UTC
        - cutoff_aligned = 2016-04-14T00:00:00 UTC  (floor of now − 87600 h)
        - diff = 3650 days → 3650 + 1 = 3651 buckets
        """
        now = datetime(2026, 4, 11, 12, 7, 30, tzinfo=UTC)
        result = await queue_depth_timeseries(empty_merge_events_conn, hours=87600, now=now)

        labels = result['labels']
        assert len(labels) == 3651, (
            f"Expected exactly 3651 daily buckets, got {len(labels)} "
            f"(regression guard against the 350 401-bucket blowup)"
        )

        # ALL consecutive labels must be exactly 1 day apart (UTC, no DST drift)
        for i in range(1, len(labels)):
            prev = datetime.fromisoformat(labels[i - 1])
            curr = datetime.fromisoformat(labels[i])
            assert curr - prev == timedelta(days=1), (
                f"Gap at index {i}: {prev.isoformat()} → {curr.isoformat()} "
                f"is not exactly 1 day"
            )

    @pytest.mark.asyncio
    async def test_first_bucket_includes_events_before_cutoff(self, merge_events_db):
        """Events in [cutoff_aligned, cutoff) must be counted in the first bucket.

        With now=2026-04-11T12:07:30 UTC and hours=24:
          cutoff         = 2026-04-10T12:07:30+00:00
          cutoff_aligned = 2026-04-10T12:00:00+00:00  (floor to 15-min boundary)

        An event at cutoff_aligned + 1 minute (12:01:00) is inside the first
        bucket's time range but falls before cutoff (12:07:30).  The SQL WHERE
        clause must use cutoff_aligned (not cutoff) so this event is fetched.
        """
        now = datetime(2026, 4, 11, 12, 7, 30, tzinfo=UTC)
        bucket_min = _bucket_minutes_for_window(24)  # 15 for a 24h window
        cutoff = now - timedelta(hours=24)
        cutoff_aligned = _align_bucket(cutoff, bucket_min)

        # Event is 1 minute after the first bucket boundary and before cutoff
        event_ts = cutoff_aligned + timedelta(minutes=1)
        assert event_ts < cutoff, "Pre-condition: event must be before cutoff"

        conn_sync = sqlite3.connect(str(merge_events_db))
        _insert_event(conn_sync, event_type='merge_attempt', timestamp=event_ts,
                      data={'outcome': 'done'})
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await queue_depth_timeseries(db, hours=24, now=now)

        first_label = cutoff_aligned.isoformat()
        assert first_label in result['labels'], (
            f"Expected first bucket label {first_label!r} in labels"
        )
        idx = result['labels'].index(first_label)
        assert result['values'][idx] == 1, (
            f"First bucket should have count=1 for the event at {event_ts.isoformat()}, "
            f"got {result['values'][idx]}"
        )


# ---------------------------------------------------------------------------
# TestOutcomeDistribution
# ---------------------------------------------------------------------------

class TestOutcomeDistribution:
    @pytest.mark.asyncio
    async def test_populated(self, merge_events_db):
        """Outcome counts match inserted data; canonical order; sum correct."""
        now = datetime.now(UTC)
        conn_sync = sqlite3.connect(str(merge_events_db))
        outcomes = ['done'] * 3 + ['conflict'] * 2 + ['blocked'] * 1 + ['already_merged'] * 1
        for outcome in outcomes:
            _insert_event(conn_sync, event_type='merge_attempt',
                          timestamp=now - timedelta(minutes=10),
                          data={'outcome': outcome, 'attempt': 1})
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await outcome_distribution(db, hours=24)

        assert sum(result['values']) == 7
        assert 'done' in result['labels']
        assert 'conflict' in result['labels']
        assert 'blocked' in result['labels']
        assert 'already_merged' in result['labels']

        idx_done = result['labels'].index('done')
        assert result['values'][idx_done] == 3
        idx_conflict = result['labels'].index('conflict')
        assert result['values'][idx_conflict] == 2

    @pytest.mark.asyncio
    async def test_canonical_order(self, merge_events_db):
        """Canonical outcomes appear first in deterministic order."""
        now = datetime.now(UTC)
        conn_sync = sqlite3.connect(str(merge_events_db))
        for outcome in ['already_merged', 'done', 'conflict', 'blocked']:
            _insert_event(conn_sync, event_type='merge_attempt',
                          timestamp=now - timedelta(minutes=5),
                          data={'outcome': outcome})
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await outcome_distribution(db, hours=24)

        # First four labels must be the canonical ones in canonical order
        assert result['labels'][:4] == ['done', 'conflict', 'blocked', 'already_merged']

    @pytest.mark.asyncio
    async def test_unknown_outcome_included(self, merge_events_db):
        """Unknown outcomes (e.g. 'wip_halted') appear after canonical ones."""
        now = datetime.now(UTC)
        conn_sync = sqlite3.connect(str(merge_events_db))
        for outcome in ['done', 'wip_halted', 'done_wip_recovery']:
            _insert_event(conn_sync, event_type='merge_attempt',
                          timestamp=now - timedelta(minutes=5),
                          data={'outcome': outcome})
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await outcome_distribution(db, hours=24)

        assert 'wip_halted' in result['labels']
        assert 'done_wip_recovery' in result['labels']
        # done is first (canonical)
        assert result['labels'][0] == 'done'

    @pytest.mark.asyncio
    async def test_none_db(self):
        result = await outcome_distribution(None, hours=24)
        assert result == {'labels': [], 'values': []}

    @pytest.mark.asyncio
    async def test_empty_db(self, empty_merge_events_conn):
        result = await outcome_distribution(empty_merge_events_conn, hours=24)
        assert result['labels'] == []
        assert result['values'] == []


# ---------------------------------------------------------------------------
# TestLatencyStats
# ---------------------------------------------------------------------------

class TestLatencyStats:
    @pytest.mark.asyncio
    async def test_populated(self, merge_events_db):
        """Percentiles and mean computed correctly for known duration set."""
        now = datetime.now(UTC)
        conn_sync = sqlite3.connect(str(merge_events_db))
        # 10 events: 100, 200, ..., 1000 ms
        for i, ms in enumerate(range(100, 1100, 100)):
            _insert_event(conn_sync, event_type='merge_attempt',
                          timestamp=now - timedelta(minutes=10 + i),
                          data={'outcome': 'done', 'attempt': 1},
                          duration_ms=ms)
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await latency_stats(db, hours=24)

        assert result['count'] == 10
        assert result['mean_ms'] == pytest.approx(550.0, abs=1e-6)
        assert result['p50'] == pytest.approx(550.0, abs=1.0)
        assert result['p95'] > result['p50']
        assert result['p99'] >= result['p95']
        assert {'p50', 'p95', 'p99', 'count', 'mean_ms'} == set(result.keys())

    @pytest.mark.asyncio
    async def test_none_db(self):
        result = await latency_stats(None, hours=24)
        assert result == {'p50': 0, 'p95': 0, 'p99': 0, 'count': 0, 'mean_ms': 0.0}

    @pytest.mark.asyncio
    async def test_empty_db(self, empty_merge_events_conn):
        result = await latency_stats(empty_merge_events_conn, hours=24)
        assert result == {'p50': 0, 'p95': 0, 'p99': 0, 'count': 0, 'mean_ms': 0.0}

    @pytest.mark.asyncio
    async def test_all_null_durations(self, merge_events_db):
        """Rows present but duration_ms NULL → count=0, percentiles=0."""
        now = datetime.now(UTC)
        conn_sync = sqlite3.connect(str(merge_events_db))
        for i in range(3):
            _insert_event(conn_sync, event_type='merge_attempt',
                          timestamp=now - timedelta(minutes=i + 1),
                          data={'outcome': 'done'}, duration_ms=None)
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await latency_stats(db, hours=24)

        assert result == {'p50': 0, 'p95': 0, 'p99': 0, 'count': 0, 'mean_ms': 0.0}

    @pytest.mark.asyncio
    async def test_get_durations_returns_sorted(self, merge_events_db):
        """_get_durations returns a sorted list even when inserted in non-sorted order.

        Establishes the sorted-output invariant of _get_durations, which latency_stats
        relies on to avoid a redundant sorted() call.
        """
        now = datetime.now(UTC)
        conn_sync = sqlite3.connect(str(merge_events_db))
        for ms in [500, 100, 300, 200, 400]:
            _insert_event(conn_sync, event_type='merge_attempt',
                          timestamp=now - timedelta(minutes=10),
                          data={'outcome': 'done'},
                          duration_ms=ms)
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await _get_durations(db, hours=24)

        assert result == sorted([500.0, 100.0, 300.0, 200.0, 400.0])


# ---------------------------------------------------------------------------
# TestRecentMerges
# ---------------------------------------------------------------------------

class TestRecentMerges:
    @pytest.mark.asyncio
    async def test_populated(self, merge_events_db):
        """recent_merges returns up to limit rows, newest first."""
        now = datetime.now(UTC)
        conn_sync = sqlite3.connect(str(merge_events_db))
        for i in range(25):
            _insert_event(conn_sync, event_type='merge_attempt',
                          timestamp=now - timedelta(minutes=25 - i),
                          task_id=f'task-{i:03d}', run_id=f'run-{i:03d}',
                          data={'outcome': 'done' if i % 2 == 0 else 'conflict', 'attempt': 1},
                          duration_ms=1000 + i * 100)
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await recent_merges(db, limit=20)

        assert len(result) == 20
        # Ordered by timestamp DESC (newest first → task-024 is first)
        assert result[0]['task_id'] == 'task-024'
        # Keys present
        assert {'task_id', 'outcome', 'duration_ms', 'timestamp', 'run_id'} <= set(result[0].keys())

    @pytest.mark.asyncio
    async def test_none_db(self):
        result = await recent_merges(None, limit=20)
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_db(self, empty_merge_events_conn):
        result = await recent_merges(empty_merge_events_conn, limit=20)
        assert result == []

    @pytest.mark.asyncio
    async def test_custom_limit(self, merge_events_db):
        """limit=5 returns 5 rows."""
        now = datetime.now(UTC)
        conn_sync = sqlite3.connect(str(merge_events_db))
        for i in range(10):
            _insert_event(conn_sync, event_type='merge_attempt',
                          timestamp=now - timedelta(minutes=i + 1),
                          data={'outcome': 'done'})
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await recent_merges(db, limit=5)

        assert len(result) == 5

    @pytest.mark.xfail(
        reason=(
            "Known limitation: SQLite uses lexicographic string comparison for "
            "the 'timestamp >= ?' filter.  A timestamp stored with a large "
            "positive offset (e.g. +14:00) can appear after the UTC cutoff "
            "string even though its UTC-equivalent is before the cutoff.  "
            "Correct fix: normalise timestamps to UTC on write."
        ),
        strict=False,
    )
    @pytest.mark.asyncio
    async def test_recent_merges_sql_string_comparison_tz_limitation(self, merge_events_db):
        """Non-UTC offsets can bypass the SQL hours-window filter (known limitation).

        The ``AND timestamp >= ?`` clause in ``recent_merges`` compares stored
        timestamp strings against ``_cutoff_iso()`` (a UTC string) using
        SQLite's lexicographic ordering.  An event stored with offset ``+14:00``
        has a local-time component that is 14 hours ahead, so its string
        representation can sort *after* the cutoff string even though the
        event's UTC-equivalent time is *before* the cutoff.

        Example (all UTC):
            now        = T
            cutoff     = T - 1h  →  stored as  '...T(h-1):MM:SS+00:00'
            event (UTC) = T - 3h  →  stored as  '...(next day)T01:MM:SS+14:00'
            SQLite: next-day string > today string  →  INCLUDED  (wrong)
            Correct:    T-3h < T-1h                →  EXCLUDED

        This test asserts the *correct* behaviour (0 results) and is marked
        ``xfail`` because the current implementation will include the event.
        When the underlying limitation is fixed this test will pass.
        """
        now = datetime.now(UTC)
        event_utc = now - timedelta(hours=3)  # 3 h before now → before 1-h cutoff
        # Represent the same moment in +14:00 local time.
        # (event_utc + 14h) gives the local clock reading; appending '+14:00'
        # produces a valid ISO-8601 string whose UTC-equivalent == event_utc.
        local_dt = event_utc + timedelta(hours=14)
        ts_non_utc = local_dt.strftime('%Y-%m-%dT%H:%M:%S') + '+14:00'

        conn_sync = sqlite3.connect(str(merge_events_db))
        _insert_event(conn_sync, event_type='merge_attempt', timestamp=ts_non_utc,
                      task_id='old-non-utc', run_id='run-tz',
                      data={'outcome': 'done'})
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await recent_merges(db, limit=20, hours=1)

        # The event is 3 h old in UTC — well outside the 1-h window.
        # With correct UTC comparison it should not appear; with string
        # comparison it is incorrectly included.
        assert len(result) == 0, (
            f"Event stored as '{ts_non_utc}' (= event_utc {event_utc.isoformat()}) "
            "was included by the 1-hour filter despite being 3 h before the cutoff. "
            "This is the known SQLite string-comparison limitation."
        )

    @pytest.mark.asyncio
    async def test_hours_window_excludes_old_events(self, merge_events_db):
        """recent_merges with hours=1 excludes events older than 1 hour."""
        now = datetime.now(UTC)
        conn_sync = sqlite3.connect(str(merge_events_db))
        # 3 events at now-30min (within the 1-hour window)
        for i in range(3):
            _insert_event(conn_sync, event_type='merge_attempt',
                          timestamp=now - timedelta(minutes=30 + i),
                          task_id=f'recent-{i}', run_id=f'run-r{i}',
                          data={'outcome': 'done'})
        # 2 events at now-3hours (outside the 1-hour window)
        for i in range(2):
            _insert_event(conn_sync, event_type='merge_attempt',
                          timestamp=now - timedelta(hours=3 + i),
                          task_id=f'old-{i}', run_id=f'run-o{i}',
                          data={'outcome': 'done'})
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await recent_merges(db, limit=20, hours=1)

        assert len(result) == 3
        task_ids = [r['task_id'] for r in result]
        assert all(tid.startswith('recent-') for tid in task_ids)


# ---------------------------------------------------------------------------
# TestSpeculativeStats
# ---------------------------------------------------------------------------

class TestSpeculativeStats:
    @pytest.mark.asyncio
    async def test_populated(self, merge_events_db):
        """hit_count, discard_count, total, hit_rate computed correctly."""
        now = datetime.now(UTC)
        conn_sync = sqlite3.connect(str(merge_events_db))
        for i in range(5):
            _insert_event(conn_sync, event_type='speculative_merge',
                          timestamp=now - timedelta(minutes=i + 1),
                          data={'base_sha': f'abc{i}'})
        for i in range(2):
            _insert_event(conn_sync, event_type='speculative_discard',
                          timestamp=now - timedelta(minutes=i + 10),
                          data={'reason': 'previous_failed'})
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await speculative_stats(db, hours=24)

        assert result['hit_count'] == 5
        assert result['discard_count'] == 2
        assert result['total'] == 7
        assert result['hit_rate'] == pytest.approx(5 / 7, abs=1e-6)

    @pytest.mark.asyncio
    async def test_none_db(self):
        result = await speculative_stats(None, hours=24)
        assert result == {'hit_count': 0, 'discard_count': 0, 'total': 0, 'hit_rate': 0.0}

    @pytest.mark.asyncio
    async def test_empty_db(self, empty_merge_events_conn):
        result = await speculative_stats(empty_merge_events_conn, hours=24)
        assert result == {'hit_count': 0, 'discard_count': 0, 'total': 0, 'hit_rate': 0.0}

    @pytest.mark.asyncio
    async def test_all_hits(self, merge_events_db):
        """All speculative_merge → hit_rate=1.0."""
        now = datetime.now(UTC)
        conn_sync = sqlite3.connect(str(merge_events_db))
        for i in range(3):
            _insert_event(conn_sync, event_type='speculative_merge',
                          timestamp=now - timedelta(minutes=i + 1),
                          data={'base_sha': f'sha{i}'})
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await speculative_stats(db, hours=24)

        assert result['hit_rate'] == pytest.approx(1.0)
        assert result['discard_count'] == 0

    @pytest.mark.asyncio
    async def test_all_discards(self, merge_events_db):
        """All speculative_discard → hit_rate=0.0."""
        now = datetime.now(UTC)
        conn_sync = sqlite3.connect(str(merge_events_db))
        for i in range(3):
            _insert_event(conn_sync, event_type='speculative_discard',
                          timestamp=now - timedelta(minutes=i + 1),
                          data={'reason': 'chain_invalidated'})
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await speculative_stats(db, hours=24)

        assert result['hit_rate'] == pytest.approx(0.0)
        assert result['hit_count'] == 0


# ---------------------------------------------------------------------------
# TestMultiDbAggregation
# ---------------------------------------------------------------------------

class TestMultiDbAggregation:
    def _make_db(self, tmp_path, name, events):
        """Create a populated DB with given events list of dicts."""
        db_path = tmp_path / name
        conn = sqlite3.connect(str(db_path))
        conn.executescript(MERGE_EVENTS_SCHEMA)
        for evt in events:
            _insert_event(conn, **evt)
        conn.commit()
        conn.close()
        return db_path

    def _make_broken_mock(self):
        """Return a mock connection that raises on any real method call.

        The mock just needs to cause *some* exception that escapes with_db() —
        the exact failure point doesn't matter.  asyncio.gather(return_exceptions=True)
        catches BaseException at the coroutine level, so any unhandled exception
        (TypeError from awaiting a non-async attribute, RuntimeError from
        execute_fetchall, etc.) is silently skipped by the aggregators.
        """
        broken = MagicMock()
        broken.execute_fetchall = AsyncMock(side_effect=RuntimeError('simulated broken DB'))
        return broken

    async def _run_with_one_broken_db(self, tmp_path, aggregate_fn, events, **kwargs):
        """Scaffold for resilience tests: one valid DB + one broken mock connection."""
        db1 = self._make_db(tmp_path, 'runs1.db', events)
        broken = self._make_broken_mock()
        async with aiosqlite.connect(str(db1)) as conn1:
            conn1.row_factory = aiosqlite.Row
            return await aggregate_fn([conn1, broken], **kwargs)

    async def _run_with_none_entry(self, tmp_path, aggregate_fn, events, **kwargs):
        """Scaffold for None-entry tests: one valid DB + None in dbs list."""
        db1 = self._make_db(tmp_path, 'runs1.db', events)
        async with aiosqlite.connect(str(db1)) as conn1:
            conn1.row_factory = aiosqlite.Row
            return await aggregate_fn([conn1, None], **kwargs)

    @pytest.mark.asyncio
    async def test_aggregate_queue_depth_timeseries(self, tmp_path):
        """Counts per bucket sum across two DBs."""
        now = datetime(2026, 4, 11, 12, 7, 30, tzinfo=UTC)
        cutoff = now - timedelta(hours=24)
        bucket = _bucket_start(cutoff) + timedelta(hours=22)

        db1 = self._make_db(tmp_path, 'runs1.db', [
            {'event_type': 'merge_attempt', 'timestamp': bucket + timedelta(minutes=1),
             'data': {'outcome': 'done'}},
            {'event_type': 'merge_attempt', 'timestamp': bucket + timedelta(minutes=2),
             'data': {'outcome': 'done'}},
        ])
        db2 = self._make_db(tmp_path, 'runs2.db', [
            {'event_type': 'merge_attempt', 'timestamp': bucket + timedelta(minutes=3),
             'data': {'outcome': 'conflict'}},
        ])

        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await aggregate_queue_depth_timeseries([conn1, conn2], hours=24, now=now)

        assert len(result['labels']) == 97
        bucket_label = bucket.isoformat()
        assert bucket_label in result['labels']
        idx = result['labels'].index(bucket_label)
        assert result['values'][idx] == 3  # 2 + 1

    @pytest.mark.asyncio
    async def test_aggregator_uses_consistent_now_across_dbs(self, tmp_path):
        """Aggregator with explicit now threads same timestamp to all per-DB calls."""
        now = datetime(2026, 4, 11, 12, 7, 30, tzinfo=UTC)
        current_bucket_label = _bucket_start(now).isoformat()  # "2026-04-11T12:00:00+00:00"

        db1 = self._make_db(tmp_path, 'runs1.db', [
            {'event_type': 'merge_attempt',
             'timestamp': now - timedelta(seconds=30),
             'data': {'outcome': 'done'}},
        ])
        db2 = self._make_db(tmp_path, 'runs2.db', [
            {'event_type': 'merge_attempt',
             'timestamp': now - timedelta(seconds=45),
             'data': {'outcome': 'done'}},
        ])

        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await aggregate_queue_depth_timeseries([conn1, conn2], hours=24, now=now)

        assert current_bucket_label in result['labels']
        idx = result['labels'].index(current_bucket_label)
        assert result['values'][idx] == 2  # 1 from each DB

    @pytest.mark.asyncio
    async def test_aggregator_labels_stable_skewed_now(self, tmp_path, monkeypatch):
        """Aggregator captures datetime.now() once and threads it to all per-DB calls.

        Without the effective_now capture at the top of aggregate_queue_depth_timeseries,
        a stepping datetime.now() that straddles the 12:00→12:15 bucket boundary would
        cause per-DB subcalls to receive different now values, diverging their label sets.
        The strengthened test verifies: (a) both subcalls receive the same pre-boundary
        now, (b) datetime.now was called exactly once (aggregator-level capture, not
        per-DB), and (c) labels use the pre-boundary 15-min bucket alignment.
        """
        from dashboard.data import merge_queue

        # Stepping clock: first call → 12:14:59 (pre-boundary),
        # subsequent calls → 12:15:01 (post-boundary, should never be reached).
        before = datetime(2026, 4, 11, 12, 14, 59, tzinfo=UTC)
        after = datetime(2026, 4, 11, 12, 15, 1, tzinfo=UTC)

        class _SteppingDT(datetime):
            _call_count = 0

            @classmethod
            def now(cls, tz=None):
                assert tz is UTC, f'Expected UTC, got {tz}'
                val = before if cls._call_count == 0 else after
                cls._call_count += 1
                return val

        monkeypatch.setattr(merge_queue, 'datetime', _SteppingDT)

        # Async spy: record the `now` kwarg received by each per-DB subcall.
        received_nows: list = []
        _real = queue_depth_timeseries

        async def _spy(db, *, hours, now=None):
            received_nows.append(now)
            return await _real(db, hours=hours, now=now)

        monkeypatch.setattr(merge_queue, 'queue_depth_timeseries', _spy)

        db1 = self._make_db(tmp_path, 'runs1.db', [])
        db2 = self._make_db(tmp_path, 'runs2.db', [])

        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await aggregate_queue_depth_timeseries([conn1, conn2], hours=24)

        # (a) Both per-DB subcalls received the same pre-boundary now.
        assert len(received_nows) == 2
        assert received_nows[0] == before
        assert received_nows[1] == before

        # (b) datetime.now was called exactly once: aggregator-level capture, not per-DB.
        assert _SteppingDT._call_count == 1

        # (c) Labels use the pre-boundary 15-min bucket: 12:14:59 → bucket 12:00:00.
        assert result['labels'][-1] == '2026-04-11T12:00:00+00:00'

    @pytest.mark.asyncio
    async def test_aggregate_outcome_distribution(self, tmp_path):
        """Outcome counts sum across two DBs."""
        now = datetime.now(UTC)
        db1 = self._make_db(tmp_path, 'runs1.db', [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=5),
             'data': {'outcome': 'done'}},
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=6),
             'data': {'outcome': 'done'}},
        ])
        db2 = self._make_db(tmp_path, 'runs2.db', [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=7),
             'data': {'outcome': 'done'}},
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=8),
             'data': {'outcome': 'conflict'}},
        ])

        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await aggregate_outcome_distribution([conn1, conn2], hours=24)

        assert 'done' in result['labels']
        done_idx = result['labels'].index('done')
        assert result['values'][done_idx] == 3  # 2 + 1

    @pytest.mark.asyncio
    async def test_aggregate_latency_stats(self, tmp_path):
        """Latency stats recomputed from merged durations."""
        now = datetime.now(UTC)
        db1 = self._make_db(tmp_path, 'runs1.db', [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=5),
             'data': {'outcome': 'done'}, 'duration_ms': 100},
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=6),
             'data': {'outcome': 'done'}, 'duration_ms': 200},
        ])
        db2 = self._make_db(tmp_path, 'runs2.db', [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=7),
             'data': {'outcome': 'done'}, 'duration_ms': 300},
        ])

        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await aggregate_latency_stats([conn1, conn2], hours=24)

        assert result['count'] == 3
        assert result['mean_ms'] == pytest.approx(200.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_aggregate_recent_merges(self, tmp_path):
        """Recent merges concatenated, sorted by timestamp DESC, truncated."""
        now = datetime.now(UTC)
        db1 = self._make_db(tmp_path, 'runs1.db', [
            {'event_type': 'merge_attempt',
             'timestamp': now - timedelta(minutes=1), 'task_id': 'task-a',
             'data': {'outcome': 'done'}},
        ])
        db2 = self._make_db(tmp_path, 'runs2.db', [
            {'event_type': 'merge_attempt',
             'timestamp': now - timedelta(minutes=2), 'task_id': 'task-b',
             'data': {'outcome': 'done'}},
        ])

        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await aggregate_recent_merges([conn1, conn2], limit=20)

        assert len(result) == 2
        # Newest first
        assert result[0]['task_id'] == 'task-a'
        assert result[1]['task_id'] == 'task-b'

    @pytest.mark.asyncio
    async def test_aggregate_speculative_stats(self, tmp_path):
        """Speculative counts summed, hit_rate recomputed."""
        now = datetime.now(UTC)
        db1 = self._make_db(tmp_path, 'runs1.db', [
            {'event_type': 'speculative_merge',
             'timestamp': now - timedelta(minutes=1), 'data': {'base_sha': 'abc'}},
            {'event_type': 'speculative_merge',
             'timestamp': now - timedelta(minutes=2), 'data': {'base_sha': 'def'}},
        ])
        db2 = self._make_db(tmp_path, 'runs2.db', [
            {'event_type': 'speculative_discard',
             'timestamp': now - timedelta(minutes=3),
             'data': {'reason': 'previous_failed'}},
        ])

        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await aggregate_speculative_stats([conn1, conn2], hours=24)

        assert result['hit_count'] == 2
        assert result['discard_count'] == 1
        assert result['total'] == 3
        assert result['hit_rate'] == pytest.approx(2 / 3, abs=1e-6)

    @pytest.mark.asyncio
    async def test_aggregate_7d_uses_1h_buckets(self, tmp_path):
        """aggregate_queue_depth_timeseries at hours=168 produces 169 buckets,
        summing counts across two DBs into the same 1h bucket."""
        now = datetime(2026, 4, 11, 12, 7, 30, tzinfo=UTC)
        event_time = now - timedelta(hours=12)  # 12h before now

        db1 = self._make_db(tmp_path, 'runs1.db', [
            {'event_type': 'merge_attempt',
             'timestamp': event_time + timedelta(minutes=5),
             'data': {'outcome': 'done'}},
        ])
        db2 = self._make_db(tmp_path, 'runs2.db', [
            {'event_type': 'merge_attempt',
             'timestamp': event_time + timedelta(minutes=10),
             'data': {'outcome': 'conflict'}},
            {'event_type': 'merge_attempt',
             'timestamp': event_time + timedelta(minutes=15),
             'data': {'outcome': 'conflict'}},
        ])

        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await aggregate_queue_depth_timeseries([conn1, conn2], hours=168, now=now)

        labels = result['labels']
        values = result['values']

        assert len(labels) == 169

        # All 3 events share the same 1h bucket
        bucket_label = _align_bucket(event_time, 60).isoformat()
        assert bucket_label in labels
        idx = labels.index(bucket_label)
        assert values[idx] == 3  # 1 + 2

    @pytest.mark.asyncio
    async def test_aggregate_all_window_bounded(self, tmp_path):
        """aggregate_queue_depth_timeseries at hours=87600 produces ≤4000 buckets."""
        now = datetime(2026, 4, 11, 12, 7, 30, tzinfo=UTC)

        db1 = self._make_db(tmp_path, 'runs1.db', [])
        db2 = self._make_db(tmp_path, 'runs2.db', [])

        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await aggregate_queue_depth_timeseries([conn1, conn2], hours=87600, now=now)

        assert len(result['labels']) >= 3650
        assert len(result['labels']) <= 4000

    @pytest.mark.asyncio
    async def test_aggregate_recent_merges_hours_window(self, tmp_path):
        """aggregate_recent_merges with hours=1 excludes events older than 1 hour."""
        now = datetime.now(UTC)
        db1 = self._make_db(tmp_path, 'runs1.db', [
            {'event_type': 'merge_attempt',
             'timestamp': now - timedelta(minutes=30), 'task_id': 'recent-a',
             'data': {'outcome': 'done'}},
            {'event_type': 'merge_attempt',
             'timestamp': now - timedelta(hours=3), 'task_id': 'old-a',
             'data': {'outcome': 'done'}},
        ])
        db2 = self._make_db(tmp_path, 'runs2.db', [
            {'event_type': 'merge_attempt',
             'timestamp': now - timedelta(minutes=45), 'task_id': 'recent-b',
             'data': {'outcome': 'done'}},
            {'event_type': 'merge_attempt',
             'timestamp': now - timedelta(hours=4), 'task_id': 'old-b',
             'data': {'outcome': 'done'}},
        ])

        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await aggregate_recent_merges([conn1, conn2], limit=20, hours=1)

        assert len(result) == 2
        task_ids = {r['task_id'] for r in result}
        assert task_ids == {'recent-a', 'recent-b'}

    @pytest.mark.asyncio
    async def test_aggregate_recent_merges_mixed_tz_sort(self, tmp_path):
        """Merges with different UTC offsets sort correctly by chronological order.

        Lexicographic sort would compare '13:00+05:00' > '12:00+00:00' (wrong).
        Chronological sort: 12:00 UTC > 08:00 UTC (13:00+05:00), so UTC noon
        must appear first (newest).
        """
        # DB1: 2026-04-11T12:00:00+00:00  = 12:00 UTC (newer)
        # DB2: 2026-04-11T13:00:00+05:00  = 08:00 UTC (older)
        ts_utc_noon = '2026-04-11T12:00:00+00:00'
        ts_tz_plus5 = '2026-04-11T13:00:00+05:00'

        db1 = self._make_db(tmp_path, 'runs1.db', [
            {'event_type': 'merge_attempt', 'timestamp': ts_utc_noon,
             'task_id': 'utc-noon', 'data': {'outcome': 'done'}},
        ])
        db2 = self._make_db(tmp_path, 'runs2.db', [
            {'event_type': 'merge_attempt', 'timestamp': ts_tz_plus5,
             'task_id': 'tz-plus5', 'data': {'outcome': 'done'}},
        ])

        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            # Use a large hours window so both events are included
            result = await aggregate_recent_merges([conn1, conn2], limit=20, hours=87600)

        assert len(result) == 2
        # utc-noon (12:00 UTC) is newer than tz-plus5 (08:00 UTC), so must be first
        assert result[0]['task_id'] == 'utc-noon'
        assert result[1]['task_id'] == 'tz-plus5'

    # -----------------------------------------------------------------------
    # Gap coverage (a): one DB raises RuntimeError — gather-level resilience
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_one_db_raises_queue_depth_timeseries(self, tmp_path):
        """aggregate_queue_depth_timeseries silently skips a DB whose coroutine
        raises RuntimeError (escapes with_db) via gather(return_exceptions=True).
        """
        now = datetime(2026, 4, 11, 12, 7, 30, tzinfo=UTC)
        cutoff = now - timedelta(hours=24)
        bucket = _bucket_start(cutoff) + timedelta(hours=22)
        events = [
            {'event_type': 'merge_attempt',
             'timestamp': bucket + timedelta(minutes=1),
             'data': {'outcome': 'done'}},
        ]
        result = await self._run_with_one_broken_db(
            tmp_path, aggregate_queue_depth_timeseries, events, hours=24, now=now,
        )
        # Broken DB error was silently skipped — no exception raised
        assert len(result['labels']) == 97
        bucket_label = bucket.isoformat()
        assert bucket_label in result['labels']
        idx = result['labels'].index(bucket_label)
        assert result['values'][idx] == 1  # only from valid DB; broken DB skipped

    @pytest.mark.asyncio
    async def test_one_db_raises_outcome_distribution(self, tmp_path):
        """aggregate_outcome_distribution silently skips a broken DB."""
        now = datetime.now(UTC)
        events = [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=5),
             'data': {'outcome': 'done'}},
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=6),
             'data': {'outcome': 'done'}},
        ]
        result = await self._run_with_one_broken_db(
            tmp_path, aggregate_outcome_distribution, events, hours=24,
        )
        assert 'done' in result['labels']
        done_idx = result['labels'].index('done')
        assert result['values'][done_idx] == 2  # only from valid DB

    @pytest.mark.asyncio
    async def test_one_db_raises_latency_stats(self, tmp_path):
        """aggregate_latency_stats silently skips a broken DB."""
        now = datetime.now(UTC)
        events = [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=5),
             'data': {'outcome': 'done'}, 'duration_ms': 100},
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=6),
             'data': {'outcome': 'done'}, 'duration_ms': 200},
        ]
        result = await self._run_with_one_broken_db(
            tmp_path, aggregate_latency_stats, events, hours=24,
        )
        assert result['count'] == 2
        assert result['mean_ms'] == pytest.approx(150.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_one_db_raises_recent_merges(self, tmp_path):
        """aggregate_recent_merges silently skips a broken DB."""
        now = datetime.now(UTC)
        events = [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=1),
             'task_id': 'task-x', 'data': {'outcome': 'done'}},
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=2),
             'task_id': 'task-y', 'data': {'outcome': 'done'}},
        ]
        result = await self._run_with_one_broken_db(
            tmp_path, aggregate_recent_merges, events, limit=20,
        )
        assert len(result) == 2
        # Newest first
        assert result[0]['task_id'] == 'task-x'
        assert result[1]['task_id'] == 'task-y'

    @pytest.mark.asyncio
    async def test_one_db_raises_speculative_stats(self, tmp_path):
        """aggregate_speculative_stats silently skips a broken DB."""
        now = datetime.now(UTC)
        events = [
            {'event_type': 'speculative_merge', 'timestamp': now - timedelta(minutes=1),
             'data': {'base_sha': 'abc'}},
            {'event_type': 'speculative_merge', 'timestamp': now - timedelta(minutes=2),
             'data': {'base_sha': 'def'}},
        ]
        result = await self._run_with_one_broken_db(
            tmp_path, aggregate_speculative_stats, events, hours=24,
        )
        assert result['hit_count'] == 2
        assert result['discard_count'] == 0
        assert result['total'] == 2
        assert result['hit_rate'] == pytest.approx(1.0, abs=1e-6)

    # -----------------------------------------------------------------------
    # Gap coverage (b): empty dbs=[] — asyncio.gather(*[]) returns []
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_empty_dbs_queue_depth(self):
        """aggregate_queue_depth_timeseries with dbs=[] returns empty ChartData."""
        result = await aggregate_queue_depth_timeseries([], hours=24)
        assert result == {'labels': [], 'values': []}

    @pytest.mark.asyncio
    async def test_empty_dbs_outcome_distribution(self):
        """aggregate_outcome_distribution with dbs=[] returns empty ChartData."""
        result = await aggregate_outcome_distribution([], hours=24)
        assert result == {'labels': [], 'values': []}

    @pytest.mark.asyncio
    async def test_empty_dbs_latency_stats(self):
        """aggregate_latency_stats with dbs=[] returns all-zero stats dict."""
        result = await aggregate_latency_stats([], hours=24)
        assert result == {'p50': 0, 'p95': 0, 'p99': 0, 'count': 0, 'mean_ms': 0.0}

    @pytest.mark.asyncio
    async def test_empty_dbs_recent_merges(self):
        """aggregate_recent_merges with dbs=[] returns empty list."""
        result = await aggregate_recent_merges([], limit=20)
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_dbs_speculative_stats(self):
        """aggregate_speculative_stats with dbs=[] returns all-zero stats dict."""
        result = await aggregate_speculative_stats([], hours=24)
        assert result == {'hit_count': 0, 'discard_count': 0, 'total': 0, 'hit_rate': 0.0}

    # -----------------------------------------------------------------------
    # Gap coverage (b+): all-None dbs — gather(*[fn(None), fn(None)]) path
    #
    # Distinct from dbs=[] (gather(*[]) → []) because per-DB functions ARE
    # called with None and return empty defaults.  These tests confirm the
    # None-handling in each per-DB function propagates cleanly through the
    # aggregate layer.
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_all_none_dbs_queue_depth(self):
        """aggregate_queue_depth_timeseries with dbs=[None, None] returns empty ChartData."""
        result = await aggregate_queue_depth_timeseries([None, None], hours=24)
        assert result == {'labels': [], 'values': []}

    @pytest.mark.asyncio
    async def test_all_none_dbs_outcome_distribution(self):
        """aggregate_outcome_distribution with dbs=[None, None] returns empty ChartData."""
        result = await aggregate_outcome_distribution([None, None], hours=24)
        assert result == {'labels': [], 'values': []}

    @pytest.mark.asyncio
    async def test_all_none_dbs_latency_stats(self):
        """aggregate_latency_stats with dbs=[None, None] returns all-zero stats dict."""
        result = await aggregate_latency_stats([None, None], hours=24)
        assert result == {'p50': 0, 'p95': 0, 'p99': 0, 'count': 0, 'mean_ms': 0.0}

    @pytest.mark.asyncio
    async def test_all_none_dbs_recent_merges(self):
        """aggregate_recent_merges with dbs=[None, None] returns empty list."""
        result = await aggregate_recent_merges([None, None], limit=20)
        assert result == []

    @pytest.mark.asyncio
    async def test_all_none_dbs_speculative_stats(self):
        """aggregate_speculative_stats with dbs=[None, None] returns all-zero stats dict."""
        result = await aggregate_speculative_stats([None, None], hours=24)
        assert result == {'hit_count': 0, 'discard_count': 0, 'total': 0, 'hit_rate': 0.0}

    # -----------------------------------------------------------------------
    # Gap coverage (c): None entries in dbs — per-DB returns empty defaults
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_none_entry_queue_depth(self, tmp_path):
        """aggregate_queue_depth_timeseries with [valid_conn, None] merges cleanly."""
        now = datetime(2026, 4, 11, 12, 7, 30, tzinfo=UTC)
        cutoff = now - timedelta(hours=24)
        bucket = _bucket_start(cutoff) + timedelta(hours=22)
        events = [
            {'event_type': 'merge_attempt',
             'timestamp': bucket + timedelta(minutes=1),
             'data': {'outcome': 'done'}},
        ]
        result = await self._run_with_none_entry(
            tmp_path, aggregate_queue_depth_timeseries, events, hours=24, now=now,
        )
        assert len(result['labels']) == 97
        bucket_label = bucket.isoformat()
        assert bucket_label in result['labels']
        idx = result['labels'].index(bucket_label)
        assert result['values'][idx] == 1  # valid DB only; None contributed nothing

    @pytest.mark.asyncio
    async def test_none_entry_outcome_distribution(self, tmp_path):
        """aggregate_outcome_distribution with [valid_conn, None] merges cleanly."""
        now = datetime.now(UTC)
        events = [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=5),
             'data': {'outcome': 'done'}},
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=6),
             'data': {'outcome': 'done'}},
        ]
        result = await self._run_with_none_entry(
            tmp_path, aggregate_outcome_distribution, events, hours=24,
        )
        assert 'done' in result['labels']
        done_idx = result['labels'].index('done')
        assert result['values'][done_idx] == 2  # from valid DB only

    @pytest.mark.asyncio
    async def test_none_entry_latency_stats(self, tmp_path):
        """aggregate_latency_stats with [valid_conn, None] merges cleanly."""
        now = datetime.now(UTC)
        events = [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=5),
             'data': {'outcome': 'done'}, 'duration_ms': 100},
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=6),
             'data': {'outcome': 'done'}, 'duration_ms': 200},
        ]
        result = await self._run_with_none_entry(
            tmp_path, aggregate_latency_stats, events, hours=24,
        )
        assert result['count'] == 2
        assert result['mean_ms'] == pytest.approx(150.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_none_entry_recent_merges(self, tmp_path):
        """aggregate_recent_merges with [valid_conn, None] merges cleanly."""
        now = datetime.now(UTC)
        events = [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=1),
             'task_id': 'task-a', 'data': {'outcome': 'done'}},
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=2),
             'task_id': 'task-b', 'data': {'outcome': 'done'}},
        ]
        result = await self._run_with_none_entry(
            tmp_path, aggregate_recent_merges, events, limit=20,
        )
        assert len(result) == 2
        assert result[0]['task_id'] == 'task-a'
        assert result[1]['task_id'] == 'task-b'

    @pytest.mark.asyncio
    async def test_none_entry_speculative_stats(self, tmp_path):
        """aggregate_speculative_stats with [valid_conn, None] merges cleanly."""
        now = datetime.now(UTC)
        events = [
            {'event_type': 'speculative_merge', 'timestamp': now - timedelta(minutes=1),
             'data': {'base_sha': 'abc'}},
            {'event_type': 'speculative_discard', 'timestamp': now - timedelta(minutes=2),
             'data': {'reason': 'previous_failed'}},
        ]
        result = await self._run_with_none_entry(
            tmp_path, aggregate_speculative_stats, events, hours=24,
        )
        assert result['hit_count'] == 1
        assert result['discard_count'] == 1
        assert result['total'] == 2
        assert result['hit_rate'] == pytest.approx(0.5, abs=1e-6)

    # -----------------------------------------------------------------------
    # Gap coverage (d): current-bucket-tail regression
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_current_bucket_tail_regression(self, tmp_path):
        """Near-now events from two DBs land in the last (current) bucket.

        Verifies that: (a) the current bucket is always present as the last
        label, and (b) events inserted at now-1min and now-30s each contribute
        to the same current bucket so the summed count is 2.
        """
        now = datetime(2026, 4, 11, 12, 7, 30, tzinfo=UTC)

        db1 = self._make_db(tmp_path, 'runs1.db', [
            {'event_type': 'merge_attempt',
             'timestamp': now - timedelta(minutes=1),
             'data': {'outcome': 'done'}},
        ])
        db2 = self._make_db(tmp_path, 'runs2.db', [
            {'event_type': 'merge_attempt',
             'timestamp': now - timedelta(seconds=30),
             'data': {'outcome': 'done'}},
        ])

        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await aggregate_queue_depth_timeseries([conn1, conn2], hours=24, now=now)

        # The last label must be the current aligned bucket (12:00:00 for 12:07:30)
        expected_last_label = _align_bucket(now, 15).isoformat()
        assert result['labels'][-1] == expected_last_label

        # Both events are in the current bucket — summed count must be 2
        idx = result['labels'].index(expected_last_label)
        assert result['values'][idx] == 2

    @pytest.mark.asyncio
    async def test_bucket_boundary_events_in_adjacent_buckets(self, tmp_path):
        """Events straddling a 15-min boundary land in adjacent buckets, not the same one.

        DB1: event at exactly 12:15:00 → bucket 12:15.
        DB2: event at 12:14:59          → bucket 12:00.
        Tests the 'divergent label sets' scenario from the aggregate docstring:
        each DB independently aligns its timestamps, so the aggregate must merge
        label sets correctly and never collapse boundary events into a single bucket.
        """
        base = datetime(2026, 4, 11, 12, 15, 0, tzinfo=UTC)  # exact bucket boundary

        db1 = self._make_db(tmp_path, 'runs1.db', [
            {'event_type': 'merge_attempt',
             'timestamp': base,                              # 12:15:00 → bucket 12:15
             'data': {'outcome': 'done'}},
        ])
        db2 = self._make_db(tmp_path, 'runs2.db', [
            {'event_type': 'merge_attempt',
             'timestamp': base - timedelta(seconds=1),      # 12:14:59 → bucket 12:00
             'data': {'outcome': 'done'}},
        ])

        now = datetime(2026, 4, 11, 12, 30, 0, tzinfo=UTC)
        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await aggregate_queue_depth_timeseries([conn1, conn2], hours=1, now=now)

        labels = result['labels']
        values = result['values']
        bucket_1215 = datetime(2026, 4, 11, 12, 15, tzinfo=UTC).isoformat()
        bucket_1200 = datetime(2026, 4, 11, 12, 0, tzinfo=UTC).isoformat()

        assert bucket_1215 in labels, f"bucket 12:15 missing from {labels}"
        assert bucket_1200 in labels, f"bucket 12:00 missing from {labels}"

        idx_1215 = labels.index(bucket_1215)
        idx_1200 = labels.index(bucket_1200)

        # Each event is in its own bucket — no cross-boundary merging
        assert values[idx_1215] == 1
        assert values[idx_1200] == 1
        # 12:15 bucket immediately follows 12:00 bucket (adjacent, one step apart)
        assert idx_1215 == idx_1200 + 1

    # -----------------------------------------------------------------------
    # Gap coverage (e): duration_ms=0 and NULL excluded by aggregate path
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_aggregate_latency_zero_duration_excluded(self, tmp_path):
        """Zero-duration rows are excluded from aggregate latency stats.

        _get_durations filters 'AND duration_ms > 0', so duration_ms=0 rows
        must not contribute to count or percentiles in the aggregate.
        """
        now = datetime.now(UTC)
        db1 = self._make_db(tmp_path, 'runs1.db', [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=5),
             'data': {'outcome': 'done'}, 'duration_ms': 0},
        ])
        db2 = self._make_db(tmp_path, 'runs2.db', [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=6),
             'data': {'outcome': 'done'}, 'duration_ms': 500},
        ])

        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await aggregate_latency_stats([conn1, conn2], hours=24)

        assert result['count'] == 1  # zero-duration row excluded
        assert result['p50'] == 500

    @pytest.mark.asyncio
    async def test_aggregate_latency_null_duration_excluded(self, tmp_path):
        """NULL-duration rows are excluded from aggregate latency stats.

        _get_durations filters 'AND duration_ms IS NOT NULL', so NULL rows
        must not contribute to count or percentiles in the aggregate.
        """
        now = datetime.now(UTC)
        db1 = self._make_db(tmp_path, 'runs1.db', [
            # duration_ms=None → stored as NULL in SQLite
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=5),
             'data': {'outcome': 'done'}},
        ])
        db2 = self._make_db(tmp_path, 'runs2.db', [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=6),
             'data': {'outcome': 'done'}, 'duration_ms': 300},
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=7),
             'data': {'outcome': 'done'}, 'duration_ms': 600},
        ])

        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await aggregate_latency_stats([conn1, conn2], hours=24)

        assert result['count'] == 2  # NULL row excluded
        assert result['mean_ms'] == pytest.approx(450.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_aggregate_latency_mixed_zero_null_valid(self, tmp_path):
        """Only valid (non-zero, non-NULL) durations from multiple DBs contribute.

        DB1 has duration_ms=0 and a NULL-duration row; DB2 has three valid rows.
        Confirms the aggregate path preserves _get_durations' filter and only
        the three valid durations from DB2 contribute.
        """
        now = datetime.now(UTC)
        db1 = self._make_db(tmp_path, 'runs1.db', [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=5),
             'data': {'outcome': 'done'}, 'duration_ms': 0},
            # NULL duration:
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=6),
             'data': {'outcome': 'done'}},
        ])
        db2 = self._make_db(tmp_path, 'runs2.db', [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=7),
             'data': {'outcome': 'done'}, 'duration_ms': 100},
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=8),
             'data': {'outcome': 'done'}, 'duration_ms': 200},
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=9),
             'data': {'outcome': 'done'}, 'duration_ms': 300},
        ])

        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await aggregate_latency_stats([conn1, conn2], hours=24)

        assert result['count'] == 3  # only valid durations from DB2
        assert result['mean_ms'] == pytest.approx(200.0, abs=1e-6)
