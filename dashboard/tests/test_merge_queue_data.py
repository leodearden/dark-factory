"""Tests for dashboard.data.merge_queue — merge queue query functions."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

import dashboard.data.merge_queue as _mqmod
from tests._dt_helpers import make_fixed_datetime_cls

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


def _make_db(tmp_path, name, events):
    """Create a populated DB with the given events list of dicts.

    Each dict is forwarded as kwargs to _insert_event.  Returns the db_path.
    """
    db_path = tmp_path / name
    conn = sqlite3.connect(str(db_path))
    conn.executescript(MERGE_EVENTS_SCHEMA)
    for evt in events:
        _insert_event(conn, **evt)
    conn.commit()
    conn.close()
    return db_path


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
    _cutoff_iso,
    _get_durations,
    _ts_sort_key,
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
        (0, 15),
        (1, 15),
        (24, 15),
        (25, 60),
        (167, 60),
        (168, 60),
        (169, 360),
        (719, 360),
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
# TestTsSortKey
# ---------------------------------------------------------------------------

class TestTsSortKey:
    def test_non_utc_aware_datetime_normalized(self):
        """A non-UTC-offset timestamp is normalized to UTC.

        '2026-04-01T10:00:00+05:30' represents 04:30:00 UTC.  _ts_sort_key
        must return a datetime with utcoffset() == timedelta(0) so that
        downstream key comparisons and serialization are consistent.
        """
        entry = {'timestamp': '2026-04-01T10:00:00+05:30'}
        result = _ts_sort_key(entry)
        # Must be UTC-normalised
        assert result.utcoffset() == timedelta(0)
        # Point-in-time must equal the UTC equivalent
        expected_utc = datetime(2026, 4, 1, 4, 30, 0, tzinfo=UTC)
        assert result == expected_utc

    def test_utc_timestamp_unchanged(self):
        """A UTC-offset timestamp is returned unchanged (same value, UTC offset).

        .astimezone(UTC) must be a no-op for timestamps that are already UTC
        so that well-formed data passes through without any transformation.
        """
        entry = {'timestamp': '2026-04-01T10:00:00+00:00'}
        result = _ts_sort_key(entry)
        assert result.utcoffset() == timedelta(0)
        assert result == datetime(2026, 4, 1, 10, 0, 0, tzinfo=UTC)

    def test_naive_timestamp_gets_utc(self):
        """A naive timestamp (no tzinfo) gets UTC attached and normalized.

        parse_utc attaches UTC to naive datetimes via replace(tzinfo=UTC).
        .astimezone(UTC) on a UTC datetime is a no-op, so the result should
        be UTC-aware and equal to the naive value interpreted as UTC.
        """
        entry = {'timestamp': '2026-04-01T10:00:00'}
        result = _ts_sort_key(entry)
        assert result.tzinfo is not None
        assert result.utcoffset() == timedelta(0)
        assert result == datetime(2026, 4, 1, 10, 0, 0, tzinfo=UTC)

    def test_missing_timestamp_returns_datetime_min(self):
        """An entry with no 'timestamp' key returns UTC-aware datetime.min.

        Malformed entries must sort to the end of a descending sort.  The
        fallback value is datetime.min.replace(tzinfo=UTC).
        """
        result = _ts_sort_key({})
        assert result == datetime.min.replace(tzinfo=UTC)
        assert result.utcoffset() == timedelta(0)

    def test_invalid_timestamp_returns_datetime_min(self):
        """An unparseable timestamp string returns UTC-aware datetime.min.

        The ValueError branch in _ts_sort_key must catch fromisoformat failures
        and return the same fallback as the missing-key / None cases so that
        malformed entries sort consistently to the end of a descending sort.
        """
        result = _ts_sort_key({'timestamp': 'garbage'})
        assert result == datetime.min.replace(tzinfo=UTC)
        assert result.utcoffset() == timedelta(0)


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

    @pytest.mark.asyncio
    async def test_last_bucket_includes_events_after_now_aligned(self, merge_events_db):
        """Events in (now_aligned, effective_now] must be counted in the last bucket.

        With now=2026-04-11T12:07:30 UTC and hours=24:
          effective_now = 2026-04-11T12:07:30+00:00
          now_aligned   = 2026-04-11T12:00:00+00:00  (floor to 15-min boundary)

        An event at now_aligned + 1 minute (12:01:00) is after the last bucket
        boundary but before effective_now (12:07:30).  The SQL WHERE clause uses
        effective_now (not now_aligned) as the upper bound so this event is
        fetched and then floored into the last bucket (now_aligned).
        """
        now = datetime(2026, 4, 11, 12, 7, 30, tzinfo=UTC)
        bucket_min = _bucket_minutes_for_window(24)  # 15 for a 24h window
        now_aligned = _align_bucket(now, bucket_min)

        # Event is 1 minute after now_aligned and before effective_now
        event_ts = now_aligned + timedelta(minutes=1)
        assert event_ts > now_aligned, "Pre-condition: event must be after now_aligned"
        assert event_ts < now, "Pre-condition: event must be before effective_now"

        conn_sync = sqlite3.connect(str(merge_events_db))
        _insert_event(conn_sync, event_type='merge_attempt', timestamp=event_ts,
                      data={'outcome': 'done'})
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await queue_depth_timeseries(db, hours=24, now=now)

        last_label = now_aligned.isoformat()
        assert last_label in result['labels'], (
            f"Expected last bucket label {last_label!r} in labels"
        )
        idx = result['labels'].index(last_label)
        assert result['values'][idx] == 1, (
            f"Last bucket should have count=1 for the event at {event_ts.isoformat()}, "
            f"got {result['values'][idx]}"
        )

    @pytest.mark.asyncio
    async def test_last_bucket_includes_event_at_exact_effective_now(self, merge_events_db):
        """An event timestamped exactly at effective_now must be counted (SQL uses <=).

        With now=2026-04-11T12:07:30 UTC and hours=24:
          effective_now = 2026-04-11T12:07:30+00:00  (== now)
          now_aligned   = 2026-04-11T12:00:00+00:00  (floor to 15-min boundary)

        An event at exactly effective_now sits on the inclusive upper boundary of
        the SQL ``timestamp <= ?`` clause.  It must be fetched and floored into
        the last bucket (now_aligned).
        """
        now = datetime(2026, 4, 11, 12, 7, 30, tzinfo=UTC)
        bucket_min = _bucket_minutes_for_window(24)  # 15 for a 24h window
        now_aligned = _align_bucket(now, bucket_min)

        # Event is exactly at effective_now — tests the inclusive upper bound
        event_ts = now  # effective_now == now when now is passed explicitly
        assert event_ts > now_aligned, "Pre-condition: event must be after now_aligned"

        conn_sync = sqlite3.connect(str(merge_events_db))
        _insert_event(conn_sync, event_type='merge_attempt', timestamp=event_ts,
                      data={'outcome': 'done'})
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await queue_depth_timeseries(db, hours=24, now=now)

        last_label = now_aligned.isoformat()
        assert last_label in result['labels'], (
            f"Expected last bucket label {last_label!r} in labels"
        )
        idx = result['labels'].index(last_label)
        assert result['values'][idx] == 1, (
            f"Last bucket should have count=1 for the event at exact effective_now "
            f"({event_ts.isoformat()}), got {result['values'][idx]}"
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
        """_get_durations returns a sorted list regardless of insertion order.

        Establishes the sorted-output invariant of _get_durations, which latency_stats
        relies on to avoid a redundant sorted() call.

        The timestamps are staggered in *reverse* duration order so that if the
        query were ``ORDER BY timestamp DESC`` (most-recent first) it would return
        [500, 400, 300, 200, 100].  Because the expected result is [100, 200, 300,
        400, 500], this proves that ``ORDER BY duration_ms`` is what actually
        determines the output order — not the timestamp ordering.
        """
        now = datetime.now(UTC)
        conn_sync = sqlite3.connect(str(merge_events_db))
        # duration_ms → timestamp mapping: higher duration = more recent timestamp
        # timestamp order (most-recent first): 500, 400, 300, 200, 100
        # duration_ms order (ascending):       100, 200, 300, 400, 500
        events = [
            (500, now - timedelta(minutes=1)),
            (100, now - timedelta(minutes=5)),
            (300, now - timedelta(minutes=3)),
            (200, now - timedelta(minutes=4)),
            (400, now - timedelta(minutes=2)),
        ]
        for ms, ts in events:
            _insert_event(conn_sync, event_type='merge_attempt',
                          timestamp=ts,
                          data={'outcome': 'done'},
                          duration_ms=ms)
        conn_sync.commit()
        conn_sync.close()

        async with aiosqlite.connect(str(merge_events_db)) as db:
            db.row_factory = aiosqlite.Row
            result = await _get_durations(db, hours=24)

        assert result == [100.0, 200.0, 300.0, 400.0, 500.0]


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
        broken = MagicMock(spec=aiosqlite.Connection)
        broken.execute_fetchall = AsyncMock(side_effect=RuntimeError('simulated broken DB'))
        return broken

    async def _run_with_one_broken_db(self, tmp_path, aggregate_fn, events, **kwargs):
        """Scaffold for resilience tests: one valid DB + one broken mock connection.

        Returns (result, broken) so callers can assert the broken mock was exercised.
        """
        db1 = self._make_db(tmp_path, 'runs1.db', events)
        broken = self._make_broken_mock()
        async with aiosqlite.connect(str(db1)) as conn1:
            conn1.row_factory = aiosqlite.Row
            result = await aggregate_fn([conn1, broken], **kwargs)
            return result, broken

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
        """aggregate_queue_depth_timeseries at hours=87600 produces exactly 3651 buckets."""
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

        assert len(result['labels']) == 3651


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
        result, broken = await self._run_with_one_broken_db(
            tmp_path, aggregate_queue_depth_timeseries, events, hours=24, now=now,
        )
        # Broken DB error was silently skipped — no exception raised
        assert len(result['labels']) == 97
        bucket_label = bucket.isoformat()
        assert bucket_label in result['labels']
        idx = result['labels'].index(bucket_label)
        assert result['values'][idx] == 1  # only from valid DB; broken DB skipped
        broken.execute_fetchall.assert_called()

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
        result, broken = await self._run_with_one_broken_db(
            tmp_path, aggregate_outcome_distribution, events, hours=24,
        )
        assert 'done' in result['labels']
        done_idx = result['labels'].index('done')
        assert result['values'][done_idx] == 2  # only from valid DB
        broken.execute_fetchall.assert_called()

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
        result, broken = await self._run_with_one_broken_db(
            tmp_path, aggregate_latency_stats, events, hours=24,
        )
        assert result['count'] == 2
        assert result['mean_ms'] == pytest.approx(150.0, abs=1e-6)
        broken.execute_fetchall.assert_called()

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
        result, broken = await self._run_with_one_broken_db(
            tmp_path, aggregate_recent_merges, events, limit=20,
        )
        assert len(result) == 2
        # Newest first
        assert result[0]['task_id'] == 'task-x'
        assert result[1]['task_id'] == 'task-y'
        broken.execute_fetchall.assert_called()

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
        result, broken = await self._run_with_one_broken_db(
            tmp_path, aggregate_speculative_stats, events, hours=24,
        )
        assert result['hit_count'] == 2
        assert result['discard_count'] == 0
        assert result['total'] == 2
        assert result['hit_rate'] == pytest.approx(1.0, abs=1e-6)
        broken.execute_fetchall.assert_called()

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
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    @pytest.mark.parametrize('patch_target,agg_fn,kwargs,expected', [
        (
            'queue_depth_timeseries',
            aggregate_queue_depth_timeseries,
            {'hours': 24},
            {'labels': [], 'values': []},
        ),
        (
            'outcome_distribution',
            aggregate_outcome_distribution,
            {'hours': 24},
            {'labels': [], 'values': []},
        ),
        (
            '_get_durations',
            aggregate_latency_stats,
            {'hours': 24},
            {'p50': 0, 'p95': 0, 'p99': 0, 'count': 0, 'mean_ms': 0.0},
        ),
        (
            'recent_merges',
            aggregate_recent_merges,
            {'limit': 20},
            [],
        ),
        (
            'speculative_stats',
            aggregate_speculative_stats,
            {'hours': 24},
            {'hit_count': 0, 'discard_count': 0, 'total': 0, 'hit_rate': 0.0},
        ),
    ])
    async def test_all_none_dbs_spy(self, patch_target, agg_fn, kwargs, expected):
        """Parametrized: aggregate fn with dbs=[None, None] calls per-DB fn twice with None.

        Confirms the gather(*[fn(None), fn(None)]) path is distinct from
        dbs=[] (gather(*[]) → []) because per-DB functions ARE called with None.
        """
        wraps_fn = getattr(_mqmod, patch_target)
        with patch(
            f'dashboard.data.merge_queue.{patch_target}',
            wraps=wraps_fn,
        ) as spy:
            result = await agg_fn([None, None], **kwargs)
            assert result == expected
            assert spy.call_count == 2
            for call in spy.call_args_list:
                assert call.args[0] is None

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

# ---------------------------------------------------------------------------
# TestCutoffIso (step-1)
# ---------------------------------------------------------------------------

class TestCutoffIso:
    def test_cutoff_iso_uses_provided_now(self):
        """_cutoff_iso(hours=24, now=fixed_dt) returns (fixed_dt - 24h).isoformat()."""
        fixed_dt = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        expected = (fixed_dt - timedelta(hours=24)).isoformat()
        result = _cutoff_iso(24, now=fixed_dt)
        assert result == expected

    def test_cutoff_iso_no_now_uses_current_time(self):
        """Without now, _cutoff_iso uses datetime.now(UTC) — result matches mocked clock exactly."""
        FIXED_NOW = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        expected = (FIXED_NOW - timedelta(hours=24)).isoformat()

        _FixedDT = make_fixed_datetime_cls(FIXED_NOW)

        with patch.object(_mqmod, 'datetime', _FixedDT):
            result = _cutoff_iso(24)

        assert result == expected


# ---------------------------------------------------------------------------
# TestNowThreadingToCutoffIso (step-3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    'fn_under_test',
    [outcome_distribution, speculative_stats, latency_stats],
    ids=lambda fn: fn.__name__,
)
class TestNowThreadingToCutoffIso:
    @pytest.mark.asyncio
    async def test_threads_now_to_cutoff_iso(self, fn_under_test, merge_events_db):
        """fn_under_test accepts now and passes it through to _cutoff_iso."""
        fixed_now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        captured_nows: list = []

        def mock_cutoff_iso(hours: int, *, now=None) -> str:
            captured_nows.append(now)
            return '2020-01-01T00:00:00+00:00'

        async with aiosqlite.connect(str(merge_events_db)) as conn:
            conn.row_factory = aiosqlite.Row
            with patch('dashboard.data.merge_queue._cutoff_iso', side_effect=mock_cutoff_iso):
                await fn_under_test(conn, hours=24, now=fixed_now)

        assert captured_nows == [fixed_now], (
            f"Expected _cutoff_iso called once with now={fixed_now!r}, got {captured_nows!r}"
        )


# ---------------------------------------------------------------------------
# TestAggregateOutcomeDistributionNow (step-9)
# ---------------------------------------------------------------------------

class TestAggregateOutcomeDistributionNow:
    @pytest.mark.asyncio
    async def test_aggregate_outcome_distribution_consistent_now(self, tmp_path):
        """aggregate_outcome_distribution resolves now once and threads it to all per-DB calls.

        Events near fixed_now are inside the 24h window for that now, but would be
        excluded by a real datetime.now(UTC) cutoff (they are 2+ days in the past).
        Will fail before step-10 impl because aggregate_outcome_distribution has no `now` param.
        """
        fixed_now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        # Events at 1h before fixed_now — inside [fixed_now - 24h, fixed_now] window
        event_time = fixed_now - timedelta(hours=1)

        db1 = _make_db(tmp_path, 'runs1.db', [
            {'event_type': 'merge_attempt', 'timestamp': event_time, 'data': {'outcome': 'done'}},
        ])
        db2 = _make_db(tmp_path, 'runs2.db', [
            {'event_type': 'merge_attempt', 'timestamp': event_time, 'data': {'outcome': 'done'}},
        ])

        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await aggregate_outcome_distribution(
                [conn1, conn2], hours=24, now=fixed_now,
            )

        # Both events are inside the window → both should appear in the results
        assert 'done' in result['labels'], f"Expected 'done' in labels, got {result}"
        done_idx = result['labels'].index('done')
        assert result['values'][done_idx] == 2  # one event per DB


# ---------------------------------------------------------------------------
# TestAggregateLatencyStatsNow (step-11)
# ---------------------------------------------------------------------------

class TestAggregateLatencyStatsNow:
    @pytest.mark.asyncio
    async def test_aggregate_latency_stats_consistent_now(self, tmp_path):
        """aggregate_latency_stats resolves now once and threads it to all per-DB calls.

        Events near fixed_now are inside the 24h window for that now, but would be
        excluded by a real datetime.now(UTC) cutoff (they are 2+ days in the past).
        Will fail before step-12 impl because aggregate_latency_stats has no `now` param.
        """
        fixed_now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        event_time = fixed_now - timedelta(hours=1)

        db1 = _make_db(tmp_path, 'runs1.db', [
            {'event_type': 'merge_attempt', 'timestamp': event_time,
             'data': {'outcome': 'done'}, 'duration_ms': 100},
        ])
        db2 = _make_db(tmp_path, 'runs2.db', [
            {'event_type': 'merge_attempt', 'timestamp': event_time,
             'data': {'outcome': 'done'}, 'duration_ms': 200},
        ])

        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await aggregate_latency_stats([conn1, conn2], hours=24, now=fixed_now)

        # Both events are inside the window → count=2
        assert result['count'] == 2, (
            f"Expected count=2 (both events within fixed_now window), got {result}"
        )


# ---------------------------------------------------------------------------
# TestAggregateSpeculativeStatsNow (step-13)
# ---------------------------------------------------------------------------

class TestAggregateSpeculativeStatsNow:
    @pytest.mark.asyncio
    async def test_aggregate_speculative_stats_consistent_now(self, tmp_path):
        """aggregate_speculative_stats resolves now once and threads it to all per-DB calls.

        Events near fixed_now are inside the 24h window for that now, but would be
        excluded by a real datetime.now(UTC) cutoff (they are 2+ days in the past).
        Will fail before step-14 impl because aggregate_speculative_stats has no `now` param.
        """
        fixed_now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        event_time = fixed_now - timedelta(hours=1)

        db1 = _make_db(tmp_path, 'runs1.db', [
            {'event_type': 'speculative_merge',
             'timestamp': event_time, 'data': {'base_sha': 'abc'}},
        ])
        db2 = _make_db(tmp_path, 'runs2.db', [
            {'event_type': 'speculative_discard',
             'timestamp': event_time, 'data': {'reason': 'previous_failed'}},
        ])

        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await aggregate_speculative_stats(
                [conn1, conn2], hours=24, now=fixed_now,
            )

        # Both events are inside the window → hit_count=1, discard_count=1, total=2
        assert result['total'] == 2, (
            f"Expected total=2 (both events within fixed_now window), got {result}"
        )
        assert result['hit_count'] == 1
        assert result['discard_count'] == 1


# ---------------------------------------------------------------------------
# TestAggregateQueueDepthTimeseriesNow
# ---------------------------------------------------------------------------

class TestAggregateQueueDepthTimeseriesNow:
    @pytest.mark.asyncio
    async def test_aggregate_queue_depth_timeseries_consistent_now(self, tmp_path):
        """aggregate_queue_depth_timeseries resolves now once and threads it to per-DB calls.

        Events near fixed_now are inside the 24h window for that now, but would be
        excluded by a real datetime.now(UTC) cutoff (they are 2+ days in the past).
        Ensures a regression removing `now` from aggregate_queue_depth_timeseries
        would be caught by this test.
        """
        fixed_now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        event_time = fixed_now - timedelta(hours=1)

        db1 = _make_db(tmp_path, 'runs1.db', [
            {'event_type': 'merge_attempt', 'timestamp': event_time,
             'data': {'outcome': 'done'}},
        ])
        db2 = _make_db(tmp_path, 'runs2.db', [
            {'event_type': 'merge_attempt', 'timestamp': event_time,
             'data': {'outcome': 'done'}},
        ])

        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await aggregate_queue_depth_timeseries(
                [conn1, conn2], hours=24, now=fixed_now,
            )

        # Both events are inside the fixed_now window → total count == 2
        total_count = sum(result['values'])
        assert total_count == 2, (
            f"Expected 2 events (both inside fixed_now window), got total={total_count}, "
            f"result={result}"
        )


# ---------------------------------------------------------------------------
# TestProjectScopedDbsLabeled (step-1)
# ---------------------------------------------------------------------------


class TestProjectScopedDbsLabeled:
    """Tests for app._project_scoped_dbs_labeled."""

    async def test_returns_pid_db_pairs(self, tmp_path):
        """Returns (str(root), connection|None) pairs, main project first."""
        from pathlib import Path

        from dashboard.app import _project_scoped_dbs_labeled
        from dashboard.config import DashboardConfig
        from dashboard.data.db import DbPool

        root_a = tmp_path / 'A'
        root_b = tmp_path / 'B'
        root_a.mkdir()
        root_b.mkdir()

        config = DashboardConfig(
            project_root=root_a,
            known_project_roots=[root_b],
        )
        pool = DbPool()
        try:
            rel = Path('data/orchestrator/runs.db')
            result = await _project_scoped_dbs_labeled(config, pool, rel)
        finally:
            await pool.close_all()

        # (a) Returns list of 2 tuples
        assert isinstance(result, list)
        assert len(result) == 2

        # (b) each element is a (str, ...) pair
        for pid, _db in result:
            assert isinstance(pid, str)

        # (c) main project root is always index 0
        pids = [pid for pid, _ in result]
        assert pids[0] == str(config.project_root)
        assert pids[1] == str(config.known_project_roots[0])

    async def test_deduplicates_duplicate_roots(self, tmp_path):
        """When known_project_roots contains the same path as project_root, only one entry."""
        from pathlib import Path

        from dashboard.app import _project_scoped_dbs_labeled
        from dashboard.config import DashboardConfig
        from dashboard.data.db import DbPool

        root_a = tmp_path / 'A'
        root_a.mkdir()

        config = DashboardConfig(
            project_root=root_a,
            known_project_roots=[root_a],  # duplicate
        )
        pool = DbPool()
        try:
            rel = Path('data/orchestrator/runs.db')
            result = await _project_scoped_dbs_labeled(config, pool, rel)
        finally:
            await pool.close_all()

        assert len(result) == 1
        assert result[0][0] == str(config.project_root)

    async def test_db_is_none_when_file_missing(self, tmp_path):
        """Returns None connection when the DB file does not exist."""
        from pathlib import Path

        from dashboard.app import _project_scoped_dbs_labeled
        from dashboard.config import DashboardConfig
        from dashboard.data.db import DbPool

        root_a = tmp_path / 'A'
        root_a.mkdir()
        config = DashboardConfig(project_root=root_a)
        pool = DbPool()
        try:
            rel = Path('data/orchestrator/runs.db')  # file not created
            result = await _project_scoped_dbs_labeled(config, pool, rel)
        finally:
            await pool.close_all()

        assert len(result) == 1
        _pid, db = result[0]
        assert db is None  # file does not exist → None connection


# ---------------------------------------------------------------------------
# TestFilterMergesWithin (step-3)
# ---------------------------------------------------------------------------


class TestFilterMergesWithin:
    """Tests for merge_queue.filter_merges_within."""

    def _make_row(self, offset_minutes, task_id='t1'):
        """Build a merge row dict with timestamp = NOW - offset_minutes."""
        ts = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC) - timedelta(minutes=offset_minutes)
        return {'task_id': task_id, 'timestamp': ts.isoformat(), 'outcome': 'done'}

    def test_keeps_rows_within_window(self):
        """Rows at -5m, -10m, -14m59s survive a 15-minute window."""
        from dashboard.data.merge_queue import filter_merges_within

        now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        rows = [
            self._make_row(5, task_id='t_5m'),    # 5m ago  → in
            self._make_row(10, task_id='t_10m'),  # 10m ago → in
            {'task_id': 't3', 'timestamp': (now - timedelta(minutes=14, seconds=59)).isoformat(),
             'outcome': 'done'},                   # 14m59s ago → in (< 15m)
            self._make_row(20, task_id='t_20m'),  # 20m ago → out
            {'task_id': 't5', 'timestamp': (now - timedelta(minutes=15, seconds=1)).isoformat(),
             'outcome': 'done'},                   # 15m01s ago → out
        ]
        result = filter_merges_within(rows, minutes=15, now=now)
        task_ids = [r['task_id'] for r in result]
        assert 't_5m' in task_ids   # 5m ago — must survive
        assert 't_10m' in task_ids  # 10m ago — must survive (both rows, not just one)
        assert 't3' in task_ids     # 14m59s — must survive
        assert 't_20m' not in task_ids  # 20m ago — must be filtered
        assert 't5' not in task_ids     # 15m01s — must be filtered

    def test_filters_out_old_rows(self):
        """Rows older than the window are excluded."""
        from dashboard.data.merge_queue import filter_merges_within

        now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        rows = [self._make_row(20), self._make_row(30)]
        result = filter_merges_within(rows, minutes=15, now=now)
        assert result == []

    def test_empty_list_passthrough(self):
        """Empty input returns empty output."""
        from dashboard.data.merge_queue import filter_merges_within

        now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        assert filter_merges_within([], minutes=15, now=now) == []

    def test_malformed_timestamp_filtered_out(self):
        """A row with unparseable timestamp is dropped defensively."""
        from dashboard.data.merge_queue import filter_merges_within

        now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        rows = [
            {'task_id': 'bad', 'timestamp': 'not-a-date', 'outcome': 'done'},
            self._make_row(5),  # valid, should survive
        ]
        result = filter_merges_within(rows, minutes=15, now=now)
        task_ids = [r['task_id'] for r in result]
        assert 'bad' not in task_ids
        assert 't1' in task_ids

    def test_preserves_input_order(self):
        """Output order matches input order (no re-sorting)."""
        from dashboard.data.merge_queue import filter_merges_within

        now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        rows = [
            {'task_id': 'first', 'timestamp': (now - timedelta(minutes=1)).isoformat(), 'outcome': 'done'},
            {'task_id': 'second', 'timestamp': (now - timedelta(minutes=2)).isoformat(), 'outcome': 'done'},
            {'task_id': 'third', 'timestamp': (now - timedelta(minutes=3)).isoformat(), 'outcome': 'done'},
        ]
        result = filter_merges_within(rows, minutes=15, now=now)
        assert [r['task_id'] for r in result] == ['first', 'second', 'third']

    def test_now_defaults_to_current_time(self):
        """When now=None, filter uses the current wall clock (smoke test)."""
        from dashboard.data.merge_queue import filter_merges_within

        # A row timestamped 2 minutes ago should survive a 15-minute window
        ts = (datetime.now(UTC) - timedelta(minutes=2)).isoformat()
        rows = [{'task_id': 'recent', 'timestamp': ts, 'outcome': 'done'}]
        result = filter_merges_within(rows, minutes=15)
        assert len(result) == 1
        assert result[0]['task_id'] == 'recent'


# ---------------------------------------------------------------------------
# TestEnrichMergesWithTitles (step-5)
# ---------------------------------------------------------------------------


class TestEnrichMergesWithTitles:
    """Tests for merge_queue.enrich_merges_with_titles."""

    def test_maps_titles_by_task_id(self):
        """task_id keys resolve to matching title values."""
        from dashboard.data.merge_queue import enrich_merges_with_titles

        merges = [
            {'task_id': '7', 'outcome': 'done'},
            {'task_id': '42', 'outcome': 'conflict'},
        ]
        title_map = {'7': 'Fix X', '42': 'Add Y'}
        result = enrich_merges_with_titles(merges, title_map)
        assert result[0]['title'] == 'Fix X'
        assert result[1]['title'] == 'Add Y'

    def test_none_task_id_gets_empty_title(self):
        """task_id=None maps to empty string (no KeyError)."""
        from dashboard.data.merge_queue import enrich_merges_with_titles

        merges = [{'task_id': None, 'outcome': 'done'}]
        result = enrich_merges_with_titles(merges, {'7': 'Fix X'})
        assert result[0]['title'] == ''

    def test_unknown_task_id_gets_empty_title(self):
        """task_id not in title_map maps to empty string."""
        from dashboard.data.merge_queue import enrich_merges_with_titles

        merges = [{'task_id': '99', 'outcome': 'done'}]
        result = enrich_merges_with_titles(merges, {'7': 'Fix X'})
        assert result[0]['title'] == ''

    def test_extra_title_map_keys_ignored(self):
        """Keys in title_map absent from merges are ignored."""
        from dashboard.data.merge_queue import enrich_merges_with_titles

        merges = [{'task_id': '7', 'outcome': 'done'}]
        title_map = {'7': 'Fix X', '99': 'Extra'}
        result = enrich_merges_with_titles(merges, title_map)
        assert len(result) == 1
        assert result[0]['title'] == 'Fix X'

    def test_does_not_mutate_input_rows(self):
        """Input dicts are not modified (shallow-copy semantics)."""
        from dashboard.data.merge_queue import enrich_merges_with_titles

        original = {'task_id': '7', 'outcome': 'done'}
        merges = [original]
        enrich_merges_with_titles(merges, {'7': 'Fix X'})
        assert 'title' not in original  # original row not mutated

    def test_int_task_id_resolved_via_str_conversion(self):
        """Integer task_id converts to str and resolves correctly."""
        from dashboard.data.merge_queue import enrich_merges_with_titles

        merges = [{'task_id': 7, 'outcome': 'done'}]  # int, not str
        result = enrich_merges_with_titles(merges, {'7': 'Fix X'})
        assert result[0]['title'] == 'Fix X'

    def test_returns_new_list(self):
        """Return value is a new list, not the input list."""
        from dashboard.data.merge_queue import enrich_merges_with_titles

        merges = [{'task_id': '7', 'outcome': 'done'}]
        result = enrich_merges_with_titles(merges, {'7': 'Fix X'})
        assert result is not merges


# ---------------------------------------------------------------------------
# TestLoadTaskTitles (step-7)
# ---------------------------------------------------------------------------


class TestLoadTaskTitles:
    """Tests for merge_queue.load_task_titles."""

    def _write_tasks_json(self, path, tasks, format='master'):
        """Write a tasks.json file in the given format."""
        import json
        data = {'master': {'tasks': tasks}} if format == 'master' else {'tasks': tasks}
        path.write_text(json.dumps(data))

    def test_master_format_returns_str_keyed_dict(self, tmp_path):
        """Reads {'master': {'tasks': [...]}} format and returns {str(id): title}."""
        from dashboard.data.merge_queue import load_task_titles

        tasks_path = tmp_path / 'tasks.json'
        self._write_tasks_json(tasks_path, [
            {'id': 1, 'title': 'A'},
            {'id': 2, 'title': 'B'},
        ])
        result = load_task_titles(tasks_path)
        assert result == {'1': 'A', '2': 'B'}

    def test_flat_format_also_works(self, tmp_path):
        """Reads {'tasks': [...]} format correctly."""
        from dashboard.data.merge_queue import load_task_titles

        tasks_path = tmp_path / 'tasks.json'
        self._write_tasks_json(tasks_path, [
            {'id': 1, 'title': 'A'},
            {'id': 2, 'title': 'B'},
        ], format='flat')
        result = load_task_titles(tasks_path)
        assert result == {'1': 'A', '2': 'B'}

    def test_missing_file_returns_empty_dict(self, tmp_path):
        """Returns {} when the file does not exist."""
        from dashboard.data.merge_queue import load_task_titles

        result = load_task_titles(tmp_path / 'nonexistent.json')
        assert result == {}

    def test_malformed_json_returns_empty_dict(self, tmp_path):
        """Returns {} for invalid JSON."""
        from dashboard.data.merge_queue import load_task_titles

        tasks_path = tmp_path / 'tasks.json'
        tasks_path.write_text('{ invalid json ]')
        result = load_task_titles(tasks_path)
        assert result == {}

    def test_none_title_is_omitted(self, tmp_path):
        """Tasks with title=None are omitted from the result."""
        from dashboard.data.merge_queue import load_task_titles

        tasks_path = tmp_path / 'tasks.json'
        self._write_tasks_json(tasks_path, [
            {'id': 1, 'title': None},
            {'id': 2, 'title': 'B'},
        ])
        result = load_task_titles(tasks_path)
        assert '1' not in result
        assert result.get('2') == 'B'

    def test_repeat_calls_cached_via_mtime(self, tmp_path, monkeypatch):
        """Two calls with unchanged mtime invoke load_task_tree exactly once."""
        import dashboard.data.merge_queue as _mq
        from dashboard.data.merge_queue import (
            _load_task_titles_cached,
            load_task_titles,
        )

        _load_task_titles_cached.cache_clear()

        tasks_path = tmp_path / 'tasks.json'
        self._write_tasks_json(tasks_path, [{'id': 1, 'title': 'A'}])

        call_count = 0
        original_load_task_tree = _mq.load_task_tree

        def counting_load_task_tree(path):
            nonlocal call_count
            call_count += 1
            return original_load_task_tree(path)

        monkeypatch.setattr(_mq, 'load_task_tree', counting_load_task_tree)

        result1 = load_task_titles(tasks_path)
        result2 = load_task_titles(tasks_path)

        assert call_count == 1, f"load_task_tree called {call_count} times, expected 1"
        assert result1 == {'1': 'A'}
        assert result2 == {'1': 'A'}


# ---------------------------------------------------------------------------
# TestBuildPerProjectMergeQueue (step-9)
# ---------------------------------------------------------------------------


class TestBuildPerProjectMergeQueue:
    """Tests for merge_queue.build_per_project_merge_queue."""

    async def test_returns_dict_keyed_by_pid(self, tmp_path):
        """Result dict has one entry per (pid, db) pair."""
        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        db1_path = _make_db(tmp_path, 'a.db', [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=5),
             'task_id': 'task-A', 'run_id': 'rA', 'data': {'outcome': 'done'}, 'duration_ms': 1000},
        ])
        db2_path = _make_db(tmp_path, 'b.db', [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=7),
             'task_id': 'task-B', 'run_id': 'rB', 'data': {'outcome': 'conflict'}, 'duration_ms': 2000},
        ])

        async with (
            aiosqlite.connect(str(db1_path)) as conn1,
            aiosqlite.connect(str(db2_path)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            project_dbs = [('/tmp/A', conn1), ('/tmp/B', conn2)]
            result = await build_per_project_merge_queue(
                project_dbs, hours=24, now=now, recent_window_minutes=15,
            )

        # (a) dict with both pid keys
        assert isinstance(result, dict)
        assert set(result.keys()) == {'/tmp/A', '/tmp/B'}

        # (b) each value has the expected keys
        for pid_data in result.values():
            assert set(pid_data.keys()) >= {'depth_timeseries', 'outcomes', 'latency', 'recent', 'speculative'}

    async def test_per_project_isolation(self, tmp_path):
        """Each project's stats reflect only its own DB rows."""
        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        db1_path = _make_db(tmp_path, 'a.db', [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=5),
             'task_id': 'task-A1', 'run_id': 'rA1', 'data': {'outcome': 'done'}, 'duration_ms': 500},
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=6),
             'task_id': 'task-A2', 'run_id': 'rA2', 'data': {'outcome': 'done'}, 'duration_ms': 600},
        ])
        db2_path = _make_db(tmp_path, 'b.db', [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=3),
             'task_id': 'task-B1', 'run_id': 'rB1', 'data': {'outcome': 'conflict'}, 'duration_ms': 1000},
        ])

        async with (
            aiosqlite.connect(str(db1_path)) as conn1,
            aiosqlite.connect(str(db2_path)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            project_dbs = [('/tmp/A', conn1), ('/tmp/B', conn2)]
            result = await build_per_project_merge_queue(
                project_dbs, hours=24, now=now, recent_window_minutes=15,
            )

        # (c) '/tmp/A' stats reflect only db1's rows (2 attempts)
        a_recent = result['/tmp/A']['recent']
        b_recent = result['/tmp/B']['recent']
        a_task_ids = {r['task_id'] for r in a_recent}
        b_task_ids = {r['task_id'] for r in b_recent}
        assert 'task-A1' in a_task_ids
        assert 'task-A2' in a_task_ids
        assert 'task-B1' not in a_task_ids
        assert 'task-B1' in b_task_ids

    async def test_recent_trimmed_to_window(self, tmp_path):
        """The recent list is already filtered to recent_window_minutes."""
        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        db_path = _make_db(tmp_path, 'a.db', [
            # within window (5 min ago)
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=5),
             'task_id': 'in-window', 'run_id': 'r1', 'data': {'outcome': 'done'}, 'duration_ms': 1000},
            # outside window (30 min ago)
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=30),
             'task_id': 'out-window', 'run_id': 'r2', 'data': {'outcome': 'done'}, 'duration_ms': 1000},
        ])

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            project_dbs = [('/tmp/A', conn)]
            result = await build_per_project_merge_queue(
                project_dbs, hours=24, now=now, recent_window_minutes=15,
            )

        # (d) only in-window row survives
        recent = result['/tmp/A']['recent']
        task_ids = {r['task_id'] for r in recent}
        assert 'in-window' in task_ids
        assert 'out-window' not in task_ids

    async def test_none_db_entry_skipped(self, tmp_path):
        """A (pid, None) pair results in empty/default stats for that project."""
        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        project_dbs = [('/tmp/nofile', None)]
        result = await build_per_project_merge_queue(
            project_dbs, hours=24, now=now, recent_window_minutes=15,
        )

        assert '/tmp/nofile' in result
        data = result['/tmp/nofile']
        assert data['latency']['count'] == 0
        assert data['recent'] == []

    async def test_mixed_real_and_none_dbs(self, tmp_path):
        """Mixed (pid, real_conn) and (pid, None) entries all appear in the result.

        Guards the parallel gather path: a None-db project must not raise and
        must not suppress or crash the real-db project alongside it.
        """
        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        db_path = _make_db(tmp_path, 'real.db', [])

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            project_dbs = [
                ('/tmp/real', conn),
                ('/tmp/none', None),
            ]
            result = await build_per_project_merge_queue(
                project_dbs, hours=24, now=now, recent_window_minutes=15,
            )

        # Both pids present — None entry must not be dropped
        assert set(result.keys()) == {'/tmp/real', '/tmp/none'}

        # None-db entry produces defaults without crashing
        none_data = result['/tmp/none']
        assert none_data['latency']['count'] == 0
        assert none_data['recent'] == []

        # Real-db entry has the expected top-level keys
        real_data = result['/tmp/real']
        assert set(real_data.keys()) >= {'depth_timeseries', 'outcomes', 'latency', 'recent', 'speculative'}

    async def test_per_project_queries_run_concurrently(self, tmp_path):
        """All N per-project gathers must run concurrently (peak-in-flight == N).

        Patches ``outcome_distribution`` with a fake that:
        - Increments an in-flight counter on entry and tracks the peak.
        - Sets ``all_entered`` when in_flight reaches N.
        - Blocks on a ``release`` event before returning.

        On the sequential for-loop implementation only one project enters at a
        time so ``all_entered`` is never set → TimeoutError → test fails.
        On the parallel ``asyncio.gather`` implementation all N enter before
        any returns → ``all_entered`` fires → test passes.
        """
        from dashboard.data.merge_queue import build_per_project_merge_queue

        N = 3
        now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        db_paths = [_make_db(tmp_path, f'c{i}.db', []) for i in range(N)]

        all_entered = asyncio.Event()
        release = asyncio.Event()
        counter = [0]        # mutable via list to allow mutation in closure
        max_in_flight = [0]

        async def fake_outcome_distribution(db, *, hours=24, now=None):
            counter[0] += 1
            if counter[0] > max_in_flight[0]:
                max_in_flight[0] = counter[0]
            if counter[0] == N:
                all_entered.set()
            await release.wait()
            counter[0] -= 1
            return {'labels': [], 'values': []}

        async with (
            aiosqlite.connect(str(db_paths[0])) as c0,
            aiosqlite.connect(str(db_paths[1])) as c1,
            aiosqlite.connect(str(db_paths[2])) as c2,
        ):
            c0.row_factory = aiosqlite.Row
            c1.row_factory = aiosqlite.Row
            c2.row_factory = aiosqlite.Row
            project_dbs = [(f'/tmp/P{i}', c) for i, c in enumerate([c0, c1, c2])]

            with patch(
                'dashboard.data.merge_queue.outcome_distribution',
                new=fake_outcome_distribution,
            ):
                task = asyncio.create_task(
                    build_per_project_merge_queue(
                        project_dbs, hours=24, now=now, recent_window_minutes=15,
                    )
                )
                try:
                    # Fails (TimeoutError) on sequential implementation;
                    # succeeds immediately on parallel implementation.
                    await asyncio.wait_for(all_entered.wait(), timeout=2.0)
                except TimeoutError:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    raise
                release.set()
                result = await task

        assert max_in_flight[0] == N, (
            f'Expected {N} per-project gathers in-flight concurrently, '
            f'but peak was {max_in_flight[0]}'
        )
        assert set(result.keys()) == {f'/tmp/P{i}' for i in range(N)}

