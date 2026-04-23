"""Tests for dashboard.data.merge_queue — merge queue query functions."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

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


@pytest.fixture()
def counted_load_task_tree(monkeypatch):
    """Patches dashboard.data.merge_queue.load_task_tree with a counting wrapper.

    Yields an object with a ``count`` attribute that increments on every call
    to the patched function. The original is restored automatically via
    ``monkeypatch`` teardown.
    """
    import dashboard.data.merge_queue as _mq

    class _Counter:
        count = 0

    counter = _Counter()
    original = _mq.load_task_tree

    def _counting(path):
        counter.count += 1
        return original(path)

    monkeypatch.setattr(_mq, 'load_task_tree', _counting)
    return counter


# ---------------------------------------------------------------------------
# Imports under test (deferred so the test file fails gracefully before impl)
# ---------------------------------------------------------------------------

from dashboard.data.merge_queue import (  # noqa: E402
    _align_bucket,
    _bucket_minutes_for_window,
    _cutoff_iso,
    _get_durations,
    _ts_sort_key,
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

    @pytest.mark.asyncio
    async def test_limit_none_returns_all_rows(self, merge_events_db, caplog):
        """recent_merges with limit=None returns every matching row up to _RECENT_MERGES_HARD_CAP.

        Inserts 60 merge_attempt events all within the last 5 minutes, then
        asserts that limit=None returns all 60 (60 < 100_000 default hard cap,
        so no truncation occurs) and that no WARNING is emitted (the 'below
        cap → silent' branch).
        """
        now = datetime.now(UTC)
        with contextlib.closing(sqlite3.connect(str(merge_events_db))) as conn_sync:
            for i in range(60):
                _insert_event(
                    conn_sync,
                    event_type='merge_attempt',
                    timestamp=now - timedelta(seconds=i * 5),
                    task_id=f'burst-{i:03d}',
                    run_id=f'run-burst-{i:03d}',
                    data={'outcome': 'done'},
                    duration_ms=100 + i,
                )
            conn_sync.commit()

        with caplog.at_level(logging.WARNING, logger='dashboard.data.merge_queue'):
            async with aiosqlite.connect(str(merge_events_db)) as db:
                db.row_factory = aiosqlite.Row
                result = await recent_merges(db, limit=None, hours=1)

        assert len(result) == 60, (
            f'Expected 60 rows with limit=None, got {len(result)}. '
            'limit=None returns every matching row up to _RECENT_MERGES_HARD_CAP '
            '(no truncation for normal-sized result sets).'
        )
        warn_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warn_records, (
            f'Expected no WARNING for 60 rows (well below hard cap), '
            f'but got: {[r.message for r in warn_records]}'
        )

    @pytest.mark.asyncio
    async def test_recent_merges_limit_none_hard_cap_truncates_and_warns(
        self, tmp_path, monkeypatch, caplog
    ):
        """recent_merges(limit=None) truncates to _RECENT_MERGES_HARD_CAP rows and logs WARN.

        Patches _RECENT_MERGES_HARD_CAP to 5, inserts 10 events (all within the
        1-hour window), and asserts:
          1. Only 5 rows are returned (truncation to the patched cap).
          2. At least one WARNING log record contains 'hard cap' (or 'hard_cap')
             so operators can identify the cap was hit.
        """
        monkeypatch.setattr(_mqmod, '_RECENT_MERGES_HARD_CAP', 5)

        now = datetime.now(UTC)
        db_path = tmp_path / 'hard_cap_test.db'

        with contextlib.closing(sqlite3.connect(str(db_path))) as conn_sync:
            conn_sync.executescript(MERGE_EVENTS_SCHEMA)
            for i in range(10):
                _insert_event(
                    conn_sync,
                    event_type='merge_attempt',
                    timestamp=now - timedelta(seconds=i * 5),
                    task_id=f'cap-task-{i:03d}',
                    run_id=f'cap-run-{i:03d}',
                    data={'outcome': 'done'},
                    duration_ms=100 + i,
                )
            conn_sync.commit()

        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row
            with caplog.at_level(logging.WARNING, logger='dashboard.data.merge_queue'):
                result = await recent_merges(db, limit=None, hours=1)

        assert len(result) == 5, (
            f'Expected 5 rows (hard cap=5), got {len(result)}. '
            'recent_merges(limit=None) must truncate to _RECENT_MERGES_HARD_CAP rows.'
        )
        warn_msgs = [
            r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING
        ]
        assert any('hard cap' in m or 'hard_cap' in m for m in warn_msgs), (
            f'Expected a WARNING containing "hard cap" or "hard_cap", got: {warn_msgs}'
        )
        # The row count must appear in the warning so it is actionable.
        assert any(
            ('hard cap' in m or 'hard_cap' in m) and ('5' in m or '6' in m)
            for m in warn_msgs
        ), (
            f'Expected the row count (5 or 6) in the hard-cap WARNING, got: {warn_msgs}'
        )


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

    def test_repeat_calls_cached_via_mtime(self, tmp_path, counted_load_task_tree):
        """Two calls with unchanged mtime invoke load_task_tree exactly once."""
        from dashboard.data.merge_queue import (
            _load_task_titles_cached,
            load_task_titles,
        )

        _load_task_titles_cached.cache_clear()

        tasks_path = tmp_path / 'tasks.json'
        self._write_tasks_json(tasks_path, [{'id': 1, 'title': 'A'}])

        result1 = load_task_titles(tasks_path)
        result2 = load_task_titles(tasks_path)

        assert counted_load_task_tree.count == 1, f"load_task_tree called {counted_load_task_tree.count} times, expected 1"
        assert result1 == {'1': 'A'}
        assert result2 == {'1': 'A'}

    def test_mtime_change_invalidates_cache(self, tmp_path):
        """Bumping st_mtime_ns causes the next call to return fresh content."""
        import os

        from dashboard.data.merge_queue import _load_task_titles_cached, load_task_titles

        _load_task_titles_cached.cache_clear()

        tasks_path = tmp_path / 'tasks.json'
        self._write_tasks_json(tasks_path, [{'id': 1, 'title': 'A'}])

        result1 = load_task_titles(tasks_path)
        assert result1 == {'1': 'A'}

        # Overwrite content then bump mtime explicitly to guarantee a new key
        stat = os.stat(tasks_path)
        orig_mtime_ns = stat.st_mtime_ns
        self._write_tasks_json(tasks_path, [{'id': 1, 'title': 'Z'}])
        os.utime(tasks_path, ns=(stat.st_atime_ns, orig_mtime_ns + 1))

        result2 = load_task_titles(tasks_path)
        assert result2 == {'1': 'Z'}

    def test_missing_file_does_not_invoke_load_task_tree(self, tmp_path, counted_load_task_tree):
        """A missing file short-circuits before touching load_task_tree or the cache."""
        from dashboard.data.merge_queue import (
            _load_task_titles_cached,
            load_task_titles,
        )

        _load_task_titles_cached.cache_clear()

        result = load_task_titles(tmp_path / 'nope.json')

        assert result == {}
        assert counted_load_task_tree.count == 0, f"load_task_tree was called {counted_load_task_tree.count} time(s); expected 0"

    def test_distinct_paths_cached_separately(self, tmp_path):
        """Different task files produce independent cache entries."""
        from dashboard.data.merge_queue import _load_task_titles_cached, load_task_titles

        _load_task_titles_cached.cache_clear()

        path_a = tmp_path / 'proj_a' / 'tasks.json'
        path_b = tmp_path / 'proj_b' / 'tasks.json'
        path_a.parent.mkdir()
        path_b.parent.mkdir()
        self._write_tasks_json(path_a, [{'id': 1, 'title': 'Alpha'}])
        self._write_tasks_json(path_b, [{'id': 2, 'title': 'Beta'}])

        result_a = load_task_titles(path_a)
        result_b = load_task_titles(path_b)

        assert result_a == {'1': 'Alpha'}, f"project A returned unexpected titles: {result_a}"
        assert result_b == {'2': 'Beta'}, f"project B returned unexpected titles: {result_b}"
        assert result_a != result_b

    def test_symlink_shares_cache_entry_with_real_path(self, tmp_path, counted_load_task_tree):
        """A symlink and the real path resolve to the same LRU entry via os.path.realpath."""
        import os

        from dashboard.data.merge_queue import (
            _load_task_titles_cached,
            load_task_titles,
        )

        _load_task_titles_cached.cache_clear()

        real_dir = tmp_path / 'real'
        real_dir.mkdir()
        real_path = real_dir / 'tasks.json'
        self._write_tasks_json(real_path, [{'id': 1, 'title': 'A'}])

        link_path = tmp_path / 'link.json'
        try:
            link_path.symlink_to(real_path)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unsupported on this filesystem")

        assert os.path.realpath(link_path) == os.path.realpath(real_path), (
            f"symlink {link_path!r} and real {real_path!r} must share realpath before the cache test is meaningful"
        )

        result_real = load_task_titles(real_path)
        result_link = load_task_titles(link_path)

        assert result_real == {'1': 'A'}, f"real path returned unexpected titles: {result_real}"
        assert result_link == {'1': 'A'}, f"symlink returned unexpected titles: {result_link}"
        assert counted_load_task_tree.count == 1, (
            f"load_task_tree called {counted_load_task_tree.count} time(s); expected 1 "
            "(os.path.realpath should collapse both spellings to one cache key)"
        )

    def test_corrupt_json_then_valid_refreshes_cache(self, tmp_path, counted_load_task_tree):
        """Empty-dict from corrupt JSON is cached and then invalidated when mtime bumps.

        Verifies two distinct behaviors:
        1. {} is a legitimate cached value (not a cache-skipping sentinel): a second call
           with unchanged mtime must not re-invoke load_task_tree.
        2. Bumping st_mtime_ns by 1s causes the next call to pick up the valid content,
           distinct from test_mtime_change_invalidates_cache where both sides are valid.
        """
        import os

        from dashboard.data.merge_queue import _load_task_titles_cached, load_task_titles

        _load_task_titles_cached.cache_clear()

        tasks_path = tmp_path / 'tasks.json'
        tasks_path.write_text('{ invalid json ]')

        result_corrupt = load_task_titles(tasks_path)
        assert result_corrupt == {}, f"corrupt JSON should return {{}}, got {result_corrupt!r}"

        # Second call with unchanged mtime must hit the cache — {} is not a sentinel
        result_corrupt2 = load_task_titles(tasks_path)
        assert result_corrupt2 == {}
        assert counted_load_task_tree.count == 1, (
            f"load_task_tree called {counted_load_task_tree.count} time(s) for two reads with unchanged mtime; "
            "expected 1 (empty dict must be a legitimate cached value, not a sentinel)"
        )

        # Overwrite with valid content and bump mtime by 1s to guarantee a distinct cache key.
        # Capture stat before the overwrite so the bump is relative to the corrupt-JSON mtime;
        # 1_000_000_000 ns = 1 s ensures the delta is honored even on second-resolution filesystems.
        stat = os.stat(tasks_path)
        self._write_tasks_json(tasks_path, [{'id': 1, 'title': 'A'}])
        os.utime(tasks_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

        result_valid = load_task_titles(tasks_path)
        assert result_valid == {'1': 'A'}, (
            f"after mtime bump to valid content, expected {{'1': 'A'}}, got {result_valid!r}"
        )


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
        """A (pid, None) pair yields the declared full-default shape (all 5 keys,
        each matching its _DEFAULT_* constant) for that project."""
        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        project_dbs = [('/tmp/nofile', None)]
        result = await build_per_project_merge_queue(
            project_dbs, hours=24, now=now, recent_window_minutes=15,
        )

        assert '/tmp/nofile' in result
        data = result['/tmp/nofile']
        assert set(data.keys()) >= {'depth_timeseries', 'outcomes', 'latency', 'recent', 'speculative'}
        assert data['depth_timeseries'] == {'labels': [], 'values': []}
        assert data['outcomes'] == {'labels': [], 'values': []}
        assert data['latency'] == {'p50': 0, 'p95': 0, 'p99': 0, 'count': 0, 'mean_ms': 0.0}
        assert data['recent'] == []
        assert data['speculative'] == {'hit_count': 0, 'discard_count': 0, 'total': 0, 'hit_rate': 0.0}

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

    @pytest.mark.asyncio
    @pytest.mark.parametrize('recent_window_minutes,expected_hours', [
        (15, 1),
        (60, 1),
        (90, 2),
        (120, 2),
    ])
    async def test_recent_merges_called_with_ceil_hours_and_no_limit(
        self, tmp_path, recent_window_minutes, expected_hours,
    ):
        """build_per_project_merge_queue calls recent_merges with hours=ceil(window/60) and limit=None.

        Parametrized over recent_window_minutes ∈ {15, 60, 90, 120} to lock in
        the max(1, ceil(x/60)) semantics.  Fails on the pre-fix implementation
        which passes limit=50 and hours=<outer dashboard window>.
        """
        from unittest.mock import patch

        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)
        db_path = _make_db(tmp_path, 'p.db', [])

        captured_kwargs: dict = {}

        async def fake_recent_merges(db, **kwargs):
            captured_kwargs.update(kwargs)
            return []

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            with patch('dashboard.data.merge_queue.recent_merges', new=fake_recent_merges):
                await build_per_project_merge_queue(
                    [('/tmp/P', conn)],
                    hours=24,
                    now=now,
                    recent_window_minutes=recent_window_minutes,
                )

        assert captured_kwargs.get('limit') is None, (
            f'Expected limit=None, got limit={captured_kwargs.get("limit")!r}'
        )
        assert captured_kwargs.get('hours') == expected_hours, (
            f'recent_window_minutes={recent_window_minutes}: '
            f'expected hours={expected_hours}, got hours={captured_kwargs.get("hours")!r}'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('recent_window_minutes, in_window_offset_min, over_fetch_offset_min', [
        (15,  5, 30),   # SQL hours=1, Python window=15 min; over-fetch zone=(15,60]
        (30, 10, 45),   # SQL hours=1, Python window=30 min; over-fetch zone=(30,60]
        (45, 20, 50),   # SQL hours=1, Python window=45 min; over-fetch zone=(45,60]
        (90, 45, 100),  # SQL hours=2, Python window=90 min; over-fetch zone=(90,120]
    ])
    async def test_sql_over_fetches_are_trimmed_to_exact_minute_boundary(
        self, tmp_path, recent_window_minutes, in_window_offset_min, over_fetch_offset_min,
    ):
        """SQL hour-granularity over-fetches are trimmed to the exact minute boundary by filter_merges_within.

        When recent_window_minutes is not a multiple of 60, SQL uses hours=ceil(minutes/60),
        which over-fetches rows in the zone (recent_window_minutes, ceil(minutes/60)*60] minutes.
        build_per_project_merge_queue must call filter_merges_within to drop those extra rows.

        Setup: two events per case — one at -in_window_offset_min (inside Python window) and
        one at -over_fetch_offset_min (inside SQL hours window but outside Python window).
        Asserts: only the in-window event survives in result['recent'], proving the two-layer
        contract (SQL over-fetches, Python post-filter drops the excess).
        """
        from math import ceil

        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)

        # Sanity-check: over_fetch_offset_min must be in the over-fetch zone
        sql_hours = max(1, ceil(recent_window_minutes / 60))
        assert recent_window_minutes < over_fetch_offset_min <= sql_hours * 60, (
            f'Test setup error: over_fetch_offset_min={over_fetch_offset_min} not in '
            f'over-fetch zone ({recent_window_minutes}, {sql_hours * 60}]'
        )
        assert in_window_offset_min < recent_window_minutes, (
            f'Test setup error: in_window_offset_min={in_window_offset_min} not inside '
            f'Python window ({recent_window_minutes} min)'
        )

        in_window_task_id = f'in-window-{recent_window_minutes}min'
        over_fetch_task_id = f'over-fetch-{recent_window_minutes}min'

        db_path = _make_db(tmp_path, 'p.db', [
            {
                'event_type': 'merge_attempt',
                'timestamp': now - timedelta(minutes=in_window_offset_min),
                'task_id': in_window_task_id,
                'run_id': f'run-in-{recent_window_minutes}',
                'data': {'outcome': 'done'},
                'duration_ms': 200,
            },
            {
                'event_type': 'merge_attempt',
                'timestamp': now - timedelta(minutes=over_fetch_offset_min),
                'task_id': over_fetch_task_id,
                'run_id': f'run-over-{recent_window_minutes}',
                'data': {'outcome': 'done'},
                'duration_ms': 300,
            },
        ])

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await build_per_project_merge_queue(
                [('/tmp/P', conn)],
                hours=24,
                now=now,
                recent_window_minutes=recent_window_minutes,
            )

        recent = result['/tmp/P']['recent']
        assert len(recent) == 1, (
            f'recent_window_minutes={recent_window_minutes}: '
            f'expected 1 row (in-window only), got {len(recent)}. '
            f'Rows: {[r["task_id"] for r in recent]}'
        )
        assert recent[0]['task_id'] == in_window_task_id, (
            f'Expected in-window row task_id={in_window_task_id!r}, '
            f'got task_id={recent[0]["task_id"]!r}'
        )
        over_fetch_ids = [r['task_id'] for r in recent if r['task_id'] == over_fetch_task_id]
        assert not over_fetch_ids, (
            f'Over-fetch row {over_fetch_task_id!r} survived filter_merges_within — '
            f'the SQL+Python two-layer contract is broken (recent_window_minutes={recent_window_minutes})'
        )

    @pytest.mark.asyncio
    async def test_burst_exceeding_50_within_window_not_dropped(self, tmp_path):
        """60 merge_attempt events within the 15-min window are all returned (none silently dropped).

        This is the end-to-end regression gate: the old limit=50 call would silently
        truncate a burst of >50 events even if every event was within recent_window_minutes.
        With the fix (limit=None + SQL hours window), all 60 events survive the pipeline.
        """
        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)
        # 60 events spread across the last ~14 minutes (14s apart), all within 15-min window.
        events = [
            {
                'event_type': 'merge_attempt',
                'timestamp': now - timedelta(seconds=i * 14),
                'task_id': f'burst-task-{i:03d}',
                'run_id': f'burst-run-{i:03d}',
                'data': {'outcome': 'done'},
                'duration_ms': 500 + i,
            }
            for i in range(60)
        ]
        db_path = _make_db(tmp_path, 'burst.db', events)

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await build_per_project_merge_queue(
                [('/tmp/P', conn)],
                hours=24,
                now=now,
                recent_window_minutes=15,
            )

        recent = result['/tmp/P']['recent']
        assert len(recent) == 60, (
            f'Expected 60 rows in recent burst, got {len(recent)}. '
            'The SQL LIMIT cap must not silently truncate burst events within the window.'
        )
        returned_task_ids = {r['task_id'] for r in recent}
        expected_task_ids = {f'burst-task-{i:03d}' for i in range(60)}
        assert returned_task_ids == expected_task_ids, (
            f'Missing task_ids: {expected_task_ids - returned_task_ids}'
        )

    @pytest.mark.asyncio
    async def test_build_per_project_burst_warn_uses_module_constant(
        self, tmp_path, monkeypatch, caplog
    ):
        """_one_project WARN references _RECENT_MERGES_BURST_WARN, not the bare 1_000 literal.

        Patches _RECENT_MERGES_BURST_WARN to 5, inserts 10 events all within the
        15-minute recent_window_minutes window (above the patched soft threshold,
        well below the 100_000 hard cap), and asserts:
          1. At least one WARNING log record contains 'runaway burst'.
          2. The same record mentions the row count (10).
        Fails before step-4 (bare 1_000 literal ignores the patched constant),
        passes after step-4 (_RECENT_MERGES_BURST_WARN drives the comparison).
        """
        monkeypatch.setattr(_mqmod, '_RECENT_MERGES_BURST_WARN', 5)

        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)
        # 10 events spread across ~9 minutes (56 s apart), all within the 15-min window.
        events = [
            {
                'event_type': 'merge_attempt',
                'timestamp': now - timedelta(seconds=i * 56),
                'task_id': f'bw-task-{i:03d}',
                'run_id': f'bw-run-{i:03d}',
                'data': {'outcome': 'done'},
                'duration_ms': 100 + i,
            }
            for i in range(10)
        ]
        db_path = _make_db(tmp_path, 'burst_warn.db', events)

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            with caplog.at_level(logging.WARNING, logger='dashboard.data.merge_queue'):
                await build_per_project_merge_queue(
                    [('/tmp/P', conn)],
                    hours=24,
                    now=now,
                    recent_window_minutes=15,
                )

        warn_msgs = [
            r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING
        ]
        assert any('runaway burst' in m for m in warn_msgs), (
            f'Expected a WARNING containing "runaway burst", got: {warn_msgs!r}. '
            'Ensure _RECENT_MERGES_BURST_WARN (patched to 5) controls the threshold '
            'in _one_project, not the bare 1_000 literal.'
        )
        assert any('runaway burst' in m and '10' in m for m in warn_msgs), (
            f'Expected the row count (10) in the runaway-burst WARNING, got: {warn_msgs!r}'
        )

    @pytest.mark.asyncio
    async def test_cancelled_error_from_sub_query_propagates(self, tmp_path):
        """CancelledError raised inside a sub-query must propagate out of build_per_project_merge_queue.

        The inner _safe closure (pre-step-5) treats all BaseException subclasses —
        including CancelledError — as recoverable and returns the default, silently
        swallowing asyncio cancellation.  After step-5 swaps in safe_gather_result,
        CancelledError re-raises through _one_project's ``except Exception`` guard
        (which only catches Exception, not BaseException) and propagates out of
        the outer asyncio.gather call.

        This test pins the corrected behaviour: any CancelledError from a sub-query
        must NOT be swallowed.
        """
        import asyncio
        from unittest.mock import patch

        import pytest

        from dashboard.data.merge_queue import build_per_project_merge_queue

        async def _raise_cancelled(*_args, **_kwargs):
            raise asyncio.CancelledError('shutdown')

        db_path = _make_db(tmp_path, 'x.db', [])
        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            with patch(
                'dashboard.data.merge_queue.queue_depth_timeseries',
                side_effect=_raise_cancelled,
            ):
                with pytest.raises(asyncio.CancelledError):
                    await build_per_project_merge_queue(
                        [('/tmp/P', conn)],
                        hours=24,
                        now=now,
                        recent_window_minutes=15,
                    )

