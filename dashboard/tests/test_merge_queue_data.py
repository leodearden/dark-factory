"""Tests for dashboard.data.merge_queue — merge queue query functions."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import aiosqlite
import httpx
import pytest
from _dashboard_helpers import (
    cold_session_responses,
    mcp_init_response,
    mcp_notify_response,
    mcp_tool_response,
)

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
async def tableless_runs_conn(tmp_path):
    """A readable runs.db with NO ``events`` table: every query raises OperationalError."""
    db_path = tmp_path / 'tableless_runs.db'
    sqlite3.connect(str(db_path)).close()
    async with aiosqlite.connect(str(db_path)) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


# ---------------------------------------------------------------------------
# Imports under test (deferred so the test file fails gracefully before impl)
# ---------------------------------------------------------------------------

from dashboard.data import merge_queue  # noqa: E402
from dashboard.data.datum import Datum, DatumState, validate_datum  # noqa: E402
from dashboard.data.merge_queue import (  # noqa: E402
    RECENT_MERGES_CAP,
    _align_bucket,
    _bucket_minutes_for_window,
    _cutoff_iso,
    _ts_sort_key,
    enrich_merges_with_titles,
    merge_attempts,
    merge_task_refs,
    queue_depth_timeseries,
    recent_merges,
    resolve_active,
    speculative_stats,
)
from dashboard.data.stats_utils import percentile  # noqa: E402
from dashboard.data.task_lookup import TaskRef  # noqa: E402

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
# TestMergeAttempts — ONE query feeds the outcome chart and the latency block
# ---------------------------------------------------------------------------

ATTEMPTS_NOW = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
"""The injected instant every ``merge_attempts`` case reads its window against."""

ZERO_LATENCY = {
    'p50': 0, 'p95': 0, 'p99': 0, 'mean_ms': 0.0,
    'with_duration': 0, 'without_duration': 0,
}


def _attempt(minutes_ago, outcome, duration_ms=None):
    """One merge_attempt event *minutes_ago* before :data:`ATTEMPTS_NOW`.

    An *outcome* of None writes no ``outcome`` key at all, which is how the
    substrate spells an attempt whose outcome was never recorded.
    """
    return {
        'event_type': 'merge_attempt',
        'timestamp': ATTEMPTS_NOW - timedelta(minutes=minutes_ago),
        'data': {} if outcome is None else {'outcome': outcome},
        'duration_ms': duration_ms,
    }


FIVE_ATTEMPTS = [
    _attempt(5, 'done', 300),
    _attempt(6, 'done', 100),
    _attempt(7, 'conflict', 200),
    _attempt(8, 'conflict', None),
    _attempt(9, 'blocked', 0),
]
"""Three attempts with a positive duration, one NULL and one zero."""


async def _merge_attempts_over(tmp_path, events, *, hours=24):
    db_path = _make_db(tmp_path, 'attempts.db', events)
    async with aiosqlite.connect(str(db_path)) as conn:
        conn.row_factory = aiosqlite.Row
        return await merge_attempts(conn, hours=hours, now=ATTEMPTS_NOW)


class TestMergeAttempts:
    """``merge_attempts`` reads the window once; the chart and latency agree."""

    async def test_the_chart_counts_every_attempt(self, tmp_path):
        result = await _merge_attempts_over(tmp_path, FIVE_ATTEMPTS)

        assert result.outcome_chart() == {
            'labels': ['conflict', 'done', 'blocked'], 'values': [2, 2, 1],
        }

    async def test_latency_reads_only_the_attempts_with_a_duration(self, tmp_path):
        result = await _merge_attempts_over(tmp_path, FIVE_ATTEMPTS)

        latency = result.latency()
        assert latency['with_duration'] == 3
        assert latency['without_duration'] == 2
        assert latency['p50'] == 200
        assert latency['mean_ms'] == pytest.approx(200.0)
        assert latency['p95'] == round(percentile([100.0, 200.0, 300.0], 95))
        assert latency['p99'] == round(percentile([100.0, 200.0, 300.0], 99))

    async def test_every_attempt_is_counted_once_in_the_latency_split(self, tmp_path):
        """Sketch #9: the donut total IS the latency block's attempt total."""
        result = await _merge_attempts_over(tmp_path, FIVE_ATTEMPTS)

        latency = result.latency()
        assert sum(result.outcome_chart()['values']) == (
            latency['with_duration'] + latency['without_duration']
        )

    async def test_durations_are_held_sorted(self, tmp_path):
        """Timestamps run opposite to durations, so only a sort orders them."""
        events = [
            _attempt(1, 'done', 500), _attempt(2, 'done', 400),
            _attempt(3, 'done', 300), _attempt(4, 'done', 200),
            _attempt(5, 'done', 100),
        ]
        result = await _merge_attempts_over(tmp_path, events)

        assert result.durations == (100.0, 200.0, 300.0, 400.0, 500.0)

    async def test_populated_outcomes(self, tmp_path):
        outcomes = ['done'] * 3 + ['conflict'] * 2 + ['blocked', 'already_merged']
        result = await _merge_attempts_over(
            tmp_path, [_attempt(10, outcome) for outcome in outcomes],
        )

        chart = result.outcome_chart()
        assert sum(chart['values']) == 7
        assert dict(zip(chart['labels'], chart['values'], strict=True)) == {
            'done': 3, 'conflict': 2, 'blocked': 1, 'already_merged': 1,
        }

    async def test_count_descending_with_alpha_tiebreak(self, tmp_path):
        outcomes = (
            ['conflict'] * 3 + ['blocked'] * 2 + ['done'] * 2 + ['zzz', 'aaa']
        )
        result = await _merge_attempts_over(
            tmp_path, [_attempt(5, outcome) for outcome in outcomes],
        )

        assert result.outcome_chart() == {
            'labels': ['conflict', 'blocked', 'done', 'aaa', 'zzz'],
            'values': [3, 2, 2, 1, 1],
        }

    async def test_equal_counts_sorted_alphabetically(self, tmp_path):
        outcomes = ['already_merged', 'done', 'conflict', 'blocked']
        result = await _merge_attempts_over(
            tmp_path, [_attempt(5, outcome) for outcome in outcomes],
        )

        assert result.outcome_chart()['labels'] == [
            'already_merged', 'blocked', 'conflict', 'done',
        ]

    async def test_non_canonical_outcomes_are_counted(self, tmp_path):
        outcomes = ['done', 'done', 'wip_halted', 'done_wip_recovery']
        result = await _merge_attempts_over(
            tmp_path, [_attempt(5, outcome) for outcome in outcomes],
        )

        assert result.outcome_chart() == {
            'labels': ['done', 'done_wip_recovery', 'wip_halted'],
            'values': [2, 1, 1],
        }

    async def test_an_unrecorded_outcome_counts_as_unknown(self, tmp_path):
        result = await _merge_attempts_over(
            tmp_path, [_attempt(5, None, 100), _attempt(6, 'done', 200)],
        )

        assert result.outcome_chart() == {
            'labels': ['done', 'unknown'], 'values': [1, 1],
        }

    async def test_ten_durations(self, tmp_path):
        events = [
            _attempt(10 + i, 'done', ms) for i, ms in enumerate(range(100, 1100, 100))
        ]
        latency = (await _merge_attempts_over(tmp_path, events)).latency()

        assert set(latency) == set(ZERO_LATENCY)
        assert latency['with_duration'] == 10
        assert latency['without_duration'] == 0
        assert latency['mean_ms'] == pytest.approx(550.0, abs=1e-6)
        assert latency['p50'] == pytest.approx(550.0, abs=1.0)
        assert latency['p95'] > latency['p50']
        assert latency['p99'] >= latency['p95']

    async def test_all_null_durations_give_zero_latency_over_every_attempt(self, tmp_path):
        events = [_attempt(i + 1, 'done', None) for i in range(3)]
        result = await _merge_attempts_over(tmp_path, events)

        assert result.latency() == {**ZERO_LATENCY, 'without_duration': 3}
        assert sum(result.outcome_chart()['values']) == 3

    async def test_no_db_is_an_empty_record(self):
        result = await merge_attempts(None, hours=24, now=ATTEMPTS_NOW)

        assert result.outcome_chart() == {'labels': [], 'values': []}
        assert result.latency() == ZERO_LATENCY

    async def test_an_empty_db_is_an_empty_record(self, empty_merge_events_conn):
        result = await merge_attempts(
            empty_merge_events_conn, hours=24, now=ATTEMPTS_NOW,
        )

        assert result.outcome_chart() == {'labels': [], 'values': []}
        assert result.latency() == ZERO_LATENCY

    async def test_attempts_older_than_the_window_are_excluded(self, tmp_path):
        events = [
            _attempt(30, 'done', 100),
            _attempt(90, 'conflict', 200),
            _attempt(95, 'conflict', None),
        ]
        result = await _merge_attempts_over(tmp_path, events, hours=1)

        assert result.outcome_chart() == {'labels': ['done'], 'values': [1]}
        assert result.latency()['with_duration'] == 1
        assert result.latency()['without_duration'] == 0


# ---------------------------------------------------------------------------
# TestRecentMerges
# ---------------------------------------------------------------------------

class TestRecentMerges:
    """``recent_merges`` returns the newest ``limit`` rows of the window plus its total."""

    @pytest.mark.asyncio
    async def test_populated(self, merge_events_db):
        """Newest ``limit`` rows first, and ``total`` counts the whole window."""
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
            result = await recent_merges(db, limit=20, hours=24, now=now)

        assert len(result['rows']) == 20
        # Ordered by timestamp DESC (newest first → task-024 is first)
        assert result['rows'][0]['task_id'] == 'task-024'
        assert {'task_id', 'outcome', 'duration_ms', 'timestamp', 'run_id'} <= set(
            result['rows'][0].keys()
        )
        total = result['total']
        assert total.state is DatumState.FRESH
        assert total.value == 25
        assert total.as_of == now
        assert total.freshness_bound_seconds == merge_queue.RUNS_DB_READ_FRESHNESS_BOUND_SECONDS
        validate_datum(total, now)

    @pytest.mark.asyncio
    async def test_none_db(self):
        """No runs.db open: no rows, and a window total nobody measured."""
        now = datetime.now(UTC)
        result = await recent_merges(None, limit=20, hours=24, now=now)
        assert result['rows'] == []
        assert result['total'].state is DatumState.UNKNOWN
        assert 'no runs.db is open' in (result['total'].reason or '')
        validate_datum(result['total'], now)

    @pytest.mark.asyncio
    async def test_empty_db(self, empty_merge_events_conn):
        """A readable empty window is a measured zero."""
        now = datetime.now(UTC)
        result = await recent_merges(empty_merge_events_conn, limit=20, hours=24, now=now)
        assert result['rows'] == []
        assert result['total'].state is DatumState.FRESH
        assert result['total'].value == 0
        validate_datum(result['total'], now)

    @pytest.mark.asyncio
    async def test_a_failed_read_has_an_unknown_total(self, tableless_runs_conn):
        now = datetime.now(UTC)
        result = await recent_merges(tableless_runs_conn, limit=20, hours=24, now=now)
        assert result['rows'] == []
        assert result['total'].state is DatumState.UNKNOWN
        assert (result['total'].reason or '').strip()
        validate_datum(result['total'], now)

    @pytest.mark.asyncio
    async def test_custom_limit(self, merge_events_db):
        """limit=5 returns 5 rows while ``total`` still counts every in-window row."""
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
            result = await recent_merges(db, limit=5, hours=24, now=now)

        assert len(result['rows']) == 5
        assert result['total'].value == 10

    @pytest.mark.asyncio
    @pytest.mark.parametrize('limit', [0, -1])
    async def test_non_positive_limit_is_refused(self, empty_merge_events_conn, limit):
        """With LIMIT 0 the same-statement total is unobservable, so it is refused."""
        with pytest.raises(ValueError):
            await recent_merges(empty_merge_events_conn, limit=limit, hours=24)

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
            result = await recent_merges(db, limit=20, hours=1, now=now)

        # The event is 3 h old in UTC — well outside the 1-h window.
        # With correct UTC comparison it should not appear; with string
        # comparison it is incorrectly included.
        assert len(result['rows']) == 0, (
            f"Event stored as '{ts_non_utc}' (= event_utc {event_utc.isoformat()}) "
            "was included by the 1-hour filter despite being 3 h before the cutoff. "
            "This is the known SQLite string-comparison limitation."
        )

    @pytest.mark.asyncio
    async def test_hours_window_excludes_old_events(self, merge_events_db):
        """hours=1 excludes 3h-old events from BOTH the rows and the total."""
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
            result = await recent_merges(db, limit=20, hours=1, now=now)

        assert len(result['rows']) == 3
        task_ids = [r['task_id'] for r in result['rows']]
        assert all(tid.startswith('recent-') for tid in task_ids)
        assert result['total'].value == 3


def test_recent_merges_cap_is_two_hundred():
    """PRD open question 5, decided at the suggested value."""
    assert RECENT_MERGES_CAP == 200


# ---------------------------------------------------------------------------
# TestSpeculativeStats
# ---------------------------------------------------------------------------

async def _speculative_over(db_path, events, *, now):
    conn_sync = sqlite3.connect(str(db_path))
    for evt in events:
        _insert_event(conn_sync, **evt)
    conn_sync.commit()
    conn_sync.close()
    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row
        return await speculative_stats(db, hours=24, now=now)


def _speculative_events(event_type, count, *, now, first_minutes_ago=1):
    return [
        {'event_type': event_type, 'timestamp': now - timedelta(minutes=i + first_minutes_ago)}
        for i in range(count)
    ]


class TestSpeculativeStats:
    @pytest.mark.asyncio
    async def test_populated(self, merge_events_db):
        """A readable window is one FRESH Datum carrying the counts and their rate."""
        now = datetime.now(UTC)
        datum = await _speculative_over(
            merge_events_db,
            _speculative_events('speculative_merge', 5, now=now)
            + _speculative_events('speculative_discard', 3, now=now, first_minutes_ago=10),
            now=now,
        )

        assert isinstance(datum, Datum)
        assert datum.state is DatumState.FRESH
        assert datum.reason is None
        assert datum.as_of == now
        assert datum.freshness_bound_seconds == merge_queue.RUNS_DB_READ_FRESHNESS_BOUND_SECONDS
        assert datum.value == {'hit_count': 5, 'discard_count': 3, 'total': 8, 'hit_rate': 5 / 8}
        validate_datum(datum, now)

    @pytest.mark.asyncio
    async def test_empty_db(self, empty_merge_events_conn):
        """A readable empty window is measured zeros, and a zero-attempt window has no rate."""
        now = datetime.now(UTC)
        datum = await speculative_stats(empty_merge_events_conn, hours=24, now=now)

        assert datum.state is DatumState.FRESH
        assert datum.as_of == now
        assert datum.value == {'hit_count': 0, 'discard_count': 0, 'total': 0, 'hit_rate': None}
        validate_datum(datum, now)

    @pytest.mark.asyncio
    async def test_none_db(self):
        """A project with no runs.db open has no counts, not zero counts."""
        now = datetime.now(UTC)
        datum = await speculative_stats(None, hours=24, now=now)

        assert datum.state is DatumState.UNKNOWN
        assert datum.value is None
        assert datum.as_of is None
        assert 'no runs.db is open' in (datum.reason or '')
        validate_datum(datum, now)

    @pytest.mark.asyncio
    async def test_a_failed_read_is_unknown_not_zeros(self, tableless_runs_conn):
        """with_db swallows the OperationalError; its default must still say the read failed."""
        now = datetime.now(UTC)
        datum = await speculative_stats(tableless_runs_conn, hours=24, now=now)

        assert datum.state is DatumState.UNKNOWN
        assert datum.value is None
        assert 'speculative-merge events could not be read' in (datum.reason or '')
        validate_datum(datum, now)

    @pytest.mark.asyncio
    async def test_all_hits(self, merge_events_db):
        now = datetime.now(UTC)
        datum = await _speculative_over(
            merge_events_db, _speculative_events('speculative_merge', 3, now=now), now=now,
        )

        assert datum.value['hit_rate'] == pytest.approx(1.0)
        assert datum.value['discard_count'] == 0

    @pytest.mark.asyncio
    async def test_all_discards(self, merge_events_db):
        now = datetime.now(UTC)
        datum = await _speculative_over(
            merge_events_db, _speculative_events('speculative_discard', 3, now=now), now=now,
        )

        assert datum.value['hit_rate'] == pytest.approx(0.0)
        assert datum.value['hit_count'] == 0


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
        """Without now, _cutoff_iso derives its cutoff from the current UTC clock.

        Brackets the real clock read with before/after captures (rather than
        patching a module-level ``datetime`` symbol) because the no-now branch
        resolves through ``resolve_now`` in ``dashboard.data.utils``, not a
        clock read local to ``merge_queue.py``.
        """
        before = datetime.now(UTC)
        result = _cutoff_iso(24)
        after = datetime.now(UTC)

        result_dt = datetime.fromisoformat(result)
        lower = before - timedelta(hours=24) - timedelta(seconds=5)
        upper = after - timedelta(hours=24) + timedelta(seconds=5)
        assert lower <= result_dt <= upper


# ---------------------------------------------------------------------------
# TestNowThreadingToCutoffIso (step-3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    'fn_under_test',
    [merge_attempts, speculative_stats],
    ids=lambda fn: fn.__name__,
)
class TestNowThreadingToCutoffIso:
    @pytest.mark.asyncio
    async def test_threads_now_to_cutoff_iso(self, fn_under_test, merge_events_db):
        """fn_under_test passes now to _cutoff_iso, and reads its window ONCE.

        A single captured ``now`` is the proof that ``merge_attempts`` serves
        the outcome chart and the latency block from one window read.
        """
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

        from dashboard.config import DashboardConfig
        from dashboard.data.db import DbPool
        from dashboard.project_dbs import _project_scoped_dbs_labeled

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

        from dashboard.config import DashboardConfig
        from dashboard.data.db import DbPool
        from dashboard.project_dbs import _project_scoped_dbs_labeled

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

        from dashboard.config import DashboardConfig
        from dashboard.data.db import DbPool
        from dashboard.project_dbs import _project_scoped_dbs_labeled

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
# Row titles through the task_lookup datum (task 5595)
# ---------------------------------------------------------------------------

LOOKUP_ROOT = '/proj/A'
LOOKED_UP_AT = datetime(2026, 10, 1, 11, 59, 0, tzinfo=UTC)


def _row_datum(task_id, title, *, state=DatumState.FRESH, reason=None):
    return Datum({'id': task_id, 'title': title, 'status': 'done'},
                 LOOKED_UP_AT, state, reason, 1200)


class TestEnrichMergesWithTitles:
    """Each row's title is a Datum[str] carrying its lookup's provenance."""

    def test_a_found_task_titles_the_row_with_its_lookups_provenance(self):
        found = _row_datum(7, 'Fix X')

        [row] = enrich_merges_with_titles(
            [{'task_id': '7', 'outcome': 'done'}], LOOKUP_ROOT,
            {TaskRef(LOOKUP_ROOT, 7): found},
        )

        assert row['outcome'] == 'done'
        assert row['title'] == Datum('Fix X', found.as_of, found.state, found.reason,
                                     found.freshness_bound_seconds)

    def test_a_stale_lookup_keeps_its_state_and_reason(self):
        stale = _row_datum(7, 'Fix X', state=DatumState.STALE, reason='rows refresh failed')

        [row] = enrich_merges_with_titles(
            [{'task_id': 7}], LOOKUP_ROOT, {TaskRef(LOOKUP_ROOT, 7): stale},
        )

        assert (row['title'].value, row['title'].state, row['title'].reason) == (
            'Fix X', DatumState.STALE, 'rows refresh failed',
        )

    def test_an_unknown_lookup_propagates_its_reason(self):
        unknown = Datum(None, None, DatumState.UNKNOWN, 'lookup budget: past the cap', 1200)

        [row] = enrich_merges_with_titles(
            [{'task_id': '7'}], LOOKUP_ROOT, {TaskRef(LOOKUP_ROOT, 7): unknown},
        )

        assert row['title'].state is DatumState.UNKNOWN
        assert row['title'].value is None
        assert row['title'].reason == 'lookup budget: past the cap'

    @pytest.mark.parametrize('task_id', [None, 'not-an-id'])
    def test_a_row_naming_no_task_has_an_unknown_title(self, task_id):
        [row] = enrich_merges_with_titles([{'task_id': task_id}], LOOKUP_ROOT, {})

        assert row['title'].state is DatumState.UNKNOWN
        assert row['title'].reason
        validate_datum(row['title'], LOOKED_UP_AT)

    def test_a_task_the_lookup_did_not_answer_has_an_unknown_title(self):
        [row] = enrich_merges_with_titles([{'task_id': '9'}], LOOKUP_ROOT, {})

        assert row['title'].state is DatumState.UNKNOWN
        assert row['title'].reason
        validate_datum(row['title'], LOOKED_UP_AT)

    def test_inputs_are_not_mutated(self):
        original = {'task_id': '7', 'outcome': 'done'}
        rows = [original]

        result = enrich_merges_with_titles(
            rows, LOOKUP_ROOT, {TaskRef(LOOKUP_ROOT, 7): _row_datum(7, 'Fix X')},
        )

        assert 'title' not in original
        assert result is not rows


def test_merge_task_refs_names_each_int_parseable_id_once():
    rows = [{'task_id': '7'}, {'task_id': 7}, {'task_id': None},
            {'task_id': 'not-an-id'}, {'task_id': '12'}, {}]

    assert merge_task_refs(LOOKUP_ROOT, rows) == {
        TaskRef(LOOKUP_ROOT, 7), TaskRef(LOOKUP_ROOT, 12),
    }


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
                project_dbs, hours=24, now=now,
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
                project_dbs, hours=24, now=now,
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
        """The recent list covers exactly the ``hours`` window every other leg uses."""
        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        db_path = _make_db(tmp_path, 'a.db', [
            # within the 1h window (5 min ago)
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=5),
             'task_id': 'in-window', 'run_id': 'r1', 'data': {'outcome': 'done'}, 'duration_ms': 1000},
            # outside the 1h window (90 min ago)
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=90),
             'task_id': 'out-window', 'run_id': 'r2', 'data': {'outcome': 'done'}, 'duration_ms': 1000},
        ])

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            project_dbs = [('/tmp/A', conn)]
            result = await build_per_project_merge_queue(project_dbs, hours=1, now=now)

        recent = result['/tmp/A']['recent']
        task_ids = {r['task_id'] for r in recent}
        assert 'in-window' in task_ids
        assert 'out-window' not in task_ids
        assert result['/tmp/A']['recent_total'].value == 1

    async def test_none_db_entry_skipped(self, tmp_path):
        """A (pid, None) pair yields the declared full-default shape (all 5 keys,
        each matching its _DEFAULT_* constant) for that project."""
        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        project_dbs = [('/tmp/nofile', None)]
        result = await build_per_project_merge_queue(
            project_dbs, hours=24, now=now,
        )

        assert '/tmp/nofile' in result
        data = result['/tmp/nofile']
        assert set(data.keys()) >= {'depth_timeseries', 'outcomes', 'latency', 'recent', 'speculative'}
        assert data['depth_timeseries'] == {'labels': [], 'values': []}
        assert data['outcomes'] == {'labels': [], 'values': []}
        assert data['latency'] == ZERO_LATENCY
        assert data['recent'] == []
        for field in ('speculative', 'recent_total'):
            assert data[field].state is DatumState.UNKNOWN, field
            validate_datum(data[field], now)
        assert 'active' not in data, (
            'the live probe is the one source of the queue; the route reads it'
        )

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
                project_dbs, hours=24, now=now,
            )

        # Both pids present — None entry must not be dropped
        assert set(result.keys()) == {'/tmp/real', '/tmp/none'}

        # None-db entry produces defaults without crashing
        none_data = result['/tmp/none']
        assert none_data['latency']['with_duration'] == 0
        assert none_data['recent'] == []
        for field in ('speculative', 'recent_total'):
            assert none_data[field].state is DatumState.UNKNOWN, field

        # Real-but-empty entry: its zeros were measured
        real_data = result['/tmp/real']
        assert set(real_data.keys()) >= {'depth_timeseries', 'outcomes', 'latency', 'recent', 'speculative'}
        assert real_data['speculative'].state is DatumState.FRESH
        assert real_data['speculative'].value['total'] == 0
        assert real_data['recent_total'].state is DatumState.FRESH
        assert real_data['recent_total'].value == 0
        for data in result.values():
            for field in ('speculative', 'recent_total'):
                validate_datum(data[field], now)

    async def test_the_outcome_total_is_the_latency_attempt_total(self, tmp_path):
        """Sketch #9, second half: one row set behind the donut and the latency."""
        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        db_path = _make_db(tmp_path, 'split.db', [
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=5),
             'task_id': 'timed', 'run_id': 'r1', 'data': {'outcome': 'done'}, 'duration_ms': 900},
            {'event_type': 'merge_attempt', 'timestamp': now - timedelta(minutes=6),
             'task_id': 'untimed', 'run_id': 'r2', 'data': {'outcome': 'conflict'}, 'duration_ms': None},
        ])

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await build_per_project_merge_queue([('/tmp/A', conn)], hours=24, now=now)

        project = result['/tmp/A']
        latency = project['latency']
        assert (latency['with_duration'], latency['without_duration']) == (1, 1)
        assert sum(project['outcomes']['values']) == (
            latency['with_duration'] + latency['without_duration']
        )

    async def test_per_project_queries_run_concurrently(self, tmp_path):
        """All N per-project gathers must run concurrently (peak-in-flight == N).

        Patches ``merge_attempts`` with a fake that:
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

        async def fake_merge_attempts(db, *, hours=24, now=None):
            counter[0] += 1
            if counter[0] > max_in_flight[0]:
                max_in_flight[0] = counter[0]
            if counter[0] == N:
                all_entered.set()
            await release.wait()
            counter[0] -= 1
            return await merge_attempts(None, hours=hours, now=now)

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
                'dashboard.data.merge_queue.merge_attempts',
                new=fake_merge_attempts,
            ):
                task = asyncio.create_task(
                    build_per_project_merge_queue(
                        project_dbs, hours=24, now=now,
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
    @pytest.mark.parametrize('hours, expected_ids, expected_total', [
        (168, ['day0-0', 'day0-1', 'day1-0', 'day1-1', 'day1-2', 'day2-0', 'day2-1', 'day2-2', 'day2-3'], 9),
        (24, ['day0-0', 'day0-1'], 2),
    ])
    async def test_recent_follows_the_selected_window(
        self, tmp_path, hours, expected_ids, expected_total,
    ):
        """Sketch #9: widening the window widens both the rows and the total."""
        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)
        per_day = {0: 2, 1: 3, 2: 4}
        events = [
            {
                'event_type': 'merge_attempt',
                'timestamp': now - timedelta(days=day, hours=1, minutes=i),
                'task_id': f'day{day}-{i}',
                'run_id': f'run-{day}-{i}',
                'data': {'outcome': 'done'},
                'duration_ms': 100,
            }
            for day, count in per_day.items()
            for i in range(count)
        ]
        db_path = _make_db(tmp_path, 'spread.db', events)

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await build_per_project_merge_queue([('/tmp/P', conn)], hours=hours, now=now)

        project = result['/tmp/P']
        assert [r['task_id'] for r in project['recent']] == expected_ids
        assert project['recent_total'].value == expected_total

    @pytest.mark.asyncio
    async def test_burst_beyond_the_cap_keeps_the_newest_and_counts_them_all(self, tmp_path):
        """A window holding more than RECENT_MERGES_CAP merges shows the newest cap, totals all."""
        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)
        burst = RECENT_MERGES_CAP + 5
        events = [
            {
                'event_type': 'merge_attempt',
                'timestamp': now - timedelta(seconds=i * 10),
                'task_id': f'burst-task-{i:03d}',
                'run_id': f'burst-run-{i:03d}',
                'data': {'outcome': 'done'},
                'duration_ms': 500 + i,
            }
            for i in range(burst)
        ]
        db_path = _make_db(tmp_path, 'burst.db', events)

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await build_per_project_merge_queue([('/tmp/P', conn)], hours=24, now=now)

        project = result['/tmp/P']
        assert [r['task_id'] for r in project['recent']] == [
            f'burst-task-{i:03d}' for i in range(RECENT_MERGES_CAP)
        ]
        assert project['recent_total'].value == burst

    @pytest.mark.asyncio
    @pytest.mark.parametrize('leg, field, sibling', [
        ('speculative_stats', 'speculative', 'recent_total'),
        ('recent_merges', 'recent_total', 'speculative'),
    ])
    async def test_a_raising_leg_serves_unknown_not_zeros(self, tmp_path, leg, field, sibling):
        """A leg that raises is a hole in its own field; its siblings stay measured."""
        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)
        db_path = _make_db(tmp_path, 'raising.db', [])

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            with patch(
                f'dashboard.data.merge_queue.{leg}',
                new=AsyncMock(side_effect=RuntimeError('boom')),
            ):
                result = await build_per_project_merge_queue([('/tmp/P', conn)], hours=24, now=now)

        project = result['/tmp/P']
        assert project[field].state is DatumState.UNKNOWN
        assert '/tmp/P' in (project[field].reason or '')
        assert project[sibling].state is DatumState.FRESH
        for served in (field, sibling):
            validate_datum(project[served], now)

    @pytest.mark.asyncio
    async def test_the_whole_project_fallback_serves_unknown(self, tmp_path):
        """The outer ``except Exception`` arm serves holes naming the error, not zeros."""
        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)
        db_path = _make_db(tmp_path, 'fallback.db', [])

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            with patch(
                'dashboard.data.merge_queue.safe_gather_result',
                side_effect=RuntimeError('boom'),
            ):
                result = await build_per_project_merge_queue([('/tmp/P', conn)], hours=24, now=now)

        project = result['/tmp/P']
        assert project['recent'] == []
        for field in ('speculative', 'recent_total'):
            assert project[field].state is DatumState.UNKNOWN, field
            assert 'boom' in (project[field].reason or ''), field
            validate_datum(project[field], now)

    @pytest.mark.asyncio
    async def test_cancelled_error_from_sub_query_propagates(self, tmp_path):
        """CancelledError from any sub-query must not be swallowed.

        It propagates through _one_project's ``except Exception`` guard
        (which only catches Exception, not BaseException) to the outer asyncio.gather call.
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
            ), pytest.raises(asyncio.CancelledError):
                await build_per_project_merge_queue(
                    [('/tmp/P', conn)],
                    hours=24,
                    now=now,
                )


# ---------------------------------------------------------------------------
# TestRecentTrainEvents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRecentTrainEvents:
    """Tests for recent_train_events() in dashboard.data.merge_queue."""

    async def test_returns_only_train_events_newest_first(self, tmp_path) -> None:
        """Returns only train_* rows, newest first, excludes unrelated events."""
        from dashboard.data.merge_queue import recent_train_events

        now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC)
        train_types = ['train_started', 'train_member_deferred', 'train_merged', 'train_derailed']
        events = []
        for i, etype in enumerate(train_types):
            events.append(dict(
                event_type=etype,
                timestamp=now - timedelta(hours=1, minutes=10 - i),
                task_id=f'task-{i}',
                run_id=f'run-{i}',
                data={'train_id': f'train-{i}', 'extra': i},
            ))
        # Unrelated event — must be excluded
        events.append(dict(
            event_type='merge_attempt',
            timestamp=now - timedelta(hours=1),
            task_id='task-unrelated',
            run_id='run-unrelated',
            data={'outcome': 'done'},
        ))

        db_path = _make_db(tmp_path, 'train_events.db', events)

        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row
            result = await recent_train_events(db, now=now)

        assert len(result) == 4, f'Expected 4 train_* rows, got {len(result)}: {result}'
        actual_types = [r['event_type'] for r in result]
        # Newest first: train_derailed (i=3, -1h+7min) > ... > train_started (i=0, -1h+10min)
        assert actual_types[0] == 'train_derailed', f'Newest first: got {actual_types}'
        assert set(actual_types) == set(train_types)
        # Check required keys
        for r in result:
            assert 'task_id' in r
            assert 'run_id' in r
            assert 'event_type' in r
            assert 'timestamp' in r
            assert isinstance(r['data'], dict), f'data must be a dict, got {type(r["data"])}'
            assert 'train_id' in r['data']

    async def test_none_db_returns_empty_list(self) -> None:
        """Passing db=None returns [] without raising."""
        from dashboard.data.merge_queue import recent_train_events

        result = await recent_train_events(None)
        assert result == []

    async def test_hours_window_excludes_old_events(self, tmp_path) -> None:
        """Events older than hours window are excluded."""
        from dashboard.data.merge_queue import recent_train_events

        now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC)
        events = [
            dict(
                event_type='train_started',
                timestamp=now - timedelta(hours=2),  # within default 168h
                task_id='task-recent',
                run_id='run-recent',
                data={'train_id': 'train-recent'},
            ),
            dict(
                event_type='train_merged',
                timestamp=now - timedelta(hours=200),  # OUTSIDE 168h default
                task_id='task-old',
                run_id='run-old',
                data={'train_id': 'train-old'},
            ),
        ]
        db_path = _make_db(tmp_path, 'train_window.db', events)

        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row
            result = await recent_train_events(db, hours=168, now=now)

        assert len(result) == 1, f'Expected 1 recent event, got {len(result)}'
        assert result[0]['task_id'] == 'task-recent'

    async def test_limit_arg(self, tmp_path) -> None:
        """limit argument restricts number of results returned."""
        from dashboard.data.merge_queue import recent_train_events

        now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC)
        events = [
            dict(
                event_type='train_started',
                timestamp=now - timedelta(hours=1, minutes=i + 1),
                task_id=f'task-{i}',
                run_id=f'run-{i}',
                data={'train_id': f'train-{i}'},
            )
            for i in range(10)
        ]
        db_path = _make_db(tmp_path, 'train_limit.db', events)

        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row
            result = await recent_train_events(db, limit=3, now=now)

        assert len(result) == 3, f'Expected 3 results (limit=3), got {len(result)}'


# ---------------------------------------------------------------------------
# Acceptance lock: recent merges follow the selected window
# ---------------------------------------------------------------------------


class TestRecentFollowsTheWindow:
    """Acceptance-criterion lock (task-1607, retargeted by task 5593).

    At the 24h chip window, a merge ~5h old appears in result[pid]['recent']
    and a merge ~25h old does not: the recent list is bounded by the same
    ``hours`` window every other leg of the merge-queue payload uses — "a
    merge from earlier today appears; one from yesterday does not".
    """

    @pytest.mark.asyncio
    async def test_24h_window_includes_hours_old_excludes_yesterday(self, tmp_path):
        """A merge 5h ago appears; one 25h ago does not — at hours=24."""
        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)
        pid = '/tmp/test-proj'

        db_path = _make_db(tmp_path, 'acceptance.db', [
            # Inside the 24h window: 5h ago
            {
                'event_type': 'merge_attempt',
                'timestamp': now - timedelta(hours=5),
                'task_id': 'task-recent',
                'run_id': 'run-recent',
                'data': {'outcome': 'done'},
                'duration_ms': 1000,
            },
            # Outside the 24h window: 25h ago
            {
                'event_type': 'merge_attempt',
                'timestamp': now - timedelta(hours=25),
                'task_id': 'task-old',
                'run_id': 'run-old',
                'data': {'outcome': 'done'},
                'duration_ms': 2000,
            },
        ])

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await build_per_project_merge_queue(
                [(pid, conn)],
                hours=24,
                now=now,
            )

        recent_task_ids = [row['task_id'] for row in result[pid]['recent']]
        assert 'task-recent' in recent_task_ids, (
            f'Expected task-recent (5h old) in recent list at hours=24; got {recent_task_ids}'
        )
        assert 'task-old' not in recent_task_ids, (
            f'Expected task-old (25h old) absent from recent list at hours=24; got {recent_task_ids}'
        )


# ---------------------------------------------------------------------------
# TestNormalizeLiveEntry (task-1606 step-1)
# ---------------------------------------------------------------------------


class TestNormalizeLiveEntry:
    """Tests for merge_queue._normalize_entry — snapshot entry → display shape."""

    def test_full_entry_preserved(self):
        """A fully-populated snapshot entry maps 1-to-1 onto the display shape."""
        from dashboard.data.merge_queue import _normalize_entry

        raw = {
            'task_id': '3112',
            'branch': 'task/3112',
            'state': 'queued',
            'age_secs': 14400.0,
            'position': 1,
            'waiter_alive': True,
            'enqueued_at': '2026-06-04T08:00:00+00:00',
            'worktree': '/tmp/wt',
            'pre_rebased': False,
        }
        result = _normalize_entry(raw)
        assert result['task_id'] == '3112'
        assert result['branch'] == 'task/3112'
        assert result['state'] == 'queued'
        assert result['age_secs'] == 14400.0      # AC1: 4h entry must NOT be dropped
        assert result['position'] == 1
        assert result['waiter_alive'] is True

    def test_ac1_long_queued_entry_preserved(self):
        """AC1 lock: an entry queued 4h ago (age_secs=14400) survives normalization.

        The event-derived fallback's 30-min TTL would drop this entry; the live
        path must not apply any TTL, so a >30min entry must pass through intact.
        """
        from dashboard.data.merge_queue import _normalize_entry

        raw = {'task_id': '3112', 'branch': 'task/3112', 'state': 'queued', 'age_secs': 14400.0,
               'position': 1, 'waiter_alive': True}
        result = _normalize_entry(raw)
        assert result['age_secs'] == 14400.0
        assert result['state'] == 'queued'

    def test_missing_optional_fields_get_safe_defaults(self):
        """Missing waiter_alive and position get safe defaults (True and 0)."""
        from dashboard.data.merge_queue import _normalize_entry

        raw = {'task_id': '42', 'branch': 'task/42', 'state': 'merging', 'age_secs': 5.0}
        result = _normalize_entry(raw)
        assert result['waiter_alive'] is True    # safe default: assume waiter alive
        assert result['position'] == 0           # safe default: unknown position → 0
        assert result['age_secs'] == 5.0

    def test_none_optional_fields_get_safe_defaults(self):
        """None waiter_alive and None position get the same safe defaults."""
        from dashboard.data.merge_queue import _normalize_entry

        raw = {'task_id': '7', 'branch': 'task/7', 'state': 'verifying',
               'age_secs': 120.0, 'position': None, 'waiter_alive': None}
        result = _normalize_entry(raw)
        assert result['waiter_alive'] is True
        assert result['position'] == 0

    def test_only_display_keys_returned(self):
        """Result contains exactly the six display keys (no internal snapshot fields)."""
        from dashboard.data.merge_queue import _normalize_entry

        raw = {'task_id': '1', 'branch': 'b', 'state': 'queued', 'age_secs': 1.0,
               'position': 0, 'waiter_alive': True,
               'worktree': '/tmp/wt', 'pre_rebased': False, 'enqueued_at': 'ts'}
        result = _normalize_entry(raw)
        assert set(result.keys()) == {'task_id', 'branch', 'state', 'age_secs', 'position', 'waiter_alive'}

    def test_missing_age_secs_defaults_zero(self):
        """Missing age_secs defaults to 0.0 (defensive)."""
        from dashboard.data.merge_queue import _normalize_entry

        raw = {'task_id': '5', 'branch': 'b', 'state': 'queued'}
        result = _normalize_entry(raw)
        assert result['age_secs'] == 0.0


# ---------------------------------------------------------------------------
# TestFetchLiveMergeQueues — success path (task-1606 step-3)
# ---------------------------------------------------------------------------

# MCP mock envelopes come from _dashboard_helpers (task 3952) — see the
# imports at the top of this module.


class _PerPortHandler:
    """Mock httpx handler dispatching per-port behaviour for MCP calls.

    ``snapshot_responses`` maps port → snapshot dict (the inner result of get_merge_queue).
    ``fail_ports`` raise httpx.ConnectError.
    ``slow_ports`` sleep before responding (drives the timeout path).
    """

    def __init__(
        self,
        snapshot_responses: dict[int, dict] | None = None,
        *,
        fail_ports: set[int] | None = None,
        slow_ports: dict[int, float] | None = None,
    ):
        self.snapshot_responses = snapshot_responses or {}
        self.fail_ports = fail_ports or set()
        self.slow_ports = slow_ports or {}

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        port = request.url.port
        assert port is not None
        if port in self.fail_ports:
            raise httpx.ConnectError('refused')
        if port in self.slow_ports:
            await asyncio.sleep(self.slow_ports[port])
        body = json.loads(request.content)
        method = body.get('method', '')
        request_id = body.get('id', 1)
        if method == 'initialize':
            return mcp_init_response(request_id)
        if method.startswith('notifications/'):
            return mcp_notify_response()
        # tools/call → return snapshot
        inner = self.snapshot_responses.get(port, {'entries': [], 'depth': 0})
        return mcp_tool_response(inner, request_id)


@pytest.fixture(autouse=False)
def _clean_live_sessions():
    """Reset mcp_tool_call session cache before/after live-queue tests."""
    from dashboard.data.memory import reset_sessions
    reset_sessions()
    yield
    reset_sessions()


def _live_urls(*ports: int) -> dict[str, str]:
    return {f'proj{p}': f'http://127.0.0.1:{p}/mcp' for p in ports}


def _snapshot(entries: list) -> dict:
    """Build a minimal get_merge_queue snapshot dict."""
    return {
        'entries': entries,
        'depth': len(entries),
        'head_of_line': entries[0]['task_id'] if entries else None,
        'verify_in_progress': False,
        'is_wip_halted': False,
        'halt_owner_esc_id': None,
    }


class TestFetchLiveMergeQueues:
    """Tests for fetch_live_merge_queues — success paths."""

    @pytest.mark.asyncio
    async def test_empty_urls_returns_empty(self, _clean_live_sessions):
        from dashboard.data.merge_queue import fetch_live_merge_queues
        async with httpx.AsyncClient() as client:
            result = await fetch_live_merge_queues(client, {})
        assert result == {}

    @pytest.mark.asyncio
    async def test_single_project_reachable(self, _clean_live_sessions):
        """One project reachable → {label: {reachable: True, entries: [...]}}."""
        from dashboard.data.merge_queue import fetch_live_merge_queues

        entries = [
            {'task_id': '99', 'branch': 'task/99', 'state': 'queued', 'age_secs': 30.0,
             'position': 1, 'waiter_alive': True},
        ]
        handler = _PerPortHandler({8200: _snapshot(entries)})
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_live_merge_queues(client, _live_urls(8200))

        assert 'proj8200' in result
        proj = result['proj8200']
        assert proj['reachable'] is True
        assert len(proj['entries']) == 1
        assert proj['entries'][0]['task_id'] == '99'
        assert 'error' not in proj

    @pytest.mark.asyncio
    async def test_ac2_two_same_task_entries_both_preserved(self, _clean_live_sessions):
        """AC2 lock: two entries with the same task_id must both appear in the result.

        The event-derived fallback's ROW_NUMBER PARTITION BY task_id keeps only
        one event per task; the live path must return ALL queue entries as-is.
        """
        from dashboard.data.merge_queue import fetch_live_merge_queues

        # Two queue entries for the same task_id (e.g. a task queued twice)
        entries = [
            {'task_id': '3112', 'branch': 'task/3112', 'state': 'queued',
             'age_secs': 14400.0, 'position': 1, 'waiter_alive': True},
            {'task_id': '3112', 'branch': 'task/3112-retry', 'state': 'queued',
             'age_secs': 300.0, 'position': 2, 'waiter_alive': True},
        ]
        handler = _PerPortHandler({8201: _snapshot(entries)})
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_live_merge_queues(client, _live_urls(8201))

        proj = result['proj8201']
        assert proj['reachable'] is True
        assert len(proj['entries']) == 2, (
            'AC2: both entries for task_id=3112 must be present; '
            f"got {len(proj['entries'])}"
        )
        assert proj['entries'][0]['age_secs'] == 14400.0
        assert proj['entries'][1]['age_secs'] == 300.0

    @pytest.mark.asyncio
    async def test_authoritative_empty_queue_reachable_true(self, _clean_live_sessions):
        """An authoritative empty queue (entries=[], no error) → reachable=True, entries=[].

        Hard no-synthetic-data rule: if the orchestrator says 'nothing queued'
        we must show empty, not fall back to the event-derived approximation.
        """
        from dashboard.data.merge_queue import fetch_live_merge_queues

        handler = _PerPortHandler({8202: _snapshot([])})
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_live_merge_queues(client, _live_urls(8202))

        proj = result['proj8202']
        assert proj['reachable'] is True
        assert proj['entries'] == []

    @pytest.mark.asyncio
    async def test_multi_project_all_keys_present(self, _clean_live_sessions):
        """All configured project labels appear in the result even with diverse data."""
        from dashboard.data.merge_queue import fetch_live_merge_queues

        entries_8210 = [
            {'task_id': '10', 'branch': 'task/10', 'state': 'merging',
             'age_secs': 5.0, 'position': 0, 'waiter_alive': True},
        ]
        entries_8211 = []  # authoritative empty
        handler = _PerPortHandler({8210: _snapshot(entries_8210), 8211: _snapshot(entries_8211)})
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_live_merge_queues(client, _live_urls(8210, 8211))

        assert set(result) == {'proj8210', 'proj8211'}
        assert result['proj8210']['reachable'] is True
        assert len(result['proj8210']['entries']) == 1
        assert result['proj8211']['reachable'] is True
        assert result['proj8211']['entries'] == []


# ---------------------------------------------------------------------------
# TestFetchLiveMergeQueuesFailure — degraded paths (task-1606 step-5)
# ---------------------------------------------------------------------------


class TestFetchLiveMergeQueuesFailure:
    """Failure / degraded-path tests for fetch_live_merge_queues."""

    @pytest.mark.asyncio
    async def test_connect_error_yields_unreachable(self, _clean_live_sessions):
        """(a) Connect error → {reachable: False, entries: [], error present}."""
        from dashboard.data.merge_queue import fetch_live_merge_queues

        handler = _PerPortHandler(
            {8300: _snapshot([])},
            fail_ports={8301},
        )
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_live_merge_queues(client, _live_urls(8300, 8301))

        assert result['proj8300']['reachable'] is True   # control: this one is up
        assert result['proj8301']['reachable'] is False
        assert result['proj8301']['entries'] == []
        assert 'error' in result['proj8301']

    @pytest.mark.asyncio
    async def test_timeout_yields_unreachable(self, _clean_live_sessions):
        """(b) Slow port with small per_call_timeout → {reachable: False}."""
        from dashboard.data.merge_queue import fetch_live_merge_queues

        handler = _PerPortHandler(
            {8300: _snapshot([])},
            slow_ports={8302: 0.5},
        )
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_live_merge_queues(
                client, _live_urls(8300, 8302), per_call_timeout=0.05,
            )

        assert result['proj8300']['reachable'] is True
        assert result['proj8302']['reachable'] is False
        assert 'error' in result['proj8302']

    @pytest.mark.asyncio
    async def test_error_result_yields_unreachable(self, _clean_live_sessions):
        """(c) get_merge_queue {'error': ...} result → {reachable: False, entries: []}."""
        from dashboard.data.merge_queue import fetch_live_merge_queues

        error_result = {'error': 'Merge queue not available — orchestrator not running'}
        handler = _PerPortHandler({8303: error_result})
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_live_merge_queues(client, _live_urls(8303))

        proj = result['proj8303']
        assert proj['reachable'] is False
        assert proj['entries'] == []    # AC3: no fabricated rows
        assert 'error' in proj

    @pytest.mark.asyncio
    async def test_all_known_labels_present_on_mixed_failures(self, _clean_live_sessions):
        """All project labels appear in the result regardless of per-call failure."""
        from dashboard.data.merge_queue import fetch_live_merge_queues

        handler = _PerPortHandler(
            {8310: _snapshot([])},
            fail_ports={8311, 8312},
        )
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_live_merge_queues(client, _live_urls(8310, 8311, 8312))

        assert set(result) == {'proj8310', 'proj8311', 'proj8312'}

    @pytest.mark.asyncio
    async def test_no_fabricated_rows_on_unreachable(self, _clean_live_sessions):
        """AC3: unreachable orchestrator → entries=[], NOT the fallback approximation rows."""
        from dashboard.data.merge_queue import fetch_live_merge_queues

        handler = _PerPortHandler({}, fail_ports={8313})
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_live_merge_queues(client, _live_urls(8313))

        assert result['proj8313']['entries'] == []
        assert result['proj8313']['reachable'] is False


# ---------------------------------------------------------------------------
# TestResolveActive — "In queue now" as one Datum[int] (task 5595)
# ---------------------------------------------------------------------------

QUEUE_NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
_QUEUED = {'task_id': '99', 'branch': 'task/99', 'state': 'queued',
           'age_secs': 120.0, 'position': 1, 'waiter_alive': True}
_SAMPLED_AT = QUEUE_NOW - timedelta(minutes=10)
_HISTORY = {
    'labels': [(QUEUE_NOW - timedelta(minutes=20)).isoformat(), _SAMPLED_AT.isoformat()],
    'values': [3, 1],
}
_REFUSED = {'myproj': {'reachable': False, 'entries': [], 'error': 'connect refused'}}


class TestResolveActive:
    """resolve_active(label, live_map, history, *, now) -> ActiveQueue."""

    def test_a_reachable_probe_is_a_fresh_count_at_the_probe_instant(self):
        entries = [_QUEUED, {**_QUEUED, 'task_id': '100', 'position': 2}]
        live_map = {'myproj': {'reachable': True, 'entries': entries}}

        queue = resolve_active('myproj', live_map, _HISTORY, now=QUEUE_NOW)

        assert queue.in_queue.state is DatumState.FRESH
        assert (queue.in_queue.value, queue.in_queue.as_of) == (2, QUEUE_NOW)
        assert queue.entries == entries
        assert queue.probe_configured is True
        validate_datum(queue.in_queue, QUEUE_NOW)

    def test_a_reachable_empty_queue_is_an_authoritative_zero(self):
        live_map = {'myproj': {'reachable': True, 'entries': []}}

        queue = resolve_active('myproj', live_map, _HISTORY, now=QUEUE_NOW)

        assert queue.in_queue.state is DatumState.FRESH
        assert queue.in_queue.value == 0
        assert queue.entries == []
        validate_datum(queue.in_queue, QUEUE_NOW)

    def test_a_failed_probe_serves_the_last_sample_stale(self):
        queue = resolve_active('myproj', _REFUSED, _HISTORY, now=QUEUE_NOW)

        assert queue.in_queue.state is DatumState.STALE
        assert (queue.in_queue.value, queue.in_queue.as_of) == (1, _SAMPLED_AT)
        assert queue.in_queue.reason is not None
        assert 'connect refused' in queue.in_queue.reason
        assert queue.entries == []
        assert queue.probe_configured is True, 'a failed probe is still a configured one'
        validate_datum(queue.in_queue, QUEUE_NOW)

    def test_a_label_with_no_configured_probe_is_unknown_and_says_so(self):
        """No probe means no queue this dashboard reads, so no history to fall back on.

        The sampler records only the live probe, so any sample under such a
        project predates it. ``probe_configured`` is false, which keeps the
        project out of every multi-project total instead of a permanent hole.
        """
        queue = resolve_active('myproj', {}, _HISTORY, now=QUEUE_NOW)

        assert queue.in_queue.state is DatumState.UNKNOWN
        assert queue.in_queue.reason is not None
        assert 'myproj' in queue.in_queue.reason
        assert 'configured' in queue.in_queue.reason
        assert queue.entries == []
        assert queue.probe_configured is False
        validate_datum(queue.in_queue, QUEUE_NOW)

    def test_a_failed_probe_with_no_sample_is_unknown(self):
        queue = resolve_active('myproj', _REFUSED, {'labels': [], 'values': []},
                               now=QUEUE_NOW)

        assert queue.in_queue.state is DatumState.UNKNOWN
        assert queue.in_queue.value is None
        assert queue.in_queue.reason is not None
        assert 'connect refused' in queue.in_queue.reason
        assert 'no sample' in queue.in_queue.reason
        assert queue.entries == []
        validate_datum(queue.in_queue, QUEUE_NOW)

    def test_an_unparseable_last_sample_is_unknown(self):
        history = {'labels': ['not-a-timestamp'], 'values': [4]}

        queue = resolve_active('myproj', _REFUSED, history, now=QUEUE_NOW)

        assert queue.in_queue.state is DatumState.UNKNOWN
        assert queue.entries == []
        validate_datum(queue.in_queue, QUEUE_NOW)


# ---------------------------------------------------------------------------
# TestTrainThroughputStats — step-6 RED / step-7+ GREEN
# ---------------------------------------------------------------------------

_TRAIN_THROUGHPUT_DEFAULT_KEYS = {
    'trains_landed',
    'tasks_landed_via_trains',
    'train_verifies_per_landed_task',
    'baseline_solo_landed',
    'baseline_verifies_per_landed_task',
    'verifies_per_landed_task_delta',
    'train_cas_retry_rate',
    'baseline_cas_retry_rate',
    'cas_retry_rate_delta',
    'improved',
}


@pytest.mark.asyncio
class TestTrainThroughputStats:
    """Tests for train_throughput_stats(db, *, hours=24, now=None) -> dict.

    step-6: default contract (None / empty DB → all-zeros).
    step-8: verifies-per-landed-task counting.
    step-10: CAS-retry rate.
    step-12: aggregator wiring.
    """

    async def test_none_db_returns_all_zeros_default(self):
        """train_throughput_stats(None) returns the all-zeros default dict."""
        from dashboard.data.merge_queue import train_throughput_stats

        result = await train_throughput_stats(None)

        assert set(result.keys()) == _TRAIN_THROUGHPUT_DEFAULT_KEYS, (
            f"expected keys {_TRAIN_THROUGHPUT_DEFAULT_KEYS}, got: {set(result.keys())}"
        )
        assert result['trains_landed'] == 0
        assert result['tasks_landed_via_trains'] == 0
        assert result['train_verifies_per_landed_task'] == 0.0
        assert result['baseline_solo_landed'] == 0
        assert result['baseline_verifies_per_landed_task'] == 0.0
        assert result['verifies_per_landed_task_delta'] == 0.0
        assert result['train_cas_retry_rate'] == 0.0
        assert result['baseline_cas_retry_rate'] == 0.0
        assert result['cas_retry_rate_delta'] == 0.0
        assert result['improved'] is False

    async def test_empty_db_returns_all_zeros_default(self, tmp_path):
        """train_throughput_stats(db) on an empty events table returns the all-zeros default."""
        from dashboard.data.merge_queue import train_throughput_stats

        db_path = _make_db(tmp_path, 'empty.db', [])
        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await train_throughput_stats(conn)

        assert set(result.keys()) == _TRAIN_THROUGHPUT_DEFAULT_KEYS
        assert result['trains_landed'] == 0
        assert result['tasks_landed_via_trains'] == 0
        assert result['train_verifies_per_landed_task'] == 0.0
        assert result['baseline_solo_landed'] == 0
        assert result['baseline_verifies_per_landed_task'] == 0.0
        assert result['verifies_per_landed_task_delta'] == 0.0
        assert result['train_cas_retry_rate'] == 0.0
        assert result['baseline_cas_retry_rate'] == 0.0
        assert result['cas_retry_rate_delta'] == 0.0
        assert result['improved'] is False

    async def test_verifies_per_landed_task_counting(self, tmp_path):
        """train_merged + solo done rows → verifies-per-landed-task counting identity.

        One train_merged(members=['10','11']) in window + two solo merge_attempt
        done rows + one train_merged OUTSIDE the window (must be excluded).

        Asserts exact counting identity: trains_landed=1, tasks_landed_via_trains=2,
        train_verifies_per_landed_task=0.5, baseline_solo_landed=2,
        baseline_verifies_per_landed_task=1.0, verifies_per_landed_task_delta=0.5,
        improved=True.
        """
        from datetime import UTC, datetime, timedelta

        from dashboard.data.merge_queue import train_throughput_stats

        now = datetime(2026, 1, 10, 12, 0, 0, tzinfo=UTC)
        in_window = now - timedelta(hours=1)
        out_of_window = now - timedelta(hours=48)

        events = [
            # in-window: one train_merged with 2 members
            dict(
                event_type='train_merged',
                timestamp=in_window,
                run_id='run-train-1',
                task_id='11',
                data={'train_id': 't1', 'member_task_ids': ['10', '11'],
                      'merge_commit_sha': 'aaa', 'base_sha': 'bbb'},
            ),
            # in-window: two solo merge_attempt(done, no train_id)
            dict(
                event_type='merge_attempt',
                timestamp=in_window,
                run_id='run-solo-20',
                task_id='20',
                data={'outcome': 'done'},
            ),
            dict(
                event_type='merge_attempt',
                timestamp=in_window,
                run_id='run-solo-21',
                task_id='21',
                data={'outcome': 'done'},
            ),
            # OUT-of-window: must be excluded
            dict(
                event_type='train_merged',
                timestamp=out_of_window,
                run_id='run-train-old',
                task_id='99',
                data={'train_id': 't0', 'member_task_ids': ['90', '91'],
                      'merge_commit_sha': 'zzz', 'base_sha': 'yyy'},
            ),
        ]
        db_path = _make_db(tmp_path, 'verifies.db', events)
        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await train_throughput_stats(conn, hours=24, now=now)

        assert result['trains_landed'] == 1
        assert result['tasks_landed_via_trains'] == 2
        assert result['train_verifies_per_landed_task'] == 0.5, (
            f"expected 1 union verify / 2 members = 0.5, got: {result['train_verifies_per_landed_task']}"
        )
        assert result['baseline_solo_landed'] == 2
        assert result['baseline_verifies_per_landed_task'] == 1.0
        assert result['verifies_per_landed_task_delta'] == 0.5, (
            f"expected baseline(1.0) - train(0.5) = 0.5, got: {result['verifies_per_landed_task_delta']}"
        )
        assert result['improved'] is True

    async def test_cas_retry_rates(self, tmp_path):
        """CAS-retry rates: train vs baseline, delta, train excluded from baseline.

        One train_merged(members=['10','11']), one train cas_retry row,
        two solo done rows, two solo cas_retry rows.

        Asserts:
        - train_cas_retry_rate == 0.5   (1 train retry / 2 tasks_landed_via_trains)
        - baseline_cas_retry_rate == 1.0 (2 solo retries / 2 baseline_solo_landed)
        - cas_retry_rate_delta == 0.5    (baseline - train)
        - train retry rows excluded from baseline retry count
        """
        from datetime import UTC, datetime, timedelta

        from dashboard.data.merge_queue import train_throughput_stats

        now = datetime(2026, 1, 10, 12, 0, 0, tzinfo=UTC)
        in_window = now - timedelta(hours=1)

        events = [
            # train landed with 2 members
            dict(
                event_type='train_merged',
                timestamp=in_window,
                run_id='run-train-1',
                task_id='11',
                data={'train_id': 't1', 'member_task_ids': ['10', '11'],
                      'merge_commit_sha': 'aaa', 'base_sha': 'bbb'},
            ),
            # one train cas_retry row (has train_id)
            dict(
                event_type='merge_attempt',
                timestamp=in_window,
                run_id='run-train-retry',
                task_id='10',
                data={'outcome': 'cas_retry', 'train_id': 't1',
                      'member_task_ids': ['10', '11']},
            ),
            # two solo done rows (no train_id)
            dict(
                event_type='merge_attempt',
                timestamp=in_window,
                run_id='run-solo-20',
                task_id='20',
                data={'outcome': 'done'},
            ),
            dict(
                event_type='merge_attempt',
                timestamp=in_window,
                run_id='run-solo-21',
                task_id='21',
                data={'outcome': 'done'},
            ),
            # two solo cas_retry rows (no train_id) — must NOT bleed into train rate
            dict(
                event_type='merge_attempt',
                timestamp=in_window,
                run_id='run-solo-retry-20',
                task_id='20',
                data={'outcome': 'cas_retry'},
            ),
            dict(
                event_type='merge_attempt',
                timestamp=in_window,
                run_id='run-solo-retry-21',
                task_id='21',
                data={'outcome': 'cas_retry'},
            ),
        ]
        db_path = _make_db(tmp_path, 'cas_retry.db', events)
        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await train_throughput_stats(conn, hours=24, now=now)

        assert result['tasks_landed_via_trains'] == 2
        assert result['baseline_solo_landed'] == 2
        assert result['train_cas_retry_rate'] == 0.5, (
            f"expected 1 train retry / 2 tasks_landed = 0.5, got: {result['train_cas_retry_rate']}"
        )
        assert result['baseline_cas_retry_rate'] == 1.0, (
            f"expected 2 solo retries / 2 solo_landed = 1.0, got: {result['baseline_cas_retry_rate']}"
        )
        assert result['cas_retry_rate_delta'] == 0.5, (
            f"expected baseline(1.0) - train(0.5) = 0.5, got: {result['cas_retry_rate_delta']}"
        )


# ---------------------------------------------------------------------------
# TestAggregatorTrainThroughput — step-12 RED / step-13 GREEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAggregatorTrainThroughput:
    """step-12: build_per_project_merge_queue includes 'train_throughput' key.

    Verifies:
    - each project result dict contains a 'train_throughput' key
    - its value is the train_throughput_stats dict (all-zeros for empty/None db)
    - populated with a train_merged row: trains_landed=1, tasks_landed=2
    """

    async def test_none_db_includes_train_throughput_zeros(self):
        """db=None project includes train_throughput with all-zeros default."""
        from datetime import UTC, datetime

        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 1, 10, 12, 0, 0, tzinfo=UTC)
        result = await build_per_project_merge_queue(
            [('/tmp/proj-none', None)],
            hours=24,
            now=now,
        )

        assert 'train_throughput' in result['/tmp/proj-none'], (
            "expected 'train_throughput' key in per-project result (got None-db path)"
        )
        tt = result['/tmp/proj-none']['train_throughput']
        assert tt['trains_landed'] == 0
        assert tt['tasks_landed_via_trains'] == 0
        assert tt['improved'] is False

    async def test_populated_db_includes_train_throughput(self, tmp_path):
        """DB with a train_merged row → train_throughput has trains_landed=1."""
        from datetime import UTC, datetime, timedelta

        from dashboard.data.merge_queue import build_per_project_merge_queue

        now = datetime(2026, 1, 10, 12, 0, 0, tzinfo=UTC)
        events = [
            dict(
                event_type='train_merged',
                timestamp=now - timedelta(hours=1),
                run_id='run-1',
                task_id='11',
                data={'train_id': 't1', 'member_task_ids': ['10', '11'],
                      'merge_commit_sha': 'abc', 'base_sha': 'def'},
            ),
        ]
        db_path = _make_db(tmp_path, 'agg_train.db', events)
        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await build_per_project_merge_queue(
                [('/tmp/proj-train', conn)],
                hours=24,
                now=now,
            )

        assert 'train_throughput' in result['/tmp/proj-train'], (
            "expected 'train_throughput' key in per-project result"
        )
        tt = result['/tmp/proj-train']['train_throughput']
        assert tt['trains_landed'] == 1
        assert tt['tasks_landed_via_trains'] == 2


# ---------------------------------------------------------------------------
# TestProbeLiveOneMetrics (step-07 RED / step-08 GREEN)
# ---------------------------------------------------------------------------


def _snapshot_with_metrics(entries: list, metrics: dict) -> dict:
    """Build a get_merge_queue snapshot dict that includes a 'metrics' key."""
    snap = _snapshot(entries)
    snap['metrics'] = metrics
    return snap


_SAMPLE_METRICS = {
    'retries_per_landing': 1.5,
    'drift_at_detection': {'count': 2, 'last': 3, 'mean': 2.5, 'max': 3},
    'landings_total': 2,
    'retries_total': 3,
}


class TestProbeLiveOneMetrics:
    """_probe_live_one and fetch_live_merge_queues must preserve the 'metrics' block.

    RED until step-08 GREEN extends _probe_live_one to keep result['metrics'].
    """

    @pytest.mark.asyncio
    async def test_probe_live_one_preserves_metrics_key(self, _clean_live_sessions):
        """_probe_live_one returns 'metrics' when snapshot contains it."""
        from dashboard.data.merge_queue import _probe_live_one

        snap = _snapshot_with_metrics([], _SAMPLE_METRICS)
        handler = _PerPortHandler({8400: snap})
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await _probe_live_one(client, 'http://127.0.0.1:8400', timeout=5.0)

        assert result['reachable'] is True
        assert 'metrics' in result, (
            f"_probe_live_one must carry through the snapshot 'metrics' key; "
            f"got keys: {list(result.keys())}"
        )
        assert result['metrics'] == _SAMPLE_METRICS

    @pytest.mark.asyncio
    async def test_probe_live_one_metrics_with_entries(self, _clean_live_sessions):
        """'metrics' is preserved when entries are present too."""
        from dashboard.data.merge_queue import _probe_live_one

        entries = [
            {'task_id': '7', 'branch': 'task/7', 'state': 'queued',
             'age_secs': 10.0, 'position': 1, 'waiter_alive': True},
        ]
        snap = _snapshot_with_metrics(entries, _SAMPLE_METRICS)
        handler = _PerPortHandler({8401: snap})
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await _probe_live_one(client, 'http://127.0.0.1:8401', timeout=5.0)

        assert result['reachable'] is True
        assert len(result['entries']) == 1
        assert 'metrics' in result
        assert result['metrics']['retries_per_landing'] == 1.5

    @pytest.mark.asyncio
    async def test_probe_live_one_metrics_none_when_absent(self, _clean_live_sessions):
        """When snapshot has no 'metrics', result['metrics'] is None (safe default)."""
        from dashboard.data.merge_queue import _probe_live_one

        snap = _snapshot([])  # no 'metrics' key
        handler = _PerPortHandler({8402: snap})
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await _probe_live_one(client, 'http://127.0.0.1:8402', timeout=5.0)

        assert result['reachable'] is True
        assert 'metrics' in result
        assert result['metrics'] is None

    @pytest.mark.asyncio
    async def test_probe_live_one_metrics_none_on_unreachable(self, _clean_live_sessions):
        """Unreachable hosts return metrics=None in the error dict."""
        from dashboard.data.merge_queue import _probe_live_one

        handler = _PerPortHandler(fail_ports={8403})
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await _probe_live_one(client, 'http://127.0.0.1:8403', timeout=5.0)

        assert result['reachable'] is False
        assert 'metrics' in result
        assert result['metrics'] is None

    @pytest.mark.asyncio
    async def test_fetch_live_merge_queues_preserves_metrics(self, _clean_live_sessions):
        """fetch_live_merge_queues threads 'metrics' through for each project."""
        from dashboard.data.merge_queue import fetch_live_merge_queues

        snap = _snapshot_with_metrics([], _SAMPLE_METRICS)
        handler = _PerPortHandler({8410: snap})
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_live_merge_queues(client, _live_urls(8410))

        proj = result['proj8410']
        assert proj['reachable'] is True
        assert 'metrics' in proj
        assert proj['metrics']['retries_per_landing'] == 1.5
        assert proj['metrics']['drift_at_detection']['last'] == 3


class TestProbeLiveOneTimeoutBudget:
    """The live-queue probe's budget must reach client.post, not just wait_for.

    Twin of ``test_merge_halt.TestProbeOneTimeoutBudget``. ``timeout=`` on
    ``client.post`` also governs **pool acquisition** on the shared client,
    so without threading, a probe on a 2.0s budget could still block for
    httpx's 10s default waiting on a free connection slot.

    AsyncMock rather than MockTransport deliberately: MockTransport never
    surfaces the ``timeout`` kwarg to its handler.
    """

    @pytest.mark.asyncio
    async def test_budget_reaches_every_post(self, _clean_live_sessions):
        from unittest.mock import AsyncMock

        from dashboard.data.merge_queue import _probe_live_one

        url = 'http://127.0.0.1:8200'
        mock_client = AsyncMock()
        mock_client.post.side_effect = cold_session_responses(
            _snapshot([]), url,
        )

        result = await _probe_live_one(mock_client, url, 5.0)

        assert result['reachable'] is True, f'probe should have succeeded: {result}'
        timeouts = [c.kwargs['timeout'] for c in mock_client.post.call_args_list]
        assert timeouts == [5.0, 5.0, 5.0], (
            f"the probe budget must reach every post, not httpx's 10s "
            f'default, got {timeouts}'
        )
