"""Tests for dashboard.data.merge_queue — merge queue query functions."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

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
# TestQueueDepthTimeseries
# ---------------------------------------------------------------------------

class TestQueueDepthTimeseries:
    @pytest.mark.asyncio
    async def test_buckets_15min_over_24h(self, merge_events_db):
        """24h window produces 96 buckets; events fall in correct buckets."""
        now = datetime.now(UTC)
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
            result = await queue_depth_timeseries(db, hours=24)

        labels = result['labels']
        values = result['values']

        assert len(labels) == 96
        assert len(values) == 96
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
    async def test_none_db(self):
        """None DB returns empty ChartData."""
        result = await queue_depth_timeseries(None, hours=24)
        assert result == {'labels': [], 'values': []}

    @pytest.mark.asyncio
    async def test_empty_db(self, empty_merge_events_conn):
        """No events → 96 buckets all with count 0."""
        result = await queue_depth_timeseries(empty_merge_events_conn, hours=24)
        assert len(result['labels']) == 96
        assert len(result['values']) == 96
        assert all(v == 0 for v in result['values'])


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

    @pytest.mark.asyncio
    async def test_aggregate_queue_depth_timeseries(self, tmp_path):
        """Counts per bucket sum across two DBs."""
        now = datetime.now(UTC)
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
            result = await aggregate_queue_depth_timeseries([conn1, conn2], hours=24)

        assert len(result['labels']) == 96
        bucket_label = bucket.isoformat()
        assert bucket_label in result['labels']
        idx = result['labels'].index(bucket_label)
        assert result['values'][idx] == 3  # 2 + 1

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
