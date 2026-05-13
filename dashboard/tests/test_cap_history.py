"""Tests for dashboard.data.cap_history — cap-interval reader and helpers."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import aiosqlite
import pytest

# ---------------------------------------------------------------------------
# Schema — only account_events is needed for cap_history
# ---------------------------------------------------------------------------

CAP_HISTORY_SCHEMA = """\
CREATE TABLE IF NOT EXISTS account_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_name TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    project_id   TEXT,
    run_id       TEXT,
    details      TEXT,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_acct_evt_account
    ON account_events(account_name);
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cap_db(tmp_path):
    """Empty account_events DB (schema only). Tests insert their own rows."""
    db_path = tmp_path / 'cap_events.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(CAP_HISTORY_SCHEMA)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
async def cap_conn(cap_db):
    async with aiosqlite.connect(str(cap_db)) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_event(conn_sync: sqlite3.Connection, account: str, event_type: str, ts: datetime) -> None:
    conn_sync.execute(
        'INSERT INTO account_events (account_name, event_type, created_at) VALUES (?, ?, ?)',
        (account, event_type, ts.isoformat()),
    )
    conn_sync.commit()


def _make_db_with_events(tmp_path, name: str, events: list[tuple[str, str, datetime]]):
    """Create a fresh DB file and insert events; return path."""
    db_path = tmp_path / name
    conn_sync = sqlite3.connect(str(db_path))
    conn_sync.executescript(CAP_HISTORY_SCHEMA)
    for account, event_type, ts in events:
        conn_sync.execute(
            'INSERT INTO account_events (account_name, event_type, created_at) VALUES (?, ?, ?)',
            (account, event_type, ts.isoformat()),
        )
    conn_sync.commit()
    conn_sync.close()
    return db_path


# ---------------------------------------------------------------------------
# Step-1 tests: CapInterval dataclass + read_cap_intervals
# ---------------------------------------------------------------------------

from dashboard.data.cap_history import CapInterval, read_cap_intervals  # noqa: E402


class TestCapInterval:
    def test_equality(self):
        now = datetime.now(UTC)
        a = CapInterval(account_name='acc', start=now, end=None)
        b = CapInterval(account_name='acc', start=now, end=None)
        assert a == b

    def test_hashable(self):
        now = datetime.now(UTC)
        iv = CapInterval(account_name='acc', start=now, end=now + timedelta(hours=1))
        s = {iv}
        assert iv in s

    def test_end_none_permitted(self):
        iv = CapInterval(account_name='x', start=datetime.now(UTC), end=None)
        assert iv.end is None

    def test_frozen_immutable(self):
        iv = CapInterval(account_name='x', start=datetime.now(UTC), end=None)
        with pytest.raises((AttributeError, TypeError)):
            iv.account_name = 'y'  # type: ignore[misc]


class TestReadCapIntervals:
    @pytest.mark.asyncio
    async def test_empty_db(self, cap_conn):
        result = await read_cap_intervals([cap_conn], days=7)
        assert result == []

    @pytest.mark.asyncio
    async def test_single_closed_pair(self, tmp_path):
        """cap_hit@T1 + resumed@T2 → CapInterval(start=T1, end=T2)."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=2)
        t2 = now - timedelta(hours=1)
        db_path = _make_db_with_events(tmp_path, 'single.db', [
            ('acc-a', 'cap_hit', t1),
            ('acc-a', 'resumed', t2),
        ])
        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await read_cap_intervals([conn], days=7)

        assert len(result) == 1
        iv = result[0]
        assert iv.account_name == 'acc-a'
        # Allow ±1 second for isoformat round-trip
        assert abs((iv.start - t1).total_seconds()) < 1
        assert iv.end is not None
        assert abs((iv.end - t2).total_seconds()) < 1

    @pytest.mark.asyncio
    async def test_open_ended_cap(self, tmp_path):
        """cap_hit@T1 with no resumed → CapInterval(end=None)."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=3)
        db_path = _make_db_with_events(tmp_path, 'open.db', [
            ('acc-b', 'cap_hit', t1),
        ])
        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await read_cap_intervals([conn], days=7)

        assert len(result) == 1
        assert result[0].end is None
        assert result[0].account_name == 'acc-b'

    @pytest.mark.asyncio
    async def test_fifo_pairing(self, tmp_path):
        """cap1@T1, cap2@T2, resumed@T3, resumed@T4 → (T1,T3) and (T2,T4)."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=4)
        t2 = now - timedelta(hours=3)
        t3 = now - timedelta(hours=2)
        t4 = now - timedelta(hours=1)
        db_path = _make_db_with_events(tmp_path, 'fifo.db', [
            ('acc-c', 'cap_hit', t1),
            ('acc-c', 'cap_hit', t2),
            ('acc-c', 'resumed', t3),
            ('acc-c', 'resumed', t4),
        ])
        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await read_cap_intervals([conn], days=7)

        assert len(result) == 2
        # Sort by start for deterministic assertion
        result.sort(key=lambda iv: iv.start)
        assert abs((result[0].start - t1).total_seconds()) < 1
        assert result[0].end is not None
        assert abs((result[0].end - t3).total_seconds()) < 1
        assert abs((result[1].start - t2).total_seconds()) < 1
        assert result[1].end is not None
        assert abs((result[1].end - t4).total_seconds()) < 1

    @pytest.mark.asyncio
    async def test_multi_account_partitioned(self, tmp_path):
        """Interleaved events from two accounts are partitioned per account."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=4)
        t2 = now - timedelta(hours=3)
        t3 = now - timedelta(hours=2)
        t4 = now - timedelta(hours=1)
        db_path = _make_db_with_events(tmp_path, 'multi_acct.db', [
            ('alpha', 'cap_hit', t1),
            ('beta', 'cap_hit', t2),
            ('alpha', 'resumed', t3),
            ('beta', 'resumed', t4),
        ])
        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await read_cap_intervals([conn], days=7)

        assert len(result) == 2
        by_account = {iv.account_name: iv for iv in result}
        assert set(by_account.keys()) == {'alpha', 'beta'}
        # alpha: T1→T3
        assert abs((by_account['alpha'].start - t1).total_seconds()) < 1
        assert by_account['alpha'].end is not None
        assert abs((by_account['alpha'].end - t3).total_seconds()) < 1
        # beta: T2→T4
        assert abs((by_account['beta'].start - t2).total_seconds()) < 1
        assert by_account['beta'].end is not None
        assert abs((by_account['beta'].end - t4).total_seconds()) < 1

    @pytest.mark.asyncio
    async def test_multi_db_flat_union(self, tmp_path):
        """Results from multiple DBs are merged into one flat list."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=2)
        t2 = now - timedelta(hours=1)
        db1 = _make_db_with_events(tmp_path, 'db1.db', [
            ('acc-x', 'cap_hit', t1),
        ])
        db2 = _make_db_with_events(tmp_path, 'db2.db', [
            ('acc-y', 'cap_hit', t2),
        ])
        async with (
            aiosqlite.connect(str(db1)) as conn1,
            aiosqlite.connect(str(db2)) as conn2,
        ):
            conn1.row_factory = aiosqlite.Row
            conn2.row_factory = aiosqlite.Row
            result = await read_cap_intervals([conn1, conn2], days=7)

        assert len(result) == 2
        accounts = {iv.account_name for iv in result}
        assert accounts == {'acc-x', 'acc-y'}

    @pytest.mark.asyncio
    async def test_days_cutoff_excludes_old(self, tmp_path):
        """Events older than `days` are excluded."""
        now = datetime.now(UTC)
        recent = now - timedelta(hours=2)
        old = now - timedelta(days=10)
        db_path = _make_db_with_events(tmp_path, 'cutoff.db', [
            ('acc-r', 'cap_hit', recent),
            ('acc-o', 'cap_hit', old),
        ])
        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await read_cap_intervals([conn], days=7)

        accounts = {iv.account_name for iv in result}
        assert 'acc-r' in accounts
        assert 'acc-o' not in accounts

    @pytest.mark.asyncio
    async def test_none_db_skipped(self, cap_conn):
        """A None entry in dbs is skipped without raising."""
        result = await read_cap_intervals([None, cap_conn, None], days=7)
        assert result == []


# ---------------------------------------------------------------------------
# Step-3 tests: merge_all_accounts_capped
# ---------------------------------------------------------------------------

from dashboard.data.cap_history import compute_overlap_ms, merge_all_accounts_capped  # noqa: E402


def _iv(account: str, start: datetime, end: datetime | None) -> CapInterval:
    return CapInterval(account_name=account, start=start, end=end)


class TestMergeAllAccountsCapped:
    def test_empty_intervals_non_empty_accounts(self):
        """Empty intervals with non-empty account_names → []."""
        now = datetime.now(UTC)
        result = merge_all_accounts_capped([], ['alpha', 'beta'])
        assert result == []

    def test_single_account_single_closed_interval(self):
        """Single account, single closed interval → [(T1, T2)]."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=2)
        t2 = now - timedelta(hours=1)
        intervals = [_iv('alpha', t1, t2)]
        result = merge_all_accounts_capped(intervals, ['alpha'])
        assert len(result) == 1
        assert abs((result[0][0] - t1).total_seconds()) < 1
        assert result[0][1] is not None
        assert abs((result[0][1] - t2).total_seconds()) < 1

    def test_two_accounts_full_overlap(self):
        """Both accounts capped over [T1, T2] → [(T1, T2)]."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=2)
        t2 = now - timedelta(hours=1)
        intervals = [
            _iv('alpha', t1, t2),
            _iv('beta', t1, t2),
        ]
        result = merge_all_accounts_capped(intervals, ['alpha', 'beta'])
        assert len(result) == 1
        assert abs((result[0][0] - t1).total_seconds()) < 1
        assert result[0][1] is not None
        assert abs((result[0][1] - t2).total_seconds()) < 1

    def test_two_accounts_partial_overlap(self):
        """A: T1..T3, B: T2..T4 → [(T2, T3)]."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=4)
        t2 = now - timedelta(hours=3)
        t3 = now - timedelta(hours=2)
        t4 = now - timedelta(hours=1)
        intervals = [
            _iv('alpha', t1, t3),
            _iv('beta', t2, t4),
        ]
        result = merge_all_accounts_capped(intervals, ['alpha', 'beta'])
        assert len(result) == 1
        assert abs((result[0][0] - t2).total_seconds()) < 1
        assert result[0][1] is not None
        assert abs((result[0][1] - t3).total_seconds()) < 1

    def test_two_accounts_no_overlap(self):
        """A: T1..T2, B: T3..T4 (non-overlapping) → []."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=4)
        t2 = now - timedelta(hours=3)
        t3 = now - timedelta(hours=2)
        t4 = now - timedelta(hours=1)
        intervals = [
            _iv('alpha', t1, t2),
            _iv('beta', t3, t4),
        ]
        result = merge_all_accounts_capped(intervals, ['alpha', 'beta'])
        assert result == []

    def test_one_open_ended_other_closed(self):
        """A open-ended from T1, B closed T1..T3 → [(T1, T3)]."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=3)
        t3 = now - timedelta(hours=1)
        intervals = [
            _iv('alpha', t1, None),  # open-ended
            _iv('beta', t1, t3),
        ]
        result = merge_all_accounts_capped(intervals, ['alpha', 'beta'])
        assert len(result) == 1
        assert abs((result[0][0] - t1).total_seconds()) < 1
        assert result[0][1] is not None
        assert abs((result[0][1] - t3).total_seconds()) < 1

    def test_both_accounts_open_ended(self):
        """Both accounts open-ended from T1 → [(T1, None)]."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=2)
        intervals = [
            _iv('alpha', t1, None),
            _iv('beta', t1, None),
        ]
        result = merge_all_accounts_capped(intervals, ['alpha', 'beta'])
        assert len(result) == 1
        assert abs((result[0][0] - t1).total_seconds()) < 1
        assert result[0][1] is None

    def test_account_with_zero_intervals_short_circuits(self):
        """account_names includes an account with zero intervals → []."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=2)
        t2 = now - timedelta(hours=1)
        # Only alpha has intervals; beta has none
        intervals = [_iv('alpha', t1, t2)]
        result = merge_all_accounts_capped(intervals, ['alpha', 'beta'])
        assert result == []

    def test_three_accounts_partial_chain(self):
        """Three accounts; only segment where all three overlap is returned."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=5)
        t2 = now - timedelta(hours=4)
        t3 = now - timedelta(hours=3)
        t4 = now - timedelta(hours=2)
        t5 = now - timedelta(hours=1)
        # alpha: T1..T5, beta: T2..T5, gamma: T3..T4
        # All three capped: T3..T4
        intervals = [
            _iv('alpha', t1, t5),
            _iv('beta', t2, t5),
            _iv('gamma', t3, t4),
        ]
        result = merge_all_accounts_capped(intervals, ['alpha', 'beta', 'gamma'])
        assert len(result) == 1
        assert abs((result[0][0] - t3).total_seconds()) < 1
        assert result[0][1] is not None
        assert abs((result[0][1] - t4).total_seconds()) < 1


# ---------------------------------------------------------------------------
# Step-5 tests: compute_overlap_ms
# ---------------------------------------------------------------------------

class TestComputeOverlapMs:
    def _sec(self, s: float) -> int:
        """Convert seconds to milliseconds."""
        return int(s * 1000)

    def test_empty_capped_returns_zero(self):
        now = datetime.now(UTC)
        assert compute_overlap_ms(now - timedelta(hours=1), now, []) == 0

    def test_capped_entirely_before_window(self):
        now = datetime.now(UTC)
        start = now - timedelta(hours=2)
        end = now - timedelta(hours=1)
        # Capped interval ends before window starts
        c_start = now - timedelta(hours=5)
        c_end = now - timedelta(hours=3)
        assert compute_overlap_ms(start, end, [(c_start, c_end)]) == 0

    def test_capped_entirely_after_window(self):
        now = datetime.now(UTC)
        start = now - timedelta(hours=5)
        end = now - timedelta(hours=4)
        c_start = now - timedelta(hours=2)
        c_end = now - timedelta(hours=1)
        assert compute_overlap_ms(start, end, [(c_start, c_end)]) == 0

    def test_capped_contains_full_window(self):
        now = datetime.now(UTC)
        start = now - timedelta(hours=2)
        end = now - timedelta(hours=1)
        c_start = now - timedelta(hours=3)
        c_end = now
        expected = self._sec(3600)  # 1 hour
        assert compute_overlap_ms(start, end, [(c_start, c_end)]) == expected

    def test_capped_starts_before_ends_inside(self):
        now = datetime.now(UTC)
        start = now - timedelta(hours=2)
        end = now - timedelta(hours=1)
        c_start = now - timedelta(hours=3)
        c_end = now - timedelta(minutes=90)  # ends 30 min into window
        expected = self._sec(1800)  # 30 min
        assert compute_overlap_ms(start, end, [(c_start, c_end)]) == expected

    def test_capped_starts_inside_ends_after(self):
        now = datetime.now(UTC)
        start = now - timedelta(hours=2)
        end = now - timedelta(hours=1)
        c_start = now - timedelta(minutes=90)  # starts 30 min into window
        c_end = now
        expected = self._sec(1800)  # 30 min
        assert compute_overlap_ms(start, end, [(c_start, c_end)]) == expected

    def test_capped_strictly_inside_window(self):
        now = datetime.now(UTC)
        start = now - timedelta(hours=3)
        end = now - timedelta(hours=1)
        c_start = now - timedelta(hours=2, minutes=30)
        c_end = now - timedelta(hours=1, minutes=30)
        expected = self._sec(3600)  # 1 hour
        assert compute_overlap_ms(start, end, [(c_start, c_end)]) == expected

    def test_capped_open_ended_starts_before_end(self):
        now = datetime.now(UTC)
        start = now - timedelta(hours=2)
        end = now - timedelta(hours=1)
        c_start = now - timedelta(hours=3)
        # c_end = None → clamp to end
        expected = self._sec(3600)  # full window
        assert compute_overlap_ms(start, end, [(c_start, None)]) == expected

    def test_multiple_capped_tuples_summed(self):
        now = datetime.now(UTC)
        start = now - timedelta(hours=4)
        end = now
        # Two non-overlapping capped intervals: each 30 min
        c1_start = now - timedelta(hours=3)
        c1_end = now - timedelta(hours=2, minutes=30)  # 30 min
        c2_start = now - timedelta(hours=1)
        c2_end = now - timedelta(minutes=30)  # 30 min
        expected = self._sec(3600)  # 30 + 30 min
        assert compute_overlap_ms(start, end, [(c1_start, c1_end), (c2_start, c2_end)]) == expected


# ---------------------------------------------------------------------------
# Step-7 tests: bucketise_cap_sparkline
# ---------------------------------------------------------------------------

from dashboard.data.cap_history import bucketise_cap_sparkline  # noqa: E402


class TestBucketiseCapSparkline:
    def test_shape_default_params(self):
        """labels and values have length == window_hours*3600//bucket_seconds."""
        now = datetime.now(UTC)
        result = bucketise_cap_sparkline([], window_hours=24, bucket_seconds=600, now=now)
        expected_buckets = (24 * 3600) // 600  # 144
        assert len(result['labels']) == expected_buckets
        assert len(result['values']) == expected_buckets

    def test_all_zeros_when_empty_capped(self):
        """Empty capped list → all values zero."""
        now = datetime.now(UTC)
        result = bucketise_cap_sparkline([], window_hours=2, bucket_seconds=600, now=now)
        assert all(v == 0 for v in result['values'])

    def test_all_values_are_zero_or_one(self):
        """All values are in {0, 1}."""
        now = datetime.now(UTC)
        capped = [(now - timedelta(hours=1), now - timedelta(minutes=30))]
        result = bucketise_cap_sparkline(capped, window_hours=4, bucket_seconds=600, now=now)
        assert all(v in (0, 1) for v in result['values'])

    def test_labels_are_iso_parseable_utc(self):
        """All labels can be parsed as UTC datetimes."""
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = bucketise_cap_sparkline([], window_hours=1, bucket_seconds=600, now=now)
        for label in result['labels']:
            dt = datetime.fromisoformat(label)
            assert dt.tzinfo is not None

    def test_last_label_equals_now(self):
        """The last label is exactly `now` (right-edge of last bucket)."""
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = bucketise_cap_sparkline([], window_hours=2, bucket_seconds=600, now=now)
        last_label_dt = datetime.fromisoformat(result['labels'][-1])
        assert abs((last_label_dt - now).total_seconds()) < 1

    def test_labels_monotonically_increasing_by_bucket_seconds(self):
        """Consecutive labels differ by exactly bucket_seconds."""
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        bucket_seconds = 600
        result = bucketise_cap_sparkline([], window_hours=2, bucket_seconds=bucket_seconds, now=now)
        labels = [datetime.fromisoformat(lb) for lb in result['labels']]
        for i in range(1, len(labels)):
            diff = (labels[i] - labels[i - 1]).total_seconds()
            assert abs(diff - bucket_seconds) < 0.001

    def test_capped_window_marks_correct_buckets(self):
        """Buckets whose right edge falls inside [now-1h, now-30min] → 1."""
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        c_start = now - timedelta(hours=1)
        c_end = now - timedelta(minutes=30)
        # 24h window, 10-min buckets. Buckets with right edge in [c_start, c_end]
        # are those at offsets: -60, -50, -40, -30 min from now.
        result = bucketise_cap_sparkline(
            [(c_start, c_end)],
            window_hours=24,
            bucket_seconds=600,
            now=now,
        )
        labels = [datetime.fromisoformat(lb) for lb in result['labels']]
        values = result['values']
        for label, value in zip(labels, values):
            if c_start <= label <= c_end:
                assert value == 1, f"Expected 1 at {label}"
            else:
                assert value == 0, f"Expected 0 at {label}"

    def test_open_ended_cap_marks_all_buckets_after_start(self):
        """Open-ended cap starting 2h ago → all buckets from 2h onwards are 1."""
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        c_start = now - timedelta(hours=2)
        result = bucketise_cap_sparkline(
            [(c_start, None)],
            window_hours=4,
            bucket_seconds=600,
            now=now,
        )
        labels = [datetime.fromisoformat(lb) for lb in result['labels']]
        values = result['values']
        for label, value in zip(labels, values):
            if label >= c_start:
                assert value == 1, f"Expected 1 at {label} (open-ended)"
            # (buckets before c_start should be 0 — but c_start might align
            # with a bucket edge so we only assert the ≥ direction)

    def test_cap_entirely_outside_window_all_zero(self):
        """Cap interval entirely before the window → all zero."""
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        c_start = now - timedelta(hours=30)
        c_end = now - timedelta(hours=25)
        result = bucketise_cap_sparkline(
            [(c_start, c_end)],
            window_hours=24,
            bucket_seconds=600,
            now=now,
        )
        assert all(v == 0 for v in result['values'])
