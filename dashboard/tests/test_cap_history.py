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

from dashboard.data.cap_history import merge_all_accounts_capped  # noqa: E402


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
