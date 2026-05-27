"""Tests for dashboard.data.cap_history — cap-interval reader and helpers."""

from __future__ import annotations

import dataclasses
import sqlite3
from datetime import UTC, datetime, timedelta, timezone

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


def _insert_event(
    conn_sync: sqlite3.Connection, account: str, event_type: str, ts: datetime
) -> None:
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
        with pytest.raises(dataclasses.FrozenInstanceError):
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
        db_path = _make_db_with_events(
            tmp_path,
            'single.db',
            [
                ('acc-a', 'cap_hit', t1),
                ('acc-a', 'resumed', t2),
            ],
        )
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
        db_path = _make_db_with_events(
            tmp_path,
            'open.db',
            [
                ('acc-b', 'cap_hit', t1),
            ],
        )
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
        db_path = _make_db_with_events(
            tmp_path,
            'fifo.db',
            [
                ('acc-c', 'cap_hit', t1),
                ('acc-c', 'cap_hit', t2),
                ('acc-c', 'resumed', t3),
                ('acc-c', 'resumed', t4),
            ],
        )
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
        db_path = _make_db_with_events(
            tmp_path,
            'multi_acct.db',
            [
                ('alpha', 'cap_hit', t1),
                ('beta', 'cap_hit', t2),
                ('alpha', 'resumed', t3),
                ('beta', 'resumed', t4),
            ],
        )
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
        db1 = _make_db_with_events(
            tmp_path,
            'db1.db',
            [
                ('acc-x', 'cap_hit', t1),
            ],
        )
        db2 = _make_db_with_events(
            tmp_path,
            'db2.db',
            [
                ('acc-y', 'cap_hit', t2),
            ],
        )
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
        db_path = _make_db_with_events(
            tmp_path,
            'cutoff.db',
            [
                ('acc-r', 'cap_hit', recent),
                ('acc-o', 'cap_hit', old),
            ],
        )
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

    @pytest.mark.asyncio
    async def test_days_zero_raises_value_error(self, cap_conn):
        """days=0 must raise ValueError with message containing 'positive'."""
        with pytest.raises(ValueError, match='positive'):
            await read_cap_intervals([cap_conn], days=0)

    @pytest.mark.asyncio
    async def test_days_negative_raises_value_error(self, cap_conn):
        """days=-1 must raise ValueError with message containing 'positive'."""
        with pytest.raises(ValueError, match='positive'):
            await read_cap_intervals([cap_conn], days=-1)

    @pytest.mark.asyncio
    async def test_honors_now_parameter(self, tmp_path):
        """now= shifts the cutoff so events outside the default window are included.

        Inserts a single cap_hit at real_now - 30 days.

        1. read_cap_intervals([conn], days=7) with default now: the event is
           outside the 7-day window → result is [].
        2. read_cap_intervals([conn], days=7, now=real_now - 28 days): the
           synthetic now shifts the cutoff to real_now - 35 days, so the
           30-day-old event IS within the window → result has 1 CapInterval.

        Failing baseline: read_cap_intervals currently has signature
        (dbs, *, days: int) — passing now=... raises TypeError.
        """
        real_now = datetime.now(UTC)
        event_ts = real_now - timedelta(days=30)
        db_path = _make_db_with_events(
            tmp_path,
            'now_param.db',
            [
                ('acc-x', 'cap_hit', event_ts),
            ],
        )
        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row

            # 1. Default now: 30-day-old event outside 7-day window
            result_default = await read_cap_intervals([conn], days=7)
            assert result_default == [], (
                f'Expected [] with default now (event 30d old, window 7d), got {result_default}'
            )

            # 2. Synthetic now 28 days ago: cutoff shifts to 35 days ago, includes event
            synthetic_now = real_now - timedelta(days=28)
            result_shifted = await read_cap_intervals([conn], days=7, now=synthetic_now)
            assert len(result_shifted) == 1, (
                f'Expected 1 interval with now=real_now-28d (cutoff 35d ago), got {result_shifted}'
            )
            assert result_shifted[0].account_name == 'acc-x'
            assert result_shifted[0].end is None  # open-ended (no resumed)

    @pytest.mark.asyncio
    async def test_naive_now_normalized_to_utc(self, tmp_path):
        """A naive `now` must yield identical results to the equivalent tz-aware `now`.

        WHY a naive DB row is inserted:
        With tz-aware DB rows (production format), the lexicographic naive-vs-aware
        cutoff mismatch coincidentally yields identical inclusion for all realistic
        instants — so an aware-only test *cannot* go red.  The divergence is only
        observable at a naive boundary row:

        - naive `now` → cutoff is offset-less, e.g. ``"2026-05-08T12:00:00"``
          The row's created_at is the *same* naive string → ``created_at >= cutoff``
          is True (equal) → row is INCLUDED → 1 interval returned.
        - tz-aware `now` → cutoff is ``"2026-05-08T12:00:00+00:00"``
          That string sorts *after* the naive row string lexicographically → ``>=``
          is False → row is EXCLUDED → 0 intervals returned.

        Before the fix the two calls diverge (naive→1, aware→0) so the equivalence
        assertion fails red.  After the fix both normalize to tz-aware before
        computing cutoff → both yield 0 → assertion passes green.
        """
        real_now = datetime.now(UTC)
        cutoff_instant = real_now - timedelta(days=7)

        # Insert the boundary event as a NAIVE datetime so _make_db_with_events
        # stores its isoformat without a +00:00 suffix — the only configuration
        # where the lexicographic mismatch is detectable.
        db_path = _make_db_with_events(
            tmp_path,
            'naive_now.db',
            [
                ('acc-n', 'cap_hit', cutoff_instant.replace(tzinfo=None)),
            ],
        )

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row

            result_naive = await read_cap_intervals(
                [conn], days=7, now=real_now.replace(tzinfo=None)
            )
            result_aware = await read_cap_intervals([conn], days=7, now=real_now)

        # Both should be empty: the boundary-naive row is AT the cutoff, but
        # with a consistent tz-aware cutoff ``created_at >= cutoff`` is False
        # (the row string sorts before the +00:00 cutoff string).
        assert len(result_naive) == len(result_aware), (
            f'naive now → {len(result_naive)} interval(s), '
            f'aware now → {len(result_aware)} interval(s); expected equal counts. '
            f'Bug: naive cutoff has no +00:00 so it equals the naive boundary row '
            f'(>= true), whereas aware cutoff sorts after it (>= false).'
        )
        assert {iv.account_name for iv in result_naive} == {iv.account_name for iv in result_aware}
        assert result_aware == [], (
            f'Expected 0 intervals with tz-aware now (boundary row at exact cutoff '
            f'is excluded under half-open semantics), got {result_aware}'
        )

    @pytest.mark.asyncio
    async def test_non_utc_now_normalized_to_utc(self, tmp_path):
        """A tz-aware but non-UTC ``now`` must produce the same cutoff as its UTC equivalent.

        The divergence band:
        Without ``astimezone(UTC)``, a ``now`` in ``-08:00`` produces a cutoff
        string like ``"YYYY-MM-DDTHH:MM:SS-08:00"`` where the wall-clock hour
        is 8 hours *earlier* than the UTC wall-clock hour.  A DB row stored as
        ``"(Cwall-1h)...+00:00"`` — one UTC hour before the true cutoff — has
        a wall-clock prefix 7 hours *after* the buggy cutoff's wall-clock prefix
        (because 19 > 12 when UTC is 20 and the buggy -08:00 cutoff is at 12).
        That 7-hour gap means the lexicographic comparison always resolves on the
        fixed-width ``YYYY-MM-DDTHH:MM:SS`` prefix, never on the ``+``/``-``
        suffix, so the comparison is deterministic regardless of date position.

        RED/GREEN proof:
        - ``result_utc`` (control): cutoff = ``Cwall...+00:00``; row
          ``(Cwall-1h)...+00:00`` < cutoff (fixed-width prefix: "19" < "20") →
          EXCLUDED → ``result_utc == []``, both pre- and post-fix.
        - ``result_minus8`` (probe): PRE-FIX the buggy cutoff is
          ``(Cwall-8h)...-08:00``; the row ``(Cwall-1h)...+00:00`` has prefix
          "19" > "12" → row > cutoff → INCLUDED → ``len(result_minus8) == 1`` →
          ``assert result_minus8 == []`` FAILS red.  POST-FIX ``astimezone(UTC)``
          converts to UTC → cutoff = ``Cwall...+00:00`` → row excluded → green.

        WHY a tz-aware (+00:00) DB row is used:
        This exercises the *cutoff*-normalisation path, not the row-normalisation
        path (the identical ``replace(tzinfo=UTC)`` idiom at cap_history.py:121).
        """
        # A UTC reference instant
        real_now_utc = datetime.now(UTC)
        cutoff_instant_utc = real_now_utc - timedelta(days=7)

        # Equivalent now in a fixed -08:00 offset timezone
        tz_minus8 = timezone(timedelta(hours=-8))
        real_now_minus8 = real_now_utc.astimezone(tz_minus8)

        # Probe row: strictly 1 hour INSIDE the divergence band (1 hour before
        # the true cutoff).  This positions the row such that the correct UTC
        # cutoff excludes it, while the buggy -08:00 cutoff includes it.
        # The tz-aware (+00:00) format mirrors production DB rows, exercising
        # the cutoff-normalisation branch rather than the row-normalisation branch.
        probe_ts = cutoff_instant_utc - timedelta(hours=1)
        db_path = _make_db_with_events(
            tmp_path,
            'non_utc_now.db',
            [
                ('acc-x', 'cap_hit', probe_ts),
            ],
        )

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row

            result_utc = await read_cap_intervals([conn], days=7, now=real_now_utc)
            result_minus8 = await read_cap_intervals([conn], days=7, now=real_now_minus8)

        # Load-bearing exact-count assertions:
        # result_utc: correct UTC cutoff excludes probe row (pre- and post-fix)
        assert result_utc == [], (
            f'Expected [] with UTC now (probe row at cutoff-1h is before +00:00 '
            f'cutoff string), got {result_utc}'
        )
        # result_minus8: pre-fix the buggy -08:00 cutoff includes probe row (→
        # fails red); post-fix astimezone(UTC) normalises to UTC cutoff (→ green)
        assert result_minus8 == [], (
            f'Expected [] with -08:00 now (should normalize to UTC cutoff), '
            f'got {result_minus8}. '
            f'Bug: without astimezone(UTC), -08:00 cutoff string sorts before '
            f"the probe row's +00:00 string, causing false inclusion."
        )
        # Supplementary inter-result consistency checks
        assert len(result_utc) == len(result_minus8)
        assert {iv.account_name for iv in result_utc} == {iv.account_name for iv in result_minus8}

    @pytest.mark.asyncio
    async def test_db_row_non_utc_aware_canonised_to_utc(self, tmp_path):
        """DB rows with tz-aware non-UTC created_at are canonicalised to UTC.

        Locks the row-normalisation path in _to_utc: a tz-aware non-UTC
        timestamp stored in created_at (e.g. ``2026-01-01T04:00:00-08:00``)
        is parsed by fromisoformat then converted via astimezone(UTC), so the
        resulting CapInterval.start and .end carry UTC tzinfo and the correct
        UTC wall-clock value — not the original offset.
        """
        tz_minus8 = timezone(timedelta(hours=-8))
        # Represent "2026-01-01 12:00 UTC" as "2026-01-01 04:00 -08:00"
        cap_start_utc = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        cap_end_utc = datetime(2026, 1, 1, 13, 0, 0, tzinfo=UTC)
        db_path = _make_db_with_events(
            tmp_path,
            'non_utc_row.db',
            [
                ('acc-r', 'cap_hit', cap_start_utc.astimezone(tz_minus8)),
                ('acc-r', 'resumed', cap_end_utc.astimezone(tz_minus8)),
            ],
        )

        now = datetime(2026, 1, 1, 14, 0, 0, tzinfo=UTC)  # after both events
        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await read_cap_intervals([conn], days=7, now=now)

        assert len(result) == 1
        iv = result[0]
        # Bounds must be UTC-aware despite the DB row carrying -08:00 offset
        assert iv.start.tzinfo == UTC
        assert iv.end is not None
        assert iv.end.tzinfo == UTC
        # Wall-clock value must match the UTC equivalent (no shift)
        assert iv.start == cap_start_utc
        assert iv.end == cap_end_utc


# ---------------------------------------------------------------------------
# Step-3 tests: merge_all_accounts_capped
# ---------------------------------------------------------------------------

from dashboard.data.cap_history import compute_overlap_ms, merge_all_accounts_capped  # noqa: E402


def _iv(account: str, start: datetime, end: datetime | None) -> CapInterval:
    return CapInterval(account_name=account, start=start, end=end)


class TestMergeAllAccountsCapped:
    def test_empty_intervals_non_empty_accounts(self):
        """Empty intervals with non-empty account_names → []."""
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

    def test_touching_intervals_produce_no_window(self):
        """A:[t1,t2) and B:[t2,t3) merely touch at t2 → no merged window.

        Under half-open semantics, touching intervals share a single boundary
        point but have no common interior.  The zero-width filter at the end of
        merge_all_accounts_capped drops any merged window where start == end,
        so this correctly returns [].  This characterization test pins that
        behaviour: if the filter or sort is changed, this test will fail loudly.
        """
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=3)
        t2 = now - timedelta(hours=2)
        t3 = now - timedelta(hours=1)
        intervals = [
            _iv('alpha', t1, t2),
            _iv('beta', t2, t3),
        ]
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
# Step-9 (task-1280): compute_capped_now_and_windows helper
# ---------------------------------------------------------------------------

from dashboard.data.cap_history import compute_capped_now_and_windows  # noqa: E402


class TestComputeCappedNowAndWindows:
    """Tests for compute_capped_now_and_windows(intervals).

    Failing baseline for all methods: compute_capped_now_and_windows doesn't
    exist yet — the import above raises ImportError.
    """

    def test_empty_intervals_returns_zero_and_empty_windows(self):
        """Empty input → (0, [])."""
        capped_now, windows = compute_capped_now_and_windows([])
        assert capped_now == 0, f'Expected capped_now=0 for empty, got {capped_now}'
        assert windows == [], f'Expected [] windows for empty, got {windows}'

    def test_all_closed_intervals_returns_capped_now_zero(self):
        """All intervals closed → capped_now=0; windows may still be non-empty."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=3)
        t2 = now - timedelta(hours=2)
        # Two accounts both capped [t1, t2] — same window, so merge yields [(t1, t2)]
        intervals = [
            CapInterval('alpha', t1, t2),
            CapInterval('beta', t1, t2),
        ]
        capped_now, windows = compute_capped_now_and_windows(intervals)
        assert capped_now == 0, f'Expected capped_now=0 (all closed), got {capped_now}'
        assert len(windows) >= 1, (
            f'Expected >=1 merged window (both accounts fully overlapping), got {windows}'
        )

    def test_any_open_ended_returns_capped_now_one(self):
        """At least one open-ended interval → capped_now=1."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=2)
        # Simplified: just one open-ended is enough for capped_now=1
        open_only = [CapInterval('alpha', t1, None)]
        capped_now, windows = compute_capped_now_and_windows(open_only)
        assert capped_now == 1, f'Expected capped_now=1 for open-ended interval, got {capped_now}'
        assert len(windows) >= 1, f'Expected >=1 window (single open-ended account), got {windows}'

    def test_sorted_account_names_deterministic(self):
        """Two consecutive calls with differently-ordered inputs produce same windows."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=3)
        t2 = now - timedelta(hours=2)
        # Three accounts in non-sorted insertion order
        intervals_fwd = [
            CapInterval('gamma', t1, t2),
            CapInterval('alpha', t1, t2),
            CapInterval('beta', t1, t2),
        ]
        intervals_rev = list(reversed(intervals_fwd))
        _, windows1 = compute_capped_now_and_windows(intervals_fwd)
        _, windows2 = compute_capped_now_and_windows(intervals_rev)
        assert windows1 == windows2, (
            f'Expected deterministic windows regardless of input order. '
            f'Got {windows1} vs {windows2}'
        )

    def test_mixed_availability_returns_capped_now_zero(self):
        """(a) mixed: 'a' open-ended, 'b' closed → capped_now=0.

        With all-accounts-capped semantics: total=2, capped=1, available=1 → 0.
        Under old any-account semantics this would return 1 (a is open).
        """
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=3)
        t2 = now - timedelta(hours=1)
        intervals = [
            CapInterval('a', t1, None),  # open-ended
            CapInterval('b', t1, t2),    # closed
        ]
        capped_now, _ = compute_capped_now_and_windows(intervals)
        assert capped_now == 0, (
            f"mixed availability: 'b' is uncapped so capped_now must be 0; "
            f"==1 means old any-account semantics still active (got {capped_now})"
        )

    def test_account_count_denominator_returns_capped_now_zero(self):
        """(b) 1 open cap of 3 configured accounts → capped_now=0.

        This is THE reported-bug case at helper level: one account has a stale
        open-ended cap_hit while 2 healthy accounts have no cap events at all.
        Without total_accounts the universe would be 1 (all capped) → capped_now=1.
        With total_accounts=3: total=3, capped=1, available=2 → capped_now=0.
        """
        now = datetime.now(UTC)
        t1 = now - timedelta(minutes=10)
        intervals = [CapInterval('a', t1, None)]
        capped_now, _ = compute_capped_now_and_windows(intervals, total_accounts=3)
        assert capped_now == 0, (
            f'account_count denominator case: 1 capped of 3 total must return '
            f'capped_now=0 (2 available). Got {capped_now}. '
            f'Bug: if account_count ignored then total=1, available=0 → capped_now=1 '
            f'(the original 3-uncapped-accounts bug).'
        )

    def test_all_accounts_open_ended_returns_capped_now_one(self):
        """(c) both accounts open-ended → capped_now=1.

        Sanity check: all-accounts-capped semantics still returns 1 when ALL are capped.
        """
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=1)
        intervals = [
            CapInterval('a', t1, None),
            CapInterval('b', t1, None),
        ]
        capped_now, _ = compute_capped_now_and_windows(intervals)
        assert capped_now == 1, (
            f'Both accounts open-ended: expected capped_now=1, got {capped_now}'
        )


# ---------------------------------------------------------------------------
# TestSummarizeAccounts — pure helper summarize_accounts
# ---------------------------------------------------------------------------

from dashboard.data.cap_history import summarize_accounts  # noqa: E402


class TestSummarizeAccounts:
    """Tests for summarize_accounts(intervals, *, total_accounts=None).

    RED baseline: ImportError (summarize_accounts does not exist yet).
    """

    def test_empty_intervals_no_total(self):
        """(a) empty intervals, no total_accounts -> all zeros."""
        result = summarize_accounts([])
        assert result == {'total': 0, 'capped': 0, 'available': 0, 'capped_accounts': []}, (
            f'Expected all-zero dict for empty intervals, got {result}'
        )

    def test_single_open_ended_no_total(self):
        """(b) single open-ended interval, no total_accounts -> total=1, capped=1, available=0."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=2)
        intervals = [CapInterval('acc', t1, None)]
        result = summarize_accounts(intervals)
        assert result == {'total': 1, 'capped': 1, 'available': 0, 'capped_accounts': ['acc']}, (
            f'Expected 1 capped account, got {result}'
        )

    def test_single_closed_interval_no_total(self):
        """(c) single closed pair -> total=1, capped=0, available=1."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=3)
        t2 = now - timedelta(hours=1)
        intervals = [CapInterval('acc', t1, t2)]
        result = summarize_accounts(intervals)
        assert result == {'total': 1, 'capped': 0, 'available': 1, 'capped_accounts': []}, (
            f'Expected closed interval to yield capped=0, got {result}'
        )

    def test_mixed_open_and_closed(self):
        """(d) mixed: 'a' open-ended, 'b' closed -> total=2, capped=1, available=1."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=3)
        t2 = now - timedelta(hours=1)
        intervals = [
            CapInterval('a', t1, None),
            CapInterval('b', t1, t2),
        ]
        result = summarize_accounts(intervals)
        assert result == {'total': 2, 'capped': 1, 'available': 1, 'capped_accounts': ['a']}, (
            f'Expected mixed result capped=1/available=1, got {result}'
        )

    def test_account_count_denominator(self):
        """(e) total_accounts=4 with 1 open-ended interval -> total=4, available=3.

        THE key denominator case behind the bug fix:
        One account has a stale open cap while 3 healthy accounts have no recent events.
        With total_accounts=4, available=3 and capped_now should NOT be 1.
        """
        now = datetime.now(UTC)
        t1 = now - timedelta(minutes=10)
        intervals = [CapInterval('a', t1, None)]
        result = summarize_accounts(intervals, total_accounts=4)
        assert result == {'total': 4, 'capped': 1, 'available': 3, 'capped_accounts': ['a']}, (
            f'Expected total=4 (from account_count denominator), got {result}. '
            f'Bug: if total=1 then available=0 and capped_now would be wrongly 1 '
            f'(the original 3-uncapped-accounts bug).'
        )

    def test_clamp_intervals_exceed_total_accounts(self):
        """(f) 2 distinct open accounts with total_accounts=1 -> total=max(1,2)=2.

        max() guards against under-report when account_count is stale or wrong.
        """
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=2)
        intervals = [
            CapInterval('a', t1, None),
            CapInterval('b', t1, None),
        ]
        result = summarize_accounts(intervals, total_accounts=1)
        assert result['total'] == 2, (
            f'Expected total=max(1,2)=2 when intervals exceed total_accounts, got {result["total"]}'
        )
        assert result['capped'] == 2
        assert result['available'] == 0
        assert sorted(result['capped_accounts']) == ['a', 'b']

    def test_two_open_accounts_no_total(self):
        """(g) two open-ended accounts, no total_accounts -> total=2, capped=2, available=0."""
        now = datetime.now(UTC)
        t1 = now - timedelta(hours=3)
        intervals = [
            CapInterval('a', t1, None),
            CapInterval('b', t1, None),
        ]
        result = summarize_accounts(intervals)
        assert result == {'total': 2, 'capped': 2, 'available': 0, 'capped_accounts': ['a', 'b']}, (
            f'Expected all-capped result, got {result}'
        )


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
        capped: list[tuple[datetime, datetime | None]] = [
            (now - timedelta(hours=1), now - timedelta(minutes=30))
        ]
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
        """Buckets whose right edge falls inside [now-1h, now-30min) → 1.

        Intervals are half-open [start, end): the bucket at right-edge = c_end
        is NOT marked 1 (end is exclusive).
        """
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        c_start = now - timedelta(hours=1)
        c_end = now - timedelta(minutes=30)
        # 24h window, 10-min buckets. Buckets with right edge in [c_start, c_end)
        # are those at offsets: -60, -50, -40 min from now (c_end = -30min excluded).
        result = bucketise_cap_sparkline(
            [(c_start, c_end)],
            window_hours=24,
            bucket_seconds=600,
            now=now,
        )
        labels = [datetime.fromisoformat(lb) for lb in result['labels']]
        values = result['values']
        for label, value in zip(labels, values, strict=False):
            if c_start <= label < c_end:  # half-open: c_end boundary excluded
                assert value == 1, f'Expected 1 at {label}'
            else:
                assert value == 0, f'Expected 0 at {label}'

    def test_open_ended_cap_marks_all_buckets_after_start(self):
        """Open-ended cap starting 2h ago → all buckets from 2h onwards are 1.

        Buckets strictly before c_start must be 0 — asserting both directions
        guards against an over-eager open-ended check that marks the whole
        sparkline as 1.
        """
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
        for label, value in zip(labels, values, strict=False):
            if label >= c_start:
                assert value == 1, f'Expected 1 at {label} (open-ended)'
            else:
                assert value == 0, f'Expected 0 at {label} (before cap started)'

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

    def test_naive_now_does_not_raise_and_matches_utc(self):
        """naive now must not raise and must produce the same values as UTC-equivalent now."""
        now_utc = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        capped: list[tuple[datetime, datetime | None]] = [
            (now_utc - timedelta(hours=1), now_utc - timedelta(minutes=30)),
        ]
        res_aware = bucketise_cap_sparkline(
            capped, window_hours=24, bucket_seconds=600, now=now_utc
        )
        now_naive = now_utc.replace(tzinfo=None)
        # Must not raise (was: TypeError comparing naive right_edge vs tz-aware c_start)
        res_naive = bucketise_cap_sparkline(
            capped, window_hours=24, bucket_seconds=600, now=now_naive
        )
        assert len(res_naive['values']) == len(res_aware['values'])
        assert res_naive['values'] == res_aware['values']
        assert res_naive['labels'] == res_aware['labels']

    def test_non_utc_now_matches_utc_window(self):
        """Non-UTC now must produce the same values as the UTC-equivalent now."""
        now_utc = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        tz_minus8 = timezone(timedelta(hours=-8))
        now_minus8 = now_utc.astimezone(tz_minus8)
        capped: list[tuple[datetime, datetime | None]] = [
            (now_utc - timedelta(hours=2), now_utc - timedelta(hours=1)),
        ]
        res_utc = bucketise_cap_sparkline(capped, window_hours=24, bucket_seconds=600, now=now_utc)
        res_minus8 = bucketise_cap_sparkline(
            capped, window_hours=24, bucket_seconds=600, now=now_minus8
        )
        assert res_minus8['values'] == res_utc['values']
        assert res_minus8['labels'] == res_utc['labels']
