"""Tests for sampler.store.LoadSampleStore — schema, insert, window, retention, vacuum."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Step-5 tests: schema creation, pragmas, insert, idempotent open
# ---------------------------------------------------------------------------


class TestLoadSampleStoreSchema:
    def test_creates_samples_table(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'load-samples.db'
        LoadSampleStore(db_path)

        conn = sqlite3.connect(str(db_path))
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert 'samples' in tables

    def test_samples_table_columns(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'load-samples.db'
        LoadSampleStore(db_path)

        conn = sqlite3.connect(str(db_path))
        cols = {row[1] for row in conn.execute('PRAGMA table_info(samples)').fetchall()}
        conn.close()
        assert cols == {'ts', 'metric', 'value', 'window_mean', 'window_max'}

    def test_idx_samples_metric_ts_exists(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'load-samples.db'
        LoadSampleStore(db_path)

        conn = sqlite3.connect(str(db_path))
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        conn.close()
        assert 'idx_samples_metric_ts' in indexes

    def test_wal_mode(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'load-samples.db'
        LoadSampleStore(db_path)

        conn = sqlite3.connect(str(db_path))
        mode = conn.execute('PRAGMA journal_mode').fetchone()[0]
        conn.close()
        assert mode == 'wal'

    def test_synchronous_full(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'load-samples.db'
        LoadSampleStore(db_path)

        conn = sqlite3.connect(str(db_path))
        sync = conn.execute('PRAGMA synchronous').fetchone()[0]
        conn.close()
        assert sync == 2  # FULL

    def test_insert_sample_round_trips(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'load-samples.db'
        store = LoadSampleStore(db_path)

        ts = int(time.time())
        store.insert_sample(ts, 'psi_cpu_some_avg10', 1.23)

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            'SELECT ts, metric, value, window_mean, window_max FROM samples'
        ).fetchone()
        conn.close()

        assert row[0] == ts
        assert row[1] == 'psi_cpu_some_avg10'
        assert row[2] == pytest.approx(1.23)
        assert row[3] is None   # window_mean NULL
        assert row[4] is None   # window_max NULL

    def test_insert_with_window_values(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'load-samples.db'
        store = LoadSampleStore(db_path)

        ts = int(time.time())
        store.insert_sample(ts, 'verify_concurrency', 3.0, window_mean=2.5, window_max=4.0)

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            'SELECT window_mean, window_max FROM samples'
        ).fetchone()
        conn.close()

        assert row[0] == pytest.approx(2.5)
        assert row[1] == pytest.approx(4.0)

    def test_idempotent_open_preserves_rows(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'load-samples.db'
        ts = int(time.time())

        # First open: insert a row
        store1 = LoadSampleStore(db_path)
        store1.insert_sample(ts, 'test_metric', 42.0)

        # Second open: must not raise, must see the row
        LoadSampleStore(db_path)  # reopen must not raise (return value unused)
        conn = sqlite3.connect(str(db_path))
        count = conn.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
        conn.close()
        assert count == 1

    def test_creates_parent_dirs(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'deep' / 'nested' / 'load-samples.db'
        # Must not raise even if parent dirs don't exist
        LoadSampleStore(db_path)
        assert db_path.exists()


# ---------------------------------------------------------------------------
# Step-7 tests: trailing_window
# ---------------------------------------------------------------------------


class TestTrailingWindow:
    def test_no_prior_rows_returns_current(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        mean, mx = store.trailing_window('new_metric', 5.0)
        assert mean == pytest.approx(5.0)
        assert mx == pytest.approx(5.0)

    def test_with_prior_rows(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        now = 1_000_000

        # Insert two prior values
        store.insert_sample(now - 20, 'verify_concurrency', 10.0)
        store.insert_sample(now - 10, 'verify_concurrency', 20.0)

        # trailing_window called with current value 30.0
        # -> window = [10.0, 20.0, 30.0], mean=20.0, max=30.0
        mean, mx = store.trailing_window('verify_concurrency', 30.0)
        assert mean == pytest.approx(20.0)
        assert mx == pytest.approx(30.0)

    def test_ignores_other_metrics(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        now = 1_000_000

        store.insert_sample(now - 10, 'other_metric', 999.0)

        # No prior rows for 'verify_concurrency'
        mean, mx = store.trailing_window('verify_concurrency', 5.0)
        assert mean == pytest.approx(5.0)
        assert mx == pytest.approx(5.0)

    def test_caps_at_window_minus_one_prior_rows(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        base_ts = 1_000_000

        # Insert 80 rows; only the last 59 (window=60 -> window-1=59) should be used
        for i in range(80):
            store.insert_sample(base_ts + i, 'metric_x', float(i))

        # Values 0..79 inserted; most recent 59 are values 21..79 (59 rows)
        # current_value = 100.0
        # window = [21.0, 22.0, ..., 79.0, 100.0] = 59 + 1 = 60 values
        mean, mx = store.trailing_window('metric_x', 100.0, window=60)
        expected_values = [float(i) for i in range(21, 80)] + [100.0]
        expected_mean = sum(expected_values) / len(expected_values)
        expected_max = max(expected_values)
        assert mean == pytest.approx(expected_mean, rel=1e-6)
        assert mx == pytest.approx(expected_max)


# ---------------------------------------------------------------------------
# Step-9 tests: cleanup_old, should_vacuum, maybe_vacuum
# ---------------------------------------------------------------------------


class TestRetentionAndVacuum:
    def test_cleanup_old_removes_old_rows(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        now = 1_000_000
        retain = 86400

        old_ts = now - retain - 1
        recent_ts = now - 100

        store.insert_sample(old_ts, 'metric', 1.0)
        store.insert_sample(recent_ts, 'metric', 2.0)

        store.cleanup_old(now, retain_seconds=retain)

        conn = sqlite3.connect(str(tmp_path / 'db.sqlite'))
        rows = conn.execute('SELECT ts FROM samples').fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == recent_ts

    def test_cleanup_old_keeps_all_if_none_expired(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        now = 1_000_000

        store.insert_sample(now - 100, 'metric', 1.0)
        store.insert_sample(now - 200, 'metric', 2.0)

        store.cleanup_old(now, retain_seconds=86400)

        conn = sqlite3.connect(str(tmp_path / 'db.sqlite'))
        count = conn.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
        conn.close()
        assert count == 2

    def test_should_vacuum_true_when_no_prior_vacuum(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        now = 1_000_000
        assert store.should_vacuum(now) is True

    def test_should_vacuum_false_within_interval(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        now = 1_000_000

        store.maybe_vacuum(now)
        # Call again at now + 1 (less than 86400s later) -> should be False
        assert store.should_vacuum(now + 1) is False

    def test_should_vacuum_true_after_interval(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        now = 1_000_000

        store.maybe_vacuum(now)
        assert store.should_vacuum(now + 86400) is True

    def test_maybe_vacuum_runs_and_records_timestamp(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        now = 1_000_000

        store.maybe_vacuum(now)

        # Verify last_vacuum_ts was recorded in meta table
        conn = sqlite3.connect(str(tmp_path / 'db.sqlite'))
        row = conn.execute(
            "SELECT value FROM meta WHERE key='last_vacuum_ts'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert int(row[0]) == now

    def test_maybe_vacuum_same_day_noop(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        now = 1_000_000

        store.maybe_vacuum(now)
        # Second call on same day should be a no-op (no error)
        store.maybe_vacuum(now + 3600)

        # should_vacuum still False (only 1h elapsed)
        assert store.should_vacuum(now + 3600) is False


# ---------------------------------------------------------------------------
# Task 3592 step-13: the retention window widens from 24 hours to 30 days
# ---------------------------------------------------------------------------

DAY = 86_400
THIRTY_DAYS = 30 * DAY


def _count_at(db_path: Path, ts: int) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            'SELECT COUNT(*) FROM samples WHERE ts = ?', (ts,)
        ).fetchone()[0]
    finally:
        conn.close()


class TestThirtyDayRetention:
    """Asserted by BEHAVIOUR, not by signature introspection.

    A default read off the signature would pass against a `cleanup_old` that
    ignored it; what ε1/ε2 need is that a 29-day-old sample is still in the
    corpus when the calibration runs.
    """

    def test_a_row_older_than_the_old_24h_default_now_survives(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)
        now = 10_000_000
        just_over_a_day = now - DAY - 60
        just_over_thirty_days = now - THIRTY_DAYS - 60
        store.insert_sample(just_over_a_day, 'runqueue_ratio', 1.0)
        store.insert_sample(just_over_thirty_days, 'runqueue_ratio', 2.0)

        store.cleanup_old(now)

        assert _count_at(db_path, just_over_a_day) == 1, (
            '25-hour-old rows must survive the widened window'
        )
        assert _count_at(db_path, just_over_thirty_days) == 0

    def test_explicit_override_still_prunes_to_that_window(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)
        now = 10_000_000
        two_hours_old = now - 7200
        half_an_hour_old = now - 1800
        store.insert_sample(two_hours_old, 'runqueue_ratio', 1.0)
        store.insert_sample(half_an_hour_old, 'runqueue_ratio', 2.0)

        store.cleanup_old(now, retain_seconds=3600)

        assert _count_at(db_path, two_hours_old) == 0
        assert _count_at(db_path, half_an_hour_old) == 1

    def test_the_cutoff_stays_exclusive_exactly_as_today(self, tmp_path: Path):
        """Only the NUMBER widens: a row exactly at the cutoff still survives."""
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)
        now = 10_000_000
        at_cutoff = now - THIRTY_DAYS
        one_second_older = at_cutoff - 1
        store.insert_sample(at_cutoff, 'runqueue_ratio', 1.0)
        store.insert_sample(one_second_older, 'runqueue_ratio', 2.0)

        store.cleanup_old(now)

        assert _count_at(db_path, at_cutoff) == 1
        assert _count_at(db_path, one_second_older) == 0


# ---------------------------------------------------------------------------
# Task 3592 step-15: cleanup_old is interval-gated (decision 4)
# ---------------------------------------------------------------------------


class _DeleteFailingConnection:
    """A real connection with exactly one statement broken: the DELETE.

    The stamp-after-prune invariant is precisely that the DELETE can fail while
    the meta write still succeeds, so the injection has to fail that one
    statement and nothing else. Everything else — the INSERT OR REPLACE
    ``_set_meta`` issues, commit, close — delegates to the real connection.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
        if sql.lstrip().upper().startswith('DELETE'):
            raise sqlite3.OperationalError('injected: DELETE failed')
        return self._conn.execute(sql, *args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class TestCleanupIsIntervalGated:
    """Why this gate exists, pinned so it cannot be "simplified" away.

    `DELETE FROM samples WHERE ts < ?` cannot use idx_samples_metric_ts(metric,
    ts) — a leading-column index does not serve a bare-ts predicate — so every
    call is a full table SCAN. Measured at 2.5M rows, a NO-OP cleanup (nothing
    old enough to delete) costs 106.7 ms; extrapolated to the 12.96M-row
    30-day steady state that is ~550 ms every 5 s, forever, to delete nothing.

    Two fixes were measured. Adding idx_samples_ts makes the plan an index
    SEARCH at ~0 ms but grew the probe file 33% (98 -> 130 MB, i.e. 1.62 ->
    ~2.15 GB at 30 d). The interval gate amortises the same scan to once per
    interval, where 550 ms is irrelevant, and costs zero bytes. The gate wins
    on 530 MB and on reusing the should_vacuum/last_vacuum_ts machinery next
    door; its only cost is up to one interval of over-retention, which is
    meaningless for a calibration corpus. Doing both would make the index dead
    weight, so exactly one is taken.
    """

    def test_the_delete_predicate_really_is_a_full_scan(self, tmp_path: Path):
        """The measured fact the gate is the answer to — read off the schema."""
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        LoadSampleStore(db_path)

        conn = sqlite3.connect(str(db_path))
        try:
            plan = ' '.join(
                str(row[3])
                for row in conn.execute(
                    'EXPLAIN QUERY PLAN DELETE FROM samples WHERE ts < ?', (0,)
                ).fetchall()
            )
        finally:
            conn.close()

        assert 'SCAN samples' in plan, plan
        assert 'USING INDEX' not in plan, plan

    def test_trailing_windows_query_is_index_backed_and_untouched(self, tmp_path: Path):
        """The counterpart: this one IS served by the existing index."""
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        LoadSampleStore(db_path)

        conn = sqlite3.connect(str(db_path))
        try:
            plan = ' '.join(
                str(row[3])
                for row in conn.execute(
                    'EXPLAIN QUERY PLAN SELECT value FROM samples'
                    ' WHERE metric = ? ORDER BY ts DESC LIMIT ?',
                    ('runqueue_ratio', 59),
                ).fetchall()
            )
        finally:
            conn.close()

        assert 'USING INDEX idx_samples_metric_ts' in plan, plan

    def test_virgin_store_is_due_and_running_prunes_and_stamps(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)
        now = 10_000_000
        stale = now - THIRTY_DAYS - 60
        store.insert_sample(stale, 'runqueue_ratio', 1.0)

        assert store.should_cleanup(now) is True
        store.cleanup_old(now)

        assert _count_at(db_path, stale) == 0
        assert store.should_cleanup(now) is False

    def test_a_second_call_in_the_same_interval_does_not_re_scan(self, tmp_path: Path):
        """Behavioural, not a mock call count: plant an over-age row AFTER."""
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)
        now = 10_000_000
        store.cleanup_old(now)

        planted_after = now - THIRTY_DAYS - 60
        store.insert_sample(planted_after, 'runqueue_ratio', 1.0)
        store.cleanup_old(now)

        assert _count_at(db_path, planted_after) == 1, (
            'the second call in the same interval must not have run the DELETE'
        )

    def test_one_interval_later_it_runs_again(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)
        now = 10_000_000
        store.cleanup_old(now)

        planted_after = now - THIRTY_DAYS - 60
        store.insert_sample(planted_after, 'runqueue_ratio', 1.0)
        later = now + DAY
        assert store.should_cleanup(later) is True
        store.cleanup_old(later)

        assert _count_at(db_path, planted_after) == 0

    def test_interval_override_is_honoured(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)
        now = 10_000_000
        store.cleanup_old(now)

        planted_after = now - THIRTY_DAYS - 60
        store.insert_sample(planted_after, 'runqueue_ratio', 1.0)
        store.cleanup_old(now + 60, interval_seconds=30)

        assert _count_at(db_path, planted_after) == 0
        assert store.should_cleanup(now + 60, interval_seconds=30) is False

    def test_the_stamp_is_written_only_after_a_successful_prune(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Mirrors maybe_vacuum: a transient failure must not suppress retries.

        A stamp written before the DELETE would silence cleanup for a whole
        interval on one locked-database error, and the next window would then
        be double-length.

        Reaching for the private ``_connect`` is deliberate: there is no public
        seam for "make the DELETE fail and only the DELETE", and failure
        injection is the recognised reason to reach into internals — read it as
        that, not as the tests-touch-internals smell. The obvious alternative,
        chmod(0o444) on the database file, is what this test used to do and it
        was vacuous: it breaks the meta write too, so a stamp-first
        implementation satisfies every assertion below.
        """
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)
        now = 10_000_000
        stale = now - THIRTY_DAYS - 60
        store.insert_sample(stale, 'runqueue_ratio', 1.0)

        # Bind the real method BEFORE patching — calling store._connect() from
        # inside the replacement would re-enter the replacement itself.
        real_connect = store._connect
        monkeypatch.setattr(
            store, '_connect', lambda: _DeleteFailingConnection(real_connect())
        )
        with pytest.raises(sqlite3.Error):
            store.cleanup_old(now)
        monkeypatch.undo()

        assert store._get_meta('last_cleanup_ts') is None, (
            'a failed prune must leave the clock unstamped'
        )
        assert store.should_cleanup(now) is True, (
            'a failed prune must leave the store still due, not stamped'
        )
        store.cleanup_old(now)
        assert _count_at(db_path, stale) == 0

    def test_the_gate_reuses_the_vacuum_machinery_not_a_new_mechanism(
        self, tmp_path: Path
    ):
        """Two meta keys side by side, and the two gates stay independent."""
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)
        now = 10_000_000

        store.cleanup_old(now)

        conn = sqlite3.connect(str(db_path))
        try:
            keys = {row[0] for row in conn.execute('SELECT key FROM meta').fetchall()}
        finally:
            conn.close()
        assert 'last_cleanup_ts' in keys
        assert 'last_vacuum_ts' not in keys, (
            'cleanup must not stamp the vacuum clock — the two gates are separate'
        )
        assert store.should_vacuum(now) is True
