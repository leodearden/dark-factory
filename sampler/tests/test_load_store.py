"""Tests for sampler.store.LoadSampleStore — schema, insert, window, retention, vacuum."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

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
