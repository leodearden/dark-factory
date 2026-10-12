"""Tests for sampler.store.LoadSampleStore — schema, insert, window, retention, vacuum."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from sampler.metrics import SampleKey

RUNQ = SampleKey('runqueue_ratio')
VERIFY = SampleKey('verify_concurrency')
PSI_CPU = SampleKey('psi_cpu_some_avg10')

# ---------------------------------------------------------------------------
# The store's one connection seam, observed
# ---------------------------------------------------------------------------


def _no_statement(_sql: str) -> bool:
    return False


def _statements_starting(prefix: str) -> Callable[[str], bool]:
    return lambda sql: sql.lstrip().upper().startswith(prefix)


def _is_vacuum(sql: str) -> bool:
    return sql.strip().upper().startswith('VACUUM')


class _CountingConnection:
    """A real connection that records its statements and commits, and can fail some.

    A proxy and not a patched method: ``sqlite3.Connection`` is a C type, so
    its instances accept no attribute assignment and ``commit`` cannot be
    wrapped in place. A statement the spy's ``fails`` predicate matches raises
    ``sqlite3.OperationalError`` instead of running; everything else — commit,
    rollback, close — delegates to the real connection.
    """

    def __init__(self, conn: sqlite3.Connection, spy: ConnectSpy, statements: list[str]):
        self._conn = conn
        self._spy = spy
        self._statements = statements

    def _admit(self, sql: str) -> None:
        self._statements.append(sql)
        self._spy.executed.append(sql)
        if self._spy.fails(sql):
            raise sqlite3.OperationalError(f'injected: {sql.split()[0]} failed')

    def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
        self._admit(sql)
        return self._conn.execute(sql, *args)

    def executemany(self, sql: str, *args: Any) -> sqlite3.Cursor:
        self._admit(sql)
        return self._conn.executemany(sql, *args)

    def commit(self) -> None:
        self._spy.commits.append('commit')
        self._conn.commit()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class _Opened(NamedTuple):
    autocommit: bool
    statements: list[str]


class ConnectSpy:
    """An injectable ``connect`` that hands out ``connect_durably``'s own connections.

    The pragma policy under test is therefore the production one. What the spy
    adds is a record: one ``opened`` entry per connection (its autocommit flag
    and the statements run on it), every statement in ``executed``, every
    commit in ``commits``. ``fails`` names statements to break, and can be
    swapped between calls to arm or disarm the failure.
    """

    def __init__(self, *, fails: Callable[[str], bool] = _no_statement) -> None:
        self.fails = fails
        self.opened: list[_Opened] = []
        self.executed: list[str] = []
        self.commits: list[str] = []

    def __call__(self, db_path: Path, *, autocommit: bool = False) -> Any:
        from sampler.store import connect_durably

        opened = _Opened(autocommit, [])
        self.opened.append(opened)
        return _CountingConnection(
            connect_durably(db_path, autocommit=autocommit), self, opened.statements
        )

    def clear(self) -> None:
        self.opened.clear()
        self.executed.clear()
        self.commits.clear()

    def vacuums_ran_on(self) -> list[bool]:
        """The autocommit flag of each connection a VACUUM was issued on."""
        return [o.autocommit for o in self.opened if any(map(_is_vacuum, o.statements))]


def _meta(db_path: Path, key: str) -> str | None:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute('SELECT value FROM meta WHERE key = ?', (key,)).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Schema shape, and the pre-leaf file the migration starts from
# ---------------------------------------------------------------------------

# The store's DDL before task 6772, verbatim: a historical fixture, not a copy
# of anything live, so it must never be "kept in sync".
_LEGACY_SCHEMA = """\
CREATE TABLE IF NOT EXISTS samples (
    ts          INTEGER NOT NULL,
    metric      TEXT    NOT NULL,
    value       REAL    NOT NULL,
    window_mean REAL,
    window_max  REAL
);

CREATE INDEX IF NOT EXISTS idx_samples_metric_ts
    ON samples (metric, ts);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

_LegacyRow = tuple[int, str, float, float | None, float | None]

_LEAF_INDEXES = ['idx_samples_metric_leaf_ts', 'idx_samples_metric_ts_leaf']


def _legacy_store(db_path: Path, rows: list[_LegacyRow]) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_LEGACY_SCHEMA)
        conn.executemany(
            'INSERT INTO samples (ts, metric, value, window_mean, window_max)'
            ' VALUES (?, ?, ?, ?, ?)',
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def _shape(db_path: Path) -> tuple[list[tuple], list[str]]:
    """``samples``' columns as PRAGMA table_info reports them, and its index names."""
    conn = sqlite3.connect(str(db_path))
    try:
        columns = conn.execute('PRAGMA table_info(samples)').fetchall()
        indexes = sorted(row[1] for row in conn.execute('PRAGMA index_list(samples)'))
    finally:
        conn.close()
    return columns, indexes


def _all_rows(db_path: Path) -> list[tuple]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute('SELECT * FROM samples ORDER BY rowid').fetchall()
    finally:
        conn.close()


def _seed(db_path: Path, key: SampleKey, timestamps: range) -> None:
    """One row of *key* per timestamp, written in one transaction."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executemany(
            'INSERT INTO samples (ts, metric, leaf, value) VALUES (?, ?, ?, ?)',
            [(ts, key.metric, key.leaf, 1.0) for ts in timestamps],
        )
        conn.commit()
    finally:
        conn.close()


def _window_read_cost(db_path: Path, key: SampleKey) -> int:
    """SQLite VM progress callbacks that one ``trailing_window`` read of *key* runs.

    Counted on the connections the store opens through its ``connect`` seam,
    from the read alone: construction's schema statements are not counted.
    """
    from sampler.store import LoadSampleStore, connect_durably

    steps = 0

    def step() -> int:
        nonlocal steps
        steps += 1
        return 0

    def counting(db_path: Path, *, autocommit: bool = False) -> sqlite3.Connection:
        conn = connect_durably(db_path, autocommit=autocommit)
        conn.set_progress_handler(step, 1)
        return conn

    store = LoadSampleStore(db_path, connect=counting)
    steps = 0
    store.trailing_window(key, 1.0)
    return steps


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
        """``leaf`` is LAST: ADD COLUMN appends, so a migrated file matches a fresh one."""
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'load-samples.db'
        LoadSampleStore(db_path)

        columns, _indexes = _shape(db_path)
        assert [c[1] for c in columns] == [
            'ts', 'metric', 'value', 'window_mean', 'window_max', 'leaf'
        ]

    def test_the_indexes_are_metric_ts_leaf_and_a_per_leaf_metric_leaf_ts(
        self, tmp_path: Path
    ):
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'load-samples.db'
        LoadSampleStore(db_path)

        _columns, indexes = _shape(db_path)
        assert indexes == _LEAF_INDEXES
        conn = sqlite3.connect(str(db_path))
        try:
            indexed = {
                name: [row[2] for row in conn.execute(f"PRAGMA index_info('{name}')")]
                for name in indexes
            }
            partial = {
                row[1]: bool(row[4]) for row in conn.execute('PRAGMA index_list(samples)')
            }
        finally:
            conn.close()
        assert indexed == {
            'idx_samples_metric_leaf_ts': ['metric', 'leaf', 'ts'],
            'idx_samples_metric_ts_leaf': ['metric', 'ts', 'leaf'],
        }
        assert partial == {
            'idx_samples_metric_leaf_ts': True,
            'idx_samples_metric_ts_leaf': False,
        }

    def test_SCHEMA_on_a_blank_file_has_the_shape_the_store_creates(self, tmp_path: Path):
        """Every consumer fixture builds from ``SCHEMA``, so it must BE the store's DDL."""
        from sampler.store import SCHEMA, LoadSampleStore

        from_schema = tmp_path / 'from-schema.db'
        conn = sqlite3.connect(str(from_schema))
        try:
            conn.executescript(SCHEMA)
        finally:
            conn.close()
        from_store = tmp_path / 'from-store.db'
        LoadSampleStore(from_store)

        assert _shape(from_schema) == _shape(from_store)

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
        store.insert_sample(ts, PSI_CPU, 1.23)

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
        store.insert_sample(ts, VERIFY, 3.0, window_mean=2.5, window_max=4.0)

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
        store1.insert_sample(ts, SampleKey('test_metric'), 42.0)

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
# Task 6772: a pre-leaf file is migrated in place, whole or not at all
# ---------------------------------------------------------------------------

_LEGACY_ROWS: list[_LegacyRow] = [
    (1_000_000, 'own_cpu_some10:a', 1.5, 1.0, 2.0),
    (1_000_000, 'own_read_ok:a', 1.0, None, None),
    (1_000_000, 'own_read_ok:weird:leaf.slice', 1.0, None, None),
    (1_000_000, 'psi_cpu_some_avg10', 2.0, None, None),
    (1_000_000, 'runqueue_ratio', 3.0, None, None),
]


class TestLeafMigration:
    def test_a_legacy_file_is_migrated_in_place(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        legacy = tmp_path / 'legacy.db'
        _legacy_store(legacy, _LEGACY_ROWS)
        fresh = tmp_path / 'fresh.db'
        LoadSampleStore(fresh)

        LoadSampleStore(legacy)

        conn = sqlite3.connect(str(legacy))
        try:
            rows = conn.execute(
                'SELECT metric, leaf, value, window_mean, window_max'
                ' FROM samples ORDER BY rowid'
            ).fetchall()
            still_encoded = conn.execute(
                "SELECT COUNT(*) FROM samples WHERE instr(metric, ':') > 0"
            ).fetchone()[0]
        finally:
            conn.close()
        assert rows == [
            ('own_cpu_some10', 'a', 1.5, 1.0, 2.0),
            ('own_read_ok', 'a', 1.0, None, None),
            # Only the FIRST ':' splits: the leaf is everything after it.
            ('own_read_ok', 'weird:leaf.slice', 1.0, None, None),
            ('psi_cpu_some_avg10', None, 2.0, None, None),
            ('runqueue_ratio', None, 3.0, None, None),
        ]
        assert still_encoded == 0
        assert _shape(legacy) == _shape(fresh), (
            'a migrated file must be indistinguishable in shape from a fresh one, '
            'idx_samples_metric_ts dropped and both leaf indexes built'
        )

    def test_reopening_is_a_no_op_in_every_starting_state(self, tmp_path: Path):
        from sampler.store import SCHEMA, LoadSampleStore

        fresh = tmp_path / 'fresh.db'
        LoadSampleStore(fresh)
        legacy = tmp_path / 'legacy.db'
        _legacy_store(legacy, _LEGACY_ROWS)
        LoadSampleStore(legacy)
        built_from_schema = tmp_path / 'built-from-schema.db'
        conn = sqlite3.connect(str(built_from_schema))
        try:
            conn.executescript(SCHEMA)
            conn.execute(
                'INSERT INTO samples (ts, metric, leaf, value) VALUES (?, ?, ?, ?)',
                (1_000_000, 'own_cpu_some10', 'a', 1.5),
            )
            conn.commit()
        finally:
            conn.close()
        LoadSampleStore(built_from_schema)

        files = (fresh, legacy, built_from_schema)
        shapes = {path.name: _shape(path) for path in files}
        for path in files:
            before = _all_rows(path)
            spy = ConnectSpy()

            LoadSampleStore(path, connect=spy)

            assert _all_rows(path) == before, f'reopening {path.name} changed its rows'
            assert _shape(path) == shapes[path.name]
            rewrites = [
                sql
                for sql in spy.executed
                if sql.lstrip().upper().startswith(('ALTER', 'UPDATE', 'DROP INDEX'))
            ]
            assert rewrites == [], f'reopening {path.name} rewrote it again: {rewrites}'
        assert len({repr(shape) for shape in shapes.values()}) == 1, shapes

    def test_a_migration_that_fails_midway_leaves_the_file_untouched(self, tmp_path: Path):
        """The ADD COLUMN, the split, and the index swap commit together or not at all."""
        from sampler.store import LoadSampleStore

        legacy = tmp_path / 'legacy.db'
        _legacy_store(legacy, _LEGACY_ROWS)
        legacy_shape = _shape(legacy)

        with pytest.raises(sqlite3.Error):
            LoadSampleStore(legacy, connect=ConnectSpy(fails=_statements_starting('UPDATE')))

        columns, indexes = _shape(legacy)
        assert 'leaf' not in [c[1] for c in columns], 'the ADD COLUMN outlived the failure'
        assert indexes == ['idx_samples_metric_ts']
        assert (columns, indexes) == legacy_shape
        assert _all_rows(legacy) == _LEGACY_ROWS, 'rows were rewritten by a failed migration'

        LoadSampleStore(legacy)

        columns, indexes = _shape(legacy)
        assert columns[-1][1] == 'leaf', 'the next construction did not retry the migration'
        assert indexes == _LEAF_INDEXES


# ---------------------------------------------------------------------------
# Step-7 tests: trailing_window
# ---------------------------------------------------------------------------


class TestTrailingWindow:
    def test_no_prior_rows_returns_current(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        mean, mx = store.trailing_window(SampleKey('new_metric'), 5.0)
        assert mean == pytest.approx(5.0)
        assert mx == pytest.approx(5.0)

    def test_with_prior_rows(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        now = 1_000_000

        # Insert two prior values
        store.insert_sample(now - 20, VERIFY, 10.0)
        store.insert_sample(now - 10, VERIFY, 20.0)

        # trailing_window called with current value 30.0
        # -> window = [10.0, 20.0, 30.0], mean=20.0, max=30.0
        mean, mx = store.trailing_window(VERIFY, 30.0)
        assert mean == pytest.approx(20.0)
        assert mx == pytest.approx(30.0)

    def test_ignores_other_metrics(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        now = 1_000_000

        store.insert_sample(now - 10, SampleKey('other_metric'), 999.0)

        # No prior rows for 'verify_concurrency'
        mean, mx = store.trailing_window(VERIFY, 5.0)
        assert mean == pytest.approx(5.0)
        assert mx == pytest.approx(5.0)

    def test_caps_at_window_minus_one_prior_rows(self, tmp_path: Path):
        """The span is left to the DEFAULT on purpose.

        ``_TRAILING_WINDOW_SAMPLES`` is the single home for the 60-sample span
        and every other value-sensitive test here passes ``window=``
        explicitly or seeds at most two prior rows — so the default was free
        to change with nothing going red, silently reshaping every
        ``window_mean``/``window_max`` in the calibration corpus. This
        fixture's 80 rows discriminate 60 from any other span.
        """
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        base_ts = 1_000_000

        # Insert 80 rows; only the last 59 (window=60 -> window-1=59) should be used
        for i in range(80):
            store.insert_sample(base_ts + i, SampleKey('metric_x'), float(i))

        # Values 0..79 inserted; most recent 59 are values 21..79 (59 rows)
        # current_value = 100.0
        # window = [21.0, 22.0, ..., 79.0, 100.0] = 59 + 1 = 60 values
        mean, mx = store.trailing_window(SampleKey('metric_x'), 100.0)
        expected_values = [float(i) for i in range(21, 80)] + [100.0]
        expected_mean = sum(expected_values) / len(expected_values)
        expected_max = max(expected_values)
        assert mean == pytest.approx(expected_mean, rel=1e-6)
        assert mx == pytest.approx(expected_max)

    def test_each_leafs_trailing_window_spans_only_its_own_rows(self, tmp_path: Path):
        """Two leaves of one metric, and a host-level row beside them, never pool."""
        from sampler.store import LoadSampleStore

        leaf_a = SampleKey('own_cpu_some10', 'a')
        leaf_b = SampleKey('own_cpu_some10', 'b')
        store = LoadSampleStore(tmp_path / 'db.sqlite')

        def tick(i: int) -> None:
            store.write_tick(
                1_000_000 + i * 5,
                unwindowed={},
                windowed={leaf_a: float(i), leaf_b: 1000.0 + i, RUNQ: 500.0},
            )

        for i in range(69):
            tick(i)
        expected_a = store.trailing_window(leaf_a, 69.0)
        tick(69)

        conn = sqlite3.connect(str(store.db_path))
        try:
            windows = {
                leaf: (mean, mx)
                for leaf, mean, mx in conn.execute(
                    'SELECT leaf, window_mean, window_max FROM samples'
                    ' WHERE ts = ? AND metric = ?',
                    (1_000_000 + 69 * 5, 'own_cpu_some10'),
                )
            }
        finally:
            conn.close()
        own_a = [float(i) for i in range(10, 70)]
        own_b = [1000.0 + i for i in range(10, 70)]
        assert windows['a'] == (pytest.approx(sum(own_a) / 60), max(own_a))
        assert windows['b'] == (pytest.approx(sum(own_b) / 60), max(own_b))
        assert windows['a'] == expected_a

    @pytest.mark.parametrize(
        ('sparse', 'busy', 'sparse_history'),
        [
            (SampleKey('own_cpu_some10', 'new.service'), SampleKey('own_cpu_some10', 'busy.service'), range(0)),
            (SampleKey('own_cpu_some10', 'back.service'), SampleKey('own_cpu_some10', 'busy.service'), range(3)),
            (RUNQ, VERIFY, range(3)),
        ],
        ids=['a-leaf-with-no-rows', 'a-leaf-with-fewer-rows-than-a-window', 'a-host-level-metric'],
    )
    def test_a_sparse_keys_window_read_does_not_walk_a_busy_neighbours_history(
        self, tmp_path: Path, sparse: SampleKey, busy: SampleKey, sparse_history: range
    ):
        """A key new to the corpus, or back after a gap, pays for its own rows only.

        The busy neighbour's rows are all NEWER than the sparse key's, so a read
        that walks the metric newest-first and filters by leaf would cross
        every one of a sibling leaf's rows before giving up; a read that seeks
        its own (metric, leaf) does not. Ten times the neighbour's history must
        therefore cost the same.
        """
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        LoadSampleStore(db_path)
        _seed(db_path, sparse, sparse_history)
        _seed(db_path, busy, range(1_000, 2_000))
        shallow = _window_read_cost(db_path, sparse)

        _seed(db_path, busy, range(2_000, 12_000))

        assert _window_read_cost(db_path, sparse) == shallow


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

        store.insert_sample(old_ts, SampleKey('metric'), 1.0)
        store.insert_sample(recent_ts, SampleKey('metric'), 2.0)

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

        store.insert_sample(now - 100, SampleKey('metric'), 1.0)
        store.insert_sample(now - 200, SampleKey('metric'), 2.0)

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
        """The stamp is recorded on the path where a VACUUM actually RUNS.

        The store must carry reclaimable pages or
        ``_VACUUM_MIN_RECLAIMABLE_FRACTION`` short-circuits maybe_vacuum, and
        this then stamps via the SKIP path — which
        ``test_the_clock_is_still_stamped_so_the_check_is_not_per_tick``
        already covers, leaving this test asserting nothing of its own.  The
        contract it uniquely pins is the retry rule: the clock is stamped
        AFTER a successful VACUUM, not before it.
        """
        from sampler.store import LoadSampleStore

        spy = ConnectSpy()
        store = LoadSampleStore(tmp_path / 'db.sqlite', connect=spy)
        _bulk_insert(store.db_path, 'psi_cpu_some_avg10', 1_000_000, 20_000)
        now = 1_000_000 + 20_000 * 5
        store.cleanup_old(now, retain_seconds=100)
        assert store.reclaimable_fraction() >= 0.10, (
            'setup must open the reclaimable-fraction gate, or this test is vacuous'
        )

        store.maybe_vacuum(now)

        # The stamp alone cannot distinguish this path from the skip path --
        # both write the same `now` -- so pin that a VACUUM really ran.
        assert [sql for sql in spy.executed if _is_vacuum(sql)], (
            'stamped without running a VACUUM: the retry rule is not being honoured'
        )
        assert spy.vacuums_ran_on() == [True], (
            'SQLite refuses VACUUM inside a transaction, so it must run on a '
            'connection the seam opened with autocommit=True'
        )
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
        store.insert_sample(just_over_a_day, RUNQ, 1.0)
        store.insert_sample(just_over_thirty_days, RUNQ, 2.0)

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
        store.insert_sample(two_hours_old, RUNQ, 1.0)
        store.insert_sample(half_an_hour_old, RUNQ, 2.0)

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
        store.insert_sample(at_cutoff, RUNQ, 1.0)
        store.insert_sample(one_second_older, RUNQ, 2.0)

        store.cleanup_old(now)

        assert _count_at(db_path, at_cutoff) == 1
        assert _count_at(db_path, one_second_older) == 0

    def test_a_far_future_row_is_pruned_rather_than_outliving_the_corpus(
        self, tmp_path: Path
    ):
        """A past-only cutoff can never reach a clock-skew row.

        `ts` is stamped `int(time.time())` with no monotonicity guard, so an
        NTP step forward, a VM suspend/resume or a hand-seeded probe row lands
        a sample beyond every real tick.  Against a past-only cutoff that row
        is immortal — at 30-day retention it outlives the entire corpus — and
        it is not inert: every MAX(ts)-anchored consumer is dragged forward
        with it (the dashboard's recency bound served placeholders for all
        nine metrics off exactly one such row).
        """
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)
        now = 10_000_000
        healthy = now - 60
        a_year_ahead = now + 365 * DAY
        store.insert_sample(healthy, RUNQ, 1.0)
        store.insert_sample(a_year_ahead, RUNQ, 2.0)

        store.cleanup_old(now)

        assert _count_at(db_path, a_year_ahead) == 0, (
            'a future-dated row survived the sweep and will outlive the corpus'
        )
        assert _count_at(db_path, healthy) == 1

    def test_a_backwards_clock_does_not_delete_a_healthy_corpus(self, tmp_path: Path):
        """Why the future cutoff is `retain_seconds` and not a tight tolerance.

        The skew cuts both ways: a host whose NTP has not synced at boot reads
        `now` in the PAST, and every real row then looks future-dated.  A tight
        future tolerance would delete the whole corpus for a clock that is
        merely hours out.  At ±30 days only a clock wrong by more than a month
        loses data, and such a host has no usable corpus either way.
        """
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)
        real_now = 10_000_000
        clock_reads = real_now - 3 * DAY  # NTP three days behind
        store.insert_sample(real_now, RUNQ, 1.0)
        store.insert_sample(real_now - 60, RUNQ, 2.0)

        store.cleanup_old(clock_reads)

        assert _count_at(db_path, real_now) == 1, (
            'a three-day-behind clock deleted rows a healthy sampler had just written'
        )
        assert _count_at(db_path, real_now - 60) == 1

    def test_the_future_cutoff_is_exclusive_too(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)
        now = 10_000_000
        at_cutoff = now + THIRTY_DAYS
        one_second_further = at_cutoff + 1
        store.insert_sample(at_cutoff, RUNQ, 1.0)
        store.insert_sample(one_second_further, RUNQ, 2.0)

        store.cleanup_old(now)

        assert _count_at(db_path, at_cutoff) == 1
        assert _count_at(db_path, one_second_further) == 0


# ---------------------------------------------------------------------------
# Task 3592 step-15: cleanup_old is interval-gated (decision 4)
# ---------------------------------------------------------------------------


class TestCleanupIsIntervalGated:
    """Why this gate exists, pinned so it cannot be "simplified" away.

    The retention DELETE cannot use either index — both lead with metric, and a
    leading-column index does not serve a bare-ts predicate — so every call is
    a full table SCAN. Measured at 2.5M rows, a NO-OP cleanup (nothing
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

    def test_the_cleanup_delete_is_still_a_full_scan(self, tmp_path: Path):
        """The measured fact the gate is the answer to — read off the schema."""
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        LoadSampleStore(db_path)

        conn = sqlite3.connect(str(db_path))
        try:
            plan = ' '.join(
                str(row[3])
                for row in conn.execute(
                    'EXPLAIN QUERY PLAN DELETE FROM samples WHERE ts < ? OR ts > ?',
                    (0, 0),
                ).fetchall()
            )
        finally:
            conn.close()

        assert 'SCAN samples' in plan, plan
        assert 'USING INDEX' not in plan, plan

    def test_virgin_store_is_due_and_running_prunes_and_stamps(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)
        now = 10_000_000
        stale = now - THIRTY_DAYS - 60
        store.insert_sample(stale, RUNQ, 1.0)

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
        store.insert_sample(planted_after, RUNQ, 1.0)
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
        store.insert_sample(planted_after, RUNQ, 1.0)
        later = now + DAY
        assert store.should_cleanup(later) is True
        store.cleanup_old(later)

        assert _count_at(db_path, planted_after) == 0

    def test_a_stamp_from_the_future_does_not_disable_the_sweep(self, tmp_path: Path):
        """The clock steps forward, one tick stamps it, and the skew must heal.

        A plain ``elapsed >= interval`` gate reads a future stamp as "not due"
        for the whole skew, which is the one case where the suppression is
        self-sealing: the rows preserved are exactly the future-dated ones
        ``cleanup_old``'s symmetric cutoff exists to prune, so the corpus
        cannot recover from the condition that disabled its recovery. Both
        gates are checked here because ``maybe_vacuum`` is suppressed by the
        same arithmetic on its own key, and a 30-day corpus that stops being
        pruned is also one that stops being compacted.
        """
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)
        now = 10_000_000
        a_year = 365 * DAY

        # ONE tick during the skew, through the public path that stamps both
        # clocks — not a hand-planted meta row, so what is under test is the
        # state a real forward step leaves behind.
        store.cleanup_old(now + a_year)
        store.maybe_vacuum(now + a_year)

        assert store.should_cleanup(now) is True
        assert store.should_vacuum(now) is True

        stale = now - THIRTY_DAYS - 60
        from_the_future = now + THIRTY_DAYS + 60
        store.insert_sample(stale, RUNQ, 1.0)
        store.insert_sample(from_the_future, RUNQ, 2.0)
        store.cleanup_old(now)

        assert _count_at(db_path, stale) == 0
        assert _count_at(db_path, from_the_future) == 0, (
            'the future-dated row survived, so the corpus is still anchored '
            'forward with no sweep that can reach it'
        )

    def test_interval_override_is_honoured(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)
        now = 10_000_000
        store.cleanup_old(now)

        planted_after = now - THIRTY_DAYS - 60
        store.insert_sample(planted_after, RUNQ, 1.0)
        store.cleanup_old(now + 60, interval_seconds=30)

        assert _count_at(db_path, planted_after) == 0
        assert store.should_cleanup(now + 60, interval_seconds=30) is False

    def test_the_stamp_is_written_only_after_a_successful_prune(self, tmp_path: Path):
        """Mirrors maybe_vacuum: a transient failure must not suppress retries.

        A stamp written before the DELETE would silence cleanup for a whole
        interval on one locked-database error, and the next window would then
        be double-length.

        Only the DELETE is broken, through the injected connect: the stamp-
        after-prune invariant is precisely that the DELETE can fail while the
        meta write still succeeds. chmod(0o444) on the file, what this test
        used to do, was vacuous — it breaks the meta write too, so a
        stamp-first implementation satisfied every assertion below.
        """
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        spy = ConnectSpy()
        store = LoadSampleStore(db_path, connect=spy)
        now = 10_000_000
        stale = now - THIRTY_DAYS - 60
        store.insert_sample(stale, RUNQ, 1.0)

        spy.fails = _statements_starting('DELETE')
        with pytest.raises(sqlite3.Error):
            store.cleanup_old(now)
        spy.fails = _no_statement

        assert _meta(db_path, 'last_cleanup_ts') is None, (
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


# ---------------------------------------------------------------------------
# Review suggestion 1: the daily VACUUM was sized for 24h retention, not 30d
# ---------------------------------------------------------------------------


def _bulk_insert(db_path: Path, metric: str, first_ts: int, count: int, step: int = 5) -> None:
    """Seed rows straight through sqlite3 — insert_sample commits per row."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executemany(
            'INSERT INTO samples (ts, metric, value) VALUES (?, ?, ?)',
            [(first_ts + i * step, metric, float(i % 97)) for i in range(count)],
        )
        conn.commit()
    finally:
        conn.close()


class TestVacuumIsGatedOnThereBeingSomethingToReclaim:
    """VACUUM rewrites the WHOLE file, so its cost tracks the 30-day size.

    MEASURED on this host against a probe carrying the real schema and the real
    25-metrics-per-5s-tick vocabulary, built out to the 30-day steady state
    (12,960,000 rows, 1.16 GB after packing):

      VACUUM, no free pages ............................... 15.9 s
      VACUUM after one steady-state day ................... 15.5 s
      ...and it reclaimed 2 MB of 1.160 GB ................ 0.17%

    The reason it reclaims so little is the finding that decides the design:
    free pages ARE fully reused. One steady-state day — prune the oldest day,
    write a new one — left freelist_count at 9232 immediately after the DELETE
    and at 0 after the day's inserts, with page_count up 0.25%. So at steady
    state the file does not bloat, and the unconditional daily VACUUM was
    buying ~0.17% of space for ~15 s.

    Fifteen seconds matters because the systemd unit is Type=oneshot: the whole
    VACUUM is one tick, so the corpus ε1/ε2 calibrate against would get a daily
    ~15 s hole. And a VACUUM needs ~as much free space again for its temp copy,
    so on a tight filesystem it raises, is swallowed by the existing
    `except Exception`, and the file is never compacted at all.

    Gating on free pages keeps the compaction where compaction is actually
    wanted — after a retention change or a post-outage bulk prune — and skips
    it in the steady state where it accomplishes nothing.
    """

    def test_reclaimable_fraction_is_zero_with_no_free_pages(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        _bulk_insert(store.db_path, 'psi_cpu_some_avg10', 1_000_000, 5000)

        assert store.reclaimable_fraction() == 0.0

    def test_reclaimable_fraction_is_high_after_a_bulk_prune(self, tmp_path: Path):
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        _bulk_insert(store.db_path, 'psi_cpu_some_avg10', 1_000_000, 20_000)
        now = 1_000_000 + 20_000 * 5
        store.cleanup_old(now, retain_seconds=100)

        assert store.reclaimable_fraction() > 0.10

    def test_a_steady_state_store_runs_no_vacuum(self, tmp_path: Path):
        """The 15 s daily hole this closes. Nothing to reclaim -> do not rewrite."""
        from sampler.store import LoadSampleStore

        spy = ConnectSpy()
        store = LoadSampleStore(tmp_path / 'db.sqlite', connect=spy)
        _bulk_insert(store.db_path, 'psi_cpu_some_avg10', 1_000_000, 5000)

        store.maybe_vacuum(1_000_000)

        assert spy.opened, 'the connect spy never fired, so the absence below pins nothing'
        assert not [sql for sql in spy.executed if _is_vacuum(sql)], (
            'VACUUM rewrote a file with no free pages'
        )

    def test_the_clock_is_still_stamped_so_the_check_is_not_per_tick(
        self, tmp_path: Path
    ):
        """A deliberate skip is a successful evaluation, not a transient failure.

        Stamping keeps the pragma read to once per interval rather than once
        per 5 s tick, and it is what lets the four pre-existing interval tests
        above go on describing maybe_vacuum's contract unchanged.
        """
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        _bulk_insert(store.db_path, 'psi_cpu_some_avg10', 1_000_000, 5000)

        store.maybe_vacuum(1_000_000)

        assert store.should_vacuum(1_000_000 + 1) is False
        assert store.should_vacuum(1_000_000 + 86400) is True

    def test_a_bulk_prune_is_still_compacted(self, tmp_path: Path):
        """The case the VACUUM exists for must keep working."""
        from sampler.store import LoadSampleStore

        spy = ConnectSpy()
        store = LoadSampleStore(tmp_path / 'db.sqlite', connect=spy)
        _bulk_insert(store.db_path, 'psi_cpu_some_avg10', 1_000_000, 20_000)
        now = 1_000_000 + 20_000 * 5
        store.cleanup_old(now, retain_seconds=100)
        size_before = store.db_path.stat().st_size

        store.maybe_vacuum(now)

        assert [sql for sql in spy.executed if _is_vacuum(sql)], (
            'a file that is mostly free pages was left uncompacted'
        )
        assert spy.vacuums_ran_on() == [True]
        assert store.db_path.stat().st_size < size_before


# ---------------------------------------------------------------------------
# Review suggestion 1: one tick is one connection and one transaction
# ---------------------------------------------------------------------------


class TestWriteTickIsOneConnectionAndOneTransaction:
    """The per-row write path did not survive the 9 -> 25 metric widening.

    MEASURED in this worktree against warmed stores on ext4 (200 ticks of
    history), the two paths INTERLEAVED in one process so host noise hits both,
    40 ticks each:

      per-row writes, the 25-metric tick ......... 309.6 ms median (max 592.8)
      the same 25 rows, one connection, one txn ..  13.2 ms median (max  34.6)

    Every row took its own connection, its own five durability pragmas and its
    own ``synchronous=FULL`` commit, and each non-PSI metric took a SECOND
    connection for its trailing window: ~44 connections and ~25 fsyncs per
    tick. At the paired timer's 5 s cadence that is 6.2% duty forever, on a
    host that also runs seven orchestrators — the same order of magnitude as
    the ~550 ms/tick cleanup scan this change goes to considerable lengths to
    gate away. Batched it is 0.26%.

    Atomicity comes with it, and it is not incidental: the corpus ε1/ε2
    calibrate against is read tick-by-tick, and a crash between the PSI rows
    and the windowed rows used to leave a half-written tick in it.
    """

    def _tick(self, store, now: int, leaves: int = 7) -> None:
        store.write_tick(
            now,
            unwindowed={PSI_CPU: 1.0, SampleKey('psi_mem_full_avg10'): 2.0},
            windowed={
                VERIFY: 3.0,
                **{SampleKey('own_cpu_some10', f'leaf{i}'): float(i) for i in range(leaves)},
            },
        )

    def _spied_store(self, tmp_path: Path):
        """A store whose connections are counted from the first tick on.

        Counting CONNECTIONS and COMMITS rather than timing the tick is
        deliberate: a wall-clock assertion would be flaky on a loaded host
        while measuring the same thing indirectly. The two counts are kept
        apart because they pin different properties — one connection is the
        pragma cost, one commit is the fsync cost AND the atomicity — and a
        per-row ``execute(); commit()`` loop on one connection satisfies the
        first while destroying the second.
        """
        from sampler.store import LoadSampleStore

        spy = ConnectSpy()
        store = LoadSampleStore(tmp_path / 'db.sqlite', connect=spy)
        spy.clear()
        return store, spy

    def test_a_tick_opens_one_connection(self, tmp_path: Path):
        store, spy = self._spied_store(tmp_path)

        self._tick(store, 1_000_000)

        assert len(spy.opened) == 1, (
            f'one tick opened {len(spy.opened)} connections; the point of '
            'write_tick is that it opens exactly one'
        )

    def test_the_cost_of_a_tick_does_not_scale_with_its_metric_count(self, tmp_path: Path):
        """The property the measurement above is really about.

        Pinning the ABSOLUTE connection count would go red on any unrelated
        retention change; pinning that the count is INDEPENDENT of how many
        metrics the tick carries is the thing that stops the regression, and it
        is what a per-row loop can never satisfy — the cgroup leaf count is
        discovered per tick and is not bounded by anything here.
        """
        store, spy = self._spied_store(tmp_path)

        self._tick(store, 1_000_000, leaves=1)
        few = len(spy.opened)

        spy.clear()
        self._tick(store, 1_000_005, leaves=100)
        many = len(spy.opened)

        # The floor its sibling in test_load_sampler.py already carries: an
        # equality between two counts holds trivially at 0 == 0 if the spy ever
        # detaches from the seam, and would then pin nothing at all.
        assert few >= 1, 'the connect spy never fired'
        assert few == many, (
            f'a 4-metric tick opened {few} connections and a 103-metric tick '
            f'opened {many} — the write path still scales with metric count'
        )

    def test_a_tick_commits_exactly_once_whatever_it_carries(self, tmp_path: Path):
        """Atomicity and fsync cost, measured where they live: the commit count.

        Its failure-injection sibling below raises inside the row-BUILDING
        loop, before any INSERT is issued, so it only witnesses "an exception
        before the first write leaves no rows". A per-row ``execute();
        commit()`` loop passes that test AND the one-connection test above,
        while destroying both properties ``write_tick`` exists for: the tick
        lands whole or not at all, and its fsync cost is flat in the metric
        count rather than linear in it.
        """
        store, spy = self._spied_store(tmp_path)

        self._tick(store, 1_000_000, leaves=1)
        few = len(spy.commits)

        spy.clear()
        self._tick(store, 1_000_005, leaves=100)
        many = len(spy.commits)

        assert (few, many) == (1, 1), (
            f'a 4-metric tick committed {few} times and a 103-metric tick '
            f'{many} — one tick is meant to be one transaction'
        )

    def test_a_tick_that_raises_partway_leaves_no_rows_at_all(self, tmp_path: Path):
        """Atomicity: a half-written tick must never reach the corpus.

        Under the per-row path the PSI rows were already committed by the time
        a windowed metric failed, so the tick landed truncated and silently —
        ε1/ε2 would read it as a tick on which those metrics were unreadable.
        The failure lands on the SECOND trailing-window read, partway through
        the row-building loop and before any INSERT.
        """
        from sampler.store import LoadSampleStore

        window_reads: list[str] = []

        def second_window_read(sql: str) -> bool:
            if not sql.lstrip().startswith('SELECT value FROM samples'):
                return False
            window_reads.append(sql)
            return len(window_reads) == 2

        store = LoadSampleStore(
            tmp_path / 'db.sqlite', connect=ConnectSpy(fails=second_window_read)
        )

        with pytest.raises(sqlite3.OperationalError):
            self._tick(store, 1_000_000)

        conn = sqlite3.connect(str(store.db_path))
        try:
            rows = conn.execute(
                'SELECT COUNT(*) FROM samples WHERE ts = ?', (1_000_000,)
            ).fetchone()[0]
        finally:
            conn.close()
        assert rows == 0, f'a failed tick left {rows} rows behind'

    def test_unwindowed_rows_keep_null_windows_and_windowed_rows_keep_theirs(
        self, tmp_path: Path
    ):
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        store.write_tick(
            1_000_000,
            unwindowed={PSI_CPU: 1.5},
            windowed={VERIFY: 4.0},
        )

        conn = sqlite3.connect(str(store.db_path))
        try:
            rows = {
                metric: (mean, mx)
                for metric, mean, mx in conn.execute(
                    'SELECT metric, window_mean, window_max FROM samples'
                ).fetchall()
            }
        finally:
            conn.close()
        assert rows['psi_cpu_some_avg10'] == (None, None)
        assert rows['verify_concurrency'] == (4.0, 4.0)

    def test_the_batched_windows_agree_with_the_trailing_window_helper(
        self, tmp_path: Path
    ):
        """One home for the window arithmetic, reconciled (heuristic 11).

        ``write_tick`` computes the windows on its own connection rather than
        calling the public helper, so this is the test that stops the two
        drifting apart.
        """
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        for i in range(5):
            store.insert_sample(1_000_000 + i, VERIFY, float(i))
        expected = store.trailing_window(VERIFY, 99.0)

        store.write_tick(
            1_000_010, unwindowed={}, windowed={VERIFY: 99.0}
        )

        conn = sqlite3.connect(str(store.db_path))
        try:
            got = conn.execute(
                'SELECT window_mean, window_max FROM samples WHERE ts = ?',
                (1_000_010,),
            ).fetchone()
        finally:
            conn.close()
        assert got == expected


# ---------------------------------------------------------------------------
# Task 6772: every connection comes from the one injected connect
# ---------------------------------------------------------------------------


class TestEveryConnectionGoesThroughTheInjectedConnect:
    def test_every_connection_the_store_opens_goes_through_the_injected_connect(
        self, tmp_path: Path
    ):
        """Nothing is monkeypatched; the injected connect redirects instead.

        The store is handed one path and its connect opens another, so a
        connection that bypassed the seam would create the named file — and
        every public operation must ALSO have opened at least one connection
        through the spy, so none of them can be serving itself some other way.
        """
        from sampler.store import LoadSampleStore

        named = tmp_path / 'named.sqlite'
        real = tmp_path / 'real.sqlite'
        spy = ConnectSpy()

        def redirected(db_path: Path, *, autocommit: bool = False) -> Any:
            assert db_path == named
            return spy(real, autocommit=autocommit)

        now = 1_000_000 + 20_000 * 5
        operations: list[tuple[str, Callable[[LoadSampleStore], object]]] = [
            ('insert_sample', lambda s: s.insert_sample(now, RUNQ, 1.0)),
            (
                'write_tick',
                lambda s: s.write_tick(
                    now + 5,
                    unwindowed={PSI_CPU: 1.0},
                    windowed={RUNQ: 2.0},
                ),
            ),
            ('trailing_window', lambda s: s.trailing_window(RUNQ, 3.0)),
            ('cleanup_old', lambda s: s.cleanup_old(now, retain_seconds=100)),
            ('reclaimable_fraction', lambda s: s.reclaimable_fraction()),
            ('maybe_vacuum', lambda s: s.maybe_vacuum(now)),
        ]

        store = LoadSampleStore(named, connect=redirected)
        assert spy.opened, 'construction opened no connection through the seam'
        _bulk_insert(real, 'psi_cpu_some_avg10', 1_000_000, 20_000)
        bypassed = []
        for name, operation in operations:
            before = len(spy.opened)
            operation(store)
            if len(spy.opened) == before:
                bypassed.append(name)

        assert bypassed == [], f'these operations opened no connection through the seam: {bypassed}'
        assert spy.vacuums_ran_on() == [True], 'the VACUUM path was not reached'
        assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith('named')) == [], (
            'a connection bypassed the injected connect and created the named file'
        )
