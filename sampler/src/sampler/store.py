"""SQLite persistence for the load sampler's per-tick samples.

Schema
------
``SCHEMA`` is its one spelling.

samples(ts, metric, value, window_mean, window_max, leaf)
    One row per metric per tick. ``leaf`` is the cgroup leaf a per-leaf metric
    was read at, NULL for a host-level metric, and is the last column.
    idx_samples_metric_ts_leaf ON samples(metric, ts, leaf) serves every read
    by metric. idx_samples_metric_leaf_ts ON samples(metric, leaf, ts), partial
    on ``leaf IS NOT NULL``, serves the per-leaf trailing window.

meta(key, value)
    Two independent clocks: last_cleanup_ts for the retention gate and
    last_vacuum_ts for the VACUUM gate.

Migration
---------
A file written before the ``leaf`` column existed is migrated when the store
is built: additive and feature-detected, the column is added and every legacy
``metric:leaf`` row is split into its two columns, and the indexes are swapped,
in one transaction (``_migrate``).

Connections
-----------
Every connection comes from the injected ``connect``, ``connect_durably`` by
default, and lives for one call.

Writing
-------
The unit of writing is the TICK: ``write_tick`` lands a whole tick in one
transaction, or none of it.

Retention
---------
``cleanup_old`` keeps the rows within ±30 days of now and runs at most once a
day. ``maybe_vacuum`` runs at most once a day, and only when the reclaimable
fraction of the file is worth a whole-file rewrite. Both gates treat a stamp in
the future as due.

Trailing window
---------------
A windowed metric's ``window_mean``/``window_max`` cover its current value and
the most recent stored rows of the same (metric, leaf). The window is read
from the DB because every tick is a process of its own. A read seeks its own
(metric, leaf), so a leaf with little history costs no more than a busy one.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import statistics
from pathlib import Path
from typing import Protocol

from shared.sqlite_sync_base import apply_full_durability_pragmas_sync

from sampler.metrics import SampleKey, Samples

__all__ = ['SCHEMA', 'Connect', 'LoadSampleStore', 'connect_durably']

logger = logging.getLogger(__name__)

# ``leaf`` is last because ALTER TABLE ADD COLUMN appends, so a migrated file
# and a fresh one have the same shape.
_TABLES_DDL = """\
CREATE TABLE IF NOT EXISTS samples (
    ts          INTEGER NOT NULL,
    metric      TEXT    NOT NULL,
    value       REAL    NOT NULL,
    window_mean REAL,
    window_max  REAL,
    leaf        TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

_INDEXES_DDL = (
    'CREATE INDEX IF NOT EXISTS idx_samples_metric_ts_leaf ON samples (metric, ts, leaf)',
    'CREATE INDEX IF NOT EXISTS idx_samples_metric_leaf_ts ON samples (metric, leaf, ts)'
    ' WHERE leaf IS NOT NULL',
)

SCHEMA: str = '\n'.join([_TABLES_DDL, *(f'{ddl};' for ddl in _INDEXES_DDL)]) + '\n'

# The right-hand sides read the row's OLD metric, so one statement splits it.
_SPLIT_LEGACY_KEYS_SQL = (
    "UPDATE samples SET metric = substr(metric, 1, instr(metric, ':') - 1),"
    " leaf = substr(metric, instr(metric, ':') + 1)"
    " WHERE instr(metric, ':') > 0"
)


# A VACUUM is worth its whole-file rewrite only past this share of free pages.
_VACUUM_MIN_RECLAIMABLE_FRACTION = 0.10

# Samples a trailing window spans, current value included. One home for it, so
# ``write_tick`` and ``trailing_window`` cannot disagree about the span.
_TRAILING_WINDOW_SAMPLES = 60

_SET_META_SQL = 'INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)'

_INSERT_SQL = (
    'INSERT INTO samples (ts, metric, leaf, value, window_mean, window_max)'
    ' VALUES (?, ?, ?, ?, ?, ?)'
)

_HOST_WINDOW_SQL = (
    'SELECT value FROM samples WHERE metric = ? AND leaf IS NULL ORDER BY ts DESC LIMIT ?'
)

# ``leaf = ?``, never ``leaf IS ?``: only ``=`` proves the partial index's
# ``leaf IS NOT NULL``, and without that index the read walks every newer row of
# every other leaf of the metric.
_LEAF_WINDOW_SQL = (
    'SELECT value FROM samples WHERE metric = ? AND leaf = ? ORDER BY ts DESC LIMIT ?'
)


class Connect(Protocol):
    def __call__(self, db_path: Path, *, autocommit: bool = False) -> sqlite3.Connection: ...


def connect_durably(db_path: Path, *, autocommit: bool = False) -> sqlite3.Connection:
    """Open *db_path* under the full durability pragmas; ``autocommit`` is for VACUUM.

    SQLite refuses some statements inside a transaction, VACUUM among them, so
    an autocommit connection opens with ``isolation_level=None`` from the
    start and no pragma can begin an implicit transaction on it.
    """
    conn = sqlite3.connect(str(db_path), isolation_level=None if autocommit else 'DEFERRED')
    apply_full_durability_pragmas_sync(conn, busy_timeout_ms=5000)
    return conn


def _create_indexes(conn: sqlite3.Connection) -> None:
    for ddl in _INDEXES_DDL:
        conn.execute(ddl)


def _migrate(conn: sqlite3.Connection) -> None:
    """Give a pre-leaf ``samples`` its ``leaf`` column, in ONE transaction.

    The ADD COLUMN, the split of every legacy ``metric:leaf`` row, and the
    swap to the leaf indexes commit together or roll back together, so a
    failure leaves the file exactly as it was and the next construction
    retries. The feature-detect runs under the write lock, so a concurrent
    construction waits, then reads ``leaf`` present and does nothing.
    Statements are ``execute()``d one by one: ``executescript`` would commit
    first. The additive idiom is the one
    ``orchestrator/src/orchestrator/run_store.py::RunStore._migrate_task_results_block_context``
    follows; unlike it, a failure here raises, and the store is not built.
    """
    conn.execute('BEGIN IMMEDIATE')
    try:
        columns = {row[1] for row in conn.execute('PRAGMA table_info(samples)')}
        if 'leaf' in columns:
            conn.rollback()
            return
        conn.execute('ALTER TABLE samples ADD COLUMN leaf TEXT')
        split = conn.execute(_SPLIT_LEGACY_KEYS_SQL).rowcount
        _create_indexes(conn)
        conn.execute('DROP INDEX IF EXISTS idx_samples_metric_ts')
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    logger.info('migrated samples to the leaf column: %d metric:leaf rows split', split)


class LoadSampleStore:
    """Synchronous SQLite writer for per-tick load samples."""

    def __init__(self, db_path: Path, *, connect: Connect = connect_durably):
        self.db_path = db_path
        self._open = connect
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self, *, autocommit: bool = False) -> sqlite3.Connection:
        return self._open(self.db_path, autocommit=autocommit)

    def _ensure_schema(self) -> None:
        """Bring a fresh, pre-leaf or current file to ``SCHEMA``'s shape.

        Tables, then the migration, then the indexes: running ``SCHEMA`` whole on
        a pre-leaf file would fail, because its indexes name the ``leaf`` column
        the migration has not added yet.
        """
        with contextlib.closing(self._connect()) as conn:
            conn.executescript(_TABLES_DDL)
            _migrate(conn)
            _create_indexes(conn)
            conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert_sample(
        self,
        ts: int,
        key: SampleKey,
        value: float,
        *,
        window_mean: float | None = None,
        window_max: float | None = None,
    ) -> None:
        """The single-row primitive; production writes go through ``write_tick``."""
        conn = self._connect()
        try:
            conn.execute(
                _INSERT_SQL, (ts, key.metric, key.leaf, value, window_mean, window_max)
            )
            conn.commit()
        finally:
            conn.close()

    def write_tick(
        self,
        ts: int,
        *,
        unwindowed: Samples,
        windowed: Samples,
        window: int = _TRAILING_WINDOW_SAMPLES,
    ) -> None:
        """Write one whole tick on ONE connection inside ONE transaction.

        *unwindowed* rows are stored with NULL window columns because their
        source already carries its own window — a PSI ``avg10`` is kernel-
        windowed, and re-windowing it would report a window of windows.
        *windowed* rows get ``window_mean``/``window_max`` from their own
        (metric, leaf)'s trailing history. The window mode is the only axis on
        which the store treats a metric differently; which collection group a
        metric came from is the caller's business.

        Every window is read before any row is inserted, so no row of this tick
        is visible to any window of this tick, and the single commit lands the
        whole tick or none of it: the corpus never holds half a tick. The
        connection and commit cost is one per tick, whatever the metric count;
        each windowed metric still costs one indexed window read.
        """
        if not unwindowed and not windowed:
            return
        conn = self._connect()
        try:
            # Annotated because the unwindowed comprehension alone would fix the
            # element type at ``None`` for both window columns, which the
            # windowed append below then contradicts.
            rows: list[tuple[int, str, str | None, float, float | None, float | None]] = [
                (ts, key.metric, key.leaf, value, None, None)
                for key, value in unwindowed.items()
            ]
            for key, value in windowed.items():
                window_mean, window_max = self._trailing_window(
                    conn, key, value, window=window
                )
                rows.append((ts, key.metric, key.leaf, value, window_mean, window_max))
            conn.executemany(_INSERT_SQL, rows)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Trailing window
    # ------------------------------------------------------------------

    @staticmethod
    def _trailing_window(
        conn: sqlite3.Connection,
        key: SampleKey,
        current_value: float,
        *,
        window: int,
    ) -> tuple[float, float]:
        """(mean, max) over *current_value* and *key*'s ``window - 1`` latest rows.

        On the caller's connection, so ``write_tick`` reads windows inside its
        own transaction. The span is a ROW COUNT, not a time: it covers
        whatever wall-clock those rows do, outages included. It is scoped to
        one (metric, leaf), so no two leaves ever share a window.
        """
        if key.leaf is None:
            rows = conn.execute(_HOST_WINDOW_SQL, (key.metric, window - 1)).fetchall()
        else:
            rows = conn.execute(_LEAF_WINDOW_SQL, (key.metric, key.leaf, window - 1)).fetchall()
        values = [current_value] + [r[0] for r in rows]
        return statistics.fmean(values), max(values)

    def trailing_window(
        self,
        key: SampleKey,
        current_value: float,
        *,
        window: int = _TRAILING_WINDOW_SAMPLES,
    ) -> tuple[float, float]:
        """Return (window_mean, window_max) over *key*'s last `window` samples.

        The `window - 1` most recent stored rows of the same (metric, leaf),
        plus *current_value*; a row count, not a time span. With no prior rows
        it is (current_value, current_value).
        """
        conn = self._connect()
        try:
            return self._trailing_window(conn, key, current_value, window=window)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    def _is_due(self, key: str, now: int, interval_seconds: int) -> bool:
        """Return True when the interval clock at *key* has run out.

        An absent record is due, so a fresh store prunes and compacts on its
        first tick. A record in the FUTURE is due too, so a stamp written during
        a forward clock step cannot disable either sweep.
        """
        last_str = self._get_meta(key)
        if last_str is None:
            return True
        return not 0 <= now - int(last_str) < interval_seconds

    def should_cleanup(self, now: int, *, interval_seconds: int = 86400) -> bool:
        """Return True when a cleanup is due (no prior record or interval elapsed)."""
        return self._is_due('last_cleanup_ts', now, interval_seconds)

    def cleanup_old(
        self,
        now: int,
        *,
        retain_seconds: int = 30 * 24 * 60 * 60,
        interval_seconds: int = 86400,
    ) -> None:
        """Delete rows more than ``retain_seconds`` from ``now``, once per interval.

        The cutoff is SYMMETRIC about ``now``: rows that far in the future go
        too, because a forward clock step stamps rows that no past-only cutoff
        would ever reach. Both bounds are exclusive — a row at exactly
        ``now ± retain_seconds`` survives.

        The DELETE is a full table scan, since no index leads with ``ts``,
        which is why it runs at most once per interval. The interval is stamped
        only after the DELETE commits, so a failure is retried on the next tick.
        """
        if not self.should_cleanup(now, interval_seconds=interval_seconds):
            return
        conn = self._connect()
        try:
            conn.execute(
                'DELETE FROM samples WHERE ts < ? OR ts > ?',
                (now - retain_seconds, now + retain_seconds),
            )
            conn.commit()
        finally:
            conn.close()
        self._set_meta('last_cleanup_ts', str(now))

    # ------------------------------------------------------------------
    # Vacuum gating via meta table
    # ------------------------------------------------------------------

    def _get_meta(self, key: str) -> str | None:
        conn = self._connect()
        try:
            row = conn.execute('SELECT value FROM meta WHERE key = ?', (key,)).fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        conn = self._connect()
        try:
            conn.execute(_SET_META_SQL, (key, value))
            conn.commit()
        finally:
            conn.close()

    def should_vacuum(self, now: int, *, interval_seconds: int = 86400) -> bool:
        """Return True when a VACUUM is due (no prior record or interval elapsed)."""
        return self._is_due('last_vacuum_ts', now, interval_seconds)

    def reclaimable_fraction(self) -> float:
        """Free pages as a fraction of the file — what a VACUUM would reclaim.

        Both pragmas read the file header; neither scans the table.
        """
        conn = self._connect()
        try:
            free = conn.execute('PRAGMA freelist_count').fetchone()[0]
            total = conn.execute('PRAGMA page_count').fetchone()[0]
        finally:
            conn.close()
        return free / total if total else 0.0

    def maybe_vacuum(self, now: int) -> None:
        """VACUUM at most once per interval, and only when it would reclaim enough.

        A VACUUM rewrites the whole file, and in the steady state the next
        day's inserts reuse the pages a prune frees. So it runs only when
        ``reclaimable_fraction`` reaches ``_VACUUM_MIN_RECLAIMABLE_FRACTION``,
        as after a retention change or a bulk prune.

        A skip stamps the clock: finding nothing to reclaim is a successful
        evaluation. A VACUUM stamps it only on success, and a failure, opening
        the connection included, is logged and retried on the next eligible
        tick.
        """
        if not self.should_vacuum(now):
            return
        if self.reclaimable_fraction() < _VACUUM_MIN_RECLAIMABLE_FRACTION:
            self._set_meta('last_vacuum_ts', str(now))
            return
        try:
            with contextlib.closing(self._connect(autocommit=True)) as conn:
                conn.execute('VACUUM')
                conn.execute(_SET_META_SQL, ('last_vacuum_ts', str(now)))
        except Exception:
            logger.exception('VACUUM failed; will retry on next eligible tick')
