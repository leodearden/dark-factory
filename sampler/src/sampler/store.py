"""SQLite persistence for load sampler metrics.

Schema
------
samples(ts INTEGER, metric TEXT, value REAL, window_mean REAL, window_max REAL)
    idx_samples_metric_ts ON samples(metric, ts)

meta(key TEXT PRIMARY KEY, value TEXT)
    Tracks last_vacuum_ts for the daily VACUUM gate and last_cleanup_ts for
    the daily retention gate. The two are independent clocks.

Design
------
Mirrors orchestrator.run_store.RunStore:
- connect-per-call (no persistent connection)
- apply_full_durability_pragmas_sync on every connection
- IF NOT EXISTS schema (idempotent _ensure_schema)

Retention policy
----------------
- cleanup_old(now): DELETE rows older than 30 days. Called every tick, but
  gated by meta.last_cleanup_ts to run at most once per 24h — the DELETE is a
  full table SCAN and cannot be index-backed (see cleanup_old's docstring for
  the measurement and the rejected alternative).
- maybe_vacuum(now): VACUUM at most once per 24h, gated by meta.last_vacuum_ts.
  VACUUM runs outside a transaction to satisfy SQLite constraints.

The 30-day window is what the threshold calibration in PRD
``plans/load-throttle-harmonisation-prd.md`` D11 needs: a fortnight of
production load with enough margin either side to see a weekly cycle.

Sizing, measured on this host rather than estimated, so the next reader
inherits the numbers instead of re-deriving them:

    metrics/tick   25  (6 PSI + 3 process + 2 runqueue + 2 x 7 cgroup leaves)
    ticks/day      17,280  (the paired .timer's OnUnitActiveSec=5s)
    rows/day       432,000
    rows at 30 d   12,960,000
    bytes/row      125.3  including the index, which is 43% of the file
    file at 30 d   ~1.62 GB

against ~19 MB for the 9-metric, 24-hour steady state this replaces — about
85x, NOT the ~30x the originating task estimated. That estimate counted the
retention widening (24 h -> 30 d) but not the metric-count widening (9 -> 25)
that lands in the same change.

Trailing window
---------------
trailing_window(metric, current_value, window=60):
  Reads the last `window-1` rows from the DB for the metric, combines with
  current_value, returns (fmean, max).  Under the oneshot model the in-process
  ring-buffer approach is impossible; DB-backed windows are the only option.
"""

from __future__ import annotations

import logging
import sqlite3
import statistics
from pathlib import Path

from shared.sqlite_sync_base import apply_full_durability_pragmas_sync

__all__ = ['LoadSampleStore']

logger = logging.getLogger(__name__)

_SCHEMA = """\
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


class LoadSampleStore:
    """Synchronous SQLite writer for per-tick load samples."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a connection and apply the full durability pragma triad."""
        conn = sqlite3.connect(str(self.db_path))
        apply_full_durability_pragmas_sync(conn, busy_timeout_ms=5000)
        return conn

    def _ensure_schema(self) -> None:
        """Create the samples + meta tables and index idempotently."""
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert_sample(
        self,
        ts: int,
        metric: str,
        value: float,
        *,
        window_mean: float | None = None,
        window_max: float | None = None,
    ) -> None:
        """Insert a single sample row and commit."""
        conn = self._connect()
        try:
            conn.execute(
                'INSERT INTO samples (ts, metric, value, window_mean, window_max)'
                ' VALUES (?, ?, ?, ?, ?)',
                (ts, metric, value, window_mean, window_max),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Trailing window
    # ------------------------------------------------------------------

    def trailing_window(
        self,
        metric: str,
        current_value: float,
        *,
        window: int = 60,
    ) -> tuple[float, float]:
        """Return (window_mean, window_max) over the last `window` samples.

        Reads the `window-1` most-recent prior DB rows for *metric*, prepends
        *current_value*, and computes mean and max over the combined list.
        If there are no prior rows, returns (current_value, current_value).
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                'SELECT value FROM samples'
                ' WHERE metric = ?'
                ' ORDER BY ts DESC'
                ' LIMIT ?',
                (metric, window - 1),
            ).fetchall()
        finally:
            conn.close()

        values = [current_value] + [r[0] for r in rows]
        return statistics.fmean(values), max(values)

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    def should_cleanup(self, now: int, *, interval_seconds: int = 86400) -> bool:
        """Return True when a cleanup is due (no prior record or interval elapsed).

        Same semantics as ``should_vacuum``, deliberately: absent record means
        due, so a fresh store prunes on its first tick.
        """
        last_str = self._get_meta('last_cleanup_ts')
        if last_str is None:
            return True
        return (now - int(last_str)) >= interval_seconds

    def cleanup_old(
        self,
        now: int,
        *,
        retain_seconds: int = 2_592_000,
        interval_seconds: int = 86400,
    ) -> None:
        """Delete samples older than ``retain_seconds``, at most once per interval.

        The default window is 30 days (see "Retention policy" above). The
        cutoff is exclusive — a row at exactly ``now - retain_seconds``
        survives.

        The interval GATE is what makes a per-tick call affordable.
        ``DELETE FROM samples WHERE ts < ?`` plans as ``SCAN samples``:
        ``idx_samples_metric_ts(metric, ts)`` cannot serve a bare-ts
        predicate, because a leading-column index does not. Measured at 2.5M
        rows, a NO-OP cleanup — nothing old enough to delete — costs 106.7 ms;
        extrapolated to the 12.96M-row 30-day steady state that is ~550 ms of
        full-table scan every 5 s, forever, to delete nothing (~11% duty).

        The alternative measured was ``CREATE INDEX idx_samples_ts``, which
        turns the plan into ``SEARCH samples USING INDEX idx_samples_ts
        (ts<?)`` at ~0 ms but grew the probe file 33% (98 -> 130 MB, i.e.
        1.62 -> ~2.15 GB at 30 d). The gate amortises the same scan to once
        per interval, where 550 ms is irrelevant, and costs zero extra bytes —
        winning on 530 MB of disk and on reusing the ``should_vacuum`` /
        ``last_vacuum_ts`` pattern that already lives in this class. Its only
        cost, up to one interval of over-retention past 30 days, is meaningless
        for a calibration corpus. Doing both would make the index dead weight,
        so exactly one is taken.

        ``trailing_window`` is deliberately untouched: its SELECT already
        plans as ``SEARCH samples USING INDEX idx_samples_metric_ts
        (metric=?)`` and measured ~0 ms at 2.5M rows.

        The timestamp is stamped only AFTER the DELETE commits — maybe_vacuum's
        ordering — so a transient failure does not suppress retries for a whole
        interval and leave the next window double-length.
        """
        if not self.should_cleanup(now, interval_seconds=interval_seconds):
            return
        cutoff = now - retain_seconds
        conn = self._connect()
        try:
            conn.execute('DELETE FROM samples WHERE ts < ?', (cutoff,))
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
            conn.execute(
                'INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)',
                (key, value),
            )
            conn.commit()
        finally:
            conn.close()

    def should_vacuum(self, now: int, *, interval_seconds: int = 86400) -> bool:
        """Return True when a VACUUM is due (no prior record or interval elapsed)."""
        last_str = self._get_meta('last_vacuum_ts')
        if last_str is None:
            return True
        return (now - int(last_str)) >= interval_seconds

    def maybe_vacuum(self, now: int) -> None:
        """Run VACUUM and record the timestamp if the daily interval has elapsed."""
        if not self.should_vacuum(now):
            return
        # VACUUM must run outside a transaction.  Connect with isolation_level=None
        # (autocommit) from the start so that pragma application inside
        # apply_full_durability_pragmas_sync cannot begin an implicit transaction
        # that would cause VACUUM to error with
        # "cannot VACUUM from within a transaction".
        conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        try:
            apply_full_durability_pragmas_sync(conn, busy_timeout_ms=5000)
            conn.execute('VACUUM')
            # Record the timestamp only after a successful VACUUM so that a
            # transient VACUUM failure doesn't suppress retries for 24h.
            conn.execute(
                'INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)',
                ('last_vacuum_ts', str(now)),
            )
        except Exception:
            logger.exception('VACUUM failed; will retry on next eligible tick')
        finally:
            conn.close()
