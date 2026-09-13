"""SQLite persistence for load sampler metrics.

Schema
------
samples(ts INTEGER, metric TEXT, value REAL, window_mean REAL, window_max REAL)
    idx_samples_metric_ts ON samples(metric, ts)

meta(key TEXT PRIMARY KEY, value TEXT)
    Used to track last_vacuum_ts for the daily VACUUM gate.

Design
------
Mirrors orchestrator.run_store.RunStore:
- connect-per-call (no persistent connection)
- apply_full_durability_pragmas_sync on every connection
- IF NOT EXISTS schema (idempotent _ensure_schema)

Retention policy
----------------
- cleanup_old(now): DELETE rows older than 30 days, called every tick.
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

    def cleanup_old(self, now: int, *, retain_seconds: int = 2_592_000) -> None:
        """Delete samples older than ``retain_seconds`` relative to ``now``.

        The default is 30 days (see "Retention policy" above). The cutoff is
        exclusive — a row at exactly ``now - retain_seconds`` survives.
        """
        cutoff = now - retain_seconds
        conn = self._connect()
        try:
            conn.execute('DELETE FROM samples WHERE ts < ?', (cutoff,))
            conn.commit()
        finally:
            conn.close()

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
