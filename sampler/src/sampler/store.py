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

The unit of writing is the TICK, not the row: write_tick(ts, unwindowed=,
windowed=) reads every trailing window and inserts every row on one connection
inside one transaction, so a tick's cost is flat in its metric count and a
tick reaches the corpus whole or not at all (see its docstring for both
measurements). insert_sample remains the single-row primitive, with no
production caller of its own -- see its docstring.

Retention policy
----------------
- cleanup_old(now): DELETE rows outside ±30 days of now — the future half
  prunes clock-skew rows a past-only cutoff can never reach. Called every
  tick, gated by meta.last_cleanup_ts to at most once per 24h because the
  DELETE is a full table SCAN and cannot be index-backed.
- maybe_vacuum(now): VACUUM at most once per 24h, gated by meta.last_vacuum_ts
  AND by a floor on the free-page fraction, because at the 30-day size a
  steady-state VACUUM stalls a tick for seconds to reclaim almost nothing.
  VACUUM runs outside a transaction to satisfy SQLite constraints.
- Both gates read _is_due, which treats a stamp in the FUTURE as due — a
  forward clock step must not be able to disable either sweep.

Every number behind those two policies lives on the method that implements it
(``cleanup_old``, ``maybe_vacuum``): the scan cost and the rejected ts index,
the VACUUM's cost and what a steady-state day actually reclaims. They are not
restated here, so there is one copy to keep true.

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
from collections.abc import Mapping
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


# A VACUUM is worth its whole-file rewrite only past this share of free pages.
# Measured on the 30-day probe: a steady-state daily prune leaves ~3.3% free,
# which the next day's inserts fully reclaim, so 10% sits clear of the normal
# transient while still firing on a retention change or a post-outage prune.
_VACUUM_MIN_RECLAIMABLE_FRACTION = 0.10

# Samples a trailing window spans, current value included. One home for it, so
# ``write_tick`` and ``trailing_window`` cannot disagree about the span.
_TRAILING_WINDOW_SAMPLES = 60


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
        """Insert a single sample row and commit.

        NO PRODUCTION CALLER, deliberately, and said here so nobody has to grep
        to find out: every tick goes through ``write_tick``, and the ~30
        remaining call sites are all in this store's own tests, seeding rows.

        Kept rather than deleted because it is the store's one-row primitive —
        the smallest honest unit of this interface — and because deleting it
        would rewrite those call sites into ``write_tick(ts, unwindowed=...,
        windowed={})``, which is a wordier way to say the same thing and drags
        the windowing machinery into tests that are not about windows.
        """
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

    def write_tick(
        self,
        ts: int,
        *,
        unwindowed: Mapping[str, float],
        windowed: Mapping[str, float],
        window: int = _TRAILING_WINDOW_SAMPLES,
    ) -> None:
        """Write one whole tick on ONE connection inside ONE transaction.

        *unwindowed* metrics are stored with NULL window columns because their
        source already carries its own window — a PSI ``avg10`` is kernel-
        windowed, and re-windowing it here would report a window of windows.
        *windowed* metrics get ``window_mean``/``window_max`` computed from
        their own trailing history.

        The two are separate parameters rather than one pre-shaped row list
        because the window MODE is the only axis on which the store treats a
        metric differently; which collection group a metric came from is the
        caller's business and is deliberately not visible here.

        COST is why this exists. Every row used to take its own connection, its
        own five durability pragmas and its own ``synchronous=FULL`` commit,
        and every windowed metric took a second connection for its trailing
        window — about 44 connections and 25 fsyncs for the 25-metric tick this
        change's vocabulary produces. Measured on this host against warmed
        stores on ext4, the two paths INTERLEAVED in one process so host noise
        hits both, 40 ticks each:

            per-row .... 309.6 ms median (mean 330.9, max 592.8)
            batched ....  13.2 ms median (mean  15.1, max  34.6)

        23x, or 6.2% of the paired timer's 5 s cadence down to 0.26%, on a host
        that also runs seven orchestrators.

        What is FLAT in the metric count is the CONNECTION and fsync cost — one
        of each per tick, whatever the vocabulary. The tick as a whole is not:
        the trailing-window loop below still issues one indexed
        ``ORDER BY ts DESC LIMIT`` per WINDOWED metric, and the load group is
        routed there whole (``sampler.py::run_tick``), so even a 0/1
        ``*_read_ok`` flag buys a window read. That is worth stating because the
        cgroup leaf count is DISCOVERED per tick and nothing here bounds it — so
        it was measured, same host and method, 40 timed ticks after 80 warm-up:

            1 leaf ......   9 windowed metrics ....  4.5 ms median
            100 leaves .. 207 windowed metrics .... 23.4 ms median

        23x the metrics for 5.2x the time — the per-metric SELECT is real but
        each one is an index seek, so 100 leaves is 0.47% of the cadence. Leaf
        growth is affordable, not free; collapsing the N window reads into one
        grouped query is the move if a host ever carries leaves in the thousands.

        ATOMICITY comes with it and is not incidental. Every read happens
        before any insert, and the single commit lands the whole tick or none
        of it, so the corpus ε1/ε2 calibrate against can no longer contain a
        tick that was cut in half by a crash — which would have read as "those
        metrics were unreadable on that tick", a fact that never happened.
        Reading first is also what keeps the windows identical to the per-row
        path: no row of this tick is visible to any window of this tick.
        """
        if not unwindowed and not windowed:
            return
        conn = self._connect()
        try:
            # Annotated because the unwindowed comprehension alone would fix the
            # element type at ``None`` for both window columns, which the
            # windowed append below then contradicts.
            rows: list[tuple[int, str, float, float | None, float | None]] = [
                (ts, metric, value, None, None) for metric, value in unwindowed.items()
            ]
            for metric, value in windowed.items():
                window_mean, window_max = self._trailing_window(
                    conn, metric, value, window=window
                )
                rows.append((ts, metric, value, window_mean, window_max))
            conn.executemany(
                'INSERT INTO samples (ts, metric, value, window_mean, window_max)'
                ' VALUES (?, ?, ?, ?, ?)',
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Trailing window
    # ------------------------------------------------------------------

    @staticmethod
    def _trailing_window(
        conn: sqlite3.Connection,
        metric: str,
        current_value: float,
        *,
        window: int,
    ) -> tuple[float, float]:
        """The window arithmetic, on a caller-supplied connection.

        One home for it (heuristic 11): ``write_tick`` needs it on the
        connection it already holds, ``trailing_window`` needs it on one of its
        own, and a second copy would be free to drift.

        The span is bounded by ROW COUNT and not by time, so a "60-sample
        window" spans whatever wall-clock those 60 rows cover: 300 s at the
        5 s cadence, and the outage plus 300 s for a metric whose unit was
        stopped for hours. No ``ts >=`` floor is imposed, because these
        columns are a convenience over rows that are all kept anyway — a floor
        would silently narrow the window instead, and a consumer that needs
        wall-clock windows can compute them from the raw ``ts``/``value``
        rows, which is what ``scripts/load-threshold-calibration.py`` already
        does.
        """
        rows = conn.execute(
            'SELECT value FROM samples'
            ' WHERE metric = ?'
            ' ORDER BY ts DESC'
            ' LIMIT ?',
            (metric, window - 1),
        ).fetchall()
        values = [current_value] + [r[0] for r in rows]
        return statistics.fmean(values), max(values)

    def trailing_window(
        self,
        metric: str,
        current_value: float,
        *,
        window: int = _TRAILING_WINDOW_SAMPLES,
    ) -> tuple[float, float]:
        """Return (window_mean, window_max) over the last `window` samples.

        Reads the `window-1` most-recent prior DB rows for *metric*, prepends
        *current_value*, and computes mean and max over the combined list.
        If there are no prior rows, returns (current_value, current_value).
        """
        conn = self._connect()
        try:
            return self._trailing_window(conn, metric, current_value, window=window)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    def _is_due(self, key: str, now: int, interval_seconds: int) -> bool:
        """Return True when the interval clock at *key* has run out.

        One home for both gates (heuristic 11): ``should_cleanup`` and
        ``should_vacuum`` differ only in which meta key they read, and a second
        copy of this arithmetic was free to drift from the first.

        An ABSENT record means due, so a fresh store prunes and compacts on its
        first tick.

        A record in the FUTURE also means due, and that is the non-obvious
        half. ``now`` is ``int(time.time())`` with no monotonicity guard
        (``sampler/src/sampler/__main__.py``), and the stamp is whatever
        ``now`` the stamping tick carried — so one tick during a forward clock
        step writes a stamp no later tick can reach with a plain
        ``elapsed >= interval``. That reading would suppress BOTH the retention
        sweep and the VACUUM for the entire skew, and the rows it would
        preserve are exactly the future-dated ones ``cleanup_old``'s symmetric
        cutoff exists to prune: the corpus could not heal itself from the very
        condition that disabled the healing. Elapsed time outside
        ``[0, interval)`` is therefore due, which reads "the clock is either
        spent or nonsense, and running the sweep is cheap either way".
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
        """Delete samples older than ``retain_seconds``, at most once per interval.

        The default window is 30 days (see "Retention policy" above), and it
        is SYMMETRIC about ``now``: rows more than ``retain_seconds`` in the
        FUTURE are deleted too. Both cutoffs are exclusive — a row at exactly
        ``now ± retain_seconds`` survives.

        The future half is not symmetry for its own sake. ``ts`` is stamped
        ``int(time.time())`` with no monotonicity guard
        (``sampler/__main__.py``), so an NTP step forward, a VM
        suspend/resume, or a hand-seeded probe row can land a sample beyond
        any real tick. Such a row is unreachable by the past-only cutoff — it
        outlives the whole corpus — and it is not inert: every consumer that
        anchors on ``MAX(ts)`` (``dashboard/src/dashboard/data/load.py``'s
        recency bound, ``trailing_window``'s ordering) is pulled forward with
        it. Pruning it is the only path back to a healthy corpus.

        The future cutoff is ``retain_seconds`` and not something tighter
        BECAUSE THE CLOCK CUTS BOTH WAYS: a host whose clock reads
        ``now`` too EARLY (unsynced NTP at boot) sees a healthy corpus as
        future-dated, and a tight tolerance would delete all of it. At 30 days
        only a clock wrong by more than a month destroys data, and a host that
        wrong has no usable corpus either way. Consumers that need to be
        robust to a skewed row WITHIN the window clamp their own anchor rather
        than rely on this sweep — see load.py's ``_ANCHOR_SQL``.

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
            conn.execute(
                'INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)',
                (key, value),
            )
            conn.commit()
        finally:
            conn.close()

    def should_vacuum(self, now: int, *, interval_seconds: int = 86400) -> bool:
        """Return True when a VACUUM is due (no prior record or interval elapsed)."""
        return self._is_due('last_vacuum_ts', now, interval_seconds)

    def reclaimable_fraction(self) -> float:
        """Free pages as a fraction of the file — what a VACUUM would reclaim.

        Both pragmas are header reads, not scans: measured 0.49 ms for
        connect-plus-both against the 1.16 GB 30-day probe, so this is
        affordable inside ``maybe_vacuum``'s own interval gate.
        """
        conn = self._connect()
        try:
            free = conn.execute('PRAGMA freelist_count').fetchone()[0]
            total = conn.execute('PRAGMA page_count').fetchone()[0]
        finally:
            conn.close()
        return free / total if total else 0.0

    def maybe_vacuum(self, now: int) -> None:
        """VACUUM once per interval, and only when there are pages to reclaim.

        The interval gate alone was sized for the 24-hour retention this class
        shipped with. At the 30-day window (see "Retention policy" above) the
        file is ~85x larger and VACUUM rewrites ALL of it. Measured against a
        probe carrying this schema and the real 25-metrics-per-tick vocabulary
        at the 30-day steady state (12,960,000 rows, 1.16 GB): VACUUM takes
        15.5 s and reclaims 2 MB — 0.17%.

        It reclaims so little because free pages are REUSED. One steady-state
        day, measured on the same probe: the daily prune left
        ``freelist_count`` at 9232, and the day's inserts took it back to 0
        with ``page_count`` up 0.25%. So the file does not bloat at steady
        state, and an unconditional daily VACUUM buys ~0.17% for ~15 s.

        Fifteen seconds is not free: the systemd unit is Type=oneshot, so the
        rewrite is one whole tick and the corpus ε1/ε2 calibrate against would
        carry a ~15 s hole every day. A VACUUM also needs roughly the file size
        again in temp space, so on a tight filesystem it raises, is swallowed
        by the ``except`` below, and the file is then never compacted at all.

        Gating on ``reclaimable_fraction`` keeps the compaction where it earns
        its cost — after a retention change, or a post-outage bulk prune — and
        skips it in the steady state, where ``cleanup_old`` runs first in the
        same tick and leaves only ~3.3% free for the next day's inserts to
        consume.

        A skip STAMPS the clock: deciding there is nothing to reclaim is a
        successful evaluation, not the transient failure the retry rule below
        exists for, and stamping keeps the pragma read to once per interval
        instead of once per 5 s tick.
        """
        if not self.should_vacuum(now):
            return
        if self.reclaimable_fraction() < _VACUUM_MIN_RECLAIMABLE_FRACTION:
            self._set_meta('last_vacuum_ts', str(now))
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
