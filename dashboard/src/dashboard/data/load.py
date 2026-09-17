"""Dashboard data module for host load metrics — /api/load.

Source-of-truth note
--------------------
``LOAD_SAMPLES_SCHEMA`` is a copy of ``sampler.store._SCHEMA`` (task 1536).
The dashboard tests run from the ``dashboard/`` subdir and must not take a
runtime dependency on the sampler package.  If the sampler schema changes,
this constant MUST be updated to stay aligned — the DB join columns (metric,
ts, value, window_mean, window_max) are the consumer contract.

9-key allowlist
---------------
``KNOWN_METRICS`` is the stable set of metric names the dashboard serves.
Keys not in this tuple are filtered from query results, giving the frontend
a fixed shape regardless of future sampler additions.  The response always
contains all 9 keys; a known metric with nothing to serve returns the
placeholder shape
``{current: null, sparkline: [], window_mean: null, window_max: null}``.

TWO things produce that placeholder, and only the first is "absent from the
DB".  The second is a metric whose newest row is older than
``_RECENCY_SLACK_SECONDS`` before the newest row of ANY served metric — see
the recency-bound note below.  ``sampler/__main__.py`` degrades each
collection group independently, so one group can stall while its siblings
keep writing every 5 s, and after the slack elapses the stalled group's cards
go to the placeholder.  That is intended: /api/load is polled every 5 s and
the frontend renders ``current`` as the live number, so an hour-old value is
not a stale reading of host load but a reading of a collector that has
stopped, and "no data" is the honest answer.  Pinned by
``test_a_group_that_stops_writing_blanks_while_its_siblings_keep_ticking``.

PSI window columns
------------------
PSI metrics (``psi_*``) are kernel-windowed: the sampler writes NULL for
``window_mean`` and ``window_max``.  The data module passes these NULLs
through as Python ``None`` — callers must not expect numeric window values
for PSI metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import aiosqlite

from dashboard.data.db import with_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema (mirrors sampler.store._SCHEMA — keep in sync)
# ---------------------------------------------------------------------------

LOAD_SAMPLES_SCHEMA: str = """\
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

# ---------------------------------------------------------------------------
# Known metric allowlist
# ---------------------------------------------------------------------------

# PSI metrics: kernel-windowed; window_mean/window_max always NULL in DB.
PSI_METRICS: frozenset[str] = frozenset({
    'psi_cpu_some_avg10',
    'psi_cpu_full_avg10',
    'psi_mem_some_avg10',
    'psi_mem_full_avg10',
    'psi_io_some_avg10',
    'psi_io_full_avg10',
})

# Process metrics: sampler-windowed; window_mean/window_max populated.
PROCESS_METRICS: frozenset[str] = frozenset({
    'occt_queue_depth',
    'verify_concurrency',
    'verify_rss_total_bytes',
})

# Stable ordered tuple — the frontend shape contract.
KNOWN_METRICS: tuple[str, ...] = tuple(sorted(PSI_METRICS | PROCESS_METRICS))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _placeholder() -> dict[str, Any]:
    """Return a fresh placeholder dict for an absent metric."""
    return {'current': None, 'sparkline': [], 'window_mean': None, 'window_max': None}


def _default_result() -> dict[str, dict[str, Any]]:
    """Return the all-placeholders result (fresh copy, safe to mutate)."""
    return {m: _placeholder() for m in KNOWN_METRICS}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PLACEHOLDERS_SQL = ','.join('?' * len(KNOWN_METRICS))

# Slack, in seconds, for the recency bound below.  The sparkline is 60 samples
# at the sampler's 5s tick = 300s, so an hour is 12x headroom: it absorbs
# sampler restarts and missed ticks without ever truncating a full sparkline.
_RECENCY_SLACK_SECONDS = 3600

# Why the recency bound exists (task 3592)
# ---------------------------------------
# Without `ts >=`, this query's cost is LINEAR IN RETENTION: SQLite does not
# push `rn <= 60` down into the window function, so it ranks EVERY row of the
# allowlisted metrics before discarding all but 60 per metric.  When sampler
# retention widened 24h -> 30d, the 9-metric steady state went 155k -> 4.67M
# rows.  Measured on a 4,665,600-row probe DB with this exact schema:
# 11,955 ms unbounded vs 28.7 ms bounded, for byte-identical 540-row output.
# /api/load is on a 5s frontend poll (tab_overview.jsx::LOAD_POLL_INTERVAL_MS),
# so the unbounded form took ~2x the poll interval and saturated the aiosqlite
# pool -- the same unbounded-scan-on-a-polled-endpoint mechanism behind the
# 2026-07-30 dashboard outage (tasks 3304, 3519).
#
# The bound is anchored to the newest sample (MAX(ts)), NOT to wall-clock now().
# That is load-bearing in two ways.  It keeps the data layer free of any
# wall-clock dependency, and it preserves behaviour when the sampler is DOWN: a
# now()-relative bound would blank the card after an outage longer than the
# slack, whereas anchoring to the data keeps showing the last known samples,
# exactly as the unbounded query did.
#
# The anchor is GLOBAL across the 9 served metrics, not per metric, and the
# difference shows in the PARTIAL degrade: one collection group stalls while
# its siblings keep writing, the siblings advance the anchor, and after the
# slack the stalled group's cards go to the placeholder.  Kept global
# deliberately -- see the module docstring for why "no data" is the honest
# answer there, and the test that pins it.  A per-metric bound
# (`ts >= (SELECT MAX(ts) FROM samples s2 WHERE s2.metric = samples.metric)`)
# would instead report each stalled metric's last value as current forever,
# and makes the scalar subquery correlated.
#
# Cost is linear in the SLACK, not in retention, so the slack must stay modest:
# measured on the same probe, 1h = 28.7 ms, 24h = 347 ms, 7d = 2,168 ms.
#
# The anchor is CLAMPED to now() from ABOVE, and that clamp is a bound, not a
# switch to a wall-clock anchor: when the sampler is behind or down, MAX(ts) is
# already <= now() and MIN() returns MAX(ts) unchanged, so every property above
# still holds. It bites only when MAX(ts) is in the FUTURE, which no healthy
# tick produces -- the sampler stamps `ts = int(time.time())` with no
# monotonicity guard (sampler/src/sampler/__main__.py), so an NTP step forward,
# a VM suspend/resume, or a hand-seeded probe row is enough. Unclamped, ONE
# such row drags the anchor past every real sample and serves placeholders for
# all nine metrics -- the silent, total blank-dashboard failure this whole
# bound exists to prevent, and with nothing to recover it: a future-dated row
# survives every past-only retention sweep. `sampler.store.cleanup_old` now
# prunes rows beyond +/- retain_seconds, but that sweep runs daily and only
# catches the egregious ones; this clamp is what makes the endpoint robust on
# the very next request, at any skew. Pinned by
# `test_one_future_dated_row_does_not_blank_the_other_eight_metrics`.
_ANCHOR_SQL = f"""\
MIN(
    (SELECT MAX(ts) FROM samples WHERE metric IN ({_PLACEHOLDERS_SQL})),
    CAST(strftime('%s', 'now') AS INTEGER)
)"""

_QUERY_SQL = f"""\
SELECT metric, value, window_mean, window_max, ts
FROM (
    SELECT metric, value, window_mean, window_max, ts,
           ROW_NUMBER() OVER (PARTITION BY metric ORDER BY ts DESC) AS rn
    FROM samples
    WHERE metric IN ({_PLACEHOLDERS_SQL})
      AND ts >= ({_ANCHOR_SQL}) - {_RECENCY_SLACK_SECONDS}
)
WHERE rn <= 60
ORDER BY metric, ts ASC
"""


async def get_load_metrics(
    db: aiosqlite.Connection | None,
) -> dict[str, dict[str, Any]]:
    """Return the latest value + 60-sample sparkline for each of the 9 known metrics.

    Parameters
    ----------
    db:
        An open ``aiosqlite`` connection to ``load-samples.db``, or ``None``
        when the DB is absent or could not be opened.

    Returns
    -------
    dict
        Keys are exactly ``KNOWN_METRICS``.  Each value is::

            {
                "current": float | None,
                "sparkline": list[float],   # ascending ts, ≤ 60 entries
                "window_mean": float | None,
                "window_max": float | None,
            }

        A known metric absent from the DB returns the placeholder shape.
        Unknown metrics in the DB are filtered out.
    """
    async def _query(conn: aiosqlite.Connection) -> dict[str, dict[str, Any]]:
        result = _default_result()

        rows = await conn.execute_fetchall(_QUERY_SQL, KNOWN_METRICS + KNOWN_METRICS)

        # Group rows by metric (already ordered by metric, ts ASC from SQL).
        # Rows are always aiosqlite.Row objects — DbPool.get sets
        # conn.row_factory = aiosqlite.Row before returning the connection.
        groups: dict[str, list[Any]] = {m: [] for m in KNOWN_METRICS}
        for row in rows:
            metric = row['metric']
            if metric in groups:
                groups[metric].append(row)

        for metric, metric_rows in groups.items():
            if not metric_rows:
                continue  # keep placeholder
            # Rows are already in ascending ts order (SQL: ORDER BY metric, ts ASC)
            latest = metric_rows[-1]
            sparkline = [float(r['value']) for r in metric_rows]
            current = float(latest['value'])
            wm = latest['window_mean']
            wx = latest['window_max']
            result[metric] = {
                'current': current,
                'sparkline': sparkline,
                'window_mean': float(wm) if wm is not None else None,
                'window_max': float(wx) if wx is not None else None,
            }

        return result

    return await with_db(db, _query, _default_result())
