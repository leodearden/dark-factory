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
contains all 9 keys; absent-from-DB known metrics return the placeholder
shape ``{current: null, sparkline: [], window_mean: null, window_max: null}``.

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

_QUERY_SQL = f"""\
SELECT metric, value, window_mean, window_max, ts
FROM (
    SELECT metric, value, window_mean, window_max, ts,
           ROW_NUMBER() OVER (PARTITION BY metric ORDER BY ts DESC) AS rn
    FROM samples
    WHERE metric IN ({_PLACEHOLDERS_SQL})
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

        rows = await conn.execute_fetchall(_QUERY_SQL, KNOWN_METRICS)

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
