"""Periodic metrics-snapshot collection and chart queries.

Records sparse-history signals (orchestrator running count, fused-memory
store sizes, write queue depth, reconciliation state, merge queue active
depth) into a dedicated SQLite database so they can be rendered as time
series even though the underlying sources are point-in-time only.

Sampling cadence and downsampling rules match
``dashboard/data/burndown.py`` so both files retain at most 90 days of
history (10-minute raw, hourly after 7 days). The collector runs on a
sibling lifespan task in ``app.py``; route handlers read via the shared
``DbPool`` (read-only).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import aiosqlite
import httpx

from dashboard.config import DashboardConfig
from dashboard.data.cap_history import (
    compute_capped_now_and_windows,
    compute_overlap_ms,
    read_cap_intervals,
)
from dashboard.data.memory import (
    get_curator_state,
    get_memory_status,
    get_queue_stats,
    mcp_tool_call,
)
from dashboard.data.merge_queue import active_queued_merges
from dashboard.data.orchestrator import (
    _read_project_root_from_config,
    _resolve_project_root,
    find_running_orchestrators,
)
from dashboard.data.reconciliation import get_buffer_stats, get_burst_state, partition_burst_state
from dashboard.data.stats_utils import percentile
from dashboard.data.utils import safe_gather_result

logger = logging.getLogger(__name__)

# 5s ceiling on any individual fused-memory HTTP call so a hung MCP cannot
# extend the 10-minute sampling cycle.  Note: this is per-(root, url) call,
# not per-loop — worst-case latency for the pending-count accumulation loop
# scales as N_roots × M_urls × 5s if all URLs hang for every root.
_HTTP_SAMPLER_TIMEOUT_SECONDS = 5.0
# Default per-(root, url) page size for list_tickets; saturation triggers a WARNING.
_LIST_TICKETS_LIMIT = 2000


async def fan_out_list_tickets(
    http_client: httpx.AsyncClient,
    config: DashboardConfig,
    *,
    limit: int | None = None,
    timeout: float = _HTTP_SAMPLER_TIMEOUT_SECONDS,
) -> tuple[list[dict], int]:
    """Fan-out ``list_tickets`` across all project roots; first-success-per-root.

    Returns ``(tickets, pending_total)`` where:

    * ``tickets`` — raw MCP ticket-row dicts augmented with a ``project_id``
      key taken from the top-level response field.  Callers that only need the
      count can ignore this list.
    * ``pending_total`` — sum of the per-root ``count`` fields.

    Roots are queried concurrently via ``asyncio.gather``; within each root the
    URL list is tried sequentially (first-success wins via ``break``-equivalent
    early return) so URLs remain a fallback chain without double-counting.

    If any per-root count saturates at *limit* the server may have clipped the
    real depth — a WARNING is logged so it surfaces in operator logs.

    Per-(url, root) network/decode errors are swallowed at DEBUG level; timeouts
    surface at WARNING so slow fused-memory instances are visible to operators.
    Unexpected exception types are logged at WARNING level so programming bugs
    don't silently vanish into the fan-out loop.
    """
    # Resolve limit at call time so monkeypatching _LIST_TICKETS_LIMIT in tests
    # takes effect without needing to pass an explicit kwarg.
    effective_limit = limit if limit is not None else _LIST_TICKETS_LIMIT
    seen_roots: set[str] = set()
    all_roots: list[str] = []
    for root in [config.project_root, *config.known_project_roots]:
        key = str(root)
        if key not in seen_roots:
            seen_roots.add(key)
            all_roots.append(key)

    async def _fan_out_one_root(root_str: str) -> tuple[int, list[dict]]:
        """Query list_tickets for *root_str* across all configured URLs.

        URLs are tried sequentially; the first success returns immediately
        (equivalent to ``break`` in the outer URL loop) to avoid double-counting.
        Returns ``(count, tickets_with_project_id)`` or ``(0, [])`` if every URL fails.
        """
        for url in config.fused_memory_urls:
            try:
                result = await asyncio.wait_for(
                    mcp_tool_call(
                        http_client,
                        url,
                        'list_tickets',
                        {'project_root': root_str, 'status': 'pending', 'limit': effective_limit},
                    ),
                    timeout=timeout,
                )
                count = result.get('count', 0) or 0
                if count >= effective_limit:
                    logger.warning(
                        'list_tickets returned count=%d at-or-above the requested '
                        'limit of %d for %s / %s — '
                        'pending_total may be clipped at the server limit',
                        count,
                        effective_limit,
                        url,
                        root_str,
                    )
                project_id = result.get('project_id', '')
                root_tickets = [{**r, 'project_id': project_id} for r in result.get('tickets', [])]
                return count, root_tickets  # first success wins; skip remaining URLs
            except TimeoutError:
                logger.warning(
                    'list_tickets timed out for project_root=%s url=%s after %.1fs',
                    root_str,
                    url,
                    timeout,
                )
            except (httpx.HTTPError, ValueError):
                logger.debug(
                    'list_tickets failed for %s / %s',
                    url,
                    root_str,
                    exc_info=True,
                )
            except Exception:
                logger.warning(
                    'list_tickets unexpected error for %s / %s',
                    url,
                    root_str,
                    exc_info=True,
                )
        return 0, []

    raw_results = await asyncio.gather(
        *(_fan_out_one_root(r) for r in all_roots), return_exceptions=True
    )
    # safe_gather_result re-raises non-Exception BaseException (CancelledError,
    # KeyboardInterrupt, SystemExit) so shutdown cancellation propagates correctly.
    # Exception subclasses (including per-root failures) are substituted with the
    # (0, []) default and logged at WARNING — same convention as the sibling legs
    # in api_curator (curator/sparks, curator/intervals, curator/fan_out).
    per_root_results: list[tuple[int, list[dict]]] = [
        safe_gather_result(r, cast('tuple[int, list[dict]]', (0, [])), 'curator/fan_out_root')
        for r in raw_results
    ]
    pending_total = sum(c for c, _ in per_root_results)
    tickets: list[dict] = [t for _, ts in per_root_results for t in ts]
    return tickets, pending_total


METRICS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS orchestrator_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    project_id    TEXT    NOT NULL,
    running_count INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orch_snap_pid_ts ON orchestrator_snapshots(project_id, ts);

CREATE TABLE IF NOT EXISTS memory_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    project_id      TEXT    NOT NULL,
    graphiti_nodes  INTEGER,
    mem0_memories   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_mem_snap_pid_ts ON memory_snapshots(project_id, ts);

CREATE TABLE IF NOT EXISTS queue_snapshots (
    ts      TEXT PRIMARY KEY,
    pending INTEGER,
    retry   INTEGER,
    dead    INTEGER
);

CREATE TABLE IF NOT EXISTS recon_snapshots (
    ts             TEXT PRIMARY KEY,
    buffered_count INTEGER,
    active_agents  INTEGER
);

CREATE TABLE IF NOT EXISTS merge_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT    NOT NULL,
    project_id   TEXT    NOT NULL,
    active_count INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_merge_snap_pid_ts ON merge_snapshots(project_id, ts);

CREATE TABLE IF NOT EXISTS curator_snapshots (
    ts            TEXT PRIMARY KEY,
    pending_total INTEGER,
    capped_now    INTEGER,
    p50_active_ms INTEGER,
    p90_active_ms INTEGER,
    p99_active_ms INTEGER
);
"""


# ---------------------------------------------------------------------------
# Per-sampler helpers (return rows; never raise to the caller)
# ---------------------------------------------------------------------------


def _sample_orchestrators(config: DashboardConfig) -> list[tuple[str, int]]:
    """Return [(project_id, running_count)] for all detected orchestrators.

    Mirrors the discovery logic in burndown.collect_snapshot so per-project
    counts agree across files. Returns [] on failure (caller logs).
    """
    seen: dict[str, int] = {str(config.project_root): 0}
    procs = find_running_orchestrators()
    for proc in procs:
        try:
            if proc.get('prd'):
                root = _resolve_project_root(proc['prd'], config.project_root)
            elif proc.get('config_path'):
                resolved = _read_project_root_from_config(proc['config_path'])
                if resolved is None:
                    continue
                root = resolved
            else:
                root = config.project_root.resolve()
            key = str(root)
            seen[key] = seen.get(key, 0) + 1
        except (OSError, ValueError):
            logger.debug('Orchestrator entry sampling failed', exc_info=True)
    # Include known_project_roots even with zero count so the spark line
    # has continuity when an orchestrator stops running.
    for known in config.known_project_roots:
        seen.setdefault(str(known), 0)
    return list(seen.items())


def _split_status(status: dict) -> tuple[list[tuple[str, dict]], dict | None]:
    """Pull (project_id, project_dict) pairs out of a fused-memory status.

    Returns (project_pairs, queue_dict_or_none). On offline or malformed
    payloads, both pieces degrade to empty/None — never raises.
    """
    if not isinstance(status, dict) or status.get('offline'):
        return [], None
    projects = status.get('projects')
    pairs: list[tuple[str, dict]] = []
    if isinstance(projects, dict):
        for pid, payload in projects.items():
            if isinstance(pid, str) and isinstance(payload, dict):
                pairs.append((pid, payload))
    queue = status.get('queue') if isinstance(status.get('queue'), dict) else None
    return pairs, queue


def _split_queue_stats(qstats: dict) -> tuple[int | None, int | None, int | None]:
    """Extract pending/retry/dead from a get_queue_stats result.

    Returns (None, None, None) on offline / malformed payloads.
    """
    if not isinstance(qstats, dict) or qstats.get('offline'):
        return None, None, None
    counts_raw = qstats.get('counts')
    counts = counts_raw if isinstance(counts_raw, dict) else {}
    return counts.get('pending'), counts.get('retry'), counts.get('dead')


# ---------------------------------------------------------------------------
# Curator sampler helper
# ---------------------------------------------------------------------------


async def _sample_curator(
    http_client: httpx.AsyncClient,
    config: DashboardConfig,
    tickets_db: aiosqlite.Connection | None,
    runs_dbs: list[aiosqlite.Connection | None],
    *,
    now: datetime | None = None,
) -> dict:
    """Sample curator queue depth, cap state, and terminal-ticket latency.

    Args:
        http_client: Shared AsyncClient for MCP HTTP calls.
        config: Dashboard configuration (fused_memory_urls, known_project_roots).
        tickets_db: Read-only aiosqlite connection to tickets.db. If None,
            ticket latency fields are all None and only pending/capped are set.
        runs_dbs: Per-project runs.db connections for cap-event lookup.
        now: Reference time (defaults to datetime.now(UTC)). Used for both
            the 1-hour ticket cutoff AND threaded into read_cap_intervals so
            the cap-window query anchors to the same clock.

    Returns:
        Always returns a dict with keys: pending_total, capped_now,
        p50_active_ms, p90_active_ms, p99_active_ms. Individual sub-errors
        are swallowed and treated as 0/None — the function never propagates
        a top-level exception to the caller.
    """
    effective_now = now if now is not None else datetime.now(UTC)

    # 1. HTTP pending count via fan_out_list_tickets (de-duped roots, first-success-per-root).
    _, pending_total = await fan_out_list_tickets(
        http_client,
        config,
        timeout=_HTTP_SAMPLER_TIMEOUT_SECONDS,
    )

    # Curator gate dict — supplies account_count (denominator for capped_now under
    # all-accounts-capped semantics, mirroring app.py:864-866). Inline try/except
    # absorbs any HTTP/transport failure so the sampler degrades to the cap-history
    # universe rather than failing outright.
    try:
        curator_state = await get_curator_state(http_client, config)
    except Exception:
        logger.debug('get_curator_state failed in _sample_curator', exc_info=True)
        curator_state = {}
    account_count: int = (
        curator_state.get('account_count', 0) if isinstance(curator_state, dict) else 0
    )

    # 2. Cap intervals (last 7 days — long enough to catch any plausible open-ended
    # cap; account_events scan is cheap and indexed by created_at).
    try:
        intervals = await read_cap_intervals(runs_dbs, days=7, now=effective_now)
    except Exception:
        logger.debug('read_cap_intervals failed', exc_info=True)
        intervals = []

    # 3. capped_now (all-accounts-capped semantics) + capped_windows (all-accounts merge)
    # derived together by compute_capped_now_and_windows — single source of truth
    # shared with api_curator. account_count is threaded in from the curator gate;
    # on gate-offline it degrades to the cap-history universe (max(0, #distinct intervals)).
    # See cap_history.py for the all-accounts-capped semantics rationale.
    try:
        capped_now, capped_windows = compute_capped_now_and_windows(intervals, account_count)
    except Exception:
        # Raise to WARNING so the silent capped_now=0 degrade is visible in metrics logs.
        # The prior code returned `1 if any(iv.end is None ...) else 0`; that any-account
        # fallback would now produce a false positive in the wedged-interval scenario the
        # helper deliberately treats as not-capped, so it is dropped in favour of
        # degrade-safe 0.
        logger.warning(
            'compute_capped_now_and_windows failed in _sample_curator — degrading capped_now to 0',
            exc_info=True,
        )
        capped_now = 0
        capped_windows = []

    # 4. Terminal tickets resolved in the last hour.
    # NOTE: centiles use a 1-hour rolling window, not the 10-minute sampling
    # interval. A ticket resolved at minute 0 will be included in ~6 consecutive
    # snapshots. This is intentional — it produces smoother spark lines at the
    # cost of the centile rows representing overlapping (not disjoint) windows.
    # pending_total and capped_now are point-in-time; only the centiles are rolling.
    p50: int | None = None
    p90: int | None = None
    p99: int | None = None
    if tickets_db is not None:
        cutoff = (effective_now - timedelta(hours=1)).isoformat()
        try:
            async with tickets_db.execute(
                'SELECT created_at, resolved_at FROM tickets '
                "WHERE resolved_at >= ? AND status IN ('created', 'combined', 'cancelled', 'failed')",
                (cutoff,),
            ) as cur:
                ticket_rows = await cur.fetchall()

            # 5. Per-ticket active_ms with cap subtraction.
            active_ms_list: list[float] = []
            for row in ticket_rows:
                created_str = row[0]  # positional access works for both aiosqlite.Row and tuple
                resolved_str = row[1]
                if not created_str or not resolved_str:
                    continue
                created_dt = datetime.fromisoformat(created_str)
                resolved_dt = datetime.fromisoformat(resolved_str)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=UTC)
                if resolved_dt.tzinfo is None:
                    resolved_dt = resolved_dt.replace(tzinfo=UTC)
                raw_ms = (resolved_dt - created_dt).total_seconds() * 1000
                overlap = compute_overlap_ms(created_dt, resolved_dt, capped_windows)
                active_ms = max(0.0, raw_ms - overlap)
                active_ms_list.append(active_ms)

            # 6. Centiles (None if no tickets).
            if active_ms_list:
                active_ms_list.sort()
                p50 = int(percentile(active_ms_list, 50))
                p90 = int(percentile(active_ms_list, 90))
                p99 = int(percentile(active_ms_list, 99))
        except Exception:
            logger.debug('curator ticket query/centile failed', exc_info=True)

    return {
        'pending_total': pending_total,
        'capped_now': capped_now,
        'p50_active_ms': p50,
        'p90_active_ms': p90,
        'p99_active_ms': p99,
    }


# ---------------------------------------------------------------------------
# Snapshot driver — runs every 10 minutes from the lifespan task.
# ---------------------------------------------------------------------------


async def collect_metrics_snapshot(
    conn: aiosqlite.Connection,
    config: DashboardConfig,
    http_client: httpx.AsyncClient,
    recon_db: aiosqlite.Connection | None,
    merge_dbs: list[tuple[str, aiosqlite.Connection | None]],
    *,
    tickets_db: aiosqlite.Connection | None = None,
) -> None:
    """Sample every metric source and insert one row per signal.

    Each sampler runs in its own try/except; a failure in one source (e.g.,
    fused-memory HTTP timeout) does not poison the others. Per-row inserts
    commit individually so a write error on one signal cannot drop earlier
    rows.

    Args:
        tickets_db: Read-only connection to tickets.db for curator latency
            sampling. Optional — when None the curator block records None
            centiles but still captures pending_total and capped_now.
    """
    now_dt = datetime.now(UTC)
    now = now_dt.isoformat()

    # Orchestrators (synchronous; subprocess in to_thread).
    try:
        rows = await asyncio.to_thread(_sample_orchestrators, config)
        for pid, count in rows:
            await conn.execute(
                'INSERT INTO orchestrator_snapshots (ts, project_id, running_count) VALUES (?, ?, ?)',
                (now, pid, count),
            )
        await conn.commit()
    except Exception:
        logger.warning('orchestrator sampler failed', exc_info=True)
        with contextlib.suppress(Exception):
            await conn.rollback()

    # Memory (HTTP via fused-memory MCP — wrap in wait_for as belt-and-braces).
    try:
        status = await asyncio.wait_for(
            get_memory_status(http_client, config),
            timeout=_HTTP_SAMPLER_TIMEOUT_SECONDS,
        )
        pairs, _queue_inline = _split_status(status)
        for pid, payload in pairs:
            await conn.execute(
                'INSERT INTO memory_snapshots (ts, project_id, graphiti_nodes, mem0_memories) '
                'VALUES (?, ?, ?, ?)',
                (now, pid, payload.get('graphiti_nodes'), payload.get('mem0_memories')),
            )
        await conn.commit()
    except (TimeoutError, Exception):
        logger.warning('memory sampler failed', exc_info=True)
        with contextlib.suppress(Exception):
            await conn.rollback()

    # Write queue (HTTP, separate call; tolerate either source failing).
    try:
        qstats = await asyncio.wait_for(
            get_queue_stats(http_client, config),
            timeout=_HTTP_SAMPLER_TIMEOUT_SECONDS,
        )
        pending, retry, dead = _split_queue_stats(qstats)
        await conn.execute(
            'INSERT OR REPLACE INTO queue_snapshots (ts, pending, retry, dead) VALUES (?, ?, ?, ?)',
            (now, pending, retry, dead),
        )
        await conn.commit()
    except (TimeoutError, Exception):
        logger.warning('queue sampler failed', exc_info=True)
        with contextlib.suppress(Exception):
            await conn.rollback()

    # Reconciliation (read-only SQLite — get_buffer_stats + active-agent count).
    try:
        buf = await get_buffer_stats(recon_db)
        burst = await get_burst_state(recon_db)
        active, _idle = partition_burst_state(burst)
        await conn.execute(
            'INSERT OR REPLACE INTO recon_snapshots (ts, buffered_count, active_agents) '
            'VALUES (?, ?, ?)',
            (now, buf.get('buffered_count'), len(active)),
        )
        await conn.commit()
    except Exception:
        logger.warning('recon sampler failed', exc_info=True)
        with contextlib.suppress(Exception):
            await conn.rollback()

    # Merge queue (per-project active count from runs.db events).
    try:
        for pid, db in merge_dbs:
            try:
                merges = await active_queued_merges(db)
            except Exception:
                logger.debug('merge sampler failed for %s', pid, exc_info=True)
                continue
            await conn.execute(
                'INSERT INTO merge_snapshots (ts, project_id, active_count) VALUES (?, ?, ?)',
                (now, pid, len(merges)),
            )
        await conn.commit()
    except Exception:
        logger.warning('merge sampler driver failed', exc_info=True)
        with contextlib.suppress(Exception):
            await conn.rollback()

    # Curator (queue depth via HTTP list_tickets + cap state + ticket latency).
    try:
        runs_dbs = [db for _, db in merge_dbs]
        curator = await _sample_curator(
            http_client,
            config,
            tickets_db,
            runs_dbs,
            now=now_dt,
        )
        await conn.execute(
            'INSERT OR REPLACE INTO curator_snapshots '
            '(ts, pending_total, capped_now, p50_active_ms, p90_active_ms, p99_active_ms) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (
                now,
                curator['pending_total'],
                curator['capped_now'],
                curator['p50_active_ms'],
                curator['p90_active_ms'],
                curator['p99_active_ms'],
            ),
        )
        await conn.commit()
    except Exception:
        logger.warning('curator sampler failed', exc_info=True)
        with contextlib.suppress(Exception):
            await conn.rollback()


# ---------------------------------------------------------------------------
# Downsampling — same policy as burndown (raw 10min, hourly after 7d, drop 90d).
# ---------------------------------------------------------------------------


_DOWNSAMPLE_TABLES = (
    ('orchestrator_snapshots', 'project_id'),
    ('memory_snapshots', 'project_id'),
    ('queue_snapshots', None),
    ('recon_snapshots', None),
    ('merge_snapshots', 'project_id'),
    ('curator_snapshots', None),
)


async def downsample_metrics(conn: aiosqlite.Connection) -> None:
    """Compact old metrics rows: hourly after 7 days, drop after 90 days.

    For per-project tables, partition key is (project_id, hour); for
    system-wide tables, partition key is just hour.
    """
    now = datetime.now(UTC)
    cutoff_7d = (now - timedelta(days=7)).isoformat()
    cutoff_90d = (now - timedelta(days=90)).isoformat()

    for table, pid_col in _DOWNSAMPLE_TABLES:
        rowid_col = 'id' if pid_col is not None else 'rowid'
        partition = (
            f"{pid_col}, strftime('%%Y-%%m-%%dT%%H', ts)"
            if pid_col is not None
            else "strftime('%%Y-%%m-%%dT%%H', ts)"
        )
        await conn.execute(
            f"""
            DELETE FROM {table}
            WHERE ts < ?
              AND {rowid_col} NOT IN (
                  SELECT {rowid_col} FROM (
                      SELECT {rowid_col}, ROW_NUMBER() OVER (
                          PARTITION BY {partition}
                          ORDER BY ts DESC
                      ) AS rn
                      FROM {table}
                      WHERE ts < ?
                  )
                  WHERE rn = 1
              )
            """,
            (cutoff_7d, cutoff_7d),
        )
        await conn.execute(f'DELETE FROM {table} WHERE ts < ?', (cutoff_90d,))
    await conn.commit()


# ---------------------------------------------------------------------------
# Read-side queries (read-only DbPool connections from route handlers)
# ---------------------------------------------------------------------------


_EMPTY_SERIES: dict[str, list] = {'labels': [], 'values': []}


def _coerce_series(rows: list[tuple[str, Any]]) -> dict[str, list]:
    return {
        'labels': [r[0] for r in rows],
        'values': [r[1] for r in rows],
    }


async def get_orchestrators_running_series(
    db: aiosqlite.Connection | None,
    *,
    days: int = 1,
    project_id: str | None = None,
) -> dict[str, list]:
    """Return total running-orchestrator count over time.

    When *project_id* is None, returns the sum across all projects per
    timestamp; otherwise filters to a single project.
    """
    if db is None:
        return dict(_EMPTY_SERIES)
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    try:
        if project_id is None:
            sql = (
                'SELECT ts, SUM(running_count) FROM orchestrator_snapshots '
                'WHERE ts >= ? GROUP BY ts ORDER BY ts'
            )
            params: tuple = (since,)
        else:
            sql = (
                'SELECT ts, running_count FROM orchestrator_snapshots '
                'WHERE project_id = ? AND ts >= ? ORDER BY ts'
            )
            params = (project_id, since)
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
    except Exception:
        logger.debug('orchestrator series query failed', exc_info=True)
        return dict(_EMPTY_SERIES)
    return _coerce_series([(r[0], r[1]) for r in rows])


async def get_memory_sparks(
    db: aiosqlite.Connection | None,
    *,
    days: int = 1,
) -> dict[str, dict[str, list]]:
    """Return system-wide Graphiti node + Mem0 memory sparks.

    Sums per-project rows per timestamp so the spark reflects total store
    size even when multiple projects share the dashboard.
    """
    empty = {'graphiti_nodes': dict(_EMPTY_SERIES), 'mem0_memories': dict(_EMPTY_SERIES)}
    if db is None:
        return empty
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    try:
        async with db.execute(
            'SELECT ts, SUM(graphiti_nodes), SUM(mem0_memories) '
            'FROM memory_snapshots WHERE ts >= ? GROUP BY ts ORDER BY ts',
            (since,),
        ) as cur:
            rows = await cur.fetchall()
    except Exception:
        logger.debug('memory sparks query failed', exc_info=True)
        return empty
    return {
        'graphiti_nodes': _coerce_series([(r[0], r[1]) for r in rows]),
        'mem0_memories': _coerce_series([(r[0], r[2]) for r in rows]),
    }


_24H_TOLERANCE_SECONDS = 2 * 3600  # accept rows within ±2h of the target


async def get_memory_24h_ago(
    db: aiosqlite.Connection | None,
) -> dict[str, dict]:
    """Return {project_id: {graphiti_nodes, mem0_memories}} from ~24h ago.

    Picks the row whose ``ts`` is closest to (now - 24h) per project,
    but only if it falls within ±2 hours of the target. Projects whose
    only available rows are far from the 24h mark are absent from the
    result so the UI renders ``—`` rather than a misleading zero delta.
    """
    if db is None:
        return {}
    target = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
    try:
        async with db.execute(
            """
            SELECT project_id, graphiti_nodes, mem0_memories, ts FROM (
                SELECT project_id, graphiti_nodes, mem0_memories, ts,
                       ABS(strftime('%s', ts) - strftime('%s', ?)) AS d,
                       ROW_NUMBER() OVER (
                           PARTITION BY project_id
                           ORDER BY ABS(strftime('%s', ts) - strftime('%s', ?)) ASC
                       ) AS rn
                FROM memory_snapshots
            ) WHERE rn = 1 AND d <= ?
            """,
            (target, target, _24H_TOLERANCE_SECONDS),
        ) as cur:
            rows = await cur.fetchall()
    except Exception:
        logger.debug('memory 24h-ago query failed', exc_info=True)
        return {}
    return {r[0]: {'graphiti_nodes': r[1], 'mem0_memories': r[2]} for r in rows}


async def get_queue_pending_series(
    db: aiosqlite.Connection | None,
    *,
    days: int = 1,
) -> dict[str, list]:
    """Return write-queue pending depth over time."""
    if db is None:
        return dict(_EMPTY_SERIES)
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    try:
        async with db.execute(
            'SELECT ts, pending FROM queue_snapshots WHERE ts >= ? ORDER BY ts',
            (since,),
        ) as cur:
            rows = await cur.fetchall()
    except Exception:
        logger.debug('queue series query failed', exc_info=True)
        return dict(_EMPTY_SERIES)
    return _coerce_series([(r[0], r[1]) for r in rows])


async def get_recon_sparks(
    db: aiosqlite.Connection | None,
    *,
    days: int = 1,
) -> dict[str, dict[str, list]]:
    """Return reconciliation buffered_count and active_agents over time."""
    empty = {'buffered_count': dict(_EMPTY_SERIES), 'active_agents': dict(_EMPTY_SERIES)}
    if db is None:
        return empty
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    try:
        async with db.execute(
            'SELECT ts, buffered_count, active_agents FROM recon_snapshots '
            'WHERE ts >= ? ORDER BY ts',
            (since,),
        ) as cur:
            rows = await cur.fetchall()
    except Exception:
        logger.debug('recon sparks query failed', exc_info=True)
        return empty
    return {
        'buffered_count': _coerce_series([(r[0], r[1]) for r in rows]),
        'active_agents': _coerce_series([(r[0], r[2]) for r in rows]),
    }


async def get_curator_sparks(
    db: aiosqlite.Connection | None,
    *,
    days: int = 1,
) -> dict[str, dict[str, list]]:
    """Return curator pending-count and active-latency centile sparks.

    Returns four ChartData series: 'pending', 'p50', 'p90', 'p99'.
    Each is a time series of system-wide values sampled every 10 minutes.
    Returns the empty 4-key shape when *db* is None or when no rows exist.
    """
    empty = {
        'pending': dict(_EMPTY_SERIES),
        'p50': dict(_EMPTY_SERIES),
        'p90': dict(_EMPTY_SERIES),
        'p99': dict(_EMPTY_SERIES),
    }
    if db is None:
        return empty
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    try:
        async with db.execute(
            'SELECT ts, pending_total, p50_active_ms, p90_active_ms, p99_active_ms '
            'FROM curator_snapshots WHERE ts >= ? ORDER BY ts',
            (since,),
        ) as cur:
            rows = await cur.fetchall()
    except Exception:
        logger.debug('curator sparks query failed', exc_info=True)
        return empty
    return {
        'pending': _coerce_series([(r[0], r[1]) for r in rows]),
        'p50': _coerce_series([(r[0], r[2]) for r in rows]),
        'p90': _coerce_series([(r[0], r[3]) for r in rows]),
        'p99': _coerce_series([(r[0], r[4]) for r in rows]),
    }


async def get_merge_active_series(
    db: aiosqlite.Connection | None,
    *,
    project_id: str | None = None,
    days: int = 1,
) -> dict[str, list]:
    """Return active merge-queue depth over time.

    When *project_id* is None, returns the sum across all projects per
    timestamp; otherwise filters to a single project.
    """
    if db is None:
        return dict(_EMPTY_SERIES)
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    try:
        if project_id is None:
            sql = (
                'SELECT ts, SUM(active_count) FROM merge_snapshots '
                'WHERE ts >= ? GROUP BY ts ORDER BY ts'
            )
            params: tuple = (since,)
        else:
            sql = (
                'SELECT ts, active_count FROM merge_snapshots '
                'WHERE project_id = ? AND ts >= ? ORDER BY ts'
            )
            params = (project_id, since)
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
    except Exception:
        logger.debug('merge series query failed', exc_info=True)
        return dict(_EMPTY_SERIES)
    return _coerce_series([(r[0], r[1]) for r in rows])
