"""Dark Factory dashboard — FastAPI application.

Serves a single-page React UI from ``static/redux/`` and exposes a JSON API
under ``/api/v2/dashboard/*`` that the React app polls every few seconds.
The aggregator layer in ``dashboard.data.*`` is shared by every endpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import aiosqlite
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from shared.async_sqlite_base import AsyncSqliteBase

from dashboard.config import DashboardConfig
from dashboard.data import memory as memory_data
from dashboard.data import redux_api
from dashboard.data.active_tasks import (
    _MAX_DONE_PER_PROJECT,
    collect_tasks_with_counts,
)
from dashboard.data.burndown import (
    BURNDOWN_SCHEMA,
    aggregate_burndown_projects,
    aggregate_burndown_series,
    collect_snapshot,
    downsample,
)
from dashboard.data.cap_history import (
    AccountsSummary,
    CapInterval,
    bucketise_cap_sparkline,
    compute_capped_now_and_windows,
    read_cap_intervals,
    summarize_accounts,
)
from dashboard.data.chart_utils import ChartData, trim_leading_zero_buckets
from dashboard.data.costs import (
    aggregate_account_events,
    aggregate_cost_by_account,
    aggregate_cost_by_project,
    aggregate_cost_by_role,
    aggregate_cost_summary,
    aggregate_cost_trend,
)
from dashboard.data.db import DbPool
from dashboard.data.escalations import build_escalation_queues
from dashboard.data.load import get_load_metrics
from dashboard.data.merge_halt import get_merge_halt_status
from dashboard.data.merge_queue import (
    build_per_project_merge_queue,
    enrich_merges_with_titles,
    load_task_titles,
)
from dashboard.data.metrics import (
    METRICS_SCHEMA,
    collect_metrics_snapshot,
    downsample_metrics,
    fan_out_list_tickets,
    get_curator_sparks,
    get_memory_24h_ago,
    get_memory_sparks,
    get_merge_active_series,
    get_orchestrators_running_series,
    get_queue_pending_series,
    get_recon_sparks,
)
from dashboard.data.orchestrator import discover_orchestrators
from dashboard.data.performance import (
    aggregate_completion_paths,
    aggregate_escalation_rates,
    aggregate_loop_histograms,
    aggregate_performance_history,
    aggregate_time_centiles,
)
from dashboard.data.reconciliation import (
    get_buffer_stats,
    get_burst_state,
    get_latest_verdict,
    get_recent_runs,
    get_watermarks,
    partition_burst_state,
)
from dashboard.data.scheduler import get_scheduler_snapshot
from dashboard.data.tasks import fetch_tasks
from dashboard.data.utils import safe_gather_result
from dashboard.data.write_journal import (
    get_memory_timeseries,
    get_operations_breakdown,
)

_pkg_dir = Path(__file__).parent
_redux_dir = _pkg_dir / 'static' / 'redux'
logger = logging.getLogger(__name__)

_WINDOW_DAYS: dict[str, int] = {
    '24h': 1,
    '7d': 7,
    '30d': 30,
    'all': 3650,
}

_BURNDOWN_WINDOWS: dict[str, int] = {
    '24h': 1,
    '7d': 7,
    '30d': 30,
    '90d': 90,
}


def _parse_window(query_params: Mapping[str, str], default: int = 30) -> int:
    """Parse the ``?window=`` query parameter and return the corresponding days int."""
    window = query_params.get('window', f'{default}d')
    return _WINDOW_DAYS.get(window, default)


# ---------------------------------------------------------------------------
# Task-cards TTL cache (mirrors load_task_titles pattern in merge_queue.py)
# ---------------------------------------------------------------------------

_TASK_CARDS_TTL_SECONDS = 10.0
_task_cards_cache: dict[str, tuple[float, list[dict]]] = {}


def _task_cards_cache_clear() -> None:
    """Clear the task-cards TTL cache (test hook)."""
    _task_cards_cache.clear()


async def _load_task_cards(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: str,
) -> list[dict]:
    """Return the full dashboard-shaped task list for *project_root*.

    Results are cached per project_root for ``_TASK_CARDS_TTL_SECONDS`` (~10 s)
    to avoid hammering the MCP server on every dashboard poll.  An offline
    marker or MCP failure returns ``[]`` WITHOUT writing to the cache, so a
    transient blip doesn't pin empty results for the full TTL window.

    Test hook: call ``_task_cards_cache_clear()`` to reset cache state between
    test cases.
    """
    now = time.monotonic()
    cached = _task_cards_cache.get(project_root)
    if cached is not None and (now - cached[0]) < _TASK_CARDS_TTL_SECONDS:
        return list(cached[1])

    fetched = await fetch_tasks(client, config, project_root)
    if not isinstance(fetched, list):
        # Offline or MCP failure — degrade gracefully, do NOT cache.
        return []
    _task_cards_cache[project_root] = (now, fetched)
    return list(fetched)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


_SAMPLE_INTERVAL_SECONDS = 600  # 10 minutes
_DOWNSAMPLE_INTERVAL_SECONDS = 3600  # 1 hour
_CHECKPOINT_INTERVAL_SECONDS = 3600  # 1 hour


class _BurndownStore(AsyncSqliteBase):
    """Writable WAL-mode store for the burndown snapshot collector.

    Subclasses AsyncSqliteBase so that open() applies the full Phase-3
    durability pragma triad (synchronous=FULL, wal_autocheckpoint=100,
    journal_size_limit=64 MiB) and checkpoint() is available for periodic
    use by _burndown_loop.
    """

    @property
    def _schema(self) -> str:
        return BURNDOWN_SCHEMA

    @property
    def connection(self) -> aiosqlite.Connection:
        """Public accessor for the open connection; raises RuntimeError if not opened."""
        return self._require_conn()


class _MetricsStore(AsyncSqliteBase):
    """Writable WAL-mode store for the metrics snapshot collector.

    Mirrors _BurndownStore — subclasses AsyncSqliteBase so that open()
    applies the full Phase-3 durability pragma triad and checkpoint() is
    available for periodic use by _metrics_loop.
    """

    @property
    def _schema(self) -> str:
        return METRICS_SCHEMA

    @property
    def connection(self) -> aiosqlite.Connection:
        """Public accessor for the open connection; raises RuntimeError if not opened."""
        return self._require_conn()


async def _sleep_to_aligned_tick(interval: int) -> None:
    """Sleep until the next wall-clock-aligned interval boundary.

    Avoids drift across long uptimes — without alignment, a sleep(600)
    loop slowly desynchronises from minute boundaries because each
    iteration's wakeup latency accumulates.
    """
    now = time.time()
    target = (int(now) // interval + 1) * interval
    await asyncio.sleep(max(0.0, target - now))


async def _burndown_loop(
    store: _BurndownStore,
    config: DashboardConfig,
    client: httpx.AsyncClient,
) -> None:
    """Periodically snapshot task status counts into the burndown DB."""
    conn = store.connection
    try:
        await collect_snapshot(conn, config, client)
    except Exception:
        logger.warning('Initial burndown snapshot failed', exc_info=True)
    last_downsample = 0.0
    last_checkpoint = 0.0
    while True:
        await _sleep_to_aligned_tick(_SAMPLE_INTERVAL_SECONDS)
        conn = store.connection
        try:
            await collect_snapshot(conn, config, client)
            now = time.monotonic()
            if now - last_downsample > _DOWNSAMPLE_INTERVAL_SECONDS:
                await downsample(conn)
                last_downsample = now
            if now - last_checkpoint > _CHECKPOINT_INTERVAL_SECONDS:
                try:
                    await store.checkpoint()
                except Exception:
                    logger.warning('Periodic WAL checkpoint failed (burndown)', exc_info=True)
                last_checkpoint = now
        except Exception:
            logger.warning('Burndown snapshot error', exc_info=True)


async def _metrics_loop(
    store: _MetricsStore,
    app: FastAPI,
) -> None:
    """Periodically snapshot ephemeral system metrics into metrics.db.

    Uses fresh per-cycle handles for the recon DB and per-project runs.db
    files so a stale connection cannot strand the loop. Each sampler in
    collect_metrics_snapshot has its own try/except, so one failed source
    does not poison the others.
    """

    async def _run_once() -> None:
        conn = store.connection
        config: DashboardConfig = app.state.config
        pool: DbPool = app.state.db
        http_client: httpx.AsyncClient = app.state.http_client
        recon_db = await pool.get(config.reconciliation_db)
        tickets_db = await pool.get(config.tickets_db)
        merge_dbs = await _project_scoped_dbs_labeled(
            config,
            pool,
            Path('data/orchestrator/runs.db'),
        )
        await collect_metrics_snapshot(
            conn=conn,
            config=config,
            http_client=http_client,
            recon_db=recon_db,
            merge_dbs=merge_dbs,
            tickets_db=tickets_db,
        )

    try:
        await _run_once()
    except Exception:
        logger.warning('Initial metrics snapshot failed', exc_info=True)
    last_downsample = 0.0
    last_checkpoint = 0.0
    while True:
        await _sleep_to_aligned_tick(_SAMPLE_INTERVAL_SECONDS)
        try:
            await _run_once()
            conn = store.connection
            now = time.monotonic()
            if now - last_downsample > _DOWNSAMPLE_INTERVAL_SECONDS:
                await downsample_metrics(conn)
                last_downsample = now
            if now - last_checkpoint > _CHECKPOINT_INTERVAL_SECONDS:
                try:
                    await store.checkpoint()
                except Exception:
                    logger.warning('Periodic WAL checkpoint failed (metrics)', exc_info=True)
                last_checkpoint = now
        except Exception:
            logger.warning('Metrics snapshot error', exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage shared resources: HTTP client, DB connection pool."""
    app.state.http_client = httpx.AsyncClient(follow_redirects=True)
    app.state.config = DashboardConfig.from_env()
    app.state.db = DbPool()
    app.state.start_time = time.monotonic()

    # Burndown snapshot collector (writable WAL connection with full durability triad).
    burndown_path = app.state.config.burndown_db
    burndown_store = _BurndownStore(burndown_path, busy_timeout_ms=5000)
    await burndown_store.open()
    app.state.burndown_store = burndown_store
    collector_task = asyncio.create_task(
        _burndown_loop(
            burndown_store,
            app.state.config,
            app.state.http_client,
        )
    )

    # Metrics snapshot collector (separate WAL writer with full durability triad).
    metrics_path = app.state.config.metrics_db
    metrics_store = _MetricsStore(metrics_path, busy_timeout_ms=5000)
    await metrics_store.open()
    app.state.metrics_store = metrics_store
    app.state.metrics_db_path = metrics_path  # preserved for healthz / other callers
    metrics_task = asyncio.create_task(_metrics_loop(metrics_store, app))

    yield

    for task in (collector_task, metrics_task):
        task.cancel()
    for task in (collector_task, metrics_task):
        with contextlib.suppress(asyncio.CancelledError):
            await task
    await app.state.burndown_store.close()
    await app.state.metrics_store.close()
    await app.state.db.close_all()
    await app.state.http_client.aclose()


app = FastAPI(title='Dark Factory Dashboard', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=str(_pkg_dir / 'static')), name='static')


# ---------------------------------------------------------------------------
# Page route + simple endpoints
# ---------------------------------------------------------------------------


@app.get('/')
async def index() -> FileResponse:
    """Serve the React SPA entry point."""
    return FileResponse(_redux_dir / 'index.html', media_type='text/html')


@app.get('/api/health')
async def health() -> dict:
    return {'status': 'ok'}


_THREAD_LIMIT = 50
_DB_PROBE_TIMEOUT = 5.0


@app.get('/healthz')
async def healthz(request: Request) -> JSONResponse:
    """Deep health check — detects thread leaks and unresponsive DB connections."""
    checks: dict = {}
    healthy = True

    thread_count = threading.active_count()
    threads_ok = thread_count < _THREAD_LIMIT
    checks['threads'] = {'count': thread_count, 'limit': _THREAD_LIMIT, 'ok': threads_ok}
    if not threads_ok:
        healthy = False

    pool: DbPool = request.app.state.db
    config: DashboardConfig = request.app.state.config
    checks['connections'] = {'open': pool.open_count}

    for name, db_path in [
        ('reconciliation', config.reconciliation_db),
        ('write_journal', config.write_journal_db),
        ('runs', config.runs_db),
    ]:
        conn = await pool.get(db_path)
        if conn is None:
            checks[f'db_{name}'] = 'unavailable'
            continue
        try:
            async with conn.execute('SELECT 1') as cursor:
                row = await asyncio.wait_for(cursor.fetchone(), timeout=_DB_PROBE_TIMEOUT)
            checks[f'db_{name}'] = 'ok' if row is not None else 'failed'
            if row is None:
                healthy = False
        except Exception:
            checks[f'db_{name}'] = 'timeout'
            healthy = False

    checks['uptime_seconds'] = round(time.monotonic() - request.app.state.start_time, 1)

    return JSONResponse(
        content={'status': 'healthy' if healthy else 'degraded', 'checks': checks},
        status_code=200 if healthy else 503,
    )


# ---------------------------------------------------------------------------
# Multi-project DB helpers
# ---------------------------------------------------------------------------


async def _project_scoped_dbs(
    config: DashboardConfig,
    pool: DbPool,
    rel_path: Path,
) -> list[aiosqlite.Connection | None]:
    """Return DB connections for a project-scoped file across all known roots."""
    seen: set[Path] = {config.project_root}
    paths: list[Path] = [config.project_root / rel_path]
    for root in config.known_project_roots:
        if root not in seen:
            seen.add(root)
            paths.append(root / rel_path)
    return [await pool.get(p) for p in paths]


async def _project_scoped_dbs_labeled(
    config: DashboardConfig,
    pool: DbPool,
    rel_path: Path,
) -> list[tuple[str, aiosqlite.Connection | None]]:
    """Return labeled (str(root), connection|None) pairs across all known project roots."""
    seen: set[Path] = {config.project_root}
    roots: list[Path] = [config.project_root]
    for root in config.known_project_roots:
        if root not in seen:
            seen.add(root)
            roots.append(root)
    return [(str(root), await pool.get(root / rel_path)) for root in roots]


async def _cost_dbs(
    config: DashboardConfig,
    pool: DbPool,
) -> list[aiosqlite.Connection | None]:
    """Connections for all known project runs.db files (costs and performance)."""
    return await _project_scoped_dbs(config, pool, Path('data/orchestrator/runs.db'))


async def _performance_resources(
    config: DashboardConfig,
    pool: DbPool,
) -> tuple[list[aiosqlite.Connection | None], list[Path]]:
    """Return ``(dbs, esc_dirs)`` for the performance API."""
    seen: set[Path] = {config.project_root}
    run_paths: list[Path] = [config.runs_db]
    esc_dirs: list[Path] = [config.escalations_dir]
    for root in config.known_project_roots:
        if root not in seen:
            seen.add(root)
            run_paths.append(root / 'data' / 'orchestrator' / 'runs.db')
            esc_dirs.append(root / 'data' / 'escalations')
    dbs = [await pool.get(p) for p in run_paths]
    return dbs, esc_dirs


async def _burndown_dbs(
    config: DashboardConfig,
    pool: DbPool,
) -> list[aiosqlite.Connection | None]:
    """Connections for all known project burndown.db files."""
    return await _project_scoped_dbs(config, pool, Path('data/burndown/burndown.db'))


# ---------------------------------------------------------------------------
# JSON API: /api/v2/dashboard/*
# ---------------------------------------------------------------------------


@app.get('/api/v2/dashboard/orchestrators')
async def api_orchestrators(request: Request) -> JSONResponse:
    """ORCHESTRATORS + PROJECTS for the redux dashboard."""
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    http_client: httpx.AsyncClient = request.app.state.http_client
    metrics_db = await pool.get(config.metrics_db)
    orchestrators, running_spark = await asyncio.gather(
        discover_orchestrators(http_client, config),
        get_orchestrators_running_series(metrics_db, days=1),
    )
    known_roots = [config.project_root, *config.known_project_roots]
    return JSONResponse(
        redux_api.shape_orchestrators(
            orchestrators,
            known_project_roots=known_roots,
            running_spark=running_spark,
        )
    )


@app.get('/api/v2/dashboard/tasks')
async def api_tasks(request: Request) -> JSONResponse:
    """ACTIVE_TASKS (lock state surfaced via the scheduler endpoint — see /api/v2/dashboard/scheduler).

    Each task in ACTIVE_TASKS includes a ``meta_files`` field (taskmaster
    ``metadata.files``) that is retained on the wire for debugging and tooling.
    No frontend UI reads it directly — lock display routes through D.SCHEDULER.
    """
    config: DashboardConfig = request.app.state.config
    http_client: httpx.AsyncClient = request.app.state.http_client
    # Single-pass: fetch_tasks once per project, derive both active rows and
    # done counts from the same snapshot — no second fetch_statuses round-trip.
    active, offline_projects, done_counts = await collect_tasks_with_counts(
        http_client, config,
        max_done_per_project=_MAX_DONE_PER_PROJECT,
        resolve_external=True,
    )
    return JSONResponse(
        {
            'ACTIVE_TASKS': active,
            'TASKS_OFFLINE': bool(offline_projects),
            'TASKS_OFFLINE_PROJECTS': offline_projects,
            'DONE_COUNTS': done_counts,
        }
    )


@app.get('/api/v2/dashboard/memory')
async def api_memory(request: Request) -> JSONResponse:
    """MEMORY_STATUS, including queue counts and per-project totals."""
    http_client = request.app.state.http_client
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    metrics_db = await pool.get(config.metrics_db)
    status, queue, sparks, queue_spark, delta_24h, wal = await asyncio.gather(
        memory_data.get_memory_status(http_client, config),
        memory_data.get_queue_stats(http_client, config),
        get_memory_sparks(metrics_db, days=1),
        get_queue_pending_series(metrics_db, days=1),
        get_memory_24h_ago(metrics_db),
        memory_data.get_wal_status(http_client, config),
    )
    return JSONResponse(
        redux_api.shape_memory(
            status,
            queue,
            sparks=sparks,
            queue_spark=queue_spark,
            delta_24h=delta_24h,
            wal=wal,
        )
    )


@app.get('/api/v2/dashboard/memory-graphs')
async def api_memory_graphs(request: Request) -> JSONResponse:
    """MEMORY_TIMESERIES + MEMORY_OPS_BREAKDOWN from the write journal."""
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    db = await pool.get(config.write_journal_db)
    ts_r, ops_r = await asyncio.gather(
        get_memory_timeseries(db),
        get_operations_breakdown(db),
        return_exceptions=True,
    )
    timeseries = safe_gather_result(
        ts_r, {'labels': [], 'reads': [], 'writes': []}, 'memory-graphs/ts'
    )
    ops = cast(
        ChartData, safe_gather_result(ops_r, {'labels': [], 'values': []}, 'memory-graphs/ops')
    )
    return JSONResponse(redux_api.shape_memory_graphs(timeseries, ops))


@app.get('/api/v2/dashboard/recon')
async def api_recon(request: Request) -> JSONResponse:
    """RECON_STATE + AGENTS — latest reconciliation buffer/burst/runs/verdict."""
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    db = await pool.get(config.reconciliation_db)

    metrics_db = await pool.get(config.metrics_db)
    bs_r, burst_r, wm_r, verdict_r, runs_r, sparks_r = await asyncio.gather(
        get_buffer_stats(db),
        get_burst_state(db),
        get_watermarks(db),
        get_latest_verdict(db),
        get_recent_runs(db),
        get_recon_sparks(metrics_db, days=1),
        return_exceptions=True,
    )
    buffer_stats = safe_gather_result(
        bs_r,
        {'buffered_count': 0, 'oldest_event_age_seconds': None},
        'recon/buffer',
    )
    burst_state_raw = safe_gather_result(burst_r, [], 'recon/burst')
    active_burst, _ = partition_burst_state(burst_state_raw)
    watermarks = safe_gather_result(wm_r, [], 'recon/watermarks')
    verdict = safe_gather_result(verdict_r, None, 'recon/verdict')
    runs = safe_gather_result(runs_r, [], 'recon/runs')
    sparks = safe_gather_result(
        sparks_r,
        {
            'buffered_count': {'labels': [], 'values': []},
            'active_agents': {'labels': [], 'values': []},
        },
        'recon/sparks',
    )
    return JSONResponse(
        redux_api.shape_recon(
            buffer_stats=buffer_stats,
            burst_state=active_burst,
            watermarks=watermarks,
            verdict=verdict,
            runs=runs,
            sparks=sparks,
        )
    )


@app.get('/api/v2/dashboard/merge-queue')
async def api_merge_queue(request: Request) -> JSONResponse:
    """MERGE_QUEUE — per-project depth/outcomes/latency/recent/active/speculative."""
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    days = _parse_window(request.query_params)
    hours = days * 24
    effective_now = datetime.now(UTC)

    project_dbs = await _project_scoped_dbs_labeled(
        config,
        pool,
        Path('data/orchestrator/runs.db'),
    )
    http_client: httpx.AsyncClient = request.app.state.http_client
    projects_raw, halt_status = await asyncio.gather(
        build_per_project_merge_queue(
            project_dbs,
            hours=hours,
            now=effective_now,
            recent_window_minutes=15,
        ),
        get_merge_halt_status(http_client, config.escalation_urls),
    )
    pids = list(projects_raw.keys())
    title_maps = await asyncio.gather(*(load_task_titles(http_client, config, pid) for pid in pids))
    enriched: dict[str, dict] = {}
    for pid, data, titles in zip(pids, projects_raw.values(), title_maps, strict=True):
        enriched[pid] = {
            **data,
            'depth_timeseries': trim_leading_zero_buckets(
                cast(ChartData, data['depth_timeseries'])
            ),
            'recent': enrich_merges_with_titles(data['recent'], titles),
            'active': enrich_merges_with_titles(data.get('active', []), titles),
        }
    metrics_db = await pool.get(config.metrics_db)
    active_sparks: dict[str, dict] = {}
    for pid in pids:
        active_sparks[pid] = await get_merge_active_series(metrics_db, project_id=pid, days=1)
    return JSONResponse(
        redux_api.shape_merge_queue(
            enriched,
            active_sparks=active_sparks,
            halt_status=halt_status,
        )
    )


@app.get('/api/v2/dashboard/costs')
async def api_costs(request: Request) -> JSONResponse:
    """COSTS — flat summary, per-project / per-account / per-role / trend / events."""
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    days = _parse_window(request.query_params)
    dbs = await _cost_dbs(config, pool)
    summary, by_project, by_account, by_role, trend, events = await asyncio.gather(
        aggregate_cost_summary(dbs, days=days),
        aggregate_cost_by_project(dbs, days=days),
        aggregate_cost_by_account(dbs, days=days),
        aggregate_cost_by_role(dbs, days=days),
        aggregate_cost_trend(dbs, days=days),
        aggregate_account_events(dbs, days=days),
        return_exceptions=True,
    )
    return JSONResponse(
        redux_api.shape_costs(
            summary=safe_gather_result(summary, {}, 'costs/summary'),
            by_project=safe_gather_result(by_project, {}, 'costs/by_project'),
            by_account=safe_gather_result(by_account, {}, 'costs/by_account'),
            by_role=safe_gather_result(by_role, {}, 'costs/by_role'),
            trend=safe_gather_result(trend, {}, 'costs/trend'),
            events=safe_gather_result(events, [], 'costs/events'),
        )
    )


@app.get('/api/v2/dashboard/performance')
async def api_performance(request: Request) -> JSONResponse:
    """PERFORMANCE — completion paths / escalation / loop histograms / TTC."""
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    dbs, esc_dirs = await _performance_resources(config, pool)
    days = _parse_window(request.query_params, default=7)
    paths_r, esc_r, hist_r, ttc_r, history_r = await asyncio.gather(
        aggregate_completion_paths(dbs, esc_dirs),
        aggregate_escalation_rates(dbs, esc_dirs),
        aggregate_loop_histograms(dbs),
        aggregate_time_centiles(dbs),
        aggregate_performance_history(dbs, days=days),
        return_exceptions=True,
    )
    return JSONResponse(
        redux_api.shape_performance(
            paths=safe_gather_result(paths_r, {}, 'perf/paths'),
            escalations=safe_gather_result(esc_r, {}, 'perf/escalations'),
            histograms=safe_gather_result(hist_r, {}, 'perf/histograms'),
            ttc=safe_gather_result(ttc_r, {}, 'perf/ttc'),
            history=safe_gather_result(history_r, {}, 'perf/history'),
        )
    )


# Cap str(exc) inside the 502 `detail` field to bound arbitrary-length
# exception message text from any of the caught exception types (e.g. a
# long ValueError arg, or a long httpx exception message string).  Note:
# httpx.HTTPStatusError.__str__ (inherited from BaseException) returns
# only its message arg — NOT the response body — so the cap defends
# against long message strings rather than response-body leakage.  The
# WARNING log still records the full untruncated exception text.
# Intentionally cancel-handler-scoped for now; other proxy handlers can
# adopt the same constant if they gain an equivalent truncation path.
_CANCEL_DETAIL_EXC_CHAR_LIMIT = 200


@app.post('/api/v2/dashboard/curator/cancel')
async def api_curator_cancel(request: Request) -> JSONResponse:
    """Proxy POST /curator/cancel → fused-memory cancel_ticket MCP tool.

    Accepts ``{"ticket_id": "tkt_..."}`` and returns the tool's response
    verbatim.  Status-code mapping:
      - ``error == 'not_found'``    → 404
      - network / transport errors  → 502
      - everything else             → 200
    """
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        body = None

    ticket_id = body.get('ticket_id') if isinstance(body, dict) else None
    if not isinstance(ticket_id, str) or not ticket_id.startswith('tkt_'):
        return JSONResponse(
            {
                'error': 'invalid_ticket_id',
                'detail': "ticket_id must be a string starting with 'tkt_'",
            },
            status_code=400,
        )

    config: DashboardConfig = request.app.state.config
    http_client: httpx.AsyncClient = request.app.state.http_client
    errors: list[str] = []
    # Single-Homed Ticket Invariant: each ticket lives on exactly one
    # fused-memory instance, enforced at two independent levels:
    #   • OS-wide singleton lock — fused-memory/src/fused_memory/server/
    #     main.py:_acquire_singleton_lock binds abstract Unix socket
    #     \0fused-memory-singleton; any second process exits immediately.
    #   • Per-process SQLite store — fused-memory/src/fused_memory/
    #     middleware/ticket_store.py persists tickets in <data_dir>/tickets.db
    #     with no cross-instance replication.
    # See DESIGN.md → Curator Tickets & Routing Invariant for the full
    # audit trail including the failover-only role of fused_memory_urls.
    #
    # Because only one instance can own a ticket, cancel_ticket is idempotent
    # and first-success-or-not_found semantics are correct.  A not_found from
    # the first reachable instance is authoritative — we short-circuit to 404
    # and do NOT fan out to avoid double-cancelling.  Only network/transport
    # errors trigger fallthrough to the next URL.
    #
    # If a future infra change introduces ticket replication or multi-region
    # deployment, this assumption must be re-evaluated and the loop relaxed to
    # fan-out-then-quorum.
    for url in config.fused_memory_urls:
        try:
            result = await memory_data.mcp_tool_call(
                http_client,
                url,
                'cancel_ticket',
                {'ticket_id': ticket_id},
            )
        except (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.HTTPStatusError,
            ValueError,
        ) as exc:
            logger.warning('cancel_ticket failed for %s: %s', url, exc)
            errors.append(
                f'{url}: {type(exc).__name__}: {str(exc)[:_CANCEL_DETAIL_EXC_CHAR_LIMIT]}'
            )
            # Invalidate the cached MCP session so the next caller retries
            # session initialisation — mirrors get_memory_status/get_queue_stats.
            memory_data.invalidate_session(url)
            continue
        if result.get('error') == 'not_found':
            return JSONResponse(result, status_code=404)
        return JSONResponse(result)
    return JSONResponse(
        {'error': 'fused_memory_unreachable', 'detail': '; '.join(errors)},
        status_code=502,
    )


_CURATOR_ENDPOINT_TIMEOUT_SECONDS = 5.0


@app.get('/api/v2/dashboard/curator')
async def api_curator(request: Request) -> JSONResponse:
    """CURATOR_STATE — pending tickets, latency centiles, capped sparkline."""
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    http_client: httpx.AsyncClient = request.app.state.http_client
    now = datetime.now(UTC)

    # 1 + 2 + 3. Fan-out list_tickets, latency sparks, and cap intervals all
    #              run concurrently — the fan-out dominates latency under slow
    #              MCP, so overlapping it with DB queries cuts p95 materially.
    _empty_sparks: dict[str, dict[str, list]] = {
        'pending': {'labels': [], 'values': []},
        'p50': {'labels': [], 'values': []},
        'p90': {'labels': [], 'values': []},
        'p99': {'labels': [], 'values': []},
    }

    # Each leg resolves its own DB connection inside the gather so that a
    # transient pool.get failure (OSError, etc.) degrades only that leg —
    # safe_gather_result (applied below) absorbs the failure per-leg rather
    # than propagating out of api_curator and producing a 500.  The fan-out
    # leg uses mcp_tool_call (not pool.get) and is therefore unaffected.
    async def _sparks_leg() -> dict[str, dict[str, list]]:
        md = await pool.get(config.metrics_db)
        return await get_curator_sparks(md, days=1)

    async def _intervals_leg() -> list[CapInterval]:
        cd = await _cost_dbs(config, pool)
        # days=7 matches _sample_curator look-back so caps older than 1 day are
        # not misreported as uncapped (consistent with metrics.py:323).
        return await read_cap_intervals(cd, days=7)

    fanout_r, sparks_r, intervals_r, state_r = await asyncio.gather(
        fan_out_list_tickets(
            http_client,
            config,
            timeout=_CURATOR_ENDPOINT_TIMEOUT_SECONDS,
        ),
        _sparks_leg(),
        _intervals_leg(),
        memory_data.get_curator_state(http_client, config),
        return_exceptions=True,
    )
    # fan_out_list_tickets never raises a normal Exception (per-root failures are
    # swallowed internally), but non-Exception BaseExceptions (CancelledError,
    # KeyboardInterrupt, SystemExit) intentionally propagate via safe_gather_result
    # so asyncio cancellation and process signals are not silenced during shutdown.
    raw_tickets, pending_total = safe_gather_result(fanout_r, ([], 0), 'curator/fan_out')
    curator_sparks = safe_gather_result(sparks_r, _empty_sparks, 'curator/sparks')
    intervals: list[CapInterval] = safe_gather_result(intervals_r, [], 'curator/intervals')
    curator_state: dict = safe_gather_result(state_r, {'offline': True}, 'curator/state')

    # Map raw MCP rows to the dashboard display shape (age_seconds, title, etc.).
    # Pure CPU — runs after the gather with no latency penalty.
    pending: list[dict] = []
    for r in raw_tickets:
        created_at_str = r.get('created_at', '')
        age_seconds: int | None = None
        if created_at_str:
            try:
                created_dt = datetime.fromisoformat(created_at_str)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=UTC)
                age_seconds = int((now - created_dt).total_seconds())
            except (ValueError, TypeError):
                age_seconds = None
        pending.append(
            {
                'ticket_id': r.get('ticket_id', ''),
                'project_id': r.get('project_id', ''),
                'title': r.get('candidate_title') or '',
                'files': r.get('files', []),
                'created_at': created_at_str,
                'age_seconds': age_seconds,
            }
        )

    # Extract account_count from the already-fetched gate dict so healthy accounts
    # with no recent cap events are counted as available (the denominator fix).
    account_count: int = (
        curator_state.get('account_count', 0) if isinstance(curator_state, dict) else 0
    )

    # capped_now (all-accounts-capped semantics via summarize_accounts) + capped_windows
    # (all-accounts simultaneously capped merge) come from one helper — single source
    # of truth shared with _sample_curator.  Pass account_count so healthy accounts
    # without recent cap events are NOT counted as capped (fixes the reported bug where
    # 1 wedged interval among N accounts showed "Capped").
    # See dashboard.data.cap_history for the semantics rationale.
    # The sparkline downstream (bucketise_cap_sparkline) renders the strict
    # all-accounts-capped window — a 1 in the sparkline means EVERY account was
    # blocked at that bucket.
    _empty_capped: ChartData = {'labels': [], 'values': []}
    try:
        capped_now, capped_windows = compute_capped_now_and_windows(intervals, account_count)
        capped_spark: ChartData = bucketise_cap_sparkline(capped_windows)
    except Exception:
        # Raise to WARNING so the silent capped_now=0 degrade is always visible in logs.
        # The prior code returned capped_now=1 when any interval was open-ended; that
        # conservative fallback would now give a false positive because the helper
        # no longer uses any-account semantics.  WARNING is the right visibility level.
        logger.warning('curator/capped_now_and_spark computation failed — degrading capped_now to 0', exc_info=True)
        capped_now = 0
        capped_spark = _empty_capped

    # Compute per-account availability summary in its own try/except so a failure
    # here does not affect capped_now / capped_spark (or vice versa).
    _empty_summary: AccountsSummary = {'total': 0, 'capped': 0, 'available': 0, 'capped_accounts': [], 'account_names': []}
    try:
        accounts_summary: AccountsSummary = summarize_accounts(intervals, total_accounts=account_count)
    except Exception:
        logger.debug('curator/accounts_summary computation failed', exc_info=True)
        accounts_summary = _empty_summary

    if 'paused' in curator_state:
        paused_reason: str | None = curator_state.get('paused_reason')
    else:
        if curator_state:
            logger.debug('curator/state: unexpected response shape %r', curator_state)
        paused_reason = None

    return JSONResponse(
        redux_api.shape_curator(
            pending=pending,
            curator_sparks=curator_sparks,
            capped_spark=cast(dict, capped_spark),
            capped_now=capped_now,
            paused_reason=paused_reason,
            pending_total=pending_total,
            accounts_summary=accounts_summary,
        )
    )


@app.get('/api/v2/dashboard/scheduler')
async def api_scheduler(request: Request) -> JSONResponse:
    """SCHEDULER — task contention, parks, overrides, and event sparklines."""
    config: DashboardConfig = request.app.state.config
    http_client: httpx.AsyncClient = request.app.state.http_client
    (
        rows,
        modules,
        pin_queue,
        events_by_task,
        offline_projects,
        paused_projects,
    ), snapshot_at = await get_scheduler_snapshot(http_client, config)
    return JSONResponse(
        redux_api.shape_scheduler(
            rows=rows,
            modules=modules,
            pin_queue=pin_queue,
            events_by_task=events_by_task,
            offline_projects=offline_projects,
            paused_projects=paused_projects,
            snapshot_at=snapshot_at,
        )
    )


_VALID_BOOST_TIERS = frozenset({'critical', 'high', 'medium', 'low', 'polish'})
# Dashboard boundary uses 'ttl_until' (matching the row/override shape).
# The MCP tool expects 'ttl' — translation happens in api_scheduler_clear_override.
_VALID_CLEAR_FIELDS = frozenset({'boost_tier', 'pinned', 'reserve_now', 'ttl_until'})


def _sched_fan_out_error(errors: list[str]) -> JSONResponse:
    return JSONResponse(
        {'error': 'fused_memory_unreachable', 'detail': '; '.join(errors)},
        status_code=502,
    )


async def _scheduler_proxy(
    http_client: httpx.AsyncClient,
    config: DashboardConfig,
    tool_name: str,
    args: dict,
    *,
    treat_not_found_as_404: bool = True,
) -> JSONResponse:
    """Fan out an MCP tool call over all fused_memory_urls, return first success.

    Validation and argument construction happen in the calling route handler;
    this helper is responsible only for the fan-out, transport error handling,
    and status-code mapping (502 when all URLs fail, optional 404 on not_found).
    """
    errors: list[str] = []
    for url in config.fused_memory_urls:
        try:
            result = await memory_data.mcp_tool_call(http_client, url, tool_name, args)
        except (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.HTTPStatusError,
            ValueError,
        ) as exc:
            logger.warning('%s failed for %s: %s', tool_name, url, exc)
            errors.append(f'{url}: {str(exc)[:200]}')
            memory_data.invalidate_session(url)
            continue
        # Guard the not_found mapping with isinstance: an MCP tool that
        # returns a list or None (buggy/older server) would AttributeError
        # on `.get(...)` and escape as a 500.  Defensive at the single
        # boundary that all override/clear/reorder endpoints share.
        if (
            treat_not_found_as_404
            and isinstance(result, dict)
            and result.get('error') == 'not_found'
        ):
            return JSONResponse(result, status_code=404)
        return JSONResponse(result)
    return _sched_fan_out_error(errors)


@app.post('/api/v2/dashboard/scheduler/override')
async def api_scheduler_override(request: Request) -> JSONResponse:
    """Proxy POST /scheduler/override → set_task_priority_override MCP tool."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        body = None

    if not isinstance(body, dict):
        return JSONResponse({'error': 'invalid_body'}, status_code=400)

    task_id = body.get('task_id')
    if not isinstance(task_id, str) or not task_id:
        return JSONResponse({'error': 'invalid_task_id'}, status_code=400)

    project_root = body.get('project_root')
    if not isinstance(project_root, str) or not project_root:
        return JSONResponse({'error': 'invalid_project_root'}, status_code=400)

    boost_tier = body.get('boost_tier')
    if boost_tier is not None and boost_tier not in _VALID_BOOST_TIERS:
        return JSONResponse(
            {
                'error': 'invalid_boost_tier',
                'detail': f'must be one of {sorted(_VALID_BOOST_TIERS)}',
            },
            status_code=400,
        )

    # Type-check bool fields before forwarding — forwarding 'yes'/1 verbatim
    # would surface whatever fused-memory chooses to return, inconsistent with
    # the strict validation applied to boost_tier/ttl_minutes/task_id/project_root.
    pinned = body.get('pinned')
    if pinned is not None and not isinstance(pinned, bool):
        return JSONResponse({'error': 'invalid_pinned'}, status_code=400)

    reserve_now = body.get('reserve_now')
    if reserve_now is not None and not isinstance(reserve_now, bool):
        return JSONResponse({'error': 'invalid_reserve_now'}, status_code=400)

    pin_order = body.get('pin_order')
    # Reject non-integer pin_order (bool subclasses int in Python, so exclude it)
    if pin_order is not None and not (
        isinstance(pin_order, int) and not isinstance(pin_order, bool)
    ):
        return JSONResponse(
            {'error': 'invalid_pin_order', 'detail': 'pin_order must be an integer'},
            status_code=400,
        )
    if pin_order is not None and pinned is not True:
        return JSONResponse(
            {'error': 'invalid_pin_order', 'detail': 'pin_order requires pinned=True'},
            status_code=400,
        )

    # Validate and translate ttl_minutes → ttl_secs.
    # Client (scheduler_drawer.jsx) sends ttl_minutes; the MCP tool takes ttl_secs.
    # ttl_until is not forwarded — MCP uses ttl_secs, so passing ttl_until verbatim
    # was always a no-op.
    ttl_secs: int | None = None
    if 'ttl_minutes' in body:
        ttl_raw = body['ttl_minutes']
        try:
            ttl_val = float(ttl_raw)  # raises TypeError for None
        except (TypeError, ValueError):
            return JSONResponse(
                {
                    'error': 'invalid_ttl_minutes',
                    'detail': 'ttl_minutes must be a positive number ≤ 1440',
                },
                status_code=400,
            )
        if ttl_val <= 0 or ttl_val > 1440:
            return JSONResponse(
                {
                    'error': 'invalid_ttl_minutes',
                    'detail': 'ttl_minutes must be a positive number ≤ 1440',
                },
                status_code=400,
            )
        ttl_secs = int(round(ttl_val * 60))

    args: dict = {'task_id': task_id, 'project_root': project_root}
    # Forward recognised optional fields; exclude ttl_until (not a MCP param)
    # and ttl_minutes (translated above into ttl_secs).
    for key in ('boost_tier', 'pinned', 'pin_order', 'reserve_now'):
        if key in body:
            args[key] = body[key]
    if ttl_secs is not None:
        args['ttl_secs'] = ttl_secs

    config: DashboardConfig = request.app.state.config
    http_client: httpx.AsyncClient = request.app.state.http_client
    return await _scheduler_proxy(http_client, config, 'set_task_priority_override', args)


@app.post('/api/v2/dashboard/scheduler/clear-override')
async def api_scheduler_clear_override(request: Request) -> JSONResponse:
    """Proxy POST /scheduler/clear-override → clear_task_priority_override MCP tool."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        body = None

    if not isinstance(body, dict):
        return JSONResponse({'error': 'invalid_body'}, status_code=400)

    task_id = body.get('task_id')
    if not isinstance(task_id, str) or not task_id:
        return JSONResponse({'error': 'invalid_task_id'}, status_code=400)

    project_root = body.get('project_root')
    if not isinstance(project_root, str) or not project_root:
        return JSONResponse({'error': 'invalid_project_root'}, status_code=400)

    fields = body.get('fields')
    if fields is not None:
        if not isinstance(fields, list):
            return JSONResponse(
                {'error': 'invalid_fields', 'detail': 'fields must be a list'}, status_code=400
            )
        invalid = [f for f in fields if f not in _VALID_CLEAR_FIELDS]
        if invalid:
            return JSONResponse(
                {'error': 'invalid_fields', 'detail': f'unknown fields: {invalid}'},
                status_code=400,
            )

    args: dict = {'task_id': task_id, 'project_root': project_root}
    if fields is not None:
        # Translate 'ttl_until' → 'ttl': the dashboard boundary speaks 'ttl_until'
        # (matching the row shape), but the MCP tool's wire format uses 'ttl'.
        args['fields'] = ['ttl' if f == 'ttl_until' else f for f in fields]

    config: DashboardConfig = request.app.state.config
    http_client: httpx.AsyncClient = request.app.state.http_client
    return await _scheduler_proxy(http_client, config, 'clear_task_priority_override', args)


@app.post('/api/v2/dashboard/scheduler/reorder-pin-queue')
async def api_scheduler_reorder_pin_queue(request: Request) -> JSONResponse:
    """Proxy POST /scheduler/reorder-pin-queue → reorder_pin_queue MCP tool."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        body = None

    if not isinstance(body, dict):
        return JSONResponse({'error': 'invalid_body'}, status_code=400)

    task_ids = body.get('task_ids')
    if not isinstance(task_ids, list):
        return JSONResponse(
            {'error': 'invalid_task_ids', 'detail': 'task_ids must be a list'}, status_code=400
        )

    # Validate each element at the boundary — the override endpoint enforces
    # str/non-empty for task_id, and reorder should match.  Without this a
    # request like task_ids=[None, {}, []] would proceed past the duplicate
    # check (str() of dissimilar non-string objects rarely collides) and the
    # failure would surface from fused-memory rather than failing fast here.
    if not all(isinstance(t, str) and t for t in task_ids):
        return JSONResponse(
            {'error': 'invalid_task_ids', 'detail': 'task_ids must be non-empty strings'},
            status_code=400,
        )

    project_root = body.get('project_root')
    if not isinstance(project_root, str) or not project_root:
        return JSONResponse({'error': 'invalid_project_root'}, status_code=400)

    if len(task_ids) != len(set(task_ids)):
        return JSONResponse({'error': 'duplicate_task_ids'}, status_code=400)

    config: DashboardConfig = request.app.state.config
    http_client: httpx.AsyncClient = request.app.state.http_client
    # reorder_pin_queue forwards MCP results verbatim (no not_found→404 mapping),
    # so the React layer can toast on 'mismatch' without an extra round-trip.
    return await _scheduler_proxy(
        http_client,
        config,
        'reorder_pin_queue',
        {'task_ids': task_ids, 'project_root': project_root},
        treat_not_found_as_404=False,
    )


@app.get('/api/v2/dashboard/burndown')
async def api_burndown(request: Request) -> JSONResponse:
    """BURNDOWN + BURNDOWN_BY_PROJECT — per-project status time series."""
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    dbs = await _burndown_dbs(config, pool)
    window_raw = request.query_params.get('window', '30d')
    days = _BURNDOWN_WINDOWS.get(window_raw, 30)

    try:
        projects = await aggregate_burndown_projects(dbs)
        per_pid = await asyncio.gather(
            *(aggregate_burndown_series(dbs, pid, days=days) for pid in projects)
        )
        series: dict[str, dict] = dict(zip(projects, per_pid, strict=True))
    except Exception:
        logger.warning('Error fetching burndown data', exc_info=True)
        series = {}
    return JSONResponse(redux_api.shape_burndown(series))


@app.get('/api/load')
async def api_load(request: Request) -> JSONResponse:
    """Host load metrics — latest value + 60-sample sparkline for each of the 9 known metrics."""
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    db = await pool.get(config.load_samples_db)
    result = await get_load_metrics(db)
    return JSONResponse(result)


@app.get('/api/v2/dashboard/escalations')
async def api_escalations(request: Request) -> JSONResponse:
    """ESCALATIONS — per-project escalation queues with resolved task cards."""
    config: DashboardConfig = request.app.state.config
    http_client: httpx.AsyncClient = request.app.state.http_client

    queues = build_escalation_queues(config)

    # Derive fetch roots from orchestrator subsection ids — these are already
    # str(root) and are de-duped by build_escalation_queues.  Keying task_maps
    # by subsection id means the shaper can match by id directly.
    root_ids = [s['id'] for s in queues.get('subsections') or [] if s.get('kind') == 'orchestrator']

    results = await asyncio.gather(*(_load_task_cards(http_client, config, rid) for rid in root_ids))
    task_maps: dict[str, list[dict]] = {rid: tasks for rid, tasks in zip(root_ids, results, strict=True)}

    return JSONResponse(redux_api.shape_escalations(queues, task_maps))


# Tests import these helpers directly.
__all__: Sequence[str] = (
    'app',
    'lifespan',
    '_project_scoped_dbs',
    '_project_scoped_dbs_labeled',
    '_cost_dbs',
    '_performance_resources',
    '_burndown_dbs',
    '_task_cards_cache_clear',
    '_load_task_cards',
)
