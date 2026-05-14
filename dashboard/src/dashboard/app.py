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

from dashboard.config import DashboardConfig
from dashboard.data import memory as memory_data
from dashboard.data import redux_api
from dashboard.data.active_tasks import collect_active_tasks
from dashboard.data.burndown import (
    BURNDOWN_SCHEMA,
    aggregate_burndown_projects,
    aggregate_burndown_series,
    collect_snapshot,
    downsample,
)
from dashboard.data.cap_history import (
    CapInterval,
    bucketise_cap_sparkline,
    merge_all_accounts_capped,
    read_cap_intervals,
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
# Lifespan
# ---------------------------------------------------------------------------


_SAMPLE_INTERVAL_SECONDS = 600  # 10 minutes
_DOWNSAMPLE_INTERVAL_SECONDS = 3600  # 1 hour


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
    conn: aiosqlite.Connection,
    config: DashboardConfig,
    client: httpx.AsyncClient,
) -> None:
    """Periodically snapshot task status counts into the burndown DB."""
    try:
        await collect_snapshot(conn, config, client)
    except Exception:
        logger.warning('Initial burndown snapshot failed', exc_info=True)
    last_downsample = 0.0
    while True:
        await _sleep_to_aligned_tick(_SAMPLE_INTERVAL_SECONDS)
        try:
            await collect_snapshot(conn, config, client)
            now = time.monotonic()
            if now - last_downsample > _DOWNSAMPLE_INTERVAL_SECONDS:
                await downsample(conn)
                last_downsample = now
        except Exception:
            logger.warning('Burndown snapshot error', exc_info=True)


async def _metrics_loop(
    conn: aiosqlite.Connection,
    app: FastAPI,
) -> None:
    """Periodically snapshot ephemeral system metrics into metrics.db.

    Uses fresh per-cycle handles for the recon DB and per-project runs.db
    files so a stale connection cannot strand the loop. Each sampler in
    collect_metrics_snapshot has its own try/except, so one failed source
    does not poison the others.
    """
    async def _run_once() -> None:
        config: DashboardConfig = app.state.config
        pool: DbPool = app.state.db
        http_client: httpx.AsyncClient = app.state.http_client
        recon_db = await pool.get(config.reconciliation_db)
        tickets_db = await pool.get(config.tickets_db)
        merge_dbs = await _project_scoped_dbs_labeled(
            config, pool, Path('data/orchestrator/runs.db'),
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
    while True:
        await _sleep_to_aligned_tick(_SAMPLE_INTERVAL_SECONDS)
        try:
            await _run_once()
            now = time.monotonic()
            if now - last_downsample > _DOWNSAMPLE_INTERVAL_SECONDS:
                await downsample_metrics(conn)
                last_downsample = now
        except Exception:
            logger.warning('Metrics snapshot error', exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage shared resources: HTTP client, DB connection pool."""
    app.state.http_client = httpx.AsyncClient(follow_redirects=True)
    app.state.config = DashboardConfig.from_env()
    app.state.db = DbPool()
    app.state.start_time = time.monotonic()

    # Burndown snapshot collector (writable connection, WAL mode).
    burndown_path = app.state.config.burndown_db
    burndown_path.parent.mkdir(parents=True, exist_ok=True)
    burndown_conn = await aiosqlite.connect(str(burndown_path))
    await burndown_conn.execute('PRAGMA journal_mode=WAL')
    await burndown_conn.executescript(BURNDOWN_SCHEMA)
    await burndown_conn.commit()
    collector_task = asyncio.create_task(_burndown_loop(
        burndown_conn, app.state.config, app.state.http_client,
    ))

    # Metrics snapshot collector (separate WAL writer for sparse-history signals).
    metrics_path = app.state.config.metrics_db
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_conn = await aiosqlite.connect(str(metrics_path))
    await metrics_conn.execute('PRAGMA journal_mode=WAL')
    await metrics_conn.executescript(METRICS_SCHEMA)
    await metrics_conn.commit()
    metrics_task = asyncio.create_task(_metrics_loop(metrics_conn, app))
    app.state.metrics_db_path = metrics_path

    yield

    for task in (collector_task, metrics_task):
        task.cancel()
    for task in (collector_task, metrics_task):
        with contextlib.suppress(asyncio.CancelledError):
            await task
    await burndown_conn.close()
    await metrics_conn.close()
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
    return JSONResponse(redux_api.shape_orchestrators(
        orchestrators,
        known_project_roots=known_roots,
        running_spark=running_spark,
    ))


@app.get('/api/v2/dashboard/tasks')
async def api_tasks(request: Request) -> JSONResponse:
    """ACTIVE_TASKS + derived FILE_LOCKS."""
    config: DashboardConfig = request.app.state.config
    http_client: httpx.AsyncClient = request.app.state.http_client
    active, locks, offline_projects = await collect_active_tasks(http_client, config)
    return JSONResponse({
        'ACTIVE_TASKS': active,
        'FILE_LOCKS': locks,
        'TASKS_OFFLINE': bool(offline_projects),
        'TASKS_OFFLINE_PROJECTS': offline_projects,
    })


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
    return JSONResponse(redux_api.shape_memory(
        status, queue,
        sparks=sparks, queue_spark=queue_spark, delta_24h=delta_24h,
        wal=wal,
    ))


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
    timeseries = safe_gather_result(ts_r, {'labels': [], 'reads': [], 'writes': []}, 'memory-graphs/ts')
    ops = cast(ChartData, safe_gather_result(ops_r, {'labels': [], 'values': []}, 'memory-graphs/ops'))
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
        bs_r, {'buffered_count': 0, 'oldest_event_age_seconds': None}, 'recon/buffer',
    )
    burst_state_raw = safe_gather_result(burst_r, [], 'recon/burst')
    active_burst, _ = partition_burst_state(burst_state_raw)
    watermarks = safe_gather_result(wm_r, [], 'recon/watermarks')
    verdict = safe_gather_result(verdict_r, None, 'recon/verdict')
    runs = safe_gather_result(runs_r, [], 'recon/runs')
    sparks = safe_gather_result(
        sparks_r,
        {'buffered_count': {'labels': [], 'values': []},
         'active_agents': {'labels': [], 'values': []}},
        'recon/sparks',
    )
    return JSONResponse(redux_api.shape_recon(
        buffer_stats=buffer_stats, burst_state=active_burst,
        watermarks=watermarks, verdict=verdict, runs=runs, sparks=sparks,
    ))


@app.get('/api/v2/dashboard/merge-queue')
async def api_merge_queue(request: Request) -> JSONResponse:
    """MERGE_QUEUE — per-project depth/outcomes/latency/recent/active/speculative."""
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    days = _parse_window(request.query_params)
    hours = days * 24
    effective_now = datetime.now(UTC)

    project_dbs = await _project_scoped_dbs_labeled(
        config, pool, Path('data/orchestrator/runs.db'),
    )
    http_client: httpx.AsyncClient = request.app.state.http_client
    projects_raw, halt_status = await asyncio.gather(
        build_per_project_merge_queue(
            project_dbs, hours=hours, now=effective_now, recent_window_minutes=15,
        ),
        get_merge_halt_status(http_client, config.escalation_urls),
    )
    pids = list(projects_raw.keys())
    title_maps = await asyncio.gather(*(
        load_task_titles(http_client, config, pid)
        for pid in pids
    ))
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
    return JSONResponse(redux_api.shape_merge_queue(
        enriched, active_sparks=active_sparks, halt_status=halt_status,
    ))


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
    return JSONResponse(redux_api.shape_costs(
        summary=safe_gather_result(summary, {}, 'costs/summary'),
        by_project=safe_gather_result(by_project, {}, 'costs/by_project'),
        by_account=safe_gather_result(by_account, {}, 'costs/by_account'),
        by_role=safe_gather_result(by_role, {}, 'costs/by_role'),
        trend=safe_gather_result(trend, {}, 'costs/trend'),
        events=safe_gather_result(events, [], 'costs/events'),
    ))


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
    return JSONResponse(redux_api.shape_performance(
        paths=safe_gather_result(paths_r, {}, 'perf/paths'),
        escalations=safe_gather_result(esc_r, {}, 'perf/escalations'),
        histograms=safe_gather_result(hist_r, {}, 'perf/histograms'),
        ttc=safe_gather_result(ttc_r, {}, 'perf/ttc'),
        history=safe_gather_result(history_r, {}, 'perf/history'),
    ))


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
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        body = None

    ticket_id = body.get('ticket_id') if isinstance(body, dict) else None
    if not isinstance(ticket_id, str) or not ticket_id.startswith('tkt_'):
        return JSONResponse(
            {'error': 'invalid_ticket_id',
             'detail': "ticket_id must be a string starting with 'tkt_'"},
            status_code=400,
        )

    config: DashboardConfig = request.app.state.config
    http_client: httpx.AsyncClient = request.app.state.http_client
    errors: list[str] = []
    # Assumption: cancel_ticket is idempotent and each ticket lives on exactly
    # one fused-memory instance.  First-success-or-not_found semantics are
    # correct; we do NOT fan out past a successful response to avoid
    # double-cancelling.  Only network/transport errors trigger fallthrough.
    for url in config.fused_memory_urls:
        try:
            result = await memory_data.mcp_tool_call(
                http_client, url, 'cancel_ticket', {'ticket_id': ticket_id},
            )
        except (httpx.ConnectError, httpx.TimeoutException,
                httpx.HTTPStatusError, ValueError) as exc:
            logger.warning('cancel_ticket failed for %s: %s', url, exc)
            errors.append(f'{url}: {type(exc).__name__}: {exc}')
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

    # 1. Fan-out list_tickets across all project roots (de-duped, first-success-per-root).
    #    limit=2000 matches _sample_curator so the live count and the stored
    #    pending_total spark stay consistent.  fan_out_list_tickets logs a WARNING
    #    when any root's count saturates at the limit.
    raw_tickets, pending_total = await fan_out_list_tickets(
        http_client, config,
        limit=2000,
        timeout=_CURATOR_ENDPOINT_TIMEOUT_SECONDS,
    )
    # Map raw MCP rows to the dashboard display shape (age_seconds, title, etc.).
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
        pending.append({
            'ticket_id': r.get('ticket_id', ''),
            'project_id': r.get('project_id', ''),
            'title': r.get('candidate_title') or '',
            'files': r.get('files', []),
            'created_at': created_at_str,
            'age_seconds': age_seconds,
        })

    # 2 + 3. Latency sparks and cap intervals — independent queries, run concurrently.
    _empty_sparks: dict[str, dict[str, list]] = {
        'pending': {'labels': [], 'values': []},
        'p50': {'labels': [], 'values': []},
        'p90': {'labels': [], 'values': []},
        'p99': {'labels': [], 'values': []},
    }
    _empty_capped: ChartData = {'labels': [], 'values': []}

    metrics_db = await pool.get(config.metrics_db)
    cost_dbs_raw = await _cost_dbs(config, pool)

    sparks_r, intervals_r = await asyncio.gather(
        get_curator_sparks(metrics_db, days=1),
        read_cap_intervals(cost_dbs_raw, days=1),
        return_exceptions=True,
    )
    curator_sparks = safe_gather_result(sparks_r, _empty_sparks, 'curator/sparks')
    intervals: list[CapInterval] = safe_gather_result(intervals_r, [], 'curator/intervals')

    # capped_now: "any account currently capped" — one open-ended interval means
    # the curator is effectively blocked right now.  Uses "any" semantics to avoid
    # the false-negative that would arise if some accounts had no events in the
    # look-back window (those accounts produce no intervals, so ALL-semantics would
    # falsely report uncapped).  Matches _sample_curator's point-in-time check.
    capped_now = 1 if any(iv.end is None for iv in intervals) else 0

    # capped_spark: "all accounts simultaneously capped" via merge_all_accounts_capped.
    # Intentionally stricter than capped_now — the sparkline shows periods of total
    # saturation (every account blocked at once), while capped_now is a live badge
    # that fires on the first open-ended interval.  With ≥2 accounts, capped_now=1
    # while the sparkline stays flat is expected behaviour, not a bug.
    try:
        account_names = sorted({iv.account_name for iv in intervals})
        capped_windows = merge_all_accounts_capped(intervals, account_names)
        capped_spark: ChartData = bucketise_cap_sparkline(capped_windows)
    except Exception:
        logger.debug('curator/capped_spark computation failed', exc_info=True)
        capped_spark = _empty_capped

    return JSONResponse(redux_api.shape_curator(
        pending=pending,
        curator_sparks=curator_sparks,
        capped_spark=cast(dict, capped_spark),
        capped_now=capped_now,
        # paused_reason: the usage_gate's pause reason lives only in-memory and
        # is not exposed by any current MCP tool.  Returns None until a follow-up
        # task adds a snapshot column or a new get_curator_state MCP tool.
        paused_reason=None,
        pending_total=pending_total,
    ))


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


# Tests import these helpers directly.
__all__: Sequence[str] = (
    'app',
    'lifespan',
    '_project_scoped_dbs',
    '_project_scoped_dbs_labeled',
    '_cost_dbs',
    '_performance_resources',
    '_burndown_dbs',
)
