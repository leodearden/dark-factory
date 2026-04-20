"""Dark Factory dashboard — FastAPI application."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar, cast

import aiosqlite
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dashboard.config import DashboardConfig
from dashboard.data import memory as memory_data
from dashboard.data.burndown import (
    BURNDOWN_SCHEMA,
    aggregate_burndown_projects,
    aggregate_burndown_series,
    collect_snapshot,
    downsample,
)
from dashboard.data.chart_utils import ChartData, group_top_n, trim_leading_zero_buckets
from dashboard.data.costs import (
    aggregate_account_events,
    aggregate_cost_by_account,
    aggregate_cost_by_project,
    aggregate_cost_by_role,
    aggregate_cost_summary,
    aggregate_cost_trend,
    aggregate_run_cost_breakdown,
)
from dashboard.data.db import DbPool
from dashboard.data.merge_queue import (
    aggregate_latency_stats,
    aggregate_outcome_distribution,
    aggregate_queue_depth_timeseries,
    aggregate_recent_merges,
    aggregate_speculative_stats,
)
from dashboard.data.orchestrator import discover_orchestrators
from dashboard.data.performance import (
    aggregate_completion_paths,
    aggregate_escalation_rates,
    aggregate_loop_histograms,
    aggregate_time_centiles,
)
from dashboard.data.reconciliation import (
    get_buffer_stats,
    get_burst_state,
    get_journal_entries,
    get_last_attempted_run,
    get_latest_verdict,
    get_recent_runs,
    get_watermarks,
)
from dashboard.data.utils import parse_utc
from dashboard.data.write_journal import (
    get_agent_breakdown,
    get_memory_timeseries,
    get_operations_breakdown,
)

_pkg_dir = Path(__file__).parent
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


def _parse_window(request: Any) -> int:
    """Parse the ``window`` query parameter and return the corresponding days int.

    Valid values: '24h' → 1, '7d' → 7, '30d' → 30, 'all' → 3650.
    Missing or unknown values default to 30.
    """
    window = request.query_params.get('window', '30d')
    return _WINDOW_DAYS.get(window, 30)


_T = TypeVar('_T')


def _safe_gather_result(result: object, default: _T, label: str) -> _T:
    """Return *default* if *result* is an exception, otherwise return *result*.

    Used with ``asyncio.gather(return_exceptions=True)`` to inspect each result
    independently, preserving sibling results when one coroutine fails.

    Non-Exception BaseExceptions (CancelledError, KeyboardInterrupt, SystemExit)
    are re-raised so that asyncio cancellation and process signals propagate
    correctly during shutdown and disconnect.
    """
    if isinstance(result, BaseException) and not isinstance(result, Exception):
        raise result
    if isinstance(result, Exception):
        logger.warning('Error fetching %s data: %s', label, result)
        return default
    return cast(_T, result)

templates = Jinja2Templates(directory=str(_pkg_dir / 'templates'))


def timeago(value: str | None) -> str:
    """Convert an ISO timestamp string to a human-friendly relative time."""
    if value is None:
        return 'never'
    try:
        parsed = parse_utc(value).astimezone(UTC)
    except (ValueError, TypeError):
        return 'never'
    delta = datetime.now(UTC) - parsed
    total_seconds = max(delta.total_seconds(), 0)
    total_minutes = int(total_seconds // 60)
    if total_minutes >= 1440:
        return f'{total_minutes // 1440}d ago'
    if total_minutes >= 60:
        return f'{total_minutes // 60}h ago'
    if total_minutes == 0:
        return 'just now'
    return f'{total_minutes}m ago'


templates.env.filters['timeago'] = timeago

_TRIGGER_TYPE_MAP = {
    'max_staleness': 'staleness',
    'buffer_size': 'buffer',
}


def format_trigger(value: str | None) -> str:
    """Format a trigger_reason string for human-friendly display.

    Examples:
        None          -> ''
        'manual'      -> 'manual'
        'quiescent:6' -> 'quiescent (6)'
        'buffer_size:10' -> 'buffer (10)'
        'max_staleness:<ISO>' -> 'staleness (<relative>)'
        'foo:bar'     -> 'foo (bar)'
    """
    if value is None:
        return ''
    if ':' not in value:
        return value
    trigger_type, _, trigger_value = value.partition(':')
    display_type = _TRIGGER_TYPE_MAP.get(trigger_type, trigger_type)
    if trigger_type == 'max_staleness':
        trigger_value = timeago(trigger_value)
    return f'{display_type} ({trigger_value})'


templates.env.filters['format_trigger'] = format_trigger


def format_duration_ms(value: int | float | None) -> str:
    """Format a duration in milliseconds to a human-friendly string."""
    if value is None or value <= 0:
        return '-'
    seconds = value / 1000
    if seconds < 60:
        return f'{seconds:.0f}s'
    minutes = seconds / 60
    if minutes < 60:
        return f'{minutes:.0f}m'
    hours = minutes / 60
    return f'{hours:.1f}h'


templates.env.filters['format_duration_ms'] = format_duration_ms


def format_duration(value: Any) -> str:
    """Format a duration in seconds to a human-readable compound string.

    Tiers:
        < 60s   → 'Xs'      (e.g. '45s')
        < 3600s → 'Xm Ys'   (e.g. '10m 0s')
        ≥ 3600s → 'Xh Ym'   (e.g. '17h 25m')
    """
    try:
        total = int(round(value))
    except (TypeError, ValueError, OverflowError):
        return '-'
    if total <= 0:
        return '-'
    if total < 60:
        return f'{total}s'
    if total < 3600:
        minutes = total // 60
        seconds = total % 60
        return f'{minutes}m {seconds}s'
    hours = total // 3600
    minutes = (total % 3600) // 60
    return f'{hours}h {minutes}m'


templates.env.filters['format_duration'] = format_duration


def project_name(value: str | None) -> str:
    """Extract a display-friendly project name from a project_id string.

    If the value looks like a filesystem path (contains '/'), return the
    last path component (basename). Otherwise return the value as-is.

    Examples:
        None                          -> ''
        ''                            -> ''
        'dark_factory'                -> 'dark_factory'
        '/home/leo/src/dark-factory'  -> 'dark-factory'
        '/home/leo/src/dark-factory/' -> 'dark-factory'
    """
    if not value:
        return ''
    if '/' not in value:
        return value
    return value.rstrip('/').rsplit('/', 1)[-1]


templates.env.filters['project_name'] = project_name


def css_id(value: str | None) -> str:
    """Convert a string to a safe CSS/HTML ID by replacing non-alphanumeric characters.

    Rules:
    - Replace any character that is not alphanumeric or underscore with an underscore.
    - Collapse consecutive underscores into a single underscore.
    - Strip leading and trailing underscores.
    - If the result is empty, return 'unknown'.

    Examples:
        None                          -> 'unknown'
        ''                            -> 'unknown'
        'dark_factory'                -> 'dark_factory'
        '/home/leo/src/dark-factory'  -> 'home_leo_src_dark_factory'
        'my-project.name here'        -> 'my_project_name_here'
        '---'                         -> 'unknown'
    """
    if not value:
        return 'unknown'
    result = re.sub(r'[^a-zA-Z0-9_]', '_', value)
    result = re.sub(r'_+', '_', result)
    result = result.strip('_')
    return result or 'unknown'


templates.env.filters['css_id'] = css_id


async def _burndown_loop(
    conn: aiosqlite.Connection,
    config: DashboardConfig,
) -> None:
    """Periodically snapshot task status counts into the burndown DB."""
    try:
        await collect_snapshot(conn, config)
    except Exception:
        logger.warning('Initial burndown snapshot failed', exc_info=True)
    last_downsample = 0.0
    while True:
        await asyncio.sleep(600)  # 10 minutes
        try:
            await collect_snapshot(conn, config)
            now = time.monotonic()
            if now - last_downsample > 3600:
                await downsample(conn)
                last_downsample = now
        except Exception:
            logger.warning('Burndown snapshot error', exc_info=True)


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
    collector_task = asyncio.create_task(_burndown_loop(burndown_conn, app.state.config))

    yield

    collector_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await collector_task
    await burndown_conn.close()
    await app.state.db.close_all()
    await app.state.http_client.aclose()


app = FastAPI(title='Dark Factory Dashboard', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=str(_pkg_dir / 'static')), name='static')


@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse(request, 'index.html')


@app.get('/costs')
async def costs(request: Request):
    window_raw = request.query_params.get('window', '30d')
    window = window_raw if window_raw in _WINDOW_DAYS else '30d'
    return templates.TemplateResponse(request, 'costs.html', context={'window': window})


@app.get('/burndown')
async def burndown(request: Request):
    window_raw = request.query_params.get('window', '30d')
    window = window_raw if window_raw in _BURNDOWN_WINDOWS else '30d'
    return templates.TemplateResponse(request, 'burndown.html', context={'window': window})


@app.get('/api/health')
async def health():
    return {'status': 'ok'}


_THREAD_LIMIT = 50
_DB_PROBE_TIMEOUT = 5.0


@app.get('/healthz')
async def healthz(request: Request):
    """Deep health check — detects thread leaks and unresponsive DB connections."""
    checks: dict = {}
    healthy = True

    # 1. Thread count
    thread_count = threading.active_count()
    threads_ok = thread_count < _THREAD_LIMIT
    checks['threads'] = {'count': thread_count, 'limit': _THREAD_LIMIT, 'ok': threads_ok}
    if not threads_ok:
        healthy = False

    # 2. DB connectivity probe
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

    # 3. Uptime
    checks['uptime_seconds'] = round(time.monotonic() - request.app.state.start_time, 1)

    return JSONResponse(
        content={'status': 'healthy' if healthy else 'degraded', 'checks': checks},
        status_code=200 if healthy else 503,
    )


@app.get('/partials/memory')
async def memory_partial(request: Request):
    http_client = request.app.state.http_client
    config = request.app.state.config
    status = await memory_data.get_memory_status(http_client, config)
    queue = await memory_data.get_queue_stats(http_client, config)
    return templates.TemplateResponse(
        request, 'partials/memory.html', context={'status': status, 'queue': queue}
    )


@app.get('/partials/memory-graphs')
async def memory_graphs_partial(request: Request):
    """Render the memory graphs partial (htmx fragment)."""
    config = request.app.state.config
    pool: DbPool = request.app.state.db
    db = await pool.get(config.write_journal_db)
    ts_r, ops_r, agents_r = await asyncio.gather(
        get_memory_timeseries(db),
        get_operations_breakdown(db),
        get_agent_breakdown(db),
        return_exceptions=True,
    )
    timeseries = _safe_gather_result(ts_r, {'labels': [], 'reads': [], 'writes': []}, 'timeseries')
    operations: ChartData = cast(ChartData, _safe_gather_result(ops_r, {'labels': [], 'values': []}, 'operations'))
    agents: ChartData = cast(ChartData, _safe_gather_result(agents_r, {'labels': [], 'values': []}, 'agents'))
    return templates.TemplateResponse(
        request, 'partials/memory_graphs.html',
        context={
            'timeseries': timeseries,
            'operations': group_top_n(operations),
            'agents': group_top_n(agents),
        },
    )


@app.get('/partials/recon')
async def partials_recon(request: Request):
    """Render the reconciliation panel partial (htmx fragment)."""
    config = request.app.state.config
    pool: DbPool = request.app.state.db
    db = await pool.get(config.reconciliation_db)

    bs_r, burst_r, wm_r, verdict_r, runs_r, la_r = await asyncio.gather(
        get_buffer_stats(db),
        get_burst_state(db),
        get_watermarks(db),
        get_latest_verdict(db),
        get_recent_runs(db),
        get_last_attempted_run(db),
        return_exceptions=True,
    )
    buffer_stats = _safe_gather_result(
        bs_r, {'buffered_count': 0, 'oldest_event_age_seconds': None}, 'buffer_stats',
    )
    burst_state = _safe_gather_result(burst_r, [], 'burst_state')
    watermarks_list = _safe_gather_result(wm_r, [], 'watermarks')
    verdict = _safe_gather_result(verdict_r, None, 'latest_verdict')
    runs = _safe_gather_result(runs_r, [], 'recent_runs')
    last_attempted_map = _safe_gather_result(la_r, {}, 'last_attempted_run')

    # Build per-project view: merge watermarks + last attempted run
    projects: dict[str, dict] = {}
    for wm in watermarks_list:
        pid = wm['project_id']
        projects[pid] = {
            'watermarks': wm,
            'last_attempted': last_attempted_map.get(pid),
        }
    # Include projects that have runs but no watermarks yet
    for pid, la in last_attempted_map.items():
        if pid not in projects:
            projects[pid] = {'watermarks': {}, 'last_attempted': la}

    # Determine if runs span multiple project_ids (for column display)
    run_project_ids = {r['project_id'] for r in runs}

    return templates.TemplateResponse(
        request, 'partials/recon.html',
        context={
            'buffer_stats': buffer_stats,
            'burst_state': burst_state,
            'projects': projects,
            'verdict': verdict,
            'runs': runs,
            'multi_project_runs': len(run_project_ids) > 1,
        },
    )


@app.get('/partials/recon/run/{run_id}')
async def partials_recon_run_detail(request: Request, run_id: str):
    """Render journal entries for a specific reconciliation run (htmx fragment)."""
    config = request.app.state.config
    pool: DbPool = request.app.state.db
    db = await pool.get(config.reconciliation_db)
    entries = await get_journal_entries(db, run_id)
    return templates.TemplateResponse(
        request, 'partials/recon_run_detail.html',
        context={'entries': entries},
    )


@app.get('/partials/orchestrators')
async def partials_orchestrators(request: Request):
    """Render the orchestrators panel partial (htmx fragment)."""
    config = request.app.state.config
    orchestrators = await asyncio.to_thread(discover_orchestrators, config)
    return templates.TemplateResponse(
        request, 'partials/orchestrators.html', context={'orchestrators': orchestrators}
    )


@app.get('/partials/performance')
async def partials_performance(request: Request):
    """Render the performance panel partial (htmx fragment)."""
    config = request.app.state.config
    pool: DbPool = request.app.state.db
    dbs = await _cost_dbs(config, pool)

    # Build matching escalations_dirs list (same dedup order as _cost_dbs).
    esc_dirs = [config.escalations_dir]
    for root in config.known_project_roots:
        peer_dir = root / 'data' / 'escalations'
        if peer_dir.resolve() != config.escalations_dir.resolve():
            esc_dirs.append(peer_dir)

    paths_r, esc_r, hist_r, ttc_r = await asyncio.gather(
        aggregate_completion_paths(dbs, esc_dirs),
        aggregate_escalation_rates(dbs, esc_dirs),
        aggregate_loop_histograms(dbs),
        aggregate_time_centiles(dbs),
        return_exceptions=True,
    )
    paths = _safe_gather_result(paths_r, {}, 'completion_paths')
    escalations = _safe_gather_result(esc_r, {}, 'escalation_rates')
    histograms = _safe_gather_result(hist_r, {}, 'loop_histograms')
    ttc = _safe_gather_result(ttc_r, {}, 'time_centiles')
    return templates.TemplateResponse(
        request, 'partials/performance.html',
        context={
            'paths': paths,
            'escalations': escalations,
            'histograms': histograms,
            'ttc': ttc,
        },
    )


# ---------------------------------------------------------------------------
# Costs partials
# ---------------------------------------------------------------------------


async def _cost_dbs(config: DashboardConfig, pool: DbPool) -> list[aiosqlite.Connection | None]:
    """Collect DB connections for all known project runs.db files."""
    paths = [config.runs_db]
    for root in config.known_project_roots:
        p = root / 'data' / 'orchestrator' / 'runs.db'
        if p.resolve() != config.runs_db.resolve():
            paths.append(p)
    return [await pool.get(p) for p in paths]


@app.get('/costs/partials/summary')
async def costs_partials_summary(request: Request):
    """Cost summary: 4 metric cards aggregated across all projects."""
    config = request.app.state.config
    pool: DbPool = request.app.state.db
    dbs = await _cost_dbs(config, pool)
    days = _parse_window(request)
    try:
        summary = await aggregate_cost_summary(dbs, days=days)
    except Exception as exc:
        logger.warning('Error fetching cost summary: %s', exc)
        summary = {}
    return templates.TemplateResponse(
        request, 'partials/costs/summary.html',
        context={'summary': summary},
    )


@app.get('/costs/partials/by-project')
async def costs_partials_by_project(request: Request):
    """Cost by project: horizontal stacked bar chart."""
    config = request.app.state.config
    pool: DbPool = request.app.state.db
    dbs = await _cost_dbs(config, pool)
    days = _parse_window(request)
    try:
        by_project = await aggregate_cost_by_project(dbs, days=days)
    except Exception as exc:
        logger.warning('Error fetching cost by project: %s', exc)
        by_project = {}
    return templates.TemplateResponse(
        request, 'partials/costs/by_project.html',
        context={'by_project': by_project},
    )


@app.get('/costs/partials/by-account')
async def costs_partials_by_account(request: Request):
    """Cost by account: doughnut chart + table."""
    config = request.app.state.config
    pool: DbPool = request.app.state.db
    dbs = await _cost_dbs(config, pool)
    days = _parse_window(request)
    try:
        by_account = await aggregate_cost_by_account(dbs, days=days)
    except Exception as exc:
        logger.warning('Error fetching cost by account: %s', exc)
        by_account = {}
    return templates.TemplateResponse(
        request, 'partials/costs/by_account.html',
        context={'by_account': by_account},
    )


@app.get('/costs/partials/by-role')
async def costs_partials_by_role(request: Request):
    """Cost by role: horizontal stacked bar chart."""
    config = request.app.state.config
    pool: DbPool = request.app.state.db
    dbs = await _cost_dbs(config, pool)
    days = _parse_window(request)
    try:
        by_role = await aggregate_cost_by_role(dbs, days=days)
    except Exception as exc:
        logger.warning('Error fetching cost by role: %s', exc)
        by_role = {}
    return templates.TemplateResponse(
        request, 'partials/costs/by_role.html',
        context={'by_role': by_role},
    )


@app.get('/costs/partials/trend')
async def costs_partials_trend(request: Request):
    """Cost trend: daily line chart."""
    config = request.app.state.config
    pool: DbPool = request.app.state.db
    dbs = await _cost_dbs(config, pool)
    days = _parse_window(request)
    try:
        trend = await aggregate_cost_trend(dbs, days=days)
    except Exception as exc:
        logger.warning('Error fetching cost trend: %s', exc)
        trend = {}
    return templates.TemplateResponse(
        request, 'partials/costs/trend.html',
        context={'trend': trend},
    )


@app.get('/costs/partials/events')
async def costs_partials_events(request: Request):
    """Account events feed."""
    config = request.app.state.config
    pool: DbPool = request.app.state.db
    dbs = await _cost_dbs(config, pool)
    days = _parse_window(request)
    try:
        events = await aggregate_account_events(dbs, days=days)
    except Exception as exc:
        logger.warning('Error fetching account events: %s', exc)
        events = []
    return templates.TemplateResponse(
        request, 'partials/costs/events.html',
        context={'events': events},
    )


@app.get('/costs/partials/runs')
async def costs_partials_runs(request: Request):
    """Run cost breakdown: expandable drilldown table."""
    config = request.app.state.config
    pool: DbPool = request.app.state.db
    dbs = await _cost_dbs(config, pool)
    days = _parse_window(request)
    try:
        runs = await aggregate_run_cost_breakdown(dbs, days=days)
    except Exception as exc:
        logger.warning('Error fetching run cost breakdown: %s', exc)
        runs = []
    return templates.TemplateResponse(
        request, 'partials/costs/runs.html',
        context={'runs': runs},
    )


# ---------------------------------------------------------------------------
# Merge queue partial
# ---------------------------------------------------------------------------


@app.get('/partials/merge-queue')
async def partials_merge_queue(request: Request):
    """Merge queue operational status card."""
    config = request.app.state.config
    pool: DbPool = request.app.state.db
    dbs = await _cost_dbs(config, pool)
    days = _parse_window(request)
    hours = days * 24
    window_raw = request.query_params.get('window', '30d')

    effective_now = datetime.now(UTC)
    depth, outcomes, latency, recent, spec = await asyncio.gather(
        aggregate_queue_depth_timeseries(dbs, hours=hours, now=effective_now),
        aggregate_outcome_distribution(dbs, hours=hours, now=effective_now),
        aggregate_latency_stats(dbs, hours=hours, now=effective_now),
        aggregate_recent_merges(dbs, limit=20, hours=hours),
        aggregate_speculative_stats(dbs, hours=hours, now=effective_now),
        return_exceptions=True,
    )

    depth = cast(ChartData, _safe_gather_result(depth, {'labels': [], 'values': []}, 'merge_queue.depth'))
    depth = trim_leading_zero_buckets(depth)
    outcomes = _safe_gather_result(outcomes, {'labels': [], 'values': []}, 'merge_queue.outcomes')
    latency = _safe_gather_result(
        latency, {'p50': 0, 'p95': 0, 'p99': 0, 'count': 0, 'mean_ms': 0.0},
        'merge_queue.latency',
    )
    recent = _safe_gather_result(recent, [], 'merge_queue.recent')
    spec = _safe_gather_result(
        spec, {'hit_count': 0, 'discard_count': 0, 'total': 0, 'hit_rate': 0.0},
        'merge_queue.speculative',
    )

    return templates.TemplateResponse(
        request, 'partials/merge_queue.html',
        context={
            'depth_timeseries': depth,
            'outcomes': outcomes,
            'latency': latency,
            'recent': recent,
            'speculative': spec,
            'window': window_raw,
        },
    )


# ---------------------------------------------------------------------------
# Burndown partials
# ---------------------------------------------------------------------------


async def _burndown_dbs(config: DashboardConfig, pool: DbPool) -> list[aiosqlite.Connection | None]:
    """Collect DB connections for all known project burndown.db files.

    Mirrors ``_cost_dbs``: starts from the main burndown_db, then appends each
    known_project_root's burndown.db (dedup by resolved path).
    """
    paths = [config.burndown_db]
    for root in config.known_project_roots:
        p = root / 'data' / 'burndown' / 'burndown.db'
        if p.resolve() != config.burndown_db.resolve():
            paths.append(p)
    return [await pool.get(p) for p in paths]


@app.get('/burndown/partials/charts')
async def burndown_partials_charts(request: Request):
    """Burndown charts: stacked area chart per project, aggregated across all known roots."""
    config = request.app.state.config
    pool: DbPool = request.app.state.db
    dbs = await _burndown_dbs(config, pool)
    window_raw = request.query_params.get('window', '30d')
    days = _BURNDOWN_WINDOWS.get(window_raw, 30)
    try:
        projects = await aggregate_burndown_projects(dbs)
        series: dict = {}
        for pid in projects:
            series[pid] = await aggregate_burndown_series(dbs, pid, days=days)
    except Exception:
        logger.warning('Error fetching burndown data', exc_info=True)
        series = {}
    return templates.TemplateResponse(
        request, 'partials/burndown/charts.html',
        context={'series': series},
    )
