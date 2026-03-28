"""Dark Factory dashboard — FastAPI application."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dashboard.config import DashboardConfig
from dashboard.data import memory as memory_data
from dashboard.data.chart_utils import group_top_n
from dashboard.data.db import DbPool
from dashboard.data.orchestrator import discover_orchestrators
from dashboard.data.performance import (
    get_completion_paths,
    get_escalation_rates,
    get_loop_histograms,
    get_time_centiles,
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
from dashboard.data.write_journal import (
    get_agent_breakdown,
    get_memory_timeseries,
    get_operations_breakdown,
)

_pkg_dir = Path(__file__).parent

templates = Jinja2Templates(directory=str(_pkg_dir / 'templates'))


def timeago(value: str | None) -> str:
    """Convert an ISO timestamp string to a human-friendly relative time."""
    if value is None:
        return 'never'
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return 'never'
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
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
        total = int(value)
    except (TypeError, ValueError):
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage shared resources: HTTP client, DB connection pool."""
    app.state.http_client = httpx.AsyncClient(follow_redirects=True)
    app.state.config = DashboardConfig.from_env()
    app.state.db = DbPool()
    app.state.start_time = time.monotonic()
    yield
    await app.state.db.close_all()
    await app.state.http_client.aclose()


app = FastAPI(title='Dark Factory Dashboard', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=str(_pkg_dir / 'static')), name='static')


@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse(request, 'index.html')


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
    timeseries, operations, agents = await asyncio.gather(
        get_memory_timeseries(db),
        get_operations_breakdown(db),
        get_agent_breakdown(db),
    )
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

    buffer_stats, burst_state, watermarks_list, verdict, runs, last_attempted_map = (
        await asyncio.gather(
            get_buffer_stats(db),
            get_burst_state(db),
            get_watermarks(db),
            get_latest_verdict(db),
            get_recent_runs(db),
            get_last_attempted_run(db),
        )
    )

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
    db = await pool.get(config.runs_db)
    esc_dir = config.escalations_dir

    paths, escalations, histograms, ttc = await asyncio.gather(
        get_completion_paths(db, esc_dir),
        get_escalation_rates(db, esc_dir),
        get_loop_histograms(db),
        get_time_centiles(db),
    )
    return templates.TemplateResponse(
        request, 'partials/performance.html',
        context={
            'paths': paths,
            'escalations': escalations,
            'histograms': histograms,
            'ttc': ttc,
        },
    )
