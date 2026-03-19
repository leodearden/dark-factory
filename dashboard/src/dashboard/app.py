"""Dark Factory dashboard — FastAPI application."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dashboard.config import DashboardConfig
from dashboard.data import memory as memory_data
from dashboard.data.orchestrator import discover_orchestrators
from dashboard.data.reconciliation import (
    get_buffer_stats,
    get_burst_state,
    get_latest_verdict,
    get_recent_runs,
    get_watermarks,
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
    return f'{total_minutes}m ago'


templates.env.filters['timeago'] = timeago

_TRIGGER_TYPE_MAP = {
    'max_staleness': 'staleness',
    'buffer_size': 'buffer',
}


def format_trigger(value: str | None) -> str:
    """Format a trigger_reason string for human-friendly display.

    Examples:
        None          → ''
        'manual'      → 'manual'
        'quiescent:6' → 'quiescent (6)'
        'buffer_size:10' → 'buffer (10)'
        'max_staleness:<ISO>' → 'staleness (<relative>)'
        'foo:bar'     → 'foo (bar)'
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage httpx.AsyncClient lifecycle."""
    app.state.http_client = httpx.AsyncClient(follow_redirects=True)
    app.state.config = DashboardConfig.from_env()
    yield
    await app.state.http_client.aclose()


app = FastAPI(title='Dark Factory Dashboard', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=str(_pkg_dir / 'static')), name='static')


@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse(request, 'index.html')


@app.get('/api/health')
async def health():
    return {'status': 'ok'}


@app.get('/partials/memory')
async def memory_partial(request: Request):
    http_client = request.app.state.http_client
    config = request.app.state.config
    status = await memory_data.get_memory_status(http_client, config)
    queue = await memory_data.get_queue_stats(http_client, config)
    return templates.TemplateResponse(
        request, 'partials/memory.html', context={'status': status, 'queue': queue}
    )


@app.get('/partials/recon')
async def partials_recon(request: Request):
    """Render the reconciliation panel partial (htmx fragment)."""
    config = request.app.state.config
    db = config.reconciliation_db

    buffer_stats, burst_state, watermarks, verdict, runs = await asyncio.gather(
        get_buffer_stats(db),
        get_burst_state(db),
        get_watermarks(db),
        get_latest_verdict(db),
        get_recent_runs(db),
    )

    return templates.TemplateResponse(
        'partials/recon.html',
        {
            'request': request,
            'buffer_stats': buffer_stats,
            'burst_state': burst_state,
            'watermarks': watermarks,
            'verdict': verdict,
            'runs': runs,
        },
    )


@app.get('/partials/orchestrators')
async def partials_orchestrators(request: Request):
    """Render the orchestrators panel partial (htmx fragment)."""
    config = request.app.state.config
    orchestrators = await asyncio.to_thread(discover_orchestrators, config)
    return templates.TemplateResponse(
        request, 'partials/orchestrators.html', context={'orchestrators': orchestrators}
    )
