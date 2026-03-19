"""Dark Factory dashboard — FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dashboard.config import DashboardConfig
from dashboard.data.memory import get_memory_status, get_queue_stats

_pkg_dir = Path(__file__).parent

templates = Jinja2Templates(directory=str(_pkg_dir / 'templates'))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage httpx.AsyncClient lifecycle."""
    app.state.http_client = httpx.AsyncClient()
    app.state.config = DashboardConfig.from_env()
    yield
    await app.state.http_client.aclose()


app = FastAPI(title='Dark Factory Dashboard', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=str(_pkg_dir / 'static')), name='static')


@app.get('/api/health')
async def health():
    return {'status': 'ok'}


@app.get('/partials/memory')
async def memory_partial(request: Request):
    http_client = request.app.state.http_client
    config = request.app.state.config
    status = await get_memory_status(http_client, config)
    queue = await get_queue_stats(http_client, config)
    return templates.TemplateResponse(
        request, 'partials/memory.html', context={'status': status, 'queue': queue}
    )
