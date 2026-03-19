"""Dark Factory dashboard — FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage httpx.AsyncClient lifecycle."""
    app.state.http_client = httpx.AsyncClient()
    yield
    await app.state.http_client.aclose()


app = FastAPI(title='Dark Factory Dashboard', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=str(_pkg_dir / 'static')), name='static')


@app.get('/api/health')
async def health():
    return {'status': 'ok'}
