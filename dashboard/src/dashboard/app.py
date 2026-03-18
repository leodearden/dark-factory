"""Dark Factory dashboard — FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

_pkg_dir = Path(__file__).parent

templates = Jinja2Templates(directory=str(_pkg_dir / 'templates'))


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
