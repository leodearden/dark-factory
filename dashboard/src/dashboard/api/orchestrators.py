"""`/api/v2/dashboard/orchestrators` — the per-orchestrator status table.

Discovers the orchestrator processes running across every known project
root and pairs them with the running-count sparkline sampled by
``dashboard.loops``.
"""

from __future__ import annotations

import asyncio

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dashboard.config import DashboardConfig
from dashboard.data import redux_api
from dashboard.data.db import DbPool
from dashboard.data.metrics import get_orchestrators_running_series
from dashboard.data.orchestrator import discover_orchestrators

router = APIRouter()


@router.get('/api/v2/dashboard/orchestrators')
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
