"""`/api/v2/dashboard/orchestrators` — the per-orchestrator status table.

Discovers the orchestrator PROCESSES running across every known project
root and pairs them with the running-count sparkline sampled by
``dashboard.loops``. No task tree is read on this route: task counts live on
``/api/v2/dashboard/tasks``, inside a ``Datum`` that says when each was
measured.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dashboard.config import DashboardConfig
from dashboard.data import redux_api
from dashboard.data.db import DbPool
from dashboard.data.metrics import get_orchestrators_running_series
from dashboard.data.orchestrator import discover_orchestrators
from dashboard.data.utils import resolve_now

router = APIRouter()


@router.get('/api/v2/dashboard/orchestrators')
async def api_orchestrators(request: Request) -> JSONResponse:
    """ORCHESTRATORS + PROJECTS for the redux dashboard.

    ``served_at`` stamps the payload with the one instant it was shaped at, so
    every surface the dashboard serves says when it looked. Nothing here
    carries a ``Datum`` — a ``ps`` scan of the local process table has no
    staleness to disclose — but the stamp is what lets a client tell a
    still-polling tab from a frozen one.

    The ``asyncio.gather`` survives the loss of the task fetch: the sparkline
    still reads the metrics DB, and the ``ps`` scan still runs on a thread, so
    there are still two independent waits to overlap.
    """
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    served_at = resolve_now(None)
    metrics_db = await pool.get(config.metrics_db)
    orchestrators, running_spark = await asyncio.gather(
        discover_orchestrators(config),
        get_orchestrators_running_series(metrics_db, days=1),
    )
    known_roots = [config.project_root, *config.known_project_roots]
    return JSONResponse(
        {
            **redux_api.shape_orchestrators(
                orchestrators,
                known_project_roots=known_roots,
                running_spark=running_spark,
            ),
            'served_at': served_at.isoformat(),
        }
    )
