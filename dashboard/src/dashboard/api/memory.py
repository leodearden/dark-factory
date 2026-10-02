"""`/api/v2/dashboard/memory` — memory-store counts, sparklines and queue depth.

Fans out over three MCP legs plus the local metrics DB, each leg bounded by
this module's per-request budget.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dashboard.config import DashboardConfig
from dashboard.data import memory as memory_data
from dashboard.data import redux_api
from dashboard.data.db import DbPool
from dashboard.data.metrics import (
    get_memory_24h_ago,
    get_memory_sparks,
    get_queue_pending_series,
)

router = APIRouter()


# Per-HTTP-request budget for /memory's three MCP legs (task 3871), matching
# the metrics samplers' 5.0s. Without it each leg silently ran to
# mcp_tool_call's 10s default — including pool acquisition — while its
# sibling legs honoured a real budget.
#
# Deliberately NOT also wrapped in asyncio.wait_for, unlike
# ``app.py::api_curator``: this gather has no return_exceptions=True, so a
# TimeoutError raised by a wrapper would escape as a 500 rather than degrading
# one leg. The three callees swallow their own per-URL failures and return
# offline dicts, so adding a whole-operation bound here means first giving
# each leg its own exception containment — a shape change beyond this task.
_MEMORY_ENDPOINT_TIMEOUT_SECONDS = 5.0


@router.get('/api/v2/dashboard/memory')
async def api_memory(request: Request) -> JSONResponse:
    """MEMORY_STATUS, including queue counts and per-project totals."""
    http_client = request.app.state.http_client
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    metrics_db = await pool.get(config.metrics_db)
    status, queue, sparks, queue_spark, delta_24h, wal = await asyncio.gather(
        memory_data.get_memory_status(
            http_client, config, timeout=_MEMORY_ENDPOINT_TIMEOUT_SECONDS,
        ),
        memory_data.get_queue_stats(
            http_client, config, timeout=_MEMORY_ENDPOINT_TIMEOUT_SECONDS,
        ),
        get_memory_sparks(metrics_db, days=1),
        get_queue_pending_series(metrics_db, days=1),
        get_memory_24h_ago(metrics_db),
        memory_data.get_wal_status(
            http_client, config, timeout=_MEMORY_ENDPOINT_TIMEOUT_SECONDS,
        ),
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
