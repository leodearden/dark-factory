"""`/api/v2/dashboard/burndown` — task burndown series, aggregate and per-project.

Reads the snapshot history that ``dashboard.loops::_burndown_loop`` writes,
capturing ``now`` once per request and threading that one value through
every per-project aggregate so the series cannot skew across projects.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dashboard.config import DashboardConfig
from dashboard.data import redux_api
from dashboard.data.burndown import (
    aggregate_burndown_projects,
    aggregate_burndown_series,
)
from dashboard.data.db import DbPool
from dashboard.project_dbs import _burndown_dbs

logger = logging.getLogger(__name__)

router = APIRouter()


# Deliberately NOT dashboard.api.window's _WINDOW_DAYS: the burndown chart
# offers 90d where that vocabulary offers `all`, and static/redux/app.jsx
# pins the two sets as distinct. api_burndown is this one's only consumer.
_BURNDOWN_WINDOWS: dict[str, int] = {
    '24h': 1,
    '7d': 7,
    '30d': 30,
    '90d': 90,
}


@router.get('/api/v2/dashboard/burndown')
async def api_burndown(request: Request) -> JSONResponse:
    """BURNDOWN + BURNDOWN_BY_PROJECT — per-project status time series."""
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    dbs = await _burndown_dbs(config, pool)
    window_raw = request.query_params.get('window', '30d')
    days = _BURNDOWN_WINDOWS.get(window_raw, 30)

    try:
        projects = await aggregate_burndown_projects(dbs)
        now = datetime.now(UTC)  # clock-exempt: single-capture route
        per_pid = await asyncio.gather(
            *(aggregate_burndown_series(dbs, pid, days=days, now=now) for pid in projects)
        )
        series: dict[str, dict] = dict(zip(projects, per_pid, strict=True))
    except Exception:
        logger.warning('Error fetching burndown data', exc_info=True)
        series = {}
    return JSONResponse(redux_api.shape_burndown(series))
