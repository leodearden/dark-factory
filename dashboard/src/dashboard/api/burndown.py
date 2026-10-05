"""`/api/v2/dashboard/burndown` — task burndown series, aggregate and per-project.

Reads the snapshot history that ``dashboard.loops::_burndown_loop`` writes,
capturing ``now`` once per request. That one instant is the window cutoff for
the project listing and every per-project aggregate, so the series cannot skew
across projects and the listing names exactly the projects the window sampled,
and it is the instant every burndown Datum is judged at, served as
``served_at``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dashboard.api.window import _parse_window, with_window
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


# A distinct vocabulary, parsed by the shared dashboard.api.window parser:
# the burndown chip offers 90d where dashboard.api.window._WINDOW_DAYS offers
# `all`. api_burndown is this one's only consumer.
_BURNDOWN_WINDOWS: dict[str, int] = {
    '24h': 1,
    '7d': 7,
    '30d': 30,
    '90d': 90,
}


@router.get('/api/v2/dashboard/burndown')
async def api_burndown(request: Request) -> JSONResponse:
    """BURNDOWN + BURNDOWN_BY_PROJECT + served_at — per-project status time series."""
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    dbs = await _burndown_dbs(config, pool)
    window = _parse_window(request.query_params, vocabulary=_BURNDOWN_WINDOWS)
    now = datetime.now(UTC)  # clock-exempt: single-capture route

    try:
        projects = await aggregate_burndown_projects(dbs, days=window.days, now=now)
        per_pid = await asyncio.gather(
            *(aggregate_burndown_series(dbs, pid, days=window.days, now=now) for pid in projects)
        )
        series: dict[str, dict] = dict(zip(projects, per_pid, strict=True))
    except Exception:
        logger.warning('Error fetching burndown data', exc_info=True)
        series = {}
    shaped = redux_api.shape_burndown(series, served_at=now)
    return JSONResponse(with_window({**shaped, 'served_at': now.isoformat()}, window))
