"""`/api/v2/dashboard/merge-queue` — per-project merge queues and their history.

This module is the ROUTE layer. The aggregation it calls lives in
``dashboard.data.merge_queue``, whose name it deliberately mirrors; the
names imported below (``build_per_project_merge_queue``,
``fetch_live_merge_queues``, ``load_task_titles``, ``resolve_active``,
``enrich_merges_with_titles``) all come from that data module, never from
here.

``now`` is captured once per request and threaded through every leg, so a
queue and the sparkline beside it cannot disagree about the present.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dashboard.api.window import _parse_window
from dashboard.config import DashboardConfig
from dashboard.data import redux_api
from dashboard.data.chart_utils import ChartData, trim_leading_zero_buckets
from dashboard.data.db import DbPool
from dashboard.data.merge_halt import get_merge_halt_status
from dashboard.data.merge_queue import (
    build_per_project_merge_queue,
    enrich_merges_with_titles,
    fetch_live_merge_queues,
    load_task_titles,
    resolve_active,
)
from dashboard.data.metrics import get_merge_active_series
from dashboard.data.redux_api import _project_label
from dashboard.project_dbs import _project_scoped_dbs_labeled

router = APIRouter()


@router.get('/api/v2/dashboard/merge-queue')
async def api_merge_queue(request: Request) -> JSONResponse:
    """MERGE_QUEUE — per-project depth/outcomes/latency/recent/active/speculative."""
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    days = _parse_window(request.query_params)
    hours = days * 24
    effective_now = datetime.now(UTC)  # clock-exempt: single-capture route

    project_dbs = await _project_scoped_dbs_labeled(
        config,
        pool,
        Path('data/orchestrator/runs.db'),
    )
    http_client: httpx.AsyncClient = request.app.state.http_client
    projects_raw, halt_status, live_map = await asyncio.gather(
        build_per_project_merge_queue(
            project_dbs,
            hours=hours,
            now=effective_now,
            recent_window_minutes=1440,
        ),
        get_merge_halt_status(http_client, config.escalation_urls),
        fetch_live_merge_queues(http_client, config.escalation_urls),
    )
    pids = list(projects_raw.keys())
    title_maps = await asyncio.gather(*(load_task_titles(http_client, config, pid) for pid in pids))
    enriched: dict[str, dict] = {}
    for pid, data, titles in zip(pids, projects_raw.values(), title_maps, strict=True):
        label = _project_label(pid)
        resolved = resolve_active(label, live_map, data.get('active', []))
        # ι=1894: extract live metrics from the snapshot and stash for shaping
        live_metrics = live_map.get(label, {}).get('metrics')
        enriched[pid] = {
            **data,
            'depth_timeseries': trim_leading_zero_buckets(
                cast(ChartData, data['depth_timeseries'])
            ),
            'recent': enrich_merges_with_titles(data['recent'], titles),
            'active': enrich_merges_with_titles(resolved['entries'], titles),
            'active_approximate': resolved['approximate'],
            'live_metrics': live_metrics,
        }
    metrics_db = await pool.get(config.metrics_db)
    active_sparks: dict[str, dict] = {}
    for pid in pids:
        active_sparks[pid] = await get_merge_active_series(metrics_db, project_id=pid, days=1)
    return JSONResponse(
        redux_api.shape_merge_queue(
            enriched,
            active_sparks=active_sparks,
            halt_status=halt_status,
        )
    )
