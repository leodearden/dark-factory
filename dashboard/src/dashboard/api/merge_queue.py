"""`/api/v2/dashboard/merge-queue` — per-project merge queues and their history.

This module is the ROUTE layer. The aggregation it calls lives in
``dashboard.data.merge_queue``, whose name it deliberately mirrors; the
names imported below (``build_per_project_merge_queue``,
``fetch_live_merge_queues``, ``resolve_active``, ``merge_task_refs``,
``enrich_merges_with_titles``) all come from that data module, never from
here. Row titles come from ``dashboard.data.task_lookup.lookup_tasks``: one
call over every id the payload names, never a whole task tree.

``render_at`` is resolved once per request and threaded through every leg,
so a queue and the sparkline beside it cannot disagree about the present.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dashboard.api.window import _parse_window, with_window
from dashboard.config import DashboardConfig
from dashboard.data import redux_api
from dashboard.data.chart_utils import ChartData, trim_leading_zero_buckets
from dashboard.data.db import DbPool
from dashboard.data.mcp_fanout import project_label
from dashboard.data.merge_halt import get_merge_halt_status
from dashboard.data.merge_queue import (
    build_per_project_merge_queue,
    enrich_merges_with_titles,
    fetch_live_merge_queues,
    merge_task_refs,
    resolve_active,
)
from dashboard.data.metrics import get_merge_active_series
from dashboard.data.task_lookup import lookup_tasks
from dashboard.data.utils import resolve_now
from dashboard.project_dbs import _project_scoped_dbs_labeled

router = APIRouter()


@router.get('/api/v2/dashboard/merge-queue')
async def api_merge_queue(request: Request) -> JSONResponse:
    """MERGE_QUEUE — per-project depth/outcomes/latency/recent/in_queue/speculative."""
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    window = _parse_window(request.query_params)
    render_at = resolve_now(None)

    project_dbs = await _project_scoped_dbs_labeled(
        config,
        pool,
        Path('data/orchestrator/runs.db'),
    )
    http_client: httpx.AsyncClient = request.app.state.http_client
    projects_raw, halt_status, live_map = await asyncio.gather(
        build_per_project_merge_queue(
            project_dbs,
            hours=window.days * 24,
            now=render_at,
        ),
        get_merge_halt_status(http_client, config.escalation_urls),
        fetch_live_merge_queues(http_client, config.escalation_urls),
    )
    metrics_db = await pool.get(config.metrics_db)
    active_sparks = {
        pid: await get_merge_active_series(metrics_db, project_id=pid, days=1, now=render_at)
        for pid in projects_raw
    }
    queues = {
        pid: resolve_active(project_label(pid), live_map, active_sparks[pid], now=render_at)
        for pid in projects_raw
    }
    lookup = await lookup_tasks(
        http_client,
        config,
        {
            ref
            for pid, data in projects_raw.items()
            for rows in (data['recent'], queues[pid].entries)
            for ref in merge_task_refs(pid, rows)
        },
        now=render_at,
    )
    enriched = {
        pid: {
            **data,
            'depth_timeseries': trim_leading_zero_buckets(
                cast(ChartData, data['depth_timeseries'])
            ),
            'recent': enrich_merges_with_titles(data['recent'], pid, lookup),
            'active': enrich_merges_with_titles(queues[pid].entries, pid, lookup),
            'in_queue': queues[pid].in_queue,
            'live_probe_configured': queues[pid].probe_configured,
            # ι=1894: the live metrics the probe carried, stashed for shaping
            'live_metrics': live_map.get(project_label(pid), {}).get('metrics'),
        }
        for pid, data in projects_raw.items()
    }
    # Resolved AFTER the fan-out, as api/tasks.py::api_tasks does: every
    # as_of above was stamped at or before this instant, never after it.
    served_at = resolve_now(None)
    return JSONResponse(
        with_window(
            redux_api.shape_merge_queue(
                enriched,
                served_at=served_at,
                active_sparks=active_sparks,
                halt_status=halt_status,
            ),
            window,
        )
    )
