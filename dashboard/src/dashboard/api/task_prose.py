"""`/api/v2/dashboard/task/{project}/T-{task_id}` — one task's description and details.

Serves the Tasks tab's Task Detail pane, which fetches the prose of the
selected task on selection and never polls it. The ACTIVE_TASKS list
deliberately omits both fields (see
``dashboard/src/dashboard/data/active_tasks.py::_build_task_row``).
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dashboard.config import DashboardConfig
from dashboard.data.active_tasks import project_roots_for_label, task_uid
from dashboard.data.tasks import (
    DEFAULT_WHOLE_OPERATION_BUDGET,
    TaskNotFound,
    TaskProse,
    TaskProseRead,
    TaskReadOffline,
    fetch_task_prose,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Bound to the shared default by reference, as api/escalations.py's
# _TASK_CARDS_BUDGET is: this site may tighten it, never widen it.
_TASK_PROSE_BUDGET = DEFAULT_WHOLE_OPERATION_BUDGET


@router.get('/api/v2/dashboard/task/{project}/T-{task_id}')
async def api_task_prose(project: str, task_id: int, request: Request) -> JSONResponse:
    """TASK_PROSE:<uid> for the row uid ``<project>/T-<task_id>``.

    The project LABEL resolves against the configured roots only, so a request
    can never aim fused-memory at an arbitrary path. A label naming two roots
    is refused rather than guessed: a guess could show another task's prose
    under a correct-looking title.
    """
    config: DashboardConfig = request.app.state.config
    http_client: httpx.AsyncClient = request.app.state.http_client
    uid = task_uid(project, task_id)
    roots = project_roots_for_label(config, project)
    if not roots:
        return JSONResponse({'error': 'unknown_project', 'project': project}, status_code=404)
    if len(roots) > 1:
        return JSONResponse(
            {
                'error': 'ambiguous_project',
                'project': project,
                'roots': [str(root) for root in roots],
            },
            status_code=409,
        )
    try:
        read = await asyncio.wait_for(
            fetch_task_prose(http_client, config, roots[0], task_id),
            timeout=_TASK_PROSE_BUDGET,
        )
    except TimeoutError:
        # The expiry cancels first_success mid-flight, and it logs nothing on a
        # cancellation, so this line is the only journal trace.
        logger.warning(
            'task prose %s: no answer within the %.1fs whole-operation budget',
            uid, _TASK_PROSE_BUDGET,
        )
        return JSONResponse(
            {
                'error': 'budget_exceeded',
                'detail': f'no answer within the {_TASK_PROSE_BUDGET:.1f}s budget',
            },
            status_code=504,
        )
    return _response_for(uid, read)


def _response_for(uid: str, read: TaskProseRead) -> JSONResponse:
    match read:
        case TaskProse():
            return JSONResponse({f'TASK_PROSE:{uid}': read.to_wire()})
        case TaskNotFound(detail=detail):
            return JSONResponse({'error': 'task_not_found', 'detail': detail}, status_code=404)
        case TaskReadOffline(detail=detail):
            return JSONResponse(
                {'error': 'fused_memory_unreachable', 'detail': detail}, status_code=502,
            )
