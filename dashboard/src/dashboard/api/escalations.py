"""`/api/v2/dashboard/escalations` — escalation queues with resolved task cards.

The route itself is thin: `dashboard.data.escalations.build_escalation_queues`
assembles the queues, and this module resolves a task card list per
orchestrator root to hang off them.

It also owns the task-cards TTL cache that resolution runs through. Cache
and consumer live in one file because the cache exists only for this route:
its TTL, its whole-operation budget and its degraded-return shape are all
statements about how the escalations tab should behave when the MCP server
is slow, and splitting them from the handler would leave neither half
legible alone.

``fetch_tasks`` is called raw here, deliberately — see the note in
``_load_task_cards`` on why the whole operation, not the HTTP request, is
what needs bounding.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dashboard.config import DashboardConfig
from dashboard.data import redux_api
from dashboard.data.escalations import build_escalation_queues
from dashboard.data.mcp_fanout import TTLCache
from dashboard.data.tasks import DEFAULT_WHOLE_OPERATION_BUDGET, fetch_tasks

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Task-cards TTL cache (mirrors load_task_titles pattern in merge_queue.py)
# ---------------------------------------------------------------------------

_TASK_CARDS_TTL_SECONDS = 10.0

# Whole-operation bound for ``_load_task_cards``, enforced with
# ``asyncio.wait_for``. Bound to the shared default rather than restating the
# literal, so the arithmetic lives in exactly one place; this site may later
# TIGHTEN its own constant (the structural test enforces it can never widen
# it). Single-root call whose fan-out happens at the CALLER via
# ``asyncio.gather`` over root ids, so the handler cost is max-of-N rather
# than sum-of-N — no whole-loop deadline is needed. (``discover_orchestrators``
# used to be the contrasting case, a sequential per-root walk that needed one.
# It reads no task tree since task 5587.)
_TASK_CARDS_BUDGET = DEFAULT_WHOLE_OPERATION_BUDGET

_task_cards_cache: TTLCache[list[dict] | dict] = TTLCache(
    ttl_seconds=lambda: _TASK_CARDS_TTL_SECONDS
)


def _task_cards_cache_clear() -> None:
    """Clear the task-cards TTL cache (test hook)."""
    _task_cards_cache.clear()


async def _load_task_cards(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: str,
) -> list[dict]:
    """Return the full dashboard-shaped task list for *project_root*.

    Results are cached per project_root for ``_TASK_CARDS_TTL_SECONDS`` (~10 s)
    to avoid hammering the MCP server on every dashboard poll.  An offline
    marker or MCP failure returns ``[]`` WITHOUT writing to the cache, so a
    transient blip doesn't pin empty results for the full TTL window.
    Concurrent cold callers for the same project_root collapse onto one
    in-flight fetch_tasks call (TTLCache single-flight).

    Test hook: call ``_task_cards_cache_clear()`` to reset cache state between
    test cases.

    **Bounded as a whole.** The whole operation is bounded by
    ``_TASK_CARDS_BUDGET`` via ``asyncio.wait_for``. ``fetch_tasks``' own
    *timeout* is a PER-HTTP-REQUEST budget — it bounds connect/read/write and
    pool acquisition, never the operation as a whole — so without this layer a
    hang that opens no socket (a connection-pool lock, say) is unbounded, and
    that is exactly what wedged /api/v2/dashboard/escalations for 19.8 h. A
    timeout returns the SAME ``[]``, so the tab renders cardless rather than
    hanging, and nothing is cached on that path.

    The ``wait_for`` deliberately encloses ``get_or_refresh`` rather than the
    inner ``fetch_tasks``. ``TTLCache.get_or_refresh`` serializes cold callers
    for one key behind a per-key lock and runs the refresh WHILE HOLDING it,
    so an inner-only wrap would leave a QUEUED caller waiting unbounded for
    the holder's full budget before paying its own: the pair costs 2x and N
    waiters cost N x, and the dashboard's 3 s poll makes waiters routine.
    Enclosing the outer call bounds the lock wait too, and is safe —
    ``wait_for`` cancels the inner task, cancellation unwinds
    ``async with lock``, and ``__aexit__`` releases it rather than leaking it.

    The five-line ``wait_for``/``except TimeoutError``/warn/degrade construct
    below, and the lock-placement rationale above, are duplicated verbatim at
    the sibling call site (``merge_queue.load_task_titles``). That duplication is
    KNOWN and deliberate for now: the mechanism is a property of
    ``TTLCache`` — not of either call site — so the idiom belongs on
    ``dashboard/src/dashboard/data/mcp_fanout.py::TTLCache`` as a
    ``get_or_refresh_bounded`` that owns the timeout, the warning and the
    degraded return. That file is outside this change's lock set, so the
    extraction is left to the sibling TTLCache task referenced below.

    This bounds THIS caller only. It does not fix the general TTLCache
    queue-amplifier class across all of its call sites; that is the sibling
    task filed in the same batch.
    """

    async def _refresh() -> list[dict] | dict:
        return await fetch_tasks(client, config, project_root)

    try:
        result = await asyncio.wait_for(
            _task_cards_cache.get_or_refresh(
                project_root, _refresh, cache_ok=lambda v: isinstance(v, list),
            ),
            timeout=_TASK_CARDS_BUDGET,
        )
    except TimeoutError:
        # Broader than the ``wait_for`` expiry, deliberately. On 3.11+
        # ``asyncio.TimeoutError`` IS the builtin, and ``socket.timeout`` is
        # too, so a ``TimeoutError`` raised INSIDE the refresh is folded into
        # this same budget path rather than 500ing the escalations tab. The
        # message below is therefore authoritative about the OUTCOME — the
        # cards are unknown for this poll — and not about the cause.
        logger.warning(
            '_load_task_cards %s: exceeded the %.1fs whole-operation budget — '
            "the escalation tab's task cards are UNKNOWN for this poll "
            '(not absent)',
            project_root, _TASK_CARDS_BUDGET,
        )
        return []
    return list(result) if isinstance(result, list) else []


@router.get('/api/v2/dashboard/escalations')
async def api_escalations(request: Request) -> JSONResponse:
    """ESCALATIONS — per-project escalation queues with resolved task cards."""
    config: DashboardConfig = request.app.state.config
    http_client: httpx.AsyncClient = request.app.state.http_client

    queues = build_escalation_queues(config)

    # Derive fetch roots from orchestrator subsection ids — these are already
    # str(root) and are de-duped by build_escalation_queues.  Keying task_maps
    # by subsection id means the shaper can match by id directly.
    root_ids = [s['id'] for s in queues.get('subsections') or [] if s.get('kind') == 'orchestrator']

    results = await asyncio.gather(*(_load_task_cards(http_client, config, rid) for rid in root_ids))
    task_maps: dict[str, list[dict]] = {rid: tasks for rid, tasks in zip(root_ids, results, strict=True)}

    return JSONResponse(redux_api.shape_escalations(queues, task_maps))
