"""Fan-out probe of per-task runtime state across all known orchestrators.

Each orchestrator's escalation MCP server owns the live runtime snapshot for
its own worktrees (loops/attempts/started/lane/phase/lane_state — see
``shared.task_runtime_state``). To surface that on the dashboard we fan out
``get_task_runtime_state`` calls across every known escalation URL
concurrently and aggregate the results into a ``{project_label:
TaskRuntimeSnapshot}`` dict.

Clones the ``dashboard.data.merge_halt`` fan-out pattern: ``asyncio.gather``,
per-project offline degradation so one unreachable/misbehaving orchestrator
never takes down the others, and the same **two complementary layers** of
timeout bound (the default ``mcp_tool_call`` timeout is 10s — too slow for a
fast-polling dashboard):

1. **Per HTTP request** — ``per_call_timeout`` is threaded into
   ``mcp_tool_call``, which hands it to ``client.post``, bounding
   connect/read/write *and pool acquisition* on the shared client.
2. **Per whole operation** — the enclosing ``asyncio.wait_for`` stays,
   because layer 1 alone cannot cap the call: a *cold* session performs
   three posts (initialize, notifications/initialized, tools/call), so a
   per-request bound of T permits roughly 3T overall.

The wire payload is decoded through ``TaskRuntimeSnapshot.model_validate`` —
the SAME shared model the server projects into (INV-1 contracts-machine-
checked; no structurally-mirrored second copy). Any failure mode (unreachable,
timeout, malformed payload) degrades that ONE project to
``TaskRuntimeSnapshot(offline=True)`` — the model's own ``offline`` field,
synthesized client-side exactly as its docstring prescribes.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from shared.task_runtime_state import TaskRuntimeSnapshot

from dashboard.data.memory import mcp_tool_call

logger = logging.getLogger(__name__)

DEFAULT_PER_CALL_TIMEOUT = 2.0


async def _probe_one(
    client: httpx.AsyncClient,
    base_url: str,
    timeout: float,
) -> TaskRuntimeSnapshot:
    try:
        result = await asyncio.wait_for(
            mcp_tool_call(
                client, base_url, 'get_task_runtime_state', {}, timeout=timeout,
            ),
            timeout=timeout,
        )
    except (TimeoutError, httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError, ValueError, OSError) as exc:
        logger.debug('get_task_runtime_state failed for %s: %s', base_url, exc)
        return TaskRuntimeSnapshot(offline=True)
    try:
        return TaskRuntimeSnapshot.model_validate(result)
    except ValueError as exc:
        # pydantic v2 ValidationError subclasses ValueError.
        logger.warning(
            'get_task_runtime_state: malformed payload from %s, degrading to offline: %s',
            base_url, exc,
        )
        return TaskRuntimeSnapshot(offline=True)


async def fetch_task_runtime(
    client: httpx.AsyncClient,
    escalation_urls: dict[str, str],
    *,
    per_call_timeout: float = DEFAULT_PER_CALL_TIMEOUT,
) -> dict[str, TaskRuntimeSnapshot]:
    """Probe every escalation URL concurrently; return ``{label: TaskRuntimeSnapshot}``.

    Per-project labels match the keys used elsewhere in the dashboard
    (project basename — see ``active_tasks._project_label``). Failures
    (unreachable, timeout, malformed payload) degrade that project's entry to
    ``TaskRuntimeSnapshot(offline=True)`` so the caller can render an honest
    ``—`` rather than a fabricated zero.
    """
    if not escalation_urls:
        return {}
    labels = list(escalation_urls.keys())
    urls = [escalation_urls[lbl] for lbl in labels]
    base_urls = [u.removesuffix('/mcp').rstrip('/') for u in urls]
    results = await asyncio.gather(
        *(_probe_one(client, base, per_call_timeout) for base in base_urls),
        return_exceptions=False,
    )
    return dict(zip(labels, results, strict=True))
