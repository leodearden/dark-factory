"""Fan-out probe of per-orchestrator merge-queue halt status.

Each orchestrator runs its own escalation MCP, and that MCP owns the
merge-queue halt state for its orchestrator. To surface halt status on
the dashboard we fan out ``get_merge_halt_status`` calls across every
known escalation URL concurrently and aggregate the results into a
``{project_label: {wired, halted, owner_esc_id, offline}}`` dict.

The default ``mcp_tool_call`` timeout is 10s — too slow for a 3s polling
loop. We wrap each call in ``asyncio.wait_for`` so worst-case latency is
capped close to ``per_call_timeout`` even when every orchestrator is down.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from dashboard.data.memory import mcp_tool_call

logger = logging.getLogger(__name__)

DEFAULT_PER_CALL_TIMEOUT = 2.0


async def _probe_one(
    client: httpx.AsyncClient,
    base_url: str,
    timeout: float,
) -> dict[str, Any]:
    try:
        result = await asyncio.wait_for(
            mcp_tool_call(client, base_url, 'get_merge_halt_status', {}),
            timeout=timeout,
        )
    except (asyncio.TimeoutError, httpx.ConnectError, httpx.TimeoutException,
            httpx.HTTPStatusError, ValueError, OSError) as exc:
        logger.debug('get_merge_halt_status failed for %s: %s', base_url, exc)
        return {'wired': False, 'halted': False, 'offline': True, 'error': str(exc)}
    return {
        'wired': bool(result.get('wired', True)),
        'halted': bool(result.get('halted', False)),
        'owner_esc_id': result.get('owner_esc_id'),
        'offline': False,
    }


async def get_merge_halt_status(
    client: httpx.AsyncClient,
    escalation_urls: dict[str, str],
    *,
    per_call_timeout: float = DEFAULT_PER_CALL_TIMEOUT,
) -> dict[str, dict[str, Any]]:
    """Probe every escalation URL concurrently; return ``{label: status}``.

    Per-project labels match the keys used by ``shape_merge_queue`` (project
    basename). Failures are reported as ``{wired, halted: False, offline: True}``
    so the UI can render an Offline pill rather than a blank panel.
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
