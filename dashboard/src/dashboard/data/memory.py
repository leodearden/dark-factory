"""Async functions for memory health metrics via fused-memory MCP HTTP endpoint.

Each function makes a JSON-RPC call to the fused-memory MCP server and returns
structured data. Network errors are handled at the caller level (get_* functions),
not in the generic mcp_tool_call transport.
"""

from __future__ import annotations

import json

import httpx

from dashboard.config import DashboardConfig


async def mcp_tool_call(
    client: httpx.AsyncClient,
    base_url: str,
    tool_name: str,
    arguments: dict,
) -> dict:
    """Make a JSON-RPC tools/call request to the fused-memory MCP endpoint.

    Posts to {base_url}/mcp/ with a JSON-RPC 2.0 envelope, parses the MCP
    response content, and returns the inner dict.

    Raises:
        ValueError: If the HTTP status is not 200.
        httpx.TimeoutException: If the request times out (propagated).
        httpx.ConnectError: If the connection fails (propagated).
    """
    resp = await client.post(
        f'{base_url}/mcp/',
        json={
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'tools/call',
            'params': {
                'name': tool_name,
                'arguments': arguments,
            },
        },
        timeout=10,
    )

    if resp.status_code != 200:
        raise ValueError(f'MCP call {tool_name} returned non-200 status: {resp.status_code}')

    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError):
        return {}
    content = data.get('result', {}).get('content', [])
    if not content:
        return {}

    text = content[0].get('text', '')
    if not text:
        return {}

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}


async def get_memory_status(client: httpx.AsyncClient, config: DashboardConfig) -> dict:
    """Fetch memory subsystem status from the fused-memory MCP server.

    Returns the status dict on success, or {offline: True, error: str} on
    connection/timeout/server-error failure.
    """
    try:
        return await mcp_tool_call(
            client,
            config.fused_memory_url,
            'get_status',
            {'project_id': config.fused_memory_project_id},
        )
    except (httpx.ConnectError, httpx.TimeoutException, ValueError) as e:
        return {'offline': True, 'error': str(e)}


async def get_queue_stats(client: httpx.AsyncClient, config: DashboardConfig) -> dict:
    """Fetch write queue statistics from the fused-memory MCP server.

    Returns the queue stats dict on success, or {offline: True, error: str} on
    connection/timeout/server-error failure.
    """
    try:
        return await mcp_tool_call(
            client,
            config.fused_memory_url,
            'get_queue_stats',
            {},
        )
    except (httpx.ConnectError, httpx.TimeoutException, ValueError) as e:
        return {'offline': True, 'error': str(e)}
