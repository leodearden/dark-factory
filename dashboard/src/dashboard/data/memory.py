"""Async functions for memory health metrics via fused-memory MCP HTTP endpoint.

Each function makes a JSON-RPC call to the fused-memory MCP server and returns
structured data. Network errors are handled at the caller level (get_* functions),
not in the generic mcp_tool_call transport.
"""

from __future__ import annotations

import json

import httpx


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

    data = resp.json()
    content = data.get('result', {}).get('content', [])
    if not content:
        return {}

    text = content[0].get('text', '')
    if not text:
        return {}

    return json.loads(text)
