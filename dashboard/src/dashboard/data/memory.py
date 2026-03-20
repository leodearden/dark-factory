"""Async functions for memory health metrics via fused-memory MCP HTTP endpoint.

Each function talks to one or more fused-memory MCP Streamable HTTP servers,
handling session initialization and SSE response parsing transparently.
Network errors are caught at the get_* level and returned as offline dicts.
"""

from __future__ import annotations

import json
import logging

import httpx

from dashboard.config import DashboardConfig

logger = logging.getLogger(__name__)

MCP_HEADERS = {
    'Content-Type': 'application/json',
    'Accept': 'application/json, text/event-stream',
}


def _parse_mcp_response(resp: httpx.Response) -> dict:
    """Parse an MCP JSON-RPC response (JSON or SSE)."""
    content_type = resp.headers.get('content-type', '')
    if 'text/event-stream' in content_type:
        return _parse_sse_response(resp.text)
    try:
        return resp.json()
    except (json.JSONDecodeError, ValueError):
        return _parse_sse_response(resp.text)


def _parse_sse_response(text: str) -> dict:
    """Extract the last ``data:`` line from an SSE response and parse as JSON."""
    last_data = None
    for line in text.split('\n'):
        if line.startswith('data: '):
            last_data = line[6:]
        elif line.startswith('data:'):
            last_data = line[5:]
    if last_data:
        return json.loads(last_data)
    raise ValueError(f'No data line in SSE response: {text[:200]}')


def _extract_tool_result(rpc_response: dict) -> dict:
    """Pull the inner dict out of a JSON-RPC tools/call result."""
    content = rpc_response.get('result', {}).get('content', [])
    if not content:
        return {}
    first = content[0]
    text = first.get('text', '') if isinstance(first, dict) else ''
    if not text:
        return {}
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.warning('MCP inner-text parse error', exc_info=True)
        return {}


class McpSession:
    """Lightweight MCP Streamable HTTP session with automatic initialization."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.mcp_endpoint = f'{self.base_url}/mcp'
        self._session_id: str | None = None
        self._initialized = False
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _ensure_initialized(self, client: httpx.AsyncClient) -> None:
        if self._initialized:
            return
        await self._raw_call(client, 'initialize', {
            'protocolVersion': '2025-03-26',
            'capabilities': {},
            'clientInfo': {'name': 'dashboard', 'version': '0.1'},
        })
        await self._raw_notify(client, 'notifications/initialized')
        self._initialized = True

    async def call_tool(
        self,
        client: httpx.AsyncClient,
        tool_name: str,
        arguments: dict,
        timeout: float = 10,
    ) -> dict:
        """Initialize (if needed), call a tool, return the inner result dict."""
        await self._ensure_initialized(client)
        rpc = await self._raw_call(
            client, 'tools/call',
            {'name': tool_name, 'arguments': arguments},
            timeout=timeout,
        )
        return _extract_tool_result(rpc)

    async def _raw_call(
        self,
        client: httpx.AsyncClient,
        method: str,
        params: dict | None = None,
        timeout: float = 10,
    ) -> dict:
        payload: dict = {'jsonrpc': '2.0', 'id': self._next_id(), 'method': method}
        if params is not None:
            payload['params'] = params

        headers = dict(MCP_HEADERS)
        if self._session_id:
            headers['Mcp-Session-Id'] = self._session_id

        resp = await client.post(
            self.mcp_endpoint, json=payload, headers=headers, timeout=timeout,
        )
        resp.raise_for_status()

        if sid := resp.headers.get('mcp-session-id'):
            self._session_id = sid

        return _parse_mcp_response(resp)

    async def _raw_notify(
        self,
        client: httpx.AsyncClient,
        method: str,
    ) -> None:
        payload: dict = {'jsonrpc': '2.0', 'method': method}
        headers = dict(MCP_HEADERS)
        if self._session_id:
            headers['Mcp-Session-Id'] = self._session_id
        resp = await client.post(
            self.mcp_endpoint, json=payload, headers=headers, timeout=10,
        )
        if resp.status_code not in (200, 202, 204):
            logger.warning('MCP notify %s returned %s', method, resp.status_code)


# Session cache — one per base URL, reused across poll cycles.
_sessions: dict[str, McpSession] = {}


def _get_session(base_url: str) -> McpSession:
    base_url = base_url.rstrip('/')
    if base_url not in _sessions:
        _sessions[base_url] = McpSession(base_url)
    return _sessions[base_url]


def reset_sessions() -> None:
    """Clear cached sessions (useful in tests)."""
    _sessions.clear()


# ── Public API (backward-compatible return types) ──────────────────


async def mcp_tool_call(
    client: httpx.AsyncClient,
    base_url: str,
    tool_name: str,
    arguments: dict,
) -> dict:
    """Make a JSON-RPC tools/call request via a cached MCP session.

    Raises on HTTP or connection errors (caller is expected to catch).
    """
    session = _get_session(base_url)
    return await session.call_tool(client, tool_name, arguments)


async def get_memory_status(client: httpx.AsyncClient, config: DashboardConfig) -> dict:
    """Fetch memory subsystem status, trying each configured URL.

    Returns the first successful status dict, or {offline: True, error: ...}.
    """
    errors: list[str] = []
    for url in config.fused_memory_urls:
        try:
            return await mcp_tool_call(
                client, url, 'get_status',
                {'project_id': config.fused_memory_project_id},
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError,
                ValueError) as e:
            logger.debug('get_status failed for %s: %s', url, e)
            errors.append(f'{url}: {e}')
            # Invalidate stale session so next poll retries init
            _sessions.pop(url.rstrip('/'), None)
    return {'offline': True, 'error': '; '.join(errors)}


async def get_queue_stats(client: httpx.AsyncClient, config: DashboardConfig) -> dict:
    """Fetch and aggregate write-queue stats from all configured servers.

    Counts are summed; oldest_pending_age_seconds is the max across servers.
    """
    merged_counts: dict[str, int] = {}
    oldest_age: float | None = None
    any_success = False

    for url in config.fused_memory_urls:
        try:
            result = await mcp_tool_call(client, url, 'get_queue_stats', {})
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError,
                ValueError) as e:
            logger.debug('get_queue_stats failed for %s: %s', url, e)
            _sessions.pop(url.rstrip('/'), None)
            continue

        any_success = True
        for key, val in result.get('counts', {}).items():
            merged_counts[key] = merged_counts.get(key, 0) + (val or 0)

        age = result.get('oldest_pending_age_seconds')
        if age is not None and (oldest_age is None or age > oldest_age):
            oldest_age = age

    if not any_success:
        return {'offline': True, 'error': 'All servers unreachable'}
    return {'counts': merged_counts, 'oldest_pending_age_seconds': oldest_age}
