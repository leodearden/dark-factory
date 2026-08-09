"""Async functions for memory health metrics via fused-memory MCP HTTP endpoint.

Each function talks to one or more fused-memory MCP Streamable HTTP servers,
handling session initialization and SSE response parsing transparently.
Network errors are caught at the get_* level and returned as offline dicts.
"""

from __future__ import annotations

import json
import logging

import httpx
from shared.mcp_idempotency import maybe_inject_client_op_id

from dashboard.config import DashboardConfig
from dashboard.data.mcp_fanout import first_success

logger = logging.getLogger(__name__)

MCP_HEADERS = {
    'Content-Type': 'application/json',
    'Accept': 'application/json, text/event-stream',
}

# Twin of orchestrator.mcp_lifecycle's use of shared.mcp_idempotency (task
# 2712; hoisted into the shared module by task 2766 so the two McpSession
# twins share one frozenset + injection helper instead of drifting copies):
# mutating task tools get a client-supplied idempotency key so a retried
# write dedupes server-side instead of double-applying. Inert today (the
# dashboard issues only reads).
#
# CAVEAT (not yet write-safe by construction on this path): unlike the
# orchestrator twin, this _raw_call has NO transport retry loop, so the
# injection generates a FRESH key per invocation. The orchestrator's
# inject-once-before-the-retry-loop invariant — the property that makes
# transport retries reuse one key and dedupe — does NOT hold here. If the
# dashboard ever gains a mutating call that a higher layer retries by
# re-invoking _raw_call, that caller must thread one stable client_op_id
# across attempts (or a retry loop must be added here); a fresh per-attempt
# uuid4 would NOT trigger server-side dedup. Safe today only because reads
# never dedup and a caller-supplied key is preserved.


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

    async def _ensure_initialized(
        self, client: httpx.AsyncClient, timeout: float = 10,
    ) -> None:
        """Perform the initialize handshake once, under the caller's budget.

        *timeout* is threaded into both handshake posts so a caller working
        to a tight budget is not silently held to the 10s default while the
        session warms up.
        """
        if self._initialized:
            return
        await self._raw_call(client, 'initialize', {
            'protocolVersion': '2025-03-26',
            'capabilities': {},
            'clientInfo': {'name': 'dashboard', 'version': '0.1'},
        }, timeout=timeout)
        await self._raw_notify(client, 'notifications/initialized', timeout=timeout)
        self._initialized = True

    async def call_tool(
        self,
        client: httpx.AsyncClient,
        tool_name: str,
        arguments: dict,
        timeout: float = 10,
    ) -> dict:
        """Initialize (if needed), call a tool, return the inner result dict.

        *timeout* is a PER-HTTP-REQUEST budget applied to every post this
        performs — up to three on a cold session (see :func:`mcp_tool_call`).
        """
        await self._ensure_initialized(client, timeout=timeout)
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

        maybe_inject_client_op_id(method, params)

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
        timeout: float = 10,
    ) -> None:
        payload: dict = {'jsonrpc': '2.0', 'method': method}
        headers = dict(MCP_HEADERS)
        if self._session_id:
            headers['Mcp-Session-Id'] = self._session_id
        resp = await client.post(
            self.mcp_endpoint, json=payload, headers=headers, timeout=timeout,
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


def invalidate_session(url: str) -> None:
    """Remove a single cached MCP session, forcing re-initialisation on next call.

    Symmetric with reset_sessions(). Uses the same URL-normalisation rule
    (.rstrip('/')) as _get_session so callers need not strip manually.
    """
    _sessions.pop(url.rstrip('/'), None)


# ── Public API (backward-compatible return types) ──────────────────


async def mcp_tool_call(
    client: httpx.AsyncClient,
    base_url: str,
    tool_name: str,
    arguments: dict,
    timeout: float = 10,
) -> dict:
    """Make a JSON-RPC tools/call request via a cached MCP session.

    Raises on HTTP or connection errors (caller is expected to catch).

    *timeout* is a **per-HTTP-request** budget, not a whole-operation one. It
    is handed to ``client.post``, so it bounds connect/read/write **and pool
    acquisition** — the last of which is why threading it matters: a caller
    working to a 2.0s budget would otherwise still block up to httpx's 10s
    default waiting for a free connection slot.

    Because it is per-request, it does **not** bound this call as a whole: a
    *cold* session performs three posts (``initialize``,
    ``notifications/initialized``, then ``tools/call``), so the worst case is
    roughly ``3 * timeout`` plus the server's own think time. Callers needing
    a hard whole-operation bound must still wrap this in
    ``asyncio.wait_for`` — every existing probe caller does, deliberately, and
    the two layers are complementary rather than redundant.
    """
    session = _get_session(base_url)
    return await session.call_tool(client, tool_name, arguments, timeout=timeout)


async def _first_success(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    tool_name: str,
    args: dict,
    log_label: str,
    timeout: float = 10,
) -> dict:
    """Call an MCP tool on each configured URL; return the first success.

    On all-fail returns ``{'offline': True, 'error': '; '.join(errors)}``.
    This is correct for singleton-per-instance tools (e.g. ``get_status``,
    ``get_curator_state``); aggregating helpers (``get_queue_stats``,
    ``get_wal_status``) handle their own per-URL loops.

    *timeout* is the per-HTTP-request budget forwarded to each URL's
    ``mcp_tool_call``. ``first_success`` itself stays timeout-agnostic (it
    also serves tasks.py and metrics.py, which carry their own budgets) —
    the budget rides along on the ``call`` closure below.
    """
    return await first_success(
        config.fused_memory_urls,
        lambda url: mcp_tool_call(client, url, tool_name, args, timeout=timeout),
        log_label=log_label,
        offline_result=lambda errs: {'offline': True, 'error': '; '.join(errs)},
    )


async def get_memory_status(
    client: httpx.AsyncClient, config: DashboardConfig, timeout: float = 10,
) -> dict:
    """Fetch memory subsystem status, trying each configured URL.

    Returns the first successful status dict, or {offline: True, error: ...}.

    *timeout* is a per-HTTP-request budget (see :func:`mcp_tool_call`), NOT a
    bound on this call: the failover walks up to N URLs, each of which may
    perform a three-post cold-session handshake. Callers needing a hard bound
    must still wrap this in ``asyncio.wait_for`` — ``metrics.py`` does.
    """
    return await _first_success(
        client, config, 'get_status', {}, 'get_status', timeout=timeout,
    )


# Intentionally NOT converted to first_success: sums/collects across ALL
# configured URLs rather than short-circuiting on the first success.
async def get_queue_stats(
    client: httpx.AsyncClient, config: DashboardConfig, timeout: float = 10,
) -> dict:
    """Fetch and aggregate write-queue stats from all configured servers.

    Counts are summed; oldest_pending_age_seconds is the max across servers.

    *timeout* is a per-HTTP-request budget (see :func:`mcp_tool_call`), NOT a
    bound on this call: unlike a first-success failover this visits ALL N
    configured URLs, so the aggregate cost scales with N. Callers needing a
    hard bound must still wrap this in ``asyncio.wait_for`` — ``metrics.py``
    does.
    """
    merged_counts: dict[str, int] = {}
    oldest_age: float | None = None
    any_success = False

    for url in config.fused_memory_urls:
        try:
            result = await mcp_tool_call(
                client, url, 'get_queue_stats', {}, timeout=timeout,
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError,
                ValueError) as e:
            logger.debug('get_queue_stats failed for %s: %s', url, e)
            invalidate_session(url)
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


# Intentionally NOT converted to first_success: collects a per-URL entry from
# ALL configured URLs rather than short-circuiting on the first success.
async def get_wal_status(client: httpx.AsyncClient, config: DashboardConfig) -> dict:
    """Fetch per-store WAL checkpoint status from each fused-memory server.

    Returns ``{'stores': {server_url: {store_name: row, ...}}}`` — one
    column per server, one row per SQLite store. The frontend renders
    these as a small badge in the memory panel (red on ``busy>0`` or
    stale ``ts``, amber on missing rows, green otherwise).

    Returns ``{'offline': True, 'error': ...}`` if all configured servers
    are unreachable. Added 2026-05-14 in response to the 2026-05-13
    task-DB-loss incident — see ``docs/task-recovery-2026-05-13/``.
    """
    per_server: dict[str, dict] = {}
    errors: list[str] = []
    for url in config.fused_memory_urls:
        try:
            result = await mcp_tool_call(client, url, 'get_wal_status', {})
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError,
                ValueError) as e:
            logger.debug('get_wal_status failed for %s: %s', url, e)
            invalidate_session(url)
            errors.append(f'{url}: {e}')
            continue
        per_server[url] = result.get('stores') or {}

    if not per_server:
        return {'offline': True, 'error': '; '.join(errors)}
    return {'stores': per_server}


async def get_curator_state(client: httpx.AsyncClient, config: DashboardConfig) -> dict:
    """Fetch the curator UsageGate state from the fused-memory server.

    Returns the first successful result from any configured URL; on all-fail
    returns ``{'offline': True, 'error': ...}``. First-success semantics are
    correct because the curator UsageGate lives on a single fused-memory
    instance per the singleton-lock invariant.

    Result shape (on success):
      ``{'paused': bool, 'paused_reason': str | None,
         'soonest_open_at': str | None, 'account_count': int}``
    """
    return await _first_success(
        client, config, 'get_curator_state', {}, 'get_curator_state'
    )
