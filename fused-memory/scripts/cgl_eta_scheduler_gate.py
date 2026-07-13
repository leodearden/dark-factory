#!/usr/bin/env python3
"""CGL-η scheduler gate — halt/resume BOTH the reify and dark_factory
orchestrator schedulers around the live bulk migration.

The bulk cross-graph apply writes into the reify and dark_factory FalkorDB
graphs (the two largest foreign populations). Halting both projects' schedulers
for the duration keeps their agents from bursting concurrent memory writes onto
those same graphs while the migration runs — bounding the load well under
FalkorDB's MAX_QUEUED_QUERIES cap. The apply itself is already safe under
concurrent load (uuid-indexed writes, no write-query clock timeout, lean shim /
no 2nd MemoryService, create-before-delete); this is defence-in-depth requested
by the operator, NOT a correctness precondition — so `halt` is best-effort and
non-fatal (a project whose orchestrator is down or unreachable is logged and
skipped, the migration still proceeds).

`resume` is likewise best-effort but ALWAYS attempts every endpoint, so a
wrapper `trap` can call it unconditionally on any exit path and never strand a
scheduler halted.

Usage:
    cgl_eta_scheduler_gate.py halt   [--reason "..."]
    cgl_eta_scheduler_gate.py resume [--reason "..."]

Endpoints default to reify (8100) + dark_factory (8102); override with
$CGL_SCHED_ENDPOINTS as a comma list of "name=url" pairs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid

import httpx

# name -> escalation MCP base url. Ports per orchestrator.yaml scheme
# (reify=xxx0, dark-factory=xxx2); each project's own escalation server owns
# its own scheduler, so halting both means one call to each.
DEFAULT_ENDPOINTS = {
    'reify': 'http://127.0.0.1:8100',
    'dark_factory': 'http://127.0.0.1:8102',
}


def _endpoints() -> dict[str, str]:
    raw = os.environ.get('CGL_SCHED_ENDPOINTS', '').strip()
    if not raw:
        return dict(DEFAULT_ENDPOINTS)
    out: dict[str, str] = {}
    for pair in raw.split(','):
        pair = pair.strip()
        if not pair:
            continue
        name, _, url = pair.partition('=')
        out[name.strip()] = url.strip()
    return out


def _log(msg: str) -> None:
    print(f'[cgl-sched-gate] {msg}', flush=True)


class McpClient:
    """Minimal HTTP/JSON-RPC client for an escalation MCP server."""

    def __init__(self, url: str, transport: httpx.AsyncBaseTransport | None = None):
        self._url = url.rstrip('/')
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._transport = transport

    async def __aenter__(self) -> McpClient:
        self._client = httpx.AsyncClient(
            timeout=30.0, follow_redirects=True, transport=self._transport,
        )
        try:
            # No session id on the FIRST request: the MCP streamable-HTTP contract
            # requires `initialize` to be sent session-less. A STATEFUL server
            # (e.g. the escalation servers) 404s "Session not found" if a client
            # invents its own id here. The server-assigned id is captured from the
            # initialize response in `_post` below and reused from then on.
            await self._post({
                'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
                'params': {'protocolVersion': '2024-11-05',
                           'clientInfo': {'name': 'cgl-sched-gate', 'version': '1.0'},
                           'capabilities': {}},
            })
            await self._post({'jsonrpc': '2.0', 'method': 'notifications/initialized', 'params': {}})
        except Exception:
            # __aexit__ is never called if __aenter__ raises, so close the
            # just-created client ourselves to avoid leaking the connection
            # pool (e.g. when initialize 404s against a stateful server).
            await self._client.aclose()
            raise
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def _post(self, payload: dict) -> dict:
        assert self._client is not None
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
        }
        if self._session_id is not None:
            headers['mcp-session-id'] = self._session_id
        resp = await self._client.post(f'{self._url}/mcp', json=payload, headers=headers)
        resp.raise_for_status()
        # Capture the server-assigned session id (returned on `initialize`) so
        # it can be reused on every subsequent request. Must happen before the
        # 202/empty-content early return below, since `initialize` responses
        # carry a JSON body but `notifications/initialized` may not.
        sid = resp.headers.get('mcp-session-id')
        if sid:
            self._session_id = sid
        if resp.status_code == 202 or not resp.content:
            return {}
        if 'text/event-stream' in resp.headers.get('content-type', ''):
            for line in resp.text.splitlines():
                if line.startswith('data:'):
                    return json.loads(line[5:].strip())
            raise RuntimeError(f'no SSE data line: {resp.text[:200]}')
        return resp.json()

    async def call_tool(self, name: str, arguments: dict) -> dict:
        result = await self._post({
            'jsonrpc': '2.0', 'id': uuid.uuid4().hex, 'method': 'tools/call',
            'params': {'name': name, 'arguments': arguments},
        })
        if 'error' in result:
            raise RuntimeError(f'{name} failed: {result["error"]}')
        content = result.get('result', {})
        if 'structuredContent' in content:
            return content['structuredContent']
        for entry in content.get('content', []) or []:
            if entry.get('type') == 'text':
                try:
                    return json.loads(entry['text'])
                except json.JSONDecodeError:
                    return {'_raw': entry['text']}
        return content


async def _one(name: str, url: str, tool: str, reason: str) -> bool:
    """Call halt/resume on one endpoint. Returns True on a clean apply."""
    try:
        async with McpClient(url) as client:
            res = await client.call_tool(tool, {'reason': reason})
        if res.get('error'):
            _log(f'{name} ({url}): {tool} -> error: {res["error"]}')
            return False
        key = 'halted' if tool == 'halt_scheduler' else 'resumed'
        _log(f'{name} ({url}): {tool} ok ({key}={res.get(key)}, was_paused={res.get("was_paused")})')
        return bool(res.get(key))
    except Exception as exc:  # unreachable server, timeout, etc. — non-fatal
        _log(f'{name} ({url}): {tool} UNREACHABLE/failed: {exc!r}')
        return False


async def main_async(action: str, reason: str) -> int:
    tool = 'halt_scheduler' if action == 'halt' else 'resume_scheduler'
    endpoints = _endpoints()
    results = await asyncio.gather(
        *[_one(name, url, tool, reason) for name, url in endpoints.items()]
    )
    ok = sum(1 for r in results if r)
    _log(f'{action}: {ok}/{len(endpoints)} scheduler(s) confirmed.')
    # Best-effort by design: NEVER fail the wrapper on a halt/resume miss — the
    # migration is safe without the halt, and a resume miss is surfaced loudly
    # for the operator rather than aborting cleanup.
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['halt', 'resume'])
    ap.add_argument('--reason', default='CGL-eta Phase-1 bulk cross-graph migration (auto-apply)')
    args = ap.parse_args()
    return asyncio.run(main_async(args.action, args.reason))


if __name__ == '__main__':
    sys.exit(main())
