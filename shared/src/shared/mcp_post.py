"""Loud-and-safe primitives for raw ``httpx`` POSTs to an MCP server.

WHY THIS MODULE EXISTS (task 4023).  Five near-identical raw-POST blocks in
the orchestrator built their target URL as ``f'{base_url}/mcp/'`` — WITH a
trailing slash — on a bare ``httpx.AsyncClient``.  The fused-memory server
mounts its Streamable-HTTP endpoint at ``/mcp`` and answers ``/mcp/`` with a
``307 Temporary Redirect``; a bare ``httpx.AsyncClient`` has
``follow_redirects=False``, so the POST body was never delivered.  A 307 is a
perfectly successful HTTP exchange, so nothing raised and nothing logged: the
writes were silently discarded for months.  ``.mcp.json`` declares both
servers slash-less (``http://127.0.0.1:8002/mcp``) — that is the canonical
form this module produces.

The fix has FOUR parts and every call site needs all four:

1. build the URL with :func:`mcp_endpoint_url` (drops the slash — load-bearing),
2. send :data:`MCP_POST_HEADERS` (the ``Accept`` header is equally load-bearing
   — see below),
3. construct the client with ``follow_redirects=True`` (defence in depth),
4. hand the response to :func:`check_mcp_post_response` (makes the NEXT
   transport failure loud instead of silent).

WHY THE ``Accept`` HEADER IS NOT OPTIONAL.  Measured against the live server
(127.0.0.1:8002) while fixing this: dropping the slash alone is NECESSARY BUT
NOT SUFFICIENT.  The five defect sites also sent no ``Accept`` header, so once
the request reaches ``/mcp`` the server answers ``406 Not Acceptable``
(``{"error": {"code": -32600, "message": "Not Acceptable: Client must accept
application/json"}}``) — and following the 307 changes nothing, because the
redirected request carries the same headers.  Fixing only the slash would move
the failure from a silent 307 no-op to a silent 406 no-op.  With
:data:`MCP_POST_HEADERS` the same POST returns ``200`` and a real JSON-RPC
result, with no ``initialize``/session handshake needed.

Public API::

    from shared.mcp_post import (
        MCP_POST_HEADERS,
        check_mcp_post_response,
        mcp_endpoint_url,
    )

The module is intentionally NOT re-exported from ``shared/__init__.py``.
Consumers import via the fully-qualified path (see above), consistent with the
``mcp_envelope``/``proc_group``/``config_dir`` sub-module convention.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

__all__ = [
    'MCP_POST_HEADERS',
    'check_mcp_post_response',
    'mcp_endpoint_url',
]

logger = logging.getLogger(__name__)

#: Headers every raw MCP POST must send.  Byte-identical to
#: ``orchestrator.mcp_lifecycle.MCP_HEADERS`` — the already-correct pattern the
#: five defect sites never adopted.  Without the ``Accept`` member the server
#: answers ``406 Not Acceptable`` and the payload is discarded (see the module
#: docstring for the measurement).  Copy with ``dict(MCP_POST_HEADERS)`` before
#: mutating: this is a module-level singleton.
MCP_POST_HEADERS = {
    'Content-Type': 'application/json',
    'Accept': 'application/json, text/event-stream',
}


def mcp_endpoint_url(base_url: str) -> str:
    """Return the canonical, slash-less MCP endpoint for *base_url*.

    ``base_url`` is a SERVER BASE (``http://127.0.0.1:8002``), not an endpoint:
    that is the contract of ``self.mcp.url`` at every orchestrator call site.

    The ``rstrip('/')`` is load-bearing, not cosmetic — a configured base that
    already ends in ``/`` would otherwise produce ``…//mcp``, which is a
    different path than the server mounts.  Mirrors the canonicalization
    ``McpSession.__init__`` (``orchestrator/mcp_lifecycle.py``) has been doing
    correctly all along; the defect sites simply never used it.

    >>> mcp_endpoint_url('http://127.0.0.1:8002')
    'http://127.0.0.1:8002/mcp'
    >>> mcp_endpoint_url('http://127.0.0.1:8002/')
    'http://127.0.0.1:8002/mcp'
    """
    return f"{base_url.rstrip('/')}/mcp"


def _decode_body(resp: Any) -> Any:
    """Decode a JSON or SSE response body.

    Mirrors ``McpSession._parse_response`` — FastMCP may answer a Streamable
    HTTP POST with ``text/event-stream`` instead of ``application/json``, and a
    naive ``resp.json()`` would then warn on every SUCCESSFUL write.  Raises on
    an undecodable body; the sole caller turns that into a warning.
    """
    content_type = str(resp.headers.get('content-type', ''))
    if 'text/event-stream' in content_type:
        return _parse_sse(resp.text)
    try:
        return resp.json()
    except Exception:
        # Unlabelled or mislabelled body — try the SSE spelling before giving up.
        return _parse_sse(resp.text)


def _parse_sse(text: str) -> Any:
    """Extract the JSON-RPC payload from the last ``data:`` line of an SSE body."""
    last_data = None
    for line in str(text).split('\n'):
        if line.startswith('data: '):
            last_data = line[6:]
        elif line.startswith('data:'):
            last_data = line[5:]
    if last_data:
        return json.loads(last_data)
    raise ValueError(f'no data line in SSE body: {str(text)[:200]!r}')


def check_mcp_post_response(resp: Any, *, context: str) -> bool:
    """Inspect a raw MCP POST response.  Return True only if the call landed.

    NEVER RAISES — that is the contract, not an implementation detail.  All
    five call sites are deliberately fire-and-forget (``_post_submit_tasks``
    runs under ``asyncio.create_task``; the memory writes are best-effort side
    channels whose failure must not fail a task).  A checker that raised on a
    duck-typed or malformed response would convert a silent no-op into a new
    failure mode.  Every abnormal shape instead produces a ``logger.warning``
    naming *context*, so the next transport failure is loud rather than
    invisible — which is the whole point of this task.

    Checks in order:

    1. **3xx** — a redirect that was NOT followed, i.e. THIS TASK'S defect.
       It gets its own distinct message rather than folding into a generic
       transport warning, because a redirect is a *successful* HTTP exchange
       and is therefore the failure mode most likely to go unnoticed again.
    2. **>=400** — an outright rejection (``406`` when the ``Accept`` header is
       missing, ``5xx`` when the server is unwell).
    3. **JSON-RPC ``error`` member** — HTTP 200 with an application-level
       failure.  ``{'error': None}`` is the JSON-RPC absent-error spelling and
       is NOT a failure.

    :param resp: an ``httpx.Response`` (duck-typed; any shape is tolerated).
    :param context: identifies the call site, interpolated into every message.
    :returns: True on a 2xx carrying no JSON-RPC error, False otherwise.
    """
    try:
        status = int(resp.status_code)

        if 300 <= status < 400:
            location = 'unknown'
            with contextlib.suppress(Exception):
                location = resp.headers.get('location', 'unknown') or 'unknown'
            logger.warning(
                'MCP POST got redirect %d (Location: %s) and it was NOT followed — '
                'the payload was silently discarded [%s]. The endpoint is /mcp '
                '(no trailing slash); build it with shared.mcp_post.mcp_endpoint_url '
                'and pass follow_redirects=True.',
                status, location, context,
            )
            return False

        if status >= 400:
            body = ''
            with contextlib.suppress(Exception):
                body = str(resp.text)[:200]
            logger.warning(
                'MCP POST failed with HTTP %d [%s]: %s', status, context, body,
            )
            return False

        payload = _decode_body(resp)

        if isinstance(payload, dict):
            error = payload.get('error')
            if error is not None:
                logger.warning(
                    'MCP POST returned a JSON-RPC error [%s]: %s', context, error,
                )
                return False

        return True

    except Exception as exc:
        # Covers a duck-typed/None response, an undecodable body, and any
        # shape not anticipated above.  Loud, but never fatal.
        logger.warning(
            'MCP POST response could not be inspected [%s]: %s: %s',
            context, type(exc).__name__, exc,
        )
        return False
