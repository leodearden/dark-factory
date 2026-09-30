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

USE THE COMPOSED FORM.  Exposing only the three ingredients left the failure
mode this module exists to close still reachable: a sixth call site could get
three of the four parts right — slash-less URL and ``follow_redirects`` but no
``Accept`` header — and still discard its payload, this time to a silent 406.
:func:`post_mcp_tool_call` and :func:`open_mcp_client` apply the four parts as
a unit, so they cannot be partially applied::

    from shared.mcp_post import open_mcp_client, post_mcp_tool_call

    async with open_mcp_client() as client:            # parts 3
        await post_mcp_tool_call(                      # parts 1, 2, 4
            client, self.mcp.url, 'add_memory', arguments,
            context=f'completion memory write for task {self.task_id}',
        )

The client is a SEPARATE call rather than folded into ``post_mcp_tool_call``
because three of the five call sites POST a batch and deliberately share one
connection pool across it (``_post_submit_tasks``' documented intent); a
primitive that constructed its own client per call would have silently undone
that.  ``post_mcp_tool_call`` warns if handed a client that is not following
redirects, so splitting them does not reopen the partial-application hole.

The ingredients stay public — :data:`MCP_POST_HEADERS`,
:func:`mcp_endpoint_url`, :func:`check_mcp_post_response`,
:func:`decode_mcp_response_body` — for a caller that genuinely needs a
non-``tools/call`` request or a hand-built envelope.

``orchestrator.mcp_lifecycle.McpSession`` (the session-handshake transport)
consumes :data:`MCP_POST_HEADERS`, :func:`mcp_endpoint_url` and
:func:`decode_mcp_response_body`, so this module is the single copy of each.

Public API::

    from shared.mcp_post import (
        MCP_POST_HEADERS,
        check_mcp_post_response,
        decode_mcp_response_body,
        mcp_endpoint_url,
        mcp_tool_call_payload,
        open_mcp_client,
        post_mcp_tool_call,
    )

The module is intentionally NOT re-exported from ``shared/__init__.py``.
Consumers import via the fully-qualified path (see above), consistent with the
``mcp_envelope``/``proc_group``/``config_dir`` sub-module convention.

``httpx`` is imported LAZILY, inside :func:`open_mcp_client`, so importing
this module does not pull httpx into every ``shared`` consumer — and so that
``patch('httpx.AsyncClient', ...)`` still intercepts client construction,
which is the seam every transport test in this repo relies on.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

__all__ = [
    'MCP_POST_HEADERS',
    'check_mcp_post_response',
    'decode_mcp_response_body',
    'mcp_endpoint_url',
    'mcp_tool_call_payload',
    'open_mcp_client',
    'post_mcp_tool_call',
]

logger = logging.getLogger(__name__)

#: Headers every raw MCP POST must send.  Without the ``Accept`` member the
#: server answers ``406 Not Acceptable`` and the payload is discarded (see the
#: module docstring for the measurement).  Copy with ``dict(MCP_POST_HEADERS)``
#: before mutating: this is a module-level singleton.
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
    different path than the server mounts.  ``McpSession.__init__``
    (``orchestrator/src/orchestrator/mcp_lifecycle.py``) builds its endpoint
    with this function.

    >>> mcp_endpoint_url('http://127.0.0.1:8002')
    'http://127.0.0.1:8002/mcp'
    >>> mcp_endpoint_url('http://127.0.0.1:8002/')
    'http://127.0.0.1:8002/mcp'
    """
    return f"{base_url.rstrip('/')}/mcp"


def decode_mcp_response_body(resp: Any) -> Any:
    """Decode an MCP response body sent as JSON or as SSE (Streamable HTTP).

    FastMCP may answer with ``text/event-stream`` instead of
    ``application/json``; for SSE the last ``data:`` frame wins.  An
    unlabelled or mislabelled body falls back to the SSE spelling.  RAISES
    ``ValueError`` when the body is neither JSON nor carries a ``data:`` line —
    :func:`check_mcp_post_response` is the never-raises wrapper.
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

        payload = decode_mcp_response_body(resp)

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


def mcp_tool_call_payload(
    tool: str, arguments: dict, *, request_id: Any = 1,
) -> dict:
    """Return the JSON-RPC envelope for an MCP ``tools/call``.

    Split out from :func:`post_mcp_tool_call` so a caller that must send the
    request some other way (a batching transport, a replay fixture) still gets
    the envelope from one place rather than re-typing four keys.
    """
    return {
        'jsonrpc': '2.0',
        'id': request_id,
        'method': 'tools/call',
        'params': {'name': tool, 'arguments': arguments},
    }


def open_mcp_client(**kwargs: Any) -> Any:
    """Return an ``httpx.AsyncClient`` configured for raw MCP POSTs.

    Part 3 of the four-part fix: ``follow_redirects=True``, defence in depth
    against a re-introduced slash.  Returned rather than yielded so the caller
    keeps the ``async with`` and therefore keeps the connection pool for a
    whole batch — which three of the five call sites deliberately do.

    ``httpx`` is imported here, not at module scope: it keeps ``shared`` free
    of an import-time httpx dependency, and it keeps ``AsyncClient`` a MODULE
    ATTRIBUTE resolved at call time, which is what makes
    ``patch('httpx.AsyncClient', ...)`` able to intercept every MCP client this
    repo constructs.  Binding it at import time here would silently blind every
    transport test.

    Extra *kwargs* are forwarded to ``AsyncClient``; ``follow_redirects`` is a
    default, so an explicit override is still possible (nothing needs one
    today).
    """
    import httpx

    kwargs.setdefault('follow_redirects', True)
    return httpx.AsyncClient(**kwargs)


async def post_mcp_tool_call(
    client: Any,
    base_url: str,
    tool: str,
    arguments: dict,
    *,
    context: str,
    request_id: Any = 1,
    timeout: float = 10,
) -> bool:
    """POST one MCP ``tools/call`` over *client* and inspect the answer.

    THE COMPOSED PRIMITIVE.  Applies parts 1, 2 and 4 of the fix together —
    slash-less URL, ``Accept`` header, loud response check — so a call site
    cannot get some of them and silently lose its payload to whichever one it
    missed.  Part 3 belongs to *client*; if that client is not following
    redirects this warns rather than proceeding quietly, which is the only way
    splitting client construction out stays safe.

    Never raises for a response-shaped reason (that is
    :func:`check_mcp_post_response`'s contract).  It DOES propagate a transport
    exception from ``client.post`` — every call site already wraps this in the
    ``try/except`` its fire-and-forget behaviour needs, and swallowing a
    connection error here would hide a server that is simply down.

    :param client: an ``httpx.AsyncClient`` (duck-typed).
    :param base_url: the server BASE (``http://127.0.0.1:8002``), not an endpoint.
    :param tool: MCP tool name, e.g. ``'add_memory'`` / ``'submit_task'``.
    :param arguments: the tool's ``arguments`` mapping.
    :param context: identifies the call site in any warning.
    :returns: True only if the call actually landed.
    """
    # ``True`` by default so a duck-typed stub with no such attribute (several
    # existing test doubles) does not produce a spurious warning.
    if not getattr(client, 'follow_redirects', True):
        logger.warning(
            'MCP POST client was built without follow_redirects=True [%s]; use '
            'shared.mcp_post.open_mcp_client so a re-introduced trailing slash '
            'cannot silently discard the payload.',
            context,
        )

    resp = await client.post(
        mcp_endpoint_url(base_url),
        headers=MCP_POST_HEADERS,
        json=mcp_tool_call_payload(tool, arguments, request_id=request_id),
        timeout=timeout,
    )
    return check_mcp_post_response(resp, context=context)
