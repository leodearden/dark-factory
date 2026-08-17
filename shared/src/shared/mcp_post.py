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

The fix has three halves and every call site needs all three:

1. build the URL with :func:`mcp_endpoint_url` (drops the slash — load-bearing),
2. construct the client with ``follow_redirects=True`` (defence in depth),
3. hand the response to :func:`check_mcp_post_response` (makes the NEXT
   transport failure loud instead of silent).

Public API::

    from shared.mcp_post import check_mcp_post_response, mcp_endpoint_url

The module is intentionally NOT re-exported from ``shared/__init__.py``.
Consumers import via the fully-qualified path (see above), consistent with the
``mcp_envelope``/``proc_group``/``config_dir`` sub-module convention.
"""

from __future__ import annotations

import logging

__all__ = [
    'mcp_endpoint_url',
]

logger = logging.getLogger(__name__)


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
