"""Shared invocation wrappers for the merge MCP tools.

``merge_status``, ``merge_request``, ``merge_cancel`` and ``get_merge_queue``
are FastMCP tools, so a test cannot call them directly.  This module names that
family's tools for every escalation suite that drives one; the call shape
itself lives in ``_tool_calls.py::call_tool``.

NOT a ``conftest.py`` fixture: these are plain callables taking arguments, not
per-test state, and ``escalation/tests`` is on ``sys.path`` under pytest's
default prepend import mode (its ``conftest.py`` also inserts the directory
explicitly), so a flat import works from every collection configuration — the
same shape and the same reasoning as ``_scan_race_helpers.py`` and
``_escalation_http.py``.
"""

from __future__ import annotations

from typing import Any

from _tool_calls import call_tool
from fastmcp import FastMCP

__all__ = [
    'call_get_merge_queue', 'call_merge_cancel', 'call_merge_request', 'call_merge_status',
]


async def call_merge_status(server: FastMCP, **kwargs: Any) -> dict[str, Any]:
    """Invoke the server's ``merge_status`` tool with *kwargs*."""
    return await call_tool(server, 'merge_status', **kwargs)


async def call_merge_request(server: FastMCP, **kwargs: Any) -> dict[str, Any]:
    """Invoke the server's ``merge_request`` tool with *kwargs*."""
    return await call_tool(server, 'merge_request', **kwargs)


async def call_merge_cancel(server: FastMCP, **kwargs: Any) -> dict[str, Any]:
    """Invoke the server's ``merge_cancel`` tool with *kwargs*."""
    return await call_tool(server, 'merge_cancel', **kwargs)


async def call_get_merge_queue(server: FastMCP) -> dict[str, Any]:
    """Invoke the server's ``get_merge_queue`` tool."""
    return await call_tool(server, 'get_merge_queue')
