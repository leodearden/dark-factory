"""The named invocation wrapper for the ``resolve_issue`` MCP tool.

This module names the tool; the call shape itself lives in
``_tool_calls.py::call_tool``, as it does for ``_merge_tool_calls.py`` and
``_filing_tools.py``.
"""

from __future__ import annotations

from typing import Any

from _tool_calls import call_tool
from fastmcp import FastMCP

__all__ = ['call_resolve_issue']


async def call_resolve_issue(server: FastMCP, **kwargs: Any) -> dict[str, Any]:
    """Invoke the server's ``resolve_issue`` tool with *kwargs*."""
    return await call_tool(server, 'resolve_issue', **kwargs)
