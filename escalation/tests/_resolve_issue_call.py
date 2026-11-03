"""The one invocation wrapper for the ``resolve_issue`` MCP tool.

``resolve_issue`` is a sync FastMCP tool, so a test fetches it from the server
and calls its underlying function directly; this module is the single home for
that shape, as ``_merge_tool_calls.py`` and ``_filing_tools.py`` are for theirs.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP
from fastmcp.tools import FunctionTool

__all__ = ['call_resolve_issue']


async def call_resolve_issue(server: FastMCP, **kwargs: Any) -> dict[str, Any]:
    """Invoke the server's ``resolve_issue`` tool with *kwargs*."""
    tool = await server.get_tool('resolve_issue')
    assert isinstance(tool, FunctionTool)
    return tool.fn(**kwargs)
