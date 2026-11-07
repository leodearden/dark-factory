"""Call a registered tool by name from an in-process test.

``call_tool`` fetches the tool from the server and calls its underlying
function, awaiting the result only when the tool body is ``async def``.

A plain importable module rather than a ``conftest.py`` fixture, for the same
reason as ``_escalation_http.py``.  The per-family modules
(``_filing_tools.py``, ``_merge_tool_calls.py``, ``_resolve_issue_call.py``)
keep their named wrappers as thin delegates over it.
"""

from __future__ import annotations

import inspect
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools import FunctionTool

__all__ = ['call_tool']


async def call_tool(server: FastMCP, name: str, /, **kwargs: Any) -> Any:
    tool = await server.get_tool(name)
    assert isinstance(tool, FunctionTool), (
        f'{name!r} is not registered on the server as a FunctionTool (got {tool!r})'
    )
    result = tool.fn(**kwargs)
    return await result if inspect.isawaitable(result) else result
