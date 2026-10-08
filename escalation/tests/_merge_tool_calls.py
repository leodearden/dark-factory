"""Shared invocation wrappers for the three merge MCP tools.

``merge_status``, ``merge_request`` and ``merge_cancel`` are FastMCP tools, so a
test cannot call them directly: it must fetch the registered tool from the
server and invoke its underlying function
(``(await server.get_tool('merge_status')).fn(**kwargs)``).  This module is the
one home for that two-step shape: every escalation suite that drives a merge
tool imports these wrappers instead of re-typing them.

NOT a ``conftest.py`` fixture: these are plain callables taking arguments, not
per-test state, and ``escalation/tests`` is on ``sys.path`` under pytest's
default prepend import mode (its ``conftest.py`` also inserts the directory
explicitly), so a flat import works from every collection configuration — the
same shape and the same reasoning as ``_scan_race_helpers.py`` and
``_escalation_http.py``.

``server`` is deliberately left unannotated.  ``create_server`` returns a
``FastMCP``, whose ``get_tool`` is typed to return a ``Tool`` whose ``fn`` is a
bare ``Callable`` — calling it with keyword arguments is what forced the
``# type: ignore[reportAttributeAccessIssue]`` comments on the direct call sites
in ``test_merge_state_wiring.py``.  Keeping the parameter untyped keeps the
suppressions out of the shared helper rather than spreading them.
"""

from __future__ import annotations

from typing import Any

__all__ = ['call_merge_cancel', 'call_merge_request', 'call_merge_status']


async def call_merge_status(server, **kwargs: Any) -> dict[str, Any]:
    """Invoke the server's ``merge_status`` tool with *kwargs*."""
    tool = await server.get_tool('merge_status')
    return await tool.fn(**kwargs)


async def call_merge_request(server, **kwargs: Any) -> dict[str, Any]:
    """Invoke the server's ``merge_request`` tool with *kwargs*."""
    tool = await server.get_tool('merge_request')
    return await tool.fn(**kwargs)


async def call_merge_cancel(server, **kwargs: Any) -> dict[str, Any]:
    """Invoke the server's ``merge_cancel`` tool with *kwargs*."""
    tool = await server.get_tool('merge_cancel')
    return await tool.fn(**kwargs)
