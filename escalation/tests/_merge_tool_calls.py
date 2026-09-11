"""Shared invocation wrappers for the three merge MCP tools.

``merge_status``, ``merge_request`` and ``merge_cancel`` are FastMCP tools, so a
test cannot call them directly: it must fetch the registered tool from the
server and invoke its underlying function
(``(await server.get_tool('merge_status')).fn(**kwargs)``).  That two-step shape
had been re-typed verbatim in every suite that drives a merge tool — four copies
of the ``merge_status`` wrapper and three each of the other two, across
``test_server.py``, ``test_server_chokepoint.py``,
``test_merge_status_git_authority.py`` and ``test_merge_state_wiring.py``.

This module is the shared home, added by task 4829's amendment pass so the copy
count stops growing.  MIGRATING the existing call sites is deliberately NOT part
of it: those four test modules are outside this task's declared scope, and a
sweep of them belongs in its own change where the diff can be reviewed as one.
New merge-tool tests should import from here.

NOT a ``conftest.py`` fixture, for two reasons.  The mechanical one: these are
plain callables taking arguments, not per-test state, and ``escalation/tests``
is on ``sys.path`` under pytest's default prepend import mode (its
``conftest.py`` also inserts the directory explicitly), so a flat import works
from every collection configuration — the same shape and the same reasoning as
``_scan_race_helpers.py`` and ``_escalation_http.py``.  The scope one:
``escalation/tests/conftest.py``, which the review named as the natural home, is
an existing file this task holds no lock on.

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
