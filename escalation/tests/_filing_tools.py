"""Named wrappers for the escalation-filing MCP tools.

``call_blocker``, ``call_info`` and ``call_promote`` name ``escalate_blocker``,
``escalate_info`` and ``promote_to_l2``; the call shape itself lives in
``_tool_calls.py::call_tool``.

A uniquely-named sibling support module rather than a ``conftest.py`` fixture,
matching ``_escalation_http.py``: these are plain callables with no setup or
teardown, so a fixture would buy nothing and would cost every caller an
argument.  ``escalation/tests/conftest.py`` puts this directory on ``sys.path``
in every collection configuration, which is what makes
``from _filing_tools import ...`` resolve in a repo-root multi-package run as
well as a ``cd escalation && pytest tests/`` one.

The filing KWARGS are deliberately NOT shared.  Each suite pins its own
``task_id`` / ``category`` / ``summary`` and those values carry local meaning —
the chokepoint suite files against an already-done task, the unpersisted-filing
suite files an ``infra_issue`` precisely because that is the one category the
stock dedupe config folds.  Sharing them would couple two suites through a
constant neither controls.  Only the tool each wrapper names is common, so only
it lives here.
"""

from __future__ import annotations

from typing import Any

from _tool_calls import call_tool
from fastmcp import FastMCP


async def call_blocker(server: FastMCP, **kwargs: Any) -> dict[str, Any]:
    """Invoke ``escalate_blocker`` on *server* and return its response dict."""
    return await call_tool(server, 'escalate_blocker', **kwargs)


async def call_info(server: FastMCP, **kwargs: Any) -> dict[str, Any]:
    """Invoke ``escalate_info`` on *server* and return its response dict."""
    return await call_tool(server, 'escalate_info', **kwargs)


async def call_promote(server: FastMCP, **kwargs: Any) -> dict[str, Any]:
    """Invoke ``promote_to_l2`` on *server* and return its response dict."""
    return await call_tool(server, 'promote_to_l2', **kwargs)
