"""The two-line dance every test that files an escalation has to repeat.

``escalate_blocker`` and ``escalate_info`` are async FastMCP tools, so calling
one means ``await server.get_tool(name)`` and then ``await tool.fn(**kwargs)``.
That pair was copied verbatim into a THIRD test module before this one existed
(``test_server_chokepoint.py`` and ``test_observed_submit_response.py`` each
carried its own identical copy) — the point at which a copy stops being cheaper
than a name.

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
constant neither controls.  Only the call SHAPE is common, so only it lives
here.
"""

from __future__ import annotations

from typing import Any


async def call_blocker(server, **kwargs: Any) -> dict[str, Any]:
    """Invoke ``escalate_blocker`` on *server* and return its response dict."""
    tool = await server.get_tool('escalate_blocker')
    return await tool.fn(**kwargs)


async def call_info(server, **kwargs: Any) -> dict[str, Any]:
    """Invoke ``escalate_info`` on *server* and return its response dict."""
    tool = await server.get_tool('escalate_info')
    return await tool.fn(**kwargs)
