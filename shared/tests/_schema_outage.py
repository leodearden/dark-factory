"""A toy server whose tool catalogue middleware cannot read, and what that outage must log.

Every ``get_tool`` raises, but the tool bodies still run, so a guard's fail-safe answer
to an unreadable schema is observable through an ordinary ``fastmcp.Client`` call.
"""

from __future__ import annotations

import logging
import types
from collections.abc import Iterable

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware


class _UnreachableToolCatalog:
    async def get_tool(self, name: str):
        raise RuntimeError(f'tool catalogue unreachable for {name!r}')


class _BlindingMiddleware(Middleware):
    async def on_call_tool(self, context, call_next):
        blinded = types.SimpleNamespace(fastmcp=_UnreachableToolCatalog())
        return await call_next(context.copy(fastmcp_context=blinded))


def install_schema_outage(mcp: FastMCP) -> None:
    """Blind every middleware already on *mcp*: fastmcp 3.2.2 runs ``middleware[0]`` outermost."""
    mcp.middleware.insert(0, _BlindingMiddleware())


def assert_reported_once_by_shared_reader(records: Iterable[logging.LogRecord]) -> None:
    """One traceback, at WARNING from ``shared.mcp_tool_schema``, and no ``shared.*`` ERROR.

    The messages carry the offending records because pytest does not rewrite asserts here.
    """
    shared_records = [r for r in records if r.name.startswith('shared.')]
    tracebacks = [(r.name, r.levelno) for r in shared_records if r.exc_info]
    assert tracebacks == [('shared.mcp_tool_schema', logging.WARNING)], tracebacks
    errors = [(r.name, r.getMessage()) for r in shared_records if r.levelno >= logging.ERROR]
    assert errors == [], errors
