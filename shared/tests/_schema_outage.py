"""A toy server whose tool catalogue middleware cannot read: every ``get_tool`` raises.

The tool bodies still run, so a guard's fail-safe answer to an unreadable schema is
observable through an ordinary ``fastmcp.Client`` call.
"""

from __future__ import annotations

import types

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
