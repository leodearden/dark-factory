"""Tests for ``shared.mcp_tool_schema`` — the live-schema read middleware shares.

Its only input is a live ``MiddlewareContext``, so every read happens inside a
real middleware on a toy server driven through ``fastmcp.Client``.
"""
from __future__ import annotations

import logging
from typing import Any

from fastmcp import Client, FastMCP
from fastmcp.server.middleware import Middleware

from shared.mcp_tool_schema import live_tool_parameters


class _Reader(Middleware):
    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self.seen: list[Any] = []

    async def on_call_tool(self, context, call_next):
        self.seen.append(await live_tool_parameters(context, self.name or context.message.name))
        return await call_next(context)


async def _read_during_a_call(reader: _Reader) -> Any:
    mcp = FastMCP('tool-schema-harness')
    mcp.add_middleware(reader)

    @mcp.tool
    def pair(alpha: str, beta: int = 0) -> dict:
        return {'ok': True}

    async with Client(mcp) as client:
        await client.call_tool('pair', {'alpha': 'a'})
    [parameters] = reader.seen
    return parameters


async def test_the_invoked_tools_full_json_schema_is_returned():
    parameters = await _read_during_a_call(_Reader())

    assert list(parameters['properties']) == ['alpha', 'beta']
    assert parameters['properties']['beta']['type'] == 'integer'
    assert parameters['required'] == ['alpha']


async def test_an_unresolvable_tool_yields_none_and_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger='shared.mcp_tool_schema'):
        parameters = await _read_during_a_call(_Reader('no_such_tool'))

    assert parameters is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any('no_such_tool' in message for message in warnings)
