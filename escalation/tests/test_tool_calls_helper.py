"""Contract tests for ``_tool_calls.call_tool``."""

from __future__ import annotations

from pathlib import Path

import pytest
from _tool_calls import call_tool
from fastmcp import FastMCP

from escalation.queue import EscalationQueue
from escalation.server import create_server

pytestmark = pytest.mark.asyncio


def toy_server() -> FastMCP:
    server = FastMCP('toy')

    @server.tool()
    def sync_echo(x: int) -> dict:
        return {'x': x, 'kind': 'sync'}

    @server.tool()
    async def async_echo(x: int) -> dict:
        return {'x': x, 'kind': 'async'}

    return server


async def test_an_async_tool_body_is_awaited() -> None:
    assert await call_tool(toy_server(), 'async_echo', x=3) == {'x': 3, 'kind': 'async'}


async def test_a_sync_tool_body_is_returned_as_is() -> None:
    assert await call_tool(toy_server(), 'sync_echo', x=4) == {'x': 4, 'kind': 'sync'}


async def test_an_unregistered_name_fails_naming_the_tool() -> None:
    with pytest.raises(AssertionError, match='no_such_tool'):
        await call_tool(toy_server(), 'no_such_tool')


async def test_both_shapes_of_real_escalation_tool_are_served(tmp_path: Path) -> None:
    server = create_server(EscalationQueue(tmp_path / 'esc'), startup_sweep=False)

    assert await call_tool(server, 'get_task_escalations', task_id='t-none') == []
    assert await call_tool(server, 'get_pending_escalations') == []
