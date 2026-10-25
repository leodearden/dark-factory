"""The escalation server tells every MCP client where task records live.

An autonomous watcher called ``mcp__escalation__get_task`` (reify #7925,
session 151229c9) — a tool this server has never had — because nothing it was
shown said that task records live on fused-memory. The server's MCP
``instructions`` are the one text every client of ``create_server`` receives,
even with its tools deferred, so they carry that fact.

Everything here is read OFF THE HANDSHAKE through ``fastmcp.Client``: the
behaviour under test is that a connecting client receives the text and that
the text is true of the tools the same client can list.

The check that each cited ``mcp__fused-memory__<tool>`` is registered on
fused-memory is in
``fused-memory/tests/test_escalation_instructions_cite_registered_tools.py``,
because fused-memory's server cannot be built from this package's test
environment.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastmcp import Client

from escalation.queue import EscalationQueue
from escalation.server import create_server

# Claude Code truncates MCP server instructions at 2048 chars (measured, task
# 4395; FUSED_MEMORY_INSTRUCTIONS arrives cut at exactly char 2048).
CLIENT_INSTRUCTIONS_CAP = 2048


async def handshake(tmp_path: Path) -> tuple[str | None, set[str]]:
    """The served instructions and registered tool names, as a client sees them."""
    server = create_server(EscalationQueue(tmp_path / 'esc'), startup_sweep=False)
    async with Client(server) as client:
        assert client.initialize_result is not None, 'the MCP handshake did not complete'
        instructions = client.initialize_result.instructions
        tool_names = {tool.name for tool in await client.list_tools()}
    return instructions, tool_names


@pytest.mark.asyncio
async def test_a_connecting_client_is_told_task_records_live_on_fused_memory(tmp_path):
    instructions, _ = await handshake(tmp_path)

    assert isinstance(instructions, str) and instructions, (
        'the escalation server sends no MCP instructions; pass '
        'instructions=ESCALATION_SERVER_INSTRUCTIONS to FastMCP in create_server'
    )
    assert 'mcp__fused-memory__get_task' in instructions, (
        'the instructions must name mcp__fused-memory__get_task, the call a '
        'client makes instead of the non-existent mcp__escalation__get_task'
    )


@pytest.mark.asyncio
async def test_every_escalation_tool_the_instructions_cite_is_registered(tmp_path):
    instructions, registered = await handshake(tmp_path)
    cited = set(re.findall(r'mcp__escalation__([a-z_]+)', instructions or ''))

    assert cited, 'the instructions cite no mcp__escalation__<tool> by name'
    assert cited <= registered, (
        f'the instructions cite unregistered tools: {sorted(cited - registered)}'
    )
    assert not {'get_task', 'get_tasks'} & registered, (
        'the instructions tell clients this server has no get_task/get_tasks; '
        'if you add such a tool here, rewrite '
        'escalation/src/escalation/server_instructions.py'
    )


@pytest.mark.asyncio
async def test_instructions_fit_the_client_delivery_cap(tmp_path):
    instructions, _ = await handshake(tmp_path)

    assert instructions is not None, 'the escalation server sends no MCP instructions'
    assert len(instructions) <= CLIENT_INSTRUCTIONS_CAP, (
        f'the instructions are {len(instructions)} chars; text past '
        f'{CLIENT_INSTRUCTIONS_CAP} never reaches the agent — shorten them'
    )
