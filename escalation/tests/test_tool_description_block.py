"""``AppendToEveryToolDescription`` appends one block to every served tool description.

Driven against small toy servers and read through ``fastmcp.Client``, which is
what a connecting MCP caller sees.
"""

from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP

from escalation.tool_description_block import AppendToEveryToolDescription

BLOCK = 'TRAILING BLOCK'


def toy_server(*, with_transform: bool = True) -> FastMCP:
    server = FastMCP('toy')

    @server.tool()
    def documented(x: int) -> dict:
        """Documented tool."""
        return {'x': x}

    if with_transform:
        server.add_transform(AppendToEveryToolDescription(BLOCK))

    @server.tool()
    async def undocumented() -> str:
        return 'ok'

    return server


async def listed_descriptions(server: FastMCP) -> dict[str, str | None]:
    async with Client(server) as client:
        return {tool.name: tool.description for tool in await client.list_tools()}


@pytest.mark.asyncio
async def test_a_documented_tool_lists_its_docstring_then_the_block() -> None:
    descriptions = await listed_descriptions(toy_server())

    assert descriptions['documented'] == 'Documented tool.' + '\n\n' + BLOCK


@pytest.mark.asyncio
async def test_an_undocumented_tool_lists_the_block_alone() -> None:
    descriptions = await listed_descriptions(toy_server())

    assert descriptions['undocumented'] == BLOCK


@pytest.mark.asyncio
async def test_tools_registered_before_and_after_the_transform_all_carry_the_block() -> None:
    descriptions = await listed_descriptions(toy_server())

    assert set(descriptions) == {'documented', 'undocumented'}
    for name, description in descriptions.items():
        assert description is not None and description.endswith(BLOCK), name


@pytest.mark.asyncio
async def test_get_tool_agrees_with_the_client_listing() -> None:
    server = toy_server()
    descriptions = await listed_descriptions(server)

    tool = await server.get_tool('documented')

    assert tool is not None
    assert tool.description == descriptions['documented']


@pytest.mark.asyncio
async def test_calls_still_dispatch_to_the_original_function() -> None:
    async with Client(toy_server()) as client:
        result = await client.call_tool('documented', {'x': 3})

    assert result.data == {'x': 3}


@pytest.mark.asyncio
async def test_without_the_transform_the_docstring_is_listed_unchanged() -> None:
    descriptions = await listed_descriptions(toy_server(with_transform=False))

    assert descriptions['documented'] == 'Documented tool.'
