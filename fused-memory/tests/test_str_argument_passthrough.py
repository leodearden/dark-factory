"""Strings sent to a ``str | None`` tool parameter arrive exactly as sent (task 7472).

The bundled FastMCP json.loads every string argument whose parameter is not
annotated exactly ``str``, so before this fix a ``str | None`` parameter
refused a JSON-object/array string with a validation error and turned
``'null'`` into None. Parameters typed list, dict, Any or a mixed union keep
that coercion, which clients rely on.
"""

from __future__ import annotations

from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from fused_memory.server.str_argument_passthrough import install_str_argument_passthrough


def _recording_server(received: dict[str, Any]) -> FastMCP:
    server = FastMCP('t')

    @server.tool()
    def echo(
        a: str | None = None,
        b: list[str] | None = None,
        c: dict | None = None,
        d: str | dict | None = None,
        e: Any = None,
    ) -> str:
        received.update(a=a, b=b, c=c, d=d, e=e)
        return 'ok'

    return server


@pytest.mark.asyncio
@pytest.mark.parametrize('sent', ['{"a": 1}', '[1, 2]', 'null'])
async def test_a_str_parameter_receives_a_json_looking_string_verbatim(sent):
    received: dict[str, Any] = {}
    server = _recording_server(received)
    install_str_argument_passthrough(server)

    await server._tool_manager.call_tool('echo', {'a': sent})

    assert received['a'] == sent


@pytest.mark.asyncio
async def test_list_dict_mixed_union_and_any_parameters_keep_the_coercion():
    received: dict[str, Any] = {}
    server = _recording_server(received)
    install_str_argument_passthrough(server)

    await server._tool_manager.call_tool(
        'echo', {'b': '["x"]', 'c': '{"k": 1}', 'd': '{"k": 1}', 'e': '[1]'},
    )

    assert received['b'] == ['x']
    assert received['c'] == {'k': 1}
    assert received['d'] == {'k': 1}
    assert received['e'] == [1]


@pytest.mark.asyncio
async def test_a_tool_registered_after_install_is_covered_too():
    server = FastMCP('t')
    install_str_argument_passthrough(server)
    received: list[str | None] = []

    @server.tool()
    def late(a: str | None = None) -> str:
        received.append(a)
        return 'ok'

    await server._tool_manager.call_tool('late', {'a': '{"a": 1}'})
    await server._tool_manager.call_tool('late', {'a': 'null'})

    assert received == ['{"a": 1}', 'null']


def test_installing_twice_does_not_wrap_twice():
    server = FastMCP('t')
    install_str_argument_passthrough(server)
    first_wrapper = server._tool_manager.call_tool

    install_str_argument_passthrough(server)

    assert server._tool_manager.call_tool is first_wrapper


@pytest.mark.asyncio
async def test_without_the_passthrough_fastmcp_still_coerces_a_str_parameter():
    received: dict[str, Any] = {}
    server = _recording_server(received)

    await server._tool_manager.call_tool('echo', {'a': 'null'})
    assert received['a'] is None

    with pytest.raises(ToolError, match='Input should be a valid string'):
        await server._tool_manager.call_tool('echo', {'a': '{"a": 1}'})
