"""The middleware chain at the bundled FastMCP's ``ToolManager.call_tool`` (task 4643, D5).

Every test drives a TOY bundled server with three tools, one per raw result
shape the bundled dispatcher produces:

* ``any_values`` (``-> dict[str, Any]``): a ``(content, structured)`` tuple,
  output schema ``{type: object, additionalProperties: true}``.
* ``str_values`` (``-> dict[str, str]``): the same tuple, but a schema that
  refuses a dict-valued key, like fused-memory's ``get_external_statuses``.
* ``bare`` (``-> dict``): a bare content list, no output schema.

Wire assertions go through a real in-memory ``ClientSession``, which also runs
the client's own ``outputSchema`` validation. Members use public fastmcp names
only.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware
from fastmcp.tools.base import ToolResult
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent

from fused_memory.server.tool_middleware_bridge import install_tool_middleware

_SENTINEL = '_toy_chain_installed'
_TOOLS = ('any_values', 'str_values', 'bare')
_PROBE = {'k': 1}


class _ToyTools:
    """Records each tool run, and hands back dicts the test can inspect afterwards."""

    def __init__(self) -> None:
        self.runs: list[tuple[str, str]] = []
        self.returned: list[dict[str, Any]] = []

    def record(self, tool: str, note: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.runs.append((tool, note))
        self.returned.append(payload)
        return payload


def _toy_server(tools: _ToyTools) -> FastMCP:
    server = FastMCP('bridge-toy')

    @server.tool()
    def any_values(note: str = '', own_probe: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {'note': note, 'nested': {'n': 1}}
        if own_probe:
            payload['probe'] = 'the tool said so'
        return tools.record('any_values', note, payload)

    @server.tool()
    def str_values(note: str = '') -> dict[str, str]:
        return tools.record('str_values', note, {'note': note})

    @server.tool()
    def bare(note: str = '') -> dict:
        return tools.record('bare', note, {'note': note})

    return server


class _PassThrough(Middleware):
    async def on_call_tool(self, context, call_next):
        return await call_next(context)


class _MetaAdder(Middleware):
    """Forwards, then authors a public ``ToolResult`` with extra meta keys."""

    def __init__(self, **extra: Any) -> None:
        self.extra = extra

    async def on_call_tool(self, context, call_next):
        result = await call_next(context)
        return ToolResult(
            content=result.content,
            structured_content=result.structured_content,
            meta={**(result.meta or {}), **self.extra},
        )


class _Recorder(Middleware):
    def __init__(self, label: str, log: list[tuple[str, dict[str, Any]]]) -> None:
        self.label = label
        self.log = log

    async def on_call_tool(self, context, call_next):
        self.log.append((self.label, dict(context.message.arguments)))
        return await call_next(context)


class _RaisesBeforeForwarding(Middleware):
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    async def on_call_tool(self, context, call_next):
        context.message.arguments['note'] = 'edited by a member that then crashed'
        context.message.arguments['smuggled'] = True
        raise self.exc


class _RaisesAfterForwarding(Middleware):
    async def on_call_tool(self, context, call_next):
        await call_next(context)
        raise RuntimeError('defect after the call was forwarded')


def _install(server: FastMCP, members: tuple[Middleware, ...], **kwargs: Any) -> None:
    install_tool_middleware(server, members, sentinel=_SENTINEL, label='toy chain', **kwargs)


async def _wire(server: FastMCP, tool: str, arguments: dict[str, Any]) -> CallToolResult:
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        return await session.call_tool(tool, arguments)


def _single_text_json(result: CallToolResult) -> dict[str, Any]:
    [block] = result.content
    assert isinstance(block, TextContent)
    return json.loads(block.text)


class TestPassThroughIsByteIdentical:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('tool', _TOOLS)
    async def test_the_in_process_dict_equals_the_unguarded_twin(self, tool):
        guarded, twin = _toy_server(_ToyTools()), _toy_server(_ToyTools())
        _install(guarded, (_PassThrough(), _PassThrough()))

        got = await guarded._tool_manager.call_tool(tool, {'note': 'x'})
        want = await twin._tool_manager.call_tool(tool, {'note': 'x'})

        assert got == want

    @pytest.mark.asyncio
    @pytest.mark.parametrize('tool', _TOOLS)
    async def test_the_wire_result_equals_the_unguarded_twin(self, tool):
        guarded, twin = _toy_server(_ToyTools()), _toy_server(_ToyTools())
        _install(guarded, (_PassThrough(), _PassThrough()))

        got = await _wire(guarded, tool, {'note': 'x'})
        want = await _wire(twin, tool, {'note': 'x'})

        assert got.content == want.content
        assert got.structuredContent == want.structuredContent
        assert got.meta is None and want.meta is None
        assert got.isError is want.isError is False


class TestAMemberAuthoredToolResultReachesTheClient:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('tool', _TOOLS)
    async def test_meta_arrives_and_the_payload_is_unchanged(self, tool):
        guarded, twin = _toy_server(_ToyTools()), _toy_server(_ToyTools())
        _install(guarded, (_MetaAdder(probe=_PROBE),))

        got = await _wire(guarded, tool, {'note': 'x'})
        want = await _wire(twin, tool, {'note': 'x'})

        assert got.meta is not None and got.meta['probe'] == _PROBE
        assert got.isError is False
        assert got.content == want.content
        assert got.structuredContent == want.structuredContent

    @pytest.mark.asyncio
    async def test_the_dispatcher_hands_back_a_call_tool_result(self):
        server = _toy_server(_ToyTools())
        _install(server, (_MetaAdder(probe=_PROBE),))

        result = await server._tool_manager.call_tool(
            'any_values', {'note': 'x'}, convert_result=True
        )

        assert isinstance(result, CallToolResult)
        assert result.meta == {'probe': _PROBE}


class TestOrderAndOneWrapping:
    @pytest.mark.asyncio
    async def test_the_outer_member_runs_first(self):
        log: list[tuple[str, dict[str, Any]]] = []
        server = _toy_server(_ToyTools())
        _install(server, (_Recorder('outer', log), _Recorder('inner', log)))

        await server._tool_manager.call_tool('bare', {'note': 'x'})

        assert [label for label, _ in log] == ['outer', 'inner']

    def test_a_second_install_with_the_same_sentinel_is_a_no_op(self):
        server = _toy_server(_ToyTools())
        _install(server, (_PassThrough(),))
        wrapped = server._tool_manager.call_tool

        _install(server, (_PassThrough(), _PassThrough()))

        assert server._tool_manager.call_tool is wrapped


class TestPerMemberContainment:
    @pytest.mark.asyncio
    async def test_a_member_failing_before_forwarding_is_skipped_not_the_chain(self, caplog):
        log: list[tuple[str, dict[str, Any]]] = []
        tools = _ToyTools()
        server = _toy_server(tools)
        _install(server, (_RaisesBeforeForwarding(RuntimeError('defect')), _Recorder('next', log)))

        with caplog.at_level('ERROR', logger='fused_memory.server.tool_middleware_bridge'):
            result = await server._tool_manager.call_tool('bare', {'note': 'x'})

        assert log == [('next', {'note': 'x'})]
        assert tools.runs == [('bare', 'x')]
        assert result == {'note': 'x'}
        text = caplog.text
        assert 'NOT scanned' in text and 'bare' in text

    @pytest.mark.asyncio
    async def test_a_member_failing_after_forwarding_re_raises_and_the_tool_ran_once(self):
        tools = _ToyTools()
        server = _toy_server(tools)
        _install(server, (_RaisesAfterForwarding(),))

        with pytest.raises(RuntimeError, match='after the call was forwarded'):
            await server._tool_manager.call_tool('bare', {'note': 'x'})

        assert tools.runs == [('bare', 'x')]

    @pytest.mark.asyncio
    async def test_cancellation_propagates(self):
        tools = _ToyTools()
        server = _toy_server(tools)
        _install(server, (_RaisesBeforeForwarding(asyncio.CancelledError()),))

        with pytest.raises(asyncio.CancelledError):
            await server._tool_manager.call_tool('bare', {'note': 'x'})

        assert tools.runs == []

    @pytest.mark.asyncio
    async def test_a_tool_error_propagates_unchanged(self):
        refusal = ToolError('{"refused": true}')
        tools = _ToyTools()
        server = _toy_server(tools)
        _install(server, (_PassThrough(), _RaisesBeforeForwarding(refusal)))

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool('bare', {'note': 'x'})

        assert exc_info.value is refusal
        assert tools.runs == []


class TestTheEchoFold:
    @pytest.mark.asyncio
    async def test_an_echoed_key_rides_in_structured_content_and_the_text_block(self):
        server = _toy_server(_ToyTools())
        _install(server, (_MetaAdder(probe=_PROBE),), echoed_meta_keys=frozenset({'probe'}))

        result = await _wire(server, 'any_values', {'note': 'x'})

        assert result.isError is False
        assert result.meta == {'probe': _PROBE}
        assert result.structuredContent == {'note': 'x', 'nested': {'n': 1}, 'probe': _PROBE}
        assert _single_text_json(result) == {'note': 'x', 'nested': {'n': 1}, 'probe': _PROBE}

    @pytest.mark.asyncio
    async def test_a_schema_that_refuses_the_key_keeps_structured_content_unfolded(self):
        server = _toy_server(_ToyTools())
        _install(server, (_MetaAdder(probe=_PROBE),), echoed_meta_keys=frozenset({'probe'}))

        result = await _wire(server, 'str_values', {'note': 'x'})

        assert result.isError is False
        assert result.structuredContent == {'note': 'x'}
        assert result.meta == {'probe': _PROBE}

    @pytest.mark.asyncio
    async def test_a_key_the_tool_already_returned_is_not_overwritten(self):
        server = _toy_server(_ToyTools())
        _install(server, (_MetaAdder(probe=_PROBE),), echoed_meta_keys=frozenset({'probe'}))

        result = await _wire(server, 'any_values', {'note': 'x', 'own_probe': True})

        assert result.structuredContent is not None
        assert result.structuredContent['probe'] == 'the tool said so'
        assert _single_text_json(result)['probe'] == 'the tool said so'
        assert result.meta == {'probe': _PROBE}

    @pytest.mark.asyncio
    async def test_the_in_process_dict_gains_the_key_without_mutating_the_tools_own(self):
        tools = _ToyTools()
        server = _toy_server(tools)
        _install(server, (_MetaAdder(probe=_PROBE),), echoed_meta_keys=frozenset({'probe'}))

        result = await server._tool_manager.call_tool('any_values', {'note': 'x'})

        assert result == {'note': 'x', 'nested': {'n': 1}, 'probe': _PROBE}
        [returned] = tools.returned
        assert 'probe' not in returned

    @pytest.mark.asyncio
    async def test_a_key_not_declared_echoed_stays_on_meta_only(self):
        tools = _ToyTools()
        server = _toy_server(tools)
        _install(
            server,
            (_MetaAdder(probe=_PROBE, private={'p': 2}),),
            echoed_meta_keys=frozenset({'probe'}),
        )

        wire = await _wire(server, 'any_values', {'note': 'x'})
        in_process = await server._tool_manager.call_tool('any_values', {'note': 'x'})

        assert wire.meta == {'probe': _PROBE, 'private': {'p': 2}}
        assert wire.structuredContent is not None
        assert 'private' not in wire.structuredContent
        assert 'private' not in _single_text_json(wire)
        assert 'private' not in in_process
