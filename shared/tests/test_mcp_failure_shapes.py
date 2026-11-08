"""B9: no failure answer from an MCP server ever reads as a successful write.

Each scripted failure answer of :class:`shared.testing_mcp_stub.StubMcpServer`,
against both a stateless and a stateful server, must surface as a typed failure:
an :class:`McpCallError` of a known kind, or an :class:`EnvelopeParseError` of a
known shape once the reply is parsed. Composed end to end, ``call_tool`` →
``parse_tool_result`` → ``write_failure_reason`` never yields a success verdict.
``SyncMcpSession`` folds the parse in, so it raises for every one of them.
"""
from __future__ import annotations

import pytest

from shared.mcp_envelope import EnvelopeShape, parse_tool_result, write_failure_reason
from shared.mcp_session import McpCallError, McpFailureKind, McpSession, SyncMcpSession
from shared.testing_mcp_stub import StubAnswer, StubMcpServer

UPDATE = ('update_task', {'id': '7', 'status': 'done'})

ASYNC_EXPECTED: dict[StubAnswer, McpFailureKind | EnvelopeShape] = {
    StubAnswer.IS_ERROR: EnvelopeShape.IS_ERROR,
    StubAnswer.NON_JSON_BODY: McpFailureKind.UNDECODABLE,
    StubAnswer.NON_JSON_TEXT: EnvelopeShape.JSON_DECODE_ERROR,
    StubAnswer.EMPTY_202: McpFailureKind.UNDECODABLE,
    StubAnswer.REDIRECT_307: McpFailureKind.TRANSPORT,
    StubAnswer.JSON_RPC_ERROR: EnvelopeShape.JSON_RPC_ERROR,
}

SYNC_EXPECTED: dict[StubAnswer, McpFailureKind] = {
    StubAnswer.IS_ERROR: McpFailureKind.TOOL_ERROR,
    StubAnswer.NON_JSON_BODY: McpFailureKind.UNDECODABLE,
    StubAnswer.NON_JSON_TEXT: McpFailureKind.REPLY_SHAPE,
    StubAnswer.EMPTY_202: McpFailureKind.UNDECODABLE,
    StubAnswer.REDIRECT_307: McpFailureKind.TRANSPORT,
    StubAnswer.JSON_RPC_ERROR: McpFailureKind.JSON_RPC_ERROR,
}

STATEFULNESS = pytest.mark.parametrize('stateful', [False, True], ids=['stateless', 'stateful'])


async def _async_failure(server: StubMcpServer) -> McpFailureKind | EnvelopeShape | None:
    try:
        body = await McpSession(server.url, client_name='b9').call_tool(*UPDATE)
    except McpCallError as exc:
        return exc.kind
    _reply, err = parse_tool_result(body, None, dict)
    return None if err is None else err.shape


@STATEFULNESS
@pytest.mark.parametrize('answer', list(ASYNC_EXPECTED), ids=str)
async def test_async_session_surfaces_each_failure_typed(stateful, answer):
    with StubMcpServer(stateful=stateful, answer=answer) as server:
        assert await _async_failure(server) == ASYNC_EXPECTED[answer]


@STATEFULNESS
@pytest.mark.parametrize('answer', list(ASYNC_EXPECTED), ids=str)
async def test_no_failure_shape_reaches_write_failure_reason_as_success(stateful, answer):
    with StubMcpServer(stateful=stateful, answer=answer) as server:
        try:
            body = await McpSession(server.url, client_name='b9').call_tool(*UPDATE)
        except McpCallError:
            return
    reply, _err = parse_tool_result(body, None, dict)
    assert write_failure_reason(reply) is not None


@STATEFULNESS
async def test_an_ok_answer_is_the_one_success(stateful):
    """The control row: the composed path does report a real success."""
    with StubMcpServer(stateful=stateful, payload={'success': True}) as server:
        body = await McpSession(server.url, client_name='b9').call_tool(*UPDATE)
    reply, err = parse_tool_result(body, None, dict)
    assert err is None
    assert write_failure_reason(reply) is None


@STATEFULNESS
@pytest.mark.parametrize('answer', list(SYNC_EXPECTED), ids=str)
def test_sync_session_raises_each_failure_typed(stateful, answer):
    with (
        StubMcpServer(stateful=stateful, answer=answer) as server,
        SyncMcpSession(server.url, client_name='b9') as session,
        pytest.raises(McpCallError) as excinfo,
    ):
        session.call_tool(*UPDATE)

    assert excinfo.value.kind is SYNC_EXPECTED[answer]
