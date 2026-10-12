"""Tests for shared.mcp_session.McpSession, the async MCP client every package uses.

Transport retry behaviour is driven through ``patch('httpx.AsyncClient')`` with
scripted answers: ConnectError/ReadTimeout/OSError sequences cannot be produced
cheaply by a socket. Wire facts (path, Accept, session id, clientInfo) are
observed by a real-socket :class:`shared.testing_mcp_stub.StubMcpServer`.
Everything is driven through ``initialize``/``call``/``call_tool``.
"""
from __future__ import annotations

import copy
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from shared.mcp_post import MCP_POST_HEADERS
from shared.mcp_session import (
    MCP_PROTOCOL_VERSION,
    McpCallError,
    McpFailureKind,
    McpSession,
    is_timeout_failure,
)
from shared.testing_mcp_stub import StubAnswer, StubMcpServer

BASE_URL = 'http://localhost:8002'
OK_BODY = {'jsonrpc': '2.0', 'id': 1, 'result': {'ok': True}}


def _session(retry_backoffs: Sequence[float] = ()) -> McpSession:
    return McpSession(BASE_URL, client_name='test-client', retry_backoffs=retry_backoffs)


def _response(
    status_code: int = 200,
    json_body: dict | None = None,
    session_id: str | None = None,
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=json_body or OK_BODY,
        headers={'content-type': 'application/json', **(
            {'mcp-session-id': session_id} if session_id else {}
        )},
        request=httpx.Request('POST', f'{BASE_URL}/mcp'),
    )


def _handshake(session_id: str | None = None) -> list[Any]:
    """The two answers an initialize handshake consumes."""
    return [_response(session_id=session_id), _response(status_code=202)]


@dataclass
class _Transport:
    client: AsyncMock
    sleep: AsyncMock

    def posted(self) -> list[dict]:
        return [c.kwargs['json'] for c in self.client.post.call_args_list]

    def posted_methods(self) -> list[str]:
        return [body['method'] for body in self.posted()]

    def session_header(self, index: int) -> str | None:
        return self.client.post.call_args_list[index].kwargs['headers'].get('Mcp-Session-Id')


@contextmanager
def _scripted(answers: Any) -> Iterator[_Transport]:
    """Patch the httpx client seam so each post() takes the next scripted answer."""
    client = AsyncMock()
    client.post.side_effect = answers
    with (
        patch('httpx.AsyncClient') as client_cls,
        patch('asyncio.sleep', new_callable=AsyncMock) as sleep,
    ):
        client_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        yield _Transport(client, sleep)


# ---------------------------------------------------------------------------
# Transport retry (moved from orchestrator/tests/test_mcp_retry.py)
# ---------------------------------------------------------------------------


class TestRequestRetry:
    async def test_succeeds_first_attempt(self):
        with _scripted([*_handshake('s1'), _response()]) as transport:
            result = await _session().call_tool('test', {})

        assert result == OK_BODY
        assert transport.posted_methods() == ['initialize', 'notifications/initialized', 'tools/call']
        assert transport.session_header(2) == 's1'

    async def test_retries_on_connect_error(self):
        fixed = (1.0, 2.0)
        answers = [
            httpx.ConnectError('Connection refused'),
            httpx.ConnectError('Connection refused'),
            *_handshake('new-session'),
        ]
        with _scripted(answers) as transport:
            await _session(fixed).initialize()

        assert transport.client.post.call_count == 4
        assert [c.args[0] for c in transport.sleep.call_args_list] == pytest.approx(list(fixed))
        assert transport.session_header(3) == 'new-session'

    async def test_retries_on_503(self):
        with _scripted([*_handshake(), _response(status_code=503), _response()]) as transport:
            result = await _session((1.0,)).call_tool('t', {})

        assert result['result'] == {'ok': True}
        assert transport.posted_methods()[2:] == ['tools/call', 'tools/call']

    async def test_no_retry_on_400(self):
        with (
            _scripted([*_handshake(), _response(status_code=400)]) as transport,
            pytest.raises(McpCallError) as excinfo,
        ):
            await _session((1.0, 2.0)).call_tool('t', {})

        assert excinfo.value.kind is McpFailureKind.HTTP_STATUS
        assert excinfo.value.status == 400
        assert isinstance(excinfo.value.__cause__, httpx.HTTPStatusError)
        assert transport.posted_methods().count('tools/call') == 1

    async def test_exhausts_retries(self):
        fixed = (0.01, 0.01)
        with (
            _scripted(httpx.ConnectError('Connection refused')) as transport,
            pytest.raises(McpCallError) as excinfo,
        ):
            await _session(fixed).call_tool('t', {})

        err = excinfo.value
        assert transport.client.post.call_count == len(fixed) + 1
        assert err.kind is McpFailureKind.TRANSPORT
        assert isinstance(err, RuntimeError)
        assert f'{len(fixed) + 1} attempts' in str(err)
        assert 'ConnectError' in str(err)
        assert isinstance(err.__cause__, httpx.ConnectError)

    async def test_transient_failure_drops_the_session_and_reinitializes(self):
        answers = [
            *_handshake('old'),
            httpx.ConnectError('refused'),
            _response(),
            *_handshake('fresh'),
            _response(),
        ]
        with _scripted(answers) as transport:
            session = _session((0.0,))
            await session.call_tool('t', {})
            await session.call_tool('t', {})

        assert transport.posted_methods() == [
            'initialize', 'notifications/initialized', 'tools/call', 'tools/call',
            'initialize', 'notifications/initialized', 'tools/call',
        ]
        assert transport.session_header(2) == 'old'
        assert transport.session_header(3) is None
        assert transport.session_header(6) == 'fresh'

    @pytest.mark.parametrize(
        'transient',
        [httpx.ReadTimeout(''), FileNotFoundError(2, 'No such file or directory')],
        ids=['read-timeout', 'oserror'],
    )
    async def test_retries_on_timeout_and_raw_oserror(self, transient):
        """ReadTimeout (empty str) and a raw OSError that httpx did not re-wrap
        both retry instead of escaping the loop (2026-04-20 incidents)."""
        answers = [*_handshake(), transient, transient, _response()]
        with _scripted(answers) as transport:
            result = await _session((0.0, 0.0)).call_tool('t', {})

        assert transport.posted_methods().count('tools/call') == 3
        assert result['result']['ok'] is True

    @pytest.mark.parametrize(
        'cause, type_name',
        [
            (FileNotFoundError(2, 'No such file or directory'), 'FileNotFoundError'),
            (httpx.ReadTimeout(''), 'ReadTimeout'),
        ],
        ids=['oserror', 'read-timeout'],
    )
    async def test_exhausted_failure_names_the_cause_type(self, cause, type_name):
        """str(cause) is empty or misleading, so the wrap must carry the type name."""
        fixed = (0.01, 0.01)
        with _scripted(cause) as transport, pytest.raises(McpCallError) as excinfo:
            await _session(fixed).call_tool('t', {})

        assert transport.client.post.call_count == len(fixed) + 1
        assert type_name in str(excinfo.value)
        assert excinfo.value.__cause__ is cause


class TestIsTimeoutFailure:
    """``is_timeout_failure`` recovers the timeout fact the exhausted-retry wrap erases."""

    async def _drive(self, cause: Exception) -> McpCallError:
        """The error a session really raises for *cause*, not an imitation of it."""
        with _scripted(cause), pytest.raises(McpCallError) as excinfo:
            await _session((0.01, 0.01)).call_tool('t', {})
        return excinfo.value

    async def test_a_real_exhausted_read_timeout_is_a_timeout(self):
        err = await self._drive(httpx.ReadTimeout(''))

        assert not isinstance(err, TimeoutError), (
            'the wrap is why a type check cannot answer this question'
        )
        assert is_timeout_failure(err)

    async def test_a_real_exhausted_connect_error_is_not_a_timeout(self):
        err = await self._drive(httpx.ConnectError('refused'))

        assert not is_timeout_failure(err)

    def test_bare_timeout_exceptions_are_timeouts(self):
        assert is_timeout_failure(httpx.TimeoutException('slow'))
        assert is_timeout_failure(httpx.ReadTimeout(''))
        assert is_timeout_failure(TimeoutError())

    def test_non_timeouts_are_not_timeouts(self):
        assert not is_timeout_failure(httpx.ConnectError('refused'))
        assert not is_timeout_failure(RuntimeError('no cause at all'))
        assert not is_timeout_failure(ValueError('unrelated'))

    def test_a_timeout_deep_in_a_none_terminated_chain_is_found(self):
        inner = RuntimeError('MCP tools/call failed after 3 attempts: ReadTimeout: ')
        inner.__cause__ = httpx.ReadTimeout('')
        outer = RuntimeError('search failed')
        outer.__cause__ = inner

        assert is_timeout_failure(outer)

    def test_a_cyclic_cause_chain_terminates(self):
        first = RuntimeError('first')
        second = RuntimeError('second')
        first.__cause__ = second
        second.__cause__ = first

        assert not is_timeout_failure(first)

    def test_a_cycle_containing_a_timeout_still_reports_it(self):
        slow = httpx.ReadTimeout('')
        wrapper = RuntimeError('wrapper')
        wrapper.__cause__ = slow
        slow.__cause__ = wrapper

        assert is_timeout_failure(wrapper)


class TestClientOpIdInjection:
    """A mutating task tool gets one client_op_id per logical call (task 2712)."""

    async def test_injects_client_op_id_for_mutating_tool(self):
        with _scripted([*_handshake(), _response()]) as transport:
            await _session().call_tool('update_task', {'id': '1'})

        op_id = transport.posted()[-1]['params']['arguments'].get('client_op_id')
        assert isinstance(op_id, str) and op_id

    async def test_same_client_op_id_across_transport_retries(self):
        """Bodies are snapshotted per post(): the payload is re-serialized each attempt."""
        bodies: list[dict] = []
        answers = iter([*_handshake(), httpx.ConnectError('x'), _response()])

        async def capture_post(*args, **kwargs):
            bodies.append(copy.deepcopy(kwargs['json']))
            answer = next(answers)
            if isinstance(answer, Exception):
                raise answer
            return answer

        with _scripted(capture_post):
            await _session((0.0,)).call_tool('update_task', {'id': '1'})

        op_ids = [b['params']['arguments']['client_op_id'] for b in bodies if b['method'] == 'tools/call']
        assert len(op_ids) == 2
        assert op_ids[0] and op_ids[0] == op_ids[1]

    async def test_no_injection_for_read_tool(self):
        with _scripted([*_handshake(), _response()]) as transport:
            await _session().call_tool('get_tasks', {})

        assert 'client_op_id' not in transport.posted()[-1]['params']['arguments']

    async def test_preserves_caller_supplied_client_op_id(self):
        with _scripted([*_handshake(), _response()]) as transport:
            await _session().call_tool('update_task', {'id': '1', 'client_op_id': 'caller-set'})

        assert transport.posted()[-1]['params']['arguments']['client_op_id'] == 'caller-set'

    async def test_the_callers_arguments_are_not_written_to(self):
        arguments = {'id': '1'}
        params = {'name': 'update_task', 'arguments': {'id': '2'}}
        with _scripted([*_handshake(), _response(), _response()]) as transport:
            session = _session()
            await session.call_tool('update_task', arguments)
            await session.call('tools/call', params)

        assert arguments == {'id': '1'}
        assert params == {'name': 'update_task', 'arguments': {'id': '2'}}
        assert all('client_op_id' in body['params']['arguments'] for body in transport.posted()[2:])


class TestNotifyRetry:
    async def test_notification_is_the_second_post_and_carries_no_id(self):
        with _scripted(_handshake()) as transport:
            await _session().initialize()

        notification = transport.posted()[1]
        assert notification['method'] == 'notifications/initialized'
        assert 'id' not in notification

    async def test_retries_on_connect_error(self):
        answers = [_response(), httpx.ConnectError('refused'), _response(status_code=200)]
        with _scripted(answers) as transport:
            await _session((0.0,)).initialize()

        assert transport.posted_methods() == [
            'initialize', 'notifications/initialized', 'notifications/initialized',
        ]


class TestInjectedRetrySchedule:
    """The retry schedule is injected at construction (task 2706's window, task 2880's sentinel)."""

    FIXED = (0.5, 1.5, 3.0, 6.0)

    async def test_request_consumes_the_schedule_until_recovery(self):
        answers = [*_handshake(), *[httpx.ConnectError('refused')] * 4, _response()]
        with _scripted(answers) as transport:
            result = await _session(self.FIXED).call_tool('t', {})

        assert result['result'] == {'ok': True}
        assert transport.posted_methods().count('tools/call') == 5
        assert [c.args[0] for c in transport.sleep.call_args_list] == list(self.FIXED)

    async def test_request_exhausts_the_schedule(self):
        with (
            _scripted(httpx.ConnectError('refused')) as transport,
            pytest.raises(McpCallError) as excinfo,
        ):
            await _session(self.FIXED).call_tool('t', {})

        assert transport.client.post.call_count == 5
        assert '5 attempts' in str(excinfo.value)

    async def test_notify_consumes_the_schedule(self):
        answers = [_response(), *[httpx.ConnectError('refused')] * 4, _response(status_code=202)]
        with _scripted(answers) as transport:
            await _session(self.FIXED).initialize()

        assert transport.client.post.call_count == 6
        assert [c.args[0] for c in transport.sleep.call_args_list] == list(self.FIXED)

    async def test_the_default_schedule_is_a_single_attempt(self):
        with (
            _scripted(httpx.ConnectError('refused')) as transport,
            pytest.raises(McpCallError) as excinfo,
        ):
            await McpSession(BASE_URL, client_name='test-client').call_tool('t', {})

        assert excinfo.value.kind is McpFailureKind.TRANSPORT
        assert transport.client.post.call_count == 1
        assert transport.sleep.call_count == 0


# ---------------------------------------------------------------------------
# Wire facts, observed by a real-socket stub server
# ---------------------------------------------------------------------------


def _session_headers(server: StubMcpServer) -> list[str | None]:
    return [r.headers.get('mcp-session-id') for r in server.requests]


class TestAgainstStubServer:
    async def test_initialize_advertises_the_protocol_version_and_client_name(self):
        with StubMcpServer(stateful=False) as server:
            await McpSession(server.url, client_name='alarm-poster').initialize()
            init = server.requests[0].body

        assert init['method'] == 'initialize'
        assert init['params']['protocolVersion'] == MCP_PROTOCOL_VERSION
        assert init['params']['clientInfo']['name'] == 'alarm-poster'

    async def test_every_request_hits_the_slashless_endpoint_with_the_canonical_accept(self):
        with StubMcpServer(stateful=False) as server:
            await McpSession(f'{server.url}/', client_name='c').call_tool('get_status', {})
            requests = server.requests

        assert [r.body['method'] for r in requests] == [
            'initialize', 'notifications/initialized', 'tools/call',
        ]
        assert {r.path for r in requests} == {'/mcp'}
        assert {r.headers['accept'] for r in requests} == {MCP_POST_HEADERS['Accept']}

    async def test_a_stateful_server_minted_id_is_echoed_and_never_invented(self):
        with StubMcpServer(stateful=True) as server:
            session = McpSession(server.url, client_name='c')
            await session.call_tool('get_status', {})
            await session.call_tool('get_status', {})
            sent = _session_headers(server)
            minted = server.minted_session_ids

        assert len(minted) == 1
        assert sent[0] is None
        assert sent[1:] == [minted[0]] * (len(sent) - 1)

    async def test_a_stateless_server_is_sent_no_session_id(self):
        with StubMcpServer(stateful=False) as server:
            await McpSession(server.url, client_name='c').call_tool('get_status', {})
            sent = _session_headers(server)

        assert sent == [None, None, None]

    async def test_an_sse_framed_tool_result_decodes_to_the_json_rpc_body(self):
        payload = {'statuses': {'7': 'done'}}
        with StubMcpServer(stateful=True, payload=payload) as server:
            body = await McpSession(server.url, client_name='c').call_tool('get_statuses', {})

        assert body['jsonrpc'] == '2.0'
        assert body['result']['structuredContent'] == payload

    async def test_call_auto_initializes_before_any_method(self):
        with StubMcpServer(stateful=True) as server:
            body = await McpSession(server.url, client_name='c').call('tools/list')
            methods = [r.body['method'] for r in server.requests]

        assert methods == ['initialize', 'notifications/initialized', 'tools/list']
        assert body['result'] == {'tools': []}

    async def test_an_undecodable_body_is_a_typed_failure(self):
        with (
            StubMcpServer(stateful=False, answer=StubAnswer.NON_JSON_BODY) as server,
            pytest.raises(McpCallError) as excinfo,
        ):
            await McpSession(server.url, client_name='c').call_tool('update_task', {})

        assert excinfo.value.kind is McpFailureKind.UNDECODABLE
