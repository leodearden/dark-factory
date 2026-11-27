"""Tests for shared.mcp_session.SyncMcpSession, the MCP client for scripts.

Every test runs against a real-socket :class:`StubMcpServer`: the handshake,
the session id echo and the release are wire facts the server observes.
"""
from __future__ import annotations

import json

import httpx
import pytest

from shared.mcp_post import MCP_POST_HEADERS
from shared.mcp_session import (
    MCP_PROTOCOL_VERSION,
    McpCallError,
    McpFailureKind,
    SyncMcpSession,
)
from shared.testing_mcp_stub import RecordedRequest, StubAnswer, StubMcpServer

PAYLOAD = {'success': True, 'escalation_id': 'esc-1'}
ARGUMENTS = {'task_id': '7', 'summary': 'disk nearly full'}


def _post_one_alarm(server: StubMcpServer, base_url: str | None = None) -> dict:
    with SyncMcpSession(base_url or server.url, client_name='alarm-poster') as session:
        return session.call_tool('escalate_info', ARGUMENTS)


def _posts(requests: tuple[RecordedRequest, ...]) -> list[RecordedRequest]:
    return [r for r in requests if r.http_method == 'POST']


def _deletes(requests: tuple[RecordedRequest, ...]) -> list[RecordedRequest]:
    return [r for r in requests if r.http_method == 'DELETE']


class TestStatefulServer:
    def test_initialize_is_session_less_and_identifies_the_client(self):
        with StubMcpServer(stateful=True, payload=PAYLOAD) as server:
            _post_one_alarm(server)
            first = server.requests[0]

        assert first.body['method'] == 'initialize'
        assert 'mcp-session-id' not in first.headers
        assert first.body['params']['clientInfo']['name'] == 'alarm-poster'
        assert first.body['params']['protocolVersion'] == MCP_PROTOCOL_VERSION

    def test_the_minted_id_is_echoed_and_never_invented(self):
        with StubMcpServer(stateful=True, payload=PAYLOAD) as server:
            _post_one_alarm(server)
            requests = server.requests
            minted = server.minted_session_ids

        assert len(minted) == 1
        later = _posts(requests)[1:]
        assert [r.body['method'] for r in later] == ['notifications/initialized', 'tools/call']
        assert all(r.headers.get('mcp-session-id') == minted[0] for r in later)
        assert {r.headers.get('mcp-session-id') for r in requests} <= {None, minted[0]}

    def test_exit_releases_the_minted_session(self):
        with StubMcpServer(stateful=True, payload=PAYLOAD) as server:
            _post_one_alarm(server)
            deletes = _deletes(server.requests)
            minted = server.minted_session_ids
            still_open = server.open_session_ids

        assert [d.headers.get('mcp-session-id') for d in deletes] == [minted[0]]
        assert still_open == frozenset()

    def test_the_reply_is_the_unwrapped_tool_payload(self):
        with StubMcpServer(stateful=True, payload=PAYLOAD) as server:
            assert _post_one_alarm(server) == PAYLOAD


IDENTITY_HEADERS = {'X-Escalation-Identity': 'interactive:coordinator'}


class TestCallerHeaders:
    def test_caller_headers_ride_every_post_beside_the_session_echo(self):
        with StubMcpServer(stateful=True, payload=PAYLOAD) as server:
            with SyncMcpSession(server.url, client_name='t', headers=IDENTITY_HEADERS) as session:
                session.call_tool('escalate_info', ARGUMENTS)
            posts = _posts(server.requests)
            minted = server.minted_session_ids

        assert [r.body['method'] for r in posts] == [
            'initialize', 'notifications/initialized', 'tools/call',
        ]
        assert all(r.headers.get('x-escalation-identity') == 'interactive:coordinator' for r in posts)
        assert all(r.headers.get('mcp-session-id') == minted[0] for r in posts[1:])

    def test_a_session_without_caller_headers_sends_none(self):
        with StubMcpServer(stateful=True, payload=PAYLOAD) as server:
            _post_one_alarm(server)
            requests = server.requests

        assert all('x-escalation-identity' not in r.headers for r in requests)

    @pytest.mark.parametrize('name', ['Accept', 'content-type', 'Mcp-Session-Id', 'MCP-SESSION-ID'])
    def test_a_caller_header_may_not_override_a_protocol_header(self, name: str):
        with pytest.raises(ValueError, match=name):
            SyncMcpSession('http://127.0.0.1:1', client_name='t', headers={name: 'x'})


class TestStatelessServer:
    def test_no_session_id_is_sent_and_nothing_is_released(self):
        with StubMcpServer(stateful=False, payload=PAYLOAD) as server:
            reply = _post_one_alarm(server)
            requests = server.requests

        assert reply == PAYLOAD
        assert all('mcp-session-id' not in r.headers for r in requests)
        assert _deletes(requests) == []

    def test_every_post_hits_the_slashless_endpoint_with_the_canonical_accept(self):
        with StubMcpServer(stateful=False, payload=PAYLOAD) as server:
            _post_one_alarm(server, base_url=f'{server.url}/')
            posts = _posts(server.requests)

        assert [r.body['method'] for r in posts] == [
            'initialize', 'notifications/initialized', 'tools/call',
        ]
        assert {r.path for r in posts} == {'/mcp'}
        assert {r.headers['accept'] for r in posts} == {MCP_POST_HEADERS['Accept']}


class TestLifecycle:
    def test_an_exception_in_the_block_propagates_and_the_session_is_still_released(self):
        class Boom(Exception):
            pass

        with StubMcpServer(stateful=True) as server:
            with pytest.raises(Boom), SyncMcpSession(server.url, client_name='c'):
                raise Boom
            deletes = _deletes(server.requests)
            still_open = server.open_session_ids

        assert len(deletes) == 1
        assert still_open == frozenset()

    def test_a_failed_release_is_not_raised(self):
        with (
            StubMcpServer(stateful=True, payload=PAYLOAD) as server,
            SyncMcpSession(server.url, client_name='c') as session,
        ):
            assert session.call_tool('escalate_info', ARGUMENTS) == PAYLOAD
            server.close()

    def test_call_tool_outside_an_entered_session_is_refused(self):
        with StubMcpServer(stateful=False) as server:
            session = SyncMcpSession(server.url, client_name='c')
            with pytest.raises(RuntimeError):
                session.call_tool('escalate_info', ARGUMENTS)
            with session:
                pass
            with pytest.raises(RuntimeError):
                session.call_tool('escalate_info', ARGUMENTS)

    def test_a_json_rpc_error_is_a_typed_failure(self):
        with (
            StubMcpServer(stateful=True, answer=StubAnswer.JSON_RPC_ERROR) as server,
            SyncMcpSession(server.url, client_name='c') as session,
            pytest.raises(McpCallError) as excinfo,
        ):
            session.call_tool('escalate_info', ARGUMENTS)

        assert excinfo.value.kind is McpFailureKind.JSON_RPC_ERROR

    def test_an_unreachable_server_fails_the_handshake_as_transport(self):
        with StubMcpServer(stateful=False) as server:
            url = server.url

        with pytest.raises(McpCallError) as excinfo, SyncMcpSession(url, client_name='c'):
            pass

        assert excinfo.value.kind is McpFailureKind.TRANSPORT


class TestInjectedClient:
    def test_every_post_goes_through_the_injected_client(self):
        seen: list[httpx.Request] = []
        client = httpx.Client(trust_env=False, event_hooks={'request': [seen.append]})
        with StubMcpServer(stateful=True, payload=PAYLOAD) as server:
            with SyncMcpSession(server.url, client_name='c', http_client=client) as session:
                session.call_tool('escalate_info', ARGUMENTS)
            arrived = server.requests

        assert [(r.method, r.url.path) for r in seen] == [(r.http_method, r.path) for r in arrived]
        assert [r.method for r in seen].count('POST') == 3

    def test_exit_releases_the_session_but_leaves_the_injected_client_open(self):
        client = httpx.Client(trust_env=False)
        with StubMcpServer(stateful=True, payload=PAYLOAD) as server:
            with SyncMcpSession(server.url, client_name='c', http_client=client):
                pass
            still_open = server.open_session_ids

        assert still_open == frozenset()
        assert client.is_closed is False
        client.close()


def _scripted_list(pages: dict[str | None, dict]) -> httpx.Client:
    """A client whose server answers the handshake and then each tools/list cursor from *pages*."""

    def answer(request: httpx.Request) -> httpx.Response:
        message = json.loads(request.content)
        if 'id' not in message:
            return httpx.Response(202)
        if message['method'] == 'initialize':
            return httpx.Response(200, json={'jsonrpc': '2.0', 'id': message['id'], 'result': {}})
        cursor = (message.get('params') or {}).get('cursor')
        return httpx.Response(200, json={'jsonrpc': '2.0', 'id': message['id'], 'result': pages[cursor]})

    return httpx.Client(transport=httpx.MockTransport(answer))


class TestListTools:
    def test_the_listed_names_arrive_in_order(self):
        with (
            StubMcpServer(stateful=False, listed_tools=('a', 'b')) as server,
            SyncMcpSession(server.url, client_name='c') as session,
        ):
            assert session.list_tools() == ('a', 'b')

    def test_every_page_is_followed_until_no_cursor_remains(self):
        pages = {
            None: {'tools': [{'name': 'a'}], 'nextCursor': 'p2'},
            'p2': {'tools': [{'name': 'b'}]},
        }
        with SyncMcpSession('http://fused.test', client_name='c', http_client=_scripted_list(pages)) as session:
            assert session.list_tools() == ('a', 'b')

    @pytest.mark.parametrize(
        'result',
        [{}, {'tools': 'a'}, {'tools': [{'name': 7}]}, {'tools': ['a']}],
        ids=['no-tools', 'tools-not-a-list', 'name-not-a-str', 'tool-not-a-mapping'],
    )
    def test_a_result_without_a_list_of_named_tools_is_a_reply_shape_failure(self, result):
        client = _scripted_list({None: result})
        with (
            SyncMcpSession('http://fused.test', client_name='c', http_client=client) as session,
            pytest.raises(McpCallError) as excinfo,
        ):
            session.list_tools()

        assert excinfo.value.kind is McpFailureKind.REPLY_SHAPE
