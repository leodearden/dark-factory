"""Tests for shared.mcp_session.SyncMcpSession, the MCP client for scripts.

Every test runs against a real-socket :class:`StubMcpServer`: the handshake,
the session id echo and the release are wire facts the server observes.
"""
from __future__ import annotations

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
