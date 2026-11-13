"""Self-tests for the fake T3 server in _fake_t3_server: it mints values in T3 Code v0.1.6's own shapes,
and a test can script its tool replies per call."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from _fake_t3_server import HANG_UP, FakeT3Server, ToolSuccess, replies_in_turn

from orchestrator import t3_mcp_client as client

T3_PAIRING_CODE = re.compile(r'[23456789ABCDEFGHJKLMNPQRSTUVWXYZ]{12}')


def test_every_pairing_code_the_fake_mints_has_t3s_own_shape() -> None:
    with FakeT3Server.running() as server:
        codes = [server.add_pairing_code() for _ in range(256)]

    assert [code for code in codes if not T3_PAIRING_CODE.fullmatch(code)] == []
    assert len(set(codes)) == len(codes)


@pytest.fixture
def server() -> Iterator[FakeT3Server]:
    with FakeT3Server.running() as running:
        yield running


@pytest.fixture
def mcp(server: FakeT3Server, tmp_path: Path) -> client.T3McpClient:
    """A client on a server-issued bearer, stored the way a pairing stores it."""
    server.write_paired_credential(tmp_path)
    connected = client.connect(server.origin, root=tmp_path)
    assert isinstance(connected, client.T3McpClient)
    return connected


def test_a_responder_answers_with_what_it_returns_for_the_calls_arguments(
    server: FakeT3Server, mcp: client.T3McpClient
) -> None:
    def echo(arguments: Mapping[str, Any]) -> ToolSuccess:
        return ToolSuccess({'echoed': dict(arguments)})

    server.tool_replies['t3_echo'] = echo

    assert mcp.call('t3_echo', {'threadId': 'thread-1'}) == client.ToolOk({'echoed': {'threadId': 'thread-1'}})
    assert mcp.call('t3_echo', {}) == client.ToolOk({'echoed': {}})


def test_replies_in_turn_answers_each_reply_once_then_repeats_the_last(
    server: FakeT3Server, mcp: client.T3McpClient
) -> None:
    server.tool_replies['t3_turns'] = replies_in_turn(ToolSuccess({'turn': 'A'}), ToolSuccess({'turn': 'B'}))

    answers = [mcp.call('t3_turns', {}) for _ in range(3)]

    assert answers == [client.ToolOk({'turn': 'A'}), client.ToolOk({'turn': 'B'}), client.ToolOk({'turn': 'B'})]


def test_a_hang_up_drops_only_that_call_after_recording_it(server: FakeT3Server, mcp: client.T3McpClient) -> None:
    server.tool_replies['t3_flaky'] = replies_in_turn(HANG_UP, ToolSuccess({'ok': True}))

    dropped = mcp.call('t3_flaky', {'attempt': 1})
    answered = mcp.call('t3_flaky', {'attempt': 2})

    assert isinstance(dropped, client.TransportFailed)
    assert answered == client.ToolOk({'ok': True})
    assert server.tool_calls('t3_flaky') == [{'attempt': 1}, {'attempt': 2}]


def test_a_written_paired_credential_connects_on_a_bearer_the_server_issued(
    server: FakeT3Server, tmp_path: Path
) -> None:
    path = server.write_paired_credential(tmp_path)

    assert path == client.credential_path(server.environment_id, root=tmp_path)
    assert path.exists()
    stored = client.read_credential_status(server.environment_id, root=tmp_path).credential
    assert stored is not None
    assert (stored.origin, stored.environment_id, stored.server_version) == (
        server.origin,
        server.environment_id,
        server.server_version,
    )
    assert stored.access_token in server.issued_tokens()
    connected = client.connect(server.origin, root=tmp_path)
    assert isinstance(connected, client.T3McpClient)
    assert isinstance(connected.call('t3_environment_read', {}), client.ToolOk)


def test_a_paired_credential_can_carry_a_given_bearer(server: FakeT3Server, tmp_path: Path) -> None:
    server.write_paired_credential(tmp_path, access_token='revoked')

    stored = client.read_credential_status(server.environment_id, root=tmp_path).credential
    assert stored is not None
    assert stored.access_token == 'revoked'
    assert 'revoked' not in server.issued_tokens()
