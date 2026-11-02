"""Tests for orchestrator.t3_mcp_client, the /spawn T3 lane's client of T3's /mcp.

The T3 side is ``_fake_t3_server``: a real HTTP server on an ephemeral loopback
port and a real ``t3`` executable on PATH, so every test drives the client over
the wire it uses in production.
"""

from __future__ import annotations

import json
import socket
import stat
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _fake_t3_server import (
    DECISION_PATH,
    ENVIRONMENT_ID,
    REGISTER_PATH,
    TOKEN_PATH,
    FakeT3Server,
    ToolFailure,
    ToolSuccess,
)

from orchestrator import t3_mcp_client as client

NOW = datetime(2026, 11, 2, 12, 0, 0, tzinfo=UTC)

CREDENTIAL_KEYS = {
    'schema_version',
    'origin',
    'environment_id',
    'server_version',
    'access',
    'client_id',
    'access_token',
    'expires_at',
    'paired_at',
    'paired_by',
}


def make_credential(**overrides: Any) -> client.Credential:
    fields: dict[str, Any] = {
        'schema_version': client.CREDENTIAL_SCHEMA_VERSION,
        'origin': 'http://127.0.0.1:3773',
        'environment_id': ENVIRONMENT_ID,
        'server_version': '0.1.6',
        'access': client.ACCESS,
        'client_id': 'mcp-client-1',
        'access_token': 'token-1',
        'expires_at': NOW + timedelta(days=30),
        'paired_at': NOW,
        'paired_by': client.PairedBy.OPERATOR,
    }
    return client.Credential(**{**fields, **overrides})


@pytest.fixture
def fleet(tmp_path: Path) -> Path:
    return tmp_path / 'fleet'


@pytest.fixture
def t3_server() -> Iterator[FakeT3Server]:
    with FakeT3Server.running() as server:
        yield server


@pytest.fixture(params=['json', 'sse'])
def framed_server(request: pytest.FixtureRequest) -> Iterator[FakeT3Server]:
    with FakeT3Server.running(framing=request.param) as server:
        yield server


def seed_credential(
    server: FakeT3Server, root: Path, *, expires_at: datetime | None = None, access_token: str | None = None
) -> client.Credential:
    paired_at = datetime.now(UTC).replace(microsecond=0)
    credential = make_credential(
        origin=server.origin,
        environment_id=server.environment_id,
        access_token=access_token if access_token is not None else server.issue_token(),
        paired_at=paired_at,
        expires_at=expires_at if expires_at is not None else paired_at + timedelta(days=30),
    )
    client.write_credential(credential, root=root)
    return credential


def closed_origin() -> str:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    return f'http://127.0.0.1:{port}'


def connected(server: FakeT3Server, root: Path, **options: Any) -> client.T3McpClient:
    result = client.connect(server.origin, root=root, **options)
    assert isinstance(result, client.T3McpClient), result
    return result


def mcp_methods(server: FakeT3Server) -> list[str]:
    return [request.body['method'] for request in server.mcp_requests()]


def no_credential_written(root: Path) -> bool:
    return not (root / 't3-mcp').exists()


def touched_oauth(server: FakeT3Server) -> list[str]:
    return [request.path for request in server.requests if request.path.startswith('/oauth/')]


def touched_mcp_or_oauth(server: FakeT3Server) -> list[str]:
    return [request.path for request in server.requests if request.path == '/mcp' or request.path.startswith('/oauth/')]


class TestCredentialFile:
    @pytest.mark.parametrize('paired_by', list(client.PairedBy))
    def test_write_lays_out_one_private_file_per_environment(self, fleet: Path, paired_by: client.PairedBy):
        client.write_credential(make_credential(paired_by=paired_by), root=fleet)

        path = fleet / 't3-mcp' / f'{ENVIRONMENT_ID}.json'
        assert client.credential_path(ENVIRONMENT_ID, root=fleet) == path
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        written = json.loads(path.read_text())
        assert set(written) == CREDENTIAL_KEYS
        assert written['schema_version'] == 1
        assert written['paired_by'] == paired_by.value
        assert {member.value for member in client.PairedBy} == {'operator', 'self-pair'}

    def test_a_credential_well_inside_its_lifetime_reads_back_paired_and_equal(self, fleet: Path):
        credential = make_credential(expires_at=NOW + timedelta(hours=2))
        client.write_credential(credential, root=fleet)

        status = client.read_credential_status(ENVIRONMENT_ID, root=fleet, now=NOW)

        assert status.state is client.CredentialState.PAIRED
        assert status.environment_id == ENVIRONMENT_ID
        assert status.credential == credential

    @pytest.mark.parametrize(
        'expires_at',
        [NOW + timedelta(minutes=30), NOW + timedelta(hours=1), NOW - timedelta(days=1)],
        ids=['inside-the-margin', 'exactly-at-the-margin', 'in-the-past'],
    )
    def test_a_credential_within_the_margin_of_expiry_reads_back_expired(self, fleet: Path, expires_at: datetime):
        credential = make_credential(expires_at=expires_at)
        client.write_credential(credential, root=fleet)

        status = client.read_credential_status(ENVIRONMENT_ID, root=fleet, now=NOW)

        assert status.state is client.CredentialState.EXPIRED
        assert status.credential == credential

    def test_a_missing_file_reads_back_unpaired(self, fleet: Path):
        status = client.read_credential_status(ENVIRONMENT_ID, root=fleet, now=NOW)

        assert status.state is client.CredentialState.UNPAIRED
        assert status.environment_id == ENVIRONMENT_ID
        assert status.credential is None

    @pytest.mark.parametrize(
        'mutate',
        [
            lambda record: 'not json at all {',
            lambda record: json.dumps({key: value for key, value in record.items() if key != 'access_token'}),
            lambda record: json.dumps({**record, 'schema_version': 2}),
            lambda record: json.dumps({**record, 'expires_at': '2026-12-02T12:00:00'}),
            lambda record: json.dumps({**record, 'paired_by': 'somebody'}),
            lambda record: json.dumps({**record, 'access_token': 7}),
        ],
        ids=[
            'garbage',
            'missing-access-token',
            'wrong-schema-version',
            'naive-instant',
            'unknown-paired-by',
            'mistyped-field',
        ],
    )
    def test_an_unreadable_file_reads_back_unpaired(self, fleet: Path, mutate: Any):
        client.write_credential(make_credential(), root=fleet)
        path = client.credential_path(ENVIRONMENT_ID, root=fleet)
        path.write_text(mutate(json.loads(path.read_text())))

        status = client.read_credential_status(ENVIRONMENT_ID, root=fleet, now=NOW)

        assert status.state is client.CredentialState.UNPAIRED
        assert status.credential is None

    def test_the_three_states_are_distinct_words(self):
        states = [client.CredentialState.PAIRED, client.CredentialState.EXPIRED, client.CredentialState.UNPAIRED]

        assert len(set(states)) == 3
        assert [state.value for state in states] == ['paired', 'expired', 'unpaired']


class TestCheckedOrigin:
    @pytest.mark.parametrize(
        ('url', 'origin'),
        [
            ('http://127.0.0.1:3773/mcp', 'http://127.0.0.1:3773'),
            ('http://localhost:3773', 'http://localhost:3773'),
            ('http://[::1]:3773/', 'http://[::1]:3773'),
            ('https://host.ts.net', 'https://host.ts.net'),
        ],
    )
    def test_accepts_https_anywhere_and_plain_http_on_loopback(self, url: str, origin: str):
        assert client.checked_origin(url) == origin

    @pytest.mark.parametrize('url', ['http://192.168.1.105:3773', 'ftp://x', 'http://', '127.0.0.1:3773'])
    def test_refuses_plain_http_off_loopback_and_other_schemes(self, url: str):
        with pytest.raises(ValueError):
            client.checked_origin(url)


class TestCall:
    def test_connect_then_call_runs_the_mcp_handshake_once_and_reads_the_tool(
        self, framed_server: FakeT3Server, fleet: Path
    ):
        token = seed_credential(framed_server, fleet).access_token

        mcp = connected(framed_server, fleet)
        result = mcp.call('t3_environment_read', {})

        assert mcp.environment_id == framed_server.environment_id
        assert isinstance(result, client.ToolOk)
        assert result.value['environmentId'] == framed_server.environment_id
        requests = framed_server.mcp_requests()
        assert mcp_methods(framed_server) == ['initialize', 'notifications/initialized', 'tools/call']
        assert requests[0].body['params']['protocolVersion'] == '2025-06-18'
        assert requests[2].body['params'] == {'name': 't3_environment_read', 'arguments': {}}
        for request in requests:
            assert request.headers['authorization'] == f'Bearer {token}'
            assert 'application/json' in request.headers['accept']
            assert 'text/event-stream' in request.headers['accept']
        [session_id] = framed_server.issued_sessions()
        for request in requests[1:]:
            assert request.headers['mcp-protocol-version'] == '2025-06-18'
            assert request.headers['mcp-session-id'] == session_id

    def test_a_second_call_reuses_the_session(self, framed_server: FakeT3Server, fleet: Path):
        seed_credential(framed_server, fleet)
        mcp = connected(framed_server, fleet)

        first = mcp.call('t3_environment_read', {})
        second = mcp.call('t3_environment_read', {})

        assert isinstance(first, client.ToolOk)
        assert isinstance(second, client.ToolOk)
        assert mcp_methods(framed_server) == ['initialize', 'notifications/initialized', 'tools/call', 'tools/call']

    def test_a_declared_tool_failure_is_tool_failed(self, framed_server: FakeT3Server, fleet: Path):
        seed_credential(framed_server, fleet)
        framed_server.tool_replies['t3_thread_read'] = ToolFailure('thread_not_found', 'm')

        result = connected(framed_server, fleet).call('t3_thread_read', {'threadId': 't'})

        assert result == client.ToolFailed('thread_not_found', 'm')

    def test_a_json_rpc_error_is_a_protocol_fault(self, framed_server: FakeT3Server, fleet: Path):
        seed_credential(framed_server, fleet)

        result = connected(framed_server, fleet).call('t3_no_such_tool', {})

        assert isinstance(result, client.TransportFailed)
        assert result.fault is client.TransportFault.PROTOCOL

    def test_a_server_gone_after_connect_is_unreachable(self, fleet: Path):
        with FakeT3Server.running() as server:
            seed_credential(server, fleet)
            mcp = connected(server, fleet)
            assert isinstance(mcp.call('t3_environment_read', {}), client.ToolOk)

        result = mcp.call('t3_environment_read', {})

        assert isinstance(result, client.TransportFailed)
        assert result.fault is client.TransportFault.UNREACHABLE

    def test_connect_to_a_closed_port_fails_on_the_descriptor(self, fleet: Path):
        result = client.connect(closed_origin(), root=fleet)

        assert isinstance(result, client.ConnectFailed)
        assert result.reason is client.ConnectFailure.DESCRIPTOR_UNREACHABLE

    def test_connect_with_no_credential_file_is_unpaired_and_sends_nothing(self, t3_server: FakeT3Server, fleet: Path):
        result = client.connect(t3_server.origin, root=fleet)

        assert isinstance(result, client.ConnectFailed)
        assert result.reason is client.ConnectFailure.UNPAIRED
        assert touched_mcp_or_oauth(t3_server) == []

    def test_connect_with_an_expired_file_and_no_repair_sends_nothing(self, t3_server: FakeT3Server, fleet: Path):
        seed_credential(t3_server, fleet, expires_at=datetime.now(UTC) + timedelta(minutes=30))

        result = client.connect(t3_server.origin, root=fleet)

        assert isinstance(result, client.ConnectFailed)
        assert result.reason is client.ConnectFailure.CREDENTIAL_EXPIRED
        assert touched_mcp_or_oauth(t3_server) == []

    def test_credential_status_at_selects_the_file_the_descriptor_names(self, t3_server: FakeT3Server, fleet: Path):
        seed_credential(t3_server, fleet)

        status = client.credential_status_at(t3_server.origin, root=fleet, now=NOW)

        assert status == client.read_credential_status(t3_server.environment_id, root=fleet, now=NOW)
        assert isinstance(status, client.CredentialStatus)
        assert status.state is client.CredentialState.PAIRED

    def test_credential_status_at_a_closed_port_is_unreachable(self, fleet: Path):
        status = client.credential_status_at(closed_origin(), root=fleet)

        assert isinstance(status, client.TransportFailed)
        assert status.fault is client.TransportFault.UNREACHABLE


class TestPairWithOperatorCode:
    @pytest.mark.parametrize('expires_in', [2_592_000, 3 * 3600])
    def test_an_operator_code_pairs_and_writes_the_validated_credential(
        self, t3_server: FakeT3Server, fleet: Path, expires_in: int
    ):
        t3_server.expires_in = expires_in
        code = t3_server.add_pairing_code()

        result = client.pair(t3_server.origin, client.OperatorCode(code), root=fleet)

        assert isinstance(result, client.Paired)
        credential = result.credential
        assert credential.origin == t3_server.origin
        assert credential.environment_id == t3_server.environment_id
        assert credential.server_version == '0.1.6'
        assert credential.access == 'full-access'
        assert [credential.client_id] == t3_server.issued_client_ids()
        assert [credential.access_token] == t3_server.issued_tokens()
        assert credential.paired_by is client.PairedBy.OPERATOR
        assert credential.expires_at - credential.paired_at == timedelta(seconds=expires_in)
        status = client.read_credential_status(t3_server.environment_id, root=fleet)
        assert status.state is client.CredentialState.PAIRED
        assert status.credential == credential
        path = client.credential_path(t3_server.environment_id, root=fleet)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_the_exchange_registers_decides_and_redeems_once_then_validates_once(
        self, t3_server: FakeT3Server, fleet: Path
    ):
        code = t3_server.add_pairing_code()

        client.pair(t3_server.origin, client.OperatorCode(code), root=fleet)

        [register] = t3_server.oauth_calls(REGISTER_PATH)
        assert register['software_id'] == client.SOFTWARE_ID
        assert register['redirect_uris'] == [client.REDIRECT_URI]
        [decision] = t3_server.oauth_calls(DECISION_PATH)
        assert decision['decision'] == {'_tag': 'pairing-code', 'access': 'full-access', 'code': code}
        assert decision['authorization']['code_challenge_method'] == 'S256'
        assert decision['authorization']['resource'] == t3_server.origin + '/mcp'
        [token] = t3_server.oauth_calls(TOKEN_PATH)
        assert token['grant_type'] == 'authorization_code'
        assert token['client_id'] == t3_server.issued_client_ids()[0]
        assert token['redirect_uri'] == client.REDIRECT_URI
        assert t3_server.tool_calls('t3_environment_read') == [{}]

    def test_a_wrong_code_is_rejected_before_any_token_request(self, t3_server: FakeT3Server, fleet: Path):
        result = client.pair(t3_server.origin, client.OperatorCode('not-a-code'), root=fleet)

        assert isinstance(result, client.PairFailed)
        assert result.reason is client.PairFailure.PAIRING_REJECTED
        assert result.reason.value == 'pairing_rejected'
        assert result.detail is not None and 'The pairing code is invalid or expired.' in result.detail
        assert t3_server.oauth_calls(TOKEN_PATH) == []
        assert no_credential_written(fleet)

    def test_an_unreachable_descriptor_fails_before_any_exchange(self, fleet: Path):
        result = client.pair(closed_origin(), client.OperatorCode('any'), root=fleet)

        assert isinstance(result, client.PairFailed)
        assert result.reason is client.PairFailure.DESCRIPTOR_UNREACHABLE
        assert no_credential_written(fleet)

    def test_a_token_answering_for_another_environment_is_not_stored(self, t3_server: FakeT3Server, fleet: Path):
        t3_server.tool_replies['t3_environment_read'] = ToolSuccess(
            {'environmentId': 'another-environment', 'label': 'x', 'serverVersion': '0.1.6', 'platform': {}, 'preferences': {}}
        )

        result = client.pair(t3_server.origin, client.OperatorCode(t3_server.add_pairing_code()), root=fleet)

        assert isinstance(result, client.PairFailed)
        assert result.reason is client.PairFailure.ENVIRONMENT_MISMATCH
        assert no_credential_written(fleet)
