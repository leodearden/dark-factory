"""Tests for orchestrator.t3_mcp_client, the /spawn T3 lane's client of T3's /mcp.

The T3 side is ``_fake_t3_server``: a real HTTP server on an ephemeral loopback
port and a real ``t3`` executable on PATH, so every test drives the client over
the wire it uses in production.
"""

from __future__ import annotations

import json
import math
import socket
import stat
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _fake_t3_server import (
    DEAD_ORIGIN,
    DECISION_PATH,
    ENVIRONMENT_ID,
    PAIRING_CREATE_MODE_ENV,
    REGISTER_PATH,
    TOKEN_PATH,
    FakeT3Server,
    PairingCreateMode,
    ToolFailure,
    ToolSuccess,
    cli_calls,
    put_fake_t3_cli_on_path,
)

from orchestrator import t3_mcp_client as client

NOW = datetime(2026, 11, 2, 12, 0, 0, tzinfo=UTC)
OTHER_ENVIRONMENT_ID = '7c1e0d9b-2a3f-4b5c-8d6e-0f1a2b3c4d5e'

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
    server: FakeT3Server,
    root: Path,
    *,
    expires_at: datetime | None = None,
    access_token: str | None = None,
    origin: str | None = None,
) -> client.Credential:
    paired_at = datetime.now(UTC).replace(microsecond=0)
    credential = make_credential(
        origin=origin if origin is not None else server.origin,
        environment_id=server.environment_id,
        access_token=access_token if access_token is not None else server.issue_token(),
        paired_at=paired_at,
        expires_at=expires_at if expires_at is not None else paired_at + timedelta(days=30),
    )
    client.write_credential(credential, root=root)
    return credential


@pytest.fixture
def t3_cli(t3_server: FakeT3Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The real CLI's version line for the fake server's version."""
    return put_fake_t3_cli_on_path(t3_server, tmp_path, monkeypatch, version_line='t3 v0.1.6')


PAIRING_CREATE = ['auth', 'pairing', 'create', '--json']


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

    def test_a_server_hanging_up_after_connect_is_unreachable(self, t3_server: FakeT3Server, fleet: Path):
        seed_credential(t3_server, fleet)
        mcp = connected(t3_server, fleet)
        assert isinstance(mcp.call('t3_environment_read', {}), client.ToolOk)
        t3_server.hangs_up = True

        result = mcp.call('t3_environment_read', {})

        assert isinstance(result, client.TransportFailed)
        assert result.fault is client.TransportFault.UNREACHABLE

    def test_a_client_of_a_dead_origin_is_unreachable(self):
        result = client.T3McpClient(DEAD_ORIGIN, make_credential(origin=DEAD_ORIGIN)).call('t3_environment_read', {})

        assert isinstance(result, client.TransportFailed)
        assert result.fault is client.TransportFault.UNREACHABLE

    def test_a_server_request_bearing_the_request_id_is_not_taken_for_the_reply(
        self, t3_server: FakeT3Server, fleet: Path
    ):
        t3_server.framing = 'sse'
        t3_server.pings_with_the_request_id = True
        seed_credential(t3_server, fleet)

        result = connected(t3_server, fleet).call('t3_environment_read', {})

        assert isinstance(result, client.ToolOk)

    def test_the_handshake_on_a_silent_server_gives_up_after_the_http_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(client, 'HTTP_TIMEOUT_SECS', 0.2)
        with socket.create_server(('127.0.0.1', 0)) as silent:
            origin = f'http://127.0.0.1:{silent.getsockname()[1]}'
            mcp = client.T3McpClient(origin, make_credential(origin=origin))
            outcome: list[client.CallResult] = []
            caller = threading.Thread(target=lambda: outcome.append(mcp.call('t3_environment_read', {})), daemon=True)
            caller.start()
            caller.join(timeout=10)
            gave_up_in_time = not caller.is_alive()
        caller.join(timeout=10)

        assert gave_up_in_time
        [result] = outcome
        assert isinstance(result, client.TransportFailed)
        assert result.fault is client.TransportFault.UNREACHABLE

    def test_connect_to_a_closed_port_fails_on_the_descriptor(self, fleet: Path):
        result = client.connect(DEAD_ORIGIN, root=fleet)

        assert isinstance(result, client.ConnectFailed)
        assert result.reason is client.DescriptorFailure.UNREACHABLE

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

    def test_connect_never_sends_a_bearer_issued_for_another_origin(self, t3_server: FakeT3Server, fleet: Path):
        seed_credential(t3_server, fleet, origin='https://elsewhere.example')

        result = client.connect(t3_server.origin, root=fleet)
        status = client.credential_status_at(t3_server.origin, root=fleet)

        assert isinstance(result, client.ConnectFailed)
        assert result.reason is client.ConnectFailure.UNPAIRED
        assert touched_mcp_or_oauth(t3_server) == []
        assert isinstance(status, client.CredentialStatus)
        assert status.state is client.CredentialState.UNPAIRED

    def test_credential_status_at_selects_the_file_the_descriptor_names(self, t3_server: FakeT3Server, fleet: Path):
        seed_credential(t3_server, fleet)

        status = client.credential_status_at(t3_server.origin, root=fleet, now=NOW)

        assert status == client.read_credential_status(t3_server.environment_id, root=fleet, now=NOW)
        assert isinstance(status, client.CredentialStatus)
        assert status.state is client.CredentialState.PAIRED

    def test_credential_status_at_a_closed_port_is_unreachable(self, fleet: Path):
        status = client.credential_status_at(DEAD_ORIGIN, root=fleet)

        assert isinstance(status, client.DescriptorFailed)
        assert status.reason is client.DescriptorFailure.UNREACHABLE


@pytest.fixture(params=['descriptor-404', 'unusable-environment-id'])
def incompatible_server(request: pytest.FixtureRequest) -> Iterator[FakeT3Server]:
    """A server that answers, but not with a descriptor this client can use."""
    options: dict[str, Any] = (
        {'serves_descriptor': False} if request.param == 'descriptor-404' else {'environment_id': '../elsewhere'}
    )
    with FakeT3Server.running(**options) as server:
        yield server


class TestDescriptor:
    def test_a_server_answering_an_unusable_descriptor_is_incompatible(self, incompatible_server: FakeT3Server):
        result = client.fetch_descriptor(incompatible_server.origin)

        assert isinstance(result, client.DescriptorFailed)
        assert result.reason is client.DescriptorFailure.INCOMPATIBLE
        assert result.reason.value == 'descriptor-incompatible'
        assert result.detail

    def test_a_closed_port_is_unreachable(self):
        result = client.fetch_descriptor(DEAD_ORIGIN)

        assert isinstance(result, client.DescriptorFailed)
        assert result.reason is client.DescriptorFailure.UNREACHABLE
        assert result.reason.value == 'descriptor-unreachable'

    def test_connect_to_an_incompatible_server_neither_calls_nor_pairs(
        self, incompatible_server: FakeT3Server, fleet: Path
    ):
        result = client.connect(incompatible_server.origin, root=fleet, repair=client.OperatorCode('any'))

        assert isinstance(result, client.ConnectFailed)
        assert result.reason is client.DescriptorFailure.INCOMPATIBLE
        assert touched_mcp_or_oauth(incompatible_server) == []
        assert no_credential_written(fleet)

    def test_pair_with_an_incompatible_server_sends_no_oauth_request(
        self, incompatible_server: FakeT3Server, fleet: Path
    ):
        result = client.pair(incompatible_server.origin, client.OperatorCode('any'), root=fleet)

        assert isinstance(result, client.PairFailed)
        assert result.reason is client.DescriptorFailure.INCOMPATIBLE
        assert touched_oauth(incompatible_server) == []
        assert no_credential_written(fleet)


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

    def test_a_redirect_for_another_state_is_rejected_before_any_token_request(
        self, t3_server: FakeT3Server, fleet: Path
    ):
        t3_server.decision_state_override = 'forged-state'

        result = client.pair(t3_server.origin, client.OperatorCode(t3_server.add_pairing_code()), root=fleet)

        assert isinstance(result, client.PairFailed)
        assert result.reason is client.PairFailure.PAIRING_REJECTED
        assert t3_server.oauth_calls(TOKEN_PATH) == []
        assert no_credential_written(fleet)

    @pytest.mark.parametrize(
        'override',
        [
            {'token_type': 'mac'},
            {'expires_in': math.inf},
            {'expires_in': math.nan},
            {'expires_in': 1e12},
            {'expires_in': 0},
            {'expires_in': '3600'},
            {'expires_in': True},
        ],
        ids=['not-bearer', 'infinite', 'nan', 'beyond-datetime', 'zero', 'string', 'bool'],
    )
    def test_an_unusable_token_reply_is_incompatible_and_not_stored(
        self, t3_server: FakeT3Server, fleet: Path, override: dict[str, Any]
    ):
        t3_server.token_reply_overrides = override

        result = client.pair(t3_server.origin, client.OperatorCode(t3_server.add_pairing_code()), root=fleet)

        assert isinstance(result, client.PairFailed)
        assert result.reason is client.PairFailure.INCOMPATIBLE
        assert t3_server.mcp_requests() == []
        assert no_credential_written(fleet)

    def test_a_token_whose_validation_call_is_refused_is_not_stored(self, t3_server: FakeT3Server, fleet: Path):
        t3_server.tool_replies['t3_environment_read'] = ToolFailure('forbidden', 'This credential cannot read.')

        result = client.pair(t3_server.origin, client.OperatorCode(t3_server.add_pairing_code()), root=fleet)

        assert isinstance(result, client.PairFailed)
        assert result.reason is client.PairFailure.PAIRING_REJECTED
        assert result.detail is not None and 'forbidden' in result.detail
        assert no_credential_written(fleet)

    def test_an_unreachable_descriptor_fails_before_any_exchange(self, fleet: Path):
        result = client.pair(DEAD_ORIGIN, client.OperatorCode('any'), root=fleet)

        assert isinstance(result, client.PairFailed)
        assert result.reason is client.DescriptorFailure.UNREACHABLE
        assert no_credential_written(fleet)

    def test_a_token_answering_for_another_environment_is_not_stored(self, t3_server: FakeT3Server, fleet: Path):
        t3_server.tool_replies['t3_environment_read'] = ToolSuccess({'environmentId': OTHER_ENVIRONMENT_ID})

        result = client.pair(t3_server.origin, client.OperatorCode(t3_server.add_pairing_code()), root=fleet)

        assert isinstance(result, client.PairFailed)
        assert result.reason is client.PairFailure.ENVIRONMENT_MISMATCH
        assert no_credential_written(fleet)


class TestSelfPair:
    def test_a_matching_cli_mints_the_code_and_pairs(self, t3_server: FakeT3Server, t3_cli: Path, fleet: Path):
        result = client.pair(t3_server.origin, client.CliCodeSource(), root=fleet)

        assert isinstance(result, client.Paired)
        assert result.credential.paired_by is client.PairedBy.SELF_PAIR
        assert cli_calls(t3_cli) == [['--version'], PAIRING_CREATE]
        [minted] = t3_server.issued_pairing_codes()
        [decision] = t3_server.oauth_calls(DECISION_PATH)
        assert decision['decision']['code'] == minted
        path = client.credential_path(t3_server.environment_id, root=fleet)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_a_cli_of_another_version_mints_nothing(
        self, t3_server: FakeT3Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fleet: Path
    ):
        log = put_fake_t3_cli_on_path(t3_server, tmp_path, monkeypatch, version_line='t3 v0.1.4')

        result = client.pair(t3_server.origin, client.CliCodeSource(), root=fleet)

        assert result == client.PairFailed(
            client.PairFailure.CLI_VERSION_MISMATCH, cli_version='0.1.4', server_version='0.1.6'
        )
        assert isinstance(result, client.PairFailed)
        assert result.reason.value == 'cli-version-mismatch'
        assert cli_calls(log) == [['--version']]
        assert t3_server.issued_pairing_codes() == []
        assert touched_oauth(t3_server) == []
        assert no_credential_written(fleet)

    def test_an_unparsable_version_line_mints_nothing(
        self, t3_server: FakeT3Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fleet: Path
    ):
        log = put_fake_t3_cli_on_path(t3_server, tmp_path, monkeypatch, version_line='garbage')

        result = client.pair(t3_server.origin, client.CliCodeSource(), root=fleet)

        assert isinstance(result, client.PairFailed)
        assert result.reason is client.PairFailure.CLI_VERSION_UNREADABLE
        assert cli_calls(log) == [['--version']]

    def test_no_t3_on_path_is_t3_missing(
        self, t3_server: FakeT3Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fleet: Path
    ):
        empty = tmp_path / 'empty-bin'
        empty.mkdir()
        monkeypatch.setenv('PATH', str(empty))

        result = client.pair(t3_server.origin, client.CliCodeSource(), root=fleet)

        assert isinstance(result, client.PairFailed)
        assert result.reason is client.PairFailure.T3_MISSING
        assert no_credential_written(fleet)

    @pytest.mark.parametrize('mode', [PairingCreateMode.FAILS, PairingCreateMode.OMITS_CREDENTIAL])
    def test_a_failed_pairing_create_sends_no_oauth_request(
        self,
        t3_server: FakeT3Server,
        t3_cli: Path,
        monkeypatch: pytest.MonkeyPatch,
        fleet: Path,
        mode: PairingCreateMode,
    ):
        monkeypatch.setenv(PAIRING_CREATE_MODE_ENV, mode.value)

        result = client.pair(t3_server.origin, client.CliCodeSource(), root=fleet)

        assert isinstance(result, client.PairFailed)
        assert result.reason is client.PairFailure.PAIRING_CREATE_FAILED
        assert cli_calls(t3_cli) == [['--version'], PAIRING_CREATE]
        assert touched_oauth(t3_server) == []
        assert no_credential_written(fleet)


class TestCredentialRecovery:
    @pytest.mark.parametrize('framing', ['json', 'sse'])
    def test_a_revoked_token_is_repaired_once_per_client(
        self, t3_server: FakeT3Server, t3_cli: Path, fleet: Path, framing: str
    ):
        t3_server.framing = framing
        seeded = seed_credential(t3_server, fleet)
        mcp = connected(t3_server, fleet, repair=client.CliCodeSource())
        assert isinstance(mcp.call('t3_environment_read', {}), client.ToolOk)
        t3_server.revoke_tokens()

        repaired = mcp.call('t3_environment_read', {})

        assert isinstance(repaired, client.ToolOk)
        assert cli_calls(t3_cli).count(PAIRING_CREATE) == 1
        stored = client.read_credential_status(t3_server.environment_id, root=fleet).credential
        assert stored is not None
        assert stored.access_token != seeded.access_token
        assert stored.paired_by is client.PairedBy.SELF_PAIR

        t3_server.revoke_tokens()
        refused = mcp.call('t3_environment_read', {})

        assert isinstance(refused, client.TransportFailed)
        assert refused.fault is client.TransportFault.CREDENTIAL_EXPIRED
        assert refused.fault.value == 'credential-expired'
        assert cli_calls(t3_cli).count(PAIRING_CREATE) == 1

    def test_a_revoked_token_without_a_repair_source_is_credential_expired(
        self, t3_server: FakeT3Server, t3_cli: Path, fleet: Path
    ):
        seed_credential(t3_server, fleet)
        mcp = connected(t3_server, fleet)
        assert isinstance(mcp.call('t3_environment_read', {}), client.ToolOk)
        t3_server.revoke_tokens()

        result = mcp.call('t3_environment_read', {})

        assert isinstance(result, client.TransportFailed)
        assert result.fault is client.TransportFault.CREDENTIAL_EXPIRED
        assert cli_calls(t3_cli) == []

    def test_a_token_refused_at_initialize_is_repaired(self, t3_server: FakeT3Server, t3_cli: Path, fleet: Path):
        seed_credential(t3_server, fleet, access_token='never-issued')
        mcp = connected(t3_server, fleet, repair=client.CliCodeSource())

        result = mcp.call('t3_environment_read', {})

        assert isinstance(result, client.ToolOk)
        assert cli_calls(t3_cli).count(PAIRING_CREATE) == 1

    def test_a_repair_answering_as_another_environment_does_not_switch_the_client(
        self, t3_server: FakeT3Server, t3_cli: Path, fleet: Path
    ):
        seed_credential(t3_server, fleet)
        mcp = connected(t3_server, fleet, repair=client.CliCodeSource())
        assert isinstance(mcp.call('t3_environment_read', {}), client.ToolOk)
        t3_server.answer_as(OTHER_ENVIRONMENT_ID)
        t3_server.revoke_tokens()

        result = mcp.call('t3_environment_read', {})

        assert isinstance(result, client.TransportFailed)
        assert result.fault is client.TransportFault.CREDENTIAL_EXPIRED
        assert ENVIRONMENT_ID in result.detail and OTHER_ENVIRONMENT_ID in result.detail
        assert mcp.environment_id == ENVIRONMENT_ID

    @pytest.mark.parametrize('stored', ['expired', 'unpaired', 'another-origin'])
    def test_connect_with_a_repair_source_self_pairs_an_unusable_credential(
        self, t3_server: FakeT3Server, t3_cli: Path, fleet: Path, stored: str
    ):
        if stored == 'expired':
            seed_credential(t3_server, fleet, expires_at=datetime.now(UTC) + timedelta(minutes=30))
        if stored == 'another-origin':
            seed_credential(t3_server, fleet, origin='https://elsewhere.example')

        result = client.connect(t3_server.origin, root=fleet, repair=client.CliCodeSource())

        assert isinstance(result, client.T3McpClient)
        status = client.read_credential_status(t3_server.environment_id, root=fleet)
        assert status.state is client.CredentialState.PAIRED
        assert status.credential is not None
        assert status.credential.paired_by is client.PairedBy.SELF_PAIR
        assert status.credential.origin == t3_server.origin
        assert isinstance(result.call('t3_environment_read', {}), client.ToolOk)

    def test_connect_whose_self_pair_is_refused_leaves_the_old_file(
        self, t3_server: FakeT3Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fleet: Path
    ):
        put_fake_t3_cli_on_path(t3_server, tmp_path, monkeypatch, version_line='t3 v0.1.4')
        seed_credential(t3_server, fleet, expires_at=datetime.now(UTC) + timedelta(minutes=30))
        path = client.credential_path(t3_server.environment_id, root=fleet)
        before = path.read_bytes()

        result = client.connect(t3_server.origin, root=fleet, repair=client.CliCodeSource())

        assert isinstance(result, client.ConnectFailed)
        assert result.reason is client.ConnectFailure.PAIR_FAILED
        assert result.detail is not None and 'cli-version-mismatch' in result.detail
        assert path.read_bytes() == before

    def test_a_lost_session_reinitializes_without_repairing(self, t3_server: FakeT3Server, t3_cli: Path, fleet: Path):
        seed_credential(t3_server, fleet)
        mcp = connected(t3_server, fleet, repair=client.CliCodeSource())
        assert isinstance(mcp.call('t3_environment_read', {}), client.ToolOk)
        t3_server.forget_sessions()

        result = mcp.call('t3_environment_read', {})

        assert isinstance(result, client.ToolOk)
        assert mcp_methods(t3_server).count('initialize') == 2
        assert len(t3_server.issued_sessions()) == 2
        assert cli_calls(t3_cli) == []
