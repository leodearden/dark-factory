"""Tests for orchestrator.t3_mcp_client, the /spawn T3 lane's client of T3's /mcp.

The T3 side is ``_fake_t3_server``: a real HTTP server on an ephemeral loopback
port and a real ``t3`` executable on PATH, so every test drives the client over
the wire it uses in production.
"""

from __future__ import annotations

import json
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _fake_t3_server import ENVIRONMENT_ID

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
