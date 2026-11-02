"""orchestrator.t3_mcp_client — the /spawn T3 lane's client of T3 Code's ``/mcp``.

It pairs this host with a T3 server through the pairing-code OAuth exchange,
keeps the resulting bearer in a per-environment credential file under the fleet
root, and makes JSON-RPC tool calls over T3's streamable-HTTP MCP endpoint.
Every transport outcome is returned as a typed value; nothing here raises for
a refusal, an outage or a malformed reply.

It is imported at module scope by ``spawn_t3_lane.py``, which the spawn shell
runs with no venv, so it is stdlib-only plus ``shared`` leaves reached through
``session_registry``'s bootstrap: the tier-2 contract pinned by
``orchestrator/tests/test_session_registry.py::TestStdlibOnlySelfContainment``.

Contracts: ``plans/spawn-t3-lane-mcp-and-host-registry-prd.md`` §6.2 (the
credential file) and §6.6 (the T3 0.1.6 surface).
"""

from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlsplit

from orchestrator import session_registry

# session_registry's import puts this checkout's shared/src on sys.path for the bare shell.
# isort: split
from shared.safe_io import atomic_write_text

PROTOCOL_VERSION = '2025-06-18'
SOFTWARE_ID = 'dark-factory-spawn-t3-lane'
CLIENT_NAME = 'Dark Factory /spawn T3 lane'
REDIRECT_URI = 'http://127.0.0.1/dark-factory-spawn-t3-lane'
"""Registered with T3 and never fetched: the code comes back in the decision reply."""
ACCESS = 'full-access'
CREDENTIAL_SCHEMA_VERSION = 1
EXPIRY_MARGIN = timedelta(hours=1)
T3_MAX_WAIT_SLICE_SECS = 3600
"""T3's ``OrchestratorMcpService.ts`` ``MAX_WAIT_TIMEOUT_MS``: the longest wait a tool call may hold."""
MCP_READ_TIMEOUT_SECS = T3_MAX_WAIT_SLICE_SECS + 60
HTTP_TIMEOUT_SECS = 30
CLI_TIMEOUT_SECS = 60

_ENVIRONMENT_ID = re.compile(r'[0-9A-Za-z][0-9A-Za-z_-]*')
_T = TypeVar('_T')


# ---------------------------------------------------------------------------
# Origin
# ---------------------------------------------------------------------------


def _is_loopback(host: str) -> bool:
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def checked_origin(url: str) -> str:
    """``scheme://host[:port]`` of ``url``: https to any host, plain http only to loopback.

    A bearer token and a pairing code travel to this origin, so anything else
    raises ValueError.
    """
    parts = urlsplit(url)
    host = parts.hostname
    if parts.scheme not in ('http', 'https') or not host:
        raise ValueError(f'{url!r} is not an http(s) URL with a host')
    if parts.username is not None or parts.password is not None:
        raise ValueError(f'{url!r} carries credentials in its authority')
    if parts.scheme == 'http' and not _is_loopback(host):
        raise ValueError(f'{url!r}: plain http is allowed only to a loopback host; use https')
    authority = f'[{host}]' if ':' in host else host
    port = parts.port
    return f'{parts.scheme}://{authority}' if port is None else f'{parts.scheme}://{authority}:{port}'


# ---------------------------------------------------------------------------
# Credential file
# ---------------------------------------------------------------------------


class PairedBy(StrEnum):
    SELF_PAIR = 'self-pair'
    OPERATOR = 'operator'


class CredentialState(StrEnum):
    PAIRED = 'paired'
    EXPIRED = 'expired'
    UNPAIRED = 'unpaired'


def _typed(record: dict[str, Any], name: str, kind: type[_T]) -> _T:
    value = record.get(name)
    if type(value) is not kind:
        raise ValueError(f'credential field {name!r} is missing or not a {kind.__name__}')
    return value


def _instant(record: dict[str, Any], name: str) -> datetime:
    instant = datetime.fromisoformat(_typed(record, name, str))
    if instant.utcoffset() is None:
        raise ValueError(f'credential field {name!r} has no UTC offset')
    return instant


@dataclass(frozen=True)
class Credential:
    """One pairing's bearer, in the field order of PRD §6.2."""

    schema_version: int
    origin: str
    environment_id: str
    server_version: str
    access: str
    client_id: str
    access_token: str
    expires_at: datetime
    paired_at: datetime
    paired_by: PairedBy

    def to_json(self) -> str:
        record: dict[str, Any] = {field.name: getattr(self, field.name) for field in fields(self)}
        record['expires_at'] = self.expires_at.isoformat()
        record['paired_at'] = self.paired_at.isoformat()
        record['paired_by'] = self.paired_by.value
        return json.dumps(record, indent=2) + '\n'

    @classmethod
    def from_json(cls, text: str) -> Credential:
        record = json.loads(text)
        if not isinstance(record, dict):
            raise ValueError('a credential file holds one JSON object')
        if _typed(record, 'schema_version', int) != CREDENTIAL_SCHEMA_VERSION:
            raise ValueError(f'unknown credential schema_version {record["schema_version"]!r}')
        return cls(
            schema_version=CREDENTIAL_SCHEMA_VERSION,
            origin=_typed(record, 'origin', str),
            environment_id=_typed(record, 'environment_id', str),
            server_version=_typed(record, 'server_version', str),
            access=_typed(record, 'access', str),
            client_id=_typed(record, 'client_id', str),
            access_token=_typed(record, 'access_token', str),
            expires_at=_instant(record, 'expires_at'),
            paired_at=_instant(record, 'paired_at'),
            paired_by=PairedBy(_typed(record, 'paired_by', str)),
        )


@dataclass(frozen=True)
class CredentialStatus:
    state: CredentialState
    environment_id: str
    credential: Credential | None


def credential_path(environment_id: str, *, root: Path | str | None = None) -> Path:
    if not _ENVIRONMENT_ID.fullmatch(environment_id):
        raise ValueError(f'environment id {environment_id!r} cannot name a credential file')
    return session_registry.fleet_root(root) / 't3-mcp' / f'{environment_id}.json'


def write_credential(credential: Credential, *, root: Path | str | None = None) -> None:
    path = credential_path(credential.environment_id, root=root)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_write_text(path, credential.to_json(), mode=0o600)


def read_credential_status(
    environment_id: str, *, root: Path | str | None = None, now: datetime | None = None
) -> CredentialStatus:
    """PAIRED, EXPIRED within :data:`EXPIRY_MARGIN` of ``expires_at``, or UNPAIRED when unreadable."""
    try:
        credential = Credential.from_json(credential_path(environment_id, root=root).read_text())
    except (OSError, ValueError):
        return CredentialStatus(CredentialState.UNPAIRED, environment_id, None)
    if credential.environment_id != environment_id:
        return CredentialStatus(CredentialState.UNPAIRED, environment_id, None)
    current = now if now is not None else datetime.now(UTC)
    if credential.expires_at <= current + EXPIRY_MARGIN:
        return CredentialStatus(CredentialState.EXPIRED, environment_id, credential)
    return CredentialStatus(CredentialState.PAIRED, environment_id, credential)
