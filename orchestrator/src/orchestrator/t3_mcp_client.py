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

import base64
import contextlib
import hashlib
import http.client
import ipaddress
import itertools
import json
import math
import os
import re
import shutil
import subprocess
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, TypeVar
from urllib.parse import parse_qs, urlencode, urlsplit

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
"""How long a tool call may wait for its reply by default: long enough for a full wait slice."""
HTTP_TIMEOUT_SECS = 30
"""How long every other request waits: the descriptor, OAuth, and the MCP handshake."""
MAX_TOKEN_LIFETIME_SECS = 10 * 365 * 24 * 3600
"""The longest ``expires_in`` accepted, far past T3's 30 days and well inside datetime's range."""
CLI_TIMEOUT_SECS = 60

_DESCRIPTOR_PATH = '/.well-known/t3/environment'
_MCP_PATH = '/mcp'
_REGISTER_PATH = '/oauth/mcp/register'
_DECISION_PATH = '/oauth/mcp/decision'
_TOKEN_PATH = '/oauth/mcp/token'
_ENVIRONMENT_ID = re.compile(r'[0-9A-Za-z][0-9A-Za-z_-]*')
_CLI_VERSION_LINE = re.compile(r'\S+ v(\S+)')
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


# ---------------------------------------------------------------------------
# Typed results
# ---------------------------------------------------------------------------


class TransportFault(StrEnum):
    UNREACHABLE = 'unreachable'
    CREDENTIAL_EXPIRED = 'credential-expired'
    PROTOCOL = 'protocol'


@dataclass(frozen=True)
class TransportFailed:
    fault: TransportFault
    detail: str


@dataclass(frozen=True)
class ToolOk:
    value: Mapping[str, Any]


@dataclass(frozen=True)
class ToolFailed:
    """A declared ``OrchestratorMcpFailure``: T3 answered and refused."""

    code: str
    message: str


CallResult = ToolOk | ToolFailed | TransportFailed


# ---------------------------------------------------------------------------
# HTTP seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Response:
    status: int
    stream: http.client.HTTPResponse
    connection: http.client.HTTPConnection

    def header(self, name: str) -> str | None:
        return self.stream.getheader(name)

    def close(self) -> None:
        self.connection.close()


def _open(
    origin: str, method: str, path: str, *, body: bytes | None, headers: Mapping[str, str], timeout: float
) -> _Response | TransportFailed:
    """Send one request to ``origin``; the response stays open for the caller to read.

    http.client follows no redirect, so a token or a pairing code never leaves
    the checked origin.
    """
    parts = urlsplit(checked_origin(origin))
    connection_class = http.client.HTTPSConnection if parts.scheme == 'https' else http.client.HTTPConnection
    connection = connection_class(parts.netloc, timeout=timeout)
    try:
        connection.request(method, path, body=body, headers=dict(headers))
        stream = connection.getresponse()
    except (OSError, http.client.HTTPException) as error:
        connection.close()
        return TransportFailed(TransportFault.UNREACHABLE, f'{method} {path}: {error!r}')
    return _Response(stream.status, stream, connection)


@dataclass(frozen=True)
class _JsonReply:
    status: int
    body: Mapping[str, Any] | None
    """The decoded body when it is a JSON object, else None."""


def _request_json(
    origin: str, method: str, path: str, *, body: bytes | None = None, content_type: str | None = None
) -> _JsonReply | TransportFailed:
    headers = {'Accept': 'application/json'} | ({'Content-Type': content_type} if content_type else {})
    response = _open(origin, method, path, body=body, headers=headers, timeout=HTTP_TIMEOUT_SECS)
    if isinstance(response, TransportFailed):
        return response
    with contextlib.closing(response):
        try:
            raw = response.stream.read()
        except (OSError, http.client.HTTPException) as error:
            return TransportFailed(TransportFault.UNREACHABLE, f'{method} {path}: {error!r}')
    try:
        decoded = json.loads(raw)
    except ValueError:
        decoded = None
    return _JsonReply(response.status, decoded if isinstance(decoded, dict) else None)


# ---------------------------------------------------------------------------
# Environment descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Descriptor:
    environment_id: str
    server_version: str
    label: str


class DescriptorFailure(StrEnum):
    """Why no descriptor: nothing answered, or something answered that is not a usable T3."""

    UNREACHABLE = 'descriptor-unreachable'
    INCOMPATIBLE = 'descriptor-incompatible'


@dataclass(frozen=True)
class DescriptorFailed:
    reason: DescriptorFailure
    detail: str


def fetch_descriptor(origin: str) -> Descriptor | DescriptorFailed:
    """GET T3's unauthenticated ``/.well-known/t3/environment``."""
    reply = _request_json(origin, 'GET', _DESCRIPTOR_PATH)
    if isinstance(reply, TransportFailed):
        return DescriptorFailed(DescriptorFailure.UNREACHABLE, reply.detail)
    body = reply.body
    if reply.status != 200 or body is None:
        return DescriptorFailed(DescriptorFailure.INCOMPATIBLE, f'descriptor: HTTP {reply.status} without a JSON object')
    environment_id, server_version, label = body.get('environmentId'), body.get('serverVersion'), body.get('label')
    if not isinstance(environment_id, str) or not _ENVIRONMENT_ID.fullmatch(environment_id):
        return DescriptorFailed(DescriptorFailure.INCOMPATIBLE, f'descriptor: unusable environmentId {environment_id!r}')
    if not isinstance(server_version, str) or not server_version:
        return DescriptorFailed(DescriptorFailure.INCOMPATIBLE, f'descriptor: unusable serverVersion {server_version!r}')
    return Descriptor(environment_id, server_version, label if isinstance(label, str) else '')


def credential_status_at(
    origin: str, *, root: Path | str | None = None, now: datetime | None = None
) -> CredentialStatus | DescriptorFailed:
    """The credential state for whichever environment ``origin`` answers as.

    A credential issued for another origin is UNPAIRED here: its bearer was
    issued for that origin's ``/mcp``, so it is never sent to this one.
    """
    checked = checked_origin(origin)
    descriptor = fetch_descriptor(checked)
    if isinstance(descriptor, DescriptorFailed):
        return descriptor
    status = read_credential_status(descriptor.environment_id, root=root, now=now)
    if status.credential is not None and status.credential.origin != checked:
        return CredentialStatus(CredentialState.UNPAIRED, descriptor.environment_id, None)
    return status


# ---------------------------------------------------------------------------
# JSON-RPC over streamable HTTP
# ---------------------------------------------------------------------------


def _sse_event_data(stream: http.client.HTTPResponse) -> Iterator[str]:
    """The data of each ``message`` (or unnamed) event, read line by line as it arrives."""
    event, data = 'message', []
    for raw in stream:
        line = raw.decode('utf-8').rstrip('\r\n')
        if not line:
            if data and event == 'message':
                yield '\n'.join(data)
            event, data = 'message', []
            continue
        name, _, value = line.partition(':')
        value = value.removeprefix(' ')
        if name == 'event':
            event = value
        elif name == 'data':
            data.append(value)


def _is_reply_to(message: object, request_id: int) -> bool:
    """A response to ``request_id``; a server-to-client request such as ``ping`` carries an id of its own."""
    return (
        isinstance(message, dict)
        and 'method' not in message
        and ('result' in message or 'error' in message)
        and message.get('id') == request_id
    )


def _read_message(response: _Response, request_id: int) -> dict[str, Any] | TransportFailed:
    """The JSON-RPC message answering ``request_id``, from a JSON body or an SSE stream."""
    try:
        if (response.header('Content-Type') or '').startswith('text/event-stream'):
            for data in _sse_event_data(response.stream):
                message = json.loads(data)
                if _is_reply_to(message, request_id):
                    return message
            return TransportFailed(TransportFault.PROTOCOL, f'request {request_id}: the stream ended before the reply')
        message = json.loads(response.stream.read())
    except (OSError, http.client.HTTPException) as error:
        return TransportFailed(TransportFault.UNREACHABLE, f'request {request_id}: {error!r}')
    except ValueError as error:
        return TransportFailed(TransportFault.PROTOCOL, f'request {request_id}: undecodable reply: {error}')
    if not _is_reply_to(message, request_id):
        return TransportFailed(TransportFault.PROTOCOL, f'request {request_id}: the body is not a reply to it')
    return message


def _first_text_json(result: Mapping[str, Any]) -> dict[str, Any] | None:
    content = result.get('content')
    if not isinstance(content, list) or not content or not isinstance(content[0], dict):
        return None
    text = content[0].get('text')
    if not isinstance(text, str):
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _tool_result(result: Mapping[str, Any]) -> CallResult:
    """Read a ``CallToolResult``: a declared failure, else its structured (or text) value."""
    if result.get('isError') is True:
        failure = _first_text_json(result) or {}
        code, message = failure.get('code'), failure.get('message')
        if not isinstance(code, str) or not isinstance(message, str):
            return TransportFailed(TransportFault.PROTOCOL, 'a failed tool result without a failure code')
        return ToolFailed(code, message)
    structured = result.get('structuredContent')
    value = structured if isinstance(structured, dict) else _first_text_json(result)
    if value is None:
        return TransportFailed(TransportFault.PROTOCOL, 'a tool result without a JSON object value')
    return ToolOk(value)


@dataclass(frozen=True)
class _RpcResult:
    result: Mapping[str, Any]
    session_id: str | None


class _Refusal(StrEnum):
    """The two HTTP refusals a client recovers from; ``call`` never returns one."""

    UNAUTHORIZED = 'unauthorized'
    SESSION_LOST = 'session-lost'


_UNRECOVERED_FAULT = {
    _Refusal.UNAUTHORIZED: TransportFault.CREDENTIAL_EXPIRED,
    _Refusal.SESSION_LOST: TransportFault.PROTOCOL,
}


class T3McpClient:
    """An MCP session with one T3 environment, authenticated by one credential.

    A lost session (404) is re-initialized once per call. A refused bearer
    (401) is re-paired through ``repair`` at most once in the client's life,
    which bounds how many pairing codes a server refusing every token can mint.
    """

    def __init__(
        self,
        origin: str,
        credential: Credential,
        *,
        root: Path | str | None = None,
        repair: CodeSource | None = None,
    ) -> None:
        self._origin = checked_origin(origin)
        self._credential = credential
        self._root = root
        self._repair = repair
        self._repaired = False
        self._session: dict[str, str] | None = None
        """The headers that carry the session; None until ``initialize`` succeeds."""
        self._request_ids = itertools.count(1)

    @property
    def environment_id(self) -> str:
        return self._credential.environment_id

    def call(
        self, tool: str, arguments: Mapping[str, Any], *, read_timeout: float = MCP_READ_TIMEOUT_SECS
    ) -> CallResult:
        """One ``tools/call``, waiting up to ``read_timeout`` for its reply; the handshake waits :data:`HTTP_TIMEOUT_SECS`."""
        outcome = self._attempt(tool, arguments, read_timeout)
        if outcome is _Refusal.SESSION_LOST:
            self._session = None
            outcome = self._attempt(tool, arguments, read_timeout)
        if outcome is _Refusal.UNAUTHORIZED:
            outcome = self._repaired_attempt(tool, arguments, read_timeout)
        if isinstance(outcome, _Refusal):
            return TransportFailed(_UNRECOVERED_FAULT[outcome], f'{tool}: {outcome.value} again after recovery')
        return outcome

    def _repaired_attempt(self, tool: str, arguments: Mapping[str, Any], read_timeout: float) -> CallResult | _Refusal:
        if self._repair is None or self._repaired:
            spent = 'no repair source' if self._repair is None else 'already re-paired once'
            return TransportFailed(TransportFault.CREDENTIAL_EXPIRED, f'{tool}: T3 refused the bearer ({spent})')
        self._repaired = True
        paired = pair(self._origin, self._repair, root=self._root)
        if isinstance(paired, PairFailed):
            return TransportFailed(
                TransportFault.CREDENTIAL_EXPIRED, f'{tool}: T3 refused the bearer; re-pair {paired.summary()}'
            )
        if paired.credential.environment_id != self.environment_id:
            detail = (
                f'{tool}: T3 refused the bearer; {self._origin} now answers as environment '
                f'{paired.credential.environment_id}, not {self.environment_id}'
            )
            return TransportFailed(TransportFault.CREDENTIAL_EXPIRED, detail)
        self._credential = paired.credential
        self._session = None
        return self._attempt(tool, arguments, read_timeout)

    def _attempt(self, tool: str, arguments: Mapping[str, Any], read_timeout: float) -> CallResult | _Refusal:
        if self._session is None:
            refused = self._initialize()
            if refused is not None:
                return refused
        reply = self._rpc('tools/call', {'name': tool, 'arguments': dict(arguments)}, timeout=read_timeout)
        if isinstance(reply, _RpcResult):
            return _tool_result(reply.result)
        return reply

    def _initialize(self) -> TransportFailed | _Refusal | None:
        params = {'protocolVersion': PROTOCOL_VERSION, 'capabilities': {}, 'clientInfo': {'name': SOFTWARE_ID, 'version': '1'}}
        reply = self._rpc('initialize', params, timeout=HTTP_TIMEOUT_SECS)
        if not isinstance(reply, _RpcResult):
            return reply
        self._session = {'Mcp-Session-Id': reply.session_id} if reply.session_id else {}
        refused = self._notify_initialized()
        if refused is not None:
            self._session = None
        return refused

    def _notify_initialized(self) -> TransportFailed | _Refusal | None:
        response = self._post({'jsonrpc': '2.0', 'method': 'notifications/initialized'}, timeout=HTTP_TIMEOUT_SECS)
        if isinstance(response, TransportFailed):
            return response
        with contextlib.closing(response):
            if response.status in (200, 202):
                return None
        return self._unanswered('notifications/initialized', response.status)

    def _unanswered(self, method: str, status: int) -> TransportFailed | _Refusal:
        """A non-success status: a 401, a 404 on a held session, or PROTOCOL."""
        if status == 401:
            return _Refusal.UNAUTHORIZED
        if status == 404 and self._session:
            return _Refusal.SESSION_LOST
        return TransportFailed(TransportFault.PROTOCOL, f'{method}: HTTP {status}')

    def _rpc(
        self, method: str, params: Mapping[str, Any], *, timeout: float
    ) -> _RpcResult | TransportFailed | _Refusal:
        request_id = next(self._request_ids)
        response = self._post({'jsonrpc': '2.0', 'id': request_id, 'method': method, 'params': params}, timeout=timeout)
        if isinstance(response, TransportFailed):
            return response
        with contextlib.closing(response):
            if response.status != 200:
                return self._unanswered(method, response.status)
            message = _read_message(response, request_id)
        if isinstance(message, TransportFailed):
            return message
        error = message.get('error')
        if error is not None:
            detail = error.get('message') if isinstance(error, dict) else error
            return TransportFailed(TransportFault.PROTOCOL, f'{method}: {detail}')
        result = message.get('result')
        if not isinstance(result, dict):
            return TransportFailed(TransportFault.PROTOCOL, f'{method}: a reply without a result object')
        return _RpcResult(result, response.header('Mcp-Session-Id'))

    def _post(self, message: Mapping[str, Any], *, timeout: float) -> _Response | TransportFailed:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
            'Authorization': f'Bearer {self._credential.access_token}',
            'MCP-Protocol-Version': PROTOCOL_VERSION,
            **(self._session or {}),
        }
        body = json.dumps(message).encode()
        return _open(self._origin, 'POST', _MCP_PATH, body=body, headers=headers, timeout=timeout)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


class PairFailure(StrEnum):
    PAIRING_REJECTED = 'pairing_rejected'
    CLI_VERSION_MISMATCH = 'cli-version-mismatch'
    T3_MISSING = 't3-missing'
    CLI_VERSION_UNREADABLE = 'cli-version-unreadable'
    PAIRING_CREATE_FAILED = 'pairing-create-failed'
    UNREACHABLE = 'unreachable'
    INCOMPATIBLE = 'incompatible'
    ENVIRONMENT_MISMATCH = 'environment-mismatch'


@dataclass(frozen=True)
class PairFailed:
    reason: PairFailure | DescriptorFailure
    detail: str | None = None
    cli_version: str | None = None
    server_version: str | None = None

    def summary(self) -> str:
        """One line, the reason's word first."""
        parts = [self.reason.value]
        if self.cli_version is not None or self.server_version is not None:
            parts.append(f'cli {self.cli_version} server {self.server_version}')
        if self.detail is not None:
            parts.append(self.detail)
        return ': '.join(parts)


@dataclass(frozen=True)
class Paired:
    credential: Credential


class CodeSource(Protocol):
    """Where a one-time pairing code comes from: the one axis along which pairing varies."""

    @property
    def paired_by(self) -> PairedBy: ...

    def mint(self, descriptor: Descriptor) -> str | PairFailed: ...


@dataclass(frozen=True)
class OperatorCode:
    """A code an operator minted on the T3 host and handed over."""

    code: str

    @property
    def paired_by(self) -> PairedBy:
        return PairedBy.OPERATOR

    def mint(self, descriptor: Descriptor) -> str | PairFailed:
        return self.code


def parse_cli_version(stdout: str) -> str | None:
    """The bare version in ``t3 --version``'s ``t3 v0.1.6``.

    The CLI prints Effect's ``<name> v<semver>`` (``apps/server/src/binCli.ts``)
    while the descriptor's serverVersion is the bare ``0.1.6``, so the two are
    compared after this parse; any other line is None, never a guess.
    """
    match = _CLI_VERSION_LINE.fullmatch(stdout.strip())
    return match.group(1) if match else None


def _run_t3(executable: str, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            [executable, *args], capture_output=True, text=True, timeout=CLI_TIMEOUT_SECS, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _cli_version_refusal(executable: str, descriptor: Descriptor) -> PairFailed | None:
    """None when this host's ``t3`` is the server's own version."""
    ran = _run_t3(executable, '--version')
    if ran is None or ran.returncode != 0:
        return PairFailed(PairFailure.CLI_VERSION_UNREADABLE, 't3 --version did not run cleanly')
    cli_version = parse_cli_version(ran.stdout)
    if cli_version is None:
        return PairFailed(PairFailure.CLI_VERSION_UNREADABLE, f't3 --version printed {ran.stdout.strip()!r}')
    if cli_version != descriptor.server_version:
        return PairFailed(
            PairFailure.CLI_VERSION_MISMATCH, cli_version=cli_version, server_version=descriptor.server_version
        )
    return None


def _created_pairing_code(created: subprocess.CompletedProcess[str] | None) -> str | PairFailed:
    """The ``credential`` of ``t3 auth pairing create --json`` (``cliAuthFormat.ts::formatIssuedPairingCredential``)."""
    if created is None:
        return PairFailed(PairFailure.PAIRING_CREATE_FAILED, 't3 auth pairing create did not run')
    if created.returncode != 0:
        return PairFailed(PairFailure.PAIRING_CREATE_FAILED, created.stderr.strip() or f'exit {created.returncode}')
    try:
        issued = json.loads(created.stdout)
    except ValueError:
        return PairFailed(PairFailure.PAIRING_CREATE_FAILED, 'output is not JSON')
    code = issued.get('credential') if isinstance(issued, dict) else None
    if not isinstance(code, str) or not code:
        return PairFailed(PairFailure.PAIRING_CREATE_FAILED, 'output has no credential')
    return code


@dataclass(frozen=True)
class CliCodeSource:
    """Mint a code with this host's own ``t3``, refusing one of another version before minting."""

    @property
    def paired_by(self) -> PairedBy:
        return PairedBy.SELF_PAIR

    def mint(self, descriptor: Descriptor) -> str | PairFailed:
        executable = shutil.which('t3')
        if executable is None:
            return PairFailed(PairFailure.T3_MISSING, 'no t3 on PATH')
        refusal = _cli_version_refusal(executable, descriptor)
        if refusal is not None:
            return refusal
        return _created_pairing_code(_run_t3(executable, 'auth', 'pairing', 'create', '--json'))


@dataclass(frozen=True)
class _Issued:
    client_id: str
    access_token: str
    expires_in: int


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode('ascii').rstrip('=')


def _oauth_post(
    origin: str, path: str, *, body: bytes, content_type: str, expected_status: int
) -> Mapping[str, Any] | PairFailed:
    """One OAuth endpoint's JSON object reply; a 400 is T3 refusing, anything else odd is INCOMPATIBLE."""
    reply = _request_json(origin, 'POST', path, body=body, content_type=content_type)
    if isinstance(reply, TransportFailed):
        return PairFailed(PairFailure.UNREACHABLE, reply.detail)
    if reply.status == 400:
        refusal = reply.body or {}
        message = refusal.get('error_description') or refusal.get('message') or 'HTTP 400'
        return PairFailed(PairFailure.PAIRING_REJECTED, f'{path}: {message}')
    if reply.status != expected_status or reply.body is None:
        return PairFailed(PairFailure.INCOMPATIBLE, f'{path}: HTTP {reply.status}')
    return reply.body


def _register(origin: str) -> str | PairFailed:
    payload = {'client_name': CLIENT_NAME, 'redirect_uris': [REDIRECT_URI], 'software_id': SOFTWARE_ID}
    registered = _oauth_post(
        origin, _REGISTER_PATH, body=json.dumps(payload).encode(), content_type='application/json', expected_status=201
    )
    if isinstance(registered, PairFailed):
        return registered
    client_id = registered.get('client_id')
    if not isinstance(client_id, str) or not client_id:
        return PairFailed(PairFailure.INCOMPATIBLE, f'{_REGISTER_PATH}: no client_id')
    return client_id


def _approve(origin: str, client_id: str, code: str, *, challenge: str, state: str) -> str | PairFailed:
    """Spend the pairing code on an authorization code."""
    authorization = {
        'response_type': 'code',
        'client_id': client_id,
        'redirect_uri': REDIRECT_URI,
        'code_challenge': challenge,
        'code_challenge_method': 'S256',
        'state': state,
        'resource': f'{origin}{_MCP_PATH}',
    }
    payload = {'authorization': authorization, 'decision': {'_tag': 'pairing-code', 'access': ACCESS, 'code': code}}
    approved = _oauth_post(
        origin, _DECISION_PATH, body=json.dumps(payload).encode(), content_type='application/json', expected_status=200
    )
    if isinstance(approved, PairFailed):
        return approved
    redirect_to = approved.get('redirectTo')
    if not isinstance(redirect_to, str):
        return PairFailed(PairFailure.INCOMPATIBLE, f'{_DECISION_PATH}: no redirectTo')
    query = parse_qs(urlsplit(redirect_to).query)
    if query.get('state') != [state] or len(query.get('code', [])) != 1:
        reason = query.get('error_description', ['no code for this state'])[0]
        return PairFailed(PairFailure.PAIRING_REJECTED, f'{_DECISION_PATH}: {reason}')
    return query['code'][0]


def _redeem(origin: str, client_id: str, authorization_code: str, *, verifier: str) -> _Issued | PairFailed:
    form = {
        'grant_type': 'authorization_code',
        'code': authorization_code,
        'redirect_uri': REDIRECT_URI,
        'client_id': client_id,
        'code_verifier': verifier,
        'resource': f'{origin}{_MCP_PATH}',
    }
    token = _oauth_post(
        origin,
        _TOKEN_PATH,
        body=urlencode(form).encode(),
        content_type='application/x-www-form-urlencoded',
        expected_status=200,
    )
    if isinstance(token, PairFailed):
        return token
    access_token, expires_in = token.get('access_token'), token.get('expires_in')
    if token.get('token_type') != 'Bearer' or not isinstance(access_token, str) or not access_token:
        return PairFailed(PairFailure.INCOMPATIBLE, f'{_TOKEN_PATH}: no Bearer access_token')
    lifetime_secs = _lifetime_secs(expires_in)
    if lifetime_secs is None:
        return PairFailed(PairFailure.INCOMPATIBLE, f'{_TOKEN_PATH}: unusable expires_in {expires_in!r}')
    return _Issued(client_id, access_token, lifetime_secs)


def _lifetime_secs(expires_in: object) -> int | None:
    """Whole seconds when ``expires_in`` is finite and in (0, :data:`MAX_TOKEN_LIFETIME_SECS`]; JSON admits ``Infinity`` and ``NaN``."""
    if isinstance(expires_in, bool) or not isinstance(expires_in, int | float):
        return None
    if not math.isfinite(expires_in) or not 0 < expires_in <= MAX_TOKEN_LIFETIME_SECS:
        return None
    return int(expires_in)


def _exchange(origin: str, code: str) -> _Issued | PairFailed:
    """The pairing-code sign-in of T3's own ``apps/server/src/peer/PeerLinks.ts::signIn``, PKCE S256."""
    client_id = _register(origin)
    if isinstance(client_id, PairFailed):
        return client_id
    verifier = _b64url(os.urandom(32))
    challenge = _b64url(hashlib.sha256(verifier.encode('ascii')).digest())
    authorization_code = _approve(origin, client_id, code, challenge=challenge, state=_b64url(os.urandom(16)))
    if isinstance(authorization_code, PairFailed):
        return authorization_code
    return _redeem(origin, client_id, authorization_code, verifier=verifier)


_VALIDATION_FAILURE = {
    TransportFault.CREDENTIAL_EXPIRED: PairFailure.PAIRING_REJECTED,
    TransportFault.UNREACHABLE: PairFailure.UNREACHABLE,
    TransportFault.PROTOCOL: PairFailure.INCOMPATIBLE,
}


def _validation_refusal(credential: Credential, *, root: Path | str | None) -> PairFailed | None:
    """None when one ``t3_environment_read`` on the new token answers for the descriptor's environment."""
    validator = T3McpClient(credential.origin, credential, root=root)
    result = validator.call('t3_environment_read', {}, read_timeout=HTTP_TIMEOUT_SECS)
    if isinstance(result, TransportFailed):
        return PairFailed(_VALIDATION_FAILURE[result.fault], f't3_environment_read: {result.detail}')
    if isinstance(result, ToolFailed):
        return PairFailed(PairFailure.PAIRING_REJECTED, f't3_environment_read: {result.code}: {result.message}')
    answered = result.value.get('environmentId')
    if answered != credential.environment_id:
        detail = f'the token answers for {answered!r}, the descriptor for {credential.environment_id!r}'
        return PairFailed(PairFailure.ENVIRONMENT_MISMATCH, detail)
    return None


def pair(origin: str, source: CodeSource, *, root: Path | str | None = None) -> Paired | PairFailed:
    """Pair with the environment at ``origin``; the credential is stored only once it is validated."""
    checked = checked_origin(origin)
    descriptor = fetch_descriptor(checked)
    if isinstance(descriptor, DescriptorFailed):
        return PairFailed(descriptor.reason, descriptor.detail)
    code = source.mint(descriptor)
    if isinstance(code, PairFailed):
        return code
    issued = _exchange(checked, code)
    if isinstance(issued, PairFailed):
        return issued
    paired_at = datetime.now(UTC).replace(microsecond=0)
    credential = Credential(
        schema_version=CREDENTIAL_SCHEMA_VERSION,
        origin=checked,
        environment_id=descriptor.environment_id,
        server_version=descriptor.server_version,
        access=ACCESS,
        client_id=issued.client_id,
        access_token=issued.access_token,
        expires_at=paired_at + timedelta(seconds=issued.expires_in),
        paired_at=paired_at,
        paired_by=source.paired_by,
    )
    refusal = _validation_refusal(credential, root=root)
    if refusal is not None:
        return refusal
    write_credential(credential, root=root)
    return Paired(credential)


# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------


class ConnectFailure(StrEnum):
    UNPAIRED = 'unpaired'
    CREDENTIAL_EXPIRED = 'credential-expired'
    PAIR_FAILED = 'pair-failed'


@dataclass(frozen=True)
class ConnectFailed:
    reason: ConnectFailure | DescriptorFailure
    detail: str | None = None


def connect(
    origin: str, *, root: Path | str | None = None, repair: CodeSource | None = None
) -> T3McpClient | ConnectFailed:
    """A client for the environment ``origin`` answers as, on its stored credential.

    An expired or missing credential is re-paired through ``repair`` when one
    is given (``CliCodeSource`` on the T3 host), else refused.
    """
    status = credential_status_at(origin, root=root)
    if isinstance(status, DescriptorFailed):
        return ConnectFailed(status.reason, status.detail)
    if status.state is CredentialState.PAIRED and status.credential is not None:
        return T3McpClient(origin, status.credential, root=root, repair=repair)
    if repair is None:
        unpaired = status.state is CredentialState.UNPAIRED
        reason = ConnectFailure.UNPAIRED if unpaired else ConnectFailure.CREDENTIAL_EXPIRED
        return ConnectFailed(reason, f'environment {status.environment_id}')
    paired = pair(origin, repair, root=root)
    if isinstance(paired, PairFailed):
        return ConnectFailed(ConnectFailure.PAIR_FAILED, paired.summary())
    return T3McpClient(origin, paired.credential, root=root, repair=repair)
