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

import contextlib
import http.client
import ipaddress
import itertools
import json
import re
from collections.abc import Iterator, Mapping
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

_DESCRIPTOR_PATH = '/.well-known/t3/environment'
_MCP_PATH = '/mcp'
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


def fetch_descriptor(origin: str) -> Descriptor | TransportFailed:
    """GET T3's unauthenticated ``/.well-known/t3/environment``."""
    reply = _request_json(origin, 'GET', _DESCRIPTOR_PATH)
    if isinstance(reply, TransportFailed):
        return reply
    body = reply.body
    if reply.status != 200 or body is None:
        return TransportFailed(TransportFault.PROTOCOL, f'descriptor: HTTP {reply.status} without a JSON object')
    environment_id, server_version, label = body.get('environmentId'), body.get('serverVersion'), body.get('label')
    if not isinstance(environment_id, str) or not _ENVIRONMENT_ID.fullmatch(environment_id):
        return TransportFailed(TransportFault.PROTOCOL, f'descriptor: unusable environmentId {environment_id!r}')
    if not isinstance(server_version, str) or not server_version:
        return TransportFailed(TransportFault.PROTOCOL, f'descriptor: unusable serverVersion {server_version!r}')
    return Descriptor(environment_id, server_version, label if isinstance(label, str) else '')


def credential_status_at(
    origin: str, *, root: Path | str | None = None, now: datetime | None = None
) -> CredentialStatus | TransportFailed:
    """The credential state for whichever environment ``origin`` answers as."""
    descriptor = fetch_descriptor(origin)
    if isinstance(descriptor, TransportFailed):
        return descriptor
    return read_credential_status(descriptor.environment_id, root=root, now=now)


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
    return isinstance(message, dict) and message.get('id') == request_id


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
        return TransportFailed(TransportFault.PROTOCOL, f'request {request_id}: the reply names another request')
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


class T3McpClient:
    """An MCP session with one T3 environment, authenticated by one credential."""

    def __init__(self, origin: str, credential: Credential, *, root: Path | str | None = None) -> None:
        self._origin = checked_origin(origin)
        self._credential = credential
        self._root = root
        self._session: dict[str, str] | None = None
        """The headers that carry the session; None until ``initialize`` succeeds."""
        self._request_ids = itertools.count(1)

    @property
    def environment_id(self) -> str:
        return self._credential.environment_id

    def call(self, tool: str, arguments: Mapping[str, Any]) -> CallResult:
        if self._session is None:
            opened = self._initialize()
            if isinstance(opened, TransportFailed):
                return opened
        reply = self._rpc('tools/call', {'name': tool, 'arguments': dict(arguments)})
        if isinstance(reply, TransportFailed):
            return reply
        return _tool_result(reply.result)

    def _initialize(self) -> None | TransportFailed:
        params = {'protocolVersion': PROTOCOL_VERSION, 'capabilities': {}, 'clientInfo': {'name': SOFTWARE_ID, 'version': '1'}}
        reply = self._rpc('initialize', params)
        if isinstance(reply, TransportFailed):
            return reply
        self._session = {'Mcp-Session-Id': reply.session_id} if reply.session_id else {}
        notified = self._notify_initialized()
        if isinstance(notified, TransportFailed):
            self._session = None
        return notified

    def _notify_initialized(self) -> None | TransportFailed:
        response = self._post({'jsonrpc': '2.0', 'method': 'notifications/initialized'})
        if isinstance(response, TransportFailed):
            return response
        with contextlib.closing(response):
            if response.status not in (200, 202):
                return TransportFailed(TransportFault.PROTOCOL, f'notifications/initialized: HTTP {response.status}')
        return None

    def _rpc(self, method: str, params: Mapping[str, Any]) -> _RpcResult | TransportFailed:
        request_id = next(self._request_ids)
        response = self._post({'jsonrpc': '2.0', 'id': request_id, 'method': method, 'params': params})
        if isinstance(response, TransportFailed):
            return response
        with contextlib.closing(response):
            if response.status != 200:
                return TransportFailed(TransportFault.PROTOCOL, f'{method}: HTTP {response.status}')
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

    def _post(self, message: Mapping[str, Any]) -> _Response | TransportFailed:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
            'Authorization': f'Bearer {self._credential.access_token}',
            'MCP-Protocol-Version': PROTOCOL_VERSION,
            **(self._session or {}),
        }
        body = json.dumps(message).encode()
        return _open(self._origin, 'POST', _MCP_PATH, body=body, headers=headers, timeout=MCP_READ_TIMEOUT_SECS)


# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------


class ConnectFailure(StrEnum):
    DESCRIPTOR_UNREACHABLE = 'descriptor-unreachable'
    UNPAIRED = 'unpaired'
    CREDENTIAL_EXPIRED = 'credential-expired'
    PAIR_FAILED = 'pair-failed'


@dataclass(frozen=True)
class ConnectFailed:
    reason: ConnectFailure
    detail: str | None = None


def connect(origin: str, *, root: Path | str | None = None) -> T3McpClient | ConnectFailed:
    """A client on the stored credential for the environment ``origin`` answers as."""
    status = credential_status_at(origin, root=root)
    if isinstance(status, TransportFailed):
        return ConnectFailed(ConnectFailure.DESCRIPTOR_UNREACHABLE, status.detail)
    if status.state is CredentialState.PAIRED and status.credential is not None:
        return T3McpClient(origin, status.credential, root=root)
    reason = ConnectFailure.UNPAIRED if status.state is CredentialState.UNPAIRED else ConnectFailure.CREDENTIAL_EXPIRED
    return ConnectFailed(reason, f'environment {status.environment_id}')
