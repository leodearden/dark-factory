"""A fake T3 Code server and a fake ``t3`` CLI for the /spawn T3 lane's MCP client.

WHAT IT FAKES.  :class:`FakeT3Server` is one in-process HTTP server answering every
surface ``orchestrator/src/orchestrator/t3_mcp_client.py`` talks to: the
unauthenticated environment descriptor, the pairing-code OAuth exchange
(register, decision, token, with PKCE S256 checked for real), and the
streamable-HTTP ``/mcp`` endpoint, framed as a JSON body or as SSE frames.
:func:`install_fake_t3_cli` writes an executable ``t3`` that answers
``--version`` and mints pairing codes on that server, the way the real CLI
writes the real server's state; :func:`put_fake_t3_cli_on_path` installs it
first on a test's PATH. :meth:`FakeT3Server.write_paired_credential` stands in for
a completed pairing, storing this host's credential for the server.
:data:`DEAD_ORIGIN` is an origin nothing answers.

WHERE THE SHAPES COME FROM.  Every canned payload and refusal is copied from
T3 Code v0.1.6 (``~/src/t3code-v0.1.6``):
``packages/contracts/src/auth.ts`` (``AuthMcpClientRegistration``,
``AuthMcpRegisteredClient``, ``AuthMcpRegistrationError``,
``AuthMcpApprovalDecisionRequest``, ``AuthMcpApprovalRedirect``,
``AuthMcpApprovalError``, ``AuthMcpTokenRequest``, ``AuthMcpTokenResult``,
``AuthMcpTokenError``), ``packages/contracts/src/environmentHttp.ts`` (the
``mcpOAuth`` group and the ``descriptor`` endpoint),
``packages/contracts/src/environment.ts::ExecutionEnvironmentDescriptor``,
``apps/server/src/mcp/McpHttpServer.ts`` (the 401 body),
``apps/server/src/auth/PairingGrantStore.ts`` (``PAIRING_TOKEN_ALPHABET``,
``PAIRING_TOKEN_LENGTH``: the pairing codes),
``apps/server/src/cliAuthFormat.ts::formatIssuedPairingCredential`` (the CLI's
``--json`` output) and effect's ``McpServer`` ``CallToolResult``.

WHY ``cross_suite_helpers/``.  So the root pyright table can name this helper
without exposing ``orchestrator/tests``; see the root ``pyproject.toml``
``extraPaths``. The ``_`` prefix is the convention
``orchestrator/tests/_mcp_transport_harness.py`` records. No pytest fixtures
live here, so no consumer imports one it never names: each test module wraps
:meth:`FakeT3Server.running` in its own short fixture. A test plays T3's state per call with a :data:`ToolResponder` in
``tool_replies``, such as :func:`replies_in_turn`; :data:`HANG_UP` drops one call.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import secrets
import sys
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.message import Message
from enum import Enum, StrEnum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qs, urlencode, urlsplit

from orchestrator import t3_mcp_client

if TYPE_CHECKING:
    import pytest

ENVIRONMENT_ID = '3f9a1c2e-5b7d-4e8f-9a0b-1c2d3e4f5a6b'
SERVER_VERSION = '0.1.6'
PROTOCOL_VERSION = '2025-06-18'
DEFAULT_EXPIRES_IN = 2_592_000
FULL_ACCESS_SCOPES = ('orchestration:read', 'orchestration:operate', 'settings:write', 'source-control:write')
DEAD_ORIGIN = 'http://127.0.0.1:1'
"""Nothing answers here: binding port 1 needs root, so unlike a freed ephemeral
port no concurrently running fake can be handed it."""
PAIRING_CODE_ALPHABET = '23456789ABCDEFGHJKLMNPQRSTUVWXYZ'
PAIRING_CODE_LENGTH = 12
"""T3's ``PAIRING_TOKEN_ALPHABET`` and ``PAIRING_TOKEN_LENGTH``: no minted code
begins with '-', so each one passes on argv as ``--code <code>`` the way an operator's does."""

DESCRIPTOR_PATH = '/.well-known/t3/environment'
REGISTER_PATH = '/oauth/mcp/register'
DECISION_PATH = '/oauth/mcp/decision'
TOKEN_PATH = '/oauth/mcp/token'
MCP_PATH = '/mcp'
FAKE_PAIRING_CODE_PATH = '/__fake/pairing-code'
"""Test-only: mints a pairing code. Only the fake CLI calls it."""

PAIRING_CREATE_MODE_ENV = 'FAKE_T3_PAIRING_CREATE'
"""Read by the fake CLI's ``auth pairing create`` branch; values are :class:`PairingCreateMode`."""


class PairingCreateMode(StrEnum):
    OK = 'ok'
    FAILS = 'fails'
    OMITS_CREDENTIAL = 'omits-credential'


@dataclass(frozen=True)
class ToolSuccess:
    value: Mapping[str, Any]


@dataclass(frozen=True)
class ToolFailure:
    code: str
    message: str


ToolReply = ToolSuccess | ToolFailure


class _HangUp(Enum):
    HANG_UP = 'hang-up'


HANG_UP = _HangUp.HANG_UP
"""A responder's answer that closes this one call's connection without a reply."""

ToolAnswer = ToolReply | Literal[_HangUp.HANG_UP]
ToolResponder = Callable[[Mapping[str, Any]], ToolAnswer]
"""Called with each call's ``arguments`` (an empty dict when absent); the request is recorded first."""


def replies_in_turn(*replies: ToolAnswer | ToolResponder) -> ToolResponder:
    """Answer each call with the next of ``replies``, repeating the last; a responder among them is called."""
    if not replies:
        raise ValueError('replies_in_turn needs at least one reply')
    pending, lock = list(replies), threading.Lock()

    def respond(arguments: Mapping[str, Any]) -> ToolAnswer:
        with lock:
            reply = pending.pop(0) if len(pending) > 1 else pending[0]
        return reply(arguments) if callable(reply) else reply

    return respond


@dataclass(frozen=True)
class RecordedRequest:
    method: str
    path: str
    headers: Mapping[str, str]
    """Lower-cased header names."""
    body: Any
    """Decoded JSON, a flat dict for a form body, or None when empty."""


@dataclass(frozen=True)
class _Reply:
    status: int
    body: bytes = b''
    content_type: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _AuthCode:
    client_id: str
    redirect_uri: str
    code_challenge: str


_UNRELATED_FRAME = {
    'jsonrpc': '2.0',
    'method': 'notifications/message',
    'params': {'level': 'info', 'data': 'fake-t3: a frame that is not the reply'},
}


def _json_reply(status: int, value: Any, headers: Mapping[str, str] | None = None) -> _Reply:
    return _Reply(status, json.dumps(value).encode(), 'application/json', headers or {})


def _approval_error(message: str) -> _Reply:
    return _json_reply(400, {'_tag': 'AuthMcpApprovalError', 'message': message})


def _invalid_grant(description: str) -> _Reply:
    return _json_reply(400, {'error': 'invalid_grant', 'error_description': description})


def _s256(verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')


def _bearer(authorization: str | None) -> str | None:
    scheme, _, token = (authorization or '').partition(' ')
    return token if scheme == 'Bearer' and token else None


def _parse_body(content_type: str, raw: bytes) -> Any:
    if not raw:
        return None
    text = raw.decode('utf-8', 'replace')
    if content_type.startswith('application/x-www-form-urlencoded'):
        return {key: values[0] for key, values in parse_qs(text, keep_blank_values=True).items()}
    try:
        return json.loads(text)
    except ValueError:
        return text


def _environment_read(environment_id: str, server_version: str) -> ToolSuccess:
    return ToolSuccess(
        {
            'environmentId': environment_id,
            'label': 'fake-t3',
            'serverVersion': server_version,
            'platform': {'os': 'linux', 'arch': 'x64'},
            'preferences': {},
        }
    )


def _call_tool_result(reply: ToolReply) -> dict[str, Any]:
    if isinstance(reply, ToolSuccess):
        value = dict(reply.value)
        return {'content': [{'type': 'text', 'text': json.dumps(value)}], 'structuredContent': value, 'isError': False}
    failure = {'_tag': 'OrchestratorMcpFailure', 'code': reply.code, 'message': reply.message}
    return {'content': [{'type': 'text', 'text': json.dumps(failure)}], 'isError': True}


class FakeT3Server(ThreadingHTTPServer):
    """T3's HTTP surface on 127.0.0.1:<ephemeral>; enter it with :meth:`running`."""

    daemon_threads = True

    def __init__(
        self,
        *,
        framing: str = 'json',
        expires_in: int = DEFAULT_EXPIRES_IN,
        environment_id: str = ENVIRONMENT_ID,
        server_version: str = SERVER_VERSION,
        serves_descriptor: bool = True,
    ) -> None:
        super().__init__(('127.0.0.1', 0), _Handler)
        self.origin = f'http://127.0.0.1:{self.server_port}'
        self.serves_descriptor = serves_descriptor
        """False answers the descriptor path 404, as a pre-0.1.6 T3 or a non-T3 service on the port would."""
        self.environment_id = environment_id
        self.server_version = server_version
        self.framing = framing
        self.expires_in = expires_in
        self.token_reply_overrides: dict[str, Any] = {}
        """Merged over the well-formed token reply, to answer a malformed one."""
        self.decision_state_override: str | None = None
        """When set, the decision's redirect carries this ``state`` instead of echoing the client's."""
        self.pings_with_the_request_id = False
        """True puts a server-to-client ``ping`` bearing the client's own request id before each SSE reply."""
        self.hangs_up = False
        """True closes every connection without a reply, as a server dying mid-request does."""
        self.tool_replies: dict[str, ToolReply | ToolResponder] = {
            't3_environment_read': _environment_read(environment_id, server_version)
        }
        self.requests: list[RecordedRequest] = []
        self._lock = threading.Lock()
        self._pairing_codes: set[str] = set()
        self._issued_pairing_codes: list[str] = []
        self._clients: dict[str, list[str]] = {}
        self._auth_codes: dict[str, _AuthCode] = {}
        self._tokens: set[str] = set()
        self._issued_tokens: list[str] = []
        self._issued_client_ids: list[str] = []
        self._sessions: set[str] = set()
        self._issued_sessions: list[str] = []

    @classmethod
    @contextlib.contextmanager
    def running(cls, **options: Any) -> Iterator[FakeT3Server]:
        server = cls(**options)
        thread = threading.Thread(target=server.serve_forever, name='fake-t3-server', daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def answer_as(self, environment_id: str) -> None:
        """Answer the descriptor and ``t3_environment_read`` as ``environment_id``, as a T3 reinstalled on this port would."""
        self.environment_id = environment_id
        self.tool_replies['t3_environment_read'] = _environment_read(environment_id, self.server_version)

    def add_pairing_code(self) -> str:
        code = ''.join(secrets.choice(PAIRING_CODE_ALPHABET) for _ in range(PAIRING_CODE_LENGTH))
        with self._lock:
            self._pairing_codes.add(code)
            self._issued_pairing_codes.append(code)
        return code

    def issued_pairing_codes(self) -> list[str]:
        """Every pairing code minted, by the fake CLI or :meth:`add_pairing_code`, oldest first."""
        with self._lock:
            return list(self._issued_pairing_codes)

    def issue_token(self) -> str:
        token = secrets.token_urlsafe(24)
        with self._lock:
            self._tokens.add(token)
            self._issued_tokens.append(token)
        return token

    def write_paired_credential(self, root: Path | str, *, access_token: str | None = None) -> Path:
        """This host's credential for the server under fleet ``root``, on a bearer it issued unless one is given."""
        paired_at = datetime.now(UTC).replace(microsecond=0)
        credential = t3_mcp_client.Credential(
            schema_version=t3_mcp_client.CREDENTIAL_SCHEMA_VERSION,
            origin=self.origin,
            environment_id=self.environment_id,
            server_version=self.server_version,
            access=t3_mcp_client.ACCESS,
            client_id='mcp-client-1',
            access_token=access_token if access_token is not None else self.issue_token(),
            expires_at=paired_at + timedelta(days=30),
            paired_at=paired_at,
            paired_by=t3_mcp_client.PairedBy.SELF_PAIR,
        )
        t3_mcp_client.write_credential(credential, root=root)
        return t3_mcp_client.credential_path(self.environment_id, root=root)

    def revoke_tokens(self) -> None:
        with self._lock:
            self._tokens.clear()

    def forget_sessions(self) -> None:
        with self._lock:
            self._sessions.clear()

    def issued_tokens(self) -> list[str]:
        """Every bearer issued, by the token endpoint or :meth:`issue_token`, oldest first."""
        with self._lock:
            return list(self._issued_tokens)

    def issued_client_ids(self) -> list[str]:
        """Every client_id the register endpoint answered with, oldest first."""
        with self._lock:
            return list(self._issued_client_ids)

    def issued_sessions(self) -> list[str]:
        """Every Mcp-Session-Id an ``initialize`` was answered with, oldest first."""
        with self._lock:
            return list(self._issued_sessions)

    def mcp_requests(self) -> list[RecordedRequest]:
        return [request for request in self.requests if request.path == MCP_PATH]

    def tool_calls(self, name: str) -> list[Any]:
        return [
            request.body['params'].get('arguments')
            for request in self.mcp_requests()
            if isinstance(request.body, dict)
            and request.body.get('method') == 'tools/call'
            and request.body.get('params', {}).get('name') == name
        ]

    def oauth_calls(self, path: str) -> list[Any]:
        return [request.body for request in self.requests if request.path == path]

    def record(self, request: RecordedRequest) -> None:
        with self._lock:
            self.requests.append(request)

    def answer_get(self, path: str) -> _Reply:
        if path != DESCRIPTOR_PATH or not self.serves_descriptor:
            return _json_reply(404, {'error': 'not_found'})
        return _json_reply(
            200,
            {
                'environmentId': self.environment_id,
                'label': 'fake-t3',
                'platform': {'os': 'linux', 'arch': 'x64'},
                'serverVersion': self.server_version,
                'capabilities': {},
            },
        )

    def answer_post(self, path: str, headers: Message, body: Any) -> _Reply | None:
        """The reply to send, or None to close the connection without one."""
        if path == MCP_PATH:
            return self._mcp(headers, body)
        if not isinstance(body, dict):
            return _json_reply(400, {'error': 'invalid_request', 'error_description': 'Expected an object body.'})
        if path == REGISTER_PATH:
            return self._register(body)
        if path == DECISION_PATH:
            return self._decision(body)
        if path == TOKEN_PATH:
            return self._token(body)
        if path == FAKE_PAIRING_CODE_PATH:
            return _json_reply(200, {'code': self.add_pairing_code()})
        return _json_reply(404, {'error': 'not_found'})

    def _register(self, body: dict[str, Any]) -> _Reply:
        redirect_uris = body.get('redirect_uris')
        if not isinstance(redirect_uris, list) or not redirect_uris or not all(isinstance(u, str) for u in redirect_uris):
            return _json_reply(
                400, {'error': 'invalid_redirect_uri', 'error_description': 'Register between 1 and 5 redirect URIs.'}
            )
        client_id = f'mcp-client-{secrets.token_hex(8)}'
        with self._lock:
            self._clients[client_id] = list(redirect_uris)
            self._issued_client_ids.append(client_id)
        return _json_reply(
            201,
            {
                'client_id': client_id,
                'client_name': body.get('client_name', 'MCP client'),
                'redirect_uris': redirect_uris,
                'grant_types': ['authorization_code'],
                'response_types': ['code'],
                'token_endpoint_auth_method': 'none',
            },
        )

    def _decision(self, body: dict[str, Any]) -> _Reply:
        authorization, decision = body.get('authorization'), body.get('decision')
        if not isinstance(authorization, dict) or not isinstance(decision, dict):
            return _approval_error('The sign-in request is malformed.')
        if decision.get('_tag') != 'pairing-code':
            return _approval_error('Enter a pairing code.')
        if authorization.get('code_challenge_method') != 'S256' or not authorization.get('code_challenge'):
            return _approval_error('The sign-in request must use PKCE S256.')
        if authorization.get('resource') != f'{self.origin}{MCP_PATH}':
            return _approval_error('The sign-in request names another resource.')
        client_id, redirect_uri = authorization.get('client_id'), authorization.get('redirect_uri')
        with self._lock:
            if not isinstance(client_id, str) or redirect_uri not in self._clients.get(client_id, []):
                return _approval_error('The client or its redirect URI is not registered.')
            if decision.get('code') not in self._pairing_codes:
                return _approval_error('The pairing code is invalid or expired.')
            self._pairing_codes.discard(decision['code'])
            auth_code = secrets.token_urlsafe(16)
            self._auth_codes[auth_code] = _AuthCode(client_id, redirect_uri, authorization['code_challenge'])
        state = self.decision_state_override if self.decision_state_override is not None else authorization.get('state', '')
        query = urlencode({'code': auth_code, 'state': state, 'iss': self.origin})
        return _json_reply(200, {'redirectTo': f'{redirect_uri}?{query}'})

    def _token(self, form: dict[str, Any]) -> _Reply:
        with self._lock:
            grant = self._auth_codes.pop(form.get('code', ''), None)
        if (
            form.get('grant_type') != 'authorization_code'
            or grant is None
            or form.get('client_id') != grant.client_id
            or form.get('redirect_uri') != grant.redirect_uri
            or _s256(form.get('code_verifier', '')) != grant.code_challenge
        ):
            return _invalid_grant('The authorization code is invalid or expired.')
        issued = {
            'access_token': self.issue_token(),
            'token_type': 'Bearer',
            'expires_in': self.expires_in,
            'scope': ' '.join(FULL_ACCESS_SCOPES),
        }
        return _json_reply(200, issued | self.token_reply_overrides)

    def _mcp(self, headers: Message, body: Any) -> _Reply | None:
        with self._lock:
            authorized = _bearer(headers.get('Authorization')) in self._tokens
        if not authorized:
            return _json_reply(
                401, {'error': 'invalid_mcp_credential', 'message': 'A valid T3 Code MCP credential is required.'}
            )
        accept = headers.get('Accept', '')
        if 'application/json' not in accept or 'text/event-stream' not in accept:
            return _json_reply(406, self._rpc_error(None, -32000, 'Client must accept both JSON and SSE.'))
        if not isinstance(body, dict) or not isinstance(body.get('method'), str):
            return _json_reply(400, self._rpc_error(None, -32700, 'Parse error'))
        if body['method'] == 'initialize':
            session_id = str(uuid.uuid4())
            with self._lock:
                self._sessions.add(session_id)
                self._issued_sessions.append(session_id)
            result = {
                'protocolVersion': PROTOCOL_VERSION,
                'capabilities': {'tools': {}},
                'serverInfo': {'name': 't3code', 'version': self.server_version},
            }
            return self._framed({'jsonrpc': '2.0', 'id': body.get('id'), 'result': result}, {'Mcp-Session-Id': session_id})
        if headers.get('MCP-Protocol-Version') != PROTOCOL_VERSION:
            return _json_reply(400, self._rpc_error(None, -32000, 'Unsupported or missing MCP-Protocol-Version.'))
        session_id = headers.get('Mcp-Session-Id')
        if session_id is None:
            return _json_reply(400, self._rpc_error(None, -32000, 'Mcp-Session-Id is required.'))
        with self._lock:
            known = session_id in self._sessions
        if not known:
            return _json_reply(404, self._rpc_error(None, -32001, 'Session not found'))
        if 'id' not in body:
            return _Reply(202)
        if body['method'] != 'tools/call':
            return self._framed(self._rpc_error(body['id'], -32601, f'Method not found: {body["method"]}'))
        params = body.get('params') or {}
        name = params.get('name')
        entry = self.tool_replies.get(name) if isinstance(name, str) else None
        if entry is None:
            return self._framed(self._rpc_error(body['id'], -32602, f'Tool {name} not found'))
        reply = entry(params.get('arguments') or {}) if callable(entry) else entry
        if reply is HANG_UP:
            return None
        return self._framed({'jsonrpc': '2.0', 'id': body['id'], 'result': _call_tool_result(reply)})

    @staticmethod
    def _rpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {'jsonrpc': '2.0', 'id': request_id, 'error': {'code': code, 'message': message}}

    def _framed(self, message: dict[str, Any], headers: Mapping[str, str] | None = None) -> _Reply:
        if self.framing != 'sse':
            return _json_reply(200, message, headers)
        prelude = [_UNRELATED_FRAME]
        if self.pings_with_the_request_id:
            prelude.append({'jsonrpc': '2.0', 'id': message.get('id'), 'method': 'ping'})
        frames = ''.join(f'event: message\ndata: {json.dumps(frame)}\n\n' for frame in (*prelude, message))
        return _Reply(200, frames.encode(), 'text/event-stream', headers or {})


class _Handler(BaseHTTPRequestHandler):
    server: FakeT3Server

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        self.server.record(RecordedRequest('GET', path, self._header_map(), None))
        if self.server.hangs_up:
            return
        self._send(self.server.answer_get(path))

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        raw = self.rfile.read(int(self.headers.get('Content-Length') or 0))
        body = _parse_body(self.headers.get('Content-Type', ''), raw)
        self.server.record(RecordedRequest('POST', path, self._header_map(), body))
        if self.server.hangs_up:
            return
        reply = self.server.answer_post(path, self.headers, body)
        if reply is not None:
            self._send(reply)

    def _header_map(self) -> dict[str, str]:
        return {name.lower(): value for name, value in self.headers.items()}

    def _send(self, reply: _Reply) -> None:
        self.send_response(reply.status)
        if reply.content_type is not None:
            self.send_header('Content-Type', reply.content_type)
        self.send_header('Content-Length', str(len(reply.body)))
        for name, value in reply.headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(reply.body)


_FAKE_T3_CLI_BODY = '''
import http.client
import json
import os
import sys
from datetime import datetime, timedelta, timezone

args = sys.argv[1:]
with open(LOG_PATH, "a") as log:
    log.write(json.dumps(args) + "\\n")
if args == ["--version"]:
    print(VERSION_LINE)
    sys.exit(0)
if args == ["auth", "pairing", "create", "--json"]:
    mode = os.environ.get(MODE_ENV, "ok")
    if mode == "fails":
        print("Could not issue a pairing credential.", file=sys.stderr)
        sys.exit(1)
    connection = http.client.HTTPConnection(ORIGIN.split("://", 1)[1], timeout=10)
    connection.request("POST", PAIRING_CODE_PATH, body=b"{}", headers={"Content-Type": "application/json"})
    code = json.loads(connection.getresponse().read())["code"]
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    issued = {"id": "pairing-1", "credential": code, "scopes": SCOPES, "expiresAt": expires_at.isoformat()}
    if mode == "omits-credential":
        del issued["credential"]
    print(json.dumps(issued, indent=2))
    sys.exit(0)
print(f"fake t3: unexpected arguments {args}", file=sys.stderr)
sys.exit(2)
'''


def install_fake_t3_cli(bin_dir: Path, *, origin: str, version_line: str, log_path: Path) -> Path:
    """Write ``bin_dir/t3``, logging each argv as one JSON line to ``log_path``.

    ``--version`` prints ``version_line`` (the real CLI prints ``t3 v0.1.6``).
    ``auth pairing create --json`` mints a code on the fake server at ``origin``
    and prints it as ``formatIssuedPairingCredential`` does, unless
    ``$FAKE_T3_PAIRING_CREATE`` names another :class:`PairingCreateMode`.
    Any other argv exits 2.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    header = (
        f'#!{sys.executable}\n'
        f'ORIGIN = {origin!r}\n'
        f'VERSION_LINE = {version_line!r}\n'
        f'LOG_PATH = {str(log_path)!r}\n'
        f'MODE_ENV = {PAIRING_CREATE_MODE_ENV!r}\n'
        f'PAIRING_CODE_PATH = {FAKE_PAIRING_CODE_PATH!r}\n'
        f'SCOPES = {list(FULL_ACCESS_SCOPES)!r}\n'
    )
    executable = bin_dir / 't3'
    executable.write_text(header + _FAKE_T3_CLI_BODY)
    executable.chmod(0o755)
    return executable


def put_fake_t3_cli_on_path(
    server: FakeT3Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, version_line: str
) -> Path:
    """Install the fake ``t3`` minting on ``server`` first on PATH; returns its argv log."""
    bin_dir, log_path = tmp_path / 'fake-t3-mcp-bin', tmp_path / 'fake-t3-mcp-cli.log'
    install_fake_t3_cli(bin_dir, origin=server.origin, version_line=version_line, log_path=log_path)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    return log_path


def cli_calls(log_path: Path) -> list[list[str]]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines()]
