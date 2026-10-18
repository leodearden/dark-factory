"""A real-socket stand-in for the fleet's two MCP servers, for tests.

STATEFUL mode imitates the escalation server: ``initialize`` mints an
``mcp-session-id``, any other request without one is refused 400 'Missing
session ID', an unknown id is 404, answers are SSE-framed, and DELETE releases
the session. STATELESS mode imitates fused-memory: no session, JSON answers.
Both redirect ``/mcp/`` to ``/mcp`` with a 307 and refuse a POST whose Accept
lacks the needed media types with a 406. ``tools/call`` can be scripted to fail
in each way a client must not mistake for success (:class:`StubAnswer`), and
its first *unavailable_tool_calls* requests can be answered 503, as a gateway
answers while the server behind it restarts.

Real sockets rather than patching a client's modules: the session header, path
and Accept a client sends are wire facts the server observes in
:attr:`StubMcpServer.requests`, so no client internals are pinned. Pure stdlib,
registered in ``shared/tests/test_pure_stdlib_leaves.py``.
"""

from __future__ import annotations

import enum
import functools
import json
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import MappingProxyType, TracebackType
from typing import Any

__all__ = ['RecordedRequest', 'StubAnswer', 'StubMcpServer']

_ENDPOINT = '/mcp'
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_DEFAULT_PAYLOAD: Mapping[str, Any] = MappingProxyType({'success': True})
#: ``shutdown()`` waits up to one poll; the stdlib default 0.5s is paid by every test.
_SHUTDOWN_POLL_SECS = 0.01


class StubAnswer(enum.StrEnum):
    """How the stub answers ``tools/call``; the handshake is never affected."""

    OK = 'OK'
    IS_ERROR = 'IS_ERROR'
    NON_JSON_BODY = 'NON_JSON_BODY'
    NON_JSON_TEXT = 'NON_JSON_TEXT'
    EMPTY_202 = 'EMPTY_202'
    REDIRECT_307 = 'REDIRECT_307'
    JSON_RPC_ERROR = 'JSON_RPC_ERROR'


@dataclass(frozen=True)
class RecordedRequest:
    http_method: str
    path: str
    headers: Mapping[str, str]
    body: Any


@dataclass(frozen=True)
class _Reply:
    status: int
    body: bytes = b''
    content_type: str | None = None
    location: str | None = None
    session_id: str | None = None


def _rpc_error(status: int, message: str, code: int = _INVALID_REQUEST) -> _Reply:
    body = {'jsonrpc': '2.0', 'id': 'server-error', 'error': {'code': code, 'message': message}}
    return _Reply(status, json.dumps(body).encode(), 'application/json')


def _accepted_media(accept: str) -> tuple[bool, bool]:
    types = [part.split(';')[0].strip().lower() for part in accept.split(',')]
    return 'application/json' in types, 'text/event-stream' in types


class StubMcpServer:
    """An MCP Streamable-HTTP server on ``127.0.0.1``, used as a context manager."""

    def __init__(
        self,
        *,
        stateful: bool,
        answer: StubAnswer = StubAnswer.OK,
        payload: Mapping[str, Any] = _DEFAULT_PAYLOAD,
        unavailable_tool_calls: int = 0,
    ) -> None:
        self._stateful = stateful
        self._answer = answer
        self._unavailable_left = unavailable_tool_calls
        self._payload_text = json.dumps(dict(payload))
        self._lock = threading.Lock()
        self._requests: list[RecordedRequest] = []
        self._minted: list[str] = []
        self._open: set[str] = set()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._url = ''

    def __enter__(self) -> StubMcpServer:
        handler = functools.partial(_Handler, respond=self._respond)
        self._httpd = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        self._url = f'http://127.0.0.1:{self._httpd.server_address[1]}'
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={'poll_interval': _SHUTDOWN_POLL_SECS}, daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Stop serving; idempotent, and :attr:`url` keeps naming the closed port."""
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join()
        self._httpd = None

    @property
    def url(self) -> str:
        return self._url

    @property
    def requests(self) -> tuple[RecordedRequest, ...]:
        with self._lock:
            return tuple(self._requests)

    @property
    def minted_session_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._minted)

    @property
    def open_session_ids(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._open)

    # --- request handling (runs on the server's handler threads) -------------

    def _respond(self, request: RecordedRequest) -> _Reply:
        with self._lock:
            self._requests.append(request)
        if request.path == f'{_ENDPOINT}/':
            return _Reply(307, location=_ENDPOINT)
        if request.path != _ENDPOINT:
            return _Reply(404, b'Not Found', 'text/plain')
        if request.http_method == 'DELETE':
            return self._release(request.headers.get('mcp-session-id'))
        refusal = self._refuse_media(request.headers.get('accept', ''))
        if refusal is not None:
            return refusal
        message = request.body
        if not isinstance(message, dict) or not isinstance(message.get('method'), str):
            return _rpc_error(400, 'Parse error')
        if self._gateway_unavailable(message['method']):
            return _Reply(503, b'Service Unavailable', 'text/plain')
        session_id, refusal = self._session_for(message['method'], request.headers)
        if refusal is not None:
            return refusal
        return self._answer_message(message, session_id)

    def _refuse_media(self, accept: str) -> _Reply | None:
        has_json, has_sse = _accepted_media(accept)
        if self._stateful and not (has_json and has_sse):
            return _rpc_error(
                406, 'Not Acceptable: Client must accept both application/json and text/event-stream',
            )
        if not has_json:
            return _rpc_error(406, 'Not Acceptable: Client must accept application/json')
        return None

    def _gateway_unavailable(self, method: str) -> bool:
        if method != 'tools/call':
            return False
        with self._lock:
            if self._unavailable_left == 0:
                return False
            self._unavailable_left -= 1
            return True

    def _session_for(
        self, method: str, headers: Mapping[str, str],
    ) -> tuple[str | None, _Reply | None]:
        if not self._stateful:
            return None, None
        session_id = headers.get('mcp-session-id')
        if session_id is None:
            if method != 'initialize':
                return None, _rpc_error(400, 'Bad Request: Missing session ID')
            return self._mint(), None
        with self._lock:
            known = session_id in self._open
        if not known:
            return None, _rpc_error(404, 'Session not found')
        return session_id, None

    def _mint(self) -> str:
        session_id = uuid.uuid4().hex
        with self._lock:
            self._minted.append(session_id)
            self._open.add(session_id)
        return session_id

    def _release(self, session_id: str | None) -> _Reply:
        if not self._stateful:
            return _rpc_error(405, 'Method Not Allowed: Session termination not supported')
        if session_id is None:
            return _rpc_error(400, 'Bad Request: Missing session ID')
        with self._lock:
            known = session_id in self._open
            self._open.discard(session_id)
        if not known:
            return _rpc_error(404, 'Session not found')
        return _Reply(200, content_type='application/json')

    def _answer_message(self, message: dict, session_id: str | None) -> _Reply:
        if 'id' not in message:
            return _Reply(202, session_id=session_id)
        method = message['method']
        if method == 'tools/call' and self._answer is not StubAnswer.OK:
            return self._scripted(message, session_id)
        if method == 'initialize':
            return self._frame(message, self._initialize_result(message), session_id)
        if method == 'tools/call':
            return self._frame(message, self._tool_result(), session_id)
        if method == 'tools/list':
            return self._frame(message, {'tools': []}, session_id)
        error = {'code': _METHOD_NOT_FOUND, 'message': 'Method not found'}
        return self._frame(message, None, session_id, error=error)

    def _scripted(self, message: dict, session_id: str | None) -> _Reply:
        match self._answer:
            case StubAnswer.IS_ERROR:
                name = message.get('params', {}).get('name')
                text = f'Error calling tool {name!r}: scripted stub failure'
                result = {'content': [{'type': 'text', 'text': text}], 'isError': True}
                return self._frame(message, result, session_id)
            case StubAnswer.NON_JSON_TEXT:
                result = {'content': [{'type': 'text', 'text': 'done, but not JSON'}], 'isError': False}
                return self._frame(message, result, session_id)
            case StubAnswer.NON_JSON_BODY:
                return _Reply(200, b'Internal Server Error', 'text/plain', session_id=session_id)
            case StubAnswer.EMPTY_202:
                return _Reply(202, session_id=session_id)
            case StubAnswer.REDIRECT_307:
                return _Reply(307, location=_ENDPOINT, session_id=session_id)
            case StubAnswer.JSON_RPC_ERROR:
                error = {'code': _INVALID_PARAMS, 'message': 'Invalid params: scripted stub failure'}
                return self._frame(message, None, session_id, error=error)
        raise AssertionError(f'{self._answer} is not a scripted failure')

    def _initialize_result(self, message: dict) -> dict:
        requested = message.get('params', {}).get('protocolVersion', '2025-03-26')
        return {
            'protocolVersion': requested,
            'capabilities': {},
            'serverInfo': {'name': 'shared-testing-mcp-stub', 'version': '0'},
        }

    def _tool_result(self) -> dict:
        return {
            'content': [{'type': 'text', 'text': self._payload_text}],
            'structuredContent': json.loads(self._payload_text),
            'isError': False,
        }

    def _frame(
        self,
        message: dict,
        result: dict | None,
        session_id: str | None,
        *,
        error: dict | None = None,
    ) -> _Reply:
        rpc: dict[str, Any] = {'jsonrpc': '2.0', 'id': message['id']}
        if error is not None:
            rpc['error'] = error
        else:
            rpc['result'] = result
        text = json.dumps(rpc)
        if self._stateful:
            frame = f'event: message\ndata: {text}\n\n'
            return _Reply(200, frame.encode(), 'text/event-stream', session_id=session_id)
        return _Reply(200, text.encode(), 'application/json', session_id=session_id)


class _Handler(BaseHTTPRequestHandler):
    def __init__(
        self, *args: Any, respond: Callable[[RecordedRequest], _Reply], **kwargs: Any,
    ) -> None:
        self._respond = respond
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — stdlib signature
        return

    def do_POST(self) -> None:
        self._handle('POST')

    def do_DELETE(self) -> None:
        self._handle('DELETE')

    def _handle(self, http_method: str) -> None:
        headers = MappingProxyType({name.lower(): value for name, value in self.headers.items()})
        request = RecordedRequest(http_method, self.path, headers, self._read_json_body())
        self._send(self._respond(request))

    def _read_json_body(self) -> Any:
        raw = self.rfile.read(int(self.headers.get('content-length') or 0))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    def _send(self, reply: _Reply) -> None:
        self.send_response(reply.status)
        if reply.content_type is not None:
            self.send_header('content-type', reply.content_type)
        if reply.location is not None:
            self.send_header('location', reply.location)
        if reply.session_id is not None:
            self.send_header('mcp-session-id', reply.session_id)
        self.send_header('content-length', str(len(reply.body)))
        self.end_headers()
        self.wfile.write(reply.body)
