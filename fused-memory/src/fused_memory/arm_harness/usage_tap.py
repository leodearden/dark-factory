"""A recording reverse proxy: forwards an arm's OpenAI-compatible traffic unchanged and logs what the arm's own server reported for each call (η context-fit gate)."""

import http.client
import json
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import Message
from enum import StrEnum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit

from pydantic import AwareDatetime, Field

from fused_memory.arm_harness.frozen_model import FrozenModel

UPSTREAM_TIMEOUT_S = 900
ERROR_EXCERPT_CHARS = 500
UPSTREAM_UNREACHABLE_STATUS = 502
STREAM_REFUSED_STATUS = 501

_HOP_BY_HOP = frozenset({
    'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
    'te', 'trailer', 'trailers', 'transfer-encoding', 'upgrade', 'content-length',
})
_NOT_FORWARDED = _HOP_BY_HOP | {'host', 'accept-encoding'}
_NOT_RETURNED = _HOP_BY_HOP | {'date', 'server'}

Headers = tuple[tuple[str, str], ...]


class StatusOrigin(StrEnum):
    UPSTREAM = 'upstream'
    TAP = 'tap'


class CallRecord(FrozenModel):
    started_at: AwareDatetime
    duration_ms: float = Field(ge=0)
    method: str
    path: str
    status: int
    request_model: str | None
    request_max_tokens: int | None
    request_response_format: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    error_excerpt: str | None
    outlived_session: bool = False
    # None marks a record written before the tap recorded who set the status.
    status_origin: StatusOrigin | None = None

    @property
    def succeeded(self) -> bool:
        return is_success_status(self.status)


def _json_object(body: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _int_or_none(value: object) -> int | None:
    return value if type(value) is int else None


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def is_success_status(status: int) -> bool:
    return 200 <= status < 300


def call_record(
    *,
    started_at: datetime,
    duration_ms: float,
    method: str,
    path: str,
    request_body: bytes,
    status: int,
    response_body: bytes,
    status_origin: StatusOrigin,
) -> CallRecord:
    request = _json_object(request_body)
    response = _json_object(response_body) if is_success_status(status) else {}
    usage = _mapping(response.get('usage'))
    choices = response.get('choices')
    first_choice = _mapping(choices[0]) if isinstance(choices, list) and choices else {}
    return CallRecord(
        started_at=started_at,
        duration_ms=duration_ms,
        method=method,
        path=path,
        status=status,
        request_model=_str_or_none(request.get('model')),
        request_max_tokens=_int_or_none(request.get('max_tokens')),
        request_response_format=_str_or_none(_mapping(request.get('response_format')).get('type')),
        prompt_tokens=_int_or_none(usage.get('prompt_tokens')),
        completion_tokens=_int_or_none(usage.get('completion_tokens')),
        finish_reason=_str_or_none(first_choice.get('finish_reason')),
        error_excerpt=None
        if is_success_status(status)
        else response_body.decode('utf-8', 'replace')[:ERROR_EXCERPT_CHARS],
        status_origin=status_origin,
    )


def load_call_records(path: Path | str) -> tuple[CallRecord, ...]:
    lines = Path(path).read_text().splitlines()
    return tuple(CallRecord.model_validate_json(line) for line in lines if line.strip())


def _notice(line: str) -> None:
    print(line, flush=True)


class _SessionCallLog:
    """One tap session's calls; closing it drains the calls still in flight."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._state = threading.Condition()
        self._in_flight = 0
        self._closed = False

    @contextmanager
    def in_flight(self) -> Iterator[None]:
        with self._state:
            self._in_flight += 1
        try:
            yield
        finally:
            with self._state:
                self._in_flight -= 1
                self._state.notify_all()

    def close(self, drain_timeout_s: float) -> None:
        with self._state:
            try:
                self._drain(drain_timeout_s)
            finally:
                self._closed = True

    def _drain(self, drain_timeout_s: float) -> None:
        if not self._in_flight:
            return
        _notice(
            f'usage tap: waiting up to {drain_timeout_s:g} s for {self._in_flight} '
            'in-flight call(s) to land before the session closes'
        )
        if not self._state.wait_for(lambda: self._in_flight == 0, timeout=drain_timeout_s):
            _notice(
                f'usage tap: drain bound of {drain_timeout_s:g} s expired with '
                f'{self._in_flight} call(s) still in flight; they land marked outlived_session'
            )

    def append(self, record: CallRecord) -> None:
        with self._state, self._path.open('a') as stream:
            if self._closed:
                record = record.model_copy(update={'outlived_session': True})
            stream.write(record.model_dump_json() + '\n')
            stream.flush()


@dataclass(frozen=True)
class _Reply:
    status: int
    headers: Headers
    body: bytes
    origin: StatusOrigin


def _tap_error(status: int, message: str) -> _Reply:
    body = json.dumps({'error': {'message': message, 'type': 'usage_tap'}}).encode()
    return _Reply(status, (('Content-Type', 'application/json'),), body, StatusOrigin.TAP)


def _exchange(
    upstream: SplitResult, method: str, path: str, headers: Message, body: bytes
) -> _Reply:
    if _json_object(body).get('stream') is True:
        return _tap_error(
            STREAM_REFUSED_STATUS,
            'the usage tap does not proxy streamed requests: their usage is not measurable here',
        )
    forwarded = {
        name: value for name, value in headers.items() if name.lower() not in _NOT_FORWARDED
    }
    forwarded['Host'] = upstream.netloc
    connection = http.client.HTTPConnection(
        upstream.hostname or '', upstream.port, timeout=UPSTREAM_TIMEOUT_S
    )
    try:
        connection.request(method, path, body=body, headers=forwarded)
        response = connection.getresponse()
        return _Reply(
            response.status, tuple(response.getheaders()), response.read(), StatusOrigin.UPSTREAM
        )
    except (OSError, http.client.HTTPException) as error:
        return _tap_error(
            UPSTREAM_UNREACHABLE_STATUS,
            f'usage tap: upstream {upstream.netloc} failed: {type(error).__name__}: {error}',
        )
    finally:
        connection.close()


def _handler_class(
    upstream: SplitResult, call_log: _SessionCallLog
) -> type[BaseHTTPRequestHandler]:
    class TapHandler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args: Any) -> None:  # noqa: A002 - stdlib signature
            """The call log is the access log."""

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            self._proxy()

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            self._proxy()

        def _proxy(self) -> None:
            with call_log.in_flight():
                started_at = datetime.now(UTC)
                started = time.perf_counter()
                body = self.rfile.read(int(self.headers.get('Content-Length') or 0))
                reply = _exchange(upstream, self.command, self.path, self.headers, body)
                call_log.append(call_record(
                    started_at=started_at,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    method=self.command,
                    path=self.path,
                    request_body=body,
                    status=reply.status,
                    response_body=reply.body,
                    status_origin=reply.origin,
                ))
            self.send_response(reply.status)
            for name, value in reply.headers:
                if name.lower() not in _NOT_RETURNED:
                    self.send_header(name, value)
            self.send_header('Content-Length', str(len(reply.body)))
            self.end_headers()
            self.wfile.write(reply.body)

    return TapHandler


def _upstream_origin(upstream_url: str) -> SplitResult:
    upstream = urlsplit(upstream_url)
    if upstream.scheme != 'http' or not upstream.hostname or upstream.path not in ('', '/'):
        raise ValueError(
            f'usage tap upstream {upstream_url!r} must be a bare http origin: '
            'the tap forwards each request path unchanged'
        )
    return upstream


@contextmanager
def usage_tap(
    upstream_url: str,
    *,
    log_path: Path | str,
    host: str = '127.0.0.1',
    port: int = 0,
    drain_timeout_s: float = UPSTREAM_TIMEOUT_S,
) -> Iterator[str]:
    call_log_path = Path(log_path)
    call_log_path.touch()
    call_log = _SessionCallLog(call_log_path)
    handler = _handler_class(_upstream_origin(upstream_url), call_log)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True, name='usage-tap')
    thread.start()
    try:
        yield f'http://{host}:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()
        call_log.close(drain_timeout_s)
        thread.join(timeout=5)
