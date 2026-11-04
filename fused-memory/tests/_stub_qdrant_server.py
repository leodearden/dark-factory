"""A dependency-free stub of the Qdrant REST surface that mem0 and Mem0Backend touch.

Why a REAL socket: the defect this exists to reproduce lives between real mem0,
real qdrant-client and httpx's library-default read timeout, all of which sit
above the transport.  Only a server that is genuinely slow to answer exercises
that seam.  Built on stdlib ``http.server`` for the reason
``_mock_openai_server.py`` gives: no HTTP-mock library is installed.

Collection meta-ops (create) run one after another under a single lock, as a
real Qdrant runs them.  A create's latency is set per collection, and it is
waited out on :attr:`StubQdrantServer.release_event` rather than slept, so a
test can end the wait early and no handler thread outlives the server.

Binds 127.0.0.1 on port 0, so ``-n auto`` xdist workers cannot collide.
"""

from __future__ import annotations

import json
import sys
import threading
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlsplit

#: The server version reported on ``GET /``, which qdrant-client reads for its
#: client/server compatibility check.
STUB_SERVER_VERSION = '1.17.1'


def _ok(result: Any) -> dict[str, Any]:
    return {'result': result, 'status': 'ok', 'time': 0}


def _error(message: str) -> dict[str, Any]:
    return {'status': {'error': message}, 'time': 0}


def _already_exists(name: str) -> dict[str, Any]:
    """Qdrant's 409 body for a create of a live collection (measured 2026-11-02)."""
    return _error(f'Wrong input: Collection `{name}` already exists!')


class StubQdrantServer:
    """Handle yielded by :func:`stub_qdrant_server`."""

    def __init__(
        self,
        *,
        create_delay_s: Mapping[str, float],
        preexisting: Iterable[str],
        conflict_on_create: Iterable[str],
        create_status: Mapping[str, int],
    ) -> None:
        self.url = ''
        self.release_event = threading.Event()
        """Set it to finish every pending and future create without its delay."""
        self._create_delay_s = dict(create_delay_s)
        self._conflict_on_create = frozenset(conflict_on_create)
        self._create_status = dict(create_status)
        self._state_lock = threading.Lock()
        self._meta_op_lock = threading.Lock()
        self._create_requests: Counter[str] = Counter()
        self._exists_requests: Counter[str] = Counter()
        self._create_bodies: dict[str, list[dict[str, Any]]] = {}
        self._created: dict[str, threading.Event] = {}
        for name in preexisting:
            self._created_event(name).set()

    def creates(self, name: str) -> int:
        """How many ``PUT /collections/{name}`` requests the server has received."""
        with self._state_lock:
            return self._create_requests[name]

    def create_bodies(self, name: str) -> list[dict[str, Any]]:
        """The parsed JSON body of every ``PUT /collections/{name}``, in arrival order."""
        with self._state_lock:
            return list(self._create_bodies.get(name, []))

    def exists_checks(self, name: str) -> int:
        """How many ``GET /collections/{name}/exists`` requests the server has received."""
        with self._state_lock:
            return self._exists_requests[name]

    def wait_created(self, name: str, timeout: float) -> bool:
        """Block until collection *name* exists server-side; False on timeout."""
        return self._created_event(name).wait(timeout)

    def _created_event(self, name: str) -> threading.Event:
        with self._state_lock:
            return self._created.setdefault(name, threading.Event())

    def _exists(self, name: str) -> bool:
        with self._state_lock:
            event = self._created.get(name)
        return event is not None and event.is_set()

    def _check_exists(self, name: str) -> tuple[int, dict[str, Any]]:
        with self._state_lock:
            self._exists_requests[name] += 1
        return 200, _ok({'exists': self._exists(name)})

    def _collection_names(self) -> list[str]:
        with self._state_lock:
            return [name for name, event in self._created.items() if event.is_set()]

    def _create(self, name: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        with self._state_lock:
            self._create_requests[name] += 1
            self._create_bodies.setdefault(name, []).append(body)
        with self._meta_op_lock:
            if self._exists(name):
                return 409, _already_exists(name)
            if name in self._conflict_on_create:
                self._created_event(name).set()
                return 409, _already_exists(name)
            if name in self._create_status:
                return self._create_status[name], _error(f'stub rejected the create of `{name}`')
            self.release_event.wait(self._create_delay_s.get(name, 0.0))
            self._created_event(name).set()
            return 200, _ok(True)

    def respond(self, method: str, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Route one request to its canned Qdrant answer."""
        segments = [unquote(part) for part in urlsplit(path).path.split('/') if part]
        match method, segments:
            case 'GET', []:
                return 200, {'title': 'qdrant - vector search engine', 'version': STUB_SERVER_VERSION}
            case 'GET', ['collections']:
                return 200, _ok({'collections': [{'name': name} for name in self._collection_names()]})
            case 'GET', ['collections', name, 'exists']:
                return self._check_exists(name)
            case 'PUT', ['collections', name]:
                return self._create(name, body)
            case 'PUT', ['collections', _, 'index']:
                return 200, _ok({'operation_id': 0, 'status': 'completed'})
            case 'POST', ['collections', _, 'points']:
                return 200, _ok([])
        return 404, _error(f'stub has no route for {method} {path}')


def _make_handler(state: StubQdrantServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args: Any) -> None:
            """Silence the default stderr access log."""

        def _serve(self) -> None:
            length = int(self.headers.get('Content-Length') or 0)
            request_body = json.loads(self.rfile.read(length)) if length else {}
            status, body = state.respond(self.command, self.path, request_body)
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_PUT = do_POST = _serve

    return Handler


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Stay quiet for a client that hung up mid-delay; report any other handler fault."""
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


@contextmanager
def stub_qdrant_server(
    *,
    create_delay_s: Mapping[str, float] | None = None,
    preexisting: Iterable[str] = (),
    conflict_on_create: Iterable[str] = (),
    create_status: Mapping[str, int] | None = None,
) -> Iterator[StubQdrantServer]:
    """Serve the stub on an ephemeral local port.

    *create_delay_s* is how long a create of each named collection takes.
    *preexisting* collections exist from the start.  A create of a collection
    in *conflict_on_create* answers 409 AND leaves it created, as when another
    creator wins between an exists check and the create.  A create of a
    collection in *create_status* answers that error status and leaves it
    absent.
    """
    state = StubQdrantServer(
        create_delay_s=create_delay_s or {},
        preexisting=preexisting,
        conflict_on_create=conflict_on_create,
        create_status=create_status or {},
    )
    httpd = _QuietServer(('127.0.0.1', 0), _make_handler(state))
    host, port = httpd.server_address[:2]
    state.url = f'http://{host!s}:{port}'
    thread = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        state.release_event.set()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
