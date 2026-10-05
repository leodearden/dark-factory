"""A dependency-free, OpenAI-compatible recording mock server.

Why a REAL socket rather than transport injection or a mocked AsyncOpenAI: the
failure class this exists to disprove is a ``base_url`` that is accepted and
then IGNORED (graphiti-#912). ``OpenAIEmbedderConfig`` is a pydantic model with
the default ``extra='ignore'``, so a misspelled kwarg is dropped silently and
every constructor-kwarg assertion still passes while traffic goes to
api.openai.com. Transport injection shares that blind spot — it intercepts
before URL resolution matters. Only a real port proves the configured host
received the bytes.

Built on stdlib ``http.server`` because no HTTP-mock library is installed
(respx / pytest-httpserver / aioresponses / responses are all absent), and
adding a dependency for one test is not warranted.

A plain importable module, not a conftest fixture, following the
``_fm_helpers.py`` convention, so several test modules can import it.

Binds 127.0.0.1 on port 0 (ephemeral) — never a fixed port, so ``-n auto``
xdist workers cannot collide.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# Long enough to survive graphiti's OpenAIEmbedder, which TRUNCATES the vector
# to embedding_dim — a short vector would surface as a length mismatch and be
# misread as a plumbing bug.
DEFAULT_EMBEDDING_LEN = 1536


def chat_completion_body(content: str) -> dict[str, Any]:
    """A minimal valid chat.completions payload.

    ``content`` is a JSON *string*: OpenAIGenericClient._generate_response does
    ``json.loads(response.choices[0].message.content)``.
    """
    return {
        'id': 'chatcmpl-mock',
        'object': 'chat.completion',
        'created': 1700000000,
        'model': 'mock-model',
        'choices': [
            {
                'index': 0,
                'message': {'role': 'assistant', 'content': content},
                'finish_reason': 'stop',
            },
        ],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
    }


def _embeddings_body(count: int, length: int) -> dict[str, Any]:
    return {
        'object': 'list',
        'data': [
            {
                'object': 'embedding',
                'index': i,
                'embedding': [0.01] * length,
            }
            for i in range(max(count, 1))
        ],
        'model': 'mock-embedding-model',
        'usage': {'prompt_tokens': 1, 'total_tokens': 1},
    }


class MockOpenAIServer:
    """Handle yielded by :func:`mock_openai_server`."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        # ThreadingHTTPServer serves concurrently — guard the record list.
        self._lock = threading.Lock()
        self._requests: list[dict[str, Any]] = []
        self._responses: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        self.chat_content = '{"ok": true}'
        self.chat_responder: Callable[[Any], str] | None = None
        """When set, each chat reply's content is computed from that request's JSON body."""
        self.embedding_len = DEFAULT_EMBEDDING_LEN

    @property
    def base_url(self) -> str:
        """The value to configure as ``api_url`` — carries the /v1 suffix, as
        our OpenAIProviderConfig.api_url values do."""
        return f'http://{self.host}:{self.port}/v1'

    @property
    def requests(self) -> list[dict[str, Any]]:
        """A snapshot of the recorded requests."""
        with self._lock:
            return list(self._requests)

    def requests_to(self, suffix: str) -> list[dict[str, Any]]:
        """Recorded requests whose path ends with ``suffix``."""
        return [r for r in self.requests if r['path'].endswith(suffix)]

    def set_response(self, path_suffix: str, body: dict[str, Any]) -> None:
        """Override the canned response for paths ending in ``path_suffix``."""
        self.set_response_sequence(path_suffix, [(200, body)])

    def set_response_sequence(
        self, path_suffix: str, bodies: Sequence[tuple[int, dict[str, Any]]]
    ) -> None:
        """Serve ``(status, body)`` pairs in order for ``path_suffix``; the last one repeats."""
        if not bodies:
            raise ValueError('a response sequence needs at least one (status, body) pair')
        with self._lock:
            self._responses[path_suffix] = list(bodies)

    # -- internals used by the handler --

    def _record(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self._requests.append(entry)

    def _canned(self, path: str) -> tuple[int, dict[str, Any]] | None:
        with self._lock:
            for suffix, sequence in self._responses.items():
                if path.endswith(suffix):
                    return sequence.pop(0) if len(sequence) > 1 else sequence[0]
        return None


def _make_handler(state: MockOpenAIServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args: Any) -> None:  # noqa: A002 - stdlib signature
            """Silence the default stderr access log."""

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            length = int(self.headers.get('Content-Length') or 0)
            raw = self.rfile.read(length) if length else b''
            try:
                json_body = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                json_body = None

            state._record({
                'path': self.path,
                'method': 'POST',
                'headers': dict(self.headers),
                'json_body': json_body,
            })

            canned = state._canned(self.path)
            if canned is not None:
                self._send_json(*canned)
                return

            if self.path.endswith('/chat/completions'):
                responder = state.chat_responder
                content = responder(json_body) if responder else state.chat_content
                self._send_json(200, chat_completion_body(content))
                return

            if self.path.endswith('/embeddings'):
                inputs = (json_body or {}).get('input')
                count = len(inputs) if isinstance(inputs, list) else 1
                # Honour an explicitly requested dimensionality so a
                # dimensions-plumbing test gets a correctly-sized vector back.
                length = (json_body or {}).get('dimensions') or state.embedding_len
                self._send_json(200, _embeddings_body(count, int(length)))
                return

            self._send_json(404, {'error': {'message': f'unhandled path {self.path}'}})

    return Handler


@contextmanager
def mock_openai_server() -> Iterator[MockOpenAIServer]:
    """Run a recording OpenAI-compatible server on an ephemeral local port."""
    # Bind port 0 ONCE and read back what the kernel assigned. Probing for a
    # free port and then rebinding it would be a TOCTOU race against any other
    # process (or xdist worker) on the box.
    state = MockOpenAIServer('127.0.0.1', 0)
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), _make_handler(state))
    state.host, state.port = str(httpd.server_address[0]), int(httpd.server_address[1])
    # Don't let a still-open client connection block shutdown().
    httpd.daemon_threads = True

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
