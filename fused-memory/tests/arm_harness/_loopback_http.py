"""Loopback HTTP doubles for driving the usage tap from both sides."""

import http.client
import json
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from _mock_openai_server import chat_completion_body

HELD_PROMPT_TOKENS = 321


def post_json(
    base_url: str, path: str, body: bytes, headers: dict[str, str] | None = None
) -> tuple[int, bytes]:
    parts = urlsplit(base_url)
    connection = http.client.HTTPConnection(parts.hostname or '', parts.port, timeout=30)
    try:
        connection.request(
            'POST', path, body=body, headers={'Content-Type': 'application/json', **(headers or {})}
        )
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def unused_port() -> int:
    """A loopback port nothing listens on."""
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def usage_body(prompt_tokens: int, completion_tokens: int, finish_reason: str) -> dict:
    body = chat_completion_body('{"entities": []}')
    body['usage'] = {
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': prompt_tokens + completion_tokens,
    }
    body['choices'][0]['finish_reason'] = finish_reason
    return body


@contextmanager
def held_upstream(received: threading.Event, release: threading.Event) -> Iterator[str]:
    """An upstream that answers each call only once ``release`` is set."""
    body = json.dumps(usage_body(HELD_PROMPT_TOKENS, 5, 'stop')).encode()

    class HeldHandler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args: Any) -> None:  # noqa: A002 - stdlib signature
            """Silent."""

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            self.rfile.read(int(self.headers.get('Content-Length') or 0))
            received.set()
            release.wait(timeout=30)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), HeldHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
