"""Tests for the shared ``serve_escalation_mcp`` harness (``conftest.py``).

``serve_escalation_mcp`` is the SINGLE implementation of the escalation
"real MCP server over real HTTP" test harness in this suite: task 3736 folded
``test_capability_guard_http.py`` and ``test_status_authority_gate.py`` onto
it, and ``test_legibility_census_escalation_e2e.py`` already drove it. Because
it is shared, its contract is tested HERE, once — rather than as N byte-similar
copies of the same regression test, one per consumer module, which is the
lockstep duplication (INV-5) task 3736 exists to remove.

``import conftest`` rather than fixture injection: these tests need the RAW
module attributes — the undecorated fixture generator via ``__wrapped__``, so
one server's startup and teardown can be observed in isolation from the
module-scoped instance serving other tests, plus module-level helpers that are
not fixtures at all. conftest.py's own header notes that ``from conftest
import ...`` is fragile under the repo-wide ``--import-mode=importlib``
addopts; a plain ``import conftest`` is verified to resolve to
``escalation/tests/conftest.py`` under BOTH rootdirs this suite is run from
(``cd escalation && uv run pytest tests/`` in the verify lane, and ``uv run
pytest escalation/tests`` from the repo root, where a repo-root conftest.py
also exists).
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import conftest

# ---------------------------------------------------------------------------
# A live TCP endpoint that is NOT an MCP server — the readiness discriminator.
# ---------------------------------------------------------------------------


class _NotMcpHandler(BaseHTTPRequestHandler):
    """Answer 404 to everything: a live HTTP port with no ``/mcp/`` route."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's own API
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's own API
        self.send_error(404)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        """Silence the handler's default per-request stderr log."""


@contextlib.contextmanager
def _listener_without_mcp_route() -> Iterator[int]:
    """Yield the port of a live HTTP listener that has no ``/mcp/`` route.

    This reproduces exactly the window the handshake readiness gate exists to
    close: the OS accept queue is up — a bare ``socket.create_connection``
    probe succeeds instantly — but nothing is mounted at ``/mcp/`` yet, so an
    MCP ``initialize`` cannot complete.

    A ``listen()``-only socket that never ``accept()``s is deliberately NOT
    used here even though it is the smaller fake: measured, the MCP client's
    request then blocks on the never-served connection until its own transport
    timeout (>180s observed, i.e. no result at all) instead of failing fast,
    which would blow this suite's 60s pytest-timeout. Answering 404 is also
    the more faithful fake — a not-yet-mounted route is what a real FastMCP
    app serves mid-startup.
    """
    server = ThreadingHTTPServer(('127.0.0.1', 0), _NotMcpHandler)
    thread = threading.Thread(
        target=server.serve_forever, name='not-mcp-listener', daemon=True,
    )
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
        assert not thread.is_alive(), (
            'the not-mcp-listener helper leaked its daemon thread past the test'
        )


# ---------------------------------------------------------------------------
# Readiness is gated on a real MCP handshake, not a bare TCP connect.
# ---------------------------------------------------------------------------


def test_handshake_readiness_rejects_a_live_port_without_the_mcp_route() -> None:
    """``_mcp_handshake_ready`` must reject a port that merely ACCEPTS.

    Both assertions below are made against the SAME port, so the
    discrimination between the two readiness notions is the subject of this
    test rather than an assumption about it: the bare TCP connect (the
    readiness probe ``serve_escalation_mcp`` used before task 3736) succeeds,
    and the handshake predicate must still say "not ready".

    That difference is load-bearing, not pedantic: a successful connect only
    proves the OS accept queue is up, not that the FastMCP ASGI app has
    finished mounting the ``/mcp/`` route, so a TCP-gated fixture can hand a
    caller a base_url whose first real call races a 404 against the
    not-yet-live route.
    """
    with _listener_without_mcp_route() as port:
        with socket.create_connection(('127.0.0.1', port), timeout=1.0):
            pass  # the weaker probe succeeds here, i.e. would report "ready"

        ready = asyncio.run(conftest._mcp_handshake_ready(f'http://127.0.0.1:{port}'))

        assert ready is False, (
            'a live TCP port with no /mcp/ route must NOT read as ready -- '
            'gating readiness on a bare TCP connect is what lets the first '
            'call race a 404 against the not-yet-mounted /mcp/ route'
        )
