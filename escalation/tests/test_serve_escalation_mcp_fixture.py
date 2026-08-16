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
from pathlib import Path

import conftest
import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

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


# ---------------------------------------------------------------------------
# A startup failure must NAME itself in the readiness-timeout error.
# ---------------------------------------------------------------------------


def test_startup_failure_that_is_not_a_runtimeerror_is_named_in_the_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that dies during startup must name its cause on timeout.

    ``_free_escalation_port()`` binds-then-closes, so there is an inherent
    TOCTOU window in which another process (or a concurrent worker) steals the
    port before the real bind. That loss arrives as an OSError, or as the
    SystemExit uvicorn raises via ``sys.exit`` on a failed bind -- neither of
    which is a ``RuntimeError``. A fixture that only captures RuntimeError
    therefore reports the generic "did not become ready" and DROPS the one
    fact a reader needs, which is what this test forbids.

    Driven by monkeypatching the port allocator to hand back a port this test
    is already holding, so the failure is a real bind conflict rather than a
    simulated one.

    NOTE: this test necessarily waits out the fixture's full ~10s readiness
    bound before raising -- that is the assertion, not a hang. It sits well
    inside the suite's 60s per-test pytest-timeout.
    """
    with _listener_without_mcp_route() as port:
        monkeypatch.setattr(conftest, '_free_escalation_port', lambda: port)

        gen = conftest.serve_escalation_mcp.__wrapped__()  # pyright: ignore[reportFunctionMemberAccess]
        try:
            start = next(gen)
            with pytest.raises(RuntimeError) as excinfo:
                start(tmp_path / 'queue')
        finally:
            # Finalize the fixture even on failure, so a RED run does not
            # leave the half-started server's thread behind for the rest of
            # the suite.
            gen.close()

        message = str(excinfo.value)
        assert 'did not' in message, (
            f'expected the readiness-timeout error text, got: {message!r}'
        )
        _, marker, detail = message.partition('server thread raised: ')
        assert marker and detail.strip(' )'), (
            'the readiness-timeout error must name the startup failure that '
            'actually happened (the bind conflict), not just report that the '
            'server never became ready; got: ' + repr(message)
        )


# ---------------------------------------------------------------------------
# serve_escalation_mcp must be MODULE-scoped: the migration rests on it.
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def module_scoped_consumer(
    tmp_path_factory: pytest.TempPathFactory,
    serve_escalation_mcp,
) -> tuple[str, object]:
    """A module-scoped consumer of ``serve_escalation_mcp``, exactly the shape
    both converted ``http_server`` fixtures take.

    This dependency is the whole reason ``serve_escalation_mcp`` is
    module-scoped: pytest forbids a fixture from depending on a NARROWER-scoped
    one, so while the factory was function-scoped this raised ``ScopeMismatch``
    at setup and no module-scoped consumer could exist.
    """
    queue_dir = tmp_path_factory.mktemp('serve_escalation_mcp_scope')
    base_url, _port, queue = serve_escalation_mcp(queue_dir)
    return base_url, queue


def test_a_module_scoped_fixture_may_depend_on_serve_escalation_mcp(
    module_scoped_consumer: tuple[str, object],
) -> None:
    """The shared factory is usable from a module-scoped fixture, and the
    server it returns answers a real MCP call.

    This is the self-enforcing guard for the rest of task 3736: it takes the
    same dependency ``test_capability_guard_http.py`` and
    ``test_status_authority_gate.py`` now take, so a future narrowing of
    ``serve_escalation_mcp`` back to function scope fails HERE, loudly and by
    name, instead of surfacing as two unrelated modules going red for reasons
    that read like a server bug.

    The ``list_tools()`` round-trip is deliberate: a fixture that resolved
    scope correctly but handed back a dead base_url would otherwise pass.
    """
    base_url, _queue = module_scoped_consumer

    async def _list_tools() -> list:
        transport = StreamableHttpTransport(f'{base_url}/mcp/')
        async with Client(transport) as client:
            return await client.list_tools()

    tools = asyncio.run(_list_tools())

    assert tools, f'the module-scoped server at {base_url} returned no tools'
