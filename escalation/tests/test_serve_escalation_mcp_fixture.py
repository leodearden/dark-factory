"""Tests for the shared ``serve_escalation_mcp`` harness (``conftest.py``).

``_serve_escalation_mcp_impl`` is the SINGLE implementation of the escalation
"real MCP server over real HTTP" test harness in this suite, exposed at two
scopes by the thin ``serve_escalation_mcp`` (function) and
``serve_escalation_mcp_module`` (module) delegates: task 3736 folded
``test_capability_guard_http.py`` and ``test_status_authority_gate.py`` onto
it, and ``test_legibility_census_escalation_e2e.py`` already drove it. Because
it is shared, its contract is tested HERE, once — rather than as N byte-similar
copies of the same regression test, one per consumer module, which is the
lockstep duplication (INV-5) task 3736 exists to remove.

These tests need the RAW module attributes — the undecorated fixture generator
via ``__wrapped__``, so one server's startup and teardown can be observed in
isolation from the module-scoped instance serving other tests, plus
module-level helpers and constants that are not fixtures at all. conftest hands
ITSELF over for exactly that, through the ``escalation_conftest`` fixture.
Injection, not a module-level ``import conftest``, is what makes the reference
correct under any collection order: under the repo-wide
``--import-mode=importlib`` addopts pytest names a conftest BY COLLISION, so
the bare name belongs to whichever conftest claimed it first — which in any
repo-root multi-package run is not this one. See
``test_the_injected_conftest_is_this_directorys_conftest``.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# The injected ``escalation_conftest`` is annotated ``Any`` for one reason:
# ``@pytest.fixture`` types its result as ``FixtureFunctionDefinition``, which
# does not declare the ``__wrapped__`` that pytest sets at runtime -- and the
# undecorated generator this module needs (see the docstring) is reachable only
# through that undeclared attribute.

# ---------------------------------------------------------------------------
# The conftest reference must resolve to THIS directory's conftest.
# ---------------------------------------------------------------------------


def test_the_injected_conftest_is_this_directorys_conftest(
    escalation_conftest: Any,
) -> None:
    """The conftest these tests reach into must be THIS directory's conftest.

    Under the repo-wide ``--import-mode=importlib`` addopts, pytest names a
    conftest module BY COLLISION, so a bare name is not a stable handle.
    Measured in this repo: ``escalation/tests/conftest.py`` is
    ``__name__ == 'conftest'`` under ``cd escalation && pytest tests/...``, but
    once the repo-root ``conftest.py`` shim claims the bare name first (any
    repo-root multi-package run) escalation's is disambiguated to
    ``'escalation.tests.conftest'`` -- and a module-level ``import conftest``
    then binds the ROOT SHIM. Measured: 3 of this module's tests died with
    ``AttributeError: module 'conftest' has no attribute
    'serve_escalation_mcp'``. That is the loud version; the silent version --
    the winner happening to carry a same-named attribute, so these tests run
    green against the WRONG object -- is why this is pinned at all.

    Only the file assertion is made. The stronger property, that the injected
    object is the LIVE module pytest loaded rather than a second import of the
    same file, is pinned BEHAVIOURALLY by the tests below that
    ``monkeypatch.setattr`` ``_bind_escalation_listener`` / ``_READY_TIMEOUT_S``
    on it and observe the fixture body pick both up -- something no second
    import could satisfy. Asserting it here instead against
    ``request.config.pluginmanager.get_plugin(str(path))`` would buy nothing
    extra while adding a dependency on pytest's undocumented
    conftest-keyed-by-str(path) internals, which can change on a version bump.
    """
    assert Path(escalation_conftest.__file__) == Path(__file__).with_name('conftest.py'), (
        'escalation_conftest resolved to the wrong file -- a bare-name '
        f'collision is exactly this failure; got {escalation_conftest.__file__}'
    )


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

    A ``listen()``-only socket that never ``accept()``s would be the smaller
    fake, but it is a different case: there the MCP client's request blocks on
    the never-served connection (>180s measured, until the client's own
    transport timeout) and fails only by the attempt's ``timeout_s`` bound,
    which is the subject of its own test below. Answering 404 is the faithful
    fake for THIS window — a not-yet-mounted route is what a real FastMCP app
    serves mid-startup.
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


def test_handshake_readiness_rejects_a_live_port_without_the_mcp_route(
    escalation_conftest: Any,
) -> None:
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

        ready = asyncio.run(
            escalation_conftest._mcp_handshake_ready(
                f'http://127.0.0.1:{port}', timeout_s=5.0,
            )
        )

        assert ready is False, (
            'a live TCP port with no /mcp/ route must NOT read as ready -- '
            'gating readiness on a bare TCP connect is what lets the first '
            'call race a 404 against the not-yet-mounted /mcp/ route'
        )


@pytest.mark.timeout(60)
def test_a_handshake_attempt_against_a_listener_that_never_answers_is_bounded(
    escalation_conftest: Any,
) -> None:
    """One readiness attempt must end within its ``timeout_s``, even against
    an endpoint that accepts the connection and never answers.

    That endpoint is the incident's foreign listener (task 5934): once the
    fixture's own server had died on a lost bind, readiness kept probing the
    port, found someone else's listener there, and a single unbounded attempt
    ran on until pytest-timeout fired 300s later -- the MCP client's own read
    timeout is minutes. A ``listen()``-only socket reproduces it exactly: the
    kernel completes the handshake into the backlog, and nothing ever reads.

    The last recorded error must be the ``TimeoutError`` of the bound itself,
    so the attempt is known to have ended BY the bound and not by some other
    failure that happened to be quick.
    """
    hung = escalation_conftest._bind_escalation_listener()
    try:
        hung.listen()
        errors: list[BaseException] = []

        ready = asyncio.run(
            escalation_conftest._mcp_handshake_ready(
                f'http://127.0.0.1:{hung.getsockname()[1]}', errors, timeout_s=0.5,
            )
        )

        assert ready is False
        assert errors, 'a failed attempt must record why it failed'
        assert isinstance(errors[-1], TimeoutError), (
            f'expected the attempt to end by its timeout_s bound; got {errors[-1]!r}'
        )
    finally:
        hung.close()


# ---------------------------------------------------------------------------
# The fixture OWNS its port from allocation through serve (task 5934).
# ---------------------------------------------------------------------------


def test_an_allocated_listener_holds_its_port_and_refuses_connections_until_served(
    escalation_conftest: Any,
) -> None:
    """An allocated listener is a hold on the port, not a guess at a free one.

    The incident this pins (task 5934): the old picker bound port 0, CLOSED
    the socket and returned only the number, so under parallel load another
    process bound that number before uvicorn did, and the server thread died
    with EADDRINUSE. The thief here uses SO_REUSEADDR -- the most permissive
    ordinary bind, and the one uvicorn itself makes -- so if even it is
    refused, no ordinary bind can take the port.

    The refused connect is the second half of the same property: a held but
    not-yet-served port must REFUSE, not accept into a backlog nobody drains.
    Accept-and-never-answer is the hang shape that turned the incident's bind
    loss into a 300s pytest-timeout instead of a fast failure.
    """
    listener = escalation_conftest._bind_escalation_listener()
    try:
        port = listener.getsockname()[1]

        thief = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            thief.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            with pytest.raises(OSError) as excinfo:
                thief.bind(('127.0.0.1', port))
            assert excinfo.value.errno == errno.EADDRINUSE, (
                f'expected EADDRINUSE from a second bind of the held port; '
                f'got {excinfo.value!r}'
            )
        finally:
            thief.close()

        with pytest.raises(ConnectionRefusedError):
            socket.create_connection(('127.0.0.1', port), timeout=1.0)
    finally:
        listener.close()


def test_the_fixture_serves_on_the_listener_it_allocated_and_releases_it(
    escalation_conftest: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server listens on the very socket the fixture allocated, and
    teardown releases it.

    ``SO_ACCEPTCONN`` on the recorded object is what tells the two designs
    apart: a close-then-rebind picker would leave the allocated socket closed
    or merely bound, whereas handing that socket to the server makes it the
    kernel's LISTEN socket for the served port. The ``list_tools()``
    round-trip rules out a listener that is up but not serving MCP.
    """
    real = escalation_conftest._bind_escalation_listener
    allocated: list[socket.socket] = []

    def _recording() -> socket.socket:
        sock = real()
        allocated.append(sock)
        return sock

    monkeypatch.setattr(escalation_conftest, '_bind_escalation_listener', _recording)

    gen = escalation_conftest.serve_escalation_mcp.__wrapped__()
    try:
        start = next(gen)
        base_url, port, _queue = start(tmp_path / 'queue')

        assert len(allocated) == 1, (
            f'expected the fixture to allocate exactly one listener; got {allocated}'
        )
        assert port == allocated[0].getsockname()[1]
        assert allocated[0].getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1, (
            'the allocated socket is not the one listening: the server bound '
            'the port afresh instead of serving on the socket the fixture held'
        )

        async def _list_tools() -> list:
            transport = StreamableHttpTransport(f'{base_url}/mcp/')
            async with Client(transport) as client:
                return await client.list_tools()

        assert asyncio.run(_list_tools()), (
            f'the server on the allocated listener at {base_url} returned no tools'
        )

        with pytest.raises(StopIteration):
            next(gen)

        assert allocated[0].fileno() == -1, (
            'teardown must close the allocated listener, releasing its port'
        )
    finally:
        gen.close()


# ---------------------------------------------------------------------------
# A readiness failure is STRUCTURED, and names whichever side of the wire failed.
# ---------------------------------------------------------------------------


def test_a_server_that_dies_during_startup_ends_the_wait_and_is_named(
    escalation_conftest: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server thread that dies during startup ENDS the readiness wait, and
    the error carries the startup failure as an object.

    A dying server thread is the first half of the task-5934 incident, where
    it was a lost bind. Readiness then kept probing the dead server's port
    until the deadline, and so ended up talking to whatever listener held that
    port next. Ending the wait as soon as the thread is gone is what stops
    that. ``_READY_TIMEOUT_S`` is deliberately NOT shortened: ending early is
    the subject, so a regression pays the production bound once and then fails
    on ``server_exited``, which records why the wait ended.

    The failure is a ValueError, not a ``RuntimeError``: asyncio's
    ``create_server`` rejects the datagram socket patched in as the listener.
    That makes it a deterministic, real startup failure, and a fixture that
    captured only RuntimeError would drop it. Whether a handshake attempt ran
    before the thread died is timing-dependent, so ``last_handshake_error`` is
    not asserted.
    """
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        udp.bind(('127.0.0.1', 0))
        udp_port = udp.getsockname()[1]
        monkeypatch.setattr(escalation_conftest, '_bind_escalation_listener', lambda: udp)

        gen = escalation_conftest.serve_escalation_mcp.__wrapped__()
        try:
            start = next(gen)
            with pytest.raises(escalation_conftest.EscalationServerNotReady) as excinfo:
                start(tmp_path / 'queue')
        finally:
            # Finalize the fixture even on failure, so a RED run does not
            # leave the half-started server's thread behind for the rest of
            # the suite.
            gen.close()

        not_ready = excinfo.value
        assert not_ready.server_exited is True, (
            'the readiness wait must end because the serving thread exited, '
            f'not run on to the deadline; got {not_ready!r}'
        )
        assert isinstance(not_ready.serve_error, ValueError), (
            'the startup failure that actually happened must be carried as '
            f'the object it was; got {not_ready.serve_error!r}'
        )
        assert not_ready.port == udp_port
    finally:
        udp.close()


def test_a_healthy_server_the_client_cannot_reach_names_the_client_error(
    escalation_conftest: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the server is healthy and only the CLIENT half fails, the error
    names the client's failure.

    ``_mcp_handshake_ready`` swallows every exception and returns False, so
    unless it records the last one, a purely client-side failure (a proxy env
    var, a transport/protocol mismatch, a fastmcp version skew) leaves a live
    server thread, no ``serve_error``, and nothing to say about why the
    handshake never completed.

    Driven by a real client-side failure: the proxy env vars name a port held
    by ``_bind_escalation_listener``, which refuses connects, and httpx honours
    them. ``_READY_TIMEOUT_S`` is shortened because this server never exits,
    so the deadline is the only way the wait can end.
    """
    dead_proxy = escalation_conftest._bind_escalation_listener()
    try:
        proxy_url = f'http://127.0.0.1:{dead_proxy.getsockname()[1]}'
        for var in ('HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy'):
            monkeypatch.setenv(var, proxy_url)
        for var in ('NO_PROXY', 'no_proxy'):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(escalation_conftest, '_READY_TIMEOUT_S', 0.5)

        gen = escalation_conftest.serve_escalation_mcp.__wrapped__()
        try:
            start = next(gen)
            with pytest.raises(escalation_conftest.EscalationServerNotReady) as excinfo:
                start(tmp_path / 'queue')
        finally:
            gen.close()

        not_ready = excinfo.value
        assert not_ready.server_exited is False, (
            f'the server thread was healthy; got {not_ready!r}'
        )
        assert not_ready.serve_error is None
        assert not_ready.last_handshake_error is not None, (
            'a timeout whose server thread is healthy must name the last '
            'client-side handshake failure'
        )
    finally:
        dead_proxy.close()


# ---------------------------------------------------------------------------
# serve_escalation_mcp_module must be MODULE-scoped: the migration rests on it.
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def module_scoped_consumer(
    tmp_path_factory: pytest.TempPathFactory,
    serve_escalation_mcp_module,
) -> tuple[str, object]:
    """A module-scoped consumer of ``serve_escalation_mcp_module``, exactly the
    shape both converted ``http_server`` fixtures take.

    This dependency is the whole reason the ``_module`` variant exists: pytest
    forbids a fixture from depending on a NARROWER-scoped one, so requesting
    the function-scoped ``serve_escalation_mcp`` from here raises
    ``ScopeMismatch`` at setup and no module-scoped consumer could exist.
    """
    queue_dir = tmp_path_factory.mktemp('serve_escalation_mcp_scope')
    base_url, _port, queue = serve_escalation_mcp_module(queue_dir)
    return base_url, queue


def test_a_module_scoped_fixture_may_depend_on_serve_escalation_mcp(
    module_scoped_consumer: tuple[str, object],
) -> None:
    """The shared factory is usable from a module-scoped fixture, and the
    server it returns answers a real MCP call.

    This is the self-enforcing guard for the rest of task 3736: it takes the
    same dependency ``test_capability_guard_http.py`` and
    ``test_status_authority_gate.py`` now take, so a future narrowing of
    ``serve_escalation_mcp_module`` back to function scope fails HERE, loudly
    and by name, instead of surfacing as two unrelated modules going red for
    reasons that read like a server bug.

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


# ---------------------------------------------------------------------------
# Teardown must stop the daemon serving thread (task 2741).
# ---------------------------------------------------------------------------


def test_the_fixture_stops_its_serving_thread_on_teardown(
    escalation_conftest: Any,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Regression test (task 2741): ``serve_escalation_mcp`` must explicitly
    stop the daemon serving thread + event loop of every server it started,
    instead of relying on process exit to kill them.

    Getting this wrong leaks a daemon thread whose event loop then acts as a
    background ``time.monotonic()`` caller for the rest of the process, and
    unwinds anyio's shielded lifespan cleanup at uncontrolled GC time.

    THE canonical copy: this assertion used to exist verbatim in both
    ``test_capability_guard_http.py`` and ``test_status_authority_gate.py``,
    each driving its own module-local fixture. Once both delegate here those
    would be two byte-similar tests of ONE fixture -- the lockstep duplication
    (INV-5) task 3736 exists to remove -- so they were folded into this one.

    Drives the fixture generator directly via ``__wrapped__``, bypassing
    pytest's fixture caching, so this test's own server is isolated from the
    module-scoped instance already serving the other tests in this file: the
    ``threading.enumerate()`` before/after diff plus the port-qualified thread
    name identify exactly one thread, this test's.
    """
    before = set(threading.enumerate())
    gen = escalation_conftest.serve_escalation_mcp.__wrapped__()
    try:
        start = next(gen)
        _base_url, port, _queue = start(tmp_path_factory.mktemp('serve_teardown'))

        new = [
            t for t in set(threading.enumerate()) - before
            if t.name == f'escalation-mcp-http-{port}'
        ]
        assert len(new) == 1, (
            f'Expected exactly one new escalation-mcp-http-{port} serving '
            f'thread; found {new}'
        )
        serving = new[0]
        assert serving.is_alive(), 'Serving thread must be alive during yield'

        with pytest.raises(StopIteration):
            next(gen)

        assert not serving.is_alive(), (
            'serve_escalation_mcp leaked its daemon serving thread past '
            'teardown (generator finalized via StopIteration) -- the fixture '
            'must explicitly stop its event loop and join the thread instead '
            'of relying on process exit to kill it.'
        )
    finally:
        # Best-effort: ensure finalization ran even if an assertion above
        # failed, so a RED failure does not leave extra servers running for
        # the rest of the suite. No-op if already exhausted.
        gen.close()
