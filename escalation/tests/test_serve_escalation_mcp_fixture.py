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
from typing import Any

import conftest as _conftest_module
import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# Deliberately re-bound through an ``Any`` alias, so the accesses below are not
# type-checked against whichever ``conftest`` pyright happened to resolve. Two
# things make a statically-checked ``conftest.<attr>`` unreliable HERE, and the
# alias closes both:
#
# 1. WHICH conftest. ``escalation/pyproject.toml``'s ``[tool.pyright]`` carries
#    BOTH ``tests`` and ``..`` on extraPaths, and each holds a ``conftest.py``
#    (this suite's, and the repo-root sys.path shim). Measured: pyright resolves
#    the bare name to a different one of the two depending on the directory it
#    is invoked from, so the attributes below read as missing from "module
#    conftest" in some invocations and resolve fine in others.
# 2. ``__wrapped__``. ``@pytest.fixture`` types its result as
#    ``FixtureFunctionDefinition``, which does not declare the ``__wrapped__``
#    that pytest sets at runtime -- the undecorated generator this module needs
#    (see the docstring) is reachable only through that undeclared attribute.
#
# Runtime is untouched: this is the same ``import conftest``, so the empirically
# verified resolution to ``escalation/tests/conftest.py`` still applies.
conftest: Any = _conftest_module

# ---------------------------------------------------------------------------
# The conftest reference must be resolved by IDENTITY, not by bare module name.
# ---------------------------------------------------------------------------


def test_the_conftest_reference_is_resolved_by_identity_not_by_bare_name(
    escalation_conftest: Any,
    request: pytest.FixtureRequest,
) -> None:
    """The conftest these tests reach into must be THIS directory's conftest.

    Under the repo-wide ``--import-mode=importlib`` addopts, pytest names a
    conftest module BY COLLISION, so a bare name is not a stable handle.
    Measured in this repo:

    * ``cd escalation && pytest tests/...`` -- ``escalation/tests/conftest.py``
      is ``__name__ == 'conftest'``; a bare ``import conftest`` finds it.
    * ``pytest shared/tests/test_task_statuses.py <this file>`` from the repo
      root (same with ``orchestrator/tests/test_routing.py`` leading) -- the
      repo-root ``conftest.py`` sys.path shim claims the bare name first, and
      escalation's is disambiguated to ``'escalation.tests.conftest'``. A
      module-level ``import conftest`` then binds the ROOT SHIM, and 3 of this
      module's 4 tests died with ``AttributeError: module 'conftest' has no
      attribute 'serve_escalation_mcp'``.

    That is the loud version of the failure. The silent version is worse and is
    the real reason this contract is pinned: if the module that won the bare
    name happened to carry a same-named attribute, the tests below would run
    green against the WRONG object instead of erroring. This module is the
    single contract test for a harness three modules depend on, so it must not
    be able to lose its subject to collection order.

    The identity assertion is made against the plugin manager, which keys
    conftest plugins by absolute PATH (measured) rather than by module name --
    so it pins the exact module object pytest itself loaded for this directory,
    which is what makes ``monkeypatch.setattr`` on it patch the live copy the
    fixtures actually read.
    """
    conftest_path = Path(__file__).with_name('conftest.py')
    registered = request.config.pluginmanager.get_plugin(str(conftest_path))

    assert escalation_conftest is registered, (
        'escalation_conftest must BE the conftest module object pytest loaded '
        f'for {conftest_path} (keyed by path in the plugin manager), not a '
        f'second import of the same file; got {escalation_conftest!r} vs '
        f'{registered!r}'
    )
    assert Path(escalation_conftest.__file__) == conftest_path, (
        'escalation_conftest resolved to the wrong file -- a bare-name '
        f'collision is exactly this failure; got {escalation_conftest.__file__}'
    )
    assert hasattr(escalation_conftest, 'serve_escalation_mcp'), (
        'the resolved conftest does not carry serve_escalation_mcp, so it is '
        'not this suite\'s harness conftest'
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

        gen = conftest.serve_escalation_mcp.__wrapped__()
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


# ---------------------------------------------------------------------------
# Teardown must stop the daemon serving thread (task 2741).
# ---------------------------------------------------------------------------


def test_the_fixture_stops_its_serving_thread_on_teardown(
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
    gen = conftest.serve_escalation_mcp.__wrapped__()
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
