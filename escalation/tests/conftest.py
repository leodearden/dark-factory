"""pytest configuration — ensure local src takes precedence over installed package.

Without this conftest, ``import escalation`` resolves to the editable
install in the prod venv (which points at /home/leo/src/dark-factory/),
not the worktree under test.  Injecting the worktree's ``src`` at the
front of ``sys.path`` mirrors the orchestrator/tests pattern.

Also injects ``orchestrator/src`` and ``shared/src`` so that
TestMergeRequestDedup and TestGetMergeQueue can do inline
``from orchestrator.*`` imports.  Those tests live in the escalation test
suite but exercise code paths that require the neighbouring orchestrator
package (InFlightMergeRegistry, MergeOutcome, MergeResult).

Because the escalation project-scoped venv intentionally excludes
``aiosqlite`` (pulled in by ``shared/__init__`` via
``shared.async_sqlite_base``), we pre-populate ``sys.modules`` with a
minimal stub for the async-sqlite layer before adding the paths.  Only
the stubs that satisfy the orchestrator import chain are provided;
everything else is loaded from the actual source files.
"""

from __future__ import annotations

import collections
import sys
import types
from pathlib import Path

import pytest

_SRC = Path(__file__).parent.parent / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Orchestrator/shared path injection for TestMergeRequestDedup /
# TestGetMergeQueue tests.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.parent
_SHARED_SRC = _REPO_ROOT / 'shared' / 'src'
_ORCH_SRC = _REPO_ROOT / 'orchestrator' / 'src'

if _SHARED_SRC.is_dir() and _ORCH_SRC.is_dir():
    # Stub ``shared`` package before adding shared/src to sys.path so that
    # Python never executes ``shared/__init__.py`` (which imports aiosqlite,
    # not installed in the escalation project-scoped venv).
    if 'shared' not in sys.modules:
        _shared_stub = types.ModuleType('shared')
        _shared_stub.__path__ = [str(_SHARED_SRC / 'shared')]
        _shared_stub.__package__ = 'shared'
        sys.modules['shared'] = _shared_stub

    # ``shared.async_sqlite_base`` imports aiosqlite.
    # ``shared.sqlite_sync_base`` re-exports CheckpointResult from it.
    # Provide minimal stubs so orchestrator modules that import these can load.
    if 'shared.async_sqlite_base' not in sys.modules:
        _async_base_stub = types.ModuleType('shared.async_sqlite_base')
        _CheckpointResult = collections.namedtuple('CheckpointResult', ['log', 'checkpointed'])
        # Populate via __dict__.update: pyright rejects attribute assignment on a
        # bare ModuleType instance (reportAttributeAccessIssue), but a module
        # namespace is a plain dict[str, Any], so this is runtime-equivalent and
        # type-clean.
        _async_base_stub.__dict__.update(
            CheckpointResult=_CheckpointResult,
            AsyncSqliteBase=type('AsyncSqliteBase', (), {}),
            apply_full_durability_pragmas=lambda *a, **kw: None,
            apply_wal_pragmas=lambda *a, **kw: None,
            connect_daemon=lambda *a, **kw: None,
        )
        sys.modules['shared.async_sqlite_base'] = _async_base_stub

    for _p in (str(_SHARED_SRC), str(_ORCH_SRC)):
        if _p not in sys.path:
            sys.path.insert(0, _p)

# NOTE (task 3644, review finding #6): the `scripts/` + `scripts/legibility/`
# path injection the census<->escalation E2E contract test needs is NOT done
# here.  It lives in that one test module
# (`test_legibility_census_escalation_e2e.py`), scoped to its only consumer and
# APPENDING rather than inserting.  `scripts/legibility/` contributes very
# generic flat top-level module names — `config.py`, `inventory.py`,
# `sampling.py`, `digest.py`, `coder.py`, `codebook.py` — and putting those at
# `sys.path[0]` for all ~1080 tests in this directory would shadow any
# same-named module for every one of them, with a wrong-module import (not an
# ImportError) as the failure mode.  The two-entry idiom in
# `scripts/tests/conftest.py` is fine THERE because every test in that
# directory needs those modules; here exactly one does.

# Suite-wide git isolation (task 3355, incident esc-3072-3).  The verify lane
# runs `cd escalation && uv run pytest tests/`, which makes rootdir the
# SUBPROJECT — the repo-root conftest.py is never loaded, so each test-root
# conftest wires the defence itself.  APPEND the repo root, never insert(0, ...):
# at sys.path[0] it would make the subproject directories resolve as namespace
# packages pointing at the project folder instead of src/<pkg>/, beating the
# inserts above.  Placed after the stubbing block to keep that contiguous;
# df_pytest_isolation is stdlib + pytest only, so it imports cleanly in this
# venv, which deliberately excludes aiosqlite.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from df_pytest_isolation import (  # noqa: E402
    _df_deploy_clocks_unwritten,  # noqa: F401  — the binding IS the wiring
    _df_git_ceiling_at_basetemp,  # noqa: F401  — the binding IS the wiring
    reject_unsafe_basetemp,
)


def pytest_configure(config):
    """Refuse a --basetemp aimed inside a live task worktree (esc-3072-3)."""
    reject_unsafe_basetemp(config)


# ---------------------------------------------------------------------------
# serve_escalation_mcp — a REAL escalation MCP server over REAL HTTP (task 3644)
# ---------------------------------------------------------------------------
#
# A pytest FIXTURE rather than an importable helper, for the same reason
# `scripts/tests/conftest.py` gives for its own fixtures: fixtures auto-resolve
# for every file in this directory, whereas `from conftest import ...` is
# fragile under the repo-wide `--import-mode=importlib` addopts.
#
# This is the SINGLE home of the escalation HTTP test harness. Task 3736 folded
# the last two module-local copies into it (`test_capability_guard_http.py` and
# `test_status_authority_gate.py`, whose `http_server` fixtures are now thin
# scope/shape adapters over the factory below); `test_legibility_census_
# escalation_e2e.py` already drove it. There is no other copy to keep in
# lockstep — which is the point, per INV-5.
#
# It is exposed at TWO scopes — `serve_escalation_mcp` (function) and
# `serve_escalation_mcp_module` (module) — both of which are thin `yield from`
# delegates to the one `_serve_escalation_mcp_impl` generator below. The split
# is a scope adapter, NOT a second implementation; see the two fixture
# docstrings for why each scope exists.
#
# Two properties below are load-bearing and must not be softened:
#
# * The teardown semantics (`stopping` flag, `_drain_pending_tasks`, bounded
#   5s join with hung threads collected then asserted). Task 2741 exists
#   precisely because getting them wrong leaks a daemon thread and crashes
#   anyio's shielded lifespan cleanup at GC time.
# * Readiness is an MCP handshake, not a TCP connect (`_mcp_handshake_ready`),
#   and any startup failure is captured as BaseException — reasoning inherited
#   from the hardened `test_status_authority_gate.py` variant when task 3736
#   deduped these copies by UNION rather than by intersection.
#
# Both are pinned by tests in `test_serve_escalation_mcp_fixture.py`.

# How long `_start` waits for a server to complete its first MCP handshake.
# A module-level constant rather than a literal inside `_start` so a test that
# deliberately drives the timeout path (the not-a-RuntimeError startup-failure
# regression test) can `monkeypatch.setattr` it down instead of paying the full
# production bound on every run -- `_start` reads it as a global at call time.
_READY_TIMEOUT_S = 10.0


def _free_escalation_port() -> int:
    """Return an ephemeral TCP port free on 127.0.0.1 at the time of the call.

    Binds to port 0 (OS-assigned free port) and immediately closes the socket.
    There is an inherent (small) TOCTOU window before the real server binds the
    same port; acceptable for a single-threaded test run.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _drain_pending_tasks(loop) -> None:
    """Cancel and await every task still pending on *loop* before it closes.

    Mirrors ``asyncio.run()``'s own internal shutdown sequence
    (``_cancel_all_tasks``).  Without this, a task still suspended
    mid-``await`` when the loop is stopped -- the server's own root task
    (``run_http_async``), or an internal one it spawns (e.g. sse_starlette's
    ``_shutdown_watcher``) -- is only unwound later at uncontrolled
    garbage-collection time, outside of any running task context, which
    crashes anyio's shielded lifespan cleanup with an unraisable
    ``TypeError``/``NoEventLoopError`` instead of shutting down cleanly
    (task 2741).
    """
    import asyncio

    pending = asyncio.all_tasks(loop)
    if not pending:
        return
    for task in pending:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


async def _mcp_handshake_ready(
    base_url: str, error_box: list[BaseException] | None = None,
) -> bool:
    """True iff a real MCP client completes initialize+ping against
    ``{base_url}/mcp/``.

    A bare TCP connect only proves the OS accept queue is up; it does not
    prove the FastMCP ASGI app has finished mounting the ``/mcp/`` route.
    Polling a real handshake instead of a TCP probe closes that window — the
    first call racing a 404/hang before the route is live.

    Every failure reads as "not ready yet", so this returns a plain bool and
    never raises. But swallowing the exception ENTIRELY would leave the
    readiness timeout with nothing to say whenever the failure is on the CLIENT
    side of the wire -- a transport/protocol mismatch, a proxy env var, a
    fastmcp version skew -- because the server thread is healthy in those cases
    and ``serve_error`` stays None. That is the same "drops the one fact a
    reader needs" failure the BaseException widening in ``_serve_forever``
    exists to prevent, just from the other end. So when *error_box* is passed,
    the most recent exception is recorded into it (last write wins: the final
    poll's failure is the one that describes why the deadline was reached) for
    the caller to name in its timeout message.
    """
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    transport = StreamableHttpTransport(f'{base_url}/mcp/')
    try:
        async with Client(transport) as client:
            await client.ping()
    except Exception as exc:  # noqa: BLE001 - any failure just means "not ready yet"
        if error_box is not None:
            error_box[:] = [exc]
        return False
    return True


def _serve_escalation_mcp_impl():
    """The ONE body behind ``serve_escalation_mcp`` and
    ``serve_escalation_mcp_module``.

    A plain generator, not a fixture: pytest caches one fixture instance per
    scope, and the two consumer shapes in this suite need DIFFERENT scopes (see
    the two fixture docstrings below). Both delegate here with ``yield from``,
    so there is exactly one copy of the start/serve/teardown protocol to keep
    correct -- the whole point of task 3736, per INV-5. Do not inline this into
    either fixture.

    Call as ``serve_escalation_mcp(queue_dir)`` -> ``(base_url, port, queue)``.
    Builds ``EscalationQueue(queue_dir)`` + ``create_server(queue,
    startup_sweep=False, dedupe_config=...)`` and serves it via
    ``FastMCP.run_http_async`` on an OS-assigned free localhost port inside a
    daemon thread running its own event loop.  Readiness is polled by a real
    MCP handshake (``_mcp_handshake_ready``, bounded ~10s total) -- not a bare
    TCP connect, and no fixed sleep.  Every server started through the factory
    is torn down when the test finishes.

    ``startup_sweep=False`` so pre-seeded queue files are not relocated by the
    startup sweep (the existing test convention in this suite).

    Deduplication is DISABLED by default (``DedupeConfig(infra_dedupe_enabled=
    False)``).  ``create_server`` otherwise defaults to ``DedupeConfig()`` --
    infra_issue dedupe ON with a 600s window -- and the fail-loud legibility
    posters this fixture exists to test all file ``category='infra_issue'``.
    With dedupe on, a regression that filed the SAME escalation twice would be
    collapsed server-side into one record and an "exactly one escalation"
    assertion would still pass: green on paper, broken in practice, which is
    the precise failure mode task 3644 exists to close.  Pass an explicit
    ``dedupe_config=`` to opt back in when dedupe is the thing under test.

    Teardown is explicit (task 2741): the event loop is created in the fixture
    body (NOT inside the thread target) so it can be stopped thread-safely; a
    ``stopping`` flag distinguishes the deliberate stop-induced
    ``RuntimeError`` from a genuine startup failure; ``_drain_pending_tasks``
    cancels+gathers inside the serving thread before ``loop.close()``; and the
    join is bounded at 5s with an assert, so a hang surfaces loudly instead of
    silently leaking a daemon thread past this fixture.

    Two teardown details are load-bearing (review finding #7, task 3644):
    the ``loop.stop`` is suppressed on ``RuntimeError`` because a server thread
    that DIED during startup has already run ``loop.close()`` in its own
    ``finally`` -- stopping a closed loop would then raise out of this
    fixture's own ``finally`` and MASK the informative "did not become ready"
    error; and hung threads are COLLECTED rather than asserted in-loop, so one
    hang cannot abort the teardown of every server started after it and leak
    exactly the daemon threads the assert exists to prevent.
    """
    import asyncio
    import contextlib
    import threading
    import time

    from escalation.dedupe import DedupeConfig
    from escalation.queue import EscalationQueue
    from escalation.server import create_server

    started: list[tuple] = []

    def _start(queue_dir, *, dedupe_config=None, startup_sweep=False):
        queue = EscalationQueue(queue_dir)
        if dedupe_config is None:
            dedupe_config = DedupeConfig(infra_dedupe_enabled=False)
        mcp = create_server(
            queue, startup_sweep=startup_sweep, dedupe_config=dedupe_config,
        )
        port = _free_escalation_port()
        loop = asyncio.new_event_loop()
        serve_error: BaseException | None = None
        state = {'stopping': False}

        def _serve_forever() -> None:
            nonlocal serve_error
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    mcp.run_http_async(
                        host='127.0.0.1', port=port,
                        show_banner=False, log_level='error',
                    )
                )
            except BaseException as exc:  # noqa: BLE001 - surfaced on timeout below
                # Two distinct cases, told apart by the `stopping` flag:
                #
                # * Teardown stopped the loop mid-serve -- a deliberate,
                #   expected RuntimeError ("Event loop stopped before Future
                #   completed"), which must NOT be reported as an error.
                # * Anything raised before teardown began is a genuine startup
                #   failure and must be surfaced below.
                #
                # BaseException, not RuntimeError: _free_escalation_port()
                # binds-then-closes, so a TOCTOU port steal (or any other bind
                # failure) reaches here as an OSError -- or, since uvicorn
                # calls sys.exit(1) on a failed bind, as SystemExit, which is
                # not even an Exception subclass. Narrowing to RuntimeError
                # leaves serve_error None for both, so the readiness timeout
                # below reports only the generic "did not complete an MCP
                # handshake" and silently drops the one fact a reader needs.
                if not state['stopping']:
                    serve_error = exc
            finally:
                _drain_pending_tasks(loop)
                loop.close()

        thread = threading.Thread(
            target=_serve_forever, name=f'escalation-mcp-http-{port}', daemon=True,
        )
        thread.start()
        started.append((loop, thread, state, port))

        # Gate readiness on a real MCP handshake (initialize+ping), not a bare
        # TCP connect: a successful connect only proves the OS accept queue is
        # up, not that the FastMCP ASGI app has finished mounting /mcp/. Every
        # poll implicitly re-probes the TCP layer too (a refused/reset connect
        # just fails the handshake and retries), so no separate TCP wait is
        # needed. Bounded ~10s instead of a fixed sleep.
        base_url = f'http://127.0.0.1:{port}'
        deadline = time.monotonic() + _READY_TIMEOUT_S
        handshake_errors: list[BaseException] = []
        ready = False
        while time.monotonic() < deadline:
            if asyncio.run(_mcp_handshake_ready(base_url, handshake_errors)):
                ready = True
                break
            time.sleep(0.05)
        if not ready:
            # BOTH sides of the wire get named. `serve_error` covers a server
            # thread that died; `handshake_errors` covers the case where the
            # thread is perfectly healthy and it is the CLIENT half that cannot
            # complete -- without it that case reports only "did not complete an
            # MCP handshake", i.e. restates the timeout and explains nothing.
            detail = f' (server thread raised: {serve_error!r})' if serve_error else ''
            handshake_detail = (
                f' (last handshake error: {handshake_errors[-1]!r})'
                if handshake_errors else ''
            )
            raise RuntimeError(
                f'escalation HTTP test server did not complete an MCP handshake '
                f'on 127.0.0.1:{port} within {_READY_TIMEOUT_S:g}s'
                f'{detail}{handshake_detail}'
            )

        return f'http://127.0.0.1:{port}', port, queue

    try:
        yield _start
    finally:
        hung = []
        for loop, thread, state, port in started:
            state['stopping'] = True
            # A thread that died during startup already ran `loop.close()` in
            # its own `finally`, and stopping a CLOSED loop raises
            # RuntimeError. Letting that escape here would replace the
            # fixture's informative "did not become ready ... (server thread
            # raised: ...)" with a bare "Event loop is closed" from teardown.
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5.0)
            if thread.is_alive():
                hung.append(f'escalation-mcp-http-{port}')
        # Collected, not asserted in-loop: an in-loop assert would abort
        # teardown of every server started AFTER the first hung one, leaking
        # precisely the daemon threads this assert exists to prevent.
        assert not hung, (
            f'serving thread(s) {hung} did not stop within the 5s teardown '
            'bound -- event loop stop is hung or the server task is stuck in a '
            'non-cancellable await (task 2741)'
        )


@pytest.fixture
def serve_escalation_mcp():
    """FUNCTION-scoped ``serve_escalation_mcp`` -- the default, and the one
    ``test_legibility_census_escalation_e2e.py`` takes.

    Every server started inside one test is torn down at the END OF THAT TEST.
    That is worth keeping (task 3736 review): the alternative -- letting a
    module's servers accumulate and stop together -- leaves N daemon threads,
    event loops and listening sockets live for the rest of the module, and
    attributes the bounded-join hung-thread assert to the MODULE rather than to
    the test that actually hung, which is precisely the failure mode task
    2741's assert exists to make legible.

    Body: ``_serve_escalation_mcp_impl`` (see it for the full contract).
    """
    yield from _serve_escalation_mcp_impl()


@pytest.fixture(scope='module')
def serve_escalation_mcp_module():
    """MODULE-scoped ``serve_escalation_mcp`` -- for consumers that must share
    ONE server across a whole module.

    Its existence is a scope contract, not a tuning choice: pytest forbids a
    fixture from depending on a NARROWER-scoped one, so a module-scoped
    ``http_server`` cannot request the function-scoped fixture above at all
    (``ScopeMismatch``) -- which is exactly why
    ``test_capability_guard_http.py`` and ``test_status_authority_gate.py``
    each used to carry their own copy of this machinery. Both now adapt over
    this fixture instead (task 3736, INV-5).

    Request this ONLY from a module- or session-scoped consumer. A
    function-scoped test that takes it gets deferred teardown (all of the
    module's servers stopping together at module end) for no benefit; take
    ``serve_escalation_mcp`` instead.

    Body: the SAME ``_serve_escalation_mcp_impl`` generator -- the scope is the
    only difference, and the two must never grow separate bodies. That is a
    review-time invariant, deliberately NOT pinned by a test: the only checks
    available are on the helper's name and the wrapper's bytecode shape, which
    go red on a pure rename while still passing a wrapper that both delegates
    AND carries a duplicated copy (task 3736 review). Keep both fixture bodies
    one-line ``yield from`` delegates.
    """
    yield from _serve_escalation_mcp_impl()


@pytest.fixture
def escalation_conftest():
    """This conftest module itself, for tests that need its RAW attributes.

    The fixtures above auto-resolve by name, but some of this module's
    contents are not fixtures at all (``_mcp_handshake_ready``,
    ``_free_escalation_port``) and some tests need the UNDECORATED generator
    behind ``serve_escalation_mcp`` via ``__wrapped__``, so that one server's
    startup and teardown can be observed in isolation. Handing the module over
    through a fixture is how those tests reach it; see
    ``test_serve_escalation_mcp_fixture.py``.

    ``sys.modules[__name__]`` rather than a bare ``import conftest`` in the
    test module, and the two are NOT interchangeable. A module's ``__name__``
    is by definition the key it is registered under in ``sys.modules``, so this
    lookup returns THIS module whatever name pytest assigned it. That matters
    because under the repo-wide ``--import-mode=importlib`` addopts pytest
    names a conftest BY COLLISION: measured, this file is ``'conftest'`` under
    ``cd escalation && pytest tests/...`` but ``'escalation.tests.conftest'``
    once the repo-root ``conftest.py`` shim claims the bare name first (any
    repo-root multi-package run). A bare ``import conftest`` asks for a name
    another package's conftest may already own, and binds whatever won it.

    The property callers actually rely on -- and the reason this is a fixture
    and not a second import of the same file -- is that ``sys.modules[__name__]``
    is the LIVE module object pytest itself loaded, so ``monkeypatch.setattr``
    against it really does patch what ``_serve_escalation_mcp_impl``'s body
    reads. That is pinned behaviourally rather than by introspection: the
    startup-failure and readiness-bound tests in
    ``test_serve_escalation_mcp_fixture.py`` monkeypatch
    ``_free_escalation_port`` / ``_READY_TIMEOUT_S`` here and observe the
    fixture pick both up, which no second import could satisfy.

    Function-scoped deliberately -- it holds no state and is cheap, so it
    imposes no scope floor on the tests that request it.
    """
    return sys.modules[__name__]
