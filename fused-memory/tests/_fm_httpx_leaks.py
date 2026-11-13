"""Per-test teardown tracking and draining of httpx.AsyncClient instances
their owners never close.

Lives outside conftest.py, imported by name, under the convention stated in
_fm_helpers' module docstring.
"""

import asyncio
import contextlib
import functools
import warnings
import weakref

import httpx

# ---------------------------------------------------------------------------
# Leaked async httpx client reaping (task 4412)
#
# Sibling of `_fm_helpers.py::reap_leaked_ticket_workers`, and of
# orchestrator/tests/_orch_helpers.py::reap_leaked_aiosqlite_connections
# (task 2413): a per-test teardown drain for a resource whose owner never
# closes it, written here rather than at the ~40 call sites across 7 test
# modules that leak one.
# ---------------------------------------------------------------------------

#: Every httpx.AsyncClient constructed since track_async_httpx_clients() ran.
#: A WeakSet so tracking never keeps a client alive past its natural lifetime —
#: the reaper only ever sees clients something else is still holding.
_TRACKED_ASYNC_HTTPX_CLIENTS: weakref.WeakSet = weakref.WeakSet()

#: Per-client bound on the drain's aclose(). The reaper runs in EVERY test's
#: teardown, so an aclose() that never returns must cost that test a bounded
#: pause rather than hang it until pytest-timeout kills the whole xdist worker
#: (see the timeout notes in fused-memory/pyproject.toml). A module-level knob
#: rather than a literal so the bound itself is testable in ~50ms — see
#: test_async_httpx_leak_isolation.test_reap_is_bounded_when_aclose_hangs.
ASYNC_HTTPX_ACLOSE_TIMEOUT = 10.0


#: Stamped on the installed wrapper so a second install is a no-op. Named on
#: the FUNCTION rather than kept in a module global because the thing being
#: guarded is the state of ``httpx.AsyncClient.__init__`` itself: a test that
#: restores a pristine ``__init__`` (see
#: test_async_httpx_leak_isolation.test_track_async_httpx_clients_is_idempotent)
#: must be able to re-install, which a module-level "already ran" flag would
#: wrongly refuse.
_TRACK_SENTINEL = '_df_tracks_async_httpx_clients'


def track_async_httpx_clients() -> bool:
    """Record every ``httpx.AsyncClient`` built from now on, for later reaping.

    Wraps ``httpx.AsyncClient.__init__`` — ONE patch point that catches every
    library shipping the leaky wrapper, present and future. Measured in this
    worktree: ``openai._base_client.AsyncHttpxClientWrapper`` and
    ``anthropic._base_client.AsyncHttpxClientWrapper`` both have MRO
    ``(Wrapper -> _DefaultAsyncHttpxClient -> httpx.AsyncClient)`` and chain to
    ``httpx.AsyncClient.__init__``, so patching only the base tracked all of
    ``AsyncOpenAI``, ``AsyncAnthropic``, a ``graphiti_core`` ``OpenAIClient``
    and a bare ``httpx.AsyncClient``. That needs no per-library import list to
    keep in sync and cannot break on import order.

    Instances are recorded AFTER the original ``__init__`` returns, so a
    constructor that raises leaves nothing half-built in the WeakSet.

    Idempotent, and that is load-bearing rather than hygiene: ``conftest``'s
    ``pytest_configure`` installs the hook at session start and each test in
    ``test_async_httpx_leak_isolation.py`` calls it again, so an unguarded
    re-wrap would nest one extra frame around the saved original per call for
    the rest of the session.

    Returns:
        True if this call installed the hook, False if it was already present.
    """
    original = httpx.AsyncClient.__init__
    if getattr(original, _TRACK_SENTINEL, False):
        return False

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _TRACKED_ASYNC_HTTPX_CLIENTS.add(self)

    # functools.wraps copies __dict__ from the wrapped function, so stamp the
    # sentinel AFTER it — otherwise re-installing over an already-stamped
    # original would carry the flag straight onto the new wrapper anyway.
    setattr(__init__, _TRACK_SENTINEL, True)
    httpx.AsyncClient.__init__ = __init__
    return True


def _leaked_async_httpx_clients() -> list:
    """Return the tracked clients that are unclosed AND resurrect-capable.

    "Resurrect-capable" means ``type(c)`` defines ``__del__`` — the finaliser
    that reschedules ``aclose()`` onto a running loop and so produces the
    flake. Measured: True for both libraries' ``AsyncHttpxClientWrapper``,
    False for a bare ``httpx.AsyncClient`` (``'__del__' not in
    httpx.AsyncClient.__dict__``). Deriving the predicate from the MECHANISM
    rather than allow-listing ``openai``/``anthropic`` by name means any
    future dependency shipping the same finaliser is covered without a code
    change, and — just as important — that a bare async httpx client owned by
    an unrelated fixture is left strictly alone. Closing everything tracked
    would give the reaper a much larger blast radius than a flake fix earns.

    Also the cheap predicate ``conftest``'s SYNC autouse fixture checks before
    deciding whether to spin up an event loop at all, so the ~14100-of-14147
    tests that leak nothing pay only a WeakSet scan.

    NO NOTION OF OWNERSHIP OR AGE, deliberately: every tracked, open,
    resurrect-capable client is selected, whoever built it and whenever. A
    client built by a fixture scoped WIDER than ``function`` would therefore
    be closed at the first test's teardown, out from under its owner. See the
    CONSTRAINT block above the two autouse arms in ``conftest.py`` for why
    scoping the reap to the current test was considered and rejected (it opens
    the inverse hole), and for the warning that makes the trap discoverable if
    anyone ever walks into it.
    """
    return [
        client
        for client in list(_TRACKED_ASYNC_HTTPX_CLIENTS)
        if not client.is_closed and hasattr(type(client), '__del__')
    ]


async def reap_leaked_async_httpx_clients() -> int:
    """Close any tracked ``httpx.AsyncClient`` that its owner never closed.

    Task 4412 — fix for the fused-memory caplog flake CLASS where an unclosed
    ``openai``/``anthropic`` client emits an ERROR record onto the root
    ``asyncio`` logger from inside an unrelated later test's caplog window.

    ROOT CAUSE: ``openai._base_client.AsyncHttpxClientWrapper`` and its
    ``anthropic`` twin define::

        def __del__(self) -> None:
            if self.is_closed:
                return
            try:
                asyncio.get_running_loop().create_task(self.aclose())
            except Exception:
                pass

    An ``AsyncOpenAI``/``AsyncAnthropic`` that is never closed is GC-finalised
    at a nondeterministic point; if a loop happens to be running, ``__del__``
    RESURRECTS the object as ``create_task(self.aclose())``. That Task's
    coroutine is httpx's inherited ``AsyncClient.aclose`` — the
    ``coro=<AsyncClient.aclose() ...>`` in the symptom. Its ``aclose()`` then
    hits a connection pool bound to an already-closed loop and raises
    ``RuntimeError('Event loop is closed')``; nobody retrieves it, so
    ``Task.__del__`` logs ``Task exception was never retrieved`` at ERROR and
    it is blamed on whichever test's caplog is open at the time.

    MEASURED (this worktree, full default-lane suite, 14147 passed): 40
    wrapper instances constructed, ALL 40 GC-finalised, ZERO closed — i.e. 40
    chances per suite run for ``__del__`` to resurrect. They come from
    ``graphiti_core``'s ``OpenAIClient`` / ``OpenAIGenericClient`` /
    ``OpenAIEmbedder`` / ``OpenAIRerankerClient``, each of which mints its own
    ``AsyncOpenAI`` in ``__init__`` and exposes no ``close()`` — so the tests
    that trigger them have no call-site fix available.

    FIX: close every tracked client at each test's teardown boundary.
    ``__del__``'s own ``if self.is_closed: return`` then short-circuits, so the
    resurrect path is unreachable and the ERROR record can never be emitted.

    Best-effort and bounded, and a cheap no-op for the (vast majority of)
    tests that leak nothing. Only ``asyncio.TimeoutError`` and ``RuntimeError``
    are suppressed: a client whose pool belongs to an already-closed loop
    raises ``RuntimeError`` on ``aclose()`` and must not fail an innocent test.
    A genuine bug inside the reaper itself — or a ``CancelledError`` targeting
    the reaper's own task — is deliberately left to propagate rather than
    masked by a blanket ``except Exception``
    (loud-over-silent-degradation). Both halves of that robustness contract
    are pinned behaviourally rather than left as prose:
    ``test_reap_survives_a_client_whose_aclose_raises`` (the suppressed
    ``RuntimeError``, and that a client which failed to close is NOT counted)
    and ``test_reap_is_bounded_when_aclose_hangs`` (the
    ``ASYNC_HTTPX_ACLOSE_TIMEOUT`` bound, exercised at 50ms).

    MAINTENANCE NOTE: this relies on third-party internals that are not part
    of any public API — the ``AsyncHttpxClientWrapper.__del__`` finaliser in
    ``openai``/``anthropic``, and those wrappers chaining to
    ``httpx.AsyncClient.__init__`` so a single base-class hook sees them.
    Verified against httpx 0.28.1 / openai 2.31.0 / anthropic 0.92.0. If a
    future release drops the finaliser or stops chaining, this helper degrades
    to a no-op (returns 0) rather than raising — the reaping is then lost. That
    loss is NOT silent: ``test_async_httpx_leak_isolation.py``'s
    ``test_a_reaped_client_del_schedules_no_aclose_task`` asserts an UNCLOSED
    client's ``__del__`` schedules exactly 1 ``AsyncClient.aclose()`` task, so
    a dropped finaliser fails that test loudly (mirrors
    ``reap_leaked_aiosqlite_connections``'s MAINTENANCE NOTE and its
    ``PytestUnhandledThreadExceptionWarning`` backstop, task 4075). Re-verify
    this helper against both libraries' ``_base_client.py`` whenever the
    pinned versions change.

    Returns:
        The number of clients this call actually closed (``is_closed``
        confirmed post-drain) — a client that fails to close within the
        bounded timeout is not counted.
    """
    reaped = 0
    for client in _leaked_async_httpx_clients():
        with contextlib.suppress(asyncio.TimeoutError, RuntimeError):
            await asyncio.wait_for(
                client.aclose(), timeout=ASYNC_HTTPX_ACLOSE_TIMEOUT,
            )
        if client.is_closed:
            reaped += 1
    return reaped


def _warn_if_drain_closed_a_foreign_client(preexisting: weakref.WeakSet) -> None:
    """Warn if the drain closed a client that pre-dated the current test.

    *preexisting* is the snapshot ``conftest``'s SYNC autouse arm takes at
    SETUP: the clients that were already tracked, open and resurrect-capable
    before the test started.
    Any of them found closed at teardown was closed by this test or by the
    drain — either way, something owns a client that outlives a single test,
    which the drain cannot support (see the CONSTRAINT block above the two
    autouse arms in ``conftest.py``, and ``_leaked_async_httpx_clients``
    above for why the selection predicate has no notion of ownership).

    A warning rather than a failure: the drain closing a foreign client is a
    design violation by the FIXTURE, not by the test that happens to be
    finishing, and failing that test would reproduce the very
    blame-the-innocent-test pattern being removed. The warning is greppable in
    CI output and names the fix.
    """
    closed = [client for client in preexisting if client.is_closed]
    if not closed:
        return
    warnings.warn(
        f'async-httpx drain closed {len(closed)} openai/anthropic client(s) '
        f'that existed BEFORE this test started. The drain has no notion of '
        f'ownership: it closes every tracked, open, resurrect-capable client '
        f'at every test teardown. If a fixture scoped wider than `function` '
        f'(or a module-level cache) owns one, it has just been closed out from '
        f'under its owner and will fail later with "Cannot send a request, as '
        f'the client has been closed". Move it to function scope (task 4412).',
        stacklevel=2,
    )
