"""Tests for reap_leaked_async_httpx_clients() and its autouse fixture wiring.

Task 4412 — retire the fused-memory caplog flake CLASS at its root by closing
the leaked ``openai``/``anthropic`` ``AsyncHttpxClientWrapper`` that produces
``Task exception was never retrieved`` / ``coro=<AsyncClient.aclose() ...>``
ERROR records on the root ``asyncio`` logger, which then land in whichever
unrelated test's ``caplog`` window happens to be open.

MECHANISM. ``openai._base_client`` and ``anthropic._base_client`` both define::

    class AsyncHttpxClientWrapper(DefaultAsyncHttpxClient):   # -> httpx.AsyncClient
        def __del__(self) -> None:
            if self.is_closed:
                return
            try:
                asyncio.get_running_loop().create_task(self.aclose())
            except Exception:
                pass

So an ``AsyncOpenAI`` / ``AsyncAnthropic`` that is never closed gets
GC-finalised at a nondeterministic point; if a loop happens to be running,
``__del__`` RESURRECTS the object as ``create_task(self.aclose())``. That
Task's coroutine qualname is ``AsyncClient.aclose`` (inherited from httpx) —
exactly the ``coro=<AsyncClient.aclose() ...>`` in the symptom. Its
``aclose()`` then hits a connection pool bound to an already-closed loop and
raises ``RuntimeError('Event loop is closed')``. Nobody retrieves it, so
``Task.__del__`` logs ``Task exception was never retrieved`` at ERROR on the
root ``asyncio`` logger, and it is attributed to an innocent later test.

The fix rides on ``__del__``'s own ``if self.is_closed: return``
short-circuit: closing every tracked client at each test's teardown makes the
resurrect path unreachable, so the ERROR record can never be emitted at all.

Design decisions (mirroring orchestrator/tests/test_aiosqlite_leak_isolation.py,
the house template for a leak-drain defence):
- The helper's own contracts are pinned by self-contained unit tests (not a
  cross-test polluter/victim pair), so they are deterministic under
  ``-n auto --dist loadgroup`` without requiring an ``xdist_group`` tag. The
  ONE exception is the teardown-affinity pair at the bottom of this module
  (``test_aaa_leaked_client_records_its_closing_loop`` /
  ``test_aab_the_leak_was_closed_in_its_own_test_loop``): it measures which
  event loop actually closed a leaked client, which cannot be observed from
  inside the test that leaked it, so that pair DOES require
  ``@pytest.mark.xdist_group('async_httpx_reap_ordering')`` to co-locate both
  arms on one worker in collection order.
- Each test installs the tracking hook itself and reaps any residual leaked
  client from a *prior* test before measuring — the same flush-before-measure
  pattern that module uses for ``reap_leaked_aiosqlite_connections()``.

===========================================================================
ACCEPTANCE EVIDENCE (recorded here so the next reader need not re-derive it)
===========================================================================

HOW THE SITE WAS FOUND. The leaked object is not a bare ``httpx.AsyncClient``
— ``grep -rn 'httpx.AsyncClient(' fused-memory/`` has ZERO hits under ``src/``
and ``tests/``, which is why several earlier attempts missed it. Instrumenting
``AsyncHttpxClientWrapper.__init__`` with ``traceback.format_stack()`` gave the
allocation path::

    tests/test_graphiti_llm_client_construction.py, line 53, in test_returns_openai_client
    src/fused_memory/backends/graphiti_client.py, line 238, in build_llm_client
    .venv/.../graphiti_core/llm_client/openai_client.py, line 61, in __init__
    .venv/.../openai/_client.py, line 617, in __init__
    .venv/.../openai/_base_client.py, line 1501, in __init__

``graphiti_core``'s ``OpenAIClient`` / ``OpenAIGenericClient`` /
``OpenAIEmbedder`` / ``OpenAIRerankerClient`` each mint their own
``AsyncOpenAI`` in ``__init__`` and expose no ``close()``, so the tests that
trigger them have no call-site fix available — hence a teardown drain.

WHERE THE 40 COME FROM (per-module census of the wrappers built by the
default-lane suite)::

    16  tests/test_graphiti_llm_client_construction.py   (build_llm_client)
     8  tests/test_local_endpoint_base_url_integration.py (real round-trips)
     8  tests/test_startup_identity_scan.py              (embedder + reranker)
     2  tests/test_per_group_client_cache.py
     2  tests/test_openai_responses_preflight.py
     2  tests/test_operational_routing_boundary_matrix.py
     1  tests/test_task_curator.py

BEFORE / AFTER. Measured as a control/treatment A/B over those 7 modules with
one throwaway probe (control = this reaper neutered; 376 passed in both arms).
The probe wraps ``AsyncHttpxClientWrapper.__del__`` and records, for each
finalisation of an UNCLOSED wrapper, whether a loop was running at that moment
— i.e. whether ``__del__`` actually reached ``create_task(self.aclose())``.
Each such RESURRECTION is exactly one potential ERROR record::

                              CONTROL   TREATMENT
    wrappers built                 40          40
    ever closed                     2          22
    GC-finalised                   40          40
    RESURRECTIONS                   2  ->        0
    finalised, no loop running     38          18

Both control resurrections were attributed to one async test,
``test_startup_identity_scan.py::TestInitializeSkipMaintenance::test_skip_maintenance_true_skips_both_blocks``.

NOTE, correcting the task's own analysis: the 18 wrappers still finalised
unclosed under treatment are the SYNC cohort
(``test_graphiti_llm_client_construction.py``). They die by refcount at test
exit, before any teardown fixture runs, and with no loop running — so
``__del__``'s ``get_running_loop()`` raises, the ``except`` swallows it, and no
task is created. They are structurally incapable of producing the flake, and
the sync autouse arm is therefore defence-in-depth (against a client held past
its test body by a mock's ``call_args`` or a reference cycle, and collected
later inside a running loop) rather than same-run coverage of that cohort.

FULL SUITE, three orderings (``-n auto`` / ``-n 8`` / ``-n 16``, all
``--dist loadgroup``): 14564 passed, 2 skipped, 0 failed in each. Identical
census every time — 43 wrappers built (the 40 above plus 3 built by this
module), 25 closed, 1 resurrection, and that one resurrection is
``test_a_reaped_client_del_schedules_no_aclose_task`` DELIBERATELY invoking
``__del__`` on an unclosed client to prove the mechanism. Grepping all three
runs' output for ``Task exception was never retrieved``, ``AsyncClient.aclose``
and ``Event loop is closed`` returned 0 hits each.
"""
from __future__ import annotations

import asyncio
import contextlib

import httpx
import openai
import pytest
from _fm_helpers import (
    _TRACK_SENTINEL,
    reap_leaked_async_httpx_clients,
    track_async_httpx_clients,
)

#: Whether the tracking hook was ALREADY installed when this module was
#: imported — i.e. at collection time, before any test here had a chance to
#: install it itself. Snapshotting at import rather than inside the test body
#: is what makes test_tracking_hook_is_installed_by_pytest_configure
#: order-independent (and therefore honestly red when the wiring is absent):
#: read inside a test, it would be satisfied by whichever sibling test in this
#: module happened to run first and call track_async_httpx_clients().
_SENTINEL_AT_IMPORT = getattr(httpx.AsyncClient.__init__, _TRACK_SENTINEL, False)

#: The two autouse teardown fixtures conftest must declare.
_SYNC_FIXTURE = '_reap_leaked_async_httpx_clients_sync'
_ASYNC_FIXTURE = '_reap_leaked_async_httpx_clients'

#: Cross-test channel for the teardown-affinity pair at the bottom of this
#: module. The polluter records the loop it ran in and the loop its leaked
#: client was actually closed in; the victim, which runs after the polluter's
#: teardown has finished, compares them. A module global is the only way to
#: observe a teardown that has by definition already completed.
_REAP_ORDERING_PROBE: dict = {}


@pytest.mark.asyncio
async def test_reap_closes_a_leaked_openai_client():
    """reap_leaked_async_httpx_clients() closes a client nothing else closed.

    ``AsyncOpenAI`` mints an ``AsyncHttpxClientWrapper`` in its constructor and
    exposes no ``close()`` on the graphiti_core wrappers that build it, so a
    test that constructs one has no way to clean it up — it is reaped here or
    not at all.
    """
    track_async_httpx_clients()
    # Flush any client a prior test left behind before measuring.
    await reap_leaked_async_httpx_clients()

    client = openai.AsyncOpenAI(api_key='test-key')
    inner = client._client
    assert inner.is_closed is False, 'expected a freshly-built AsyncOpenAI to hold an open httpx client'

    reaped = await reap_leaked_async_httpx_clients()
    assert reaped == 1, (
        f'expected the reaper to close exactly the 1 leaked client this test '
        f'built, got {reaped}'
    )
    # Independent observable: the specific client's own state, not just the
    # aggregate count returned above.
    assert inner.is_closed is True, 'reap must aclose() the leaked client'


@pytest.mark.asyncio
async def test_a_reaped_client_del_schedules_no_aclose_task():
    """A reaped client's ``__del__`` schedules no ``AsyncClient.aclose()`` task.

    The causal pin for the whole flake class: it measures the resurrection
    itself rather than the reaper's bookkeeping. An UNCLOSED wrapper's
    ``__del__`` creates exactly one task — the ``coro=<AsyncClient.aclose()>``
    named in the symptom — whereas after the reaper has closed it, ``__del__``
    short-circuits on ``if self.is_closed: return`` and creates none. That
    0 is why the ERROR record can no longer be emitted.

    It is also the tripwire for a third-party bump that drops the finaliser:
    if a future ``openai`` release stops resurrecting, the 1-task assertion
    fails loudly instead of the reaper silently protecting against nothing.
    """
    track_async_httpx_clients()
    await reap_leaked_async_httpx_clients()

    client = openai.AsyncOpenAI(api_key='test-key')
    inner = client._client

    # ARM 1 — unclosed: __del__ resurrects the object as a pending aclose task.
    before = set(asyncio.all_tasks())
    # `inner` is STATICALLY an httpx.AsyncClient, which declares no __del__ — the
    # very asymmetry test_reap_leaves_a_bare_httpx_client_open asserts on. At RUNTIME
    # it is openai's AsyncHttpxClientWrapper subclass, which does define one, and
    # calling that finaliser explicitly is the whole point of this test.
    inner.__del__()  # pyright: ignore[reportAttributeAccessIssue]
    resurrected = set(asyncio.all_tasks()) - before
    assert len(resurrected) == 1, (
        f'expected an UNCLOSED AsyncHttpxClientWrapper.__del__ to schedule '
        f'exactly 1 aclose() task (the flake source), got {len(resurrected)}. '
        f'If openai dropped the __del__ finaliser, this reaper now guards '
        f'nothing — re-derive the mechanism (task 4412).'
    )
    (task,) = resurrected
    assert 'AsyncClient.aclose' in repr(task.get_coro()), (
        f"the resurrected task's coroutine must be httpx's AsyncClient.aclose "
        f'— the literal coro named in the flake symptom — got '
        f'{task.get_coro()!r}'
    )
    # Cancel rather than await: the coroutine has not started, so cancelling
    # leaves the client UNCLOSED (which is what the reaper below must fix)
    # while leaving no pending task behind for the loop teardown to complain
    # about.
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert inner.is_closed is False, 'cancelling the resurrected task must not close the client'

    # ARM 2 — reaped: __del__ short-circuits and schedules nothing.
    await reap_leaked_async_httpx_clients()
    assert inner.is_closed is True, 'reap must aclose() the leaked client'

    before = set(asyncio.all_tasks())
    inner.__del__()  # pyright: ignore[reportAttributeAccessIssue]  # see ARM 1
    assert set(asyncio.all_tasks()) - before == set(), (
        'a CLOSED AsyncHttpxClientWrapper.__del__ must schedule no task — '
        'this is the short-circuit the whole fix rides on'
    )


@pytest.mark.asyncio
async def test_reap_leaves_a_bare_httpx_client_open():
    """A bare ``httpx.AsyncClient`` is not resurrect-capable, so reap leaves it.

    The blast-radius contract. Only a class defining ``__del__`` can resurrect
    itself via ``create_task(self.aclose())`` and emit the ERROR record;
    measured, ``'__del__' not in httpx.AsyncClient.__dict__``. A plain async
    httpx client therefore cannot produce the flake, and closing one would
    mean the reaper could yank a client an unrelated fixture still intends to
    use — the wrong trade for a flake fix. Deriving the predicate from the
    ``__del__`` mechanism (rather than allow-listing ``openai``/``anthropic``
    by name) also means any future library shipping the same finaliser is
    covered without a code change here.
    """
    track_async_httpx_clients()
    await reap_leaked_async_httpx_clients()

    assert '__del__' not in httpx.AsyncClient.__dict__, (
        'httpx.AsyncClient now defines __del__, so a bare client CAN resurrect '
        'itself and this exclusion is no longer safe — re-derive the reaper '
        'predicate in _fm_helpers.reap_leaked_async_httpx_clients (task 4412).'
    )

    bare = httpx.AsyncClient()
    try:
        reaped = await reap_leaked_async_httpx_clients()
        assert bare.is_closed is False, (
            'the reaper must leave a bare httpx.AsyncClient open: it cannot '
            'emit the ERROR record, and closing it would break whoever owns it'
        )
        assert reaped == 0, (
            f'nothing resurrect-capable was leaked, so the reaper should have '
            f'closed nothing, got {reaped}'
        )
    finally:
        await bare.aclose()


def test_track_async_httpx_clients_is_idempotent(monkeypatch):
    """Installing the tracking hook twice does not double-wrap ``__init__``.

    Load-bearing, not hygiene: ``pytest_configure`` installs the hook at
    session start AND every test in this module calls it again, so without a
    guard each call would nest another wrapper around the saved original —
    one extra frame per call, for the whole session.

    ``monkeypatch`` restores a pristine (unhooked) ``__init__`` first so this
    measures the FIRST install regardless of whether ``pytest_configure``
    already ran one, and puts the session's hook back at teardown.
    """
    pristine = getattr(httpx.AsyncClient.__init__, '__wrapped__', httpx.AsyncClient.__init__)
    monkeypatch.setattr(httpx.AsyncClient, '__init__', pristine)

    assert track_async_httpx_clients() is True, (
        'the first install on a pristine httpx.AsyncClient must report that it '
        'installed the hook'
    )
    installed = httpx.AsyncClient.__init__

    assert track_async_httpx_clients() is False, (
        'a second install must report that the hook was already present'
    )
    assert httpx.AsyncClient.__init__ is installed, (
        'a second install must not re-wrap httpx.AsyncClient.__init__ — each '
        're-wrap nests another frame around the saved original for the rest '
        'of the session'
    )


@pytest.mark.asyncio
async def test_reap_is_safe_when_nothing_is_leaked():
    """reap_leaked_async_httpx_clients() returns 0 when nothing is leaked."""
    track_async_httpx_clients()
    # First reap to flush any residual leak from a prior test.
    await reap_leaked_async_httpx_clients()

    count = await reap_leaked_async_httpx_clients()
    assert count == 0, (
        f'Expected 0 leaked async httpx clients but got {count}. '
        'A prior test may have leaked a client.'
    )


@pytest.mark.asyncio
async def test_reap_ignores_an_already_closed_client():
    """A client that was properly closed is not counted or touched by reap.

    Safety-net assertion: reap must not double-close (and must not raise on) a
    client a test already cleaned up correctly.
    """
    track_async_httpx_clients()
    await reap_leaked_async_httpx_clients()

    client = openai.AsyncOpenAI(api_key='test-key')
    await client.close()
    assert client._client.is_closed is True, 'AsyncOpenAI.close() should have closed the httpx client'

    count = await reap_leaked_async_httpx_clients()
    assert count == 0, 'An already-closed client must not be counted as reaped'


# ---------------------------------------------------------------------------
# conftest wiring (task 4412)
#
# The helper alone retires nothing: it changes the suite's behaviour only once
# it runs at EVERY test's teardown. These four guards are what keep the wiring
# from being silently dropped by a later refactor — mirroring
# orchestrator/tests/test_aiosqlite_leak_isolation.py::test_autouse_reap_fixture_is_active.
# ---------------------------------------------------------------------------


def test_autouse_async_reap_fixture_is_active(request):
    """The async drain fixture is wired as an autouse teardown fixture.

    Deterministic and worker-placement-agnostic: an autouse fixture appears in
    ``request.fixturenames`` for every test; an absent one does not. This is a
    behavioural assertion that the reaper is applied to arbitrary tests, not a
    docstring lint.
    """
    assert _ASYNC_FIXTURE in request.fixturenames, (
        f'{_ASYNC_FIXTURE} must be registered as an autouse pytest_asyncio '
        f'fixture in conftest.py so every ASYNC test closes its leaked '
        f'openai/anthropic clients inside its own still-open event loop, '
        f'before AsyncHttpxClientWrapper.__del__ can resurrect them (task 4412).'
    )


def test_autouse_sync_reap_fixture_is_active(request):
    """The sync-test drain fixture is wired as an autouse teardown fixture.

    BOTH arms are required because the leak cohorts are mixed. Measured: the
    largest single cohort — ``test_graphiti_llm_client_construction.py``'s
    ``test_returns_openai_client``, 16 of the 40 leaked clients — is a SYNC
    ``def test_``, while ``test_startup_identity_scan.py``,
    ``test_per_group_client_cache.py``,
    ``test_local_endpoint_base_url_integration.py`` and
    ``test_openai_responses_preflight.py`` are ``async def``. A
    ``pytest_asyncio`` autouse fixture only covers async tests, so that arm
    alone would silently miss 40% of the leaks.
    """
    assert _SYNC_FIXTURE in request.fixturenames, (
        f'{_SYNC_FIXTURE} must be registered as an autouse fixture in '
        f'conftest.py so SYNC tests — which the pytest_asyncio arm never runs '
        f'for — also drain their leaked openai/anthropic clients (task 4412).'
    )


def test_tracking_hook_is_installed_by_pytest_configure():
    """``pytest_configure`` installs the tracking hook at session start.

    Timing is the contract, not merely installation: a hook installed later
    (say, lazily by the first test that needs it) would miss every client
    constructed at import time during collection. The snapshot this asserts on
    is taken when THIS MODULE is imported — i.e. during collection, before any
    test in it has run — so it cannot be satisfied by a sibling test in this
    module having called ``track_async_httpx_clients()`` first.
    """
    assert _SENTINEL_AT_IMPORT is True, (
        'httpx.AsyncClient.__init__ did not carry the tracking sentinel when '
        'this module was imported, so conftest.pytest_configure is not calling '
        'track_async_httpx_clients(). Without it, clients built during '
        'collection are never tracked and so are never reaped (task 4412).'
    )


# ---------------------------------------------------------------------------
# Teardown affinity (task 4412) — the ONE cross-test pair in this module.
#
# The two-arm autouse split in conftest exists for exactly one property: an
# async test's leaked clients must be closed inside THAT TEST'S OWN still-open
# event loop, because their connection pool has affinity to it. That property
# is invisible from inside the test that leaks (the close happens later, at
# teardown), so it takes a polluter/victim pair — and therefore an
# ``xdist_group`` tag so ``-n auto --dist loadgroup`` keeps both arms on one
# worker in collection order.
#
# The names are ``test_aaa_``/``test_aab_`` so the pair stays adjacent and in
# order under any name-sorting collector, not merely under definition order.
# ---------------------------------------------------------------------------


@pytest.mark.xdist_group('async_httpx_reap_ordering')
@pytest.mark.asyncio
async def test_aaa_leaked_client_records_its_closing_loop():
    """Polluter: leak a client that records which loop eventually closes it.

    Deliberately never closes the client — the whole point is to let the
    conftest teardown drain do it, and to capture where. The recording wraps
    the INSTANCE attribute ``inner.aclose`` rather than the class, because the
    reaper calls ``client.aclose()`` — an instance lookup
    (``_fm_helpers.reap_leaked_async_httpx_clients``) — so an instance-level
    wrapper is both sufficient and free of cross-test blast radius.

    Asserts nothing itself: the measurement it takes is only observable after
    its own teardown has run, so the assertion lives in the victim below.
    """
    track_async_httpx_clients()
    # Flush any client a prior test left behind, so the only leak in flight is
    # the one this test is about to build (the module's flush-before-measure
    # pattern).
    await reap_leaked_async_httpx_clients()

    _REAP_ORDERING_PROBE.clear()
    _REAP_ORDERING_PROBE['test_loop'] = asyncio.get_running_loop()

    client = openai.AsyncOpenAI(api_key='test-key')
    inner = client._client
    original_aclose = inner.aclose

    async def _recording_aclose(*args, **kwargs):
        _REAP_ORDERING_PROBE.setdefault('closing_loop', asyncio.get_running_loop())
        return await original_aclose(*args, **kwargs)

    inner.aclose = _recording_aclose
    # Hold a strong reference so the client survives to teardown rather than
    # being refcount-finalised at test exit (which would close nothing and
    # make the victim's not-None guard fire instead of the affinity assertion).
    _REAP_ORDERING_PROBE['client'] = client


@pytest.mark.xdist_group('async_httpx_reap_ordering')
def test_aab_the_leak_was_closed_in_its_own_test_loop():
    """Victim: the polluter's leaked client was closed in the polluter's loop.

    The behavioural pin for the two-arm teardown ordering, replacing an
    earlier AST meta-test that read conftest's declaration order. That test
    passed GREEN while the RUNTIME order was the exact inverse of what it
    claimed, because declaration order does not in fact decide it:
    ``pytest_asyncio`` 1.x async fixtures acquire an event-loop dependency
    that reorders them relative to plain autouse fixtures. Only an observation
    of which loop did the closing can tell the two apart.

    A cross-loop close is not merely untidy: the client's connection pool is
    bound to the loop that opened it, so closing it from the sync arm's
    throwaway ``asyncio.run`` loop is the very ``RuntimeError('Event loop is
    closed')`` path the whole task exists to remove — most sharply for
    ``test_local_endpoint_base_url_integration.py``, the one measured cohort
    doing real I/O.

    Measured against python 3.13.9 / pytest 9.0.3 / pytest-asyncio 1.3.0
    (``asyncio_mode=strict``); a bump to any of those is exactly when this
    pair should be re-run.
    """
    try:
        closing_loop = _REAP_ORDERING_PROBE.get('closing_loop')
        test_loop = _REAP_ORDERING_PROBE.get('test_loop')

        # Guard FIRST: if the pair got split across xdist workers (or the
        # polluter was deselected), both reads are None and the affinity
        # assertion below would pass vacuously on `None is None`.
        assert closing_loop is not None, (
            'no aclose() was observed for the client leaked by '
            'test_aaa_leaked_client_records_its_closing_loop, so this pair '
            'measured nothing. Either the conftest drain never ran, or the '
            "pair was split across xdist workers — check both arms still "
            "carry @pytest.mark.xdist_group('async_httpx_reap_ordering') "
            'and remain adjacent in collection order (task 4412).'
        )
        assert closing_loop is test_loop, (
            'the leaked client was closed in a FOREIGN event loop '
            f'({closing_loop!r}), not the test\'s own ({test_loop!r}). The '
            'async autouse arm must tear down BEFORE the sync arm, so an '
            "async test's clients are closed inside its own still-open loop "
            'rather than by the sync arm\'s throwaway asyncio.run loop. In '
            'conftest.py, the async arm gets that priority by REQUESTING the '
            'sync arm as a fixture argument — not by declaration order, which '
            'pytest-asyncio does not honour here (task 4412).'
        )
    finally:
        # Drop the closed loop and the leaked client rather than holding them
        # for the rest of the session.
        _REAP_ORDERING_PROBE.clear()
