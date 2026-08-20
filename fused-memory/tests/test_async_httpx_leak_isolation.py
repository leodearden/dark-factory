"""Tests for reap_leaked_async_httpx_clients() and its autouse fixture wiring.

Task 4412 — retire the fused-memory caplog flake CLASS at its root by closing
the leaked ``openai``/``anthropic`` ``AsyncHttpxClientWrapper`` that produces
``Task exception was never retrieved`` / ``coro=<AsyncClient.aclose() ...>``
ERROR records on the root ``asyncio`` logger, which then land in whichever
unrelated test's ``caplog`` window happens to be open.

MECHANISM — stated ONCE, in ``_fm_helpers.reap_leaked_async_httpx_clients``'s
docstring, next to the code that depends on it. In a line: both libraries'
``AsyncHttpxClientWrapper`` defines a ``__del__`` that RESURRECTS an unclosed
client as ``create_task(self.aclose())`` if any loop is running when it is
GC-finalised, and the fix rides on that same finaliser's
``if self.is_closed: return`` short-circuit. Read the helper before changing
anything here: it also carries the third-party version pins the whole design
rests on, and it is the ONE place to update on an ``openai``/``anthropic``
bump. (Four near-verbatim copies of that narrative were collapsed into it —
copies silently disagree the first time only one is edited, which already
happened once in-flight with the teardown-order story below.)

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
Taken against the drain's FIRST cut, i.e. before the teardown ordering was
fixed — see MEASURED TEARDOWN ORDER below; what it establishes (the reaper
takes RESURRECTIONS to zero) is independent of which arm does the closing, and
the full-suite census further down re-establishes it under the shipped order.
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

MEASURED TEARDOWN ORDER (re-measured after the ordering fix, because the first
cut of this drain had it backwards). A spy on conftest's own
``reap_leaked_async_httpx_clients`` / ``_leaked_async_httpx_clients``,
installed at ``pytest_collection_finish`` so it survives function-scoped
teardown — a ``monkeypatch`` spy would be undone before the arms run — records
per test which arm fired, in which loop::

    PRE-FIX  sync  test_returns_openai_client
               [('sync-arm-empty-check', -), ('reap', L1)]
    PRE-FIX  async test_skip_maintenance_true_skips_both_blocks
               [('sync-arm-empty-check', -), ('reap', L1), ('reap', L2)]
    POST-FIX sync  test_returns_openai_client
               [('reap', L1), ('sync-arm-empty-check', -)]
    POST-FIX async test_skip_maintenance_true_skips_both_blocks
               [('reap', L1), ('sync-arm-empty-check', -)]

Three things fall out. (1) The order SHIPPED by the first cut was
sync-arm-FIRST — the exact inverse of the declaration-order story it was
documented with. It is now async-arm-first for both sync and async tests.
(2) Pre-fix, an async test reaped TWICE in TWO DIFFERENT loops: the sync arm's
throwaway ``asyncio.run`` loop closed the clients and the async arm then found
nothing — i.e. the real-I/O cohort was being closed CROSS-LOOP, the very thing
this drain exists to avoid. Post-fix there is exactly one reap, in the test's
own loop. (3) The ``pytest_asyncio`` arm DOES fire for a plain ``def test_``
(in a throwaway loop of its own), so the sync arm is defence in depth rather
than coverage of a gap.

The ordering is bought by an explicit FIXTURE DEPENDENCY — the async arm takes
the sync arm as an argument, so the sync arm is set up first and torn down last
— NOT by declaration order, which pytest-asyncio 1.x does not honour here
because its async fixtures acquire an event-loop dependency that reorders them
against plain autouse fixtures. Pinned behaviourally by this module's
``test_aaa_leaked_client_records_its_closing_loop`` /
``test_aab_the_leak_was_closed_in_its_own_test_loop`` pair.

FULL SUITE (re-run under the shipped ordering), two distributions —
``-n auto`` and ``-n 8``, both ``--dist loadgroup``, the flake being
ordering-sensitive: 14621 passed, 2 skipped, 0 failed in each (921s / 980s).
Identical census both times — 44 wrapper instances built (the 40 above plus 4
built by this module), 27 already closed by the time they were finalised, 0
still alive and open at session end, and exactly 1 RESURRECTION, which is
``test_a_reaped_client_del_schedules_no_aclose_task`` DELIBERATELY invoking
``__del__`` on an unclosed client to prove the mechanism. (46 finalisation
EVENTS against 44 objects: that same test calls ``__del__`` explicitly twice
before the object is really collected.) The other 18 are the sync cohort dying
by refcount at test exit with no loop running — see the NOTE above; they cannot
resurrect. Grepping both runs' output for ``Task exception was never
retrieved``, ``AsyncClient.aclose`` and ``Event loop is closed`` returned 0
hits each.

Toolchain all of the above was measured against: python 3.13.9, pytest 9.0.3,
pytest-asyncio 1.3.0 (``asyncio_mode=strict``), httpx 0.28.1, openai 2.31.0,
anthropic 0.92.0. The ordering is a property of pytest-asyncio's fixture graph
and the drain rides on openai/anthropic internals, so a bump to any of these is
exactly when the affinity pair and
``test_a_reaped_client_del_schedules_no_aclose_task`` should be re-run.
"""
from __future__ import annotations

import asyncio
import contextlib
import warnings
import weakref

import _fm_helpers
import anthropic
import httpx
import openai
import pytest
from _fm_helpers import (
    _TRACK_SENTINEL,
    _warn_if_drain_closed_a_foreign_client,
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


@pytest.mark.asyncio
async def test_reap_closes_a_leaked_anthropic_client():
    """The ONE patch point covers ``anthropic`` too, not just ``openai``.

    Both libraries ship their own ``AsyncHttpxClientWrapper``, and the drain
    hooks NEITHER of them: it hooks their shared base,
    ``httpx.AsyncClient.__init__``. "One base-class hook catches every library
    with this pattern, present and future" is the load-bearing design claim of
    ``track_async_httpx_clients`` — without this test it is asserted for
    ``openai`` only and merely asserted in prose for ``anthropic``, so an
    ``anthropic`` release that stopped chaining to the base ``__init__`` would
    silently drop out of coverage.
    """
    track_async_httpx_clients()
    await reap_leaked_async_httpx_clients()

    client = anthropic.AsyncAnthropic(api_key='test-key')
    inner = client._client
    assert hasattr(type(inner), '__del__'), (
        f'{type(inner).__module__}.{type(inner).__qualname__} no longer defines '
        f'__del__, so anthropic clients are no longer resurrect-capable and the '
        f'reaper now deliberately skips them — re-derive the predicate in '
        f'_fm_helpers._leaked_async_httpx_clients (task 4412).'
    )
    assert inner.is_closed is False, 'expected a freshly-built AsyncAnthropic to hold an open httpx client'

    reaped = await reap_leaked_async_httpx_clients()
    assert reaped == 1, (
        f'the anthropic wrapper must be tracked by the same httpx.AsyncClient '
        f'base hook as the openai one, so the reaper should have closed exactly '
        f'the 1 client this test built, got {reaped}'
    )
    assert inner.is_closed is True, 'reap must aclose() the leaked anthropic client'


@pytest.mark.asyncio
async def test_reap_survives_a_client_whose_aclose_raises():
    """A client that refuses to close does not fail the innocent test around it.

    The reaper runs at EVERY test's teardown, so its FAILURE mode matters more
    than its success one: an exception escaping it is attributed to whichever
    test happened to be finishing — the same "innocent test blamed" shape this
    drain exists to remove. ``RuntimeError('Event loop is closed')`` is the
    realistic case (a pool bound to a loop that is already gone — exactly what
    the sync fallback arm's throwaway ``asyncio.run`` loop can provoke), so it
    is the one pinned here, together with the return contract: a client that
    failed to close is NOT counted as reaped.

    Without this, ``contextlib.suppress(asyncio.TimeoutError, RuntimeError)``
    is unexercised and a later edit narrowing or dropping it would surface as
    a CI failure in an unrelated test's teardown.
    """
    track_async_httpx_clients()
    await reap_leaked_async_httpx_clients()

    client = openai.AsyncOpenAI(api_key='test-key')
    inner = client._client
    real_aclose = inner.aclose

    async def _refuses_to_close(*args, **kwargs):
        raise RuntimeError('Event loop is closed')

    # Instance attribute, not the class: the reaper does client.aclose(), an
    # instance lookup, so this is both sufficient and free of blast radius.
    inner.aclose = _refuses_to_close
    try:
        reaped = await reap_leaked_async_httpx_clients()  # must not raise
        assert reaped == 0, (
            f'a client whose aclose() raised is still open and must not be '
            f'counted as reaped, got {reaped}'
        )
        assert inner.is_closed is False, (
            'sanity: the refusing aclose() left the client open, which is what '
            'makes the count contract above meaningful'
        )
    finally:
        del inner.aclose
        await real_aclose()


@pytest.mark.asyncio
async def test_reap_is_bounded_when_aclose_hangs(monkeypatch):
    """A hanging ``aclose()`` costs a bounded pause, not the whole xdist worker.

    The other half of the "best-effort and bounded" contract. fused-memory runs
    with ``timeout_method = "thread"``, whose handler ends in ``os._exit(1)`` —
    a teardown that blocks forever therefore kills the entire xdist worker, not
    just one test. The bound is exercised at 50ms via
    ``ASYNC_HTTPX_ACLOSE_TIMEOUT`` rather than by waiting out the shipped 10s.
    """
    track_async_httpx_clients()
    await reap_leaked_async_httpx_clients()
    monkeypatch.setattr(_fm_helpers, 'ASYNC_HTTPX_ACLOSE_TIMEOUT', 0.05)

    client = openai.AsyncOpenAI(api_key='test-key')
    inner = client._client
    real_aclose = inner.aclose

    async def _never_returns(*args, **kwargs):
        await asyncio.Event().wait()

    inner.aclose = _never_returns
    try:
        loop = asyncio.get_running_loop()
        started = loop.time()
        reaped = await reap_leaked_async_httpx_clients()  # must not hang or raise
        elapsed = loop.time() - started
        assert elapsed < 5.0, (
            f'the reaper must abandon a hanging aclose() after '
            f'ASYNC_HTTPX_ACLOSE_TIMEOUT (patched to 0.05s here), but it took '
            f'{elapsed:.2f}s — the timeout bound is gone, and a stuck client '
            f'can now hang a test teardown until pytest-timeout kills the '
            f'whole worker (task 4412).'
        )
        assert reaped == 0, (
            f'a client whose aclose() timed out is still open and must not be '
            f'counted as reaped, got {reaped}'
        )
    finally:
        del inner.aclose
        await real_aclose()


@pytest.mark.asyncio
async def test_foreign_client_close_is_warned_about_not_silent():
    """Closing a client that pre-dated the test warns; leaving it open does not.

    The drain's selection predicate has NO notion of ownership or age (see the
    CONSTRAINT block above the two autouse arms in ``conftest.py``): a client
    built by a fixture scoped wider than ``function`` WOULD be closed at the
    first test's teardown, out from under its owner, and the owner would then
    fail with ``Cannot send a request, as the client has been closed`` in a
    later, apparently unrelated test — the "innocent test blamed" shape this
    drain exists to remove. Scoping the reap to the current test was rejected
    (it opens the inverse hole: a wider-scoped client dropped at its own
    teardown is then never drained and can still resurrect), so the trap is
    made DISCOVERABLE instead. This pins that: silent while the pre-existing
    client is untouched, loud the moment the drain has closed it.
    """
    track_async_httpx_clients()
    await reap_leaked_async_httpx_clients()

    client = openai.AsyncOpenAI(api_key='test-key')
    inner = client._client
    # Stand-in for the sync arm's setup snapshot: a client that was already
    # open and tracked before "this test" started.
    preexisting = weakref.WeakSet([inner])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _warn_if_drain_closed_a_foreign_client(preexisting)
    assert caught == [], (
        f'a pre-existing client the drain did NOT close is the normal case and '
        f'must stay silent, got {[str(w.message) for w in caught]}'
    )

    reaped = await reap_leaked_async_httpx_clients()
    assert reaped == 1, 'sanity: the drain closes a foreign client exactly as it closes any other'

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _warn_if_drain_closed_a_foreign_client(preexisting)
    assert len(caught) == 1, (
        f'the drain closed a client that pre-dated the test and said nothing '
        f'— the constraint "never build an async openai/anthropic client in a '
        f'fixture scoped wider than function" is then undiscoverable until it '
        f'surfaces as a closed-client error in an unrelated test (task 4412). '
        f'Got {len(caught)} warnings.'
    )
    assert 'function scope' in str(caught[0].message), (
        f'the warning must name the fix, not just the symptom: {caught[0].message}'
    )


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

    The leak cohorts genuinely ARE mixed: the largest single one —
    ``test_graphiti_llm_client_construction.py``'s
    ``test_returns_openai_client``, 16 of the 40 measured leaks — is a SYNC
    ``def test_``, while ``test_startup_identity_scan.py``,
    ``test_per_group_client_cache.py``,
    ``test_local_endpoint_base_url_integration.py`` and
    ``test_openai_responses_preflight.py`` are ``async def``.

    That is NOT, however, why this arm exists. The inference it used to rest
    on — "a ``pytest_asyncio`` autouse fixture only covers async tests, so
    that arm alone would silently miss 40% of the leaks" — is measurably
    false: instrumented on the pinned toolchain, the async arm fires for a
    plain ``def test_`` too, in a throwaway loop of its own. This arm is
    DEFENCE IN DEPTH — a cheap, version-independent backstop that does not
    depend on pytest-asyncio's fixture graph continuing to behave that way
    across a bump, and that still drains if the async arm is skipped or
    errors. Cheap: it scans a WeakSet and returns before creating a loop
    unless something is actually leaked.
    """
    assert _SYNC_FIXTURE in request.fixturenames, (
        f'{_SYNC_FIXTURE} must be registered as an autouse fixture in '
        f'conftest.py as the version-independent backstop arm of the drain, '
        f'behind the pytest_asyncio arm (task 4412).'
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
def test_aab_the_leak_was_closed_in_its_own_test_loop(request):
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

    ``closing_loop is None`` covers two states that need OPPOSITE handling, so
    they are separated by whether the polluter was collected in this session at
    all (``request.session.items``): DESELECTED — a ``-k`` filter, ``--lf``
    after an unrelated failure, an IDE run-one-test — means this pair simply
    measured nothing and SKIPS, rather than manufacturing a red that reads like
    a drain regression; COLLECTED but silent means the drain genuinely closed
    nothing, and fails loudly. Deselection removes items from
    ``session.items``, whereas an xdist worker collects the whole suite before
    running its share, so a pair split across workers still lands in the loud
    branch — which is correct: that means the ``xdist_group`` marks stopped
    working.
    """
    polluter = 'test_aaa_leaked_client_records_its_closing_loop'
    polluter_collected = any(item.name == polluter for item in request.session.items)
    try:
        closing_loop = _REAP_ORDERING_PROBE.get('closing_loop')
        test_loop = _REAP_ORDERING_PROBE.get('test_loop')

        # Guard FIRST: with no measurement in hand, the affinity assertion
        # below would pass vacuously on `None is None`.
        if test_loop is None and not polluter_collected:
            pytest.skip(
                f'{polluter} was not collected in this session (subset '
                f'selection: -k / --lf / run-one-test), so the teardown '
                f'affinity pair has nothing to compare. The full-suite signal '
                f'is unaffected.'
            )
        assert test_loop is not None, (
            f'{polluter} WAS collected in this session but recorded no loop, '
            f'so it never ran before this test. Either the pair was split '
            f"across xdist workers — check both arms still carry "
            f"@pytest.mark.xdist_group('async_httpx_reap_ordering') and remain "
            f'adjacent in collection order — or it errored out (task 4412).'
        )
        assert closing_loop is not None, (
            'the client leaked by '
            'test_aaa_leaked_client_records_its_closing_loop ran but its '
            'aclose() was never observed: the conftest drain closed nothing. '
            'Check that both autouse arms are still wired in conftest.py '
            '(task 4412).'
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
