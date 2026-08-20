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
- Tests are self-contained unit tests of the helper itself (not a cross-test
  polluter/victim pair), so they are deterministic under
  ``-n auto --dist loadgroup`` without requiring an ``xdist_group`` tag.
- Each test installs the tracking hook itself and reaps any residual leaked
  client from a *prior* test before measuring — the same flush-before-measure
  pattern that module uses for ``reap_leaked_aiosqlite_connections()``.
"""
from __future__ import annotations

import asyncio
import contextlib

import httpx
import openai
import pytest
from _fm_helpers import (
    reap_leaked_async_httpx_clients,
    track_async_httpx_clients,
)


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
    inner.__del__()
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
    inner.__del__()
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
