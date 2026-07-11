"""Tests for reap_leaked_aiosqlite_connections() and its autouse fixture wiring.

Task 2413 — fix for the scheduler-test xdist flake where a leaked aiosqlite
connection's background worker thread outlives the test that created it,
then later raises ``RuntimeError: Event loop is closed`` and is blamed on
whatever unrelated test happens to be running when the thread fires.

Design decisions:
- Tests are self-contained unit tests of the helper itself (not a
  cross-test polluter/victim pair), so they are deterministic under
  -n auto --dist loadgroup without requiring an xdist_group co-location tag —
  mirrors test_async_mock_coroutine_isolation.py.
- Each test reaps any residual leaked connection from a *prior* test before
  measuring, matching test_async_mock_coroutine_isolation.py's own
  flush-before-measure pattern for drain_async_mock_coroutines().
"""
from __future__ import annotations

import aiosqlite
import pytest
from _orch_helpers import reap_leaked_aiosqlite_connections


@pytest.mark.asyncio
async def test_reap_closes_leaked_open_connection():
    """reap_leaked_aiosqlite_connections() closes a connection nothing else closed.

    Open an aiosqlite connection and execute a statement so its worker thread
    is proven live, then leak it (never call ``.close()``). After reap, the
    thread must be dead — checked directly via the connection's own
    ``_running`` / ``_thread.is_alive()`` state, an observable independent of
    the helper's internal selection predicate.
    """
    # Flush any leaked connection a prior test left behind before measuring.
    await reap_leaked_aiosqlite_connections()

    conn = await aiosqlite.connect(':memory:')
    await conn.execute('CREATE TABLE t(x)')
    assert conn._thread.is_alive(), 'expected the worker thread to be alive after connect'

    closed = await reap_leaked_aiosqlite_connections()
    assert closed >= 1, 'reap_leaked_aiosqlite_connections() should have closed >= 1 connection'

    # Independent observable: verify the specific connection's own state, not
    # just the aggregate count returned above.
    assert conn._running is False, 'reap must stop() the leaked connection'
    assert not conn._thread.is_alive(), 'reap must join the worker thread before returning'


@pytest.mark.asyncio
async def test_reap_is_safe_when_no_leaked_connections():
    """reap_leaked_aiosqlite_connections() returns 0 when nothing is leaked."""
    # First reap to flush any residual leak from a prior test.
    await reap_leaked_aiosqlite_connections()

    count = await reap_leaked_aiosqlite_connections()
    assert count == 0, (
        f'Expected 0 leaked aiosqlite connections but got {count}. '
        'A prior test may have leaked a connection.'
    )


@pytest.mark.asyncio
async def test_reap_ignores_already_closed_connection():
    """A connection that was properly closed is not counted or touched by reap.

    Safety-net assertion: reap must not double-close (and must not raise on)
    a connection a test already cleaned up correctly.
    """
    await reap_leaked_aiosqlite_connections()

    conn = await aiosqlite.connect(':memory:')
    await conn.close()
    assert not conn._thread.is_alive()

    count = await reap_leaked_aiosqlite_connections()
    assert count == 0, 'An already-closed connection must not be counted as reaped'


def test_autouse_reap_fixture_is_active(request):
    """_reap_leaked_aiosqlite_connections is wired as an autouse teardown fixture.

    This assertion is deterministic and worker-placement-agnostic: existing
    autouse fixtures (_drain_async_mock_coroutines, _reap_leaked_merge_workers)
    appear in request.fixturenames; absent names do not. This is a behavioral
    assertion that the reaper is applied to arbitrary tests — not a docstring
    lint.
    """
    assert '_reap_leaked_aiosqlite_connections' in request.fixturenames, (
        '_reap_leaked_aiosqlite_connections must be registered as an autouse '
        'fixture in conftest.py so it runs after every test and closes any '
        'leaked aiosqlite connection before the per-test event loop closes.'
    )
