"""Tests for drain_async_mock_coroutines() and its autouse fixture wiring.

Task 1714 / esc-1702-13: order-dependent orchestrator test failures caused by
un-awaited AsyncMock coroutines that survive into sibling tests via GC cycles.

Design decisions:
- Tests are self-contained unit tests of the helper itself (not a cross-test
  polluter→victim pair), so they are deterministic under -n auto --dist loadgroup
  without requiring an xdist_group co-location tag.
- The _count_open_mock_coros() probe and drain use the same selection predicate
  (cr_code.co_name == "_execute_mock_call", state CORO_CREATED), so the test
  is a direct behavioral assertion of the helper's correctness.
"""
from __future__ import annotations

import gc
import inspect
from unittest.mock import AsyncMock

import pytest
from _orch_helpers import drain_async_mock_coroutines


def _count_open_mock_coros() -> int:
    """Count gc-reachable _execute_mock_call coroutines still in CORO_CREATED state."""
    gc.collect()
    return sum(
        1
        for obj in gc.get_objects()
        if inspect.iscoroutine(obj)
        and getattr(getattr(obj, 'cr_code', None), 'co_name', None) == '_execute_mock_call'
        and inspect.getcoroutinestate(obj) == inspect.CORO_CREATED
    )


def test_drain_closes_orphaned_asyncmock_coroutine_in_cycle():
    """drain_async_mock_coroutines() closes orphaned _execute_mock_call coroutines.

    Build an AsyncMock, call it (without awaiting), stash the coroutine in a
    self-referential dict cycle, then delete all local refs.  The GC can find
    it via gc.get_objects() but ref-count reclamation cannot reclaim it.  After
    drain the count drops to zero.
    """
    # Flush any orphans from prior tests before measuring
    drain_async_mock_coroutines()

    m = AsyncMock()
    coro = m()  # produces an orphaned _execute_mock_call coroutine
    cycle: dict = {}
    cycle['self'] = cycle
    cycle['coro'] = coro
    del coro
    del m

    # Force GC to discover the cycle
    gc.collect()
    assert _count_open_mock_coros() >= 1, (
        'Expected at least one orphaned _execute_mock_call coroutine in the GC graph'
    )

    closed = drain_async_mock_coroutines()
    assert closed >= 1, 'drain_async_mock_coroutines() should have closed >= 1 coroutine'
    assert _count_open_mock_coros() == 0, (
        'After drain, no orphaned _execute_mock_call coroutines should remain'
    )


def test_drain_is_safe_when_no_orphans():
    """drain_async_mock_coroutines() returns 0 when there are no orphaned coroutines."""
    # First drain to flush any residual orphans from prior tests
    drain_async_mock_coroutines()
    gc.collect()

    count = drain_async_mock_coroutines()
    assert count == 0, (
        f'Expected 0 orphaned coroutines but got {count}. '
        'A prior test may have leaked an un-awaited AsyncMock call.'
    )


@pytest.mark.asyncio
async def test_awaited_asyncmock_is_not_an_orphan():
    """An awaited AsyncMock coroutine is not treated as an orphan by drain.

    After awaiting AsyncMock(return_value=7)(), the coroutine reaches CORO_CLOSED
    state and is no longer in gc.get_objects() as a live CORO_CREATED coroutine.
    drain_async_mock_coroutines() must not count or close it.
    """
    result = await AsyncMock(return_value=7)()
    assert result == 7

    drain_async_mock_coroutines()
    assert _count_open_mock_coros() == 0, (
        'Awaited AsyncMock coroutine should not appear as an orphan after drain'
    )


def test_product_coroutine_is_not_closed_by_drain():
    """drain_async_mock_coroutines() does NOT close product (non-mock) coroutines.

    A genuine product async def has co_name == 'product_coro' (not
    '_execute_mock_call'), so it must survive a drain call intact.  This is the
    safety-net assertion: a forgotten `await` on product code must still trip
    filterwarnings=error and fail the test; drain must not silently swallow it.
    """

    async def product_coro():
        return 42  # pragma: no cover

    coro = product_coro()
    try:
        assert coro.cr_code.co_name == 'product_coro', (
            'Expected product_coro to have co_name "product_coro"'
        )
        assert inspect.getcoroutinestate(coro) == inspect.CORO_CREATED

        drain_async_mock_coroutines()

        # The product coroutine must still be CORO_CREATED after the drain
        assert inspect.getcoroutinestate(coro) == inspect.CORO_CREATED, (
            'drain_async_mock_coroutines() must not close product coroutines '
            '(co_name != "_execute_mock_call")'
        )
    finally:
        # Close the coro explicitly here to avoid "was never awaited" warning
        coro.close()


def test_autouse_drain_fixture_is_active(request):
    """_drain_async_mock_coroutines is wired as an autouse teardown fixture.

    This assertion is deterministic and worker-placement-agnostic: existing
    autouse fixtures (_clear_probe_cache, _isolate_orch_config) appear in
    request.fixturenames; absent names do not.  This is a behavioral assertion
    that the drain is applied to arbitrary tests — not a docstring lint.
    """
    assert '_drain_async_mock_coroutines' in request.fixturenames, (
        '_drain_async_mock_coroutines must be registered as an autouse fixture '
        'in conftest.py so it runs after every test and prevents orphaned '
        'AsyncMock coroutines from being GC-promoted into sibling tests.'
    )
