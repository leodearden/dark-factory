"""Tests for drain_async_mock_coroutines() and its autouse fixture wiring.

Task 1714 / esc-1702-13: order-dependent orchestrator test failures caused by
un-awaited AsyncMock coroutines that survive into sibling tests via GC cycles.

Design decisions:
- Tests are self-contained unit tests of the helper itself (not a cross-test
  polluter→victim pair), so they are deterministic under -n auto --dist loadgroup
  without requiring an xdist_group co-location tag.
- The _count_open_mock_coros() probe searches the heap for CORO_CREATED
  ``_execute_mock_call`` coroutines; the drain finds them through the registry
  ``track_async_mock_coroutines`` fills.  The two share no selection logic, so
  the probe is an independent observation of what the drain left behind.
"""
from __future__ import annotations

import gc
import inspect
import threading
from unittest.mock import AsyncMock, AsyncMockMixin

import pytest
from _orch_helpers import drain_async_mock_coroutines, track_async_mock_coroutines


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
    self-referential dict cycle.  The GC can find it via gc.get_objects() but
    ref-count reclamation cannot reclaim it (the cycle keeps it alive).  After
    drain the probe count drops to zero AND the specific coroutine object is in
    CORO_CLOSED state — providing an independent observable that does not depend
    on the probe's selection predicate.
    """
    # Flush any orphans from prior tests before measuring
    drain_async_mock_coroutines()

    m = AsyncMock()
    coro = m()  # produces an orphaned _execute_mock_call coroutine
    cycle: dict = {}
    cycle['self'] = cycle
    cycle['coro'] = coro
    del m
    # Keep `coro` as a direct reference so we can inspect its state after drain.
    # The cycle still holds it (preventing ref-count reclamation); the direct
    # reference does not change that — it only lets us verify the object state
    # independently of _count_open_mock_coros()'s predicate.

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
    # Independent observable: verify the specific coroutine object (not just the
    # aggregate count) reached CORO_CLOSED state.  This assertion does not share
    # the _count_open_mock_coros() predicate, so it catches a bug where drain
    # mis-selects objects (the predicate would be wrong in lockstep with the probe,
    # but this direct state check would still fail if .close() was not called).
    assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED, (
        'drain_async_mock_coroutines() must call .close() on the specific '
        '_execute_mock_call coroutine, bringing it to CORO_CLOSED state'
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

    A genuine product async def is never registered by
    ``track_async_mock_coroutines``, so it must survive a drain call intact.  This is the
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
            '(only AsyncMock call coroutines are registered for draining)'
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


def test_orphan_created_on_a_background_thread_is_closed():
    created = []
    thread = threading.Thread(target=lambda: created.append(AsyncMock()()))
    thread.start()
    thread.join()

    assert drain_async_mock_coroutines() >= 1
    assert inspect.getcoroutinestate(created[0]) == inspect.CORO_CLOSED


def test_drain_tolerates_a_thread_calling_mocks_while_it_runs():
    """Task 5668: a registry that is iterated raised here in 170 of 8,139 drains on CPython 3.13.9.

    The thread's coroutines are born and die while the drain works through the
    long-lived ones, which is what resizes an iterated container under it.
    """
    long_lived = [AsyncMock()() for _ in range(2000)]
    stop = threading.Event()

    def churn_mock_coroutines():
        mock = AsyncMock()
        while not stop.is_set():
            mock().close()

    thread = threading.Thread(target=churn_mock_coroutines)
    thread.start()
    try:
        for _ in range(200):
            drain_async_mock_coroutines()
    finally:
        stop.set()
        thread.join()

    assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in long_lived)


def test_drain_finds_orphans_without_searching_the_heap(monkeypatch):
    """Task 5668: a per-test gc.get_objects() search was 88% of a full-suite run's CPU."""
    coro = AsyncMock()()

    def heap_search_is_forbidden():
        raise AssertionError('drain_async_mock_coroutines searched the heap')

    monkeypatch.setattr(gc, 'get_objects', heap_search_is_forbidden)

    assert drain_async_mock_coroutines() >= 1
    assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED


def test_tracking_does_not_hide_a_leak_that_refcounting_reclaims():
    with pytest.warns(RuntimeWarning, match='never awaited'):
        AsyncMock()()


def test_tracking_refuses_an_interpreter_whose_mock_call_path_it_cannot_follow(monkeypatch):
    monkeypatch.setattr(AsyncMockMixin, '_execute_mock_call', lambda self, *args, **kwargs: None)

    with pytest.raises(RuntimeError, match='_execute_mock_call'):
        track_async_mock_coroutines()
