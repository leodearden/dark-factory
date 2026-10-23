"""Close AsyncMock call coroutines that a test created and never awaited.

Calling an ``AsyncMock`` without awaiting the result leaves an orphaned
``_execute_mock_call`` coroutine.  Held in a reference cycle it outlives its
test, and whichever later test the garbage collector finalizes it in fails on
``RuntimeWarning("coroutine '...' was never awaited")`` under
filterwarnings=error (task 1714 / esc-1702-13).  Tracking happens when each
call coroutine is created, so the drain never searches the heap (task 5668).
"""

from __future__ import annotations

import functools
import inspect
import logging
import weakref
from collections import deque
from unittest.mock import AsyncMockMixin

_log = logging.getLogger(__name__)


_created_mock_call_coroutines: deque[weakref.ref] = deque()


def track_async_mock_coroutines() -> None:
    """Register every AsyncMock call coroutine at creation, for ``drain_async_mock_coroutines``.

    The registry holds weak references, so an orphan that ref-counting reclaims
    still warns inside the test that leaked it; only an orphan kept alive by a
    reference cycle survives to the drain.  ``deque.append`` is atomic, so a
    background thread calling an AsyncMock can never corrupt a drain in progress.

    A silent no-op here would hand the task-1714 order-dependent failures back
    to innocent tests, so this refuses to start when ``_execute_mock_call`` is
    no longer a coroutine function.  That AsyncMock calls still GO THROUGH it is
    pinned by test_async_mock_coroutine_isolation.py, which plants an orphan and
    requires the drain to find it without searching the heap.
    """
    create_coroutine = AsyncMockMixin._execute_mock_call
    if not inspect.iscoroutinefunction(create_coroutine):
        raise RuntimeError(
            'unittest.mock.AsyncMockMixin._execute_mock_call is no longer a '
            'coroutine function, so AsyncMock call coroutines cannot be tracked '
            'at creation — re-derive track_async_mock_coroutines against this '
            "interpreter's unittest.mock before trusting the suite's verdicts"
        )

    @functools.wraps(create_coroutine)
    def create_tracked_coroutine(self, /, *args, **kwargs):
        coroutine = create_coroutine(self, *args, **kwargs)
        _created_mock_call_coroutines.append(weakref.ref(coroutine))
        return coroutine

    # pyright reads this package as Python 3.11; markcoroutinefunction arrived in 3.12.
    AsyncMockMixin._execute_mock_call = inspect.markcoroutinefunction(  # pyright: ignore[reportAttributeAccessIssue]
        create_tracked_coroutine
    )


def drain_async_mock_coroutines() -> int:
    """Close every AsyncMock call coroutine created since the last drain and never awaited.

    Task 1714 / esc-1702-13.  Calling an AsyncMock without awaiting the result
    leaves an ``_execute_mock_call`` coroutine in CORO_CREATED.  Held in a
    reference cycle (common in mock object graphs) it outlives its test and is
    finalized by whichever later test the garbage collector happens to run in,
    where ``RuntimeWarning("coroutine '...' was never awaited")`` — promoted to
    an error by orchestrator/pyproject.toml — fails that innocent test.
    ``.close()`` on a CORO_CREATED coroutine finalizes it without the warning,
    so closing each test's orphans at its own boundary keeps them out of its
    siblings.

    Only coroutines registered by ``track_async_mock_coroutines`` are touched:
    a forgotten ``await`` on product code still trips filterwarnings=error.
    The cost is proportional to the AsyncMock calls made since the last drain.
    Its predecessor searched ``gc.get_objects()`` instead, which with the whole
    suite collected was 88% of a full run's CPU (task 5668).

    Returns the number of coroutines closed.
    """
    closed = 0
    for _ in range(len(_created_mock_call_coroutines)):
        coroutine = _created_mock_call_coroutines.popleft()()
        if coroutine is not None and inspect.getcoroutinestate(coroutine) == inspect.CORO_CREATED:
            coroutine.close()
            closed += 1
    if closed:
        _log.debug(
            'drain_async_mock_coroutines: closed %d orphaned AsyncMock '
            '_execute_mock_call coroutine(s) — consider auditing un-awaited '
            'AsyncMock calls in the preceding test',
            closed,
        )
    return closed
