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
import os
import types
import weakref
from collections import deque
from typing import Any, NamedTuple
from unittest.mock import AsyncMockMixin

_log = logging.getLogger(__name__)


class OrphanedAsyncMockCall(NamedTuple):
    """An AsyncMock call that was never awaited, and where it was made.

    ``created_during`` is pytest's ``PYTEST_CURRENT_TEST`` label when the call
    was made, such as ``'tests/test_x.py::test_y (call)'``, or None outside any
    test phase (collection).  pytest gives that label no stable format, so it is
    displayed, never parsed.
    """

    mock_name: str
    created_during: str | None


_created_mock_call_coroutines: deque[tuple[weakref.ref, str | None]] = deque()


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
        _created_mock_call_coroutines.append(
            (weakref.ref(coroutine), os.environ.get('PYTEST_CURRENT_TEST'))
        )
        return coroutine

    # pyright reads this package as Python 3.11; markcoroutinefunction arrived in 3.12.
    AsyncMockMixin._execute_mock_call = inspect.markcoroutinefunction(  # pyright: ignore[reportAttributeAccessIssue]
        create_tracked_coroutine
    )


def drain_async_mock_coroutines() -> tuple[OrphanedAsyncMockCall, ...]:
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

    Returns the orphans it closed, each naming the mock that produced it and
    the test phase it was created in.
    """
    orphans: list[OrphanedAsyncMockCall] = []
    for _ in range(len(_created_mock_call_coroutines)):
        coroutine_ref, created_during = _created_mock_call_coroutines.popleft()
        coroutine = coroutine_ref()
        if coroutine is None or inspect.getcoroutinestate(coroutine) != inspect.CORO_CREATED:
            continue
        mock_name = _calling_mock_name(coroutine)
        coroutine.close()
        if mock_name is not None:
            orphans.append(OrphanedAsyncMockCall(mock_name, created_during))
    if orphans:
        _log.debug(
            'drain_async_mock_coroutines: closed %d orphaned AsyncMock '
            '_execute_mock_call coroutine(s) — consider auditing un-awaited '
            'AsyncMock calls in the preceding test',
            len(orphans),
        )
    return tuple(orphans)


def _calling_mock_name(coroutine: types.CoroutineType[Any, Any, Any]) -> str | None:
    """Name the mock whose call created *coroutine*: ``_execute_mock_call``'s ``self``.

    None when another thread finished the coroutine since its state was read,
    which leaves no frame to read ``self`` from.  The frame must be read before
    ``close()``, which discards it.
    """
    frame = coroutine.cr_frame
    if frame is None:
        return None
    mock = frame.f_locals.get('self')
    if mock is None:
        return None
    return mock._extract_mock_name()
