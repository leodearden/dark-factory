"""Helpers that let a caller see asyncio tasks end.

:func:`track_task` holds a strong reference to a fire-and-forget task until it
ends, and :func:`abandon_task` cancels a task its caller stopped waiting on and
tracks it. The event loop keeps only a WEAK reference to a Task, so a task
that nothing else references can be garbage-collected mid-flight. Membership
in a caller's registry set supplies the strong reference; one done-callback
releases it and consumes the task's exception so asyncio does not log
"exception was never retrieved".

This is the single home (task 4530) for a pattern that was previously
hand-rolled three times: ``dashboard/src/dashboard/data/db.py``,
``dashboard/src/dashboard/app.py`` (task 4089) and
``orchestrator/src/orchestrator/merge_skew_tripwire.py`` (task 4233). Only the
helper is shared; each call site keeps its OWN registry set, because callers
drain, await or gauge their registry independently.

:func:`cancel_bounded` is the bounded counterpart for a caller that must see
the task end, or learn that it would not, within a known time (task 6660). A
single ``cancel()`` followed by an unbounded ``await`` hangs forever on a task
that survives one cancel. :func:`cancel_all_within` bounds the wait for many
tasks at once, returning the ones still running for the caller to dispose of
(task 7247).

:func:`gather_unwound` is ``asyncio.gather`` for children that own
subprocesses: an exception, an outer cancel included, propagates only once
every child has ended (tasks 7977, 8078).

This module is intentionally NOT re-exported from ``shared/__init__.py``.
Consumers import via the fully-qualified path
(``from shared.asyncio_tasks import track_task``), consistent with the
``task_statuses``/``timestamps``/``task_claimant`` sub-module convention.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Collection, Sequence
from typing import TypeVar

__all__ = ['abandon_task', 'cancel_all_within', 'cancel_bounded', 'gather_unwound', 'track_task']

logger = logging.getLogger(__name__)


def track_task(
    task: asyncio.Task,
    *registries: set[asyncio.Task],
    on_done: Callable[[asyncio.Task], None] | None = None,
) -> None:
    """Keep *task* alive in every set in *registries* until it ends.

    One done-callback removes the task from ALL the registries and consumes its
    exception ONCE — a task tracked in two registries does not get two
    callbacks retrieving the same exception twice.

    *on_done* lets a call site add diagnostics (e.g. logging how the task
    ended) without installing a SECOND done-callback that would retrieve the
    same exception again. It runs for every outcome, cancellation included,
    after the release; ``finished.exception()`` still works inside it. If it
    raises, the failure is logged at WARNING and never propagates.
    """
    for registry in registries:
        registry.add(task)

    def _release(finished: asyncio.Task) -> None:
        # Order is the contract: release first so a raising hook cannot strand
        # the strong reference; consume before the hook so the "never
        # retrieved" guarantee does not depend on what the hook does.
        for registry in registries:
            registry.discard(finished)
        if not finished.cancelled():
            finished.exception()
        if on_done is None:
            return
        try:
            on_done(finished)
        except Exception:
            logger.warning(
                'on_done hook %s raised for task %r',
                getattr(on_done, '__qualname__', on_done),
                finished,
                exc_info=True,
            )

    task.add_done_callback(_release)


def abandon_task(
    task: asyncio.Task,
    *registries: set[asyncio.Task],
    on_done: Callable[[asyncio.Task], None] | None = None,
) -> None:
    """Cancel *task* without awaiting it, and track it until it has unwound.

    For a caller that stopped waiting on *task* exceptionally — its own
    budget expired, or it was itself cancelled. ``asyncio.wait(...,
    timeout=...)`` does not cancel what it waits on, and awaiting the
    cancellation would re-introduce the very hang the timeout exists to
    bound, so the task is cancelled and handed to :func:`track_task`.

    The task is in every registry when this returns, before the loop runs
    again, so a caller may read ``len(registry)`` straight afterwards as a
    backlog gauge.
    """
    task.cancel()
    track_task(task, *registries, on_done=on_done)


async def cancel_bounded(
    task: asyncio.Task,
    *registries: set[asyncio.Task],
    grace_secs: float,
) -> bool:
    """Cancel *task* and wait a bounded time for it to end; True iff it ended.

    The cancel policy is :func:`_cancel_twice`'s, with both windows
    *grace_secs* long. A task that outlives both windows is handed to
    :func:`track_task` with *registries*, so it stays referenced until it ends,
    and False is returned.

    Never raises the task's own exception; it is consumed, as
    :func:`track_task` consumes it.
    """
    if not await _cancel_twice({task}, first_window_secs=grace_secs, second_window_secs=grace_secs):
        return True
    track_task(task, *registries)
    return False


async def cancel_all_within(
    tasks: Collection[asyncio.Task],
    *,
    timeout_secs: float,
) -> set[asyncio.Task]:
    """Cancel every task and wait at most *timeout_secs* in all; return those still running.

    The cancel policy is :func:`_cancel_twice`'s. The first window is two
    thirds of *timeout_secs* and the second the remaining third: the second
    cancel is the fallback for a task that absorbed the first, so it should not
    cut short a task still honestly unwinding from the first, for instance
    still killing its subprocess. Unlike ``wait_for(gather(...))``, a task that
    survives both cancels cannot extend the wait. The returned tasks are
    neither cancelled again nor awaited: the caller decides what becomes of
    them, for example :func:`track_task`. The exceptions of the tasks that
    ended are consumed.
    """
    first_window_secs = timeout_secs * 2 / 3
    return await _cancel_twice(
        tasks, first_window_secs=first_window_secs, second_window_secs=timeout_secs - first_window_secs,
    )


async def _cancel_twice(
    tasks: Collection[asyncio.Task], *, first_window_secs: float, second_window_secs: float,
) -> set[asyncio.Task]:
    """Cancel *tasks* and wait *first_window_secs*; cancel those still running and wait *second_window_secs*.

    A task can survive one ``cancel()``: an inner ``except CancelledError``
    that awaits again, or asyncio's subprocess transport cancelled between
    spawning a child and connecting its pipes. The second cancel ends it. The
    tasks that outlive both windows are returned; the exceptions of those that
    ended are consumed.
    """
    pending = set(tasks)
    for window_secs in (first_window_secs, second_window_secs):
        if not pending:
            break
        for task in pending:
            task.cancel()
        done, pending = await asyncio.wait(pending, timeout=window_secs)
        for task in done:
            if not task.cancelled():
                task.exception()
    return pending


_Gathered = TypeVar('_Gathered')


async def gather_unwound(*aws: Awaitable[_Gathered]) -> list[_Gathered]:
    """``asyncio.gather``, except that it raises only once every child has ended.

    A plain gather raises at its first cancelled or failed child, so a sibling
    still unwinding, for instance still killing its subprocess, would outlive
    the call that gathered it. Here, as in a ``TaskGroup``, the other children are cancelled
    and then awaited, and cancelling the caller again meanwhile does not cut
    that wait short. The exception raised is the one the gather raised, unless
    the caller was cancelled: that cancel is raised instead, so it is never
    swallowed. Cancelling the caller, once or repeatedly, asks each child to
    cancel at most once, and never a child already cancelling: asking again
    would interrupt the kill it is awaiting.
    """
    children = [asyncio.ensure_future(aw) for aw in aws]
    gathered = asyncio.gather(*children)
    gathered.add_done_callback(_consume_outcome)
    try:
        await asyncio.wait((gathered,))
        return gathered.result()
    except BaseException:
        for child in children:
            if not (isinstance(child, asyncio.Task) and child.cancelling()):
                child.cancel()
        await _await_all_through_cancels(children)
        raise


def _consume_outcome(gathered: asyncio.Future) -> None:
    """Retrieve *gathered*'s exception, which nothing else retrieves once the caller has stopped waiting for it."""
    if not gathered.cancelled():
        gathered.exception()


async def _await_all_through_cancels(children: Sequence[asyncio.Future]) -> None:
    """Return once every child is done; re-raise the caller's first cancel then, if any."""
    first_cancel: asyncio.CancelledError | None = None
    while pending := [child for child in children if not child.done()]:
        try:
            await asyncio.wait(pending)
        except asyncio.CancelledError as cancel:
            first_cancel = first_cancel or cancel
    if first_cancel is not None:
        raise first_cancel
