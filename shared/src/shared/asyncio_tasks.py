"""Hold a strong reference to a fire-and-forget asyncio task until it ends.

The event loop keeps only a WEAK reference to a Task, so a task that nothing
else references can be garbage-collected mid-flight. Membership in a caller's
registry set supplies the strong reference; one done-callback releases it and
consumes the task's exception so asyncio does not log "exception was never
retrieved".

This is the single home (task 4530) for a pattern that was previously
hand-rolled three times: ``dashboard/src/dashboard/data/db.py``,
``dashboard/src/dashboard/app.py`` (task 4089) and
``orchestrator/src/orchestrator/merge_skew_tripwire.py`` (task 4233). Only the
helper is shared; each call site keeps its OWN registry set, because callers
drain, await or gauge their registry independently.

This module is intentionally NOT re-exported from ``shared/__init__.py``.
Consumers import via the fully-qualified path
(``from shared.asyncio_tasks import track_task``), consistent with the
``task_statuses``/``timestamps``/``task_claimant`` sub-module convention.
"""

from __future__ import annotations

import asyncio
import logging

__all__ = ['track_task']

logger = logging.getLogger(__name__)


def track_task(task: asyncio.Task, *registries: set[asyncio.Task]) -> None:
    """Keep *task* alive in every set in *registries* until it ends.

    One done-callback removes the task from ALL the registries and consumes its
    exception ONCE — a task tracked in two registries does not get two
    callbacks retrieving the same exception twice.
    """
    for registry in registries:
        registry.add(task)

    def _release(finished: asyncio.Task) -> None:
        for registry in registries:
            registry.discard(finished)
        if not finished.cancelled():
            finished.exception()

    task.add_done_callback(_release)
