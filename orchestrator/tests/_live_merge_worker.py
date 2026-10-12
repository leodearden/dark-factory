"""Driving a live merge worker through a real-``git`` merge round-trip in a test.

Imported by bare module name (``from _live_merge_worker import ...``), matching
the flat-test-helper convention of ``_orch_helpers.py`` (orchestrator/tests/ has
no ``__init__.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Protocol

# Nominal ``wait_responsive`` budget for a ``MergeRequest.result`` whose merge
# runs real ``git`` subprocesses.  These waits were ``asyncio.wait_for(...,
# timeout=60)`` before task 4920 moved them onto ``wait_responsive``, and the
# move keeps 60 rather than lowering it to MERGE_RESULT_TIMEOUT (45).  Such a
# round-trip is dominated by child ``git`` processes, and ``wait_responsive``
# grants no stretch when only those are starved (its "Limitation" section), so
# 45 would narrow the wait in exactly the regime that matters.  The wall-clock
# bill is unchanged: min(RESPONSIVE_WAIT_STRETCH * 60, RESPONSIVE_WAIT_WALL_CAP)
# is the same 90s a MERGE_RESULT_TIMEOUT site is billed.  Never-narrow.
REAL_GIT_MERGE_RESULT_TIMEOUT = 60


class _MergeWorker(Protocol):
    async def run(self) -> None: ...

    async def stop(self) -> None: ...


@contextlib.asynccontextmanager
async def running_merge_worker(worker: _MergeWorker) -> AsyncIterator[None]:
    """Run *worker* for the body of the block, then stop it and join its run task.

    The stop sits in a ``finally`` because ``wait_responsive`` gives up by raising
    ``pytest.fail.Exception`` (``_pytest.outcomes.Failed``), a BaseException.  On a
    straight-line body that give-up, like a failed assertion, skips ``stop()`` and
    leaks a live worker and its run task into pytest-asyncio teardown, where one
    red test cascades into unrelated failures (esc-3980-4).

    The join is neither bounded nor suppressed, so a run task that raises after
    ``stop()`` still fails the test.
    """
    run_task = asyncio.create_task(worker.run())
    try:
        yield
    finally:
        await worker.stop()
        await run_task
