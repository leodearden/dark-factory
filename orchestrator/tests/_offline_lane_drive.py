"""Drive a real OfflineLaneWorker's run loop until a counted number of passes complete."""

from __future__ import annotations

import asyncio
import contextlib

from orchestrator.offline_lane import OfflineLaneWorker


async def await_lane_passes(worker: OfflineLaneWorker, expected_passes: int) -> None:
    """Drive worker.run() as a real background task until *expected_passes*
    full passes (``_run_once`` calls, so INCLUDING each pass's confirm, engine
    observe and filing) have completed, then cancel the loop task.

    A pass that raises fails here at once with its own exception, rather than
    being swallowed into ``run()``'s fail-open backoff and retried. Requires
    ``worker._dirty`` (or the wake event) to already be set by a prior trigger,
    exactly as production wiring would leave it.
    """
    inner_run_once = worker._run_once
    completed = asyncio.get_running_loop().create_future()
    count = {'n': 0}

    async def _counting_run_once():
        try:
            result = await inner_run_once()
        except Exception as exc:
            if not completed.done():
                completed.set_exception(exc)
            raise
        count['n'] += 1
        if count['n'] >= expected_passes and not completed.done():
            completed.set_result(None)
        return result

    worker._run_once = _counting_run_once
    task = asyncio.create_task(worker.run())
    try:
        await completed
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        worker._run_once = inner_run_once
