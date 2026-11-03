"""Hermetic tests for ``_live_merge_worker.running_merge_worker``.

The context manager exists for one path: a ``wait_responsive`` give-up raises
``pytest.fail.Exception``, a BaseException, and the worker must still be stopped
and joined.  A recording double stands in for the merge worker, so nothing here
touches git or the real lane.
"""

from __future__ import annotations

import asyncio

import pytest
from _live_merge_worker import running_merge_worker


class _RecordingWorker:
    """Runs until stopped; optionally raises from ``run`` once stopped."""

    def __init__(self, *, raise_on_exit: BaseException | None = None) -> None:
        self.started = asyncio.Event()
        self.stop_calls = 0
        self.run_returned = False
        self._stopped = asyncio.Event()
        self._raise_on_exit = raise_on_exit

    async def run(self) -> None:
        self.started.set()
        await self._stopped.wait()
        if self._raise_on_exit is not None:
            raise self._raise_on_exit
        self.run_returned = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self._stopped.set()


@pytest.mark.asyncio
async def test_the_worker_runs_for_the_block_and_is_stopped_and_joined_after_it() -> None:
    worker = _RecordingWorker()

    async with running_merge_worker(worker):
        await asyncio.wait_for(worker.started.wait(), timeout=5)
        assert worker.stop_calls == 0

    assert worker.stop_calls == 1
    assert worker.run_returned


@pytest.mark.asyncio
async def test_a_give_up_inside_the_block_still_stops_and_joins_the_worker() -> None:
    worker = _RecordingWorker()

    with pytest.raises(pytest.fail.Exception, match='gave up'):
        async with running_merge_worker(worker):
            await asyncio.wait_for(worker.started.wait(), timeout=5)
            pytest.fail('gave up')

    assert worker.stop_calls == 1, 'a give-up must not skip stop(): esc-3980-4'
    assert worker.run_returned, 'the run task must be joined, not leaked into teardown'


@pytest.mark.asyncio
async def test_a_run_task_that_raises_after_stop_fails_the_block() -> None:
    worker = _RecordingWorker(raise_on_exit=RuntimeError('run() crashed on shutdown'))

    with pytest.raises(RuntimeError, match='crashed on shutdown'):
        async with running_merge_worker(worker):
            await asyncio.wait_for(worker.started.wait(), timeout=5)

    assert worker.stop_calls == 1
