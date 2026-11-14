"""A cancellation between a run's drain and its journal row re-buffers that run's
events, and a cancellation during a failed chunk's restore does not suppress its
escalation (task 6695)."""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from escalation.queue import EscalationQueue

from fused_memory.models.reconciliation import EventSource, EventType, ReconciliationEvent
from fused_memory.reconciliation.harness import BacklogIterator
from reconciliation.harness_fixtures import make_harness_config, open_harness

PROJECT_ID = 'test-project'


@pytest.fixture
def escalation_queue(tmp_path):
    return EscalationQueue(tmp_path / 'esc')


@pytest_asyncio.fixture
async def harness(tmp_path, escalation_queue):
    async with open_harness(
        tmp_path,
        taskmaster=AsyncMock(),
        config=make_harness_config(judge_enabled=False),
        known_projects={PROJECT_ID: '/tmp/test-project'},
        escalation_queue=escalation_queue,
    ) as h:
        yield h


async def _push_events(harness, count: int) -> None:
    for _ in range(count):
        await harness.buffer.push(ReconciliationEvent(
            id=str(uuid4()),
            type=EventType.episode_added,
            source=EventSource.agent,
            project_id=PROJECT_ID,
            timestamp=datetime.now(UTC),
            payload={},
        ))


def _block_first_call(monkeypatch: pytest.MonkeyPatch, obj: object, name: str) -> asyncio.Event:
    """Make the first call of ``obj.<name>`` set the returned event and never return."""
    original = getattr(obj, name)
    entered = asyncio.Event()

    async def blocking_first_call(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            await asyncio.Event().wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(obj, name, blocking_first_call)
    return entered


async def _cancel_once_entered(
    coro: Coroutine[Any, Any, None], entered: asyncio.Event,
) -> asyncio.Task[None]:
    task = asyncio.create_task(coro)
    entered_wait = asyncio.ensure_future(entered.wait())
    try:
        await asyncio.wait({entered_wait, task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        entered_wait.cancel()
    assert entered.is_set(), f'the blocked call was never reached; task: {task!r}'
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    return task


@pytest.mark.asyncio
async def test_project_loop_shutdown_cancel_before_the_run_row_restores_the_drained_events(
    harness, monkeypatch,
):
    await _push_events(harness, 2)
    entered = _block_first_call(monkeypatch, harness.journal, 'start_run')

    task = await _cancel_once_entered(harness._project_loop(PROJECT_ID), entered)

    assert task.cancelled()
    assert await harness.buffer.count_buffered(PROJECT_ID) == 2
    assert await harness.journal.get_recent_runs(PROJECT_ID) == []


@pytest.mark.asyncio
@pytest.mark.parametrize('blocked', ['record_chunk_boundary', 'start_run'])
async def test_backlog_iterator_shutdown_cancel_before_the_run_row_restores_the_in_flight_chunk(
    harness, monkeypatch, blocked,
):
    await _push_events(harness, 3)
    entered = _block_first_call(monkeypatch, harness.journal, blocked)
    iterator = BacklogIterator(harness.config, harness.journal, harness.buffer, harness)

    task = await _cancel_once_entered(iterator.run(PROJECT_ID), entered)

    assert task.cancelled()
    assert await harness.buffer.count_buffered(PROJECT_ID) == 3
    assert await harness.journal.get_recent_runs(PROJECT_ID) == []


@pytest.mark.asyncio
async def test_backlog_chunk_failure_escalates_even_when_a_cancel_lands_during_its_restore(
    harness, escalation_queue, monkeypatch,
):
    """The cancel stands in for the iteration deadline expiring mid-restore."""
    await _push_events(harness, 1)
    monkeypatch.setattr(
        harness.journal, 'record_chunk_boundary',
        AsyncMock(side_effect=RuntimeError('disk I/O error')),
    )
    original_restore = harness.buffer.restore_drained
    iterator_task_ref: list[asyncio.Task[None]] = []

    async def restore_cancelling_the_iterator(*args, **kwargs):
        iterator_task_ref[0].cancel()
        return await original_restore(*args, **kwargs)

    monkeypatch.setattr(harness.buffer, 'restore_drained', restore_cancelling_the_iterator)
    iterator = BacklogIterator(harness.config, harness.journal, harness.buffer, harness)

    tasks_before = asyncio.all_tasks()
    task = asyncio.create_task(iterator.run(PROJECT_ID))
    iterator_task_ref.append(task)
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await asyncio.gather(*(asyncio.all_tasks() - tasks_before))

    assert task.cancelled()
    assert [e for e in escalation_queue.get_pending() if e.category == 'recon_backlog_overflow']
    assert await harness.buffer.count_buffered(PROJECT_ID) == 1
