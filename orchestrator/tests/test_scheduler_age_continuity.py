"""A task's pending wait survives requeues and restarts (task 6659, PRD D3).

D3 in ``plans/scheduler-dispatch-scoring-and-lock-layer-prd.md``: only a task
that left the active population and came back has its age reset.  A requeue,
an unblock, an undefer or a restart-cancel keeps the wait it has accrued, and a
reset, once applied, is not re-applied by every later process start.

Every case is observed through the dispatch decision.  A second
``_scheduler()`` in one test IS a restarted process: conftest's autouse
``_isolate_orch_config`` pins ``project_root`` to the test's ``tmp_path``, so
it shares the guards directory and starts with empty memory.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.scheduler import Scheduler

pytestmark = pytest.mark.asyncio


def _scheduler(event_store: EventStore | None = None) -> Scheduler:
    scheduler = Scheduler(OrchestratorConfig(max_per_module=1), event_store=event_store)
    scheduler.finish_startup()
    scheduler.get_statuses = AsyncMock(return_value=({}, None))
    return scheduler


def _task(tid: str, status: str = 'pending', priority: str = 'medium') -> dict:
    return {
        'id': tid,
        'title': f'Task {tid}',
        'status': status,
        'priority': priority,
        'dependencies': [],
        'metadata': {'files': [f'mod{tid}']},
    }


async def _tick(scheduler: Scheduler, tasks: list[dict]) -> str | None:
    scheduler.get_tasks = AsyncMock(return_value=tasks)
    assignment = await scheduler.acquire_next()
    return assignment.task_id if assignment is not None else None


async def _dispatch_10_then_observe_it_in_progress() -> Scheduler:
    scheduler = _scheduler()
    assert await _tick(scheduler, [_task('10'), _task('4000', 'in-progress')]) == '10'
    await _tick(scheduler, [_task('10', 'in-progress'), _task('4000', 'in-progress')])
    return scheduler


class TestAccruedWaitIsKept:
    async def test_a_restart_cancelled_task_keeps_its_wait(self):
        """The restart that cancels an in-progress task back to pending must
        not also zero the age it accrued before it was dispatched."""
        await _dispatch_10_then_observe_it_in_progress()

        restarted = _scheduler()
        tasks = [_task('10'), _task('4995'), _task('5000', 'in-progress')]

        assert await _tick(restarted, tasks) == '10'

    async def test_an_in_process_requeue_keeps_its_wait(self):
        scheduler = await _dispatch_10_then_observe_it_in_progress()
        scheduler.release('10')
        tasks = [_task('10'), _task('4995'), _task('5000', 'in-progress')]

        assert await _tick(scheduler, tasks) == '10'
