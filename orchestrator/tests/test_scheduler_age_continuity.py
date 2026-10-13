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

from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import shared.safe_io as _safe_io

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.guard_state import PersistentMap, guard_path
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


def _guard(name: str) -> Path:
    path = guard_path(OrchestratorConfig().project_root, name)
    assert path is not None
    return path


async def _resurrect_100(event_store: EventStore | None = None) -> None:
    """'100' leaves the active fetch, then returns and is reset at max_id 1000."""
    scheduler = _scheduler(event_store)
    await _tick(scheduler, [_task('100'), _task('1000', 'in-progress')])
    await _tick(scheduler, [_task('1000', 'in-progress')])
    await _tick(scheduler, [_task('100'), _task('1000', 'in-progress')])


_AFTER_RESURRECTION = [_task('100'), _task('1040'), _task('1050', 'in-progress')]


class TestAResetIsAppliedOnce:
    async def test_every_later_start_restores_the_reset_instead_of_re_zeroing(self):
        """Age 50 since the reset beats '1040''s 10, at each of two restarts."""
        await _resurrect_100()

        assert await _tick(_scheduler(), _AFTER_RESURRECTION) == '100'
        assert await _tick(_scheduler(), _AFTER_RESURRECTION) == '100'

    async def test_a_restart_does_not_undo_the_reset(self):
        """The resurrected task does not regain its own-id age (task 5352):
        '900''s 150 beats the 50 accrued since the reset."""
        await _resurrect_100()

        tasks = [_task('100'), _task('900'), _task('1050', 'in-progress')]

        assert await _tick(_scheduler(), tasks) == '900'

    async def test_only_the_applied_reset_is_journaled(self, tmp_path: Path):
        """INV-11: one row for the reset, none for the restores."""
        store = EventStore(tmp_path / 'events.db', 'run-6659')
        await _resurrect_100(store)
        await _tick(_scheduler(store), _AFTER_RESURRECTION)
        await _tick(_scheduler(store), _AFTER_RESURRECTION)

        rows = store.fetch_events_by_type(EventType.age_anchor_reset)

        assert [(r['task_id'], r['data']) for r in rows] == [('100', {'anchor': 1000})]

    async def test_a_mass_return_writes_each_age_guard_once(self, monkeypatch):
        """'2999' stays away, so its mark keeps the owed file non-empty and
        the consumption is a write rather than the empty store's unlink."""
        returning = [_task(str(tid)) for tid in range(2000, 2050)]
        holder = _task('3000', 'in-progress')
        scheduler = _scheduler()
        await _tick(scheduler, [*returning, _task('2999'), holder])
        await _tick(scheduler, [holder])

        writes: list[Path] = []
        real = _safe_io.atomic_write_text

        def recorder(path, text, **kwargs):
            writes.append(Path(path))
            return real(path, text, **kwargs)

        monkeypatch.setattr(_safe_io, 'atomic_write_text', recorder)
        await _tick(scheduler, [*returning, holder])

        assert writes.count(_guard('scheduler_age_reset_owed.json')) == 1
        assert writes.count(_guard('scheduler_age_reset_anchors.json')) == 1

    async def test_a_non_integer_recorded_anchor_is_ignored(self):
        anchors = PersistentMap(_guard('scheduler_age_reset_anchors.json'), ttl=timedelta(days=30))
        anchors['10'] = 'not-an-int'

        assert await _tick(_scheduler(), [_task('10'), _task('4995')]) == '10'
