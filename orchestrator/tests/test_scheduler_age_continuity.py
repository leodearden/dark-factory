"""A task's pending wait survives requeues and restarts (task 6659, PRD D3).

D3 in ``plans/scheduler-dispatch-scoring-and-lock-layer-prd.md``: a task keeps
its anchor across every departure from the active fetch (infra-hold, done, or
a transient vanish) and across every requeue and restart.  Only a changed
``pending_since``, the one overwrite D3 makes on cancelled -> pending, resets
the wait, and that reset is applied once, not again at every later start.

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


_FILED = '2026-10-01T00:00:00+00:00'
_REOPENED = '2026-10-05T00:00:00+00:00'


def _task(
    tid: str,
    status: str = 'pending',
    priority: str = 'medium',
    pending_since: str | None = None,
) -> dict:
    metadata: dict = {'files': [f'mod{tid}']}
    if pending_since is not None:
        metadata['pending_since'] = pending_since
    return {
        'id': tid,
        'title': f'Task {tid}',
        'status': status,
        'priority': priority,
        'dependencies': [],
        'metadata': metadata,
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


_DECISION_TICK = [
    _task('10', pending_since=_FILED), _task('4995'), _task('5000', 'in-progress'),
]


async def _observe_10_in_progress(scheduler: Scheduler) -> None:
    await _tick(scheduler, [
        _task('10', 'in-progress', pending_since=_FILED), _task('4000', 'in-progress'),
    ])


class TestLeavingTheFetchKeepsTheWait:
    """Leaving the active fetch is not a cancellation, so a return keeps the
    anchor the task left with: '10''s 4990 beats '4995''s 5."""

    async def test_an_infra_hold_resume_keeps_its_wait(self):
        """infra-hold is outside ACTIVE_TASK_STATUSES, so '10' leaves the fetch."""
        scheduler = _scheduler()
        await _observe_10_in_progress(scheduler)
        await _tick(scheduler, [_task('4000', 'in-progress')])

        assert await _tick(scheduler, _DECISION_TICK) == '10'

    async def test_an_infra_hold_resume_keeps_its_wait_across_a_restart(self):
        scheduler = _scheduler()
        await _observe_10_in_progress(scheduler)
        await _tick(scheduler, [_task('4000', 'in-progress')])

        assert await _tick(_scheduler(), _DECISION_TICK) == '10'

    async def test_a_done_reopen_keeps_its_wait(self):
        scheduler = _scheduler()
        await _observe_10_in_progress(scheduler)
        await _tick(scheduler, [
            _task('10', 'done', pending_since=_FILED), _task('4000', 'in-progress'),
        ])

        assert await _tick(scheduler, _DECISION_TICK) == '10'

    async def test_a_first_stamp_is_not_a_reset(self):
        """D3's undefer: the stamp goes from absent to present unchanged."""
        scheduler = _scheduler()
        await _tick(scheduler, [_task('10', 'deferred'), _task('4000', 'in-progress')])

        assert await _tick(scheduler, _DECISION_TICK) == '10'


def _guard(name: str) -> Path:
    path = guard_path(OrchestratorConfig().project_root, name)
    assert path is not None
    return path


async def _resurrect_100(event_store: EventStore | None = None) -> None:
    """'100' leaves the active fetch, then returns cancelled -> pending with a
    new stamp and is reset at max_id 1000."""
    scheduler = _scheduler(event_store)
    await _tick(scheduler, [_task('100', pending_since=_FILED), _task('1000', 'in-progress')])
    await _tick(scheduler, [_task('1000', 'in-progress')])
    await _tick(scheduler, [
        _task('100', pending_since=_REOPENED), _task('1000', 'in-progress'),
    ])


_AFTER_RESURRECTION = [
    _task('100', pending_since=_REOPENED), _task('1040'), _task('1050', 'in-progress'),
]


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

        tasks = [
            _task('100', pending_since=_REOPENED), _task('900'), _task('1050', 'in-progress'),
        ]

        assert await _tick(_scheduler(), tasks) == '900'

    async def test_a_reopen_never_observed_absent_is_still_reset(self):
        """A cancel and reopen inside one tick gap.  The decision runs in a new
        process because process 1 holds '100''s lock."""
        scheduler = _scheduler()
        await _tick(scheduler, [_task('100', pending_since=_FILED), _task('1000', 'in-progress')])
        await _tick(scheduler, [
            _task('100', pending_since=_REOPENED), _task('1000', 'in-progress'),
        ])

        tasks = [
            _task('100', pending_since=_REOPENED), _task('900'), _task('1050', 'in-progress'),
        ]

        assert await _tick(_scheduler(), tasks) == '900'

    async def test_only_the_applied_reset_is_journaled(self, tmp_path: Path):
        """INV-11: one row for the reset, none for the restores."""
        store = EventStore(tmp_path / 'events.db', 'run-6659')
        await _resurrect_100(store)
        await _tick(_scheduler(store), _AFTER_RESURRECTION)
        await _tick(_scheduler(store), _AFTER_RESURRECTION)

        rows = store.fetch_events_by_type(EventType.age_anchor_reset)

        assert [(r['task_id'], r['data']) for r in rows] == [
            ('100', {'anchor': 1000, 'pending_since': _REOPENED}),
        ]

    async def test_a_mass_departure_or_return_writes_the_age_guard_at_most_once(
        self, monkeypatch,
    ):
        holder = _task('3000', 'in-progress')
        ids = range(2000, 2050)
        scheduler = _scheduler()
        await _tick(scheduler, [
            *[_task(str(tid), 'in-progress', pending_since=_FILED) for tid in ids], holder,
        ])

        writes: list[Path] = []
        real = _safe_io.atomic_write_text

        def recorder(path, text, **kwargs):
            writes.append(Path(path))
            return real(path, text, **kwargs)

        monkeypatch.setattr(_safe_io, 'atomic_write_text', recorder)
        guard = _guard('scheduler_age_anchors.json')

        await _tick(scheduler, [holder])
        assert writes.count(guard) == 1, '50 departures are recorded in one write'

        writes.clear()
        await _tick(scheduler, [*[_task(str(tid), pending_since=_FILED) for tid in ids], holder])
        assert writes.count(guard) == 0, 'a restore writes nothing'

        writes.clear()
        await _tick(scheduler, [holder])
        assert writes.count(guard) == 1

        writes.clear()
        await _tick(scheduler, [
            *[_task(str(tid), pending_since=_REOPENED) for tid in ids], holder,
        ])
        assert writes.count(guard) == 1, '50 resets are recorded in one write'

    async def test_a_malformed_recorded_anchor_is_ignored(self):
        anchors = PersistentMap(_guard('scheduler_age_anchors.json'), ttl=timedelta(days=30))
        anchors['10'] = {'anchor': 'not-an-int', 'pending_since': None}
        anchors['4995'] = 'garbage'

        assert await _tick(_scheduler(), [_task('10'), _task('4995')]) == '10'
