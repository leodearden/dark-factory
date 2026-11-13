"""A pure gate re-held behind a new precondition re-fires instead of closing (task 9520).

A watcher that finds a gate fired prematurely holds it behind its real
precondition (a dependency, an external dependency or a milestone) and
resumes it. When that precondition lands the gate must ask the human again,
not close as 'pure gate resolved'. Every row drives ``DeterministicRunner.run``
only, against an in-memory scheduler double whose ``update_task`` merges into
the task's metadata as the real merge write does, and a real
``EscalationQueue``, so the record written by one dispatch is the one read by
the next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.deterministic_runner import DETERMINISTIC_AGENT_ROLE, DeterministicRunner
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import WorkflowOutcome

TASK_ID = '9520'
GATE_CATEGORY = 'milestone_gate'


@dataclass
class _SchedulerDouble:
    """The scheduler surface the runner touches, recording every write in order."""

    task: dict
    writes: list[tuple[str, Any]] = field(default_factory=list)

    async def get_task(self, task_id: str) -> dict:
        return self.task

    async def update_task(
        self, task_id: str, metadata: dict, *, metadata_mode: str | None = None,
        append: bool = False,
    ) -> bool:
        self.writes.append(('update_task', {'metadata': dict(metadata), 'mode': metadata_mode}))
        self.task['metadata'].update(metadata)
        return True

    async def set_task_status(self, task_id: str, status: str, **kwargs: Any) -> None:
        self.writes.append(('set_task_status', {'status': status, **kwargs}))

    def statuses(self) -> list[dict]:
        return [w for kind, w in self.writes if kind == 'set_task_status']

    def done_writes(self) -> list[dict]:
        return [w for w in self.statuses() if w['status'] == 'done']


def _gate_task(
    *, deps: list, milestone: dict | None = None, external_deps: list[str] | None = None,
    **stamps: Any,
) -> dict:
    metadata: dict = {
        'task_kind': 'deterministic',
        'always_escalates': True,
        'before_done': None,
        **stamps,
    }
    if milestone is not None:
        metadata['milestone'] = milestone
    if external_deps is not None:
        metadata['external_deps'] = external_deps
    return {
        'id': TASK_ID,
        'title': 'Approve the rollout',
        'description': 'Human gate over the rollout',
        'dependencies': deps,
        'metadata': metadata,
    }


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'escalations')


async def _dispatch(scheduler: _SchedulerDouble, queue: EscalationQueue) -> WorkflowOutcome:
    runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
    return await runner.run(TaskAssignment(task_id=TASK_ID, task=scheduler.task, modules=[]))


def _pending(queue: EscalationQueue) -> list[Escalation]:
    return queue.get_by_task(TASK_ID, status='pending')


def _resolve_pending(queue: EscalationQueue) -> Escalation:
    pending = _pending(queue)
    assert len(pending) == 1, pending
    escalation = pending[0]
    queue.resolve(
        escalation.id, 'held behind its precondition',
        resolved_by='escalation-watcher', resolution_action='resume',
    )
    return escalation


async def _fire_and_resolve(scheduler: _SchedulerDouble, queue: EscalationQueue) -> Escalation:
    assert await _dispatch(scheduler, queue) == WorkflowOutcome.BLOCKED
    fired = _pending(queue)
    assert [e.category for e in fired] == [GATE_CATEGORY]
    return _resolve_pending(queue)


def _assert_refired(
    scheduler: _SchedulerDouble, queue: EscalationQueue, outcome: WorkflowOutcome,
    first: Escalation,
) -> Escalation:
    assert outcome == WorkflowOutcome.BLOCKED
    assert scheduler.done_writes() == []
    assert scheduler.statuses()[-1]['status'] == 'blocked'
    pending = _pending(queue)
    assert len(pending) == 1, pending
    refiled = pending[0]
    assert refiled.category == GATE_CATEGORY
    assert refiled.id != first.id
    return refiled


def _assert_closed_citing(
    scheduler: _SchedulerDouble, outcome: WorkflowOutcome, escalation_id: str,
) -> None:
    assert outcome == WorkflowOutcome.DONE
    assert scheduler.done_writes() == [{
        'status': 'done',
        'done_provenance': {
            'kind': 'deterministic-gate',
            'note': 'pure gate resolved',
            'escalation_id': escalation_id,
        },
    }]


@pytest.mark.asyncio
async def test_gate_held_behind_a_new_dependency_refires_when_it_lands(queue):
    scheduler = _SchedulerDouble(_gate_task(deps=[10]))
    first = await _fire_and_resolve(scheduler, queue)
    scheduler.task['dependencies'] = [10, 20]

    outcome = await _dispatch(scheduler, queue)

    _assert_refired(scheduler, queue, outcome, first)


@pytest.mark.asyncio
async def test_refired_gate_closes_on_its_next_resolution(queue):
    scheduler = _SchedulerDouble(_gate_task(deps=[10]))
    first = await _fire_and_resolve(scheduler, queue)
    scheduler.task['dependencies'] = [10, 20]
    refiled = _assert_refired(scheduler, queue, await _dispatch(scheduler, queue), first)
    _resolve_pending(queue)

    outcome = await _dispatch(scheduler, queue)

    _assert_closed_citing(scheduler, outcome, refiled.id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('milestone_at_fire', 'milestone_after_resolve'),
    [
        pytest.param(None, {'mode': 'delayed', 'after_secs': 3600}, id='milestone-added'),
        pytest.param(
            {'mode': 'dated', 'at': '2026-11-12T00:00:00+00:00'},
            {'mode': 'dated', 'at': '2026-12-01T00:00:00+00:00'},
            id='milestone-moved',
        ),
    ],
)
async def test_gate_held_behind_a_moved_milestone_refires(
    queue, milestone_at_fire, milestone_after_resolve,
):
    scheduler = _SchedulerDouble(_gate_task(deps=[10], milestone=milestone_at_fire))
    first = await _fire_and_resolve(scheduler, queue)
    scheduler.task['metadata']['milestone'] = milestone_after_resolve

    outcome = await _dispatch(scheduler, queue)

    _assert_refired(scheduler, queue, outcome, first)


@pytest.mark.asyncio
async def test_gate_held_behind_a_new_external_dependency_refires(queue):
    scheduler = _SchedulerDouble(_gate_task(deps=[10]))
    first = await _fire_and_resolve(scheduler, queue)
    scheduler.task['metadata']['external_deps'] = ['reify:42']

    outcome = await _dispatch(scheduler, queue)

    _assert_refired(scheduler, queue, outcome, first)


@pytest.mark.asyncio
async def test_unchanged_preconditions_drive_to_done(queue):
    scheduler = _SchedulerDouble(_gate_task(deps=[{'id': 10}]))
    first = await _fire_and_resolve(scheduler, queue)

    outcome = await _dispatch(scheduler, queue)

    _assert_closed_citing(scheduler, outcome, first.id)


@pytest.mark.asyncio
async def test_removed_dependency_does_not_refire(queue):
    scheduler = _SchedulerDouble(_gate_task(deps=[10, 11]))
    first = await _fire_and_resolve(scheduler, queue)
    scheduler.task['dependencies'] = [10]

    outcome = await _dispatch(scheduler, queue)

    _assert_closed_citing(scheduler, outcome, first.id)


@pytest.mark.asyncio
async def test_unreadable_preconditions_stamp_refires(queue):
    from orchestrator.gate_preconditions import GATE_ESCALATED_PRECONDITIONS_KEY

    scheduler = _SchedulerDouble(_gate_task(deps=[10]))
    first = await _fire_and_resolve(scheduler, queue)
    scheduler.task['metadata'][GATE_ESCALATED_PRECONDITIONS_KEY] = 'hand-edited'

    outcome = await _dispatch(scheduler, queue)

    _assert_refired(scheduler, queue, outcome, first)


@pytest.mark.asyncio
async def test_gate_stamped_before_preconditions_were_recorded_still_closes(queue):
    scheduler = _SchedulerDouble(
        _gate_task(deps=[10, 20], gate_escalated_at='2026-11-01T00:00:00+00:00'),
    )
    legacy = Escalation(
        id=queue.make_id(TASK_ID),
        task_id=TASK_ID,
        agent_role=DETERMINISTIC_AGENT_ROLE,
        severity='critical',
        category=GATE_CATEGORY,
        summary='Approve the rollout',
        level=2,
    )
    queue.submit(legacy)
    _resolve_pending(queue)

    outcome = await _dispatch(scheduler, queue)

    _assert_closed_citing(scheduler, outcome, legacy.id)
