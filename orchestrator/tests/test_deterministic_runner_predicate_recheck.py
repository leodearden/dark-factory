"""DeterministicRunner's predicate verdict stamps and exit-75 re-arm (PRD C1).

plans/completion-driven-triggers-prd.md C1. Every row drives
``DeterministicRunner.run`` only, against an in-memory scheduler double whose
``update_task`` merges into the task's metadata as the real merge write does,
a real ``EscalationQueue`` and a real ``EventStore``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from escalation.queue import EscalationQueue

from orchestrator.deterministic_runner import DeterministicRunner
from orchestrator.event_store import EventStore
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import WorkflowOutcome

TASK_ID = '880'
DATED = {'mode': 'dated', 'at': '2026-10-25T00:00:00+00:00'}


@dataclass
class _SchedulerDouble:
    """The scheduler surface the runner touches, recording every write in order."""

    task: dict
    event_store: EventStore
    update_ok: bool = True
    status_error: Exception | None = None
    writes: list[tuple[str, Any]] = field(default_factory=list)

    async def get_task(self, task_id: str) -> dict:
        return self.task

    async def update_task(
        self, task_id: str, metadata: dict, *, metadata_mode: str | None = None,
        append: bool = False,
    ) -> bool:
        self.writes.append(('update_task', {'metadata': dict(metadata), 'mode': metadata_mode}))
        if self.update_ok:
            self.task['metadata'].update(metadata)
        return self.update_ok

    async def set_task_status(self, task_id: str, status: str, **kwargs: Any) -> None:
        self.writes.append(('set_task_status', {'status': status, **kwargs}))
        if self.status_error is not None:
            raise self.status_error

    def updates(self) -> list[dict]:
        return [w['metadata'] for kind, w in self.writes if kind == 'update_task']

    def statuses(self) -> list[dict]:
        return [w for kind, w in self.writes if kind == 'set_task_status']


@dataclass
class _ScriptRunner:
    """An injected script runner returning a scripted ``(rc, out)``."""

    rc: int
    out: str
    calls: int = 0

    async def __call__(self, before_done: dict) -> tuple[int, str]:
        self.calls += 1
        return self.rc, self.out


def _predicate_task(
    *,
    recheck_secs: int | None = None,
    milestone: dict | None = None,
    **stamps: Any,
) -> dict:
    before_done: dict = {'kind': 'predicate', 'script': 'check.sh', 'timeout_secs': 60}
    if recheck_secs is not None:
        before_done['recheck_secs'] = recheck_secs
    metadata: dict = {'task_kind': 'deterministic', 'before_done': before_done, **stamps}
    if milestone is not None:
        metadata['milestone'] = milestone
    return {
        'id': TASK_ID,
        'title': 'Wait for the share to cross 0.8',
        'description': 'Predicate gate over the rollout share',
        'metadata': metadata,
    }


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'escalations')


@pytest.fixture
def event_store(tmp_path: Path) -> EventStore:
    return EventStore(tmp_path / 'events.db', run_id='t')


def _scheduler(task: dict, event_store: EventStore, **kwargs: Any) -> _SchedulerDouble:
    return _SchedulerDouble(task=task, event_store=event_store, **kwargs)


async def _run(
    scheduler: _SchedulerDouble, queue: EscalationQueue, script: _ScriptRunner, **kwargs: Any,
) -> WorkflowOutcome:
    runner = DeterministicRunner(
        scheduler=scheduler, escalation_queue=queue, script_runner=script, **kwargs,
    )
    return await runner.run(TaskAssignment(task_id=TASK_ID, task=scheduler.task, modules=[]))


def _pending(queue: EscalationQueue) -> list:
    return queue.get_by_task(TASK_ID, status='pending')


@pytest.mark.asyncio
class TestVerdictStampOnPass:
    async def test_trailing_json_object_is_stamped_before_done(self, queue, event_store):
        scheduler = _scheduler(_predicate_task(), event_store)
        script = _ScriptRunner(0, 'log line\n{"verdict":"pass","share":0.8}')

        outcome = await _run(scheduler, queue, script)

        assert outcome == WorkflowOutcome.DONE
        assert [kind for kind, _ in scheduler.writes] == ['update_task', 'set_task_status']
        assert scheduler.updates() == [{'predicate_verdict': {'verdict': 'pass', 'share': 0.8}}]
        done = scheduler.statuses()[0]
        assert done['status'] == 'done'
        assert done['done_provenance']['note'] == (
            'predicate check passed (rc=0): {"verdict":"pass","share":0.8}'
        )

    @pytest.mark.parametrize(
        'out', ['log line\n[1, 2]', 'check ok: 0 flakes'], ids=['json-array', 'clean-line'],
    )
    async def test_no_object_verdict_writes_no_metadata(self, queue, event_store, out):
        scheduler = _scheduler(_predicate_task(), event_store)

        outcome = await _run(scheduler, queue, _ScriptRunner(0, out))

        assert outcome == WorkflowOutcome.DONE
        assert scheduler.updates() == []

    async def test_oversized_verdict_is_elided_with_its_size(self, queue, event_store):
        verdict = {'detail': 'x' * 500}
        compact = json.dumps(verdict, separators=(',', ':'))
        scheduler = _scheduler(_predicate_task(), event_store)

        await _run(scheduler, queue, _ScriptRunner(0, f'log\n{json.dumps(verdict)}'))

        assert scheduler.updates() == [
            {'predicate_verdict': {'elided': True, 'chars': len(compact)}},
        ]

    async def test_a_refused_stamp_never_blocks_done(self, queue, event_store):
        scheduler = _scheduler(_predicate_task(), event_store, update_ok=False)

        outcome = await _run(scheduler, queue, _ScriptRunner(0, '{"verdict":"pass"}'))

        assert outcome == WorkflowOutcome.DONE
        assert scheduler.statuses()[-1]['status'] == 'done'


@pytest.mark.asyncio
class TestVerdictStampOnFailure:
    async def test_json_verdict_rides_in_the_single_gate_write(self, queue, event_store):
        scheduler = _scheduler(_predicate_task(), event_store)

        outcome = await _run(scheduler, queue, _ScriptRunner(1, 'log\n{"verdict":"fail"}'))

        assert outcome == WorkflowOutcome.BLOCKED
        [payload] = scheduler.updates()
        assert set(payload) == {'gate_escalated_at', 'predicate_verdict'}
        assert payload['predicate_verdict'] == {'verdict': 'fail'}
        [escalation] = _pending(queue)
        assert escalation.category == 'milestone_check_failed'

    async def test_legacy_failure_writes_only_the_gate_stamp(self, queue, event_store):
        scheduler = _scheduler(_predicate_task(), event_store)

        await _run(scheduler, queue, _ScriptRunner(1, 'FAIL: 3 merge flakes detected'))

        [payload] = scheduler.updates()
        assert set(payload) == {'gate_escalated_at'}

    async def test_failure_resets_a_running_streak(self, queue, event_store):
        streak = {
            'consecutive': 3,
            'first_rearmed_at': '2026-10-20T00:00:00+00:00',
            'last_checked_at': '2026-10-24T00:00:00+00:00',
            'next_check_at': '2026-10-25T00:00:00+00:00',
        }
        task = _predicate_task(recheck_secs=86400, milestone=DATED, predicate_recheck=streak)
        scheduler = _scheduler(task, event_store)

        await _run(scheduler, queue, _ScriptRunner(1, 'FAIL'))

        [payload] = scheduler.updates()
        assert set(payload) == {'gate_escalated_at', 'predicate_recheck'}
        record = payload['predicate_recheck']
        assert record['consecutive'] == 0
        assert record['first_rearmed_at'] == '2026-10-20T00:00:00+00:00'
        assert record['next_check_at'] is None

    async def test_failure_without_json_clears_a_stale_verdict(self, queue, event_store):
        task = _predicate_task(predicate_verdict={'verdict': 'stale'})
        scheduler = _scheduler(task, event_store)

        await _run(scheduler, queue, _ScriptRunner(1, 'FAIL'))

        [payload] = scheduler.updates()
        assert set(payload) == {'gate_escalated_at', 'predicate_verdict'}
        assert payload['predicate_verdict'] is None
