"""DeterministicRunner's predicate verdict stamps and exit-75 re-arm (PRD C1).

plans/completion-driven-triggers-prd.md C1. Every row drives
``DeterministicRunner.run`` only, against an in-memory scheduler double whose
``update_task`` merges into the task's metadata as the real merge write does,
a real ``EscalationQueue`` and a real ``EventStore``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from escalation.authority import L2_AUTO_CLOSE_DENY_CATEGORIES
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.milestone_rearm import scheduled_fire_at

from orchestrator.deterministic_runner import DETERMINISTIC_AGENT_ROLE, DeterministicRunner
from orchestrator.event_store import EventStore, EventType
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import WorkflowOutcome

TASK_ID = '880'
DATED = {'mode': 'dated', 'at': '2026-10-25T00:00:00+00:00'}
FIXED_NOW = datetime(2026, 10, 25, 6, 0, tzinfo=UTC)
NOT_YET = 'probe log line\n{"run":"R","verdict":"not_yet","share":0.63}'
NOT_YET_VERDICT = {'run': 'R', 'verdict': 'not_yet', 'share': 0.63}


@dataclass
class _Clock:
    """The runner's injected clock, advanced by hand between runs."""

    now: datetime = FIXED_NOW

    def __call__(self) -> datetime:
        return self.now


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


DECLARED_MILESTONE = {**DATED, 'x_note': 'kept across the re-arm'}


def _declared_task(**stamps: Any) -> dict:
    return _predicate_task(recheck_secs=86400, milestone=dict(DECLARED_MILESTONE), **stamps)


def _seed_resolved_escalation(queue: EscalationQueue) -> str:
    escalation = Escalation(
        id=queue.make_id(TASK_ID),
        task_id=TASK_ID,
        agent_role=DETERMINISTIC_AGENT_ROLE,
        severity='critical',
        category='milestone_check_failed',
        summary='Milestone predicate check failed (rc=1)',
        detail='earlier failure',
        level=2,
    )
    queue.submit(escalation)
    queue.resolve(escalation.id, 'fixed upstream')
    return escalation.id


def _own_records(queue: EscalationQueue) -> list:
    return queue.get_by_task(TASK_ID, agent_role=DETERMINISTIC_AGENT_ROLE)


def _rearm_kwargs(clock: _Clock, cap: int = 30) -> dict:
    return {'clock': clock, 'predicate_recheck_stall_cap': cap}


@pytest.mark.asyncio
class TestRearmOnNotYet:
    async def test_declared_not_yet_advances_the_dated_milestone(
        self, queue, event_store, caplog,
    ):
        scheduler = _scheduler(_declared_task(), event_store)
        next_at = (FIXED_NOW + timedelta(seconds=86400)).isoformat()

        with caplog.at_level(logging.INFO, logger='orchestrator.deterministic_runner'):
            outcome = await _run(
                scheduler, queue, _ScriptRunner(75, NOT_YET), **_rearm_kwargs(_Clock()),
            )

        assert outcome == WorkflowOutcome.REQUEUED
        assert [kind for kind, _ in scheduler.writes] == ['update_task', 'set_task_status']
        [(_, write)] = [w for w in scheduler.writes if w[0] == 'update_task']
        assert write['mode'] == 'merge'
        payload = write['metadata']
        assert set(payload) == {'milestone', 'predicate_recheck', 'predicate_verdict'}
        assert payload['milestone'] == {
            **DECLARED_MILESTONE, 'at': next_at, 'scheduled_fire_at': DATED['at'],
        }
        assert payload['predicate_recheck'] == {
            'consecutive': 1,
            'first_rearmed_at': FIXED_NOW.isoformat(),
            'last_checked_at': FIXED_NOW.isoformat(),
            'next_check_at': next_at,
        }
        assert payload['predicate_verdict'] == NOT_YET_VERDICT
        assert scheduler.statuses() == [{'status': 'pending'}]
        assert _own_records(queue) == []
        [event] = event_store.fetch_events_by_type(EventType.predicate_rearmed)
        assert event['task_id'] == TASK_ID
        assert set(event['data']) == {
            'rc', 'consecutive', 'stall_cap', 'recheck_secs', 'next_check_at', 'verdict',
        }
        assert (event['data']['rc'], event['data']['consecutive'], event['data']['stall_cap']) == (
            75, 1, 30,
        )
        assert f'predicate re-armed (rc=75, consecutive=1/30, next_check_at={next_at})' in (
            caplog.text
        )

    async def test_a_second_not_yet_extends_the_streak(self, queue, event_store):
        scheduler = _scheduler(_declared_task(), event_store)
        clock = _Clock()
        await _run(scheduler, queue, _ScriptRunner(75, NOT_YET), **_rearm_kwargs(clock))
        clock.now = FIXED_NOW + timedelta(days=1)

        outcome = await _run(scheduler, queue, _ScriptRunner(75, NOT_YET), **_rearm_kwargs(clock))

        assert outcome == WorkflowOutcome.REQUEUED
        record = scheduler.task['metadata']['predicate_recheck']
        assert record['consecutive'] == 2
        assert record['first_rearmed_at'] == FIXED_NOW.isoformat()
        assert record['last_checked_at'] == clock.now.isoformat()
        milestone = scheduler.task['metadata']['milestone']
        assert milestone['at'] == (clock.now + timedelta(seconds=86400)).isoformat()
        assert scheduled_fire_at(milestone) == DATED['at']


@pytest.mark.asyncio
class TestUndeclaredNotYetIsAFailure:
    @pytest.mark.parametrize(
        ('task_fields', 'gap_fragment'),
        [
            ({'milestone': DATED}, 'before_done.recheck_secs'),
            (
                {'recheck_secs': 86400, 'milestone': {'mode': 'delayed', 'after_secs': 60}},
                "mode='dated'",
            ),
            (
                {'recheck_secs': 10, 'milestone': DATED},
                'before_done.recheck_secs: Input should be greater than or equal to 300',
            ),
        ],
        ids=['no-recheck-secs', 'delayed-milestone', 'invalid-before-done'],
    )
    async def test_files_milestone_check_failed_naming_the_gap(
        self, queue, event_store, task_fields, gap_fragment,
    ):
        scheduler = _scheduler(_predicate_task(**task_fields), event_store)

        outcome = await _run(
            scheduler, queue, _ScriptRunner(75, NOT_YET), **_rearm_kwargs(_Clock()),
        )

        assert outcome == WorkflowOutcome.BLOCKED
        [escalation] = _pending(queue)
        assert escalation.category == 'milestone_check_failed'
        assert escalation.summary == 'Milestone predicate check failed (rc=75)'
        detail_lines = escalation.detail.splitlines()
        exit_line = detail_lines.index('Predicate check exit code: rc=75')
        assert gap_fragment in detail_lines[exit_line + 1]
        assert event_store.fetch_events_by_type(EventType.predicate_rearmed) == []
        assert 'milestone' not in scheduler.updates()[0]


@pytest.mark.asyncio
class TestRearmAfterAResolvedGate:
    async def test_a_stale_gate_stamp_re_runs_and_re_arms(self, queue, event_store):
        stale = '2026-10-20T00:00:00+00:00'
        _seed_resolved_escalation(queue)
        scheduler = _scheduler(_declared_task(gate_escalated_at=stale), event_store)
        script = _ScriptRunner(75, NOT_YET)
        clock = _Clock()

        outcome = await _run(scheduler, queue, script, **_rearm_kwargs(clock))

        assert outcome == WorkflowOutcome.REQUEUED
        assert scheduler.task['metadata']['gate_escalated_at'] == stale
        assert len(_own_records(queue)) == 1

        clock.now = FIXED_NOW + timedelta(days=1)
        outcome = await _run(scheduler, queue, script, **_rearm_kwargs(clock))

        assert outcome == WorkflowOutcome.REQUEUED
        assert script.calls == 2
        assert scheduler.task['metadata']['predicate_recheck']['consecutive'] == 2
        assert len(_own_records(queue)) == 1


@pytest.mark.asyncio
class TestRearmWriteFailures:
    async def test_a_lost_status_write_still_requeues(self, queue, event_store):
        scheduler = _scheduler(
            _declared_task(), event_store, status_error=RuntimeError('connection severed'),
        )

        outcome = await _run(
            scheduler, queue, _ScriptRunner(75, NOT_YET), **_rearm_kwargs(_Clock()),
        )

        assert outcome == WorkflowOutcome.REQUEUED
        assert _own_records(queue) == []
        assert scheduler.task['metadata']['milestone']['at'] == (
            FIXED_NOW + timedelta(seconds=86400)
        ).isoformat()

    async def test_a_refused_metadata_write_escalates_instead_of_re_dispatching(
        self, queue, event_store,
    ):
        scheduler = _scheduler(_declared_task(), event_store, update_ok=False)

        outcome = await _run(
            scheduler, queue, _ScriptRunner(75, NOT_YET), **_rearm_kwargs(_Clock()),
        )

        assert outcome == WorkflowOutcome.BLOCKED
        assert [s['status'] for s in scheduler.statuses()] == ['blocked']
        [escalation] = _pending(queue)
        assert escalation.category == 'infra_issue'
        assert escalation.level == 2
        assert 're-arm write' in escalation.summary
        assert len(scheduler.updates()) == 1
        assert 'gate_escalated_at' not in scheduler.updates()[0]
        assert event_store.fetch_events_by_type(EventType.predicate_rearmed) == []

    async def test_an_unreadable_fire_time_escalates_instead_of_re_arming(
        self, queue, event_store,
    ):
        milestone = {**DECLARED_MILESTONE, 'scheduled_fire_at': None}
        scheduler = _scheduler(
            _predicate_task(recheck_secs=86400, milestone=milestone), event_store,
        )

        outcome = await _run(
            scheduler, queue, _ScriptRunner(75, NOT_YET), **_rearm_kwargs(_Clock()),
        )

        assert outcome == WorkflowOutcome.BLOCKED
        assert [s['status'] for s in scheduler.statuses()] == ['blocked']
        [escalation] = _pending(queue)
        assert escalation.category == 'infra_issue'
        assert 're-arm refused' in escalation.summary
        assert 'ISO-8601 scheduled fire time' in escalation.detail
        assert scheduler.updates() == []
        assert event_store.fetch_events_by_type(EventType.predicate_rearmed) == []


STALL_SUMMARY = 'Milestone predicate still not satisfied after {cap} consecutive checks'


@pytest.mark.asyncio
class TestStallCap:
    async def test_reaching_the_cap_files_one_stall_escalation(self, queue, event_store):
        scheduler = _scheduler(_declared_task(), event_store)
        script = _ScriptRunner(75, NOT_YET)
        clock = _Clock()
        first = await _run(scheduler, queue, script, **_rearm_kwargs(clock, cap=2))
        assert first == WorkflowOutcome.REQUEUED
        milestone_after_rearm = dict(scheduler.task['metadata']['milestone'])
        clock.now = FIXED_NOW + timedelta(days=1)
        writes_before = len(scheduler.writes)

        outcome = await _run(scheduler, queue, script, **_rearm_kwargs(clock, cap=2))

        assert outcome == WorkflowOutcome.BLOCKED
        [escalation] = _pending(queue)
        assert escalation.category == 'milestone_check_stalled'
        assert escalation.level == 2
        assert escalation.severity == 'critical'
        assert escalation.agent_role == DETERMINISTIC_AGENT_ROLE
        assert escalation.category in L2_AUTO_CLOSE_DENY_CATEGORIES
        assert escalation.summary == STALL_SUMMARY.format(cap=2)
        for fragment in (
            'streak 2/2',
            FIXED_NOW.isoformat(),
            str(timedelta(days=1)),
            '86400',
            json.dumps(NOT_YET_VERDICT, separators=(',', ':')),
            'kept across',
            'resume',
            'abandon',
        ):
            assert fragment in escalation.detail, fragment
        stall_writes = scheduler.writes[writes_before:]
        assert [kind for kind, _ in stall_writes] == ['update_task', 'set_task_status']
        [payload] = [w['metadata'] for kind, w in stall_writes if kind == 'update_task']
        assert set(payload) == {'gate_escalated_at', 'predicate_recheck', 'predicate_verdict'}
        assert payload['gate_escalated_at'] == clock.now.isoformat()
        assert payload['predicate_recheck'] == {
            'consecutive': 0,
            'first_rearmed_at': FIXED_NOW.isoformat(),
            'last_checked_at': clock.now.isoformat(),
            'next_check_at': None,
        }
        assert payload['predicate_verdict'] == NOT_YET_VERDICT
        assert scheduler.task['metadata']['milestone'] == milestone_after_rearm
        assert stall_writes[-1][1]['status'] == 'blocked'
        [event] = event_store.fetch_events_by_type(EventType.predicate_recheck_stalled)
        assert event['task_id'] == TASK_ID
        assert (
            event['data']['rc'],
            event['data']['consecutive'],
            event['data']['stall_cap'],
            event['data']['next_check_at'],
        ) == (75, 2, 2, None)
        assert len(event_store.fetch_events_by_type(EventType.predicate_rearmed)) == 1

    async def test_a_pending_stall_is_quiescent(self, queue, event_store):
        scheduler = _scheduler(_declared_task(), event_store)
        script = _ScriptRunner(75, NOT_YET)
        clock = _Clock()
        await _run(scheduler, queue, script, **_rearm_kwargs(clock, cap=2))
        await _run(scheduler, queue, script, **_rearm_kwargs(clock, cap=2))

        outcome = await _run(scheduler, queue, script, **_rearm_kwargs(clock, cap=2))

        assert outcome == WorkflowOutcome.BLOCKED
        assert script.calls == 2
        assert len(_own_records(queue)) == 1

    async def test_resume_after_a_stall_restarts_the_streak(self, queue, event_store):
        scheduler = _scheduler(_declared_task(), event_store)
        script = _ScriptRunner(75, NOT_YET)
        clock = _Clock()
        await _run(scheduler, queue, script, **_rearm_kwargs(clock, cap=2))
        clock.now = FIXED_NOW + timedelta(days=1)
        await _run(scheduler, queue, script, **_rearm_kwargs(clock, cap=2))
        [stall] = _pending(queue)
        queue.resolve(stall.id, 'resume')
        clock.now = FIXED_NOW + timedelta(days=2)

        outcome = await _run(scheduler, queue, script, **_rearm_kwargs(clock, cap=2))

        assert outcome == WorkflowOutcome.REQUEUED
        assert script.calls == 3
        record = scheduler.task['metadata']['predicate_recheck']
        assert record['consecutive'] == 1
        assert record['first_rearmed_at'] == FIXED_NOW.isoformat()
        assert len(_own_records(queue)) == 1
        assert _pending(queue) == []

    async def test_a_cap_of_one_stalls_on_the_first_not_yet(self, queue, event_store):
        scheduler = _scheduler(_declared_task(), event_store)

        outcome = await _run(
            scheduler, queue, _ScriptRunner(75, NOT_YET), **_rearm_kwargs(_Clock(), cap=1),
        )

        assert outcome == WorkflowOutcome.BLOCKED
        [escalation] = _pending(queue)
        assert escalation.category == 'milestone_check_stalled'
        assert escalation.summary == STALL_SUMMARY.format(cap=1)
        record = scheduler.task['metadata']['predicate_recheck']
        assert record['consecutive'] == 0
        assert record['first_rearmed_at'] is None
        assert scheduler.task['metadata']['milestone'] == DECLARED_MILESTONE
        assert event_store.fetch_events_by_type(EventType.predicate_rearmed) == []
        [event] = event_store.fetch_events_by_type(EventType.predicate_recheck_stalled)
        assert (event['data']['consecutive'], event['data']['stall_cap']) == (1, 1)
