"""The phase writer for a scheduled self-deploy (task 7276).

A self-target deploy goes done while its restart is only scheduled, and its
dependents wait until ``deploy_state.phase`` reads verified.  The settler is
what moves the phase: VERIFIED once this orchestrator started after the
transient fired and the transient has finished, FAILED on an own-role
failure escalation or a failed unit, otherwise left at SCHEDULED.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.deploy_state import DeployPhase
from shared.task_read import TaskAbsent, TaskFound, TaskRead, TaskReadFailed

from orchestrator.scheduled_deploy_settle import (
    ScheduledDeploySettler,
    ScheduledRestart,
    restart_landed,
    settled_phase,
)

T0 = datetime(2026, 11, 9, 10, 0, tzinfo=UTC)
FIRE_DELAY = 60
FIRES_AT = T0 + timedelta(seconds=FIRE_DELAY)
STARTED_AFTER_FIRE = T0 + timedelta(minutes=5)
ESCALATED_AT = T0 + timedelta(seconds=12)
NOW = T0 + timedelta(minutes=10)
TRANSIENT = 'orch-redeploy-restart-7238.service'


def _stamp(**overrides: object) -> dict:
    return {'at': T0.isoformat(), 'transient_unit': TRANSIENT, 'fire_delay_secs': FIRE_DELAY, **overrides}


def _restart() -> ScheduledRestart:
    restart = ScheduledRestart.from_metadata({'before_done_scheduled_at': _stamp()})
    assert restart is not None
    return restart


class TestScheduledRestart:
    def test_parses_the_stamp_and_derives_the_fire_time(self) -> None:
        restart = _restart()
        assert restart.scheduled_at == T0
        assert restart.transient_unit == TRANSIENT
        assert restart.fire_delay_secs == FIRE_DELAY
        assert restart.fires_at == FIRES_AT

    @pytest.mark.parametrize(
        'metadata',
        [
            {},
            {'before_done_scheduled_at': 'yesterday'},
            {'before_done_scheduled_at': _stamp(at='not a time')},
            {'before_done_scheduled_at': _stamp(at='2026-11-09T10:00:00')},
            {'before_done_scheduled_at': _stamp(transient_unit='')},
            {'before_done_scheduled_at': _stamp(fire_delay_secs='60')},
            {'before_done_scheduled_at': _stamp(fire_delay_secs=-1)},
        ],
        ids=['missing', 'non-dict', 'unparseable-at', 'naive-at', 'empty-unit', 'str-delay', 'negative-delay'],
    )
    def test_an_unreadable_stamp_is_none(self, metadata: dict) -> None:
        assert ScheduledRestart.from_metadata(metadata) is None


@pytest.mark.parametrize(
    ('started_at', 'active_state', 'failure_escalations', 'expected'),
    [
        (STARTED_AFTER_FIRE, 'active', 1, DeployPhase.FAILED),
        (STARTED_AFTER_FIRE, 'inactive', 1, DeployPhase.FAILED),
        (STARTED_AFTER_FIRE, 'failed', 0, DeployPhase.FAILED),
        (T0, 'inactive', 0, DeployPhase.SCHEDULED),
        (FIRES_AT, 'inactive', 0, DeployPhase.SCHEDULED),
        (STARTED_AFTER_FIRE, 'active', 0, DeployPhase.SCHEDULED),
        (STARTED_AFTER_FIRE, 'activating', 0, DeployPhase.SCHEDULED),
        (STARTED_AFTER_FIRE, 'deactivating', 0, DeployPhase.SCHEDULED),
        (STARTED_AFTER_FIRE, '', 0, DeployPhase.SCHEDULED),
        (STARTED_AFTER_FIRE, 'inactive', 0, DeployPhase.VERIFIED),
    ],
    ids=[
        'failure-evidence-while-running',
        'failure-evidence',
        'transient-failed',
        'started-before-fire',
        'started-at-fire',
        'sweep-running',
        'sweep-activating',
        'sweep-deactivating',
        'inspect-wedged',
        'collected-after-fire',
    ],
)
def test_settled_phase(started_at, active_state, failure_escalations, expected) -> None:
    phase = settled_phase(
        _restart(),
        process_started_at=started_at,
        transient_active_state=active_state,
        failure_escalations=failure_escalations,
    )
    assert phase is expected


@pytest.mark.parametrize(
    ('started_at', 'active_state', 'expected'),
    [
        (T0, 'inactive', False),
        (FIRES_AT, 'inactive', False),
        (STARTED_AFTER_FIRE, 'active', False),
        (STARTED_AFTER_FIRE, 'activating', False),
        (STARTED_AFTER_FIRE, 'deactivating', False),
        (STARTED_AFTER_FIRE, '', False),
        (STARTED_AFTER_FIRE, 'failed', False),
        (STARTED_AFTER_FIRE, 'inactive', True),
    ],
    ids=[
        'started-before-fire',
        'started-at-fire',
        'sweep-running',
        'sweep-activating',
        'sweep-deactivating',
        'inspect-wedged',
        'transient-failed',
        'collected-after-fire',
    ],
)
def test_restart_landed(started_at, active_state, expected) -> None:
    landed = restart_landed(_restart(), process_started_at=started_at, transient_active_state=active_state)
    assert landed is expected


# ---------------------------------------------------------------------------
# settle_pass, with fakes for the task store, the lookup and the unit inspector
# ---------------------------------------------------------------------------


def _deploy_task(
    task_id: str = '7238', *, phase: str = 'scheduled', status: str = 'done', metadata: dict | None = None,
) -> dict:
    defaults = {
        'done_provenance': {
            'kind': 'deterministic-deploy-scheduled',
            'unit': 'orchestrator-dark-factory.service',
            'transient_unit': TRANSIENT,
            'fire_delay_secs': FIRE_DELAY,
        },
        'before_done_scheduled_at': _stamp(),
        'deploy_state': {
            'phase': phase,
            'ran_at': T0.isoformat(),
            'verify_baseline': None,
            'verified_at': None,
            'escalated_at': None,
        },
    }
    return {'id': task_id, 'status': status, 'metadata': {**defaults, **(metadata or {})}}


class _Lookup:
    def __init__(self, rows: list[dict], error: Exception | None = None) -> None:
        self.rows = rows
        self.error = error
        self.calls: list[tuple] = []

    async def find_tasks_by_metadata(self, key, value, statuses, *, include_value=False):
        self.calls.append((key, value, tuple(statuses), include_value))
        if self.error is not None:
            return [], self.error
        return self.rows, None


class _Store:
    def __init__(
        self,
        tasks: dict[str, dict],
        *,
        update_ok: bool = True,
        raises_for: str | None = None,
        fails_for: str | None = None,
    ):
        self.tasks = tasks
        self.update_ok = update_ok
        self.raises_for = raises_for
        self.fails_for = fails_for
        self.reads: list[str] = []
        self.updates: list[tuple[str, dict, str]] = []

    async def read_task(self, task_id: str) -> TaskRead:
        self.reads.append(task_id)
        if task_id == self.raises_for:
            raise RuntimeError('fused-memory down')
        if task_id == self.fails_for:
            return TaskReadFailed('get_task error=fused-memory down')
        task = self.tasks.get(task_id)
        return TaskAbsent() if task is None else TaskFound(task)

    async def update_task(self, task_id: str, metadata: dict, *, metadata_mode: str) -> bool:
        self.updates.append((task_id, metadata, metadata_mode))
        if self.update_ok:
            self.tasks[task_id]['metadata'].update(metadata)
        return self.update_ok


def _row(task: dict) -> dict:
    return {
        'task_id': task['id'],
        'title': 'self deploy',
        'status': task['status'],
        'priority': 'high',
        'updated_at': None,
        'matched_value': task['metadata']['deploy_state'],
    }


def _inspector(states: dict[str, str]):
    async def inspect(unit: str) -> dict:
        return {'ActiveState': states.get(unit, 'inactive'), 'MainPID': 0}

    return inspect


def _settler(
    store, lookup, queue, *, active_state: str = 'inactive', started_at=STARTED_AFTER_FIRE, now=NOW,
):
    return ScheduledDeploySettler(
        store,
        lookup,
        queue,
        process_started_at=started_at,
        unit_inspector=_inspector({TRANSIENT: active_state}),
        clock=lambda: now,
    )


def _own_role_escalation(
    queue: EscalationQueue,
    *,
    at: datetime,
    category: str = 'infra_issue',
    summary: str = 'RP-4: scheduled restart sweep failed',
) -> str:
    esc = Escalation(
        id=queue.make_id('7238'),
        task_id='7238',
        agent_role='orchestrator-deterministic',
        severity='critical',
        category=category,
        summary=summary,
        timestamp=at.isoformat(),
        level=2,
    )
    queue.submit(esc)
    return esc.id


def _escalated_task() -> dict:
    """The 2912 shape: done, scheduled provenance, phase escalated after a completion-failure filing."""
    task = _deploy_task(phase='escalated')
    task['metadata']['deploy_state']['escalated_at'] = ESCALATED_AT.isoformat()
    return task


def _phases_written(store: _Store) -> list[str]:
    return [metadata['deploy_state']['phase'] for _, metadata, _ in store.updates]


async def _settle_one(tmp_path, task: dict, **settler_kwargs) -> _Store:
    store = _Store({task['id']: task})
    queue = EscalationQueue(tmp_path)
    await _settler(store, _Lookup([_row(task)]), queue, **settler_kwargs).settle_pass()
    return store


class TestSettlePass:
    @pytest.mark.asyncio
    async def test_a_landed_restart_is_stamped_verified(self, tmp_path) -> None:
        task = _deploy_task()
        store = _Store({'7238': task})
        lookup = _Lookup([_row(task)])
        before = dict(task['metadata']['deploy_state'])

        await _settler(store, lookup, EscalationQueue(tmp_path)).settle_pass()

        assert lookup.calls == [('deploy_state', None, ('done',), True)]
        expected_state = {**before, 'phase': 'verified', 'verified_at': NOW.isoformat()}
        assert store.updates == [('7238', {'deploy_state': expected_state}, 'merge')]

    @pytest.mark.asyncio
    async def test_a_fire_time_escalation_stamps_failed_and_files_nothing(self, tmp_path, caplog) -> None:
        task = _deploy_task()
        store = _Store({'7238': task})
        queue = EscalationQueue(tmp_path)
        esc_id = _own_role_escalation(queue, at=T0 + timedelta(seconds=90))
        before = len(queue.get_by_task('7238'))

        with caplog.at_level(logging.ERROR, logger='orchestrator.scheduled_deploy_settle'):
            await _settler(store, _Lookup([_row(task)]), queue).settle_pass()

        [(task_id, metadata, mode)] = store.updates
        assert (task_id, metadata['deploy_state']['phase'], mode) == ('7238', 'failed', 'merge')
        assert any(
            r.levelno == logging.ERROR and esc_id in r.getMessage() and 'failure evidence' in r.getMessage()
            for r in caplog.records
        )
        assert len(queue.get_by_task('7238')) == before

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('at', 'category'),
        [(T0 - timedelta(minutes=1), 'infra_issue'), (T0 + timedelta(seconds=90), 'design_concern')],
        ids=['filed-before-the-schedule', 'other-category'],
    )
    async def test_an_unrelated_escalation_is_not_failure_evidence(self, tmp_path, at, category) -> None:
        task = _deploy_task()
        store = _Store({'7238': task})
        queue = EscalationQueue(tmp_path)
        _own_role_escalation(queue, at=at, category=category)

        await _settler(store, _Lookup([_row(task)]), queue).settle_pass()

        [(_, metadata, _)] = store.updates
        assert metadata['deploy_state']['phase'] == 'verified'

    @pytest.mark.asyncio
    async def test_a_running_transient_writes_nothing(self, tmp_path) -> None:
        store = await _settle_one(tmp_path, _deploy_task(), active_state='active')
        assert store.updates == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize('phase', ['verified', 'done'])
    async def test_a_row_already_past_scheduled_is_never_read(self, tmp_path, phase) -> None:
        task = _deploy_task(phase=phase)
        store = await _settle_one(tmp_path, task)
        assert store.reads == []
        assert store.updates == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'matched_value',
        [{'phase': 'failed'}, {'phase': 'escalated'}, {'phase': 'ran'}, 'scheduled'],
        ids=['failed', 'escalated', 'ran', 'non-dict'],
    )
    async def test_a_row_at_any_held_phase_is_read(self, tmp_path, matched_value) -> None:
        task = _deploy_task()
        store = _Store({'7238': task})
        row = {**_row(task), 'matched_value': matched_value}
        await _settler(store, _Lookup([row]), EscalationQueue(tmp_path)).settle_pass()
        assert store.reads == ['7238']

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'fresh',
        [
            _deploy_task(phase='verified'),
            _deploy_task(status='pending'),
            _deploy_task(metadata={'done_provenance': {'kind': 'deterministic-deploy'}}),
        ],
        ids=['fresh-phase-verified', 'fresh-status-not-done', 'fresh-provenance-not-scheduled'],
    )
    async def test_a_candidate_is_corroborated_before_acting(self, tmp_path, fresh) -> None:
        row = _row(_deploy_task())
        store = _Store({'7238': fresh})
        await _settler(store, _Lookup([row]), EscalationQueue(tmp_path)).settle_pass()
        assert store.reads == ['7238']
        assert store.updates == []

    @pytest.mark.asyncio
    async def test_a_failed_lookup_reads_and_writes_nothing(self, tmp_path, caplog) -> None:
        store = _Store({'7238': _deploy_task()})
        lookup = _Lookup([], error=RuntimeError('lookup down'))
        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduled_deploy_settle'):
            await _settler(store, lookup, EscalationQueue(tmp_path)).settle_pass()
        assert (store.reads, store.updates) == ([], [])
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    @pytest.mark.asyncio
    async def test_one_failing_candidate_does_not_stop_the_next(self, tmp_path) -> None:
        first, second = _deploy_task('7001'), _deploy_task('7238')
        store = _Store({'7001': first, '7238': second}, raises_for='7001')
        lookup = _Lookup([_row(first), _row(second)])

        await _settler(store, lookup, EscalationQueue(tmp_path)).settle_pass()

        assert [task_id for task_id, _, _ in store.updates] == ['7238']

    @pytest.mark.asyncio
    async def test_a_failed_re_read_is_reported_at_warning(self, tmp_path, caplog) -> None:
        task = _deploy_task()
        store = _Store({'7238': task}, fails_for='7238')
        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduled_deploy_settle'):
            await _settler(store, _Lookup([_row(task)]), EscalationQueue(tmp_path)).settle_pass()
        assert store.updates == []
        assert any(
            r.levelno == logging.WARNING and '7238' in r.getMessage() and 'fused-memory down' in r.getMessage()
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_a_vanished_candidate_writes_nothing(self, tmp_path) -> None:
        store = _Store({})
        await _settler(store, _Lookup([_row(_deploy_task())]), EscalationQueue(tmp_path)).settle_pass()
        assert (store.reads, store.updates) == (['7238'], [])

    @pytest.mark.asyncio
    async def test_an_unreadable_transient_is_reported_at_warning(self, tmp_path, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduled_deploy_settle'):
            store = await _settle_one(tmp_path, _deploy_task(), active_state='')
        assert store.updates == []
        assert any(
            r.levelno == logging.WARNING and '7238' in r.getMessage() and 'unreadable' in r.getMessage()
            for r in caplog.records
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('now', 'warned'),
        [(FIRES_AT + timedelta(hours=7), True), (FIRES_AT + timedelta(hours=5), False)],
        ids=['overdue', 'not-yet-overdue'],
    )
    async def test_a_restart_long_past_its_fire_time_is_reported_at_warning(
        self, tmp_path, caplog, now, warned,
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger='orchestrator.scheduled_deploy_settle'):
            store = await _settle_one(tmp_path, _deploy_task(), started_at=T0, now=now)
        assert store.updates == []
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and '7238' in r.getMessage()]
        assert bool(warnings) is warned, [r.getMessage() for r in caplog.records]

    @pytest.mark.asyncio
    async def test_a_refused_write_is_logged_and_the_pass_completes(self, tmp_path, caplog) -> None:
        task = _deploy_task()
        store = _Store({'7238': task}, update_ok=False)
        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduled_deploy_settle'):
            await _settler(store, _Lookup([_row(task)]), EscalationQueue(tmp_path)).settle_pass()
        assert len(store.updates) == 1
        assert any(r.levelno == logging.WARNING and '7238' in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_a_malformed_schedule_stamp_writes_nothing(self, tmp_path, caplog) -> None:
        task = _deploy_task(metadata={'before_done_scheduled_at': 'garbage'})
        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduled_deploy_settle'):
            store = await _settle_one(tmp_path, task)
        assert store.updates == []
        assert any(r.levelno == logging.WARNING and '7238' in r.getMessage() for r in caplog.records)


class TestHeldPhases:
    """Every phase that holds a dependent is one the settler advances, or reports."""

    @pytest.mark.asyncio
    async def test_an_escalated_deploy_with_no_pending_escalation_is_released_once_the_restart_landed(
        self, tmp_path, caplog,
    ) -> None:
        task = _escalated_task()
        before = task['metadata']['deploy_state']

        with caplog.at_level(logging.INFO, logger='orchestrator.scheduled_deploy_settle'):
            store = await _settle_one(tmp_path, task)

        assert store.updates == [('7238', {'deploy_state': {**before, 'phase': 'done'}}, 'merge')]
        assert before['escalated_at'] == ESCALATED_AT.isoformat()
        assert before['verified_at'] is None
        assert any(r.levelno == logging.INFO and '7238' in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_a_pending_own_role_escalation_holds_an_escalated_deploy_loudly(self, tmp_path, caplog) -> None:
        task = _escalated_task()
        store = _Store({'7238': task})
        queue = EscalationQueue(tmp_path)
        esc_id = _own_role_escalation(queue, at=ESCALATED_AT, summary='self-deploy completion write failed')
        settler = _settler(store, _Lookup([_row(task)]), queue)

        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduled_deploy_settle'):
            await settler.settle_pass()

        assert store.updates == []
        assert any(
            r.levelno == logging.WARNING and '7238' in r.getMessage() and esc_id in r.getMessage()
            for r in caplog.records
        )

        queue.resolve(esc_id, 'done write had landed')
        await settler.settle_pass()

        assert _phases_written(store) == ['done']

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('active_state', 'started_at'),
        [('active', STARTED_AFTER_FIRE), ('inactive', T0)],
        ids=['sweep-running', 'started-before-fire'],
    )
    async def test_an_escalated_deploy_waits_for_the_restart(self, tmp_path, active_state, started_at) -> None:
        store = await _settle_one(tmp_path, _escalated_task(), active_state=active_state, started_at=started_at)
        assert store.updates == []

    @pytest.mark.asyncio
    async def test_a_failed_deploy_is_routed_to_its_existing_escalation(self, tmp_path) -> None:
        task = _deploy_task(phase='failed')
        store = _Store({'7238': task})
        queue = EscalationQueue(tmp_path)
        _own_role_escalation(queue, at=T0 + timedelta(seconds=90))
        before = len(queue.get_by_task('7238'))

        await _settler(store, _Lookup([_row(task)]), queue).settle_pass()

        [(_, metadata, mode)] = store.updates
        assert (metadata['deploy_state']['phase'], mode) == ('escalated', 'merge')
        assert metadata['deploy_state']['escalated_at'] == NOW.isoformat()
        assert len(queue.get_by_task('7238')) == before

    @pytest.mark.asyncio
    async def test_a_failed_deploy_with_no_escalation_files_one(self, tmp_path) -> None:
        task = _deploy_task(phase='failed')
        store = _Store({'7238': task})
        queue = EscalationQueue(tmp_path)

        await _settler(store, _Lookup([_row(task)]), queue).settle_pass()

        [record] = queue.get_by_task('7238')
        assert (record.agent_role, record.category) == ('orchestrator-deterministic', 'infra_issue')
        assert (record.level, record.severity) == (2, 'critical')
        assert _phases_written(store) == ['escalated']

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('metadata', 'named_phase'),
        [
            ({'deploy_state': {'phase': 'ran', 'ran_at': T0.isoformat()}}, r'\bran\b'),
            ({'deploy_state': {'phase': 'bogus'}}, ''),
        ],
        ids=['ran', 'unreadable-slice'],
    )
    async def test_a_held_phase_no_writer_advances_is_reported_at_warning(
        self, tmp_path, caplog, metadata, named_phase,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduled_deploy_settle'):
            store = await _settle_one(tmp_path, _deploy_task(metadata=metadata))

        assert store.updates == []
        assert any(
            r.levelno >= logging.WARNING and '7238' in r.getMessage() and re.search(named_phase, r.getMessage())
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_a_fire_time_failure_is_held_until_its_escalation_is_resolved(self, tmp_path) -> None:
        task = _deploy_task()
        store = _Store({'7238': task})
        queue = EscalationQueue(tmp_path)
        esc_id = _own_role_escalation(queue, at=T0 + timedelta(seconds=90))
        settler = _settler(store, _Lookup([_row(task)]), queue)

        await settler.settle_pass()
        assert _phases_written(store) == ['failed']

        await settler.settle_pass()
        assert _phases_written(store) == ['failed', 'escalated']

        await settler.settle_pass()
        assert _phases_written(store) == ['failed', 'escalated']

        queue.resolve(esc_id, 'restarted by hand')
        await settler.settle_pass()
        assert _phases_written(store) == ['failed', 'escalated', 'done']
