"""The run()-exit contract: the pure judge and its recorder.

Spec ``docs/task-escalation-state-spec.md`` §5 / E11; the canonical WHY lives
in ``orchestrator/src/orchestrator/exit_contract.py`` (module docstring).
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from escalation.queue import EscalationQueue

from orchestrator.event_store import EventStore, EventType
from orchestrator.exit_contract import (
    ExitCheck,
    ExitStatusWriteFailure,
    ExitVerdict,
    ExitVerdictKind,
    judge_exit,
    record_exit_verdict,
)
from orchestrator.workflow_types import WorkflowOutcome, WorkflowState

pytestmark = pytest.mark.exit_contract_violation_expected

_LOGGER = 'orchestrator.exit_contract'
_TASK = '4242'

_DEAD_WRITE = ExitStatusWriteFailure(
    target_status='pending',
    error='set_task_status(…) failed after 4 transient retries: TimeoutError',
)


def _judge(
    outcome: object = WorkflowOutcome.BLOCKED,
    *,
    status_row: str | None = 'blocked',
    report_phase: WorkflowState = WorkflowState.BLOCKED,
    machine_state: WorkflowState = WorkflowState.BLOCKED,
    write_failure: ExitStatusWriteFailure | None = None,
) -> ExitVerdict:
    return judge_exit(
        outcome=outcome,
        report_phase=report_phase,
        machine_state=machine_state,
        status_row=status_row,
        write_failure=write_failure,
    )


class TestJudgeExit:
    @pytest.mark.parametrize(
        ('outcome', 'status_row', 'phase'),
        [
            (WorkflowOutcome.BLOCKED, 'blocked', WorkflowState.BLOCKED),
            (WorkflowOutcome.DONE, 'done', WorkflowState.DONE),
        ],
    )
    def test_allowed_pair_is_consistent(self, outcome, status_row, phase):
        verdict = _judge(
            outcome, status_row=status_row, report_phase=phase, machine_state=phase,
        )
        assert verdict.kind is ExitVerdictKind.CONSISTENT
        assert verdict.check is None

    def test_false_done_pair_is_an_outcome_status_violation(self):
        verdict = _judge(WorkflowOutcome.BLOCKED, status_row='done')
        assert verdict.kind is ExitVerdictKind.VIOLATION
        assert verdict.check is ExitCheck.OUTCOME_STATUS
        assert verdict.outcome is WorkflowOutcome.BLOCKED
        assert verdict.status_row == 'done'

    @pytest.mark.parametrize('status_row', [None, 'not-a-real-status'])
    def test_illegible_status_row_is_unchecked_not_consistent(self, status_row):
        verdict = _judge(WorkflowOutcome.BLOCKED, status_row=status_row)
        assert verdict.kind is ExitVerdictKind.UNCHECKED
        assert verdict.check is None

    def test_phase_mismatch_is_a_phase_violation_even_on_a_consistent_pair(self):
        verdict = _judge(
            WorkflowOutcome.BLOCKED,
            status_row='blocked',
            report_phase=WorkflowState.VERIFY,
            machine_state=WorkflowState.BLOCKED,
        )
        assert verdict.kind is ExitVerdictKind.VIOLATION
        assert verdict.check is ExitCheck.PHASE
        assert verdict.report_phase is WorkflowState.VERIFY
        assert verdict.machine_state is WorkflowState.BLOCKED

    def test_phase_check_is_evaluated_before_the_pair(self):
        verdict = _judge(
            WorkflowOutcome.BLOCKED,
            status_row='done',
            report_phase=WorkflowState.REVIEW,
            machine_state=WorkflowState.BLOCKED,
        )
        assert verdict.kind is ExitVerdictKind.VIOLATION
        assert verdict.check is ExitCheck.PHASE

    @pytest.mark.parametrize('status_row', ['blocked', 'done', None])
    def test_write_failure_reclassifies_any_pair_as_store_unavailable(self, status_row):
        verdict = _judge(
            WorkflowOutcome.BLOCKED, status_row=status_row, write_failure=_DEAD_WRITE,
        )
        assert verdict.kind is ExitVerdictKind.STORE_UNAVAILABLE
        assert verdict.check is None
        assert verdict.write_failure == _DEAD_WRITE

    def test_phase_mismatch_beats_a_write_failure(self):
        verdict = _judge(
            WorkflowOutcome.BLOCKED,
            status_row='blocked',
            report_phase=WorkflowState.EXECUTE,
            machine_state=WorkflowState.BLOCKED,
            write_failure=_DEAD_WRITE,
        )
        assert verdict.kind is ExitVerdictKind.VIOLATION
        assert verdict.check is ExitCheck.PHASE

    def test_outcome_with_no_contract_row_is_a_violation_not_an_exception(self):
        verdict = _judge(SimpleNamespace(value='bogus'), status_row='blocked')
        assert verdict.kind is ExitVerdictKind.VIOLATION
        assert verdict.check is ExitCheck.OUTCOME_STATUS


def _violation() -> ExitVerdict:
    return _judge(WorkflowOutcome.BLOCKED, status_row='done')


def _store_unavailable() -> ExitVerdict:
    return _judge(
        WorkflowOutcome.REQUEUED,
        status_row='in-progress',
        report_phase=WorkflowState.EXECUTE,
        machine_state=WorkflowState.EXECUTE,
        write_failure=_DEAD_WRITE,
    )


def _record(
    verdict: ExitVerdict,
    *,
    enforce: bool = False,
    event_store: EventStore | None = None,
    escalation_queue: EscalationQueue | None = None,
) -> None:
    record_exit_verdict(
        verdict,
        task_id=_TASK,
        enforce=enforce,
        event_store=event_store,
        escalation_queue=escalation_queue,
        worktree='/tmp/wt-4242',
        filing_claimant_run_id='run-abc',
    )


def _contract_events(event_store: MagicMock) -> list[dict]:
    return [
        c.kwargs['data']
        for c in event_store.emit.call_args_list
        if c.args and c.args[0] is EventType.workflow_exit_contract
    ]


def _contract_records(caplog: pytest.LogCaptureFixture, verdict: str) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records
        if r.name == _LOGGER and getattr(r, 'exit_contract_verdict', None) == verdict
    ]


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


class TestRecordExitVerdict:
    @pytest.mark.parametrize(
        'verdict',
        [
            _judge(WorkflowOutcome.BLOCKED, status_row='blocked'),
            _judge(WorkflowOutcome.BLOCKED, status_row=None),
        ],
        ids=['consistent', 'unchecked'],
    )
    def test_consistent_and_unchecked_record_nothing(self, verdict, caplog, queue):
        event_store = MagicMock()
        caplog.set_level(logging.DEBUG, logger=_LOGGER)
        _record(verdict, enforce=True, event_store=event_store, escalation_queue=queue)
        assert [r for r in caplog.records if r.name == _LOGGER] == []
        event_store.emit.assert_not_called()
        assert queue.get_by_task(_TASK) == []

    def test_log_mode_violation_warns_would_violate_and_emits_one_event(self, caplog, queue):
        event_store = MagicMock()
        caplog.set_level(logging.DEBUG, logger=_LOGGER)
        _record(_violation(), enforce=False, event_store=event_store, escalation_queue=queue)

        records = _contract_records(caplog, 'violation')
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        message = records[0].getMessage()
        assert 'would-violate' in message
        assert 'blocked' in message
        assert 'done' in message

        assert event_store.emit.call_count == 1
        call = event_store.emit.call_args
        assert call.args[0] is EventType.workflow_exit_contract
        assert call.kwargs['task_id'] == _TASK
        assert call.kwargs['data'] == {
            'verdict': 'violation',
            'mode': 'log',
            'check': 'outcome_status',
            'outcome': 'blocked',
            'status': 'done',
            'report_phase': 'blocked',
            'machine_state': 'blocked',
            'failed_write': None,
            'escalation_id': None,
        }
        assert queue.get_by_task(_TASK) == []

    def test_enforce_mode_violation_files_one_l1_and_names_it_on_the_event(self, caplog, queue):
        event_store = MagicMock()
        caplog.set_level(logging.DEBUG, logger=_LOGGER)
        _record(_violation(), enforce=True, event_store=event_store, escalation_queue=queue)

        filed = queue.get_by_task(_TASK, status='pending')
        assert len(filed) == 1
        esc = filed[0]
        assert esc.category == 'workflow_exit_contract'
        assert esc.severity == 'blocking'
        assert esc.level == 1
        assert esc.agent_role == 'orchestrator'
        assert 'blocked' in esc.summary
        assert 'done' in esc.summary
        assert esc.filing_claimant_run_id == 'run-abc'

        created = [
            c for c in event_store.emit.call_args_list
            if c.args and c.args[0] is EventType.escalation_created
        ]
        assert len(created) == 1
        assert created[0].kwargs['data']['escalation_id'] == esc.id
        assert created[0].kwargs['data']['category'] == 'workflow_exit_contract'

        events = _contract_events(event_store)
        assert len(events) == 1
        assert events[0]['mode'] == 'enforce'
        assert events[0]['verdict'] == 'violation'
        assert events[0]['escalation_id'] == esc.id
        assert len(_contract_records(caplog, 'violation')) == 1

    def test_enforce_mode_dedupes_against_an_open_l1_but_still_emits(self, queue):
        event_store = MagicMock()
        _record(_violation(), enforce=True, event_store=event_store, escalation_queue=queue)
        _record(_violation(), enforce=True, event_store=event_store, escalation_queue=queue)

        assert len(queue.get_by_task(_TASK, status='pending')) == 1
        events = _contract_events(event_store)
        assert len(events) == 2
        assert events[1]['mode'] == 'enforce'
        assert events[1]['escalation_id'] is None

    def test_only_a_literal_true_enforces(self, caplog, queue):
        event_store = MagicMock()
        caplog.set_level(logging.DEBUG, logger=_LOGGER)
        _record(_violation(), enforce=MagicMock(), event_store=event_store, escalation_queue=queue)

        assert queue.get_by_task(_TASK) == []
        records = _contract_records(caplog, 'violation')
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert 'would-violate' in records[0].getMessage()
        events = _contract_events(event_store)
        assert [e['mode'] for e in events] == ['log']

    @pytest.mark.parametrize('enforce', [False, True])
    def test_store_unavailable_is_one_record_and_never_an_escalation(
        self, enforce, caplog, queue,
    ):
        event_store = MagicMock()
        caplog.set_level(logging.DEBUG, logger=_LOGGER)
        _record(
            _store_unavailable(), enforce=enforce,
            event_store=event_store, escalation_queue=queue,
        )

        records = _contract_records(caplog, 'store_unavailable')
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert _contract_records(caplog, 'violation') == []
        events = _contract_events(event_store)
        assert len(events) == 1
        assert events[0]['verdict'] == 'store_unavailable'
        assert events[0]['failed_write'] == {
            'target_status': _DEAD_WRITE.target_status,
            'error': _DEAD_WRITE.error,
        }
        assert events[0]['escalation_id'] is None
        assert queue.get_by_task(_TASK) == []

    def test_a_failing_submit_is_logged_and_the_event_still_emitted(self, caplog):
        event_store = MagicMock()
        failing_queue = MagicMock()
        failing_queue.has_open_l1.return_value = False
        failing_queue.make_id.return_value = f'esc-{_TASK}-1'
        failing_queue.submit.side_effect = RuntimeError('queue dir unwritable')
        caplog.set_level(logging.DEBUG, logger=_LOGGER)

        _record(
            _violation(), enforce=True,
            event_store=event_store, escalation_queue=failing_queue,
        )

        assert any(
            r.name == _LOGGER and r.levelno == logging.ERROR for r in caplog.records
        )
        events = _contract_events(event_store)
        assert len(events) == 1
        assert events[0]['escalation_id'] is None

    @pytest.mark.parametrize('enforce', [False, True])
    def test_absent_collaborators_are_tolerated(self, enforce, caplog):
        caplog.set_level(logging.DEBUG, logger=_LOGGER)
        _record(_violation(), enforce=enforce, event_store=None, escalation_queue=None)
        _record(_store_unavailable(), enforce=enforce, event_store=None, escalation_queue=None)
        assert len(_contract_records(caplog, 'violation')) == 1
        assert len(_contract_records(caplog, 'store_unavailable')) == 1
