"""The run()-exit contract: the pure judge and its recorder.

Spec ``docs/task-escalation-state-spec.md`` §5 / E11; the canonical WHY lives
in ``orchestrator/src/orchestrator/exit_contract.py`` (module docstring).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from orchestrator.exit_contract import (
    ExitCheck,
    ExitStatusWriteFailure,
    ExitVerdict,
    ExitVerdictKind,
    judge_exit,
)
from orchestrator.workflow_types import WorkflowOutcome, WorkflowState

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
