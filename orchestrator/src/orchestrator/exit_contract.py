"""The run()-exit contract — judging a finished workflow's exit, in facts.

Task 3542 (theta).  PRD ``plans/task-escalation-state-graph-prd.md`` D7; spec
``docs/task-escalation-state-spec.md`` §5 (the outcome contract) and §8-E11;
design invariants INV-4 (loud, deduped) and INV-6 (a slot exit leaves a
truthful row).

WHY THIS MODULE EXISTS
----------------------
**This docstring is the single canonical explanation.**  Every other site that
touches this mechanism — ``TaskWorkflow.run``, ``TaskWorkflow``'s exit-write
ledger, ``config.OrchestratorConfig.workflow_exit_contract_enforce``,
``EventType.workflow_exit_contract``, the test-suite guard and the test modules
— carries a ONE-LINE pointer here instead of a copy.

When ``TaskWorkflow.run`` returns, two facts must agree (the W9 invariant
SM-2):

1. **Phase** — ``report.phase`` equals the live state-machine state.  A
   mismatch means the report was built from a stale or foreign source.
2. **Outcome ↔ status** — the reported ``WorkflowOutcome`` is an allowed
   pairing with the task's last-persisted status row, per the single Table A
   authority ``shared/src/shared/task_transitions.py::outcome_allows_status``.
   This is the false-done detector: a row reading ``done`` under a BLOCKED
   outcome is the class of bug it exists to catch.

Before this module, a violation raised ``AssertionError`` out of ``run()``.
``orchestrator/src/orchestrator/harness.py::Harness._run_slot``'s generic
catch then turned it into a synthetic BLOCKED report with an empty reason and
filed nothing (§8-E11): the alarm reached one log line, and the harness acted
on a mislabel.  Now ``run()`` ALWAYS returns its real report — TR-1 makes that
return value the harness's only channel — and the violation is RECORDED,
never raised.  The check never writes the task's status in any mode.

VERDICTS
--------
:func:`judge_exit` is pure: given the facts of one exit it returns an
:class:`ExitVerdict`.  The phase half is judged first, then:

* ``STORE_UNAVAILABLE`` — an exit status write failed (see relaxation 2);
* ``UNCHECKED`` — the status row is unknown (not read, or outside the closed
  vocabulary).  This is deliberately NOT "consistent": a garbled read is
  simply not evidence either way;
* ``VIOLATION`` / ``CONSISTENT`` — the Table A answer.  An outcome with no
  Table A row is a VIOLATION, never an exception.

THE TWO SANCTIONED RELAXATIONS (spec §5 — races are real)
--------------------------------------------------------
1. **An observed terminal is reported as that terminal.**  This is met on the
   PRODUCER side, not here: ``TaskWorkflow._observed_terminal_outcome`` maps
   an out-of-band ``done``/``cancelled`` row to DONE/CANCELLED, and the table's
   exact ``done``/``cancelled`` rows admit exactly that.  The judge does NOT
   admit terminal rows across the board — that would pass BLOCKED-over-done,
   the very pair the false-done detector exists for.
2. **A status-write failure at exit reclassifies the exit as crash-shaped.**
   The exit writers that swallow a failed write record it in
   ``TaskWorkflow``'s exit-write ledger; ``run()`` hands it to the judge as
   ``write_failure``.  Such an exit yields ONE ``STORE_UNAVAILABLE`` record (a
   WARNING plus an event) and never an escalation, whatever the mode: the
   stranded sweep is the sanctioned backstop for a crash-shaped exit, and a
   store outage must not become a per-task violation storm.

LOG MODE, THEN ENFORCE (task mu flips it)
-----------------------------------------
Spec §5 lands the tightened table log-mode-first behind a soak.  With
``workflow_exit_contract_enforce`` False (the shipped default) a VIOLATION is a
``would-violate`` WARNING plus a ``workflow_exit_contract`` event with
``mode='log'``; the soak counts those rows.  Task mu sets the flag True after
the soak, and a VIOLATION then also files ONE deduped L1 escalation (category
``workflow_exit_contract``, severity ``blocking``).  Only a literal ``True``
enforces: much of the test suite builds its config from a MagicMock, whose
attributes are truthy.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from escalation.models import Escalation
from shared.task_statuses import TaskStatus
from shared.task_transitions import outcome_allows_status

from orchestrator.event_store import EventStore, EventType
from orchestrator.workflow_types import WorkflowState

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

logger = logging.getLogger(__name__)

EXIT_CONTRACT_CATEGORY = 'workflow_exit_contract'
# The structured ``logging`` extra every recorded verdict carries — the test
# suite's zero-would-violate guard keys on it, never on message text.
VERDICT_LOG_ATTRIBUTE = 'exit_contract_verdict'


class ExitVerdictKind(enum.StrEnum):
    CONSISTENT = 'consistent'
    UNCHECKED = 'unchecked'
    VIOLATION = 'violation'
    STORE_UNAVAILABLE = 'store_unavailable'


class ExitCheck(enum.StrEnum):
    """Which half of the contract a VIOLATION failed."""

    PHASE = 'phase'
    OUTCOME_STATUS = 'outcome_status'


@dataclass(frozen=True)
class ExitStatusWriteFailure:
    """An exit status write that failed and was swallowed (relaxation 2)."""

    target_status: str
    error: str


@dataclass(frozen=True)
class ExitVerdict:
    kind: ExitVerdictKind
    check: ExitCheck | None
    outcome: object
    status_row: str | None
    report_phase: WorkflowState
    machine_state: WorkflowState
    write_failure: ExitStatusWriteFailure | None


def judge_exit(
    *,
    outcome: object,
    report_phase: WorkflowState,
    machine_state: WorkflowState,
    status_row: str | None,
    write_failure: ExitStatusWriteFailure | None,
) -> ExitVerdict:
    """Judge one run() exit against the contract.  Pure; never raises.

    ``outcome`` is a ``WorkflowOutcome`` or anything exposing ``.value`` — the
    shape ``outcome_allows_status`` accepts.
    """

    def verdict(kind: ExitVerdictKind, check: ExitCheck | None = None) -> ExitVerdict:
        return ExitVerdict(
            kind=kind,
            check=check,
            outcome=outcome,
            status_row=status_row,
            report_phase=report_phase,
            machine_state=machine_state,
            write_failure=write_failure,
        )

    if report_phase != machine_state:
        return verdict(ExitVerdictKind.VIOLATION, ExitCheck.PHASE)
    if write_failure is not None:
        return verdict(ExitVerdictKind.STORE_UNAVAILABLE)
    if status_row is None:
        return verdict(ExitVerdictKind.UNCHECKED)
    try:
        status = TaskStatus(status_row)
    except ValueError:
        return verdict(ExitVerdictKind.UNCHECKED)
    try:
        allowed = outcome_allows_status(outcome, status)
    except ValueError:
        # The status is already coerced, so the only ValueError left is an
        # outcome with no Table A row.
        allowed = False
    if not allowed:
        return verdict(ExitVerdictKind.VIOLATION, ExitCheck.OUTCOME_STATUS)
    return verdict(ExitVerdictKind.CONSISTENT)


def record_exit_verdict(
    verdict: ExitVerdict,
    *,
    task_id: str,
    enforce: bool,
    event_store: EventStore | None,
    escalation_queue: EscalationQueue | None,
    worktree: str | None,
    filing_claimant_run_id: str | None,
) -> None:
    """Make a VIOLATION or STORE_UNAVAILABLE verdict loud.  Never raises.

    Writes a log line and a ``workflow_exit_contract`` event; in enforce mode a
    VIOLATION also files one deduped L1.  Never touches the task's status.
    """
    if verdict.kind in (ExitVerdictKind.CONSISTENT, ExitVerdictKind.UNCHECKED):
        return
    mode = 'enforce' if enforce is True else 'log'
    extra = {VERDICT_LOG_ATTRIBUTE: verdict.kind.value}
    escalation_id: str | None = None
    if verdict.kind is ExitVerdictKind.STORE_UNAVAILABLE:
        logger.warning(
            'Task %s: run() exit status write failed — the exit is crash-shaped '
            'and the stranded sweep is its backstop, not a contract violation: %s',
            task_id, _render(_payload(verdict, mode, None)), extra=extra,
        )
    elif mode == 'log':
        logger.warning(
            'Task %s: run()-exit contract would-violate: %s',
            task_id, _render(_payload(verdict, mode, None)), extra=extra,
        )
    else:
        escalation_id = _file_l1(
            verdict,
            task_id=task_id,
            event_store=event_store,
            escalation_queue=escalation_queue,
            worktree=worktree,
            filing_claimant_run_id=filing_claimant_run_id,
        )
        logger.error(
            'Task %s: run()-exit contract violated: %s',
            task_id, _render(_payload(verdict, mode, escalation_id)), extra=extra,
        )
    if event_store is not None:
        event_store.emit(
            EventType.workflow_exit_contract,
            task_id=task_id,
            phase=verdict.machine_state.value,
            data=_payload(verdict, mode, escalation_id),
        )


def _payload(verdict: ExitVerdict, mode: str, escalation_id: str | None) -> dict:
    """The ONE rendering of a verdict — shared by the log line and the event."""
    failure = verdict.write_failure
    return {
        'verdict': verdict.kind.value,
        'mode': mode,
        'check': verdict.check.value if verdict.check is not None else None,
        'outcome': str(getattr(verdict.outcome, 'value', verdict.outcome)),
        'status': verdict.status_row,
        'report_phase': verdict.report_phase.value,
        'machine_state': verdict.machine_state.value,
        'failed_write': (
            {'target_status': failure.target_status, 'error': failure.error}
            if failure is not None else None
        ),
        'escalation_id': escalation_id,
    }


def _render(payload: dict) -> str:
    return ' '.join(f'{key}={value!r}' for key, value in payload.items())


def _file_l1(
    verdict: ExitVerdict,
    *,
    task_id: str,
    event_store: EventStore | None,
    escalation_queue: EscalationQueue | None,
    worktree: str | None,
    filing_claimant_run_id: str | None,
) -> str | None:
    """File the enforce-mode L1 unless one is already open; its id, or None."""
    if escalation_queue is None:
        return None
    facts = _payload(verdict, 'enforce', None)
    try:
        if escalation_queue.has_open_l1(task_id, category=EXIT_CONTRACT_CATEGORY):
            return None
        l1 = Escalation(
            id=escalation_queue.make_id(task_id),
            task_id=task_id,
            agent_role='orchestrator',
            severity='blocking',
            category=EXIT_CONTRACT_CATEGORY,
            summary=(
                f'run()-exit contract violated for task {task_id}: outcome '
                f'{facts["outcome"]!r} with status {facts["status"]!r} '
                f'({facts["check"]} check)'
            )[:200],
            detail=(
                f'{_render(facts)}\n'
                'The run()-exit contract (docs/task-escalation-state-spec.md §5) '
                'forbids this exit; the task status was left as found.  The WHY '
                'is orchestrator/src/orchestrator/exit_contract.py.\n'
            ),
            suggested_action='investigate_exit_contract',
            worktree=worktree,
            workflow_state=verdict.machine_state.value,
            level=1,
            filing_claimant_run_id=filing_claimant_run_id,
        )
        escalation_queue.submit(l1)
    except Exception:
        logger.exception(
            'Task %s: filing the %s L1 failed; recording the violation without it',
            task_id, EXIT_CONTRACT_CATEGORY,
        )
        return None
    if event_store is not None:
        event_store.emit(
            EventType.escalation_created,
            task_id=task_id,
            phase=verdict.machine_state.value,
            data={
                'escalation_id': l1.id,
                'category': EXIT_CONTRACT_CATEGORY,
                'severity': 'blocking',
                'level': 1,
                'summary': l1.summary[:200],
            },
        )
    return l1.id
