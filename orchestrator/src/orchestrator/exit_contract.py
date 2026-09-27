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
from dataclasses import dataclass

from shared.task_statuses import TaskStatus
from shared.task_transitions import outcome_allows_status

from orchestrator.workflow_types import WorkflowState


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
