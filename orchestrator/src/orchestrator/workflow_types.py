"""Workflow state-machine types (W9-β task).

Extracted verbatim from :mod:`orchestrator.workflow`: the ``WorkflowState``
phase enum and the ``WorkflowOutcome`` enum, following the types-module +
re-export-shim precedent established by :mod:`orchestrator.merge_types`
(extracted from ``orchestrator.merge_queue``).  ``orchestrator.workflow``
re-exports every name here through a top-level shim so existing importers
(``from orchestrator.workflow import WorkflowState``, etc.) keep working
unchanged.

``WorkflowStateMachine`` owns ``TaskWorkflow.state`` and validates every
transition as a THIN VALIDATOR over ``shared.task_transitions`` (W2, task
2168): it defines no transition table of its own (G4 decision #1 — the
escalation server, the fused-memory interceptor, and this machine all
consume the SAME table). Since ``is_legal_transition`` is keyed on
``TaskStatus`` rather than ``WorkflowState``, ``transition`` projects both
sides through ``STATE_TO_STATUS`` before delegating. ``STATE_TO_STATUS`` is
public (re-exported via ``__all__``) so tests can import the real
projection instead of maintaining an independent copy.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Literal

from shared.task_statuses import TaskStatus
from shared.task_transitions import (
    ActorClass,
    is_legal_transition,
    outcome_allows_status,  # noqa: F401  re-export for W9-γ
)

from orchestrator.verify_categories import FailureCategory

__all__ = [
    "STATE_TO_STATUS",
    "IllegalTransition",
    "StewardBudgetExhausted",
    "StewardInterrupted",
    "StewardOutcome",
    "StewardReescalatedL1",
    "StewardResolved",
    "StewardTerminalDecision",
    "TerminalReport",
    "WorkflowOutcome",
    "WorkflowState",
    "WorkflowStateMachine",
]


class WorkflowState(enum.Enum):
    PLAN = 'plan'
    EXECUTE = 'execute'
    VERIFY = 'verify'
    REVIEW = 'review'
    MERGE = 'merge'
    MERGE_DEFERRED = 'merge-deferred'
    DONE = 'done'
    BLOCKED = 'blocked'
    ESCALATED = 'escalated'
    CANCELLED = 'cancelled'


class WorkflowOutcome(enum.Enum):
    DONE = 'done'
    PLANNED = 'planned'
    BLOCKED = 'blocked'
    REQUEUED = 'requeued'
    ESCALATED = 'escalated'
    CANCELLED = 'cancelled'
    MERGE_DEFERRED = 'merge-deferred'
    SOFT_CANCELLED = 'soft-cancelled'


@dataclass(frozen=True)
class TerminalReport:
    """The workflow↔harness terminal contract, as a typed RETURN value.

    Replaces the ``_last_block_reason``/``_last_block_detail``/
    ``_last_block_phase`` side channel (three independent, mutable
    ``TaskWorkflow`` attributes that could go partially stale — bug_history
    882/883/851, esc-2073-15) with ONE atomic, immutable object built at the
    same choke point (``_mark_blocked``, plus the two non-``_mark_blocked``
    block paths) and returned by :meth:`TaskWorkflow.run`.

    ``category`` is typed ``FailureCategory | None`` for the future W9-ε
    wiring (``classify_failure`` → ``BlockDisposition``); it is always
    ``None`` through W9-γ — see that task's design decisions.

    ``phase`` is always the TERMINAL ``machine.state`` at the point this
    report was built (BLOCKED for a ``_mark_blocked`` exit; the still-current
    working phase for a REQUEUED exit that never transitions) — this is the
    field SM-2's ``report.phase == machine.state`` invariant is checked
    against. ``blocked_from_phase`` is a DISTINCT field: the PRE-block
    WORKING phase (e.g. ``VERIFY``/``REVIEW``) snapshotted immediately before
    ``_mark_blocked`` calls ``_enter_phase(BLOCKED)``. It defaults to
    ``None`` (clean DONE/CANCELLED exits, and every pre-existing
    construction site) and is what the harness maps ``TaskReport.block_phase``
    from, so the optimistic-path auto-eval phase gate
    (``config.auto_eval_phases``) keeps matching plan/execute/verify/review
    blocks instead of seeing the terminal ``'blocked'`` state (REVIEW-CYCLE-1
    fix — see that design decision).
    """

    outcome: WorkflowOutcome
    reason: str
    phase: WorkflowState
    detail: str
    category: FailureCategory | None
    blocked_from_phase: WorkflowState | None = None


@dataclass(frozen=True)
class StewardResolved:
    """The steward cleared the level-0 escalation itself (no re-escalation).

    Published by :meth:`TaskSteward._handle_escalation` on the success
    branch. ``resolution_text`` is sourced from the resolved escalation's
    ``resolution`` field, falling back to the invocation result/summary.
    """

    resolution_text: str


@dataclass(frozen=True)
class StewardReescalatedL1:
    """The steward gave up and filed a level-1 (human) escalation.

    Published by :meth:`TaskSteward._auto_escalate_to_human` — the single
    choke point for every give-up path (empty-output cap, worktree-missing
    preflight, and the wip-less branches of the retry/timeout caps below).
    ``esc_id`` is the id of the newly-filed level-1 escalation.
    """

    esc_id: str


@dataclass(frozen=True)
class StewardTerminalDecision:
    """The scheduler's task status is already terminal or ``deferred``.

    SYNTHESIZED by :meth:`TaskWorkflow._await_steward_completion` from a
    single scheduler status read — never published by the steward itself
    (the steward has no scheduler reference; only the workflow can observe
    status changes the sub-agent made via MCP tools). Overrides a channel
    ``StewardResolved`` when both are true (the terminal/deferred check
    takes precedence, preserving the pre-W9-δ ordering).
    """

    new_status: TaskStatus


@dataclass(frozen=True)
class StewardInterrupted:
    """The steward's work on this escalation was cut short.

    ``reason='attempt_cap'`` — the per-escalation retry limit
    (``steward_max_attempts``) was reached.  ``reason='timeout'`` — either
    the steward's per-escalation wall-clock timeout cap fired, or
    :meth:`TaskWorkflow._await_steward_completion`'s grace period elapsed
    with no outcome published.

    ``wip_commits_present`` is the task-2060 fix: when ``True``, the
    workflow's resume-plan branch re-pends the task instead of filing an L1
    — a wall-clock kill with partial work committed must not be triaged as
    "steward failed".  Derived via the shared ``_worktree_has_wip_commits``
    probe (workflow method, injected into the steward as ``_wip_probe``).
    """

    reason: Literal['timeout', 'attempt_cap']
    wip_commits_present: bool


@dataclass(frozen=True)
class StewardBudgetExhausted:
    """The steward's lifetime budget (``steward_lifetime_budget``) was exhausted.

    Always routes to an L1 escalation regardless of wip — unlike
    ``StewardInterrupted``, budget exhaustion is not gated on in-flight work
    because the steward cannot make further progress at all, wip or not.
    """


# The workflow↔steward RPC payload carried on the per-task in-process
# asyncio.Queue (PRD §10 / Contract §8.1-8.2 SO-1, resolved decision D6).
# A PEP-604 union (not an enum-with-payload) so every variant branches
# uniformly via isinstance/match in TaskWorkflow._mark_blocked's single
# routing choke point.
StewardOutcome = (
    StewardResolved
    | StewardReescalatedL1
    | StewardTerminalDecision
    | StewardInterrupted
    | StewardBudgetExhausted
)


class IllegalTransition(Exception):
    """Raised by :meth:`WorkflowStateMachine.transition` on an illegal move."""


# WorkflowState -> TaskStatus projection consumed by WorkflowStateMachine.transition.
# is_legal_transition is keyed on TaskStatus, not WorkflowState, so honouring W2's
# authority (G4 decision #1 — one shared table, never a fourth) requires projecting
# through this map rather than defining a parallel WorkflowState table. Every working
# phase collapses to IN_PROGRESS (phase order is not W2's authority to enforce);
# ESCALATED collapses to BLOCKED (it is a blocked-shaped phase); MERGE_DEFERRED/DONE/
# BLOCKED/CANCELLED project to themselves.
#
# Public (no leading underscore) and re-exported via __all__ so
# test_workflow_state_machine.py can import this SAME object for its
# delegation cross-check instead of maintaining an independent copy that
# could silently drift from this table (see TestStateToStatusProjection in
# that module for the hand-written per-member pins that keep the check
# meaningful despite sharing this object).
STATE_TO_STATUS: dict[WorkflowState, TaskStatus] = {
    WorkflowState.PLAN: TaskStatus.IN_PROGRESS,
    WorkflowState.EXECUTE: TaskStatus.IN_PROGRESS,
    WorkflowState.VERIFY: TaskStatus.IN_PROGRESS,
    WorkflowState.REVIEW: TaskStatus.IN_PROGRESS,
    WorkflowState.MERGE: TaskStatus.IN_PROGRESS,
    WorkflowState.ESCALATED: TaskStatus.BLOCKED,
    WorkflowState.MERGE_DEFERRED: TaskStatus.MERGE_DEFERRED,
    WorkflowState.DONE: TaskStatus.DONE,
    WorkflowState.BLOCKED: TaskStatus.BLOCKED,
    WorkflowState.CANCELLED: TaskStatus.CANCELLED,
}


class WorkflowStateMachine:
    """Owns ``TaskWorkflow``'s phase state and validates every transition.

    ``transition`` is a THIN VALIDATOR over ``shared.task_transitions`` (W2,
    task 2168): it projects the current and target ``WorkflowState`` through
    ``STATE_TO_STATUS`` into ``TaskStatus`` and delegates the legality
    decision to ``is_legal_transition`` — this class defines no transition
    table of its own (G4 decision #1: the escalation server, the
    fused-memory interceptor, and this machine all consume the SAME table).

    The enforced invariant is terminal absorption: ``DONE``/``CANCELLED``
    are members of the shared ``TERMINAL`` set, so ``is_legal_transition``
    returns ``False`` for any out-transition from either (without
    ``reopen=True``, which this machine never passes) and ``transition``
    raises :class:`IllegalTransition`, leaving the state unchanged.
    Same-state transitions — including ``DONE`` -> ``DONE`` and
    ``CANCELLED`` -> ``CANCELLED`` — are always legal no-ops.

    Phase order beyond that is intentionally NOT re-enforced: every working
    phase (``PLAN``/``EXECUTE``/``VERIFY``/``REVIEW``/``MERGE``) projects to
    the same ``IN_PROGRESS`` status, so a phase-to-phase move collapses to a
    same-status (always legal) check — phase ordering is not W2's authority.
    """

    def __init__(self, initial: WorkflowState = WorkflowState.PLAN) -> None:
        self._state = initial

    @property
    def state(self) -> WorkflowState:
        return self._state

    def is_terminal(self) -> bool:
        """Is the current state absorbing (``DONE`` or ``CANCELLED``)?"""
        return self._state in (WorkflowState.DONE, WorkflowState.CANCELLED)

    def transition(self, to: WorkflowState) -> None:
        """Advance to *to*, raising :class:`IllegalTransition` on an illegal move.

        Legality is delegated entirely to
        :func:`shared.task_transitions.is_legal_transition`, evaluated on
        the ``TaskStatus`` projection of the current and target state
        (``ActorClass.ORCHESTRATOR``, ``reopen=False``). The state is left
        UNCHANGED when the move is illegal.
        """
        frm_status = STATE_TO_STATUS[self._state]
        to_status = STATE_TO_STATUS[to]
        if not is_legal_transition(frm_status, to_status, ActorClass.ORCHESTRATOR):
            raise IllegalTransition(
                f'Illegal workflow transition: {self._state.value} -> {to.value} '
                f'(projected status {frm_status.value} -> {to_status.value})',
            )
        self._state = to

    def force_set(self, to: WorkflowState) -> None:
        """Set the state directly, bypassing legality validation.

        Backs ``TaskWorkflow``'s ``state`` property setter, which many
        existing tests rely on to stage a state directly (e.g.
        ``wf.state = WorkflowState.DONE``) without driving a real
        transition.
        """
        self._state = to
