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

from orchestrator.unblock_types import BlockClass
from orchestrator.verify_categories import FailureCategory

__all__ = [
    "STATE_TO_STATUS",
    "BlockDisposition",
    "IllegalTransition",
    "RequeueKind",
    "StewardBudgetExhausted",
    "classify_failure",
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


class RequeueKind(enum.Enum):
    """The outcome-kind a classified failure resolves to in ``_drive()``.

    The 3 values ``classify_failure`` (below) maps every ladder exception
    onto, replacing the exception-type-keyed except-clause dispatch with a
    disposition-table lookup: REQUEUE (WarmLaneRequeue and its subclasses),
    BLOCK (everything else in the ladder — cap/budget/verify-infra/OSError/
    worktree-conflict/generic), and CANCEL (reserved for future use; the
    SetTaskStatusRejected/TerminalExitRejection cancel/terminal-exit clause
    is NOT table-driven — see W9-ε's design decisions — so no ladder
    exception maps here today).
    """

    REQUEUE = 'requeue'
    BLOCK = 'block'
    CANCEL = 'cancel'


@dataclass(frozen=True)
class BlockDisposition:
    """The disposition ``classify_failure`` resolves a failure exception to.

    A single, table-driven replacement for the hand-written per-exception
    decisions that used to live inline in ``TaskWorkflow._drive()``'s
    exception ladder (escalate_to_human, block-vs-requeue, requeue-cap
    accounting, and the dry-run ``BlockClass``) — and, per BD-1, the same
    disposition metadata the four independent ``AllAccountsCappedException``
    catch sites (workflow.py/steward.py/review_checkpoint.py/
    dry_run_unblock.py) now consult instead of hand-writing their own.

    ``category`` is the ``verify_classify.classify_failure`` 12-value
    output domain (:class:`FailureCategory`) — a DISTINCT concept from the
    escalation-taxonomy string (``'infra_issue'``/``'wip_conflict'``/
    ``'task_failure'``) ``_mark_blocked`` separately accepts as its
    ``category=`` parameter. Every ladder exception is a non-verify-check
    failure, so this is always :attr:`FailureCategory.NONE` today — see
    this task's design decisions.
    """

    category: FailureCategory
    escalate_to_human: bool
    requeue_kind: RequeueKind
    counts_against_requeue_cap: bool
    reason_prefix: str
    block_class: BlockClass


# The fallback disposition classify_failure returns for any exception with no
# explicit _DISPOSITION_TABLE row (including every genuinely-unrecognized
# exception type) — mirrors the pre-W9-ε ladder's broad `except Exception`
# tail (workflow.py:2395: `_mark_blocked(f'Workflow error: {e}')`, no
# category/escalate_to_human passed, i.e. escalate_to_human=False).
_DEFAULT_BLOCK = BlockDisposition(
    category=FailureCategory.NONE,
    escalate_to_human=False,
    requeue_kind=RequeueKind.BLOCK,
    counts_against_requeue_cap=True,
    reason_prefix='Workflow error',
    block_class=BlockClass.AGENT_FAILURE,
)

# The disposition for an infra-class OSError (ENOSPC/EDQUOT/EROFS/EIO/EMFILE/
# ENFILE), matching the pre-W9-ε ladder's `except OSError as e: if
# _is_infra_oserror(e): ...` branch (workflow.py:2359: category='infra_issue',
# escalate_to_human=True). A non-infra OSError (e.g. EACCES/ENOENT) is
# INDISTINGUISHABLE from a generic failure in the old ladder — it fell
# through to the same `except Exception` tail — so classify_failure returns
# _DEFAULT_BLOCK for it (see below), not a dedicated row.
_INFRA_OSERROR_BLOCK = BlockDisposition(
    category=FailureCategory.NONE,
    escalate_to_human=True,
    requeue_kind=RequeueKind.BLOCK,
    counts_against_requeue_cap=True,
    reason_prefix='Verify infra OSError',
    block_class=BlockClass.AGENT_FAILURE,
)

# Module-level cache for _disposition_table(), populated on first call.
_disposition_table_cache: dict[type[BaseException], BlockDisposition] | None = None


def _disposition_table() -> dict[type[BaseException], BlockDisposition]:
    """Build (once) and return the exception-type -> BlockDisposition table.

    Every row's exception type is imported LAZILY, inside this function —
    mirroring ``unblock_types.classify_block_reason``'s precedent — so
    ``workflow_types`` stays importable by lightweight consumers (e.g.
    ``steward.py``, which only wants the ``Steward*`` outcome dataclasses)
    without pulling in ``orchestrator.git_ops``/``orchestrator.verify``'s
    heavier transitive dependency graph merely by importing this module.
    The built dict is cached at module level after the first call.

    Deliberately does NOT contain a row for bare ``Exception``/
    ``BaseException`` — see :func:`_lookup_disposition`'s docstring: a
    generic base-class row would make BD-2's completeness check
    (a synthetic new exception type with no row must resolve to ``None``)
    meaningless, since every exception would then MRO-match it.
    """
    global _disposition_table_cache
    if _disposition_table_cache is not None:
        return _disposition_table_cache

    from shared.cli_invoke import AllAccountsCappedException
    from shared.usage_gate import IllegalTransitionError, SessionBudgetExhausted

    from orchestrator.git_ops import (
        InteractiveWorktreeLimitError,
        MergeVerifyLeaseHeld,
        WarmLaneDiskPressure,
        WarmLanePoolExhausted,
        WarmLanePoolHardDown,
        WarmLaneRequeue,
        WorktreeConflictError,
        WorktreeMissing,
    )
    from orchestrator.verify import VerifyInfraError

    _disposition_table_cache = {
        # ── Cap/budget (BD-1: also consulted by the 3 satellite cap sites) ──
        # Cap/agent failures map through W4's InvocationOutcome.CapHit /
        # AgentFailureKind.UNKNOWN seam conceptually — classify_failure's
        # AllAccountsCappedException/generic rows are this table's local
        # projection of that same "cap hit" / "unclassified agent failure"
        # distinction onto a BlockDisposition.
        AllAccountsCappedException: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.BLOCK,
            counts_against_requeue_cap=True,
            reason_prefix='All accounts capped',
            block_class=BlockClass.AGENT_FAILURE,
        ),
        SessionBudgetExhausted: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.BLOCK,
            counts_against_requeue_cap=True,
            reason_prefix='Session budget exhausted',
            block_class=BlockClass.AGENT_FAILURE,
        ),
        # ── Warm-lane requeue family ─────────────────────────────────────
        # counts_against_requeue_cap is declared ONCE per subclass here —
        # the single source of truth replacing the buried NOTE at
        # workflow.py ~2278-2300 (EXHAUSTED is genuine backpressure and
        # counts; DISK_PRESSURE/HARD_DOWN are transient infra and do not).
        # The WarmLaneRequeue base row exists for BD-2 completeness (it is
        # itself one of git_ops.py's exported types) — MRO resolution means
        # a real subclass instance always matches its OWN row first.
        WarmLaneRequeue: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.REQUEUE,
            counts_against_requeue_cap=True,
            reason_prefix='warm_lane_requeue',
            block_class=BlockClass.AGENT_FAILURE,
        ),
        WarmLanePoolExhausted: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.REQUEUE,
            counts_against_requeue_cap=True,
            reason_prefix='warm_lane_pool_exhausted',
            block_class=BlockClass.AGENT_FAILURE,
        ),
        WarmLaneDiskPressure: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.REQUEUE,
            counts_against_requeue_cap=False,
            reason_prefix='warm_lane_disk_pressure (transient infra)',
            block_class=BlockClass.AGENT_FAILURE,
        ),
        WarmLanePoolHardDown: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.REQUEUE,
            counts_against_requeue_cap=False,
            reason_prefix='warm_lane_pool_hard_down',
            block_class=BlockClass.AGENT_FAILURE,
        ),
        # ── Verify-infra / worktree-conflict (escalate straight to human) ──
        VerifyInfraError: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=True,
            requeue_kind=RequeueKind.BLOCK,
            counts_against_requeue_cap=True,
            reason_prefix='Verify infra failure',
            block_class=BlockClass.AGENT_FAILURE,
        ),
        WorktreeConflictError: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=True,
            requeue_kind=RequeueKind.BLOCK,
            counts_against_requeue_cap=True,
            reason_prefix='WIP-save aborted: unresolved conflict in worktree',
            block_class=BlockClass.AGENT_FAILURE,
        ),
        # ── BD-2 completeness rows: exported by the 4 covered modules, but
        # not (today) caught by name anywhere in _drive()'s ladder ─────────
        WorktreeMissing: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=True,
            requeue_kind=RequeueKind.BLOCK,
            counts_against_requeue_cap=True,
            reason_prefix='Worktree missing',
            block_class=BlockClass.AGENT_FAILURE,
        ),
        InteractiveWorktreeLimitError: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.REQUEUE,
            counts_against_requeue_cap=True,
            reason_prefix='interactive_worktree_limit_reached',
            block_class=BlockClass.AGENT_FAILURE,
        ),
        MergeVerifyLeaseHeld: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.REQUEUE,
            counts_against_requeue_cap=False,
            reason_prefix='merge_verify_lease_held',
            block_class=BlockClass.AGENT_FAILURE,
        ),
        IllegalTransitionError: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=True,
            requeue_kind=RequeueKind.BLOCK,
            counts_against_requeue_cap=True,
            reason_prefix='Illegal usage-gate transition',
            block_class=BlockClass.AGENT_FAILURE,
        ),
    }
    return _disposition_table_cache


def _lookup_disposition(exc_type: type[BaseException]) -> BlockDisposition | None:
    """Return the explicit table row for *exc_type*, or ``None`` if absent.

    Walks ``exc_type.__mro__`` so a subclass with no row of its own (there
    are none among today's rows, but a future one is possible) inherits its
    nearest ancestor's disposition. Returns ``None`` — rather than a
    default — when no ancestor (up to but excluding ``Exception``/
    ``BaseException``, which are never table keys) matches: this is what
    makes BD-2's completeness check meaningful. ``classify_failure`` is the
    TOTAL wrapper that turns a ``None`` into :data:`_DEFAULT_BLOCK`.
    """
    table = _disposition_table()
    for klass in exc_type.__mro__:
        if klass in table:
            return table[klass]
    return None


def classify_failure(exc: BaseException) -> BlockDisposition:
    """Classify *exc* into a :class:`BlockDisposition` — the W9-ε TABLE.

    Total: every exception maps to exactly one disposition. Replaces
    ``TaskWorkflow._drive()``'s exception-type-keyed except-clause ladder
    (workflow.py:2175-2397) and, per BD-1, is the single classifier the
    four independent ``AllAccountsCappedException`` catch sites (workflow,
    steward, review_checkpoint, dry_run_unblock) all consult.

    ``OSError`` is disambiguated by errno (via ``verify._is_infra_oserror``)
    rather than by a table row, since two ``OSError`` instances of the SAME
    type can carry different errnos: an infra-class errno (ENOSPC/EDQUOT/
    EROFS/EIO/EMFILE/ENFILE) escalates to a human; any other OSError
    (including ``WorktreeMissing``, a ``FileNotFoundError``/``OSError``
    subclass that DOES also carry its own explicit table row for BD-2) is
    indistinguishable from a generic failure in the pre-W9-ε ladder and
    falls through to :data:`_DEFAULT_BLOCK` here too — mirroring
    workflow.py:2371's `# Non-infra OSError — treat the same as the broad
    except below` comment exactly.
    """
    if isinstance(exc, OSError):
        from orchestrator.verify import _is_infra_oserror
        if _is_infra_oserror(exc):
            return _INFRA_OSERROR_BLOCK
        return _DEFAULT_BLOCK
    return _lookup_disposition(type(exc)) or _DEFAULT_BLOCK


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
