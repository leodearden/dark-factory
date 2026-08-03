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

import asyncio
import contextlib
import enum
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

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
    "CancellationScope",
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
    "WorkflowCancelled",
    "WorkflowOutcome",
    "WorkflowState",
    "WorkflowStateMachine",
]

_T = TypeVar('_T')

logger = logging.getLogger(__name__)


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
    # Task 2988 (PRD ε / W3): whether this terminal outcome's requeue counts
    # against the per-task requeue cap.  Set from
    # ``BlockDisposition.counts_against_requeue_cap`` at the REQUEUED
    # construction site (workflow.py's WarmLaneRequeue clause), mapped onto
    # ``TaskReport.counts_against_requeue_cap`` in the harness, and finally
    # consumed by ``Scheduler.record_requeue(counts_against_cap=...)``.
    # Defaults True so every pre-existing construction site (DONE/CANCELLED
    # and all non-warm-lane block paths) is unchanged and keeps counting.
    counts_against_requeue_cap: bool = True


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
        BranchResetError,
        EphemeralWorktreeError,
        InteractiveWorktreeLimitError,
        LaneLockSelfOwnedLeak,
        MergeParkContentionError,
        MergeParkError,
        MergeVerifyLeaseContended,
        MergeVerifyLeaseHeld,
        WarmLaneDiskPressure,
        WarmLanePoolExhausted,
        WarmLanePoolHardDown,
        WarmLaneRequeue,
        WarmLaneReseedContaminated,
        WarmLaneSoftPressure,
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
        # workflow.py ~2278-2300. EXHAUSTED/DISK_PRESSURE/HARD_DOWN/
        # SOFT_PRESSURE all count=False: they are shared-resource /
        # capacity signals, NOT a fault of the requeued task, so burning
        # its per-task requeue cap punishes the wrong party.
        #
        # EXHAUSTED was flipped True->False by task 2988 (PRD ε / W3): the
        # 2026-07-22 incident showed pool exhaustion can mean a capacity
        # LEAK (lanes stuck assigned), and treating it as "genuine
        # backpressure that counts" burned every waiting task's requeue cap
        # -> retry-cap escalation -> reblock-guard L2 storm. Exhaustion is
        # now observed pool-GLOBALLY at the GitOps acquire chokepoint: a
        # deduped, born-at-L2 structural-exhaustion escalation (fired after
        # N consecutive EXHAUSTED acquires) is the SOLE loud signal, so
        # EXHAUSTED joins its transient siblings as non-counting.
        #
        # WarmLaneReseedContaminated remains count=True — it is a per-task
        # DATA-INTEGRITY fault (a lane retained a prior occupant's commits),
        # not a shared-resource signal, so a persistent contamination SHOULD
        # trip that task's requeue-cap escalation.
        # The WarmLaneRequeue base row exists for BD-2 completeness (it is
        # itself one of git_ops.py's exported types) — MRO resolution means
        # a real subclass instance always matches its OWN row first, so this
        # row is only ever reached by a bare WarmLaneRequeue instance (none
        # raised anywhere in the tree today). Its disposition deliberately
        # MIRRORS WarmLaneDiskPressure's rather than using a distinct
        # 'warm_lane_requeue' literal (amendment, reviewer_comprehensive
        # behavior-parity): the pre-W9-ε inline triage's `else:  #
        # WarmLaneDiskPressure` fallback (removed by step-10; see that
        # commit) mechanically classified ANY WarmLaneRequeue that wasn't
        # WarmLanePoolHardDown/WarmLanePoolExhausted — including a
        # hypothetical bare-base instance — as disk-pressure-shaped. Aliasing
        # this row to that same disposition keeps workflow.py's
        # WarmLaneRequeue except-clause comment ("reproduces the old
        # per-subclass strings exactly") true for the base class too, not
        # just the 3 named subclasses. Pinned by
        # test_bare_warm_lane_requeue_base_matches_old_else_branch_fallback.
        WarmLaneRequeue: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.REQUEUE,
            counts_against_requeue_cap=False,
            reason_prefix='warm_lane_disk_pressure (transient infra)',
            block_class=BlockClass.AGENT_FAILURE,
        ),
        WarmLanePoolExhausted: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.REQUEUE,
            # Task 2988 (PRD ε / W3): flipped True->False — see the family
            # comment above. Pool exhaustion no longer burns the per-task
            # requeue cap; a pool-level structural-exhaustion L2 is the loud
            # signal instead.
            counts_against_requeue_cap=False,
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
        # θ proactive soft-floor throttle (task 2443, §9.5 inv.11): pure
        # backpressure/defer for a FRESH allocation — deliberately weaker
        # than ε's hard-floor WarmLaneDiskPressure exit-75 row above, so it
        # gets its own reason_prefix and NEVER counts against the requeue
        # cap (mirrors HARD_DOWN/DISK_PRESSURE's counts_against_requeue_cap
        # =False, not EXHAUSTED's =True — soft pressure is not genuine pool
        # exhaustion).
        #
        # Confirmed intended (amendment, reviewer_comprehensive robustness):
        # escalate_to_human=False + counts_against_requeue_cap=False means a
        # FRESH allocation under sustained soft pressure requeues indefinitely
        # with no escalation path — by design, per inv.11, since soft-floor
        # throttling must never itself become an escalation or a fault. The
        # only operator-facing signal is the per-defer WARNING journal line
        # (GitOps._warm_lane_soft_pressure_defer) plus this reason_prefix. A
        # bounded consecutive-defer-then-escalate counter would need state
        # tracked across dispatch/requeue cycles at the scheduler/harness
        # layer — outside this row's (and this task's) module scope — so it's
        # left as a possible future follow-up, not implemented here.
        WarmLaneSoftPressure: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.REQUEUE,
            counts_against_requeue_cap=False,
            reason_prefix='warm_lane_soft_pressure (backpressure)',
            block_class=BlockClass.AGENT_FAILURE,
        ),
        # Reseed contamination (task 2854): a fresh-reseed acquire left the
        # lane carrying a prior occupant's commits — a DATA-INTEGRITY defect,
        # not transient backpressure. REQUEUE (re-acquire a DIFFERENT lane),
        # but counts_against_requeue_cap=True (unlike the transient
        # DiskPressure/HardDown/SoftPressure rows) so a persistent/pathological
        # contamination eventually trips the requeue-cap escalation — a loud
        # human signal — instead of requeuing forever silently. Distinctive
        # reason_prefix gives operators a greppable data-integrity signal.
        WarmLaneReseedContaminated: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.REQUEUE,
            counts_against_requeue_cap=True,
            reason_prefix='warm_lane_reseed_contaminated (data-integrity)',
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
        # BranchResetError (task 2403): a requeue/inter-iteration rebase
        # collapsed a task branch to zero commits ahead of its baseline,
        # wiping committed WIP. Caught by NAME in TaskWorkflow.run()'s
        # `isinstance(e, BranchResetError)` branch (workflow.py ~2470) and
        # routed to a per-task BLOCKED + human L1 — mirrors
        # WorktreeConflictError above (RuntimeError subclass, same
        # halt-and-escalate shape). That branch reads disp.reason_prefix for
        # the blocked message, so this row's prefix (not the old
        # _DEFAULT_BLOCK 'Workflow error') is what a human sees.
        BranchResetError: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=True,
            requeue_kind=RequeueKind.BLOCK,
            counts_against_requeue_cap=True,
            reason_prefix='Branch reset: rebase wiped committed task work',
            block_class=BlockClass.AGENT_FAILURE,
        ),
        # ── BD-2 completeness rows: exported by the 4 covered modules, but
        # not (today) caught by name anywhere in _drive()'s ladder ─────────
        # NOTE (amendment, reviewer_comprehensive correctness): this row is
        # UNREACHABLE via classify_failure() — WorktreeMissing IS-A
        # FileNotFoundError IS-A OSError, and classify_failure's
        # isinstance(exc, OSError) branch runs BEFORE _lookup_disposition
        # (see its docstring). A real WorktreeMissing always carries
        # errno=None (constructed from a single message string, not the
        # (errno, strerror) OSError form), which is never in
        # verify._INFRA_ERRNOS, so classify_failure(instance) falls all the
        # way through to _DEFAULT_BLOCK (escalate_to_human=False) — NOT this
        # row's escalate_to_human=True below. This row exists ONLY so
        # _lookup_disposition(WorktreeMissing) is non-None for BD-2's
        # completeness test; its own field values are never observed by
        # classify_failure(). Pinned by
        # TestBD2Completeness.test_worktree_missing_row_is_shadowed_by_the_oserror_branch
        # in test_block_disposition.py — do not treat that test's passing as
        # license to "clean up" this row without re-reading this note.
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
        # A merge-verify lease held by a DIFFERENT live process (task 2315,
        # BUG 1): the fail-CLOSED pre-check in
        # GitOps.reset_persistent_merge_worktree refused BEFORE touching the
        # tree, so nothing was mutated and nothing was verified — the lane is
        # simply busy. REQUEUE, never escalates, never counts against the
        # requeue cap.
        #
        # task 3003: this is no longer a BD-2-completeness-only row. The merge
        # worker's _run_inflight_verify now genuinely requeues this type (its
        # defer arm catches (MergeVerifyLeaseContended, MergeVerifyLeaseHeld)),
        # so the values below finally describe what the code does. Until then
        # the type fell to the generic `except Exception` and resolved a
        # 'blocked' MergeOutcome — the merge worker directly contradicting the
        # policy declared right here.
        MergeVerifyLeaseHeld: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.REQUEUE,
            counts_against_requeue_cap=False,
            reason_prefix='merge_verify_lease_held',
            block_class=BlockClass.AGENT_FAILURE,
        ),
        # A contended merge-verify lane lock (task 2828) is transient "come back
        # later," identical in shape to MergeVerifyLeaseHeld above: REQUEUE,
        # never escalates, never counts against the requeue cap. Requeued by
        # the merge worker's _run_inflight_verify (BD-2 forces this explicit
        # row). Raised by BOTH bounded-wait acquires on the shared
        # <lane_dir>.lock: GitOps.merge_verify_lease (the verify span, 2828)
        # and GitOps.reset_persistent_merge_worktree (the warm-swap reset,
        # task 3003 — where it additionally means the tree was never touched).
        MergeVerifyLeaseContended: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.REQUEUE,
            counts_against_requeue_cap=False,
            reason_prefix='merge_verify_lease_contended',
            block_class=BlockClass.AGENT_FAILURE,
        ),
        # Task 3081 (D8/B13). IS-A MergeVerifyLeaseContended, and keeps that
        # row's REQUEUE + no-cap-burn: a leaked lane lock is an infra fault the
        # task must not be charged for. Diverges on ONE axis --
        # escalate_to_human -- because unlike ordinary contention it can never
        # resolve by waiting: nothing releases the leaked fd before process
        # exit. In reify esc-5548-5 three tasks blocked behind one identical
        # merge_outcome_signature 3173b64436423738 and nothing surfaced until an
        # unattended restart ~3h later. An explicit row is required regardless
        # (BD-2 completeness enumerates git_ops' exports), but the MRO would
        # otherwise resolve it to the parent's escalate_to_human=False.
        #
        # HONEST LIMITATION: on the merge-worker path this exception is caught
        # by merge_queue.py's contended-defer arm (it IS-A the parent) BEFORE
        # the disposition table is consulted, so this row governs the OTHER
        # consumers -- cli.py verify-merge and workflow block classification.
        # The loud FIRST-OCCURRENCE signal is the logger.error at the detection
        # site in GitOps._lane_lock_self_owned_leak, not this flag.
        LaneLockSelfOwnedLeak: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=True,
            requeue_kind=RequeueKind.REQUEUE,
            counts_against_requeue_cap=False,
            reason_prefix='lane_lock_self_owned_leak',
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
        # advance_main's WIP-parking safety mechanism (task 2556) — both are
        # permanent-halt conditions ("halting to preserve WIP" / "halting
        # merge to prevent code loss", git_ops.py's advance_main), never
        # caught by _drive()'s ladder (they're raised/caught entirely inside
        # GitOps.advance_main), so these are BD-2-completeness-only rows —
        # same category as WorktreeMissing/MergeVerifyLeaseHeld above.
        # MergeParkContentionError IS-A MergeParkError but gets its own row
        # (distinct cause/log text) rather than relying on MRO fallback.
        MergeParkError: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=True,
            requeue_kind=RequeueKind.BLOCK,
            counts_against_requeue_cap=True,
            reason_prefix='WIP park failed before advance_main',
            block_class=BlockClass.AGENT_FAILURE,
        ),
        MergeParkContentionError: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=True,
            requeue_kind=RequeueKind.BLOCK,
            counts_against_requeue_cap=True,
            reason_prefix='Stale merge-park ref: refusing to overwrite',
            block_class=BlockClass.AGENT_FAILURE,
        ),
        # BD-2-completeness-only row (task 2147/θ, verify-plan PRD). Raised by
        # GitOps.ephemeral_worktree() when `git worktree add` fails on every
        # retry — but both verify.py probe call sites
        # (verify_failure_is_preexisting_on_main, run_main_tip_sweep) catch it
        # locally and fail safe, so it never reaches _drive()/classify_failure().
        # Mirrors MergeParkError's halt-and-escalate shape (both are
        # safety/retry-exhaustion conditions caught close to their raise site).
        EphemeralWorktreeError: BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=True,
            requeue_kind=RequeueKind.BLOCK,
            counts_against_requeue_cap=True,
            reason_prefix='Ephemeral probe worktree add failed after retries',
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


@dataclass
class WorkflowCancelled(Exception):
    """The ONE typed cancellation signal (CX-1, PRD §8.1).

    Raised by :class:`CancellationScope` and caught at EXACTLY ONE place —
    ``TaskWorkflow.run()`` — regardless of whether the cancellation
    originated from the harness's hard ``task.cancel()`` (``kind='hard'``)
    or from the workflow's own soft ``_cancel_event`` (``kind='soft'``).
    Replaces the ``sys.exc_info()`` B1 sniff (workflow.py) and the harness's
    B2 ``synthetic_cancel`` dual-guard — a two-file "both must fire" comment
    contract — with one exception type carrying the distinction as data.

    Deliberately NOT ``frozen=True`` (unlike this module's other dataclasses):
    a frozen dataclass overrides ``__setattr__`` to unconditionally raise
    ``FrozenInstanceError`` for ANY attribute, including ``__traceback__`` —
    which Python's own exception machinery assigns when an exception is
    re-raised through a ``@contextlib.contextmanager``-based ``__exit__``
    (e.g. ``pytest.MonkeyPatch.context()``, hit via ``_await_cancellable``'s
    raise). ``frozen=True`` was tried first per the plan's step-1/2 risk
    note; construct/raise/catch/read in isolation (step 1) passed, but the
    contextmanager-propagation path only surfaced once real call sites
    started raising it (step 12), confirming the anticipated fallback.
    """

    kind: Literal['hard', 'soft']


# The ordered, kind-aware terminal-cleanup list a CancellationScope runs on
# every exit from `supervise` (normal return, soft-cancel, or hard-cancel).
# `kind` is None for a normal (non-cancelled) exit.
OnTerminalEntry = tuple[str, Callable[[Literal['hard', 'soft'] | None], Awaitable[None]]]


class CancellationScope:
    """Supervises a workflow body coroutine and turns either a hard
    ``task.cancel()`` or the soft *cancel_event* firing into ONE typed
    :class:`WorkflowCancelled`, running an ordered ``on_terminal`` cleanup
    list exactly once on every exit (CX-1, PRD §8.2, this task's design
    decisions).

    Soft-cancel is DETECTED by racing *cancel_event* against the body task
    via ``asyncio.wait`` — never injected as a ``CancelledError`` into the
    body — so it can never be silently swallowed by an inner
    ``except asyncio.CancelledError`` / ``contextlib.suppress`` block
    somewhere inside the body (``_drive()`` and its callees have several).
    Hard-cancel is the outer ``await`` (this coroutine's own await point)
    being cancelled, or the body itself spontaneously raising
    ``CancelledError`` (a shutdown-race teardown) — both typed ``'hard'``.
    """

    def __init__(
        self,
        cancel_event: asyncio.Event,
        on_terminal: Sequence[OnTerminalEntry],
    ) -> None:
        self._cancel_event = cancel_event
        self._on_terminal = on_terminal

    async def supervise(self, body_coro: Awaitable[_T]) -> _T:
        """Run *body_coro* under supervision; return its result, or raise
        :class:`WorkflowCancelled` on hard/soft cancellation.

        The resolved cancellation *kind* (``None`` on a normal exit) is
        always passed to every ``on_terminal`` entry, in order, exactly
        once — via the ``finally`` below, so it runs on every exit path.
        """
        body = asyncio.ensure_future(body_coro)
        waiter = asyncio.create_task(self._cancel_event.wait())
        kind: Literal['hard', 'soft'] | None = None
        try:
            try:
                done, _pending = await asyncio.wait(
                    {body, waiter}, return_when=asyncio.FIRST_COMPLETED,
                )
            except asyncio.CancelledError:
                # This coroutine's own await was cancelled — the harness's
                # task.cancel() (or any other outer cancellation).  body and
                # waiter are untouched by asyncio.wait()'s own cancellation
                # (it never cancels its member futures on our behalf), so
                # both are handled uniformly by the finally below.
                kind = 'hard'
                raise WorkflowCancelled('hard') from None
            if body in done:
                if body.cancelled():
                    # The body itself raised CancelledError spontaneously
                    # (a shutdown-race teardown) — matches the old exc_info
                    # sniff, which caught ANY CancelledError propagating
                    # through the finally, not just an externally-injected
                    # one.
                    kind = 'hard'
                    raise WorkflowCancelled('hard')
                exc = body.exception()
                if exc is not None:
                    # The body itself raised WorkflowCancelled directly as
                    # ordinary control flow (W9-θ: e.g. the merge-retry
                    # loop's explicit cancel re-check, or _await_cancellable
                    # via _submit_to_merge_queue) rather than this scope's
                    # own event-race detecting it below. Without capturing
                    # its kind here, `kind` would stay None and on_terminal
                    # would run as if this were a normal exit — silently
                    # skipping the kind-aware lane-release policy (e.g. a
                    # soft-cancel's release, boundary row 15).
                    if isinstance(exc, WorkflowCancelled):
                        kind = exc.kind
                    raise exc
                return body.result()
            # waiter resolved first (event fired) and body is still
            # pending — a genuine soft-cancel, not a same-window race
            # (the `body in done` branch above already wins any race
            # where both resolved in the same asyncio.wait() window).
            kind = 'soft'
            raise WorkflowCancelled('soft')
        finally:
            for t in (waiter, body):
                if not t.done():
                    t.cancel()
                    # Deliberately NOT the shield()+uncancel() discipline
                    # used by _run_on_terminal below: that pattern relies on
                    # its background task NEVER being cancelled itself, so
                    # every CancelledError it catches is unambiguously
                    # outer-directed. Here `t` is a task WE just cancelled
                    # on the line above, so a CancelledError surfacing at
                    # this await is its ordinary, expected completion, not
                    # an outer re-cancel — shielding and unconditionally
                    # uncancel()-ing on every catch would misattribute that
                    # expected completion as a handled outer cancel far more
                    # often than the imbalance it would fix. A genuine
                    # repeated outer cancel (the harness's hard_cancel_workflow
                    # poll loop) CAN still land in this unshielded await and
                    # leave current_task().cancelling() one higher than
                    # "balanced" when merely suppressed here — but nothing
                    # in this codebase reads Task.cancelling(), and this
                    # method still runs to completion and returns/raises
                    # correctly either way, so the imbalance is inert.
                    with contextlib.suppress(asyncio.CancelledError):
                        await t
            await self._run_on_terminal(kind)

    async def _run_on_terminal(self, kind: Literal['hard', 'soft'] | None) -> None:
        """Run every ``on_terminal`` entry, in order, passing *kind*.

        Runs the whole ordered list as a background task shielded behind a
        retry loop, so it survives being cancelled REPEATEDLY (the harness's
        ``hard_cancel_workflow`` polls, calling ``task.cancel()`` more than
        once on the same slot task): each ``asyncio.shield`` only ever
        detaches OUR await from the background task, never cancels the
        background task itself, so a second (or third) cancel here can
        never truncate an in-flight cleanup entry.

        Each entry is also individually try/except-``Exception``-and-logged,
        so a failure in one cleanup step can never skip the REMAINING
        steps — notably ``release_lane``/``cleanup_config_dir`` — or escape
        this method and mask the in-flight ``WorkflowCancelled`` that
        ``supervise``'s ``finally`` is already unwinding (which would
        otherwise surface to ``run()`` as an arbitrary exception and have
        the harness report BLOCKED instead of CANCELLED). This generalizes
        ``_stop_claimant_heartbeat``'s own bespoke catch-and-log into one
        uniform per-entry policy, keyed on the entry name — a genuine
        ``CancelledError`` is deliberately NOT caught here so the
        done/cancelled bookkeeping below still sees it.
        """
        async def _run_all() -> None:
            for name, fn in self._on_terminal:
                try:
                    await fn(kind)
                except Exception:
                    logger.exception(
                        f'CancellationScope on_terminal entry {name!r} '
                        f'failed (non-fatal — continuing remaining '
                        f'terminal cleanup)'
                    )

        task = asyncio.ensure_future(_run_all())
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()
        if task.cancelled():
            raise asyncio.CancelledError()
        exc = task.exception()
        if exc is not None:
            raise exc
