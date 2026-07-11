"""TaskGroundTruth — the single ground-truth resolver (task 2242, W10-θ1).

PRD ``plans/harness-supervision-prd.md`` §5.4 (TG-1..3; survey findings 2.4 +
4.1, ground-truth half). ``TaskGroundTruth.derive_truth(tid)`` composes the
task's DB row, its live in-memory/durable/plan.lock claimant signal, its git
branch state, its worktree presence, its open escalations, and (for
deterministic tasks) its deploy phase into one frozen ``TruthReport``. A
single module-level ``_RECOVERY`` table then maps a ``TruthReport`` shape to
a ``RecoveryAction`` — see the comment above ``_RECOVERY`` for why this table
is deliberately distinct from W2's ``(from,to,actor)`` status-legality table.

This module delivers the resolver + table + unit tests ONLY. Migrating the
seven harness reconcile sweeps to call ``derive_truth``/``recovery_for``
instead of re-deriving recovery policy themselves is a separate task (θ2).

TG-1 (journal-first branch state) / TG-2 (one classification table) / TG-3
(liveness via the public accessor, not scheduler privates) are implemented
in :meth:`TaskGroundTruth.derive_truth` and :func:`classify_recovery` below.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from shared.deploy_state import DeployPhase
from shared.task_statuses import TaskStatus

__all__ = [
    'BranchState',
    'BranchStateKind',
    'Claimant',
    'ClaimantSource',
    'EscalationRef',
    'RecoveryAction',
    'TruthReport',
    'classify_recovery',
]


class BranchStateKind(enum.StrEnum):
    """The four resolvable shapes of a task's branch (PRD §5.4 chart).

    Genuine ``str`` members (mirrors ``shared.task_statuses.TaskStatus`` /
    ``shared.deploy_state.DeployPhase``) so equality against a plain string
    holds without an explicit ``.value``.
    """

    ON_MAIN = 'on_main'
    EXISTS_OFF_MAIN = 'exists_off_main'
    GONE_WITH_MERGE_MARKER = 'gone_with_merge_marker'
    GONE_NO_MARKER = 'gone_no_marker'


class ClaimantSource(enum.StrEnum):
    """Which signal produced a live :class:`Claimant` (TG-3)."""

    DB = 'db'
    PLAN_LOCK = 'plan_lock'
    IN_MEMORY = 'in_memory'


class RecoveryAction(enum.StrEnum):
    """The closed vocabulary of recovery actions ``_RECOVERY`` maps to."""

    MARK_DONE_WITH_PROVENANCE = 'mark_done_with_provenance'
    REVERT_TO_PENDING = 'revert_to_pending'
    RE_FILE_ESCALATION = 're_file_escalation'
    LEAVE = 'leave'


@dataclass(frozen=True)
class BranchState:
    """A task's resolved branch state.

    ``sha`` is carried only for the ``on_main`` / ``gone_with_merge_marker``
    variants — ``exists_off_main`` and ``gone_no_marker`` have no associated
    merged sha, so it defaults to ``None`` for those.
    """

    kind: BranchStateKind
    sha: str | None = None


@dataclass(frozen=True)
class Claimant:
    """A live claimant identity, folded from one of three liveness signals."""

    run_id: str | None
    heartbeat_at: str | None
    source: ClaimantSource


@dataclass(frozen=True)
class EscalationRef:
    """A lightweight reference to an open escalation."""

    id: str
    level: int


@dataclass(frozen=True)
class TruthReport:
    """The single ground-truth snapshot for one task (PRD §5.4).

    Frozen — a point-in-time snapshot, not a live view; callers re-derive
    via :meth:`TaskGroundTruth.derive_truth` for a fresh report.
    """

    db_status: str
    live_claimant: Claimant | None
    branch_state: BranchState
    worktree_present: bool
    open_escalations: list[EscalationRef]
    deploy_phase: DeployPhase | None


# ---------------------------------------------------------------------------
# TG-2 — the recovery-action classification table.
#
# `_RECOVERY` is a RECOVERY-action table (TruthReport-shape -> what a sweep
# should DO), deliberately DISTINCT from W2's `(from,to,actor)` status-
# LEGALITY table (task 2182: which status transitions are allowed for which
# actor). Recovery writes derived from this table's output still flow
# through the normal fused-memory chokepoint that W2's table validates —
# there is no seam collision (PRD §6 G4).
#
# Keyed on a discretized report-shape tuple (see `_shape`); any shape not
# explicitly listed here falls through to RecoveryAction.LEAVE via `.get`'s
# default — fail-safe by construction (never phantom-done on an
# unrecognized shape). In particular, every shape with a live claimant
# present is deliberately left OUT of this table: `_shape` already folds
# `live_claimant is not None` into the key, and no entry below has that
# element True, so any live-claimant shape (for any status/branch/deploy
# combination) resolves to the LEAVE default automatically.
# ---------------------------------------------------------------------------

_RecoveryShape = tuple[str, bool, BranchStateKind, bool, DeployPhase | None]

_RECOVERY: dict[_RecoveryShape, RecoveryAction] = {
    # (a) Stranded in-progress, branch already on main (journal or git
    # evidence) -> the work landed; mark done with provenance.
    (TaskStatus.IN_PROGRESS, False, BranchStateKind.ON_MAIN, False, None):
        RecoveryAction.MARK_DONE_WITH_PROVENANCE,
    # (b) Stranded in-progress, branch gone but a merge marker on main
    # confirms it landed -> same outcome as (a).
    (TaskStatus.IN_PROGRESS, False, BranchStateKind.GONE_WITH_MERGE_MARKER, False, None):
        RecoveryAction.MARK_DONE_WITH_PROVENANCE,
    # (c) Stranded in-progress, branch still exists off-main -> no landing
    # evidence; safe to re-dispatch from pending.
    (TaskStatus.IN_PROGRESS, False, BranchStateKind.EXISTS_OFF_MAIN, False, None):
        RecoveryAction.REVERT_TO_PENDING,
    # (d) Stranded in-progress, branch gone with no marker -> no landing
    # evidence either; same revert-to-pending outcome as (c).
    (TaskStatus.IN_PROGRESS, False, BranchStateKind.GONE_NO_MARKER, False, None):
        RecoveryAction.REVERT_TO_PENDING,
    # (f) An open L1 escalation is the deliberate human-handoff signal — a
    # sweep must never second-guess it, even with on-main landing evidence.
    (TaskStatus.IN_PROGRESS, False, BranchStateKind.ON_MAIN, True, None):
        RecoveryAction.LEAVE,
    # (g) Stranded 'blocked' with no landing evidence and no open
    # escalation: blocked discipline forbids a silent blocked->pending
    # revert, so the sweep must re-file an escalation rather than guess.
    (TaskStatus.BLOCKED, False, BranchStateKind.GONE_NO_MARKER, False, None):
        RecoveryAction.RE_FILE_ESCALATION,
    # (h) D1: a deterministic deploy crashed between 'ran' and 'verified'.
    # This is a NAMED, in-flight deploy state — never phantom-done and
    # never silently reverted (that would re-run a deploy that may have
    # already taken effect); re-file an escalation for a human/DS-2 gate.
    (TaskStatus.IN_PROGRESS, False, BranchStateKind.GONE_NO_MARKER, False, DeployPhase.RAN):
        RecoveryAction.RE_FILE_ESCALATION,
}


def _shape(report: TruthReport) -> _RecoveryShape:
    """Discretize *report* to the tuple `_RECOVERY` is keyed on."""
    has_open_l1 = any(ref.level == 1 for ref in report.open_escalations)
    return (
        report.db_status,
        report.live_claimant is not None,
        report.branch_state.kind,
        has_open_l1,
        report.deploy_phase,
    )


def classify_recovery(report: TruthReport) -> RecoveryAction:
    """Map *report* to a recovery action via the single `_RECOVERY` table (TG-2).

    Any shape absent from `_RECOVERY` — including every shape with a live
    claimant present — defaults to `RecoveryAction.LEAVE` (fail-safe: never
    phantom-done, never guess on an unrecognized shape).
    """
    return _RECOVERY.get(_shape(report), RecoveryAction.LEAVE)
