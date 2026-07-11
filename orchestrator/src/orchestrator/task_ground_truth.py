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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.deploy_state import DeployPhase

__all__ = [
    'BranchState',
    'BranchStateKind',
    'Claimant',
    'ClaimantSource',
    'EscalationRef',
    'RecoveryAction',
    'TruthReport',
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
