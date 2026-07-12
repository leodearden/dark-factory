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
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from shared.deploy_state import DeployPhase, DeployState
from shared.task_claimant import is_stranded
from shared.task_statuses import TaskStatus

from orchestrator.artifacts import TaskArtifacts
from orchestrator.landed_outbox import MergeProvenance

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from escalation.queue import EscalationQueue

    from orchestrator.git_ops import GitOps
    from orchestrator.scheduler import Scheduler

__all__ = [
    'BranchState',
    'BranchStateKind',
    'Claimant',
    'ClaimantSource',
    'EscalationRef',
    'RecoveryAction',
    'TaskGroundTruth',
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


def _utc_now() -> datetime:
    """Default ``now_fn`` — real wall-clock UTC time."""
    return datetime.now(UTC)


# Default staleness threshold for the W2 db claimant signal (TG-3 / step-10).
# Callers that care about the exact TTL (e.g. harness wiring in θ2) pass
# their own value explicitly; this default only matters for callers that
# don't.
_DEFAULT_HEARTBEAT_TTL = timedelta(minutes=10)


def _pid_alive(pid: int) -> bool:
    """Return True if the process identified by *pid* is alive.

    Duplicates ``orchestrator.harness._pid_alive`` (itself mirroring
    fused-memory's ``orchestrator_detector.py:58-72``) rather than importing
    it, to avoid a harness->task_ground_truth->harness circular import once
    θ2 migrates the harness's reconcile sweeps to call derive_truth/
    recovery_for.

    - Returns False for pid <= 0 (invalid).
    - Uses os.kill(pid, 0): success → alive; ProcessLookupError → dead;
      PermissionError → alive (we can see it but lack permission to signal it);
      other OSError → treat as dead.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


class TaskGroundTruth:
    """Composes one task's DB/git/liveness/escalation/deploy signals into a
    single frozen :class:`TruthReport` (PRD §5.4).

    Collaborators are injected so :meth:`derive_truth` stays unit-testable
    with lightweight fakes — no reach-ins into any collaborator's private
    state (TG-3: ``scheduler.is_actively_held``/``is_dispatched`` are the
    only Scheduler liveness surface consulted). ``MergeProvenance`` is
    deliberately NOT injected — its own contract is a process-global façade
    (see ``landed_outbox.py``), consumed as such via ``MergeProvenance.lookup``.
    """

    def __init__(
        self,
        git_ops: GitOps,
        scheduler: Scheduler,
        escalation_queue: EscalationQueue | None,
        worktree_resolver: Callable[[str], Path],
        *,
        now_fn: Callable[[], datetime] = _utc_now,
        heartbeat_ttl: timedelta = _DEFAULT_HEARTBEAT_TTL,
    ) -> None:
        self.git_ops = git_ops
        self.scheduler = scheduler
        self.escalation_queue = escalation_queue
        self.worktree_resolver = worktree_resolver
        self.now_fn = now_fn
        self.heartbeat_ttl = heartbeat_ttl

    async def derive_truth(self, tid: str) -> TruthReport:
        """Compose a fresh :class:`TruthReport` for *tid* (PRD §5.4).

        TG-1: ``branch_state`` resolves journal-first — see
        :meth:`_resolve_branch_state`. TG-3: ``live_claimant`` folds the
        in-memory/db/plan.lock liveness signals — see
        :meth:`_resolve_live_claimant`. The task row is fetched exactly
        ONCE here and shared across ``db_status``, the db-claimant leg of
        ``live_claimant``, and ``deploy_phase`` — no second fetch.
        """
        branch_state = await self._resolve_branch_state(tid)
        task = await self.scheduler.get_task(tid) or {}
        metadata = task.get('metadata')
        deploy_state = DeployState.from_metadata(metadata) if isinstance(metadata, dict) else None
        return TruthReport(
            db_status=task.get('status') or '',
            live_claimant=self._resolve_live_claimant(tid, task),
            branch_state=branch_state,
            worktree_present=self.worktree_resolver(tid).exists(),
            open_escalations=self._resolve_open_escalations(tid),
            deploy_phase=deploy_state.phase if deploy_state is not None else None,
        )

    async def _resolve_branch_state(self, tid: str) -> BranchState:
        """Resolve *tid*'s branch state (TG-1: journal-first).

        ``MergeProvenance.lookup`` is consulted FIRST — a journal hit is
        authoritative and returns without any git I/O at all. Git
        archaeology only runs as a fallback on a journal miss.
        """
        row = MergeProvenance.lookup(tid)
        if row is not None:
            return BranchState(BranchStateKind.ON_MAIN, row.advanced_sha)

        # Git-archaeology fallback (journal miss). EXACTLY ONCE — this is
        # the sole owner of the is_ancestor -> resolve_branch_sha ->
        # find_merge_marker sequence (mirrors the archaeology previously
        # inlined per-sweep in harness._reconcile_one_stranded).
        branch = f'{self.git_ops.config.branch_prefix}{tid}'
        main_branch = self.git_ops.config.main_branch
        if await self.git_ops.is_ancestor(branch, main_branch):
            sha = await self.git_ops.resolve_branch_sha(branch)
            return BranchState(BranchStateKind.ON_MAIN, sha)

        if await self.git_ops.resolve_branch_sha(branch) is not None:
            # Branch ref still exists but is not an ancestor of main — no
            # merged sha to carry (BranchState docstring: sha is only
            # carried for on_main / gone_with_merge_marker).
            return BranchState(BranchStateKind.EXISTS_OFF_MAIN)

        marker_sha = await self.git_ops.find_merge_marker(branch)
        if marker_sha:
            return BranchState(BranchStateKind.GONE_WITH_MERGE_MARKER, marker_sha)
        return BranchState(BranchStateKind.GONE_NO_MARKER)

    def _resolve_live_claimant(self, tid: str, task: dict) -> Claimant | None:
        """Resolve *tid*'s live claimant (TG-3), folding three signals in
        priority order:

        1. ``scheduler.is_actively_held(tid)`` — the in-memory public
           accessor (task 2235).
        2. A fresh W2 db claimant: ``claimant_run_id`` present AND NOT
           ``shared.task_claimant.is_stranded`` against the injected
           ``now_fn``/``heartbeat_ttl``.
        3. A live ``plan.lock`` (owner_pid alive) — consulted ONLY when the
           db claimant is genuinely absent (pre-2182 rows predating the
           claimant_run_id/heartbeat_at columns).

        A present-but-stale db claimant (``is_stranded`` True) collapses
        straight to ``None`` — it deliberately does NOT fall through to the
        plan.lock check, which exists solely for the claimant-absent case.

        *task* is the already-fetched row (:meth:`derive_truth` fetches it
        exactly once and shares it across every field that needs it) — this
        method makes no I/O of its own beyond the plan.lock read.
        """
        if self.scheduler.is_actively_held(tid):
            return Claimant(run_id=None, heartbeat_at=None, source=ClaimantSource.IN_MEMORY)

        claimant_run_id = task.get('claimant_run_id')
        if claimant_run_id and str(claimant_run_id).strip():
            if is_stranded(task, self.now_fn(), self.heartbeat_ttl):
                return None
            return Claimant(
                run_id=claimant_run_id,
                heartbeat_at=task.get('heartbeat_at'),
                source=ClaimantSource.DB,
            )

        lock_data = TaskArtifacts(self.worktree_resolver(tid)).read_plan_lock()
        if lock_data is not None:
            owner_pid = lock_data.get('owner_pid')
            try:
                owner_alive = owner_pid is not None and _pid_alive(int(owner_pid))
            except (TypeError, ValueError):
                owner_alive = False
            if owner_alive:
                return Claimant(
                    run_id=lock_data.get('session_id'),
                    heartbeat_at=lock_data.get('locked_at'),
                    source=ClaimantSource.PLAN_LOCK,
                )
        return None

    def _resolve_open_escalations(self, tid: str) -> list[EscalationRef]:
        """Map *tid*'s pending escalations to lightweight refs.

        ``[]`` when no ``escalation_queue`` was injected — a caller that
        doesn't wire one up gets an empty-but-valid TruthReport field rather
        than an error.
        """
        if self.escalation_queue is None:
            return []
        rows = self.escalation_queue.get_by_task(tid, status='pending')
        return [EscalationRef(id=row.id, level=row.level) for row in rows]


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
