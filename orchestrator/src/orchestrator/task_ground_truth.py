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

import asyncio
import enum
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import ValidationError
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


def _lock_fresh(locked_at: object, now: datetime, ttl: timedelta) -> bool:
    """Return True if a plan.lock's ``locked_at`` is within *ttl* of *now*.

    Mirrors ``TaskArtifacts.clear_stale_plan_lock``'s own age-based
    staleness check (its 600s default equals ``_DEFAULT_HEARTBEAT_TTL``
    here) so a plan.lock owner_pid that happens to be alive isn't honored
    as a live claimant on that basis alone — under PID reuse, a dead
    orchestrator's pid can be recycled by an unrelated, genuinely-alive
    process, which would otherwise read as a phantom-live claimant and
    silently block recovery of a genuinely stranded task (review finding).
    ``read_plan_lock`` itself does no staleness eviction (that lives only
    in ``clear_stale_plan_lock``), so this resolver applies its own check
    rather than assuming a stale lock was already evicted.

    A missing/non-string/unparseable *locked_at* degrades to "not fresh" —
    the same conservative default the ``owner_pid`` guard below uses for
    malformed data, and mirrors ``clear_stale_plan_lock`` treating an
    unparseable timestamp as stale.
    """
    if not isinstance(locked_at, str):
        return False
    try:
        age = now - datetime.fromisoformat(locked_at)
    except (ValueError, TypeError):
        return False
    return age <= ttl


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
        ``live_claimant``, and ``deploy_phase`` — no second fetch. Likewise,
        ``worktree_resolver(tid)`` is called exactly ONCE here and the
        resulting path is shared with ``_resolve_live_claimant``'s
        plan.lock read — no second resolve (review finding #2).

        ``_resolve_branch_state`` (git archaeology on a journal miss) and
        ``scheduler.get_task`` (the db row fetch) are mutually independent
        — neither reads the other's result — so they run concurrently via
        ``asyncio.gather`` rather than one after the other. θ2 calls
        ``derive_truth`` once per stranded task across seven reconcile
        sweeps, so serializing two independent awaits would needlessly
        double the per-task wall-clock in that hot path (review finding).
        """
        branch_state, task = await asyncio.gather(
            self._resolve_branch_state(tid),
            self.scheduler.get_task(tid),
        )
        task = task or {}
        worktree_path = self.worktree_resolver(tid)
        return TruthReport(
            db_status=task.get('status') or '',
            live_claimant=self._resolve_live_claimant(tid, task, worktree_path),
            branch_state=branch_state,
            worktree_present=worktree_path.exists(),
            open_escalations=self._resolve_open_escalations(tid),
            deploy_phase=self._resolve_deploy_phase(task.get('metadata')),
        )

    async def recovery_for(self, tid: str) -> tuple[TruthReport, RecoveryAction]:
        """Compose :meth:`derive_truth` -> :func:`classify_recovery` (TG-2).

        The thin seam θ2's seven harness reconcile sweeps call in place of
        re-deriving recovery policy themselves.
        """
        report = await self.derive_truth(tid)
        return report, classify_recovery(report)

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
        # Safe to call is_ancestor before confirming `branch` exists: per
        # git_ops.py, `_run` maps every git exit code to a plain
        # (rc, stdout, stderr) tuple, and `is_ancestor` reduces that to
        # `rc == 0` — so `git merge-base --is-ancestor` against a missing
        # ref exits non-zero ("fatal: not a valid object name") and is
        # folded into False exactly like a real non-ancestor result; it
        # never raises for that case. `_run` raises `WorktreeMissing` only
        # when the worktree directory itself (`cwd`) is gone, which is a
        # distinct failure mode from a missing branch ref inside an
        # existing worktree. (Verified by inspection against git_ops.py —
        # task 2242 amendment review finding #5; an integration-style test
        # against real git for this path is deferred to θ2 wiring, per that
        # review's own suggested fix.)
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

    def _resolve_live_claimant(self, tid: str, task: dict, worktree_path: Path) -> Claimant | None:
        """Resolve *tid*'s live claimant (TG-3), folding three signals in
        priority order:

        1. ``scheduler.is_actively_held(tid)`` — the in-memory public
           accessor (task 2235).
        2. A fresh W2 db claimant: ``claimant_run_id`` present AND NOT
           ``shared.task_claimant.is_stranded`` against the injected
           ``now_fn``/``heartbeat_ttl``.
        3. A live ``plan.lock`` (owner_pid alive AND ``locked_at`` within
           ``heartbeat_ttl`` of ``now_fn()`` — see :func:`_lock_fresh`; the
           staleness cross-check guards against a PID-reuse false-live read,
           review finding) — consulted ONLY when the db claimant is
           genuinely absent (pre-2182 rows predating the claimant_run_id/
           heartbeat_at columns).

        A present-but-stale db claimant (``is_stranded`` True) collapses
        straight to ``None`` — it deliberately does NOT fall through to the
        plan.lock check, which exists solely for the claimant-absent case.

        NOTE (``is_stranded`` is in-progress-only): ``is_stranded`` returns
        False unconditionally whenever ``db_status`` is anything other than
        ``'in-progress'`` (its own contract — task 2182). So a non-in-progress
        task (e.g. ``blocked``) that still carries a non-blank
        ``claimant_run_id`` — left behind by a crash rather than a clean
        release, which clears it (harness.py:5810) — is treated as LIVE here
        regardless of heartbeat age, and ``classify_recovery`` then defers to
        LEAVE rather than reaching the blocked/no-claimant
        RE_FILE_ESCALATION row (g). This is an accepted, intentionally scoped
        edge case, pinned by
        ``test_stale_db_claimant_on_blocked_task_is_treated_as_live_by_design``
        — not an oversight. Broadening the staleness check to non-in-progress
        statuses would mean re-deriving ``is_stranded``'s own contract here;
        left to a follow-up if this edge case proves to matter in practice.

        *task* is the already-fetched row (:meth:`derive_truth` fetches it
        exactly once and shares it across every field that needs it) and
        *worktree_path* is :meth:`derive_truth`'s single
        ``worktree_resolver(tid)`` resolution (also shared with
        ``worktree_present``, review finding #2) — this method makes no I/O
        of its own beyond the plan.lock read under *worktree_path*.
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

        try:
            lock_data = TaskArtifacts(worktree_path).read_plan_lock()
        except (ValueError, OSError):
            # A truncated/corrupt plan.lock is a realistic outcome of the
            # very crash this resolver recovers from — degrade to "no
            # plan-lock claimant" rather than letting one bad lock file
            # abort the whole ground-truth sweep for this task.
            # ValueError (not just json.JSONDecodeError, itself already a
            # ValueError subclass) also catches UnicodeDecodeError: a
            # byte-corrupt, non-UTF-8 plan.lock raises UnicodeDecodeError
            # from `read_plan_lock`'s `lock_path.read_text()` — a ValueError
            # subclass, but NOT an OSError/JSONDecodeError subclass — so the
            # narrower tuple let a byte-corrupt lock escape uncaught
            # (review finding #2).
            lock_data = None
        if lock_data is not None:
            owner_pid = lock_data.get('owner_pid')
            try:
                owner_alive = owner_pid is not None and _pid_alive(int(owner_pid))
            except (TypeError, ValueError):
                owner_alive = False
            lock_fresh = _lock_fresh(lock_data.get('locked_at'), self.now_fn(), self.heartbeat_ttl)
            if owner_alive and lock_fresh:
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

    def _resolve_deploy_phase(self, metadata: object) -> DeployPhase | None:
        """Resolve a deterministic task's ``deploy_state.phase`` (DS-1/ε).

        ``None`` when *metadata* isn't a mapping, when it carries no
        ``deploy_state`` slice at all, or when the slice fails to validate.

        ``DeployState.from_metadata`` only guards that the slice is a dict
        before doing ``cls(**slice_)`` — a metadata blob carrying a
        malformed slice (e.g. an invalid ``phase`` value, or one missing
        the required ``phase`` key) raises pydantic ``ValidationError``; a
        slice dict with non-string keys raises ``TypeError`` from that same
        ``**slice_`` unpack. This mirrors ``shared.task_metadata``'s
        identical ``except (ValidationError, TypeError)`` guard around the
        same ``submodel(**parsed[key])`` shape. Either failure degrades to
        "no deploy state" rather than aborting the whole ground-truth sweep
        for this task — exactly the partially-written/corrupt state this
        resolver exists to recover from (review finding #1), mirroring the
        plan.lock corruption handling in :meth:`_resolve_live_claimant`.
        """
        if not isinstance(metadata, dict):
            return None
        try:
            deploy_state = DeployState.from_metadata(metadata)
        except (ValidationError, TypeError):
            return None
        return deploy_state.phase if deploy_state is not None else None


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
    # evidence), and no escalation already open at ANY level -> the work
    # landed; mark done with provenance. (If an escalation IS already open —
    # any level — the shape instead matches row (f)'s veto below.)
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
    # (f) An escalation already open at ANY level (L0/L1/L2 — not just L1)
    # is the deliberate human/automation-handoff signal — a sweep must never
    # second-guess it, even with on-main landing evidence (review finding
    # #1: this used to check level==1 only, so an open L2 slipped through
    # and still hit row (a)'s auto-flip).
    (TaskStatus.IN_PROGRESS, False, BranchStateKind.ON_MAIN, True, None):
        RecoveryAction.LEAVE,
    # (g) Stranded 'blocked' with no landing evidence and no escalation
    # already open at any level: blocked discipline forbids a silent
    # blocked->pending revert, so the sweep must re-file an escalation
    # rather than guess. If an escalation IS already open (any level), the
    # shape is absent from this table and falls through to the LEAVE
    # default — re-filing over an already-open escalation would risk a
    # duplicate/competing one (review finding #1).
    # NOTE (θ2 visibility): unreachable for a 'blocked' task that still
    # carries a stale-but-present db claimant_run_id — is_stranded is
    # in-progress-only, so that shape resolves live_claimant=True and hits
    # the LEAVE default instead of this row. Accepted, intentionally scoped
    # blind spot — see the NOTE in _resolve_live_claimant's docstring for
    # the full rationale and the follow-up path if it bites in practice
    # (review finding).
    (TaskStatus.BLOCKED, False, BranchStateKind.GONE_NO_MARKER, False, None):
        RecoveryAction.RE_FILE_ESCALATION,
    # (h) D1: a deterministic deploy crashed between 'ran' and 'verified',
    # with no escalation already open at any level. This is a NAMED,
    # in-flight deploy state — never phantom-done and never silently
    # reverted (that would re-run a deploy that may have already taken
    # effect); re-file an escalation for a human/DS-2 gate. A
    # DeployPhase.RAN task typically already carries the runner's own
    # born-at-L2 escalation, in which case the shape falls through to the
    # LEAVE default instead — same duplicate-escalation avoidance as row
    # (g) (review finding #1).
    (TaskStatus.IN_PROGRESS, False, BranchStateKind.GONE_NO_MARKER, False, DeployPhase.RAN):
        RecoveryAction.RE_FILE_ESCALATION,
    # Deliberately-unmapped deploy phases (VERIFIED / FAILED / SCHEDULED /
    # ESCALATED / DONE): `RAN` is the only deploy_phase this table
    # discriminates on, because it is the sole phase D1 names as a crashed
    # in-flight deploy (PRD §7) — every other stranded-in-progress shape
    # falls through to the LEAVE default below, and that is a deliberate,
    # tested choice (see TestClassifyRecovery's
    # test_h2_deploy_phase_failed_defaults_to_leave_deliberately /
    # test_h3_deploy_phase_verified_defaults_to_leave_deliberately), not an
    # accidental gap:
    #   - VERIFIED / DONE are terminal-success phases; a task stranded
    #     alongside one is an inconsistent/degenerate shape, not the D1
    #     crashed-mid-deploy case, so there is no evidence-backed action
    #     to take beyond LEAVE.
    #   - FAILED / ESCALATED already have their OWN mandatory recovery path
    #     in the DS-2 deploy-phase state machine itself — `_LEGAL` (see
    #     orchestrator/deploy_state.py) requires FAILED -> ESCALATED to file
    #     a loud escalation via `enforce_transition`'s `escalation_sink`
    #     ("DS-2 loudness — never silently dropped"). A second,
    #     independent RE_FILE_ESCALATION from this table for the same
    #     failure would risk a duplicate/competing escalation rather than
    #     deferring to that dedicated machinery.
    # If fleet experience shows a stranded FAILED deploy genuinely needs a
    # THIRD path (neither DS-2's own escalation nor this table), that is a
    # follow-up table row for whoever owns that evidence — not a guess here.
}


def _shape(report: TruthReport) -> _RecoveryShape:
    """Discretize *report* to the tuple `_RECOVERY` is keyed on.

    The escalation-boolean element folds ANY open escalation, at ANY level
    (L0/L1/L2) — not just L1. An escalation already open at any level is the
    same "don't second-guess a pending human/automation handoff" signal
    regardless of which tier is currently holding it, so rows (a)/(f) and
    (g)/(h) all key off this one boolean (review finding #1: a level-1-only
    check let an open L2 slip through row (a)'s veto and let rows (g)/(h)
    re-file over an already-open L0/L2).
    """
    has_open_escalation = bool(report.open_escalations)
    return (
        report.db_status,
        report.live_claimant is not None,
        report.branch_state.kind,
        has_open_escalation,
        report.deploy_phase,
    )


def classify_recovery(report: TruthReport) -> RecoveryAction:
    """Map *report* to a recovery action via the single `_RECOVERY` table (TG-2).

    Any shape absent from `_RECOVERY` — including every shape with a live
    claimant present — defaults to `RecoveryAction.LEAVE` (fail-safe: never
    phantom-done, never guess on an unrecognized shape).
    """
    return _RECOVERY.get(_shape(report), RecoveryAction.LEAVE)
