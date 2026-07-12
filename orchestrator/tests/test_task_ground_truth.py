"""Tests for TaskGroundTruth — the single ground-truth resolver (task 2242, W10-θ1).

PRD ``plans/harness-supervision-prd.md`` §5.4 (TG-1..3). Delivers the resolver
+ ONE ``_RECOVERY`` classification table + unit tests; the seven-sweep harness
migration is a separate task (θ2) and is out of scope here.

Covers:
  step-1:  pure type-surface contract — the frozen value objects and StrEnum
           vocabularies task_ground_truth.py exports, no resolver logic yet.
  step-3:  table-driven classify_recovery(report) -> RecoveryAction over every
           meaningful TruthReport shape (TG-2's single classification table).
  step-5:  derive_truth's branch_state resolves JOURNAL-FIRST via
           MergeProvenance.lookup — git archaeology is not consulted on a hit.
  step-7:  derive_truth's branch_state GIT-FALLBACK on a journal miss, all
           four BranchStateKind shapes.
  step-9:  derive_truth's live_claimant composition — in_memory/db/plan_lock/
           None across the three folded liveness signals (TG-3).
  step-11: derive_truth's remaining TruthReport fields — db_status,
           worktree_present, open_escalations, deploy_phase.
  step-13: recovery_for(tid) — the derive_truth -> classify_recovery seam
           (G1/G2/D1 boundary scenarios from the PRD's boundary-test sketch).
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from escalation.models import Escalation
from shared.deploy_state import DeployPhase

from orchestrator.artifacts import TaskArtifacts
from orchestrator.landed_outbox import LandedOutbox, LandedRow, MergeProvenance
from orchestrator.task_ground_truth import (
    BranchState,
    BranchStateKind,
    Claimant,
    ClaimantSource,
    EscalationRef,
    RecoveryAction,
    TaskGroundTruth,
    TruthReport,
    classify_recovery,
)

# ---------------------------------------------------------------------------
# step-1 — pure type-surface contract
# ---------------------------------------------------------------------------


class TestRecoveryAction:
    def test_members_have_exact_lowercase_values(self) -> None:
        assert RecoveryAction.MARK_DONE_WITH_PROVENANCE == 'mark_done_with_provenance'
        assert RecoveryAction.REVERT_TO_PENDING == 'revert_to_pending'
        assert RecoveryAction.RE_FILE_ESCALATION == 're_file_escalation'
        assert RecoveryAction.LEAVE == 'leave'


class TestBranchStateKind:
    def test_members_have_exact_lowercase_values(self) -> None:
        assert BranchStateKind.ON_MAIN == 'on_main'
        assert BranchStateKind.EXISTS_OFF_MAIN == 'exists_off_main'
        assert BranchStateKind.GONE_WITH_MERGE_MARKER == 'gone_with_merge_marker'
        assert BranchStateKind.GONE_NO_MARKER == 'gone_no_marker'


class TestFrozenValueObjects:
    """BranchState / Claimant / EscalationRef are frozen dataclasses."""

    def test_branch_state_is_frozen(self) -> None:
        state = BranchState(BranchStateKind.ON_MAIN, 'deadbeef')
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.sha = 'other'  # type: ignore[misc]

    def test_claimant_is_frozen(self) -> None:
        claimant = Claimant(
            run_id='run-1', heartbeat_at='2026-07-12T00:00:00+00:00', source=ClaimantSource.DB,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            claimant.run_id = 'other'  # type: ignore[misc]

    def test_escalation_ref_is_frozen(self) -> None:
        ref = EscalationRef(id='esc-1-1', level=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ref.level = 2  # type: ignore[misc]


class TestTruthReportConstruction:
    def test_constructs_with_all_six_fields(self) -> None:
        report = TruthReport(
            db_status='in-progress',
            live_claimant=None,
            branch_state=BranchState(BranchStateKind.GONE_NO_MARKER, None),
            worktree_present=True,
            open_escalations=[],
            deploy_phase=None,
        )
        assert report.db_status == 'in-progress'
        assert report.live_claimant is None
        assert report.branch_state == BranchState(BranchStateKind.GONE_NO_MARKER, None)
        assert report.worktree_present is True
        assert report.open_escalations == []
        assert report.deploy_phase is None

    def test_deploy_phase_and_open_escalations_accept_real_values(self) -> None:
        claimant = Claimant(run_id='run-1', heartbeat_at=None, source=ClaimantSource.IN_MEMORY)
        report = TruthReport(
            db_status='in-progress',
            live_claimant=claimant,
            branch_state=BranchState(BranchStateKind.ON_MAIN, 'sha'),
            worktree_present=False,
            open_escalations=[EscalationRef(id='esc-1-1', level=1)],
            deploy_phase=DeployPhase.RAN,
        )
        assert report.live_claimant == claimant
        assert report.deploy_phase == DeployPhase.RAN
        assert report.open_escalations == [EscalationRef(id='esc-1-1', level=1)]


# ---------------------------------------------------------------------------
# step-3 — table-driven classify_recovery (TG-2: ONE classification table)
# ---------------------------------------------------------------------------


class TestClassifyRecovery:
    """TruthReport shape -> expected RecoveryAction.

    Each row is a distinct, meaningful shape from PRD §5.4 / plan.json
    step-3. Unmapped/degenerate shapes default to LEAVE (fail-safe — never
    phantom-done); see row (i).
    """

    @staticmethod
    def _report(
        *,
        db_status: str = 'in-progress',
        live_claimant: Claimant | None = None,
        branch_state: BranchState,
        open_escalations: list[EscalationRef] | None = None,
        deploy_phase: DeployPhase | None = None,
    ) -> TruthReport:
        return TruthReport(
            db_status=db_status,
            live_claimant=live_claimant,
            branch_state=branch_state,
            worktree_present=True,
            open_escalations=open_escalations or [],
            deploy_phase=deploy_phase,
        )

    def test_a_in_progress_no_claimant_on_main_no_open_l1_marks_done(self) -> None:
        report = self._report(branch_state=BranchState(BranchStateKind.ON_MAIN, 'sha-a'))
        assert classify_recovery(report) == RecoveryAction.MARK_DONE_WITH_PROVENANCE

    def test_b_in_progress_no_claimant_gone_with_marker_marks_done(self) -> None:
        report = self._report(
            branch_state=BranchState(BranchStateKind.GONE_WITH_MERGE_MARKER, 'sha-b'),
        )
        assert classify_recovery(report) == RecoveryAction.MARK_DONE_WITH_PROVENANCE

    def test_c_in_progress_no_claimant_exists_off_main_reverts_to_pending(self) -> None:
        report = self._report(branch_state=BranchState(BranchStateKind.EXISTS_OFF_MAIN))
        assert classify_recovery(report) == RecoveryAction.REVERT_TO_PENDING

    def test_d_in_progress_no_claimant_gone_no_marker_reverts_to_pending(self) -> None:
        report = self._report(branch_state=BranchState(BranchStateKind.GONE_NO_MARKER))
        assert classify_recovery(report) == RecoveryAction.REVERT_TO_PENDING

    def test_e_any_status_live_claimant_present_leaves(self) -> None:
        claimant = Claimant(
            run_id='run-1', heartbeat_at='2026-07-12T00:00:00+00:00', source=ClaimantSource.IN_MEMORY,
        )
        report = self._report(
            db_status='blocked',
            live_claimant=claimant,
            branch_state=BranchState(BranchStateKind.ON_MAIN, 'sha-e'),
        )
        assert classify_recovery(report) == RecoveryAction.LEAVE

    def test_f_on_main_open_l1_vetoes_auto_flip(self) -> None:
        report = self._report(
            branch_state=BranchState(BranchStateKind.ON_MAIN, 'sha-f'),
            open_escalations=[EscalationRef(id='esc-1-1', level=1)],
        )
        assert classify_recovery(report) == RecoveryAction.LEAVE

    def test_g_blocked_no_claimant_gone_no_marker_no_open_escalation_refiles(self) -> None:
        report = self._report(db_status='blocked', branch_state=BranchState(BranchStateKind.GONE_NO_MARKER))
        assert classify_recovery(report) == RecoveryAction.RE_FILE_ESCALATION

    def test_h_deterministic_task_deploy_phase_ran_refiles_not_phantom_done(self) -> None:
        # D1 (PRD §7): a deploy crashed between 'ran' and 'verified' must
        # recover to a defined action, never phantom-done.
        report = self._report(
            branch_state=BranchState(BranchStateKind.GONE_NO_MARKER),
            deploy_phase=DeployPhase.RAN,
        )
        assert classify_recovery(report) == RecoveryAction.RE_FILE_ESCALATION

    def test_i_unmapped_degenerate_shape_defaults_to_leave(self) -> None:
        report = self._report(db_status='pending', branch_state=BranchState(BranchStateKind.EXISTS_OFF_MAIN))
        assert classify_recovery(report) == RecoveryAction.LEAVE


# ---------------------------------------------------------------------------
# step-5 — derive_truth's branch_state resolves JOURNAL-FIRST (TG-1)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_merge_provenance():
    """``MergeProvenance._outbox`` is a process-global — never leak a bound
    outbox into another test (mirrors test_workflow_merge_provenance.py's
    identically-named fixture)."""
    MergeProvenance._outbox = None
    yield
    MergeProvenance._outbox = None


def _bind_landed_row(tmp_path: Path, *, task_id: str, advanced_sha: str) -> None:
    """Bind a real LandedOutbox (via MergeProvenance.bind) holding a row for *task_id*."""
    outbox = LandedOutbox(tmp_path / 'landed.json')
    outbox.record(LandedRow(
        task_id=task_id, branch_tip_sha='branchtip', advanced_sha=advanced_sha,
        landed_at=1.0,
    ))
    MergeProvenance.bind(outbox)


def _fake_git_ops(
    *,
    branch_prefix: str = 'task/',
    main_branch: str = 'main',
    is_ancestor: bool = False,
    branch_sha: str | None = None,
    marker_sha: str | None = None,
) -> MagicMock:
    """A minimal git_ops double exposing exactly the async surface
    TaskGroundTruth's branch-state archaeology consumes, plus the
    ``config.branch_prefix`` / ``config.main_branch`` attributes it derives
    the branch name and main-ref from."""
    git_ops = MagicMock()
    git_ops.config = MagicMock(branch_prefix=branch_prefix, main_branch=main_branch)
    git_ops.is_ancestor = AsyncMock(return_value=is_ancestor)
    git_ops.resolve_branch_sha = AsyncMock(return_value=branch_sha)
    git_ops.find_merge_marker = AsyncMock(return_value=marker_sha)
    return git_ops


def _fake_scheduler(
    *,
    is_actively_held: bool = False,
    task: dict | None = None,
) -> MagicMock:
    """A minimal scheduler double exposing exactly the TG-3 liveness surface
    (``is_actively_held`` — sync) and the task-fetch surface
    (``get_task`` — async) TaskGroundTruth's live_claimant resolution
    consumes."""
    scheduler = MagicMock()
    scheduler.is_actively_held = MagicMock(return_value=is_actively_held)
    scheduler.get_task = AsyncMock(return_value=task)
    return scheduler


def _make_ground_truth(
    *,
    git_ops: MagicMock | None = None,
    scheduler: MagicMock | None = None,
    escalation_queue: MagicMock | None = None,
    worktree_resolver=None,
    now_fn=None,
    heartbeat_ttl: timedelta | None = None,
) -> TaskGroundTruth:
    kwargs = {}
    if now_fn is not None:
        kwargs['now_fn'] = now_fn
    if heartbeat_ttl is not None:
        kwargs['heartbeat_ttl'] = heartbeat_ttl
    return TaskGroundTruth(
        git_ops or _fake_git_ops(),
        scheduler or _fake_scheduler(),
        escalation_queue,
        worktree_resolver or (lambda tid: Path('/nonexistent-worktree') / tid),
        **kwargs,
    )


@pytest.mark.asyncio
class TestDeriveTruthBranchStateJournalFirst:
    """A ``MergeProvenance`` journal hit is authoritative and short-circuits
    before any git archaeology runs."""

    async def test_journal_hit_resolves_on_main_without_touching_git(
        self, tmp_path: Path,
    ) -> None:
        _bind_landed_row(tmp_path, task_id='42', advanced_sha='advancedsha123')
        git_ops = _fake_git_ops()
        resolver = _make_ground_truth(git_ops=git_ops)

        report = await resolver.derive_truth('42')

        assert report.branch_state == BranchState(BranchStateKind.ON_MAIN, 'advancedsha123')
        git_ops.is_ancestor.assert_not_awaited()
        git_ops.resolve_branch_sha.assert_not_awaited()
        git_ops.find_merge_marker.assert_not_awaited()


# ---------------------------------------------------------------------------
# step-7 — derive_truth's branch_state GIT-FALLBACK on a journal miss (TG-1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDeriveTruthBranchStateGitFallback:
    """On a journal miss (``MergeProvenance`` unbound / no row for *tid*),
    derive_truth falls back to git archaeology: all four ``BranchStateKind``
    shapes, with ``find_merge_marker`` consulted only on the branch-gone
    path."""

    async def test_ancestor_true_resolves_on_main_via_branch_tip(self) -> None:
        git_ops = _fake_git_ops(is_ancestor=True, branch_sha='tipsha123')
        resolver = _make_ground_truth(git_ops=git_ops)

        report = await resolver.derive_truth('7')

        assert report.branch_state == BranchState(BranchStateKind.ON_MAIN, 'tipsha123')
        git_ops.is_ancestor.assert_awaited_once_with('task/7', 'main')
        git_ops.resolve_branch_sha.assert_awaited_once_with('task/7')
        git_ops.find_merge_marker.assert_not_awaited()

    async def test_branch_exists_not_ancestor_resolves_exists_off_main(self) -> None:
        git_ops = _fake_git_ops(is_ancestor=False, branch_sha='tipsha456')
        resolver = _make_ground_truth(git_ops=git_ops)

        report = await resolver.derive_truth('8')

        assert report.branch_state == BranchState(BranchStateKind.EXISTS_OFF_MAIN)
        git_ops.is_ancestor.assert_awaited_once_with('task/8', 'main')
        git_ops.resolve_branch_sha.assert_awaited_once_with('task/8')
        git_ops.find_merge_marker.assert_not_awaited()

    async def test_branch_gone_with_marker_resolves_gone_with_merge_marker(self) -> None:
        git_ops = _fake_git_ops(is_ancestor=False, branch_sha=None, marker_sha='markersha789')
        resolver = _make_ground_truth(git_ops=git_ops)

        report = await resolver.derive_truth('9')

        assert report.branch_state == BranchState(
            BranchStateKind.GONE_WITH_MERGE_MARKER, 'markersha789',
        )
        git_ops.is_ancestor.assert_awaited_once_with('task/9', 'main')
        git_ops.find_merge_marker.assert_awaited_once_with('task/9')

    async def test_branch_gone_no_marker_resolves_gone_no_marker(self) -> None:
        git_ops = _fake_git_ops(is_ancestor=False, branch_sha=None, marker_sha=None)
        resolver = _make_ground_truth(git_ops=git_ops)

        report = await resolver.derive_truth('10')

        assert report.branch_state == BranchState(BranchStateKind.GONE_NO_MARKER, None)
        git_ops.is_ancestor.assert_awaited_once_with('task/10', 'main')
        git_ops.find_merge_marker.assert_awaited_once_with('task/10')


# ---------------------------------------------------------------------------
# step-9 — derive_truth's live_claimant composition (TG-3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDeriveTruthLiveClaimant:
    """live_claimant folds three liveness signals, in priority order:

    1. ``scheduler.is_actively_held(tid)`` (in-memory, TG-3 public accessor).
    2. A fresh W2 db claimant (``claimant_run_id`` present AND NOT
       ``is_stranded``).
    3. A live ``plan.lock`` — consulted ONLY when the db claimant is
       genuinely absent (pre-2182 rows); a present-but-stale db claimant
       collapses straight to ``None`` and never falls through to plan.lock.
    """

    async def test_actively_held_returns_in_memory_claimant(self) -> None:
        # get_task IS still awaited once here (step-12: derive_truth fetches
        # the task row unconditionally for db_status/deploy_phase,
        # regardless of which live_claimant signal wins) — only the
        # is_actively_held short-circuit's OWN resolution skips consulting
        # the fetched row's claimant columns / plan.lock.
        scheduler = _fake_scheduler(is_actively_held=True)
        resolver = _make_ground_truth(scheduler=scheduler)

        report = await resolver.derive_truth('11')

        assert report.live_claimant is not None
        assert report.live_claimant.source == ClaimantSource.IN_MEMORY
        scheduler.is_actively_held.assert_called_once_with('11')

    async def test_fresh_db_claimant_returns_db_claimant(self) -> None:
        fixed_now = datetime(2026, 7, 12, 12, 0, 0, tzinfo=UTC)
        task = {
            'status': 'in-progress',
            'claimant_run_id': 'run-1/session-1/pid=123',
            'heartbeat_at': '2026-07-12T11:55:00+00:00',  # 5 minutes stale
        }
        scheduler = _fake_scheduler(is_actively_held=False, task=task)
        resolver = _make_ground_truth(
            scheduler=scheduler, now_fn=lambda: fixed_now, heartbeat_ttl=timedelta(minutes=10),
        )

        report = await resolver.derive_truth('12')

        assert report.live_claimant == Claimant(
            run_id='run-1/session-1/pid=123',
            heartbeat_at='2026-07-12T11:55:00+00:00',
            source=ClaimantSource.DB,
        )
        scheduler.get_task.assert_awaited_once_with('12')

    async def test_no_db_claimant_live_plan_lock_returns_plan_lock_claimant(
        self, tmp_path: Path,
    ) -> None:
        TaskArtifacts(tmp_path).root.mkdir(parents=True)
        TaskArtifacts(tmp_path).lock_plan('sess-13-abc123')
        lock_data = TaskArtifacts(tmp_path).read_plan_lock()
        assert lock_data is not None
        locked_at = lock_data['locked_at']
        task = {'status': 'pending', 'claimant_run_id': None, 'heartbeat_at': None}
        scheduler = _fake_scheduler(is_actively_held=False, task=task)
        resolver = _make_ground_truth(scheduler=scheduler, worktree_resolver=lambda tid: tmp_path)

        report = await resolver.derive_truth('13')

        assert report.live_claimant == Claimant(
            run_id='sess-13-abc123',
            heartbeat_at=locked_at,
            source=ClaimantSource.PLAN_LOCK,
        )

    async def test_stale_db_claimant_returns_none_even_with_live_plan_lock(
        self, tmp_path: Path,
    ) -> None:
        # A present-but-stale db claimant must collapse straight to None —
        # it must NOT fall through to the plan.lock fallback, which exists
        # only for rows with no db claimant at all (pre-2182 rows).
        TaskArtifacts(tmp_path).root.mkdir(parents=True)
        TaskArtifacts(tmp_path).lock_plan('sess-14-abc123')
        fixed_now = datetime(2026, 7, 12, 12, 0, 0, tzinfo=UTC)
        task = {
            'status': 'in-progress',
            'claimant_run_id': 'run-1/session-1/pid=999',
            'heartbeat_at': '2026-07-12T11:00:00+00:00',  # 60 minutes stale
        }
        scheduler = _fake_scheduler(is_actively_held=False, task=task)
        resolver = _make_ground_truth(
            scheduler=scheduler,
            worktree_resolver=lambda tid: tmp_path,
            now_fn=lambda: fixed_now,
            heartbeat_ttl=timedelta(minutes=10),
        )

        report = await resolver.derive_truth('14')

        assert report.live_claimant is None

    async def test_no_claimant_and_no_plan_lock_returns_none(self) -> None:
        task = {'status': 'pending', 'claimant_run_id': None, 'heartbeat_at': None}
        scheduler = _fake_scheduler(is_actively_held=False, task=task)
        resolver = _make_ground_truth(scheduler=scheduler)

        report = await resolver.derive_truth('15')

        assert report.live_claimant is None

    async def test_missing_task_row_returns_none(self) -> None:
        scheduler = _fake_scheduler(is_actively_held=False, task=None)
        resolver = _make_ground_truth(scheduler=scheduler)

        report = await resolver.derive_truth('16')

        assert report.live_claimant is None


# ---------------------------------------------------------------------------
# step-11 — derive_truth's remaining TruthReport fields
# ---------------------------------------------------------------------------


def _fake_escalation_queue(*, rows: list[Escalation] | None = None) -> MagicMock:
    """A minimal escalation_queue double exposing exactly the sync
    ``get_by_task`` surface TaskGroundTruth's open_escalations resolution
    consumes."""
    queue = MagicMock()
    queue.get_by_task = MagicMock(return_value=rows or [])
    return queue


@pytest.mark.asyncio
class TestDeriveTruthRemainingFields:
    """db_status / worktree_present / open_escalations / deploy_phase —
    the four TruthReport fields left stubbed through step-10. Also locks in
    that the task row backing db_status/deploy_phase is fetched exactly
    ONCE per derive_truth call (shared with live_claimant's db-claimant
    check — step-12 GREEN must not double-fetch)."""

    async def test_db_status_passes_through_from_task_row(self) -> None:
        task = {'status': 'blocked', 'claimant_run_id': None, 'heartbeat_at': None}
        scheduler = _fake_scheduler(is_actively_held=False, task=task)
        resolver = _make_ground_truth(scheduler=scheduler)

        report = await resolver.derive_truth('17')

        assert report.db_status == 'blocked'

    async def test_worktree_present_true_when_path_exists(self, tmp_path: Path) -> None:
        resolver = _make_ground_truth(worktree_resolver=lambda tid: tmp_path)

        report = await resolver.derive_truth('18')

        assert report.worktree_present is True

    async def test_worktree_present_false_when_path_missing(self) -> None:
        resolver = _make_ground_truth()

        report = await resolver.derive_truth('19')

        assert report.worktree_present is False

    async def test_open_escalations_maps_pending_rows_to_refs(self) -> None:
        rows = [
            Escalation(
                id='esc-20-1', task_id='20', agent_role='implementer',
                severity='blocking', category='scope_violation', summary='s1', level=1,
            ),
            Escalation(
                id='esc-20-2', task_id='20', agent_role='implementer',
                severity='info', category='cleanup_needed', summary='s2', level=0,
            ),
        ]
        escalation_queue = _fake_escalation_queue(rows=rows)
        resolver = _make_ground_truth(escalation_queue=escalation_queue)

        report = await resolver.derive_truth('20')

        assert report.open_escalations == [
            EscalationRef(id='esc-20-1', level=1),
            EscalationRef(id='esc-20-2', level=0),
        ]
        escalation_queue.get_by_task.assert_called_once_with('20', status='pending')

    async def test_open_escalations_empty_when_queue_none(self) -> None:
        resolver = _make_ground_truth()  # escalation_queue defaults to None

        report = await resolver.derive_truth('21')

        assert report.open_escalations == []

    async def test_deploy_phase_from_present_deploy_state_slice(self) -> None:
        task = {
            'status': 'in-progress',
            'claimant_run_id': None,
            'heartbeat_at': None,
            'metadata': {'deploy_state': {'phase': 'ran'}},
        }
        scheduler = _fake_scheduler(is_actively_held=False, task=task)
        resolver = _make_ground_truth(scheduler=scheduler)

        report = await resolver.derive_truth('22')

        assert report.deploy_phase == DeployPhase.RAN

    async def test_deploy_phase_none_when_slice_absent(self) -> None:
        task = {
            'status': 'in-progress', 'claimant_run_id': None, 'heartbeat_at': None,
            'metadata': {},
        }
        scheduler = _fake_scheduler(is_actively_held=False, task=task)
        resolver = _make_ground_truth(scheduler=scheduler)

        report = await resolver.derive_truth('23')

        assert report.deploy_phase is None

    async def test_fetches_task_row_exactly_once(self) -> None:
        task = {'status': 'pending', 'claimant_run_id': None, 'heartbeat_at': None}
        scheduler = _fake_scheduler(is_actively_held=False, task=task)
        resolver = _make_ground_truth(scheduler=scheduler)

        await resolver.derive_truth('24')

        scheduler.get_task.assert_awaited_once_with('24')


# ---------------------------------------------------------------------------
# step-13 — recovery_for(tid): the derive_truth -> classify_recovery seam
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRecoveryFor:
    """``recovery_for(tid)`` composes ``derive_truth`` -> ``classify_recovery``
    — the thin seam θ2's seven sweeps will call. Boundary scenarios from the
    PRD's boundary-test sketch (G1/G2/D1)."""

    async def test_g1_stale_in_progress_journal_on_main_marks_done(
        self, tmp_path: Path,
    ) -> None:
        _bind_landed_row(tmp_path, task_id='30', advanced_sha='g1advancedsha')
        fixed_now = datetime(2026, 7, 12, 12, 0, 0, tzinfo=UTC)
        task = {
            'status': 'in-progress',
            'claimant_run_id': 'run-1/session-1/pid=1',
            'heartbeat_at': '2026-07-12T11:00:00+00:00',  # 60 minutes stale
        }
        git_ops = _fake_git_ops()
        scheduler = _fake_scheduler(is_actively_held=False, task=task)
        resolver = _make_ground_truth(
            git_ops=git_ops, scheduler=scheduler,
            now_fn=lambda: fixed_now, heartbeat_ttl=timedelta(minutes=10),
        )

        report, action = await resolver.recovery_for('30')

        assert report.branch_state == BranchState(BranchStateKind.ON_MAIN, 'g1advancedsha')
        assert report.live_claimant is None
        assert action == RecoveryAction.MARK_DONE_WITH_PROVENANCE
        git_ops.find_merge_marker.assert_not_awaited()

    async def test_g2_journal_miss_git_marker_marks_done(self) -> None:
        fixed_now = datetime(2026, 7, 12, 12, 0, 0, tzinfo=UTC)
        task = {
            'status': 'in-progress',
            'claimant_run_id': 'run-1/session-1/pid=2',
            'heartbeat_at': '2026-07-12T11:00:00+00:00',  # 60 minutes stale
        }
        git_ops = _fake_git_ops(is_ancestor=False, branch_sha=None, marker_sha='g2markersha')
        scheduler = _fake_scheduler(is_actively_held=False, task=task)
        resolver = _make_ground_truth(
            git_ops=git_ops, scheduler=scheduler,
            now_fn=lambda: fixed_now, heartbeat_ttl=timedelta(minutes=10),
        )

        report, action = await resolver.recovery_for('31')

        assert report.branch_state == BranchState(
            BranchStateKind.GONE_WITH_MERGE_MARKER, 'g2markersha',
        )
        assert action == RecoveryAction.MARK_DONE_WITH_PROVENANCE
        git_ops.find_merge_marker.assert_awaited_once_with('task/31')

    async def test_d1_deterministic_deploy_ran_refiles_not_phantom_done(self) -> None:
        task = {
            'status': 'in-progress',
            'claimant_run_id': None,
            'heartbeat_at': None,
            'metadata': {'deploy_state': {'phase': 'ran'}},
        }
        git_ops = _fake_git_ops(is_ancestor=False, branch_sha=None, marker_sha=None)
        scheduler = _fake_scheduler(is_actively_held=False, task=task)
        resolver = _make_ground_truth(git_ops=git_ops, scheduler=scheduler)

        report, action = await resolver.recovery_for('32')

        assert report.branch_state == BranchState(BranchStateKind.GONE_NO_MARKER, None)
        assert report.deploy_phase == DeployPhase.RAN
        assert action == RecoveryAction.RE_FILE_ESCALATION
