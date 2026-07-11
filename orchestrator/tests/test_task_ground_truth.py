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

import pytest
from shared.deploy_state import DeployPhase

from orchestrator.task_ground_truth import (
    BranchState,
    BranchStateKind,
    Claimant,
    ClaimantSource,
    EscalationRef,
    RecoveryAction,
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
