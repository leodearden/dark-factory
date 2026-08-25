"""Tests for task 3539 — CONVERT_TO_BLOCKED + the already-landed merge-gate carve-out.

Task 3539 closes a measured zombie loop at its two ENDS:

  ENTRY — work that already merged gets re-dispatched, its branch is re-seeded
  from post-landing main so ``base..HEAD`` is legitimately EMPTY, and
  ``_submit_to_merge_queue``'s Decision-1 gate reads that empty range and emits
  the exact INVERSE of the truth (``plan_files_not_touched``) through
  ``_mark_blocked(..., escalate_to_human=True)``, whose default
  ``category='task_failure'`` mints the pin that fuels the loop.  The
  already-landed carve-out (:func:`orchestrator.merge_gates.resolve_already_landed_branch`)
  recognises that shape and routes it to an honest terminal outcome on the
  NORMAL ladder, so no ``task_failure`` pin is minted at all.

  EXIT — a pinned, no-claimant, stranded ``in-progress`` row used to classify
  ``RecoveryAction.LEAVE``, so the sweep held silently and the row churned
  forever (measured: 39 consecutive ``recovery_vetoed`` over 10.5h on task
  3717).  Four new ``_RECOVERY`` rows now classify
  ``RecoveryAction.CONVERT_TO_BLOCKED`` instead, so the row comes to rest in
  the honest status for "pinned, awaiting a human".

HONESTY PIN — conversion is NOT completion.  The converted task arrives in
``blocked`` still carrying its pin, and its exit is a human or task 3541,
never an automatic self-heal.  :class:`TestLandedButPinnedZombieLoop` asserts
that directly so a future reader cannot quietly re-acquire the wrong
expectation.

Covers:
  step-1:  the pure ``_RECOVERY`` half — the new action, the four rows, the
           table invariant, anti-churn/idempotence, and the ``leave_reason``
           contract.
  step-3:  the applier in LOG MODE (the shipped default) — no status write,
           disposition and emission byte-identical to pre-3539.
  step-5:  the applier in ENFORCE mode — one ``blocked`` write, no
           ``done_provenance``, no escalation, fail-open, idempotent.
  step-7:  ``resolve_already_landed_branch`` over REAL git fixtures — the
           three required signals, both traps, and fail-closed on git error.
  step-9:  the ``_submit_to_merge_queue`` carve-out — positive, negative and
           ordering cases.
  step-11: the end-to-end landed-but-pinned regression, both halves composed.
"""

from __future__ import annotations

import pytest
from shared.deploy_state import DeployPhase
from shared.task_statuses import TaskStatus

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
# step-1 — the pure _RECOVERY half
# ---------------------------------------------------------------------------

#: Every ``BranchStateKind``, so the four new rows are proved to cover the
#: whole branch-state axis rather than just the ON_MAIN specimen that
#: motivated them.
ALL_BRANCH_KINDS = [
    BranchStateKind.ON_MAIN,
    BranchStateKind.GONE_WITH_MERGE_MARKER,
    BranchStateKind.EXISTS_OFF_MAIN,
    BranchStateKind.GONE_NO_MARKER,
]

#: A sha is carried only for the two variants whose ``BranchState`` docstring
#: invariant requires one.
_SHA_FOR_KIND = {
    BranchStateKind.ON_MAIN: 'sha-on-main',
    BranchStateKind.GONE_WITH_MERGE_MARKER: 'sha-marker',
    BranchStateKind.EXISTS_OFF_MAIN: None,
    BranchStateKind.GONE_NO_MARKER: None,
}

#: One ref per level, so the row is confirmed level-agnostic exactly like the
#: boolean ``_shape`` element it keys on (``has_open_escalation`` folds ANY
#: open record at ANY level).
PIN_REFS = [
    EscalationRef(id='esc-3539-0', level=0, category='task_failure', severity='blocking'),
    EscalationRef(id='esc-3539-1', level=1, category='task_failure', severity='blocking'),
    EscalationRef(id='esc-3539-2', level=2, category='task_failure', severity='critical'),
]


def _branch(kind: BranchStateKind) -> BranchState:
    return BranchState(kind, _SHA_FOR_KIND[kind])


class TestConvertToBlockedTable:
    """``_RECOVERY``'s new escalation-pinned stranded-in-progress rows.

    Modelled on ``TestClassifyRecovery._report`` in
    ``orchestrator/tests/test_task_ground_truth.py`` — the same static builder,
    so the two suites cannot drift on what a shape means.
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

    # (1) the action itself -------------------------------------------------

    def test_convert_to_blocked_is_a_genuine_strenum_member(self) -> None:
        """Matches the other four members: a real ``str`` with a lowercase value."""
        assert RecoveryAction.CONVERT_TO_BLOCKED == 'convert_to_blocked'
        assert RecoveryAction.CONVERT_TO_BLOCKED.value == 'convert_to_blocked'
        assert isinstance(RecoveryAction.CONVERT_TO_BLOCKED, str)

    # (2) the four rows -----------------------------------------------------

    @pytest.mark.parametrize('kind', ALL_BRANCH_KINDS)
    @pytest.mark.parametrize('ref', PIN_REFS, ids=lambda r: f'L{r.level}')
    def test_pinned_stranded_in_progress_converts_for_every_branch_kind(
        self, kind: BranchStateKind, ref: EscalationRef,
    ) -> None:
        """in-progress + no claimant + pinned -> CONVERT_TO_BLOCKED, any branch, any level.

        Before 3539 only the ON_MAIN shape was even in the table (row (f), as
        LEAVE); the other three fell through to the LEAVE default.  All four
        churn identically, so all four convert.
        """
        report = self._report(branch_state=_branch(kind), open_escalations=[ref])
        assert classify_recovery(report) == RecoveryAction.CONVERT_TO_BLOCKED

    def test_an_unpinned_stranded_in_progress_row_is_untouched(self) -> None:
        """The pin is the whole trigger — without one, rows (a)/(c) still apply."""
        assert classify_recovery(
            self._report(branch_state=_branch(BranchStateKind.ON_MAIN)),
        ) == RecoveryAction.MARK_DONE_WITH_PROVENANCE
        assert classify_recovery(
            self._report(branch_state=_branch(BranchStateKind.EXISTS_OFF_MAIN)),
        ) == RecoveryAction.REVERT_TO_PENDING

    @pytest.mark.parametrize('kind', ALL_BRANCH_KINDS)
    def test_a_live_claimant_still_outranks_the_new_rows(self, kind: BranchStateKind) -> None:
        """A pinned-AND-held task is simply RUNNING — there is nothing to convert."""
        report = self._report(
            branch_state=_branch(kind),
            live_claimant=Claimant(
                run_id='r', heartbeat_at=None, source=ClaimantSource.IN_MEMORY,
            ),
            open_escalations=[PIN_REFS[1]],
        )
        assert classify_recovery(report) == RecoveryAction.LEAVE

    # (3) the table invariant -----------------------------------------------

    def test_every_convert_row_is_keyed_in_progress_no_claimant_pinned(self) -> None:
        """The invariant that makes conversion one-shot AND keeps 3539 in its lane.

        Keyed ``IN_PROGRESS`` by construction:
          * the applier may assume the pin is present (it logs the pinning ids);
          * a converted (``blocked``) row can never match a CONVERT row again,
            which is what makes conversion structurally idempotent; and
          * no CONVERT row may ever be keyed on ``pending`` / ``merge-deferred``
            — that population is task 4651's, per the 2026-08-24 ownership
            ruling (gate task 4673 / esc-4673-1).
        """
        from orchestrator.task_ground_truth import _RECOVERY

        convert_keys = [
            key for key, action in _RECOVERY.items()
            if action == RecoveryAction.CONVERT_TO_BLOCKED
        ]
        assert len(convert_keys) == len(ALL_BRANCH_KINDS), (
            'expected exactly one CONVERT row per BranchStateKind'
        )
        assert {key[2] for key in convert_keys} == set(ALL_BRANCH_KINDS)
        for db_status, live_claimant_present, _kind, has_open_escalation, deploy_phase in convert_keys:
            assert db_status == TaskStatus.IN_PROGRESS
            assert live_claimant_present is False
            assert has_open_escalation is True
            assert deploy_phase is None

    # (4) anti-churn / no oscillation ---------------------------------------

    @pytest.mark.parametrize('kind', ALL_BRANCH_KINDS)
    def test_the_post_conversion_shape_comes_to_rest(self, kind: BranchStateKind) -> None:
        """A converted row classifies LEAVE, so the sweep cannot flip it back.

        This is the whole anti-churn argument, asserted rather than asserted-in-
        prose: no persisted counter is needed because the post-conversion shape
        matches no ``_RECOVERY`` row at all.
        """
        report = self._report(
            db_status='blocked',
            branch_state=_branch(kind),
            open_escalations=[PIN_REFS[1]],
        )
        assert classify_recovery(report) == RecoveryAction.LEAVE

    # (5) idempotence --------------------------------------------------------

    @pytest.mark.parametrize('kind', ALL_BRANCH_KINDS)
    def test_conversion_is_structurally_one_shot(self, kind: BranchStateKind) -> None:
        report = self._report(
            db_status='blocked',
            branch_state=_branch(kind),
            open_escalations=[PIN_REFS[1]],
        )
        assert classify_recovery(report) != RecoveryAction.CONVERT_TO_BLOCKED

    # (6) the leave_reason contract -----------------------------------------

    @pytest.mark.parametrize('kind', ALL_BRANCH_KINDS)
    def test_leave_reason_is_none_for_every_converting_report(
        self, kind: BranchStateKind,
    ) -> None:
        """"Return None whenever classify_recovery did NOT return LEAVE."

        The contract is preserved UNCHANGED by 3539: a caller must never be
        able to mislabel an action it actually took as a hold.  The log-mode
        applier therefore threads ``LeaveReason.escalation_pinned`` explicitly
        instead of relaxing this (see the design decision, and step 3).
        """
        from orchestrator.task_ground_truth import leave_reason

        report = self._report(branch_state=_branch(kind), open_escalations=[PIN_REFS[1]])
        assert classify_recovery(report) == RecoveryAction.CONVERT_TO_BLOCKED, 'fixture drift'
        assert leave_reason(report) is None

    def test_leave_reason_still_reports_escalation_pinned_for_a_genuine_hold(self) -> None:
        """The ``escalation_pinned`` precedence link survives 3539 intact."""
        from orchestrator.task_ground_truth import LeaveReason, leave_reason

        report = self._report(
            db_status='blocked',
            branch_state=_branch(BranchStateKind.ON_MAIN),
            open_escalations=[PIN_REFS[1]],
        )
        assert classify_recovery(report) == RecoveryAction.LEAVE, 'fixture drift'
        assert leave_reason(report) == LeaveReason.escalation_pinned

    # (7) recovery_shape_str is action-independent ---------------------------

    @pytest.mark.parametrize('kind', ALL_BRANCH_KINDS)
    def test_recovery_shape_str_still_renders_the_table_key(
        self, kind: BranchStateKind,
    ) -> None:
        """It renders the SHAPE, not the action — so conversion changes nothing here."""
        from orchestrator.recovery_emission import render_shape
        from orchestrator.task_ground_truth import _shape, recovery_shape_str

        report = self._report(branch_state=_branch(kind), open_escalations=[PIN_REFS[1]])
        assert recovery_shape_str(report) == render_shape(*_shape(report))
