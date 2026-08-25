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


# ---------------------------------------------------------------------------
# step-3 — the applier in LOG MODE (the shipped default)
#
# A recovery row that has NEVER written a status before must be OBSERVED
# before it is enforced, so `convert_to_blocked_enforce` ships False and the
# applier merely LOGS the conversion it would perform.  The contract of log
# mode is therefore ZERO observable change from pre-3539: the same (absent)
# writes, the same return value, and — the part that is not free — the same
# `recovery_vetoed` row, because `leave_reason` now returns None for a
# converting report and an unthreaded chokepoint would silently drop the very
# veto stream an operator is supposed to count conversions in.
# ---------------------------------------------------------------------------

_TID = 'T3539'

#: The measured task-3717 signature: in-progress, no claimant, on main,
#: pinned, no deploy phase.  This is the row that churned 39 consecutive
#: `recovery_vetoed` emissions over 10.5h.
_SHAPE_3717 = 'in-progress|false|on_main|true|-'


class _StubGroundTruth:
    """Injects the FACTS; ``classify_recovery`` stays the real one.

    Same construction as ``_StubGroundTruth`` in
    ``orchestrator/tests/test_recovery_emission_wiring.py`` — so a disposition
    asserted here is the one ``_RECOVERY`` actually produces, never a mocked
    stand-in that could keep agreeing after the table moved.
    """

    def __init__(self, report: TruthReport):
        self.escalation_queue = None
        self.report = report

    async def recovery_for(self, tid: str):
        return self.report, classify_recovery(self.report)


def _pinned_in_progress(
    kind: BranchStateKind = BranchStateKind.ON_MAIN,
    *,
    refs: list[EscalationRef] | None = None,
) -> TruthReport:
    """The converting shape: in-progress, unclaimed, pinned, any branch."""
    return TruthReport(
        db_status=TaskStatus.IN_PROGRESS,
        live_claimant=None,
        branch_state=_branch(kind),
        worktree_present=True,
        open_escalations=list(refs or [PIN_REFS[1]]),
        deploy_phase=None,
    )


@pytest.fixture
def applier_harness(tmp_path, mock_orch_config):
    """A Harness wired for the reconcile sweep, with REAL emission collaborators.

    Fixture pattern from ``orchestrator/tests/test_reconcile_stranded.py``
    (patch the three constructor collaborators, then replace ``h.scheduler``
    wholesale); the real ``EventStore`` / ``EscalationQueue`` on ``tmp_path``
    come from ``test_recovery_emission_wiring.py``, because this suite asserts
    on what the emission path actually WROTE, not on a spy's call args.
    """
    import shutil
    from unittest.mock import AsyncMock, MagicMock, patch

    from _orch_helpers import wire_scheduler_liveness_mock
    from escalation.queue import EscalationQueue

    from orchestrator.config import RecoveryEmissionConfig
    from orchestrator.event_store import EventStore
    from orchestrator.harness import Harness

    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    wire_scheduler_liveness_mock(h.scheduler)
    h.scheduler.get_tasks = AsyncMock(return_value=[])
    h.scheduler.get_statuses = AsyncMock(
        return_value=({_TID: str(TaskStatus.IN_PROGRESS)}, None),
    )
    h.scheduler.get_task = AsyncMock(
        return_value={'status': str(TaskStatus.IN_PROGRESS), 'metadata': {}},
    )
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.mark_done = AsyncMock()

    h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()
    h.git_ops.cleanup_worktree = AsyncMock(
        side_effect=lambda path, tid: shutil.rmtree(path, ignore_errors=True),
    )

    h.event_store = EventStore(tmp_path / 'runs.db', 'run-test')
    h._escalation_queue = EscalationQueue(tmp_path / 'escalations')

    # A REAL RecoveryEmissionConfig: mock_orch_config's spec'd MagicMock would
    # hand the streak predicate a MagicMock threshold, which the emitter's
    # fail-open guard swallows — hiding whatever this suite meant to assert.
    h.config.recovery_emission = RecoveryEmissionConfig()
    h.config.stranded_blocked_escalate_enabled = False
    # THE flag under test, pinned at its shipped default.  Set explicitly
    # (never left to the fixture) because ``mock_orch_config`` is a spec'd
    # MagicMock: an unset bool field reads back as a truthy MagicMock, which
    # would silently run every test in this class in ENFORCE mode.  Same
    # precedent as ``stranded_blocked_escalate_enabled`` above.
    h.config.convert_to_blocked_enforce = False

    return h


def _bind(harness, report: TruthReport) -> None:
    harness._get_ground_truth = lambda: _StubGroundTruth(report)


def _recovery_rows(harness) -> list[dict]:
    """Every recovery event row this harness wrote, oldest first."""
    import json
    import sqlite3

    from orchestrator.event_store import EventType

    conn = sqlite3.connect(str(harness.event_store.db_path))
    try:
        wanted = (EventType.recovery_vetoed.value, EventType.recovery_left.value)
        return [
            {'task_id': tid, 'event_type': et, 'data': json.loads(raw or '{}')}
            for tid, et, raw in conn.execute(
                'SELECT task_id, event_type, data FROM events ORDER BY id',
            )
            if et in wanted
        ]
    finally:
        conn.close()


class TestConvertToBlockedEnforceFlag:
    """The flag itself, read off the REAL config model.

    Separate (synchronous) class: every other assertion in this section rides
    the spec'd-MagicMock ``mock_orch_config``, which can only ever report the
    value a fixture put there — so the shipped DEFAULT has to be asserted
    against ``OrchestratorConfig`` itself or it is not asserted at all.
    """

    def test_the_flag_ships_observe_only(self) -> None:
        """Observe-before-enforce is the SHIPPED posture, not a test-only one."""
        from orchestrator.config import OrchestratorConfig

        assert OrchestratorConfig().convert_to_blocked_enforce is False


@pytest.mark.asyncio
class TestConvertToBlockedApplierLogMode:
    """``convert_to_blocked_enforce=False`` — observe, never write."""

    # (1) zero disposition change ------------------------------------------

    async def test_no_status_write_of_any_kind(self, applier_harness) -> None:
        """The load-bearing assertion: log mode may not move a single row."""
        _bind(applier_harness, _pinned_in_progress())

        await applier_harness._reconcile_one_stranded(
            _TID, 'in-progress', mid_run=False,
        )

        assert applier_harness.scheduler.set_task_status.await_count == 0
        assert applier_harness.scheduler.mark_done.await_count == 0

    @pytest.mark.parametrize('kind', ALL_BRANCH_KINDS)
    async def test_no_blocked_write_for_any_branch_kind(
        self, applier_harness, kind: BranchStateKind,
    ) -> None:
        """The negative that proves log mode is a real gate, not a no-op.

        All four branch shapes convert, so all four must be held back by the
        flag — a gate that only covered the ON_MAIN specimen would let the
        other three write while claiming to be observe-only.
        """
        _bind(applier_harness, _pinned_in_progress(kind))

        await applier_harness._reconcile_one_stranded(
            _TID, 'in-progress', mid_run=False,
        )

        assert applier_harness.scheduler.set_task_status.await_count == 0

    async def test_the_reconciler_returns_none(self, applier_harness) -> None:
        """Exactly what the pinned in-progress LEAVE tail returns today — the
        driver counts this int, so a converting task must not read as busy."""
        _bind(applier_harness, _pinned_in_progress())

        result = await applier_harness._reconcile_one_stranded(
            _TID, 'in-progress', mid_run=False,
        )

        assert result is None

    # (2) the conversion is COUNTABLE from the journal alone ----------------

    async def test_an_info_line_names_the_conversion_it_would_perform(
        self, applier_harness, caplog,
    ) -> None:
        """The whole point of observe-before-enforce: an operator must be able
        to count intended conversions BEFORE enabling them, from the journal,
        with no event-store query."""
        import logging

        from orchestrator.task_ground_truth import recovery_shape_str

        report = _pinned_in_progress(refs=[PIN_REFS[0], PIN_REFS[2]])
        _bind(applier_harness, report)

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
            await applier_harness._reconcile_one_stranded(
                _TID, 'in-progress', mid_run=False,
            )

        lines = [
            r.getMessage() for r in caplog.records
            if r.levelno >= logging.INFO and 'convert_to_blocked' in r.getMessage()
        ]
        assert len(lines) == 1, f'expected exactly one conversion line, got {lines}'
        line = lines[0]
        assert _TID in line, 'which task would convert'
        assert recovery_shape_str(report) in line, 'which SHAPE classified it'
        assert 'esc-3539-0' in line and 'esc-3539-2' in line, (
            'which records pinned it — the ids an operator triages'
        )

    @pytest.mark.parametrize('kind', ALL_BRANCH_KINDS)
    async def test_the_line_names_the_branch_kind(
        self, applier_harness, caplog, kind: BranchStateKind,
    ) -> None:
        """Two of the four shapes carry landing evidence and two do not; an
        operator sizing the conversion population must be able to tell them
        apart without re-deriving the branch state."""
        import logging

        _bind(applier_harness, _pinned_in_progress(kind))

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
            await applier_harness._reconcile_one_stranded(
                _TID, 'in-progress', mid_run=False,
            )

        line = next(
            r.getMessage() for r in caplog.records
            if 'convert_to_blocked' in r.getMessage()
        )
        assert kind.value in line

    # (3) the veto stream must NOT go dark ----------------------------------

    async def test_the_recovery_vetoed_row_still_fires(
        self, applier_harness,
    ) -> None:
        """``leave_reason`` returns None for a converting report (its contract
        is preserved unchanged), so the chokepoint must be handed
        ``LeaveReason.escalation_pinned`` EXPLICITLY.  Without that thread, log
        mode would silently delete the veto stream it exists to be measured in.
        """
        from orchestrator.event_store import EventType

        _bind(applier_harness, _pinned_in_progress())

        await applier_harness._reconcile_one_stranded(
            _TID, 'in-progress', mid_run=False,
        )

        rows = _recovery_rows(applier_harness)
        assert len(rows) == 1, f'expected exactly one recovery row, got {rows}'
        assert rows[0]['event_type'] == EventType.recovery_vetoed.value
        assert rows[0]['task_id'] == _TID
        assert rows[0]['data']['reason'] == 'escalation_pinned'

    async def test_the_vetoed_row_carries_the_unchanged_3717_shape(
        self, applier_harness,
    ) -> None:
        """Byte-identical to what this task emits today — the signature an
        operator already greps for must not move under them."""
        _bind(applier_harness, _pinned_in_progress())

        await applier_harness._reconcile_one_stranded(
            _TID, 'in-progress', mid_run=False,
        )

        assert _recovery_rows(applier_harness)[0]['data']['shape'] == _SHAPE_3717

    async def test_the_vetoed_row_still_names_the_pinning_ids(
        self, applier_harness,
    ) -> None:
        _bind(applier_harness, _pinned_in_progress(refs=[PIN_REFS[0], PIN_REFS[2]]))

        await applier_harness._reconcile_one_stranded(
            _TID, 'in-progress', mid_run=False,
        )

        buckets = _recovery_rows(applier_harness)[0]['data']['escalation_ids']
        assert sorted(i for ids in buckets.values() for i in ids) == [
            'esc-3539-0', 'esc-3539-2',
        ]

    async def test_one_pass_still_emits_exactly_once(
        self, applier_harness,
    ) -> None:
        """The chokepoint and the open-escalation early-return both see this
        task in the SAME pass; emitting at both would double every row."""
        _bind(applier_harness, _pinned_in_progress())

        await applier_harness._reconcile_one_stranded(
            _TID, 'in-progress', mid_run=False,
        )

        assert len(_recovery_rows(applier_harness)) == 1

    # (4) the 3535 alarm may not be silently disarmed -----------------------

    async def test_the_veto_streak_is_charged_exactly_as_before(
        self, applier_harness,
    ) -> None:
        """Log mode must not disarm task 3535's alarm.

        If a converting task stopped charging the streak, a fleet left in log
        mode for a week would look like it had no held strands at all — the
        exact blindness 3535 was built to end.
        """
        _bind(applier_harness, _pinned_in_progress())

        for _ in range(3):
            await applier_harness._reconcile_one_stranded(
                _TID, 'in-progress', mid_run=False,
            )

        assert applier_harness._recovery_veto_tracker.streak(
            'reconcile_sweep', _TID,
        ) == 3

    async def test_the_sweep_tally_counts_it_as_held(
        self, applier_harness,
    ) -> None:
        """The per-sweep summary line is the OTHER half of 3535's cadence, and
        it reads the tally rather than the event rows."""
        from orchestrator.recovery_emission import RecoverySweepTally

        tally = RecoverySweepTally()
        _bind(applier_harness, _pinned_in_progress(refs=[PIN_REFS[1]]))

        await applier_harness._reconcile_one_stranded(
            _TID, 'in-progress', mid_run=False, tally=tally,
        )

        assert tally.held == 1
        assert tally.pinning_ids == ['esc-3539-1']
        assert tally.left == {}
        assert tally.observed_task_ids == {_TID}
