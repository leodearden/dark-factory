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


# ---------------------------------------------------------------------------
# step-5 — the applier in ENFORCE mode
#
# The promotion.  Everything log mode deliberately withheld now happens, and
# nothing else does: ONE `blocked` write, no `done_provenance`, no escalation,
# no event row (the sweep ACTED — it did not hold), fail-open on a rejected
# write, and structurally one-shot.
# ---------------------------------------------------------------------------

#: The marker `_reconcile_one_stranded` returns for a conversion, distinct from
#: 'marked_done' / 'reverted' / 'stale_conflict' so the sweep summary can count
#: conversions on their own.
_CONVERTED = 'converted_to_blocked'


@pytest.fixture
def enforce_harness(applier_harness):
    """The step-3 harness with the observe-before-enforce gate flipped."""
    applier_harness.config.convert_to_blocked_enforce = True
    return applier_harness


@pytest.mark.asyncio
class TestConvertToBlockedApplierEnforce:
    """``convert_to_blocked_enforce=True`` — write `blocked`, and nothing else."""

    # (1) exactly one write, and it is not a completion ---------------------

    @pytest.mark.parametrize('kind', ALL_BRANCH_KINDS)
    async def test_writes_blocked_exactly_once(
        self, enforce_harness, kind: BranchStateKind,
    ) -> None:
        _bind(enforce_harness, _pinned_in_progress(kind))

        await enforce_harness._reconcile_one_stranded(
            _TID, 'in-progress', mid_run=False,
        )

        enforce_harness.scheduler.set_task_status.assert_awaited_once_with(
            _TID, 'blocked',
        )

    @pytest.mark.parametrize('kind', ALL_BRANCH_KINDS)
    async def test_never_marks_done_and_carries_no_provenance(
        self, enforce_harness, kind: BranchStateKind,
    ) -> None:
        """Refusing to mark done IS the veto the pin exists to enforce.

        Two of the four converting shapes (ON_MAIN, GONE_WITH_MERGE_MARKER)
        carry landing evidence that would justify MARK_DONE_WITH_PROVENANCE
        without the pin — so this is the assertion that keeps 3539 from
        quietly becoming a phantom-completion path.
        """
        _bind(enforce_harness, _pinned_in_progress(kind))

        await enforce_harness._reconcile_one_stranded(
            _TID, 'in-progress', mid_run=False,
        )

        assert enforce_harness.scheduler.mark_done.await_count == 0
        call = enforce_harness.scheduler.set_task_status.await_args
        assert 'done_provenance' not in call.kwargs
        assert 'done' not in call.args

    # (2) the sweep can count conversions separately ------------------------

    @pytest.mark.parametrize('kind', ALL_BRANCH_KINDS)
    async def test_returns_its_own_marker(
        self, enforce_harness, kind: BranchStateKind,
    ) -> None:
        """A conversion is not a 'marked_done' and not a 'reverted'; folding it
        into either would make the per-sweep summary lie about what moved."""
        _bind(enforce_harness, _pinned_in_progress(kind))

        result = await enforce_harness._reconcile_one_stranded(
            _TID, 'in-progress', mid_run=False,
        )

        assert result == _CONVERTED

    # (3) a status change must explain itself -------------------------------

    async def test_a_warning_names_the_task_the_shape_and_the_pins(
        self, enforce_harness, caplog,
    ) -> None:
        """A status change with no explanation is the silent fail-soft this
        repo forbids — and this one moves a row an escalation is holding."""
        import logging

        from orchestrator.task_ground_truth import recovery_shape_str

        report = _pinned_in_progress(refs=[PIN_REFS[0], PIN_REFS[2]])
        _bind(enforce_harness, report)

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
            await enforce_harness._reconcile_one_stranded(
                _TID, 'in-progress', mid_run=False,
            )

        lines = [
            r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING
        ]
        assert len(lines) == 1, f'expected exactly one WARNING, got {lines}'
        line = lines[0]
        assert _TID in line
        assert recovery_shape_str(report) in line
        assert 'esc-3539-0' in line and 'esc-3539-2' in line

    async def test_the_log_mode_line_is_gone_once_enforcing(
        self, enforce_harness, caplog,
    ) -> None:
        """The would-convert line and the did-convert line are different
        statements; emitting both would double-count the population an
        operator sized in log mode."""
        import logging

        _bind(enforce_harness, _pinned_in_progress())

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
            await enforce_harness._reconcile_one_stranded(
                _TID, 'in-progress', mid_run=False,
            )

        assert [
            r.getMessage() for r in caplog.records
            if 'log mode' in r.getMessage()
        ] == []

    # (4) an ACTION is not a hold -------------------------------------------

    @pytest.mark.parametrize('kind', ALL_BRANCH_KINDS)
    async def test_no_recovery_event_row_is_emitted(
        self, enforce_harness, kind: BranchStateKind,
    ) -> None:
        """``leave_reason`` returns None for a non-LEAVE disposition precisely
        so a site can never mislabel an action as a hold.  Enforce mode ACTS,
        so the veto stream must fall silent for this task — that silence is
        exactly how an operator sees the promotion take effect."""
        _bind(enforce_harness, _pinned_in_progress(kind))

        await enforce_harness._reconcile_one_stranded(
            _TID, 'in-progress', mid_run=False,
        )

        assert _recovery_rows(enforce_harness) == []

    async def test_the_sweep_tally_does_not_count_it_as_held(
        self, enforce_harness,
    ) -> None:
        from orchestrator.recovery_emission import RecoverySweepTally

        tally = RecoverySweepTally()
        _bind(enforce_harness, _pinned_in_progress())

        await enforce_harness._reconcile_one_stranded(
            _TID, 'in-progress', mid_run=False, tally=tally,
        )

        assert tally.held == 0
        assert tally.left == {}

    # (5) fail-open ----------------------------------------------------------

    async def test_a_rejected_write_never_aborts_the_sweep(
        self, enforce_harness, caplog,
    ) -> None:
        """One bad write must not take the whole pass down with it — the same
        try/except discipline as every existing harness blocked-write
        (`_block_and_escalate_delivered_check` / `_external_dep` /
        `_cross_repo`)."""
        import logging

        from orchestrator.scheduler import SetTaskStatusRejected

        enforce_harness.scheduler.set_task_status.side_effect = (
            SetTaskStatusRejected(_TID, 'rejected', 'nope')
        )
        _bind(enforce_harness, _pinned_in_progress())

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
            result = await enforce_harness._reconcile_one_stranded(
                _TID, 'in-progress', mid_run=False,
            )

        assert result is None, 'a failed conversion is not a conversion'
        assert any(
            r.levelno >= logging.WARNING and _TID in r.getMessage()
            for r in caplog.records
        ), 'the failure must be named, not swallowed'

    async def test_an_arbitrary_write_error_is_also_contained(
        self, enforce_harness,
    ) -> None:
        """Fail-open covers the backend hiccup, not just the modelled refusal."""
        enforce_harness.scheduler.set_task_status.side_effect = RuntimeError('boom')
        _bind(enforce_harness, _pinned_in_progress())

        assert await enforce_harness._reconcile_one_stranded(
            _TID, 'in-progress', mid_run=False,
        ) is None

    # (6) no second record ---------------------------------------------------

    @pytest.mark.parametrize('kind', ALL_BRANCH_KINDS)
    async def test_the_conversion_files_no_escalation(
        self, enforce_harness, kind: BranchStateKind,
    ) -> None:
        """The task is ALREADY pinned.  Filing a second record is the
        duplicate/competing-escalation hazard rows (g)/(h) are written to
        avoid — and it would deepen the very hold this conversion is resting."""
        from unittest.mock import MagicMock

        enforce_harness._escalation_queue = MagicMock(
            wraps=enforce_harness._escalation_queue,
        )
        _bind(enforce_harness, _pinned_in_progress(kind))

        await enforce_harness._reconcile_one_stranded(
            _TID, 'in-progress', mid_run=False,
        )

        assert enforce_harness._escalation_queue.submit.call_count == 0

    # (7) idempotence end to end --------------------------------------------

    @pytest.mark.parametrize('kind', ALL_BRANCH_KINDS)
    async def test_enforce_mode_cannot_churn(
        self, enforce_harness, kind: BranchStateKind,
    ) -> None:
        """The anti-churn argument, end to end rather than table-only.

        The converted row is still pinned, and a pinned `blocked` row matches
        no ``_RECOVERY`` row at all — so the next sweep classifies LEAVE and
        the status stays put.  That is what makes conversion structurally
        one-shot with no persisted counter.
        """
        _bind(enforce_harness, _pinned_in_progress(kind))
        assert await enforce_harness._reconcile_one_stranded(
            _TID, 'in-progress', mid_run=False,
        ) == _CONVERTED

        # The very next sweep observes the row it just wrote.
        _bind(enforce_harness, TruthReport(
            db_status=TaskStatus.BLOCKED,
            live_claimant=None,
            branch_state=_branch(kind),
            worktree_present=True,
            open_escalations=[PIN_REFS[1]],
            deploy_phase=None,
        ))
        result = await enforce_harness._reconcile_one_stranded(
            _TID, 'blocked', mid_run=False,
        )

        assert result is None
        assert enforce_harness.scheduler.set_task_status.await_count == 1, (
            'a second write means the sweep is oscillating on its own output'
        )


# ---------------------------------------------------------------------------
# step-7 — `resolve_already_landed_branch` over REAL git fixtures
#
# This is the ENTRY half of the loop.  The gate that mints the pin is a
# *history* gate, so testing it against mocked git would only prove that the
# mock agrees with itself; every case below therefore stages a real repository
# and lets real git answer.  The fixture triple and the `_RunSpy` delegating
# wrapper are modelled on `test_merge_gates_plan_files_rename.py`, which
# covers a sibling false-positive class of the SAME gate — per-file
# duplication rather than promotion to conftest.py is the established
# convention across ~60 test files here.
#
# `_RunSpy` is the one deviation from pure real-git, used for exactly the
# property real git will not produce on demand: a non-zero rc from a chosen
# subcommand, which is what proves the predicate fails CLOSED rather than
# carving out on evidence it never measured.
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
from collections.abc import Callable, Sequence  # noqa: E402
from pathlib import Path  # noqa: E402

from orchestrator.config import GitConfig  # noqa: E402
from orchestrator.git_ops import GitOps, _run  # noqa: E402


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A real temporary git repository with an initial commit on `main`."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        # tmp repo, no real remote — disabling the push keeps the tests quiet.
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


async def _head_of(repo: Path) -> str:
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0
    return out.strip()


async def _commit_on_main(
    repo: Path, paths_content: dict[str, str], msg: str,
) -> str:
    """Write *paths_content* on main and commit it.  Returns the SHA."""
    for rel, content in paths_content.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    await _run(['git', 'add', '-A'], cwd=repo)
    rc, _, err = await _run(['git', 'commit', '-m', msg], cwd=repo)
    assert rc == 0, f'commit on main failed: {err}'
    return await _head_of(repo)


async def _land_via_merge(
    repo: Path,
    branch: str,
    paths_content: dict[str, str],
    *,
    main: str = 'main',
    subject: str | None = None,
) -> str:
    """Commit *paths_content* on *branch*, then no-ff merge it into *main*.

    The merge subject defaults to the canonical ``Merge <branch> into <main>``
    that ``git_ops._merge_subject`` derives and ``find_merge_marker`` greps
    for — writer and reader share one derivation in production, and this
    helper reproduces that exact string so the probe under test is exercised
    against the real format rather than a paraphrase.  Returns the merge SHA.
    """
    rc, _, err = await _run(['git', 'checkout', '-b', branch], cwd=repo)
    assert rc == 0, err
    for rel, content in paths_content.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    await _run(['git', 'add', '-A'], cwd=repo)
    rc, _, err = await _run(
        ['git', 'commit', '-m', f'impl: work for {branch}'], cwd=repo,
    )
    assert rc == 0, err
    rc, _, err = await _run(['git', 'checkout', main], cwd=repo)
    assert rc == 0, err
    rc, _, err = await _run(
        ['git', 'merge', '--no-ff', '-m', subject or f'Merge {branch} into {main}',
         branch],
        cwd=repo,
    )
    assert rc == 0, err
    return await _head_of(repo)


async def _reseed(repo: Path, branch: str, sha: str) -> None:
    """Park *branch* on *sha* — the re-dispatch rebase's measured end state.

    After a task's work merges, re-cutting ``task/<id>`` from post-landing
    main drops every commit as already-upstream and leaves the ref sitting on
    one of MAIN'S OWN commits at 0 commits ahead of its recorded base.  The
    ref still EXISTS, which is precisely why the attribution probe must be
    called with ``gate_on_existing_ref=False``.
    """
    rc, _, err = await _run(['git', 'branch', '-f', branch, sha], cwd=repo)
    assert rc == 0, err


class _RunSpy:
    """Delegating wrapper around ``merge_gates._run`` that records every call.

    Everything not matched by *fail_when* still runs through the real
    ``_run``, so the repository work stays real; a matched command returns
    ``(128, '', <fatal>)`` without being executed.  Copied in shape from
    ``test_merge_gates_plan_files_rename.py``, which needs the same one
    capability against the same gate.
    """

    def __init__(
        self, fail_when: Callable[[Sequence[str]], bool] | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self._fail_when = fail_when

    async def __call__(
        self, cmd: list[str], cwd: Path | None = None, **kwargs,
    ) -> tuple[int, str, str]:
        self.calls.append(list(cmd))
        if self._fail_when is not None and self._fail_when(cmd):
            return 128, '', 'fatal: injected failure (test fault injection)\n'
        return await _run(cmd, cwd, **kwargs)


def _is_rev_list_count(cmd: Sequence[str]) -> bool:
    """True for the predicate's ``git rev-list --count <base>..<head>`` probe."""
    return list(cmd[:3]) == ['git', 'rev-list', '--count']


async def _resolve(plan_files, base, head, git_ops, *, task_id, branch):
    """Thin call-through to the step-8 predicate.

    The import is LOCAL on purpose.  At step 7 the predicate does not exist
    yet; a module-level import would turn every already-green step-1/3/5 test
    in this file into a collection ERROR, which is a far worse RED signal than
    a failing new class.  Keeping it local means the RED is exactly the eight
    cases below and nothing else.
    """
    from orchestrator.merge_gates import resolve_already_landed_branch

    return await resolve_already_landed_branch(
        plan_files, base, head, git_ops, task_id=task_id, branch=branch,
    )


@pytest.mark.asyncio
class TestResolveAlreadyLandedBranch:
    """The already-landed predicate: THREE independent signals, fail closed.

    The carve-out excuses an empty ``base..HEAD`` range only when the work it
    was supposed to deliver is demonstrably already ON MAIN.  Requiring all
    three of (empty range, attribution, coverage) is what keeps it from
    becoming a blanket amnesty for genuine under-delivery — each negative case
    below removes exactly one signal and asserts the predicate declines.
    """

    # --- vocabulary ---------------------------------------------------------

    async def test_the_reason_prefix_is_distinct_from_the_one_it_replaces(
        self,
    ) -> None:
        """The carve-out needs its OWN prefix, or workflow-side routing that
        keys on ``PLAN_FILES_NOT_TOUCHED_REASON_PREFIX`` would still send this
        shape to a human."""
        from orchestrator.merge_gates import (
            ALREADY_LANDED_REASON_PREFIX,
            CROSS_REPO_DELIVERABLE_REASON_PREFIX,
            PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
        )

        assert ALREADY_LANDED_REASON_PREFIX
        assert not ALREADY_LANDED_REASON_PREFIX.startswith(
            PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
        )
        assert not PLAN_FILES_NOT_TOUCHED_REASON_PREFIX.startswith(
            ALREADY_LANDED_REASON_PREFIX,
        )
        assert ALREADY_LANDED_REASON_PREFIX != CROSS_REPO_DELIVERABLE_REASON_PREFIX

    # --- (1) POSITIVE — merge marker ---------------------------------------

    async def test_a_merge_marker_landing_with_an_empty_range_carves_out(
        self, git_repo: Path, git_ops: GitOps,
    ) -> None:
        """The headline shape, staged end to end.

        The task's declared files merge into main under the canonical subject;
        an unrelated commit then lands; the re-dispatched branch is re-cut
        from that later main tip and is 0 commits ahead of its recorded base.
        Today's gate reads that empty range and emits the exact inverse of the
        truth.  The predicate must read the three real signals instead.

        Both attribution signals are present here (the branch commit's own
        subject cites ``task/900`` too), which makes the ``mechanism``
        assertion a pin on the probe ORDER: the merge marker is the stronger,
        branch-keyed evidence and must win.
        """
        plan_files = ['src/pkg/alpha.py', 'docs/alpha.md']
        merge_sha = await _land_via_merge(
            git_repo, 'task/900',
            {f: f'# {f}\n' for f in plan_files},
        )
        parked = await _commit_on_main(
            git_repo, {'unrelated.md': 'x\n'}, 'chore: unrelated later work',
        )
        await _reseed(git_repo, 'task/900', parked)

        result = await _resolve(
            plan_files, parked, parked, git_ops,
            task_id='900', branch='task/900',
        )

        assert result is not None, 'the landing is on main and measurable'
        assert result.landed_sha == merge_sha
        assert result.mechanism == 'merge_marker'
        assert sorted(result.matched_files) == sorted(plan_files)

    # --- (2) NEGATIVE — the range is not empty ------------------------------

    async def test_a_branch_with_real_commits_is_never_excused(
        self, git_repo: Path, git_ops: GitOps,
    ) -> None:
        """The signal that keeps this from excusing genuine under-delivery.

        The marker EXISTS and the declared files ARE on main — but the branch
        also carries a real commit of its own, so this is not the
        empty-because-already-upstream shape; it is a branch that did work and
        missed the declared files.  That is exactly what
        ``plan_files_not_touched`` is FOR, so the predicate must decline and
        let the unchanged path run.
        """
        plan_files = ['src/pkg/alpha.py']
        await _land_via_merge(
            git_repo, 'task/900', {f: f'# {f}\n' for f in plan_files},
        )
        base = await _head_of(git_repo)
        await _reseed(git_repo, 'task/900', base)

        rc, _, err = await _run(['git', 'checkout', 'task/900'], cwd=git_repo)
        assert rc == 0, err
        head = await _commit_on_main(
            git_repo, {'src/pkg/beta.py': 'x\n'}, 'impl: something else',
        )
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        assert await _resolve(
            plan_files, base, head, git_ops,
            task_id='900', branch='task/900',
        ) is None

    # --- (3) TRAP 1 — the coalesce-train member -----------------------------

    async def test_a_task_citation_covers_a_train_member_with_no_marker(
        self, git_repo: Path, git_ops: GitOps,
    ) -> None:
        """A non-tip coalesce-train member lands with NO per-task marker.

        Live specimen: task 4104 inside train ``coalesce-4181-b2a290cd`` at
        ``d25b24468c`` — the TRAIN gets a merge subject, its members do not.
        The member's own commits are still on main carrying task-id subjects,
        so the citation probe is the required second attribution signal;
        without it every train member below the tip is false-negatived and
        keeps feeding the loop.
        """
        plan_files = ['src/pkg/gamma.py']
        landed = await _commit_on_main(
            git_repo, {f: f'# {f}\n' for f in plan_files},
            'impl(901): deliver the declared files inside a coalesce train',
        )
        await _reseed(git_repo, 'task/901', landed)

        result = await _resolve(
            plan_files, landed, landed, git_ops,
            task_id='901', branch='task/901',
        )

        assert result is not None
        assert result.landed_sha == landed
        assert result.mechanism == 'task_citation'
        assert sorted(result.matched_files) == sorted(plan_files)

    # --- (4) TRAP 2 — the invisible rebase landing --------------------------

    async def test_a_rebase_landing_with_no_provenance_fails_closed(
        self, git_repo: Path, git_ops: GitOps,
    ) -> None:
        """KNOWN LIMITATION, asserted rather than silently accepted.

        Specimen: task 3916, whose work is genuinely on main but reachable
        only by ``git cherry`` patch-id equivalence — no merge commit, no
        task-id citation in any subject.  A grep-based predicate cannot see
        it, so it must fall through to the UNCHANGED ``plan_files_not_touched``
        path rather than guess.  This test exists so the limitation stays
        visible in the suite instead of living only in a docstring.
        """
        plan_files = ['src/pkg/delta.py']
        landed = await _commit_on_main(
            git_repo, {f: f'# {f}\n' for f in plan_files},
            'land the files by rebase, provenance lost',
        )
        await _reseed(git_repo, 'task/903', landed)

        assert await _resolve(
            plan_files, landed, landed, git_ops,
            task_id='903', branch='task/903',
        ) is None, (
            'a landing with no merge marker and no task-id citation is '
            'invisible to a grep-based predicate and must fail CLOSED'
        )

    # --- (5) PARTIAL landing ------------------------------------------------

    async def test_a_partial_earlier_landing_cannot_excuse_a_later_empty_one(
        self, git_repo: Path, git_ops: GitOps,
    ) -> None:
        """Attribution alone is not enough — coverage is the third signal.

        A task that landed a PARTIAL increment, was re-dispatched for the
        rest, and delivered nothing this time has a marker AND an empty range.
        Excusing it on attribution alone would let real non-delivery through,
        so every declared entry must appear in the landing's own touched set.
        """
        plan_files = ['src/pkg/alpha.py', 'src/pkg/never_landed.py']
        await _land_via_merge(
            git_repo, 'task/904', {'src/pkg/alpha.py': '# alpha\n'},
        )
        parked = await _head_of(git_repo)
        await _reseed(git_repo, 'task/904', parked)

        assert await _resolve(
            plan_files, parked, parked, git_ops,
            task_id='904', branch='task/904',
        ) is None

    # --- (6) directory plan entry -------------------------------------------

    async def test_a_declared_directory_is_satisfied_by_a_file_beneath_it(
        self, git_repo: Path, git_ops: GitOps,
    ) -> None:
        """Coverage must use the gate's OWN arm-(b) prefix semantics.

        ``_check_plan_files_touched_in_branch`` accepts a declared directory
        when a touched path sits beneath it.  If the carve-out compared only
        exact paths it would decline for every directory-declaring task and
        leave that whole population in the loop.
        """
        plan_files = ['src/pkg']
        await _land_via_merge(
            git_repo, 'task/905', {'src/pkg/mod.py': '# mod\n'},
        )
        parked = await _head_of(git_repo)
        await _reseed(git_repo, 'task/905', parked)

        result = await _resolve(
            plan_files, parked, parked, git_ops,
            task_id='905', branch='task/905',
        )

        assert result is not None
        assert result.matched_files == ['src/pkg']

    # --- (7) fail closed on a git error -------------------------------------

    async def test_an_unmeasurable_range_never_carves_out(
        self, git_repo: Path, git_ops: GitOps, monkeypatch,
    ) -> None:
        """No carve-out on evidence we did not measure.

        The whole point of the empty-range signal is that it SEPARATES
        already-upstream from genuine non-delivery.  If the probe cannot
        answer, the separation was never made, so the predicate must decline
        exactly as if the range were non-empty.
        """
        plan_files = ['src/pkg/alpha.py']
        await _land_via_merge(
            git_repo, 'task/906', {f: f'# {f}\n' for f in plan_files},
        )
        parked = await _head_of(git_repo)
        await _reseed(git_repo, 'task/906', parked)

        spy = _RunSpy(fail_when=_is_rev_list_count)
        monkeypatch.setattr('orchestrator.merge_gates._run', spy)

        assert await _resolve(
            plan_files, parked, parked, git_ops,
            task_id='906', branch='task/906',
        ) is None
        assert any(_is_rev_list_count(c) for c in spy.calls), (
            'the range probe must actually have been attempted'
        )

    # --- (7b) a raising probe is also failing closed -------------------------

    async def test_a_probe_that_RAISES_is_still_failing_closed(
        self, git_repo: Path, git_ops: GitOps, monkeypatch,
    ) -> None:
        """Returning None is failing closed; RAISING is not.

        The predicate sits inside the already-failing arm of a live merge
        submission.  An exception escaping it would take down the whole
        submission for what is a purely advisory recognition step — strictly
        worse than the bug it is fixing.  And the probes really do raise:
        ``_run`` raises ``WorktreeMissing`` when its cwd is gone, which is
        exactly what a MagicMock ``git_ops.project_root`` produces.
        """
        plan_files = ['src/pkg/alpha.py']
        await _land_via_merge(
            git_repo, 'task/908', {f: f'# {f}\n' for f in plan_files},
        )
        parked = await _head_of(git_repo)
        await _reseed(git_repo, 'task/908', parked)

        async def boom(*a, **k):  # noqa: ARG001
            raise RuntimeError('probe exploded')
        monkeypatch.setattr('orchestrator.merge_gates._run', boom)

        assert await _resolve(
            plan_files, parked, parked, git_ops,
            task_id='908', branch='task/908',
        ) is None

    async def test_a_missing_project_root_is_failing_closed(
        self, git_ops: GitOps, tmp_path: Path,
    ) -> None:
        """The concrete shape that surfaced this: ``_run`` raises
        ``WorktreeMissing`` — not a git rc — when the cwd does not exist."""
        git_ops.project_root = tmp_path / 'does_not_exist'

        assert await _resolve(
            ['a.py'], 'base', 'head', git_ops,
            task_id='909', branch='task/909',
        ) is None

    async def test_cancellation_is_not_swallowed(
        self, git_ops: GitOps, monkeypatch,
    ) -> None:
        """Cancellation is not a git error.  Swallowing it would strand the
        awaiting task in a state the caller believes it left."""
        async def cancel(*a, **k):  # noqa: ARG001
            raise asyncio.CancelledError
        monkeypatch.setattr('orchestrator.merge_gates._run', cancel)

        with pytest.raises(asyncio.CancelledError):
            await _resolve(
                ['a.py'], 'base', 'head', git_ops,
                task_id='910', branch='task/910',
            )

    # --- (8) nothing declared ------------------------------------------------

    async def test_no_declared_plan_files_is_not_an_already_landed_branch(
        self, git_repo: Path, git_ops: GitOps,
    ) -> None:
        """With nothing declared there is nothing to have landed — and the
        gate this carve-out sits inside never fires for an empty
        ``plan_files`` either, so a truthy answer here would be reachable
        only by a future caller wiring it up wrong."""
        parked = await _head_of(git_repo)
        assert await _resolve(
            [], parked, parked, git_ops, task_id='907', branch='task/907',
        ) is None


# ---------------------------------------------------------------------------
# step-9 — the `_submit_to_merge_queue` carve-out
#
# Step 7/8 proved the PREDICATE reads git correctly.  These tests prove the
# gate ACTS on it: that the already-landed shape stops minting the
# `task_failure` pin, which is the single assertion that closes the loop's
# ENTRY.  Modelled member-for-member on `TestSubmitToMergeQueueCrossRepo` in
# test_workflow.py, which covers the sibling carve-out in the same arm.
#
# `_make_workflow` is imported from test_workflow rather than duplicated:
# cross-test-module helper reuse is established here (test_coalesce_
# integration_gate, test_harness_digest_rollup, test_eval_boundary_suite and
# ~6 others do it), and the helper carries a dozen non-obvious MagicMock
# landmine guards that a local copy would silently drift from.
# ---------------------------------------------------------------------------

from unittest.mock import AsyncMock  # noqa: E402

from test_workflow import _make_workflow  # noqa: E402

from orchestrator.merge_gates import PlanFilesTouchedResult  # noqa: E402
from orchestrator.workflow import WorkflowOutcome  # noqa: E402


@pytest.mark.asyncio
class TestSubmitToMergeQueueAlreadyLanded:
    """The already-landed short-circuit inside ``_submit_to_merge_queue``.

    A task whose work already merged gets re-dispatched onto a branch that is
    legitimately EMPTY, and today's Decision-1 gate reads that empty range and
    emits the exact INVERSE of the truth — ``plan_files_not_touched`` through
    ``_mark_blocked(..., escalate_to_human=True)``, whose default
    ``category='task_failure'`` mints the pin that fuels the zombie loop.

    The carve-out must route that shape to the honest
    ``plan_files_already_landed`` terminal outcome on the NORMAL ladder.  The
    load-bearing assertion is ``escalate_to_human is not True``: everything
    else is presentation, but THAT is what stops the pin being minted.
    """

    def _wire(self, wf, monkeypatch):
        """Shared stubs: capture emits, forbid narrowing, stub ``_mark_blocked``.

        The not-touched gate is stubbed to FAIL so that without the carve-out
        the flow deterministically reaches the ``plan_files_not_touched``
        escalation — a clean assertion failure rather than an
        ``await MagicMock`` crash.
        """
        async def fake_run(cmd, **kwargs):  # noqa: ARG001
            return 0, 'fake_head_sha\n', ''
        monkeypatch.setattr('orchestrator.workflow._run', fake_run)

        async def fake_check(*a, **k):  # noqa: ARG001
            return PlanFilesTouchedResult(not_touched=['a.py'])
        monkeypatch.setattr(
            'orchestrator.merge_queue._check_plan_files_touched_in_branch',
            fake_check,
        )

        emits: list = []

        def fake_emit(event_store, task_id, outcome, **kwargs):  # noqa: ARG001
            emits.append(outcome)
        monkeypatch.setattr(
            'orchestrator.merge_queue._emit_merge_attempt', fake_emit,
        )

        narrow = AsyncMock(return_value=False)
        wf._try_narrow_plan = narrow  # type: ignore[method-assign]
        mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        wf._mark_blocked = mark_blocked  # type: ignore[method-assign]
        return emits, narrow, mark_blocked

    @staticmethod
    def _stub_predicate(monkeypatch, result, *, seen: list | None = None):
        """Point the workflow's predicate at *result*, recording its args."""
        async def fake_resolve(plan_files, base_sha, branch_head, git_ops, **kw):
            if seen is not None:
                seen.append((list(plan_files), base_sha, branch_head, kw))
            return result
        monkeypatch.setattr(
            'orchestrator.merge_gates.resolve_already_landed_branch',
            fake_resolve,
        )

    @staticmethod
    def _landed():
        from orchestrator.merge_gates import AlreadyLandedResult

        return AlreadyLandedResult(
            landed_sha='9ab336bd6e' * 4,
            mechanism='merge_marker',
            matched_files=['a.py', 'b.py', 'c.py'],
        )

    # --- POSITIVE -----------------------------------------------------------

    async def test_an_already_landed_branch_mints_no_human_pin(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """THE assertion that closes the loop's ENTRY.

        Everything else here is presentation; ``escalate_to_human is not
        True`` is what stops ``_mark_blocked``'s default
        ``category='task_failure'`` from minting the escalation that pins the
        task, vetoes its recovery, and feeds it back round the loop.
        """
        from orchestrator.merge_gates import ALREADY_LANDED_REASON_PREFIX

        wf = _make_workflow(tmp_path=tmp_path)
        landed = self._landed()
        self._stub_predicate(monkeypatch, landed)
        emits, narrow, mark_blocked = self._wire(wf, monkeypatch)

        outcome = await wf._submit_to_merge_queue('task/2656', pre_rebased=False)

        assert outcome == WorkflowOutcome.BLOCKED
        assert 'plan_files_already_landed' in emits
        assert 'plan_files_not_touched' not in emits, (
            'emitting the inverse of the truth is the bug being fixed'
        )
        # `narrow.assert_not_awaited(), (msg)` would build a TUPLE, silently
        # discarding the message on failure; spelled as a real assert so the
        # rationale actually reaches whoever breaks it.
        assert not narrow.await_count, (
            'already-landed work must not be dragged through a narrowing pass'
        )
        mark_blocked.assert_awaited_once()
        args, kwargs = mark_blocked.call_args
        reason = args[0] if args else kwargs['reason']
        assert reason.startswith(ALREADY_LANDED_REASON_PREFIX)
        assert landed.landed_sha in reason, 'the landing must be citable'
        assert landed.mechanism in reason, (
            'an operator must be able to tell the two landing shapes apart '
            'without re-running the probes'
        )
        assert kwargs.get('category') == 'already_landed'
        assert kwargs.get('suggested_action') == 'verify_landing_and_close'
        assert kwargs.get('escalate_to_human') is not True

    async def test_a_train_member_landing_is_named_by_its_mechanism(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The TRAP-1 shape reaches the same honest outcome.

        A coalesce-train member has no per-task merge marker, so it is
        attributed by citation; the outcome must be identical and the reason
        must SAY which probe answered.
        """
        from orchestrator.merge_gates import AlreadyLandedResult

        wf = _make_workflow(tmp_path=tmp_path)
        self._stub_predicate(monkeypatch, AlreadyLandedResult(
            landed_sha='d25b24468c' * 4,
            mechanism='task_citation',
            matched_files=['a.py', 'b.py', 'c.py'],
        ))
        emits, narrow, mark_blocked = self._wire(wf, monkeypatch)

        await wf._submit_to_merge_queue('task/2656', pre_rebased=False)

        assert 'plan_files_already_landed' in emits
        narrow.assert_not_awaited()
        _args, kwargs = mark_blocked.call_args
        reason = _args[0] if _args else kwargs['reason']
        assert 'task_citation' in reason
        assert kwargs.get('escalate_to_human') is not True

    async def test_the_predicate_is_given_the_branch_and_the_recorded_base(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The predicate cannot measure the range or find the marker without
        both, and a plausible-looking wiring that passes neither would make
        every case fail closed — i.e. silently restore the bug."""
        wf = _make_workflow(tmp_path=tmp_path)
        seen: list = []
        self._stub_predicate(monkeypatch, self._landed(), seen=seen)
        self._wire(wf, monkeypatch)

        await wf._submit_to_merge_queue('task/2656', pre_rebased=False)

        assert len(seen) == 1
        plan_files, base_sha, branch_head, kw = seen[0]
        assert plan_files == ['a.py', 'b.py', 'c.py']
        assert base_sha == 'base_sha', 'the branch\'s RECORDED base, not HEAD'
        assert branch_head == 'fake_head_sha'
        assert kw.get('task_id') == wf.task_id
        # EXACT, not a substring: the merge-marker probe greps for the literal
        # subject `Merge task/2656 into main`, so a double-prefixed
        # `task/task/2656` would fail closed on a marker that is really there
        # — and a substring assertion would wave it through.
        assert kw.get('branch') == 'task/2656'

    # --- NEGATIVE — the unchanged path stays byte-identical ------------------

    async def test_a_genuine_miss_is_still_escalated_to_a_human(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The carve-out must not become a blanket amnesty.

        With the predicate declining, EVERY pre-3539 behaviour must survive:
        the same outcome, the same narrowing attempt, the same reason prefix
        and the same forced human ladder.
        """
        from orchestrator.merge_gates import PLAN_FILES_NOT_TOUCHED_REASON_PREFIX

        wf = _make_workflow(tmp_path=tmp_path)
        self._stub_predicate(monkeypatch, None)
        emits, narrow, mark_blocked = self._wire(wf, monkeypatch)

        outcome = await wf._submit_to_merge_queue('task/2656', pre_rebased=False)

        assert outcome == WorkflowOutcome.BLOCKED
        assert 'plan_files_not_touched' in emits
        assert 'plan_files_already_landed' not in emits
        narrow.assert_awaited_once()
        mark_blocked.assert_awaited_once()
        args, kwargs = mark_blocked.call_args
        reason = args[0] if args else kwargs['reason']
        assert reason.startswith(PLAN_FILES_NOT_TOUCHED_REASON_PREFIX)
        assert kwargs.get('escalate_to_human') is True

    # --- ORDERING — the extra git work stays off the hot path ----------------

    async def test_a_passing_gate_never_consults_the_predicate(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Three git probes per merge submission is real cost.

        The carve-out is sited INSIDE the already-failing ``not_touched`` arm
        precisely so a healthy branch — the overwhelming majority — pays
        nothing for it.
        """
        wf = _make_workflow(tmp_path=tmp_path)
        seen: list = []
        self._stub_predicate(monkeypatch, self._landed(), seen=seen)
        emits, narrow, mark_blocked = self._wire(wf, monkeypatch)

        # Re-stub the gate to PASS, overriding _wire's failing stub.
        async def passing_check(*a, **k):  # noqa: ARG001
            return PlanFilesTouchedResult()
        monkeypatch.setattr(
            'orchestrator.merge_queue._check_plan_files_touched_in_branch',
            passing_check,
        )

        # A healthy branch runs off the end of the gate and into the real
        # enqueue, which this minimal harness cannot service (its queue is a
        # MagicMock).  Rather than pin that incidental crash, stop the flow at
        # the enqueue boundary with a sentinel: reaching it is itself the
        # proof that the gate passed and nothing short-circuited.
        class _ReachedEnqueue(Exception):
            pass

        async def boom(*a, **k):  # noqa: ARG001
            raise _ReachedEnqueue
        monkeypatch.setattr(
            'orchestrator.merge_queue.register_and_enqueue_merge_request', boom,
        )

        with pytest.raises(_ReachedEnqueue):
            await wf._submit_to_merge_queue('task/2656', pre_rebased=False)

        assert seen == [], 'the predicate must not run on a healthy branch'
        assert 'plan_files_already_landed' not in emits
        mark_blocked.assert_not_awaited()
        narrow.assert_not_awaited()

    async def test_the_predicate_runs_before_any_narrowing_pass(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Ordering WITHIN the failing arm.

        Siting the carve-out after ``_try_narrow_plan`` would still reach the
        right outcome, but only after asking the architect to adjudicate work
        that is already on main — the dishonest narrowing pass the cross-repo
        carve-out's docstring names.  Assert the architect is never asked.
        """
        wf = _make_workflow(tmp_path=tmp_path)
        order: list[str] = []

        async def fake_resolve(*a, **k):  # noqa: ARG001
            order.append('predicate')
            return self._landed()
        monkeypatch.setattr(
            'orchestrator.merge_gates.resolve_already_landed_branch',
            fake_resolve,
        )
        _emits, narrow, _mark_blocked = self._wire(wf, monkeypatch)
        narrow.side_effect = lambda *a, **k: order.append('narrow') or False

        await wf._submit_to_merge_queue('task/2656', pre_rebased=False)

        assert order == ['predicate']


# ---------------------------------------------------------------------------
# step-11 — the composed regression
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLandedButPinnedZombieLoop:
    """The measured task-3717/3604 loop, reconstructed at BOTH ends.

    Loop today: work already merged -> task re-dispatched -> branch re-seeded
    from post-landing main so ``base..HEAD`` is legitimately EMPTY -> the
    Decision-1 gate reads that empty range and emits the INVERSE of the truth
    (``plan_files_not_touched``) via ``_mark_blocked(...,
    escalate_to_human=True)``, whose default ``category='task_failure'`` mints
    a pin -> that pin vetoes the recovery that would have marked the task done
    -> the row churns (39 consecutive ``recovery_vetoed`` over 10.5h on 3717)
    and is re-dispatched, repeating.

    The ENTRY half runs against a REAL git repository, and the EXIT half
    against the REAL ``Harness._reconcile_one_stranded``, precisely so an
    integration gap between the two halves surfaces HERE rather than in
    production.

    HONESTY PIN — conversion is NOT completion.  The converted row arrives in
    ``blocked`` STILL CARRYING ITS PIN.  ``MERGE_REMEDIABLE_ESC_CATEGORIES`` is
    ``{'stranded_blocked'}`` and ``_only_merge_remediable`` is an ``all(...)``,
    so a ``task_failure`` pin fails both blocked-arm upgrade clauses, and
    ``_RECOVERY``'s only BLOCKED row keys ``has_open_escalation=False``.
    ``blocked`` + pinned is therefore a TERMINAL RESTING STATE whose exit is a
    human or task 3541 — never an automatic recovery, and never this task.
    """

    _PLAN_FILES = ['src/pkg/alpha.py', 'docs/alpha.md']

    # --- ENTRY half ---------------------------------------------------------

    async def test_entry_the_gate_no_longer_mints_the_pin(
        self, git_repo: Path, git_ops: GitOps, tmp_path: Path, monkeypatch,
    ) -> None:
        """Over a REAL repo in the measured parked shape, the gate must not
        produce the ``escalate_to_human=True`` block whose default
        ``category='task_failure'`` is the loop's only fuel."""
        merge_sha = await _land_via_merge(
            git_repo, 'task/3717',
            {f: f'# {f}\n' for f in self._PLAN_FILES},
        )
        # Parked on an UNRELATED task's merge commit — the measured shape
        # (3604 on 9ab336bd6e, 3717 on ce5b830caf, ...), not merely on its own.
        parked = await _land_via_merge(
            git_repo, 'task/9999', {'unrelated.py': 'x\n'},
        )
        await _reseed(git_repo, 'task/3717', parked)

        wf = _make_workflow(tmp_path=tmp_path / 'wf', task_id='3717')
        wf.plan = {'files': list(self._PLAN_FILES)}
        wf._base_commit = parked
        wf.git_ops = git_ops
        wf.config.project_root = git_repo

        async def fake_run(cmd, **kwargs):  # noqa: ARG001
            return 0, f'{parked}\n', ''
        monkeypatch.setattr('orchestrator.workflow._run', fake_run)

        async def fake_check(*a, **k):  # noqa: ARG001
            # The REAL gate's verdict for this shape: the empty range touches
            # none of the declared files.  Stubbed rather than run so this
            # test pins the CARVE-OUT, not a second copy of the gate (which
            # test_merge_gates_plan_files_rename.py already owns).
            return PlanFilesTouchedResult(not_touched=list(self._PLAN_FILES))
        monkeypatch.setattr(
            'orchestrator.merge_queue._check_plan_files_touched_in_branch',
            fake_check,
        )

        emits: list = []
        monkeypatch.setattr(
            'orchestrator.merge_queue._emit_merge_attempt',
            lambda es, tid, outcome, **k: emits.append(outcome),  # noqa: ARG005
        )
        narrow = AsyncMock(return_value=False)
        wf._try_narrow_plan = narrow  # type: ignore[method-assign]
        mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

        await wf._submit_to_merge_queue('3717', pre_rebased=False)

        _args, kwargs = mark_blocked.call_args
        assert kwargs.get('escalate_to_human') is not True, (
            'this is the loop\'s only fuel — minting it here re-arms the loop'
        )
        assert kwargs.get('category') == 'already_landed'
        assert kwargs.get('category') != 'task_failure'
        assert 'plan_files_already_landed' in emits
        assert 'plan_files_not_touched' not in emits
        narrow.assert_not_awaited()
        # The predicate ran for real against real git and found the real
        # marker — not a stub agreeing with itself.
        reason = _args[0] if _args else kwargs['reason']
        assert merge_sha in reason
        assert 'merge_marker' in reason

    # --- EXIT half ----------------------------------------------------------

    @staticmethod
    def _landed_but_pinned() -> TruthReport:
        """3717's measured signature, pinned by exactly what
        ``_mark_blocked(..., escalate_to_human=True)`` mints."""
        return TruthReport(
            db_status=TaskStatus.IN_PROGRESS,
            live_claimant=None,
            branch_state=_branch(BranchStateKind.ON_MAIN),
            worktree_present=True,
            open_escalations=[PIN_REFS[1]],
            deploy_phase=None,
        )

    async def test_exit_the_row_comes_to_rest_and_cannot_re_arm(
        self, enforce_harness,
    ) -> None:
        """One conversion, then nothing — the churn stops for good."""
        _bind(enforce_harness, self._landed_but_pinned())

        assert await enforce_harness._reconcile_one_stranded(
            _TID, 'in-progress', mid_run=False,
        ) == _CONVERTED
        enforce_harness.scheduler.set_task_status.assert_awaited_once_with(
            _TID, 'blocked',
        )

        # The next sweep observes the row it just wrote.
        converted = TruthReport(
            db_status=TaskStatus.BLOCKED,
            live_claimant=None,
            branch_state=_branch(BranchStateKind.ON_MAIN),
            worktree_present=True,
            open_escalations=[PIN_REFS[1]],
            deploy_phase=None,
        )
        _bind(enforce_harness, converted)

        assert await enforce_harness._reconcile_one_stranded(
            _TID, 'blocked', mid_run=False,
        ) is None
        assert enforce_harness.scheduler.set_task_status.await_count == 1, (
            'a second write means the sweep is oscillating on its own output'
        )
        assert classify_recovery(converted) == RecoveryAction.LEAVE
        # The re-dispatch limb of the loop is concretely the sweep's
        # revert-to-`pending` write (`_revert_in_progress_if_no_live_claimant`).
        # Asserting on that specific write, rather than on some notion of
        # "dispatchability", pins the limb that actually re-armed 3717.
        for call in enforce_harness.scheduler.set_task_status.await_args_list:
            assert 'pending' not in call.args, (
                'reverting to pending is what re-dispatches the row and '
                'closes the loop back on itself'
            )

    async def test_exit_the_converted_row_is_never_marked_done(
        self, enforce_harness,
    ) -> None:
        """THE honesty pin, asserted at its CAUSE and not just its effect.

        A converted row is pinned by a ``task_failure`` escalation, and
        ``_only_merge_remediable`` is an ``all(...)`` over
        ``MERGE_REMEDIABLE_ESC_CATEGORIES == {'stranded_blocked'}``.  That is
        WHY both blocked-arm upgrade clauses decline, and why the row rests
        instead of self-healing.  Pinning the mechanism means a future widening
        of that category set fails HERE, loudly, rather than silently turning
        conversion into completion.
        """
        from orchestrator.harness import Harness

        assert frozenset({'stranded_blocked'}) == (
            Harness.MERGE_REMEDIABLE_ESC_CATEGORIES
        )
        assert Harness._only_merge_remediable([PIN_REFS[1]]) is False

        _bind(enforce_harness, self._landed_but_pinned())
        await enforce_harness._reconcile_one_stranded(
            _TID, 'in-progress', mid_run=False,
        )

        enforce_harness.scheduler.mark_done.assert_not_awaited()
        for call in enforce_harness.scheduler.set_task_status.await_args_list:
            assert 'done' not in call.args, (
                'conversion is NOT completion — the row keeps its pin and its '
                'exit is a human or task 3541, never this sweep'
            )

    # --- the two halves compose ---------------------------------------------

    async def test_with_the_entry_closed_the_exit_is_a_backstop_not_the_path(
        self, enforce_harness,
    ) -> None:
        """What the composition actually buys.

        With the ENTRY closed, this producer never mints the ``task_failure``
        pin at all, so on a healthy fleet the CONVERT arm sees nothing from
        it.  The arm is therefore a BACKSTOP for pins minted by OTHER
        producers — which is exactly why it must be inert on an unpinned row,
        or it would start converting rows nobody is holding.
        """
        unpinned = TruthReport(
            db_status=TaskStatus.IN_PROGRESS,
            live_claimant=None,
            branch_state=_branch(BranchStateKind.ON_MAIN),
            worktree_present=True,
            open_escalations=[],
            deploy_phase=None,
        )
        assert classify_recovery(unpinned) != RecoveryAction.CONVERT_TO_BLOCKED

        _bind(enforce_harness, unpinned)
        assert await enforce_harness._reconcile_one_stranded(
            _TID, 'in-progress', mid_run=False,
        ) != _CONVERTED
        for call in enforce_harness.scheduler.set_task_status.await_args_list:
            assert 'blocked' not in call.args
