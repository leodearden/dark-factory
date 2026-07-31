"""Tests for the startup landed-outbox reconciler (task 2155 / W1 γ).

Covers:
  step-1  (RED)  RC-1 / boundary B2 — crash between fsync and advance: no
                 phantom done, row pruned.
  step-3  (RED)  RC-2 / boundary B3 — crash between advance and done-write:
                 row drives the done-write, then is pruned.
  step-5  (RED)  RC-3 / boundary B4 — already-done at reconcile: prune only,
                 no second done-write.
  step-7  (RED)  Scan robustness — empty outbox, multi-row error isolation,
                 status-unknown fail-safe ('skipped').
  step-9  (RED)  Harness startup wiring — Harness._reconcile_landed_outbox
                 None-guard + delegation.

Crash is simulated via an injected fault point (PRD §9, NOT a real process
kill): the PRODUCER half seeds the outbox through the REAL
``_journal_landed_then_advance`` write-ahead chokepoint (β), a fresh
``LandedOutbox`` re-open simulates the restart, then the RECONCILER half runs
with an injected ``is_ancestor`` verdict and a fake scheduler
(``get_status``/``mark_done`` AsyncMocks) — mirroring the dominant
merge-queue mock-git_ops convention (test_merge_speculation.py /
test_merge_queue_train_attribution.py) with a real ``LandedOutbox`` on
``tmp_path`` so ``consume()``/``lookup()`` exercise the real durable prune.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.queue import EscalationQueue

from orchestrator.delivered_checks import DeliveredChecksBlock
from orchestrator.git_ops import AdvanceOutcome
from orchestrator.harness import Harness
from orchestrator.landed_outbox import LandedOutbox, LandedRow
from orchestrator.merge_queue import (
    _journal_landed_then_advance,
    reconcile_landed_outbox,
    reconcile_landed_row,
    reconcile_landed_task,
)
from orchestrator.provenance_conflict import ProvenanceConflictSink
from orchestrator.scheduler import StaleEvidenceRejection

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _producer_git_ops(advance_result: AdvanceOutcome) -> MagicMock:
    """AsyncMock git_ops for the producer half (only advance_main is used)."""
    git_ops = MagicMock()
    git_ops.advance_main = AsyncMock(return_value=advance_result)
    return git_ops


def _reconciler_git_ops(*, main_sha: str = 'MAIN', is_ancestor_result: bool = True) -> MagicMock:
    """AsyncMock git_ops for the reconciler half (get_main_sha + is_ancestor)."""
    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value=main_sha)
    git_ops.is_ancestor = AsyncMock(return_value=is_ancestor_result)
    return git_ops


def _fake_scheduler(*, get_status_result: str | None = None) -> MagicMock:
    """MagicMock scheduler with AsyncMock get_status/mark_done."""
    scheduler = MagicMock()
    scheduler.get_status = AsyncMock(return_value=get_status_result)
    scheduler.mark_done = AsyncMock()
    return scheduler


# ---------------------------------------------------------------------------
# step-1 — RC-1 / boundary B2: crash between fsync and advance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReconcileLandedOutboxRC1CrashBeforeAdvance:
    """RC-1: advanced_sha never actually landed on main → prune, no done-write."""

    async def test_row_not_landed_is_pruned_without_done_write(self, tmp_path: Path) -> None:
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)

        # Producer half: seed the outbox via the REAL write-ahead chokepoint.
        # advance_main reports a non-'advanced' outcome — the process died
        # before the CAS commit actually happened, but the row was already
        # fsync'd (WA-1: record-then-advance).
        git_ops_producer = _producer_git_ops(AdvanceOutcome('cas_failed'))
        await _journal_landed_then_advance(
            outbox, git_ops_producer,
            task_id='Z', branch_tip_sha='tip', advanced_sha='ADV',
            merge_wt=tmp_path,
        )

        # Restart: re-open a fresh LandedOutbox at the same path.
        reopened = LandedOutbox(path)

        # Reconciler half: is_ancestor('ADV', 'MAIN') is False — advanced_sha
        # genuinely never landed on main.
        git_ops_recon = _reconciler_git_ops(main_sha='MAIN', is_ancestor_result=False)
        scheduler = _fake_scheduler()

        report = await reconcile_landed_outbox(reopened, git_ops_recon, scheduler)

        scheduler.mark_done.assert_not_called()
        assert reopened.lookup('Z') is None, 'RC-1 row must be pruned (consumed)'
        assert report['pruned_not_landed'] == 1
        assert report['marked_done'] == 0


# ---------------------------------------------------------------------------
# step-3 — RC-2 / boundary B3: crash between advance and done-write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReconcileLandedOutboxRC2CrashBeforeDoneWrite:
    """RC-2: advanced_sha landed on main but the task was never marked done."""

    async def test_landed_row_drives_done_write_then_is_pruned(self, tmp_path: Path) -> None:
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)

        # Producer half: advance_main reports 'advanced' — record AND advance
        # both genuinely happened before the crash (which hit before the
        # done-write landed).
        git_ops_producer = _producer_git_ops(AdvanceOutcome('advanced', advanced_sha='ADV'))
        await _journal_landed_then_advance(
            outbox, git_ops_producer,
            task_id='Z', branch_tip_sha='tip', advanced_sha='ADV',
            merge_wt=tmp_path,
        )

        # Restart: re-open a fresh LandedOutbox at the same path.
        reopened = LandedOutbox(path)

        # Reconciler half: is_ancestor('ADV', 'MAIN') is True — advanced_sha
        # genuinely landed; the task's status is not yet 'done'.
        git_ops_recon = _reconciler_git_ops(main_sha='MAIN', is_ancestor_result=True)
        scheduler = _fake_scheduler(get_status_result='in-progress')

        report = await reconcile_landed_outbox(reopened, git_ops_recon, scheduler)

        scheduler.mark_done.assert_called_once_with('Z', kind='merged', sha='ADV')
        assert reopened.lookup('Z') is None, (
            'RC-2 row must be consumed AFTER the done-write'
        )
        assert report['marked_done'] == 1


# ---------------------------------------------------------------------------
# step-5 — RC-3 / boundary B4: already-done at reconcile time
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReconcileLandedOutboxRC3AlreadyDone:
    """RC-3: task already 'done' at reconcile time → prune only, no 2nd done-write."""

    async def test_already_done_row_is_pruned_without_second_done_write(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)

        # The already-done branch is pure reconciler logic — a direct record()
        # is sufficient (no need to drive the real producer chokepoint here).
        outbox.record(LandedRow(
            task_id='Z', branch_tip_sha='tip', advanced_sha='ADV', landed_at=1.0,
        ))

        reopened = LandedOutbox(path)

        git_ops_recon = _reconciler_git_ops(main_sha='MAIN', is_ancestor_result=True)
        scheduler = _fake_scheduler(get_status_result='done')

        report = await reconcile_landed_outbox(reopened, git_ops_recon, scheduler)

        scheduler.mark_done.assert_not_called()
        assert reopened.lookup('Z') is None, 'RC-3 row must be pruned'
        assert report['already_done_pruned'] == 1
        assert report['marked_done'] == 0

    @pytest.mark.parametrize('preserved_status', ['cancelled', 'blocked', 'deferred', 'merge-deferred'])
    async def test_workflow_preserve_status_row_is_pruned_without_being_resurrected(
        self, tmp_path: Path, preserved_status: str,
    ) -> None:
        """A task the steward moved to a WORKFLOW_PRESERVE_STATUSES resting
        state (amendment, reviewer_comprehensive #2) after its merge landed
        but before the done-write must NOT be resurrected to 'done' by this
        reconciler — the row is pruned exactly like the literal-'done' RC-3
        case, and 'merge-deferred' stays owned by the train-merge worker.

        Locks in the generalization of RC-3 from ``status == 'done'`` to
        ``status in WORKFLOW_PRESERVE_STATUSES`` — a landed merge must never
        override a status the workflow itself is forbidden from overwriting.
        """
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)
        outbox.record(LandedRow(
            task_id='Z', branch_tip_sha='tip', advanced_sha='ADV', landed_at=1.0,
        ))

        reopened = LandedOutbox(path)
        git_ops_recon = _reconciler_git_ops(main_sha='MAIN', is_ancestor_result=True)
        scheduler = _fake_scheduler(get_status_result=preserved_status)

        report = await reconcile_landed_outbox(reopened, git_ops_recon, scheduler)

        scheduler.mark_done.assert_not_called()
        assert reopened.lookup('Z') is None, (
            f'{preserved_status!r} row must be pruned, not resurrected to done'
        )
        assert report['already_done_pruned'] == 1
        assert report['marked_done'] == 0


# ---------------------------------------------------------------------------
# step-7 — scan robustness: empty outbox, error isolation, status-unknown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReconcileLandedOutboxRobustness:
    """Empty-outbox no-op, per-row error isolation, and the None-status fail-safe."""

    async def test_empty_outbox_returns_all_zero_report_and_makes_no_scheduler_calls(
        self, tmp_path: Path,
    ) -> None:
        outbox = LandedOutbox(tmp_path / 'landed_outbox.json')
        git_ops_recon = _reconciler_git_ops(main_sha='MAIN', is_ancestor_result=True)
        scheduler = _fake_scheduler()

        report = await reconcile_landed_outbox(outbox, git_ops_recon, scheduler)

        assert report == {
            'pruned_not_landed': 0,
            'marked_done': 0,
            'already_done_pruned': 0,
            'skipped': 0,
            'stale_conflict': 0,
            'errors': 0,
        }
        scheduler.get_status.assert_not_called()
        scheduler.mark_done.assert_not_called()

    async def test_one_bad_row_does_not_abort_the_scan(self, tmp_path: Path) -> None:
        """A single row raising during reconcile must not sink the other rows.

        Mirrors recover_pending_merges' fail-open per-record loop: row 'A' is
        already-done (prune only), row 'BAD' raises inside is_ancestor, row
        'C' is a genuine RC-2 (drives a done-write) — both good rows must
        still reach their correct disposition, and the bad row is tallied
        under 'errors' rather than aborting the whole scan.
        """
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)
        outbox.record(LandedRow(task_id='A', branch_tip_sha='tip', advanced_sha='ADV-A', landed_at=1.0))
        outbox.record(LandedRow(task_id='BAD', branch_tip_sha='tip', advanced_sha='ADV-BAD', landed_at=2.0))
        outbox.record(LandedRow(task_id='C', branch_tip_sha='tip', advanced_sha='ADV-C', landed_at=3.0))

        def _is_ancestor_side_effect(advanced_sha: str, main_sha: str) -> bool:
            if advanced_sha == 'ADV-BAD':
                raise RuntimeError('boom')
            return True

        def _get_status_side_effect(task_id: str) -> str | None:
            return {'A': 'done', 'C': 'in-progress'}.get(task_id)

        git_ops_recon = _reconciler_git_ops(main_sha='MAIN')
        git_ops_recon.is_ancestor = AsyncMock(side_effect=_is_ancestor_side_effect)
        scheduler = _fake_scheduler()
        scheduler.get_status = AsyncMock(side_effect=_get_status_side_effect)

        report = await reconcile_landed_outbox(outbox, git_ops_recon, scheduler)

        assert outbox.lookup('A') is None, 'row A (already-done) must be pruned'
        assert outbox.lookup('C') is None, 'row C (RC-2) must be pruned after its done-write'
        scheduler.mark_done.assert_called_once_with('C', kind='merged', sha='ADV-C')
        assert report['already_done_pruned'] == 1
        assert report['marked_done'] == 1
        assert report['errors'] == 1

    async def test_status_unknown_row_is_left_unconsumed(self, tmp_path: Path) -> None:
        """A transient get_status() failure (None) must fail-safe: no phantom-done,

        no premature prune — the row survives for the next startup to retry.
        """
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)
        outbox.record(LandedRow(task_id='Z', branch_tip_sha='tip', advanced_sha='ADV', landed_at=1.0))

        reopened = LandedOutbox(path)
        git_ops_recon = _reconciler_git_ops(main_sha='MAIN', is_ancestor_result=True)
        scheduler = _fake_scheduler(get_status_result=None)

        report = await reconcile_landed_outbox(reopened, git_ops_recon, scheduler)

        scheduler.mark_done.assert_not_called()
        assert reopened.lookup('Z') is not None, (
            'status-unknown row must be LEFT unconsumed for the next startup to retry'
        )
        assert report['skipped'] == 1

    async def test_mark_done_raising_leaves_row_unconsumed_for_retry(
        self, tmp_path: Path,
    ) -> None:
        """RC-2 ordering contract (amendment, reviewer_comprehensive #1):
        mark_done runs BEFORE consume(), so a crash — or any exception —
        between the two must leave the row for the next startup to retry.

        Without this test, a regression that reordered consume() before
        mark_done(), or that swallowed a mark_done exception instead of
        letting it propagate to reconcile_landed_outbox's per-row
        try/except, would still pass every other test in this file.
        """
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)
        outbox.record(LandedRow(task_id='Z', branch_tip_sha='tip', advanced_sha='ADV', landed_at=1.0))

        reopened = LandedOutbox(path)
        git_ops_recon = _reconciler_git_ops(main_sha='MAIN', is_ancestor_result=True)
        scheduler = _fake_scheduler(get_status_result='in-progress')
        scheduler.mark_done = AsyncMock(side_effect=RuntimeError('boom'))

        report = await reconcile_landed_outbox(reopened, git_ops_recon, scheduler)

        scheduler.mark_done.assert_called_once_with('Z', kind='merged', sha='ADV')
        assert reopened.lookup('Z') is not None, (
            'row must be LEFT unconsumed when mark_done raises — consume() '
            'must never run before a successful done-write'
        )
        assert report['errors'] == 1
        assert report['marked_done'] == 0


# ---------------------------------------------------------------------------
# step-9 — Harness startup wiring
# ---------------------------------------------------------------------------


def _build_harness(mock_orch_config) -> Harness:
    """Construct a Harness with heavy constructors patched out.

    Mirrors test_harness_orphan_merge_reaper_wiring._build_harness /
    test_harness_merge_store_wiring's bare-harness construction helper.
    """
    mock_orch_config.max_concurrent_tasks = 2
    mock_orch_config.fused_memory.project_id = 'test'

    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        return Harness(mock_orch_config)


@pytest.mark.asyncio
class TestHarnessReconcileLandedOutboxWiring:
    """Harness._reconcile_landed_outbox: None-guard + delegation to the module fn."""

    async def test_none_worker_is_noop(self, mock_orch_config) -> None:
        """RED until step-10 adds Harness._reconcile_landed_outbox.

        Mirrors _reap_orphaned_merge_worktrees's None-guard: no worker means
        nothing to reconcile — must not touch git_ops/scheduler.
        """
        h = _build_harness(mock_orch_config)
        h._merge_worker = None

        with patch(
            'orchestrator.harness.reconcile_landed_outbox', new=AsyncMock(),
        ) as mock_reconcile:
            await h._reconcile_landed_outbox()

        mock_reconcile.assert_not_called()

    async def test_delegates_with_worker_landed_outbox_git_ops_and_scheduler(
        self, mock_orch_config,
    ) -> None:
        """RED until step-10 adds Harness._reconcile_landed_outbox.

        When a merge worker with a bound LandedOutbox is present, the
        harness delegates to the module-level reconcile_landed_outbox with
        the worker's outbox plus the harness's own git_ops/scheduler, and
        (task 2677) the harness's shared ProvenanceConflictSink.
        """
        h = _build_harness(mock_orch_config)
        worker = MagicMock()
        worker._landed_outbox = MagicMock()
        h._merge_worker = worker

        empty_report = {
            'pruned_not_landed': 0, 'marked_done': 0,
            'already_done_pruned': 0, 'skipped': 0, 'stale_conflict': 0, 'errors': 0,
        }
        with patch(
            'orchestrator.harness.reconcile_landed_outbox',
            new=AsyncMock(return_value=empty_report),
        ) as mock_reconcile:
            await h._reconcile_landed_outbox()

        mock_reconcile.assert_awaited_once_with(
            worker._landed_outbox, h.git_ops, h.scheduler,
            provenance_conflict_sink=h._provenance_conflict_sink,
        )


# ---------------------------------------------------------------------------
# task 2677 step-9 — RC-2 done_evidence_stale: 'stale_conflict' disposition
#
# NOT to be confused with this file's own pre-existing "step-9" label above
# (Harness startup wiring) — that refers to task 2155/2156's step numbering.
# This section is task 2677's step-9: the found_on_main provenance-integrity
# gate (task 2674) can refuse RC-2's ``scheduler.mark_done`` with a
# ``StaleEvidenceRejection`` when the row's ``advanced_sha`` predates a later
# ``reopen_at``. That must route to the shared ``ProvenanceConflictSink``
# (a dedupe-guarded, born-at-L2 escalation) rather than propagate up to
# ``reconcile_landed_outbox``'s fail-open per-row ``try/except`` — which
# would silently retry (and re-reject) the same write on every future scan.
# ---------------------------------------------------------------------------


def _stale_evidence_mark_done(
    *, evidence_commit: str = 'ADV', reopen_at: str = '2026-07-15T00:00:00+00:00',
):
    """AsyncMock side_effect: scheduler.mark_done always rejects as stale.

    Mirrors test_reconcile_stranded.py's ``_wire_stale_evidence_mark_done``
    — same ``StaleEvidenceRejection`` shape, factored as a bare side_effect
    callable so each test assigns it directly onto its own fake scheduler's
    ``mark_done``.
    """
    async def _reject(task_id, *, kind, sha, note=None):  # noqa: ARG001
        raise StaleEvidenceRejection(
            task_id=task_id,
            evidence_commit=evidence_commit,
            evidence_committed_at='2026-07-10T00:00:00+00:00',
            reopen_at=reopen_at,
            agent_id='claude-recon-x',
            raw="success=False payload={'error': 'done_evidence_stale'}",
        )
    return _reject


@pytest.mark.asyncio
class TestReconcileLandedRowStaleEvidenceConflict:
    """RC-2 branch, wired sink: 'stale_conflict' disposition (task 2677)."""

    async def test_returns_stale_conflict_row_left_unconsumed_one_l2_filed(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)
        outbox.record(LandedRow(
            task_id='Z', branch_tip_sha='tip', advanced_sha='ADV', landed_at=1.0,
        ))

        git_ops_recon = _reconciler_git_ops(main_sha='MAIN', is_ancestor_result=True)
        scheduler = _fake_scheduler(get_status_result='in-progress')
        scheduler.mark_done = AsyncMock(side_effect=_stale_evidence_mark_done(evidence_commit='ADV'))

        queue = EscalationQueue(tmp_path / 'esc')
        sink = ProvenanceConflictSink(escalation_queue=queue)

        row = outbox.lookup('Z')
        assert row is not None
        disposition = await reconcile_landed_row(
            row, git_ops=git_ops_recon, scheduler=scheduler,
            outbox=outbox, main_sha='MAIN', provenance_conflict_sink=sink,
        )

        assert disposition == 'stale_conflict', f'expected stale_conflict, got {disposition!r}'
        assert outbox.lookup('Z') is not None, (
            'row must be LEFT unconsumed on a stale-evidence rejection — it '
            'is not a genuine RC-1/RC-3 prune and the task is not done'
        )

        pending = queue.get_by_task('Z', status='pending')
        conflicts = [e for e in pending if e.category == 'provenance_conflict']
        assert len(conflicts) == 1, f'expected exactly one pending L2, got {len(conflicts)}'
        assert conflicts[0].level == 2
        assert conflicts[0].severity == 'urgent'

    async def test_repeat_call_still_exactly_one_pending_record(
        self, tmp_path: Path,
    ) -> None:
        """A repeat RC-2 pass at the same evidence must fold via dedupe_count
        or short-circuit via should_skip — either mechanism is acceptable;
        only the outcome (still exactly one pending record) is pinned here,
        mirroring test_reconcile_stranded.py's identical repeat-call test."""
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)
        outbox.record(LandedRow(
            task_id='Z', branch_tip_sha='tip', advanced_sha='ADV', landed_at=1.0,
        ))

        git_ops_recon = _reconciler_git_ops(main_sha='MAIN', is_ancestor_result=True)
        scheduler = _fake_scheduler(get_status_result='in-progress')
        scheduler.mark_done = AsyncMock(side_effect=_stale_evidence_mark_done(evidence_commit='ADV'))

        queue = EscalationQueue(tmp_path / 'esc')
        sink = ProvenanceConflictSink(escalation_queue=queue)

        for _ in range(2):
            row = outbox.lookup('Z')
            assert row is not None, 'row must still be present to re-drive the RC-2 branch'
            disposition = await reconcile_landed_row(
                row, git_ops=git_ops_recon, scheduler=scheduler,
                outbox=outbox, main_sha='MAIN', provenance_conflict_sink=sink,
            )
            assert disposition == 'stale_conflict'

        pending = queue.get_by_task('Z', status='pending')
        conflicts = [e for e in pending if e.category == 'provenance_conflict']
        assert len(conflicts) == 1, (
            f'a repeat rejection at the same evidence must fold or '
            f'short-circuit, not file a second pending record, got {len(conflicts)}'
        )

    async def test_reconcile_landed_task_maps_stale_conflict_to_gated(
        self, tmp_path: Path,
    ) -> None:
        """reconcile_landed_task must report gated=True (task not dispatched)
        on a stale_conflict disposition — a contested task must never
        dispatch while under arbitration, exactly like every other
        non-'pruned_not_landed' disposition."""
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)
        outbox.record(LandedRow(
            task_id='Z', branch_tip_sha='tip', advanced_sha='ADV', landed_at=1.0,
        ))

        git_ops_recon = _reconciler_git_ops(main_sha='MAIN', is_ancestor_result=True)
        scheduler = _fake_scheduler(get_status_result='in-progress')
        scheduler.mark_done = AsyncMock(side_effect=_stale_evidence_mark_done(evidence_commit='ADV'))

        queue = EscalationQueue(tmp_path / 'esc')
        sink = ProvenanceConflictSink(escalation_queue=queue)

        gated = await reconcile_landed_task(
            'Z', git_ops=git_ops_recon, scheduler=scheduler, outbox=outbox,
            provenance_conflict_sink=sink,
        )

        assert gated is True, 'a stale_conflict disposition must gate dispatch (not marked_done, not dispatchable)'


@pytest.mark.asyncio
class TestReconcileLandedRowStaleEvidenceBareBackCompat:
    """provenance_conflict_sink=None (or omitted) preserves today's
    propagate-and-tally-as-'errors' behavior for bare (sink-less) callers."""

    async def test_none_sink_propagates_stale_evidence_rejection(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)
        outbox.record(LandedRow(
            task_id='Z', branch_tip_sha='tip', advanced_sha='ADV', landed_at=1.0,
        ))

        git_ops_recon = _reconciler_git_ops(main_sha='MAIN', is_ancestor_result=True)
        scheduler = _fake_scheduler(get_status_result='in-progress')
        scheduler.mark_done = AsyncMock(side_effect=_stale_evidence_mark_done(evidence_commit='ADV'))

        row = outbox.lookup('Z')
        assert row is not None
        with pytest.raises(StaleEvidenceRejection):
            await reconcile_landed_row(
                row, git_ops=git_ops_recon, scheduler=scheduler,
                outbox=outbox, main_sha='MAIN', provenance_conflict_sink=None,
            )

        assert outbox.lookup('Z') is not None, (
            'row must be left unconsumed when the rejection propagates'
        )

    async def test_reconcile_landed_outbox_tallies_stale_evidence_as_errors(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)
        outbox.record(LandedRow(
            task_id='Z', branch_tip_sha='tip', advanced_sha='ADV', landed_at=1.0,
        ))

        git_ops_recon = _reconciler_git_ops(main_sha='MAIN', is_ancestor_result=True)
        scheduler = _fake_scheduler(get_status_result='in-progress')
        scheduler.mark_done = AsyncMock(side_effect=_stale_evidence_mark_done(evidence_commit='ADV'))

        report = await reconcile_landed_outbox(outbox, git_ops_recon, scheduler)

        assert report['errors'] == 1
        assert report['marked_done'] == 0
        assert outbox.lookup('Z') is not None, 'row must be left unconsumed for retry'


# ---------------------------------------------------------------------------
# Task 3057 step-13 (RED) — seam 8: the RC-2 journal-recovery done-write.
#
# RC-2 fires when the process crashed between the CAS advance and the
# done-write. Git ancestry proves the BRANCH TIP reached main; it never proves
# the task's declared capability rode along with it. The guard is opt-in
# (project_root/check_timeout_secs default to None = NOT ARMED) so every
# pre-existing caller in this 24k-line module stays byte-identical.
# ---------------------------------------------------------------------------

_MQ_GATE_TARGET = 'orchestrator.merge_queue.gate_mark_done_on_delivered_checks'
_MQ_DC_CHECK = {
    'name': 'cap-x', 'kind': 'grep', 'pattern': 'SomePattern', 'expect': 'present',
}


def _mq_block(reason: str = 'failed') -> DeliveredChecksBlock:
    return DeliveredChecksBlock(
        reason=reason,  # type: ignore[arg-type]
        main_sha='MAIN',
        failed_check=_MQ_DC_CHECK if reason == 'failed' else None,
    )


def _mq_row_fixture(
    tmp_path: Path,
    *,
    status: str | None = 'in-progress',
    get_task: AsyncMock | None = None,
    metadata: dict | None = None,
) -> tuple[LandedOutbox, MagicMock, MagicMock, LandedRow]:
    outbox = LandedOutbox(tmp_path / 'landed_outbox.json')
    outbox.record(LandedRow(
        task_id='Z', branch_tip_sha='tip', advanced_sha='ADV', landed_at=1.0,
    ))
    git_ops = _reconciler_git_ops(main_sha='MAIN', is_ancestor_result=True)
    scheduler = _fake_scheduler(get_status_result=status)
    scheduler.get_task = get_task or AsyncMock(return_value={
        'id': 'Z',
        'metadata': (
            {'delivered_checks': [_MQ_DC_CHECK]} if metadata is None else metadata
        ),
    })
    row = outbox.lookup('Z')
    assert row is not None
    return outbox, git_ops, scheduler, row


@pytest.mark.asyncio
class TestReconcileLandedRowDeliveredChecksGuard:
    """The delivered-capability guard on the RC-2 crash-recovery done-write.

    A withholding leaves the row UNCONSUMED — it is the only record of the
    crash-window intent, and the task is not done, so pruning it would lose
    evidence. The new ``'delivered_checks_withheld'`` disposition maps to
    "do NOT gate dispatch", so the task re-dispatches and actually delivers.
    """

    # --- row 1: hollow-done regression / FAILED ---------------------------

    async def test_failed_block_withholds_done_write_and_retains_row(
        self, tmp_path: Path,
    ) -> None:
        outbox, git_ops, scheduler, row = _mq_row_fixture(tmp_path)

        with patch(_MQ_GATE_TARGET, AsyncMock(return_value=_mq_block('failed'))):
            disposition = await reconcile_landed_row(
                row, git_ops=git_ops, scheduler=scheduler, outbox=outbox,
                main_sha='MAIN', project_root='/tmp/proj',
                check_timeout_secs=7.5,
            )

        assert disposition == 'delivered_checks_withheld'
        scheduler.mark_done.assert_not_called()
        assert outbox.lookup('Z') is not None, (
            'the row is the only record of the crash-window intent'
        )

    # --- row 2: all_delivered -> byte-identical RC-2 -----------------------

    async def test_all_delivered_marks_done_and_consumes_as_today(
        self, tmp_path: Path,
    ) -> None:
        outbox, git_ops, scheduler, row = _mq_row_fixture(tmp_path)

        with patch(_MQ_GATE_TARGET, AsyncMock(return_value=None)):
            disposition = await reconcile_landed_row(
                row, git_ops=git_ops, scheduler=scheduler, outbox=outbox,
                main_sha='MAIN', project_root='/tmp/proj',
                check_timeout_secs=7.5,
            )

        assert disposition == 'marked_done'
        scheduler.mark_done.assert_called_once_with(
            'Z', kind='merged', sha='ADV',
        )
        assert outbox.lookup('Z') is None

    # --- row 3: NOT ARMED -> zero behavioural delta for every old caller ---

    async def test_unarmed_caller_never_reaches_the_guard(
        self, tmp_path: Path,
    ) -> None:
        """Every pre-existing caller passes neither param — the helper must
        never run and no metadata round-trip may be made."""
        outbox, git_ops, scheduler, row = _mq_row_fixture(tmp_path)
        guard = AsyncMock(return_value=_mq_block('failed'))

        with patch(_MQ_GATE_TARGET, guard):
            disposition = await reconcile_landed_row(
                row, git_ops=git_ops, scheduler=scheduler, outbox=outbox,
                main_sha='MAIN',
            )

        guard.assert_not_awaited()
        scheduler.get_task.assert_not_called()
        assert disposition == 'marked_done'
        assert outbox.lookup('Z') is None

    # --- rows 4 & 5: fail-safe blocks withhold identically ----------------

    @pytest.mark.parametrize('reason', ['errored', 'main_sha_unresolved'])
    async def test_fail_safe_blocks_withhold_and_retain_the_row(
        self, tmp_path: Path, reason: str,
    ) -> None:
        outbox, git_ops, scheduler, row = _mq_row_fixture(tmp_path)

        with patch(_MQ_GATE_TARGET, AsyncMock(return_value=_mq_block(reason))):
            disposition = await reconcile_landed_row(
                row, git_ops=git_ops, scheduler=scheduler, outbox=outbox,
                main_sha='MAIN', project_root='/tmp/proj',
                check_timeout_secs=7.5,
            )

        assert disposition == 'delivered_checks_withheld'
        scheduler.mark_done.assert_not_called()
        assert outbox.lookup('Z') is not None

    # --- row 6: kill switch is FORWARDED, never re-implemented ------------

    async def test_kill_switch_is_forwarded_not_short_circuited(
        self, tmp_path: Path,
    ) -> None:
        outbox, git_ops, scheduler, row = _mq_row_fixture(tmp_path)
        guard = AsyncMock(return_value=None)

        with patch(_MQ_GATE_TARGET, guard):
            await reconcile_landed_row(
                row, git_ops=git_ops, scheduler=scheduler, outbox=outbox,
                main_sha='MAIN', project_root='/tmp/proj',
                check_timeout_secs=7.5, delivered_checks_enabled=False,
            )

        assert guard.await_args.kwargs['enabled'] is False

    async def test_kill_switch_with_real_helper_marks_done_as_today(
        self, tmp_path: Path,
    ) -> None:
        outbox, git_ops, scheduler, row = _mq_row_fixture(tmp_path)

        disposition = await reconcile_landed_row(
            row, git_ops=git_ops, scheduler=scheduler, outbox=outbox,
            main_sha='MAIN', project_root='/tmp/proj',
            check_timeout_secs=7.5, delivered_checks_enabled=False,
        )

        assert disposition == 'marked_done'
        scheduler.mark_done.assert_called_once()
        assert outbox.lookup('Z') is None

    # --- plumbing: ONE metadata read, and the RC-1 main_sha reused --------

    async def test_metadata_read_once_and_resolved_main_sha_forwarded(
        self, tmp_path: Path,
    ) -> None:
        """The guard must audit the SAME main the RC-1 ancestry test used —
        re-resolving could evaluate a DIFFERENT (newer) main than the decision
        being gated, and would cost an extra git round-trip."""
        meta = {'delivered_checks': [_MQ_DC_CHECK]}
        outbox, git_ops, scheduler, row = _mq_row_fixture(tmp_path, metadata=meta)
        guard = AsyncMock(return_value=None)

        with patch(_MQ_GATE_TARGET, guard):
            await reconcile_landed_row(
                row, git_ops=git_ops, scheduler=scheduler, outbox=outbox,
                main_sha='MAIN', project_root='/tmp/proj',
                check_timeout_secs=7.5,
            )

        scheduler.get_task.assert_awaited_once_with('Z')
        guard.assert_awaited_once()
        assert guard.await_args.args[0] == 'Z'
        assert guard.await_args.args[1] == meta
        assert guard.await_args.kwargs['main_sha'] == 'MAIN'
        assert guard.await_args.kwargs['site'] == 'landed-reconcile-rc2'
        assert guard.await_args.kwargs['project_root'] == '/tmp/proj'
        assert guard.await_args.kwargs['check_timeout_secs'] == 7.5
        git_ops.get_main_sha.assert_not_called()

    # --- fail-safe: unknown metadata never stamps -------------------------

    @pytest.mark.parametrize('mode', ['none', 'raises'])
    async def test_unreadable_task_metadata_withholds(
        self, tmp_path: Path, mode: str,
    ) -> None:
        get_task = (
            AsyncMock(return_value=None) if mode == 'none'
            else AsyncMock(side_effect=RuntimeError('scheduler down'))
        )
        outbox, git_ops, scheduler, row = _mq_row_fixture(
            tmp_path, get_task=get_task,
        )

        disposition = await reconcile_landed_row(
            row, git_ops=git_ops, scheduler=scheduler, outbox=outbox,
            main_sha='MAIN', project_root='/tmp/proj', check_timeout_secs=7.5,
        )

        assert disposition == 'delivered_checks_withheld'
        scheduler.mark_done.assert_not_called()
        assert outbox.lookup('Z') is not None

    # --- ordering: the cheap pre-checks still win -------------------------

    async def test_status_unknown_still_short_circuits_before_the_guard(
        self, tmp_path: Path,
    ) -> None:
        outbox, git_ops, scheduler, row = _mq_row_fixture(tmp_path, status=None)
        guard = AsyncMock(return_value=None)

        with patch(_MQ_GATE_TARGET, guard):
            disposition = await reconcile_landed_row(
                row, git_ops=git_ops, scheduler=scheduler, outbox=outbox,
                main_sha='MAIN', project_root='/tmp/proj',
                check_timeout_secs=7.5,
            )

        assert disposition == 'skipped'
        guard.assert_not_awaited()

    async def test_preserve_status_still_prunes_before_the_guard(
        self, tmp_path: Path,
    ) -> None:
        outbox, git_ops, scheduler, row = _mq_row_fixture(tmp_path, status='done')
        guard = AsyncMock(return_value=None)

        with patch(_MQ_GATE_TARGET, guard):
            disposition = await reconcile_landed_row(
                row, git_ops=git_ops, scheduler=scheduler, outbox=outbox,
                main_sha='MAIN', project_root='/tmp/proj',
                check_timeout_secs=7.5,
            )

        assert disposition == 'already_done_pruned'
        guard.assert_not_awaited()

    async def test_contested_row_still_reports_stale_conflict_before_the_guard(
        self, tmp_path: Path,
    ) -> None:
        outbox, git_ops, scheduler, row = _mq_row_fixture(tmp_path)
        sink = MagicMock()
        sink.should_skip = MagicMock(return_value=True)
        guard = AsyncMock(return_value=None)

        with patch(_MQ_GATE_TARGET, guard):
            disposition = await reconcile_landed_row(
                row, git_ops=git_ops, scheduler=scheduler, outbox=outbox,
                main_sha='MAIN', provenance_conflict_sink=sink,
                project_root='/tmp/proj', check_timeout_secs=7.5,
            )

        assert disposition == 'stale_conflict'
        guard.assert_not_awaited()

    async def test_stale_evidence_handling_unchanged_when_guard_passes(
        self, tmp_path: Path,
    ) -> None:
        outbox, git_ops, scheduler, row = _mq_row_fixture(tmp_path)
        scheduler.mark_done = AsyncMock(
            side_effect=_stale_evidence_mark_done(evidence_commit='ADV'),
        )
        sink = MagicMock()
        sink.should_skip = MagicMock(return_value=False)
        sink.record_from_rejection = MagicMock()

        with patch(_MQ_GATE_TARGET, AsyncMock(return_value=None)):
            disposition = await reconcile_landed_row(
                row, git_ops=git_ops, scheduler=scheduler, outbox=outbox,
                main_sha='MAIN', provenance_conflict_sink=sink,
                project_root='/tmp/proj', check_timeout_secs=7.5,
            )

        assert disposition == 'stale_conflict'
        sink.record_from_rejection.assert_called_once()
        assert outbox.lookup('Z') is not None

    # --- scan-level tally --------------------------------------------------

    async def test_outbox_scan_tallies_the_new_disposition(
        self, tmp_path: Path,
    ) -> None:
        outbox, git_ops, scheduler, _row = _mq_row_fixture(tmp_path)

        with patch(_MQ_GATE_TARGET, AsyncMock(return_value=_mq_block('failed'))):
            report = await reconcile_landed_outbox(
                outbox, git_ops, scheduler,
                project_root='/tmp/proj', check_timeout_secs=7.5,
            )

        assert report['delivered_checks_withheld'] == 1
        assert report['marked_done'] == 0
        assert report['errors'] == 0, 'a withholding is a disposition, not an error'
        assert outbox.lookup('Z') is not None

    async def test_outbox_scan_report_key_present_when_zero(
        self, tmp_path: Path,
    ) -> None:
        """The key must exist even at zero so a dashboard/report reader never
        has to distinguish "absent" from "none withheld"."""
        outbox, git_ops, scheduler, _row = _mq_row_fixture(tmp_path)

        report = await reconcile_landed_outbox(outbox, git_ops, scheduler)

        assert report['delivered_checks_withheld'] == 0
        assert report['marked_done'] == 1
