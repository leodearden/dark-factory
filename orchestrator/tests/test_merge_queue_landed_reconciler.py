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
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.git_ops import AdvanceOutcome
from orchestrator.landed_outbox import LandedOutbox, LandedRow
from orchestrator.merge_queue import _journal_landed_then_advance, reconcile_landed_outbox

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
