"""Tests for the single-task landed-outbox dispatch-gate entry (task 2156, W1 δ).

Covers:
  step-1  (RED)  ``reconcile_landed_task`` — the merge_queue-level, single-task
                 consult shared by the scheduler's consult-before-dispatch
                 gate (PRD merge-queue-reliability §8.2 SD-1, boundary B5):
                 no-row hot-path guard, on-main + pending (gate + drive-done +
                 consume), not-on-main (allow dispatch + prune), and
                 status-unknown fail-safe (gate but leave unconsumed for the
                 next retry).

Mirrors test_merge_queue_landed_reconciler.py's mock-git_ops / fake-scheduler
convention with a REAL LandedOutbox on ``tmp_path`` so ``lookup()``/
``consume()`` exercise the real durable store, not a mock.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.landed_outbox import LandedOutbox, LandedRow
from orchestrator.merge_queue import reconcile_landed_task

# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_merge_queue_landed_reconciler.py)
# ---------------------------------------------------------------------------


def _reconciler_git_ops(*, main_sha: str = 'MAIN', is_ancestor_result: bool = True) -> MagicMock:
    """AsyncMock git_ops exposing get_main_sha + is_ancestor."""
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
# No row for task_id — hot-path guard: zero git I/O, zero scheduler calls.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReconcileLandedTaskNoRow:
    """No landed row for task_id => False, without touching git_ops/scheduler."""

    async def test_no_row_returns_false_without_git_or_scheduler_calls(
        self, tmp_path: Path,
    ) -> None:
        outbox = LandedOutbox(tmp_path / 'landed_outbox.json')
        git_ops = _reconciler_git_ops()
        scheduler = _fake_scheduler()

        result = await reconcile_landed_task(
            'Z', git_ops=git_ops, scheduler=scheduler, outbox=outbox,
        )

        assert result is False
        git_ops.get_main_sha.assert_not_called()
        git_ops.is_ancestor.assert_not_called()
        scheduler.get_status.assert_not_called()
        scheduler.mark_done.assert_not_called()


# ---------------------------------------------------------------------------
# Row present, advanced_sha on main, task still pending — gate + drive done.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReconcileLandedTaskOnMainPending:
    """Row on main + pending task => True (gate), mark_done, then consume."""

    async def test_gates_and_drives_done_write_then_consumes(self, tmp_path: Path) -> None:
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)
        outbox.record(LandedRow(
            task_id='Z', branch_tip_sha='tip', advanced_sha='ADV', landed_at=1.0,
        ))

        git_ops = _reconciler_git_ops(main_sha='MAIN', is_ancestor_result=True)
        scheduler = _fake_scheduler(get_status_result='in-progress')

        result = await reconcile_landed_task(
            'Z', git_ops=git_ops, scheduler=scheduler, outbox=outbox,
        )

        assert result is True, 'advanced_sha on main => gate (do NOT dispatch)'
        scheduler.mark_done.assert_called_once_with('Z', kind='merged', sha='ADV')
        assert outbox.lookup('Z') is None, 'row must be consumed after the done-write'


# ---------------------------------------------------------------------------
# Row present but advanced_sha NOT on main — allow dispatch, prune stale row.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReconcileLandedTaskNotOnMain:
    """Row not landed (crash before CAS advance) => False, prune, no done-write."""

    async def test_not_landed_returns_false_and_prunes_without_done_write(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)
        outbox.record(LandedRow(
            task_id='Z', branch_tip_sha='tip', advanced_sha='ADV', landed_at=1.0,
        ))

        git_ops = _reconciler_git_ops(main_sha='MAIN', is_ancestor_result=False)
        scheduler = _fake_scheduler()

        result = await reconcile_landed_task(
            'Z', git_ops=git_ops, scheduler=scheduler, outbox=outbox,
        )

        assert result is False, 'advanced_sha not on main => task stays dispatchable'
        scheduler.mark_done.assert_not_called()
        assert outbox.lookup('Z') is None, 'stale (not-landed) row must still be pruned'


# ---------------------------------------------------------------------------
# Row present, on main, but get_status is unknown (None) — fail-safe gate.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReconcileLandedTaskStatusUnknown:
    """get_status() transient None => still gate (fail-safe), row left unconsumed."""

    async def test_status_unknown_gates_but_leaves_row_unconsumed(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)
        outbox.record(LandedRow(
            task_id='Z', branch_tip_sha='tip', advanced_sha='ADV', landed_at=1.0,
        ))

        git_ops = _reconciler_git_ops(main_sha='MAIN', is_ancestor_result=True)
        scheduler = _fake_scheduler(get_status_result=None)

        result = await reconcile_landed_task(
            'Z', git_ops=git_ops, scheduler=scheduler, outbox=outbox,
        )

        assert result is True, (
            'advanced_sha is on main but status is unknown => still gate '
            '(fail-safe) rather than risk a ghost re-dispatch'
        )
        scheduler.mark_done.assert_not_called()
        assert outbox.lookup('Z') is not None, (
            'status-unknown row must be LEFT unconsumed for the next startup/tick to retry'
        )
