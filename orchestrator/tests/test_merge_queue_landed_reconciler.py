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
from orchestrator.landed_outbox import LandedOutbox
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
