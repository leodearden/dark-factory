"""Tests for A1 step-13/14/15/16: harness guards that prevent a verify-complete
infra_hold task from being re-pended.

Step 13 (RED): _revert_in_progress_if_no_live_claimant — task with infra_hold
on a non-degenerate branch must NOT be reverted to pending when the claimant
is gone.  Currently RED because the guard does not exist yet.

Step 15 (RED): _on_escalation_resolved — an infra_issue resolution for an
infra_hold task resumes-at-verify (set in-progress) instead of the default
resume→pending flip.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.harness import Harness

# ---------------------------------------------------------------------------
# Harness fixture (mirrors test_reconcile_stranded.py)
# ---------------------------------------------------------------------------

_BASE_SHA = 'a' * 40   # branch_base_sha in metadata
_TIP_SHA  = 'b' * 40   # tip SHA that is DIFFERENT from base → non-degenerate
_DEGEN_SHA = _BASE_SHA  # tip == base → degenerate


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config) -> Harness:
    """Create a Harness with mocked internals for unit testing infra-hold guards."""
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    # Replace scheduler with async mocks
    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[])
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.update_task = AsyncMock(return_value=True)

    # Default get_task: returns None (no metadata)
    h.scheduler.get_task = AsyncMock(return_value=None)

    # Keep worktree_base real so we can create fake worktrees
    h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()

    # Default cleanup_worktree: side_effect removes the dir
    def _fake_cleanup(worktree_path, tid):
        shutil.rmtree(worktree_path, ignore_errors=True)
    h.git_ops.cleanup_worktree = AsyncMock(side_effect=_fake_cleanup)

    # Default: branch is non-degenerate (tip ≠ base)
    h.git_ops.resolve_branch_sha = AsyncMock(return_value=_TIP_SHA)

    return h


def _make_worktree(harness: Harness, tid: str, *, with_lock: bool = False) -> Path:
    """Create a minimal worktree directory for *tid*.

    If ``with_lock`` is False (default), no plan.lock is created — simulates
    the no-live-claimant / orphan scenario that triggers a revert.
    """
    wt = harness.git_ops.worktree_base / tid
    (wt / '.task').mkdir(parents=True, exist_ok=True)
    if with_lock:
        (wt / '.task' / 'plan.lock').write_text(json.dumps({
            'session_id': f'{tid}-test',
            'locked_at': '2026-06-23T00:00:00Z',
            'owner_pid': os.getpid(),  # live PID → not reaped
        }))
    return wt


# ---------------------------------------------------------------------------
# Step 13: _revert_in_progress_if_no_live_claimant — infra_hold guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRevertInProgressInfraHoldGuard:
    """_revert_in_progress_if_no_live_claimant must NOT revert infra_hold tasks.

    RED before step-14: function reverts regardless of metadata.
    GREEN after step-14: metadata.infra_hold + non-degenerate branch → skip revert.
    """

    async def test_infra_hold_non_degenerate_not_reverted(
        self, harness: Harness, monkeypatch
    ):
        """Task with infra_hold + non-degenerate branch + no live claimant → intact.

        Currently FAILS (RED): _revert_in_progress_if_no_live_claimant always
        reverts when there is no plan.lock, regardless of infra_hold.
        After step-14 the guard fires and returns None (task left intact).
        """
        tid = '1883'
        _make_worktree(harness, tid)  # no lock → no live claimant

        # Task metadata has infra_hold set
        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid,
            'metadata': {
                'infra_hold': {'phase': 'warm_marker', 'errno': 28, 'reason': 'ENOSPC'},
                'branch_base_sha': _BASE_SHA,
            },
        })
        # Non-degenerate branch: tip ≠ base
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=_TIP_SHA)

        result = await harness._revert_in_progress_if_no_live_claimant(
            tid, mid_run=False,
        )

        # Must NOT revert — return None (or any non-'reverted' value) and
        # MUST NOT call set_task_status with 'pending'.
        assert result != 'reverted', (
            f'_revert_in_progress_if_no_live_claimant reverted task {tid!r} '
            f'despite metadata.infra_hold being set and branch being non-degenerate. '
            f'An infra_hold hold must survive the stranded recovery sweep.'
        )
        for call in harness.scheduler.set_task_status.await_args_list:
            assert call.args[1] != 'pending', (
                f'set_task_status({tid!r}, "pending") was called despite infra_hold: '
                f'{call}'
            )

    async def test_no_infra_hold_still_reverts(self, harness: Harness):
        """Task WITHOUT infra_hold + no live claimant → reverts normally.

        This is the contrast/no-regression case: the guard must be NARROW.
        Passes both before and after step-14 (existing revert behaviour preserved).
        """
        tid = '1884'
        _make_worktree(harness, tid)  # no lock → no live claimant

        # No infra_hold in metadata
        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid,
            'metadata': {},
        })

        result = await harness._revert_in_progress_if_no_live_claimant(
            tid, mid_run=False,
        )

        assert result == 'reverted', (
            f'Task without infra_hold should be reverted to pending; got {result!r}'
        )
        harness.scheduler.set_task_status.assert_awaited_once_with(tid, 'pending')

    async def test_infra_hold_degenerate_branch_still_reverts(
        self, harness: Harness, monkeypatch
    ):
        """Task with infra_hold BUT degenerate branch → still reverts.

        A degenerate branch (tip == branch_base_sha) has no real implementation
        commits — it was just provisioned.  The guard is narrow: infra_hold alone
        is insufficient; the branch must also be non-degenerate.

        Passes both before and after step-14 (guard requires BOTH conditions).
        """
        tid = '1885'
        _make_worktree(harness, tid)  # no lock → no live claimant

        # infra_hold set but branch is degenerate (tip == base → no work done)
        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid,
            'metadata': {
                'infra_hold': {'phase': 'warm_marker', 'errno': 28},
                'branch_base_sha': _BASE_SHA,
            },
        })
        # Degenerate: branch tip == branch_base_sha
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=_DEGEN_SHA)

        result = await harness._revert_in_progress_if_no_live_claimant(
            tid, mid_run=False,
        )

        # Degenerate branch → still reverts (no real work to preserve)
        assert result == 'reverted', (
            f'Task with infra_hold on degenerate branch should still revert; '
            f'got {result!r}'
        )
