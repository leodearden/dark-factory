"""Tests for the orphan ``_merge-*`` worktree reaper wiring in Harness (task 2060).

Verifies the startup-sweep wiring added in plan step-10:

(step-9) Test 1: ``_recover_pending_merges`` RETURNS its ``recover_pending_merges``
         report dict (previously returned ``None``) so ``run()`` can thread the
         recovered branches into the reaper.

(step-9) Test 2: ``_reap_orphaned_merge_worktrees`` delegates to
         ``self._merge_worker.reap_orphaned_merge_worktrees``, passing the
         recovered requests' branches as ``recovered_branches=[...]``.

(step-9) Test 3: ``_reap_orphaned_merge_worktrees`` is None-safe — a no-op that
         never raises when ``self._merge_worker`` is ``None`` (worker not yet
         built / disabled).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.harness import Harness

# ---------------------------------------------------------------------------
# Shared harness builder — mirrors test_harness_merge_store_wiring
# ---------------------------------------------------------------------------


def _build_harness(mock_orch_config) -> Harness:
    """Construct a Harness with heavy constructors patched out."""
    mock_orch_config.max_concurrent_tasks = 2
    mock_orch_config.fused_memory.project_id = 'test'

    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        return Harness(mock_orch_config)


# ---------------------------------------------------------------------------
# Test 1 — _recover_pending_merges returns its report
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRecoverPendingMergesReturnsReport:
    """_recover_pending_merges surfaces the recover_pending_merges report."""

    async def test_returns_report_dict(self, mock_orch_config, tmp_path: Path):
        """RED until step-10 changes _recover_pending_merges to ``return report``."""
        h = _build_harness(mock_orch_config)
        h.event_store = None

        report = {
            'recovered': 1,
            'dropped': 0,
            'requests': [MagicMock(branch='b1')],
            'journal_corrupt': False,
        }
        with patch('orchestrator.harness.recover_pending_merges',
                   new=AsyncMock(return_value=report)):
            result = await h._recover_pending_merges()

        assert result is report, (
            f'_recover_pending_merges must return its report dict; got {result!r}'
        )


# ---------------------------------------------------------------------------
# Test 2/3 — _reap_orphaned_merge_worktrees delegation + None-safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReapOrphanedMergeWorktreesWiring:
    """_reap_orphaned_merge_worktrees delegates to the worker with branches."""

    async def test_delegates_with_recovered_branches(
        self, mock_orch_config, tmp_path: Path,
    ):
        """RED until step-10 adds Harness._reap_orphaned_merge_worktrees.

        The recovered MergeRequests' ``.branch`` values are passed to the
        worker's ``reap_orphaned_merge_worktrees`` as the ``recovered_branches``
        kwarg so their in-flight worktrees are re-adopted rather than reaped.
        """
        h = _build_harness(mock_orch_config)
        h._merge_worker = MagicMock()
        h._merge_worker.reap_orphaned_merge_worktrees = AsyncMock(
            return_value={'readopted': [], 'reaped': []},
        )

        requests = [MagicMock(branch='b1'), MagicMock(branch='b2')]
        await h._reap_orphaned_merge_worktrees(requests)

        h._merge_worker.reap_orphaned_merge_worktrees.assert_awaited_once_with(
            recovered_branches=['b1', 'b2'],
        )

    async def test_none_worker_is_noop(self, mock_orch_config, tmp_path: Path):
        """None-safe: no worker → no-op that never raises.

        RED until step-10 adds the ``if self._merge_worker is None: return``
        guard (before then the method does not exist at all).
        """
        h = _build_harness(mock_orch_config)
        h._merge_worker = None

        # Must not raise despite a non-empty recovered-request list.
        await h._reap_orphaned_merge_worktrees([MagicMock(branch='b1')])
