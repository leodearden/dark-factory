"""Tests for Harness.run() startup wiring of the shared-repo auto-gc/maintenance disable.

PRD plans/os-sandbox-worktree-containment-prd.md task α5 (D2 corollary): the
orchestrator must disable auto-gc/maintenance on its managed shared repo at
startup — before the first dispatch — so ``git config get gc.auto`` == 0 after
orchestrator startup. Harness.run() calls
``git_ops.disable_shared_repo_auto_maintenance()`` as a best-effort startup
step (wrapped so a git/config failure never blocks startup).

Modeled on test_harness_startup_get_statuses.py: patch the infra constructors,
mock the side-effect methods, and break the run() loop with the acquire_next
RuntimeError sentinel.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.harness import Harness


@pytest.fixture
def startup_harness(tmp_path: Path, mock_orch_config) -> Harness:
    """Create a Harness with all run() side-effects mocked for startup tests."""
    mock_orch_config.max_concurrent_tasks = 2
    mock_orch_config.fused_memory.project_id = 'test'

    # Patch constructors so __init__ doesn't spin up real infrastructure.
    with patch('orchestrator.harness.McpLifecycle') as mock_mcp_cls, \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    # h.mcp is mock_mcp_cls.return_value — make it awaitable.
    mock_mcp = mock_mcp_cls.return_value
    mock_mcp.start = AsyncMock()
    mock_mcp.stop = AsyncMock()

    h.git_ops = MagicMock()
    h.git_ops.has_dirty_working_tree = AsyncMock(return_value=None)
    h.git_ops.worktree_base = tmp_path / '.worktrees'
    # The unit under test: seed as an AsyncMock so run()'s awaited call is
    # awaitable and we can assert it was awaited during startup.
    h.git_ops.disable_shared_repo_auto_maintenance = AsyncMock()

    # Mock all side-effect methods so the test exercises only the startup block.
    h._start_escalation_server = AsyncMock()
    h._start_merge_worker = AsyncMock()
    h._dismiss_stale_escalations = AsyncMock()
    h._tag_task_modules = AsyncMock()
    h._recover_crashed_tasks = AsyncMock()
    h._reconcile_lane_checkouts = AsyncMock()
    h._reconcile_stranded_in_progress = AsyncMock()
    h._tag_prd_metadata = AsyncMock()

    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[
        {'id': '99', 'status': 'pending'},
    ])
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))
    h.scheduler.set_task_status = AsyncMock()
    # Raise on acquire_next to stop the scheduler loop after startup. Reuse the
    # distinctive sentinel from the get_statuses model so it cannot collide with
    # unrelated RuntimeErrors.
    h.scheduler.acquire_next = AsyncMock(
        side_effect=RuntimeError('__loop_reached_sentinel__'),
    )

    return h


@pytest.mark.asyncio
async def test_startup_disables_shared_repo_auto_maintenance(
    startup_harness: Harness,
):
    """Harness.run() awaits git_ops.disable_shared_repo_auto_maintenance at startup.

    Satisfies the α5 signal 'gc.auto==0 after orchestrator startup' before any
    dispatch. The acquire_next sentinel proves the startup block ran; the
    assert_awaited proves the disable call fired within it.
    """
    h = startup_harness

    with pytest.raises(RuntimeError, match='__loop_reached_sentinel__'):
        await h.run(prd_path=None)

    h.git_ops.disable_shared_repo_auto_maintenance.assert_awaited()
