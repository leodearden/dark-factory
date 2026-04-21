"""Tests for Harness._reconcile_stranded_in_progress and the _pid_alive helper."""

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.harness import _pid_alive
from orchestrator.config import GitConfig
from orchestrator.harness import Harness


# ---------------------------------------------------------------------------
# _pid_alive helper tests
# ---------------------------------------------------------------------------

class TestPidAlive:
    def test_current_pid_is_alive(self):
        assert _pid_alive(os.getpid()) is True

    def test_impossible_pid_is_dead(self):
        # PID well beyond the Linux kernel max (2^22 on 64-bit, 2^15 on 32-bit).
        # 2**31-1 is always invalid on all Linux systems.
        assert _pid_alive(2**31 - 1) is False


# ---------------------------------------------------------------------------
# Harness fixture (mirrors test_crash_recovery.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
    )


@pytest.fixture
def harness(tmp_path: Path, git_config: GitConfig):
    """Create a Harness with mocked internals for unit testing reconciliation."""
    config = MagicMock()
    config.git = git_config
    config.project_root = tmp_path
    config.usage_cap.enabled = False

    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(config)

    # Replace scheduler with async mocks
    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[])
    h.scheduler.set_task_status = AsyncMock()

    # Keep worktree_base real (under tmp_path) so we can create fake worktrees
    h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()

    return h


# ---------------------------------------------------------------------------
# _reconcile_stranded_in_progress tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestReconcileStrandedInProgress:
    async def test_orphan_without_worktree_reverted(self, harness: Harness):
        """In-progress task with no worktree dir → reverted to pending (no-lock)."""
        harness.scheduler.get_tasks.return_value = [
            {'id': 5, 'status': 'in-progress'},
            {'id': 6, 'status': 'pending'},
        ]
        # No worktree directory for task 5 exists (worktree_base not even created)

        await harness._reconcile_stranded_in_progress()

        calls = harness.scheduler.set_task_status.call_args_list
        assert len(calls) == 1
        assert calls[0].args[0] == '5'
        assert calls[0].args[1] == 'pending'

    async def test_in_progress_with_live_owner_pid_left_alone(
        self, harness: Harness, tmp_path: Path
    ):
        """In-progress task with plan.lock pointing to live PID → untouched."""
        harness.scheduler.get_tasks.return_value = [
            {'id': 7, 'status': 'in-progress'},
        ]
        # Create worktree with a plan.lock containing our own (live) PID
        lock_dir = harness.git_ops.worktree_base / '7' / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': '7-abcd1234',
            'locked_at': datetime.now(UTC).isoformat(),
            'owner_pid': os.getpid(),
        }))

        await harness._reconcile_stranded_in_progress()

        # Must NOT revert
        harness.scheduler.set_task_status.assert_not_called()
        # Lock file must still exist
        assert lock_path.exists()
