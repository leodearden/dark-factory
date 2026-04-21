"""Tests for Harness._reconcile_stranded_in_progress and the _pid_alive helper."""

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig
from orchestrator.harness import Harness, _pid_alive

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
        harness.scheduler.get_tasks.return_value = [  # type: ignore[attr-defined]
            {'id': 5, 'status': 'in-progress'},
            {'id': 6, 'status': 'pending'},
        ]
        # No worktree directory for task 5 exists (worktree_base not even created)

        await harness._reconcile_stranded_in_progress()

        calls = harness.scheduler.set_task_status.call_args_list  # type: ignore[attr-defined]
        assert len(calls) == 1
        assert calls[0].args[0] == '5'
        assert calls[0].args[1] == 'pending'

    async def test_in_progress_with_live_owner_pid_left_alone(
        self, harness: Harness, tmp_path: Path
    ):
        """In-progress task with plan.lock pointing to live PID → untouched."""
        harness.scheduler.get_tasks.return_value = [  # type: ignore[attr-defined]
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
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        # Lock file must still exist
        assert lock_path.exists()

    async def test_stale_plan_lock_cleared_and_reverted(
        self, harness: Harness
    ):
        """In-progress task with stale plan.lock (dead PID) → lock cleared and task reverted."""
        harness.scheduler.get_tasks.return_value = [  # type: ignore[attr-defined]
            {'id': 8, 'status': 'in-progress'},
        ]
        # Spawn a process and reap it to get a guaranteed-dead PID
        proc = subprocess.Popen(['true'])
        proc.wait()
        dead_pid = proc.pid

        # Create worktree with plan.lock referencing the dead PID
        lock_dir = harness.git_ops.worktree_base / '8' / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': '8-dead0001',
            'locked_at': datetime.now(UTC).isoformat(),
            'owner_pid': dead_pid,
        }))

        await harness._reconcile_stranded_in_progress()

        # Task must be reverted to pending
        harness.scheduler.set_task_status.assert_called_once_with('8', 'pending')  # type: ignore[attr-defined]
        # Stale lock must be deleted
        assert not lock_path.exists()

    async def test_fresh_plan_lock_owner_pid_alive_left_alone(
        self, harness: Harness, caplog
    ):
        """Fresh plan.lock with live owner_pid → no revert, no log mentioning 'revert' or 'stranded'."""
        import logging
        harness.scheduler.get_tasks.return_value = [  # type: ignore[attr-defined]
            {'id': 7, 'status': 'in-progress'},
        ]
        lock_dir = harness.git_ops.worktree_base / '7' / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': '7-abcd1234',
            'locked_at': datetime.now(UTC).isoformat(),
            'owner_pid': os.getpid(),
        }))

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
            await harness._reconcile_stranded_in_progress()

        # No status change
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        # Lock file intact
        assert lock_path.exists()
        # No log message mentioning revert or stranded for this task
        revert_logs = [r for r in caplog.records if 'revert' in r.message.lower() or 'stranded' in r.message.lower()]
        assert len(revert_logs) == 0


# ---------------------------------------------------------------------------
# Harness.run() call-order test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_harness_run_invokes_reconcile_before_scheduler_loop(
    tmp_path: Path,
):
    """run() must call _recover_crashed_tasks → _reconcile_stranded_in_progress
    → scheduler.acquire_next in that order.
    """
    call_order: list[str] = []

    git_cfg = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
    )
    config = MagicMock()
    config.git = git_cfg
    config.project_root = tmp_path
    config.usage_cap.enabled = False
    config.review.enabled = False
    config.max_concurrent_tasks = 2
    config.fused_memory.project_id = 'test'

    with patch('orchestrator.harness.McpLifecycle') as mock_mcp_cls, \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(config)

    # --- mock infrastructure methods so run() doesn't fail early ---
    h.git_ops = MagicMock()
    h.git_ops.has_dirty_working_tree = AsyncMock(return_value=None)
    h.git_ops.worktree_base = tmp_path / '.worktrees'

    mock_mcp = mock_mcp_cls.return_value
    mock_mcp.start = AsyncMock()
    mock_mcp.stop = AsyncMock()

    h._start_escalation_server = AsyncMock()
    h._start_merge_worker = AsyncMock()
    h._dismiss_stale_escalations = AsyncMock()
    h._start_orphan_l0_reaper = MagicMock()
    h._tag_task_modules = AsyncMock()

    # Provide one pending task so the "no pending tasks" check passes
    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[
        {'id': 1, 'status': 'pending', 'title': 'A task'},
    ])
    h.scheduler.set_task_status = AsyncMock()

    # Track ordering: _recover_crashed_tasks
    async def _fake_recover():
        call_order.append('recover')
    h._recover_crashed_tasks = _fake_recover

    # Track ordering: _reconcile_stranded_in_progress
    async def _fake_reconcile():
        call_order.append('reconcile')
    h._reconcile_stranded_in_progress = _fake_reconcile

    # Track ordering: acquire_next — append then raise to break the loop
    async def _fake_acquire():
        call_order.append('acquire')
        raise RuntimeError('stop the loop')
    h.scheduler.acquire_next = _fake_acquire

    with pytest.raises(RuntimeError, match='stop the loop'):
        await h.run(prd_path=None)

    # _recover_crashed_tasks then _reconcile_stranded_in_progress then acquire_next
    assert 'recover' in call_order, "_recover_crashed_tasks was not called"
    assert 'reconcile' in call_order, "_reconcile_stranded_in_progress was not called"
    assert 'acquire' in call_order, "scheduler.acquire_next was not called"
    recover_idx = call_order.index('recover')
    reconcile_idx = call_order.index('reconcile')
    acquire_idx = call_order.index('acquire')
    assert recover_idx < reconcile_idx, "_recover_crashed_tasks must precede _reconcile_stranded_in_progress"
    assert reconcile_idx < acquire_idx, "_reconcile_stranded_in_progress must precede scheduler.acquire_next"


# ---------------------------------------------------------------------------
# Non-in-progress statuses are ignored (regression guard for non-goal)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_in_progress_statuses_ignored(harness: Harness):
    """The sweep only touches in-progress tasks; other statuses are untouched."""
    # Only the 'in-progress' task has no lock (worktree doesn't exist)
    harness.scheduler.get_tasks.return_value = [  # type: ignore[attr-defined]
        {'id': 20, 'status': 'pending'},
        {'id': 21, 'status': 'done'},
        {'id': 22, 'status': 'blocked'},
        {'id': 23, 'status': 'cancelled'},
        {'id': 24, 'status': 'review'},
        {'id': 25, 'status': 'in-progress'},  # <-- only this one
    ]
    # No worktree for task 25 (orphan)

    await harness._reconcile_stranded_in_progress()

    calls = harness.scheduler.set_task_status.call_args_list  # type: ignore[attr-defined]
    assert len(calls) == 1, f"Expected exactly 1 call, got: {calls}"
    assert calls[0].args[0] == '25'
    assert calls[0].args[1] == 'pending'
