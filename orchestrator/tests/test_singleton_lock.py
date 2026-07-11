"""Tests for orchestrator singleton lock — prevent concurrent instances."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.harness import Harness, _acquire_project_lock


class TestAcquireProjectLock:
    """Tests for the _acquire_project_lock function."""

    def test_acquires_lock_and_writes_diagnostic_info(self, tmp_path: Path):
        lock_file = _acquire_project_lock(tmp_path)
        try:
            lock_path = tmp_path / 'data' / 'orchestrator' / 'orchestrator.lock'
            assert lock_path.exists()
            content = lock_path.read_text()
            assert 'PID' in content
            assert 'started' in content
        finally:
            lock_file.close()

    def test_creates_parent_directories(self, tmp_path: Path):
        project_root = tmp_path / 'nested' / 'project'
        project_root.mkdir(parents=True)
        lock_file = _acquire_project_lock(project_root)
        try:
            lock_path = project_root / 'data' / 'orchestrator' / 'orchestrator.lock'
            assert lock_path.exists()
        finally:
            lock_file.close()

    def test_second_instance_raises_system_exit(self, tmp_path: Path):
        first = _acquire_project_lock(tmp_path)
        try:
            with pytest.raises(SystemExit) as exc_info:
                _acquire_project_lock(tmp_path)
            assert exc_info.value.code == 1
        finally:
            first.close()

    def test_lock_released_on_close(self, tmp_path: Path):
        first = _acquire_project_lock(tmp_path)
        first.close()
        # Second instance should succeed after first is closed
        second = _acquire_project_lock(tmp_path)
        second.close()

    def test_different_projects_dont_conflict(self, tmp_path: Path):
        project_a = tmp_path / 'a'
        project_b = tmp_path / 'b'
        project_a.mkdir()
        project_b.mkdir()
        lock_a = _acquire_project_lock(project_a)
        lock_b = _acquire_project_lock(project_b)
        lock_a.close()
        lock_b.close()


def _build_dirty_tree_startup_harness(
    mock_orch_config, tmp_path: Path, *, escalation_helper,
):
    """Build a Harness wired to drive run() through startup on a dirty tree.

    Shared by the dirty-tree-escalation integration tests below, which
    otherwise differ only in the ``_file_dirty_tree_escalation`` stub they
    install — factored out so a future startup-step change only needs
    updating in one place. Mirrors startup_harness in
    test_harness_startup_get_statuses.py (same mocks, duplicated here per
    this suite's per-file fixture convention) plus the dirty-tree mock and
    the ``escalation_helper`` stub.

    Returns ``(h, mock_mcp)``. Callers drive ``await h.run()`` under
    ``pytest.raises(RuntimeError, match='__loop_reached_sentinel__')`` —
    the scheduler's ``acquire_next`` raises that sentinel once startup
    reaches the dispatch loop.
    """
    mock_orch_config.max_concurrent_tasks = 2
    mock_orch_config.fused_memory.project_id = 'test'

    with patch('orchestrator.harness.McpLifecycle') as mock_mcp_cls, \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    mock_mcp = mock_mcp_cls.return_value
    mock_mcp.start = AsyncMock()
    mock_mcp.stop = AsyncMock()

    h.git_ops = MagicMock()
    h.git_ops.has_dirty_working_tree = AsyncMock(return_value='M dirty_file.py')
    h.git_ops.worktree_base = tmp_path / '.worktrees'

    h._start_escalation_server = AsyncMock()
    h._start_merge_worker = AsyncMock()
    h._start_offline_lane = AsyncMock()
    h._dismiss_stale_escalations = AsyncMock()
    h._start_orphan_l0_reaper = MagicMock()
    h._start_terminal_status_watcher = MagicMock()
    h._tag_task_modules = AsyncMock()
    h._recover_crashed_tasks = AsyncMock()
    h._reconcile_lane_checkouts = AsyncMock()
    h._reconcile_stranded_in_progress = AsyncMock()
    h._tag_prd_metadata = AsyncMock()

    h._file_dirty_tree_escalation = escalation_helper

    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[
        {'id': '99', 'status': 'pending'},
    ])
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.acquire_next = AsyncMock(
        side_effect=RuntimeError('__loop_reached_sentinel__'),
    )

    return h, mock_mcp


class TestHarnessSingletonIntegration:
    """Tests that Harness.run() acquires and releases the lock."""

    @pytest.mark.asyncio
    async def test_dirty_tree_starts_and_escalates(self, mock_orch_config, tmp_path):
        """Dirty tree no longer refuses to start (task 2380).

        RCA 2026-07-08: the old refuse-to-start guard composed with systemd
        Restart=on-failure + the watchdog revive loop into a silent
        multi-hour crash-loop (459 aborted runs, nothing escalated because
        the process died before it could).  A dirty project_root must now
        start normally and file a deferred born-at-L2 escalation instead.
        """
        escalation_mock = AsyncMock()
        h, mock_mcp = _build_dirty_tree_startup_harness(
            mock_orch_config, tmp_path, escalation_helper=escalation_mock,
        )

        with pytest.raises(RuntimeError, match='__loop_reached_sentinel__'):
            await h.run()

        # MCP (and other) servers now start despite the dirty tree.
        mock_mcp.start.assert_called_once()
        # The deferred dirty-tree escalation helper ran during startup,
        # replacing the old top-of-run() RuntimeError refusal.
        escalation_mock.assert_awaited_once_with(False)

    @pytest.mark.asyncio
    async def test_dirty_tree_escalation_fault_does_not_abort_startup(self, mock_orch_config, tmp_path):
        """A fault filing the dirty-tree escalation must not abort startup.

        Simulates a transient git subprocess/lock failure inside
        has_dirty_working_tree() or an fsync/rename OSError inside
        _escalation_queue.submit(). The deferred call must be guarded like
        every neighboring startup step so a fault here never re-creates the
        silent multi-hour crash-loop (RCA 2026-07-08) this task exists to
        eliminate.

        Mirrors test_dirty_tree_starts_and_escalates's fixture setup; the
        ONE difference is that the escalation helper itself faults.
        """
        escalation_mock = AsyncMock(
            side_effect=OSError('transient escalation I/O fault'),
        )
        h, mock_mcp = _build_dirty_tree_startup_harness(
            mock_orch_config, tmp_path,
            escalation_helper=escalation_mock,
        )

        # Startup must reach the scheduler loop despite the fault — the
        # OSError must not propagate out of run() in place of the sentinel.
        with pytest.raises(RuntimeError, match='__loop_reached_sentinel__'):
            await h.run()

        # MCP (and other) servers still came up despite the faulting escalation.
        mock_mcp.start.assert_called_once()
        # The faulting call was actually exercised during startup.
        escalation_mock.assert_awaited_once_with(False)

    @pytest.mark.asyncio
    async def test_escalation_port_bind_failure_raises(self, mock_orch_config):
        """Escalation server bind failure must raise, not silently continue."""
        import asyncio

        with patch('orchestrator.harness.McpLifecycle') as mock_mcp_cls, \
             patch('orchestrator.harness.Scheduler'), \
             patch('orchestrator.harness.BriefingAssembler'):
            h = Harness(mock_orch_config)

        h.git_ops = MagicMock()
        h.git_ops.has_dirty_working_tree = AsyncMock(return_value=None)
        mock_mcp = mock_mcp_cls.return_value
        mock_mcp.start = AsyncMock()
        mock_mcp.stop = AsyncMock()

        # Simulate escalation server crashing immediately on port bind
        bind_error = OSError(98, 'Address already in use')

        async def _failing_serve():
            raise bind_error

        async def _mock_start_escalation():
            """Replace _start_escalation_server to inject a pre-failed task."""
            h._escalation_task = asyncio.create_task(_failing_serve())
            await asyncio.sleep(0.1)  # let the task fail
            # Now run the fail-fast check from the real method
            if h._escalation_task.done():
                exc = h._escalation_task.exception()
                if exc:
                    raise RuntimeError(
                        f'Escalation server failed to start: {exc}'
                    ) from exc

        h._start_escalation_server = _mock_start_escalation

        with pytest.raises(RuntimeError, match='Escalation server failed to start'):
            await h.run()

        # Lock should be released via finally block
        assert h._lock_file is None
