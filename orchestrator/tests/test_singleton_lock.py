"""Tests for orchestrator singleton lock — prevent concurrent instances."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig
from orchestrator.harness import Harness, _acquire_project_lock


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
    )


@pytest.fixture
def config(tmp_path: Path, git_config: GitConfig):
    config = MagicMock()
    config.git = git_config
    config.project_root = tmp_path
    config.usage_cap.enabled = False
    config.review.enabled = False
    config.sandbox.backend = 'auto'
    return config


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


class TestHarnessSingletonIntegration:
    """Tests that Harness.run() acquires and releases the lock."""

    @pytest.mark.asyncio
    async def test_dirty_tree_check_before_servers(self, config):
        """Dirty-tree check must run before any server starts."""
        with patch('orchestrator.harness.McpLifecycle') as mock_mcp_cls, \
             patch('orchestrator.harness.Scheduler'), \
             patch('orchestrator.harness.BriefingAssembler'):
            h = Harness(config)

        h.git_ops = MagicMock()
        h.git_ops.has_dirty_working_tree = AsyncMock(return_value='M dirty_file.py')
        mock_mcp = mock_mcp_cls.return_value
        mock_mcp.start = AsyncMock()

        with pytest.raises(RuntimeError, match='uncommitted tracked changes'):
            await h.run()

        # MCP server should never have been started
        mock_mcp.start.assert_not_called()
        # Lock should be released
        assert h._lock_file is None

    @pytest.mark.asyncio
    async def test_escalation_port_bind_failure_raises(self, config):
        """Escalation server bind failure must raise, not silently continue."""
        import asyncio

        with patch('orchestrator.harness.McpLifecycle') as mock_mcp_cls, \
             patch('orchestrator.harness.Scheduler'), \
             patch('orchestrator.harness.BriefingAssembler'):
            h = Harness(config)

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
