"""Tests for Harness._dismiss_stale_escalations() — auto-dismiss stale escalations on startup."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig
from orchestrator.harness import Harness


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
    )


@pytest.fixture
def harness(tmp_path: Path, git_config: GitConfig) -> Harness:
    """Create a Harness with mocked internals for unit testing stale escalation cleanup."""
    config = MagicMock()
    config.git = git_config
    config.project_root = tmp_path
    config.usage_cap.enabled = False
    config.sandbox.backend = 'auto'

    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(config)

    # Replace scheduler with async mocks
    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[])
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))

    return h


@pytest.mark.asyncio
class TestDismissStaleEscalations:
    """Harness._dismiss_stale_escalations() auto-dismisses pending escalations."""

    async def test_no_queue_is_noop(self, harness: Harness):
        """When _escalation_queue is None (no escalation support), method is a no-op."""
        harness._escalation_queue = None
        # Should not raise; count should be 0 (nothing happens)
        await harness._dismiss_stale_escalations()
        # No assertion needed beyond "no exception raised"

    async def test_has_escalation_false_is_noop(self, harness: Harness, caplog):
        """When HAS_ESCALATION is False, method is a no-op with no side effects."""
        harness._escalation_queue = None
        with patch('orchestrator.harness.HAS_ESCALATION', False), caplog.at_level(logging.INFO):
            await harness._dismiss_stale_escalations()

        assert 'dismissed' not in caplog.text.lower()

    async def test_empty_queue_is_noop(self, harness: Harness, caplog):
        """When queue has no pending escalations, method does nothing."""
        mock_queue = MagicMock()
        mock_queue.dismiss_all_pending.return_value = 0
        harness._escalation_queue = mock_queue

        with caplog.at_level(logging.INFO):
            await harness._dismiss_stale_escalations()

        mock_queue.dismiss_all_pending.assert_called_once()
        # No "dismissed N" log line for zero count
        assert 'dismissed 0' not in caplog.text.lower()

    async def test_pending_escalations_dismissed(self, harness: Harness, caplog):
        """Pending escalations are all dismissed with the correct resolution message."""
        mock_queue = MagicMock()
        mock_queue.dismiss_all_pending.return_value = 3
        harness._escalation_queue = mock_queue

        with caplog.at_level(logging.INFO):
            await harness._dismiss_stale_escalations()

        mock_queue.dismiss_all_pending.assert_called_once()
        call_args = mock_queue.dismiss_all_pending.call_args
        resolution_msg = call_args[0][0]
        assert 'stale' in resolution_msg.lower() or 'prior' in resolution_msg.lower()

    async def test_dismissal_count_logged(self, harness: Harness, caplog):
        """When escalations are dismissed, count is logged at INFO level."""
        mock_queue = MagicMock()
        mock_queue.dismiss_all_pending.return_value = 5
        harness._escalation_queue = mock_queue

        with caplog.at_level(logging.INFO):
            await harness._dismiss_stale_escalations()

        assert '5' in caplog.text

    async def test_called_after_start_escalation_server_before_task_loop(
        self, harness: Harness, tmp_path: Path
    ):
        """_dismiss_stale_escalations is called after _start_escalation_server
        but before _recover_crashed_tasks in Harness.run()."""
        call_order: list[str] = []

        async def mock_mcp_start():
            pass

        async def mock_mcp_stop():
            pass

        async def mock_start_escalation_server():
            call_order.append('start_escalation_server')

        async def mock_dismiss_stale_escalations():
            call_order.append('dismiss_stale_escalations')

        async def mock_recover_crashed_tasks():
            call_order.append('recover_crashed_tasks')

        # Mock the PRD path
        prd_path = tmp_path / 'test.prd'
        prd_path.write_text('# Test PRD')

        harness.mcp = MagicMock()
        harness.mcp.start = AsyncMock(side_effect=mock_mcp_start)
        harness.mcp.stop = AsyncMock(side_effect=mock_mcp_stop)
        harness.mcp.url = 'http://localhost:9999'

        harness._start_escalation_server = AsyncMock(
            side_effect=mock_start_escalation_server
        )
        harness._dismiss_stale_escalations = AsyncMock(
            side_effect=mock_dismiss_stale_escalations
        )
        harness._recover_crashed_tasks = AsyncMock(
            side_effect=mock_recover_crashed_tasks
        )
        harness._stop_escalation_server = AsyncMock()
        harness._populate_tasks = AsyncMock()
        harness._tag_prd_metadata = AsyncMock()
        harness._tag_task_modules = AsyncMock()
        harness.scheduler.get_tasks = AsyncMock(return_value=[])

        # Run with dry_run to avoid the task execution loop
        await harness.run(prd_path, dry_run=True)

        # Verify ordering: start_escalation_server → dismiss_stale_escalations
        # (recover_crashed_tasks may not be in dry_run path, but server startup is)
        assert 'start_escalation_server' in call_order
        assert 'dismiss_stale_escalations' in call_order

        server_idx = call_order.index('start_escalation_server')
        dismiss_idx = call_order.index('dismiss_stale_escalations')
        assert dismiss_idx > server_idx, (
            f'dismiss_stale_escalations ({dismiss_idx}) must come after '
            f'start_escalation_server ({server_idx})'
        )

        # Also verify dismiss comes before recover_crashed_tasks when that is called
        if 'recover_crashed_tasks' in call_order:
            recover_idx = call_order.index('recover_crashed_tasks')
            assert dismiss_idx < recover_idx, (
                f'dismiss_stale_escalations ({dismiss_idx}) must come before '
                f'recover_crashed_tasks ({recover_idx})'
            )


@pytest.mark.asyncio
class TestDismissStaleEscalationsFatal:
    """_dismiss_stale_escalations() failure must not prevent harness cleanup."""

    async def test_dismiss_failure_does_not_prevent_finally(
        self, harness: Harness, tmp_path: Path
    ):
        """If _dismiss_stale_escalations() raises, the finally block still runs.

        Specifically: _stop_escalation_server() and mcp.stop() must be called
        even when _dismiss_stale_escalations() raises an OSError.
        """
        prd_path = tmp_path / 'test.prd'
        prd_path.write_text('# Test PRD')

        harness.mcp = MagicMock()
        harness.mcp.start = AsyncMock()
        harness.mcp.stop = AsyncMock()
        harness.mcp.url = 'http://localhost:9999'

        harness._start_escalation_server = AsyncMock()
        harness._dismiss_stale_escalations = AsyncMock(
            side_effect=OSError('disk full simulated failure')
        )
        harness._stop_escalation_server = AsyncMock()

        # run() catches the OSError internally; _populate_tasks (un-mocked) raises RuntimeError
        with pytest.raises(RuntimeError):
            await harness.run(prd_path, dry_run=True)

        # Finally block must have run
        harness._stop_escalation_server.assert_called_once()
        harness.mcp.stop.assert_called_once()

    async def test_dismiss_failure_logged_as_warning(
        self, harness: Harness, tmp_path: Path, caplog
    ):
        """If _dismiss_stale_escalations() raises, the exception is caught and logged,
        not re-raised as an unhandled exception that aborts the entire run."""
        prd_path = tmp_path / 'test.prd'
        prd_path.write_text('# Test PRD')

        harness.mcp = MagicMock()
        harness.mcp.start = AsyncMock()
        harness.mcp.stop = AsyncMock()
        harness.mcp.url = 'http://localhost:9999'

        harness._start_escalation_server = AsyncMock()
        harness._stop_escalation_server = AsyncMock()
        harness._populate_tasks = AsyncMock()
        harness._tag_prd_metadata = AsyncMock()
        harness._tag_task_modules = AsyncMock()
        harness._recover_crashed_tasks = AsyncMock()

        error_msg = 'disk full simulated failure'
        harness._dismiss_stale_escalations = AsyncMock(
            side_effect=OSError(error_msg)
        )

        # Run should complete without re-raising (dry_run stops after task population)
        with caplog.at_level(logging.WARNING):
            # If the exception propagates, run() would raise. If it's caught
            # and logged, run() should complete normally.
            try:
                await harness.run(prd_path, dry_run=True)
                # If we get here, the exception was caught — good
                run_completed = True
            except OSError:
                run_completed = False

        assert run_completed, (
            '_dismiss_stale_escalations() exception should be caught and logged, '
            'not re-raised to abort the run'
        )

        # The warning should appear in logs
        assert error_msg in caplog.text or 'dismiss' in caplog.text.lower()
