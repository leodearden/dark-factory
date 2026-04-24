"""Tests for the Harness.run() startup block migration to get_statuses.

These tests cover the three sites in run() that should use get_statuses
instead of get_tasks after the step-16 migration:

  (a) Line 365: pre_ids snapshot before PRD parse
  (b) Lines 372-377: pending-task check when no PRD given
  (c) Lines 389-390: total_tasks count after reconcile

Each test is RED against the current harness (which still calls get_tasks)
and GREEN after the step-16 impl migrates those three sites.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig
from orchestrator.harness import Harness

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def startup_harness(tmp_path: Path) -> Harness:
    """Create a Harness with all run() side-effects mocked for startup tests."""
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

    # Patch constructors so __init__ doesn't spin up real infrastructure.
    with patch('orchestrator.harness.McpLifecycle') as mock_mcp_cls, \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(config)

    # h.mcp is mock_mcp_cls.return_value — make it awaitable.
    mock_mcp = mock_mcp_cls.return_value
    mock_mcp.start = AsyncMock()
    mock_mcp.stop = AsyncMock()

    h.git_ops = MagicMock()
    h.git_ops.has_dirty_working_tree = AsyncMock(return_value=None)
    h.git_ops.worktree_base = tmp_path / '.worktrees'

    # Mock all side-effect methods so tests exercise only the startup block.
    h._start_escalation_server = AsyncMock()
    h._start_merge_worker = AsyncMock()
    h._dismiss_stale_escalations = AsyncMock()
    h._start_orphan_l0_reaper = MagicMock()
    h._tag_task_modules = AsyncMock()
    h._recover_crashed_tasks = AsyncMock()
    h._reconcile_stranded_in_progress = AsyncMock()
    h._populate_tasks = AsyncMock()
    h._tag_prd_metadata = AsyncMock()

    # Scheduler mock: get_tasks returns one pending task so the pre-migration
    # "no pending tasks" guard lets tests (b) and (c) proceed far enough to
    # be meaningful.  get_statuses is seeded empty — each test overrides it.
    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[
        {'id': '99', 'status': 'pending'},
    ])
    h.scheduler.get_statuses = AsyncMock(return_value={})
    h.scheduler.set_task_status = AsyncMock()
    # Default: no cached transport error (tests that need one set it explicitly).
    h.scheduler.last_get_statuses_error = None
    # Raise on acquire_next to stop the scheduler loop after startup.
    h.scheduler.acquire_next = AsyncMock(side_effect=RuntimeError('stop'))

    return h


# ---------------------------------------------------------------------------
# (a) No-PRD pending check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_startup_noprd_uses_get_statuses_to_check_pending(
    startup_harness: Harness,
):
    """No-PRD startup: get_statuses (not get_tasks) used for the pending check.

    RED — harness currently calls get_tasks for this check.
    After step-16 migration: get_statuses is called; get_tasks is not.
    """
    h = startup_harness
    # Signal "no tasks" via both mocks.
    # Pre-migration: get_tasks returns [] → any(...) is False → RuntimeError.
    # Post-migration: get_statuses returns {} → 'pending' not in values → RuntimeError.
    h.scheduler.get_tasks.return_value = []  # type: ignore[attr-defined]
    h.scheduler.get_statuses.return_value = {}  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match='No PRD given and no pending tasks found'):
        await h.run(prd_path=None)

    # After migration: get_statuses used for the check.
    h.scheduler.get_statuses.assert_called()  # type: ignore[attr-defined]
    # After migration: get_tasks NOT called in the startup block.
    h.scheduler.get_tasks.assert_not_called()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# (b) total_tasks counted via get_statuses
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_startup_total_tasks_counted_via_get_statuses(
    startup_harness: Harness,
):
    """total_tasks report field is derived from get_statuses pending count, not get_tasks.

    RED — harness currently calls get_tasks to count pending tasks at line 389.
    After step-16 migration: get_statuses is used instead.
    """
    h = startup_harness
    # Three tasks: two pending, one done.
    h.scheduler.get_statuses.return_value = {  # type: ignore[attr-defined]
        '1': 'pending',
        '2': 'done',
        '3': 'pending',
    }

    with pytest.raises(RuntimeError, match='stop'):
        await h.run(prd_path=None)

    assert h.report.total_tasks == 2
    # After migration: get_tasks never called for the startup block.
    h.scheduler.get_tasks.assert_not_called()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# (c) pre_ids snapshot via get_statuses before PRD parse
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_populate_prd_uses_get_statuses_for_pre_ids(
    startup_harness: Harness,
    tmp_path: Path,
):
    """With prd_path: pre_ids snapshot taken from get_statuses.keys(), not get_tasks.

    RED — harness currently calls get_tasks to build pre_ids at line 365.
    After step-16 migration: get_statuses is called and pre_ids = set(statuses.keys()).
    """
    prd = tmp_path / 'feature.md'
    prd.write_text('# PRD')

    h = startup_harness
    h.scheduler.get_statuses.return_value = {  # type: ignore[attr-defined]
        '10': 'done',
        '20': 'pending',
    }

    captured_pre_ids: set[str] | None = None

    async def _capture(path: Path, pre_ids: set[str]) -> None:
        nonlocal captured_pre_ids
        captured_pre_ids = set(pre_ids)

    h._tag_prd_metadata = _capture  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match='stop'):
        await h.run(prd_path=prd)

    assert captured_pre_ids == {'10', '20'}
    # After migration: get_tasks NOT called for the pre_ids snapshot.
    h.scheduler.get_tasks.assert_not_called()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Transport-failure vs genuinely-empty distinction (Task 1010)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_startup_noprd_transport_failure_raises_distinct_error(
    startup_harness: Harness,
):
    """No-PRD startup: a get_statuses transport failure raises a distinct error.

    When get_statuses returns {} AND _last_get_statuses_error is set, Harness
    must raise RuntimeError('Failed to reach fused-memory: …') chained from
    the original exception — NOT the generic 'No PRD given' message.
    """
    h = startup_harness
    # Simulate: get_statuses swallowed a transport error and cached it.
    h.scheduler.get_statuses = AsyncMock(return_value={})
    h.scheduler.last_get_statuses_error = OSError(2, 'No such file')

    with pytest.raises(RuntimeError, match=r'[Ff]ailed to reach fused-memory') as excinfo:
        await h.run(prd_path=None)

    msg = str(excinfo.value)
    # Error message must include the exception class name and the message.
    # Note: OSError(2, ...) raises as FileNotFoundError (errno 2 = ENOENT remapping).
    cached_err = h.scheduler.last_get_statuses_error
    expected_cls = type(cached_err).__name__
    assert expected_cls in msg, f'Expected {expected_cls!r} class name in message: {msg}'
    assert 'No such file' in msg, f'Expected OSError message in error: {msg}'

    # Exception must be chained from the original OSError.
    assert excinfo.value.__cause__ is cached_err


@pytest.mark.asyncio
async def test_startup_noprd_empty_without_cached_error_raises_legitimate_error(
    startup_harness: Harness,
):
    """No-PRD startup: genuinely empty task tree still raises the original error.

    When get_statuses returns {} AND _last_get_statuses_error is None (no
    transport failure), the existing 'No PRD given and no pending tasks found'
    RuntimeError is raised unchanged — regression protection for the
    legitimately-empty path.
    """
    h = startup_harness
    h.scheduler.get_statuses = AsyncMock(return_value={})
    h.scheduler._last_get_statuses_error = None

    with pytest.raises(RuntimeError, match='No PRD given and no pending tasks found'):
        await h.run(prd_path=None)
