"""Tests for the Harness.run() startup block "pure read" contract.

Harness.run() must use scheduler.get_statuses (not get_tasks) at the three
startup sites: the pre_ids snapshot before PRD parse, the pending-task check
when no PRD is given, and the total_tasks count after reconcile.  Two
companion tests cover the transport-failure vs genuinely-empty distinction
introduced by task 1010.

Task 1563 removed the 'No pending tasks found' startup refusal so the
orchestrator can cold-start and idle on an empty queue under the
run-until-stopped lifecycle.  The transport-failure raise is preserved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.harness import Harness, HarnessReport

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

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

    # Mock all side-effect methods so tests exercise only the startup block.
    h._start_escalation_server = AsyncMock()
    h._start_merge_worker = AsyncMock()
    h._dismiss_stale_escalations = AsyncMock()
    h._start_orphan_l0_reaper = MagicMock()
    h._start_terminal_status_watcher = MagicMock()
    h._tag_task_modules = AsyncMock()
    h._recover_crashed_tasks = AsyncMock()
    h._reconcile_stranded_in_progress = AsyncMock()
    h._tag_prd_metadata = AsyncMock()

    # Scheduler mock: get_tasks is seeded with a non-empty list as a trap value.
    # Each test asserts get_tasks is NOT called; a non-empty seed would surface
    # (via false-positive logic) any harness regression that re-introduces a
    # get_tasks code path.  get_statuses is seeded empty — each test overrides it.
    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[
        {'id': '99', 'status': 'pending'},
    ])
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))
    h.scheduler.set_task_status = AsyncMock()
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
    """No-PRD startup uses get_statuses (not get_tasks) for the pending-task check.

    An empty but reachable tree ({}, None) no longer raises 'No pending tasks
    found' — the harness proceeds into the main loop where the fixture's
    acquire_next sentinel raises RuntimeError('stop').
    """
    h = startup_harness
    get_tasks_mock = cast(AsyncMock, h.scheduler.get_tasks)
    get_statuses_mock = cast(AsyncMock, h.scheduler.get_statuses)
    # get_statuses seeded empty — guard removed, harness now reaches the loop.
    get_statuses_mock.return_value = ({}, None)

    with pytest.raises(RuntimeError, match='stop'):
        await h.run(prd_path=None)

    # get_statuses IS the call site for the pending-task check.
    get_statuses_mock.assert_called()
    # get_tasks must NOT be called in the startup block.
    get_tasks_mock.assert_not_called()


# ---------------------------------------------------------------------------
# (b) total_tasks counted via get_statuses
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_startup_total_tasks_counted_via_get_statuses(
    startup_harness: Harness,
):
    """total_tasks report field is derived from get_statuses pending count, not get_tasks."""
    h = startup_harness
    get_tasks_mock = cast(AsyncMock, h.scheduler.get_tasks)
    get_statuses_mock = cast(AsyncMock, h.scheduler.get_statuses)
    # Three tasks: two pending, one done.
    get_statuses_mock.return_value = (
        {'1': 'pending', '2': 'done', '3': 'pending'},
        None,
    )

    with pytest.raises(RuntimeError, match='stop'):
        await h.run(prd_path=None)

    assert h.report.total_tasks == 2
    # get_statuses IS the call site for the count.
    get_statuses_mock.assert_called()
    # get_tasks must NOT be called for the startup block.
    get_tasks_mock.assert_not_called()


# ---------------------------------------------------------------------------
# (c) pre_ids snapshot via get_statuses before PRD parse
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_populate_prd_uses_get_statuses_for_pre_ids(
    startup_harness: Harness,
    tmp_path: Path,
):
    """With prd_path: pre_ids snapshot is taken from get_statuses.keys(), not get_tasks."""
    prd = tmp_path / 'feature.md'
    prd.write_text('# PRD')

    h = startup_harness
    get_tasks_mock = cast(AsyncMock, h.scheduler.get_tasks)
    get_statuses_mock = cast(AsyncMock, h.scheduler.get_statuses)
    get_statuses_mock.return_value = (
        {'10': 'done', '20': 'pending'},
        None,
    )

    captured_pre_ids: set[str] | None = None

    async def _capture(path: Path, pre_ids: set[str]) -> None:
        nonlocal captured_pre_ids
        captured_pre_ids = set(pre_ids)

    h._tag_prd_metadata = _capture  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match='stop'):
        await h.run(prd_path=prd)

    assert captured_pre_ids == {'10', '20'}
    # get_statuses IS the call site for the pre_ids snapshot.
    get_statuses_mock.assert_called()
    # get_tasks must NOT be called for the pre_ids snapshot.
    get_tasks_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Transport-failure vs genuinely-empty distinction (Task 1010)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_startup_noprd_transport_failure_raises_distinct_error(
    startup_harness: Harness,
):
    """No-PRD startup: a get_statuses transport failure raises a distinct error.

    When get_statuses returns ({}, OSError), Harness must raise
    RuntimeError('Failed to reach fused-memory: …') chained from the original
    exception — NOT the generic 'No PRD given' message.
    """
    h = startup_harness
    # Simulate: get_statuses returns ({}, OSError) — transport failure in the tuple.
    expected_err = OSError(2, 'No such file')
    h.scheduler.get_statuses = AsyncMock(return_value=({}, expected_err))

    with pytest.raises(RuntimeError, match=r'[Ff]ailed to reach fused-memory') as excinfo:
        await h.run(prd_path=None)

    msg = str(excinfo.value)
    # Error message must include the exception class name and the message.
    # Note: OSError(2, ...) raises as FileNotFoundError (errno 2 = ENOENT remapping).
    expected_cls = type(expected_err).__name__
    assert expected_cls in msg, f'Expected {expected_cls!r} class name in message: {msg}'
    assert 'No such file' in msg, f'Expected OSError message in error: {msg}'

    # Exception must be chained from the original OSError.
    assert excinfo.value.__cause__ is expected_err


@pytest.mark.asyncio
async def test_startup_noprd_empty_reachable_tree_idles_not_raises(
    startup_harness: Harness,
    caplog: pytest.LogCaptureFixture,
):
    """No-PRD startup: reachable but empty task tree does NOT raise 'No pending tasks found'.

    When get_statuses returns ({}, None) (no transport failure), the harness
    logs an INFO idle banner and proceeds into the main loop rather than
    refusing to start.  The fixture's acquire_next sentinel breaks the loop
    with RuntimeError('stop') — asserting 'stop' proves execution passed the
    guard.
    """
    h = startup_harness
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))

    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match='stop'):
        await h.run(prd_path=None)

    # The idle banner must be emitted.
    assert 'No pending tasks at startup' in caplog.text
    # The old guard message must NOT appear.
    assert 'No pending tasks found' not in caplog.text


@pytest.mark.asyncio
async def test_startup_until_idle_empty_tree_exits_cleanly(
    startup_harness: Harness,
):
    """--until-idle with empty task tree exits cleanly, returning a HarnessReport.

    When get_statuses returns ({}, None) and until_idle=True, the harness
    proceeds through the removed startup guard, enters the main loop, and
    exits immediately via the existing 'if self._until_idle: break' drain check
    rather than raising.  Proves the --until-idle immediate-exit-on-empty
    contract is preserved by the main loop without new startup-time logic.

    GOTCHA: scheduler.is_paused must be set False — MagicMock().is_paused is
    truthy by default and would route acquire_next=None into the paused-idle
    sleep branch, hanging the test forever.
    """
    h = startup_harness
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))
    h.scheduler.is_paused = False
    h.scheduler.acquire_next = AsyncMock(return_value=None)
    # Override return_value so mid-run reconcile returns 0 (no tasks freed);
    # the fixture default returns a MagicMock which compares > 0 as truthy
    # and would spin the loop forever.
    h._reconcile_stranded_in_progress = AsyncMock(return_value=0)
    # Mock _enforce_cost_ceilings: setting is_paused=False bypasses its
    # early-return guard and the MagicMock config values cause a TypeError
    # on the ceiling comparisons.  The cost-ceiling logic is not under test here.
    h._enforce_cost_ceilings = AsyncMock()

    report = await h.run(prd_path=None, until_idle=True)

    assert isinstance(report, HarnessReport)
