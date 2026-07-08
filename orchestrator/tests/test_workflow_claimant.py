"""Tests for TaskWorkflow claimant_run_id/heartbeat_at dispatch stamp + heartbeat loop.

PRD plans/task-status-authority-prd.md contract C4/D4 (task 2188, omega1).

Covers:
  - step-13/14: the pending->in-progress dispatch write (inside
    _setup_worktree_and_artifacts) stamps claimant_run_id (composed from the
    process run_id, workflow session_id, and os.getpid()) and heartbeat_at
    (now, UTC ISO-8601) atomically with the status write.
  - step-15/16: a background heartbeat loop refreshes heartbeat_at on a
    bounded config cadence (claimant_run_id untouched — refresh, not
    restamp), and is stopped cleanly via _stop_claimant_heartbeat (the
    helper run()'s finally calls).
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec
from shared.task_claimant import compose_claimant_run_id

from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import WorktreeInfo
from orchestrator.workflow import TaskWorkflow


def _make_workflow(
    *,
    project_root: Path,
    task_id: str = '101',
    run_id: str | None = 'run-abc123',
    claimant_heartbeat_interval_secs: float = 60.0,
) -> TaskWorkflow:
    """Return a minimal TaskWorkflow with scheduler/git_ops mocked."""
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.project_root = project_root
    config.claimant_heartbeat_interval_secs = claimant_heartbeat_interval_secs

    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock(return_value=None)
    scheduler.set_task_claimant = AsyncMock(return_value=None)
    scheduler.update_task = AsyncMock(return_value=True)
    # _setup_worktree_and_artifacts reads live status at dispatch before
    # claiming 'in-progress'; None -> non-terminal -> proceed.
    scheduler.get_status = AsyncMock(return_value=None)

    git_ops = MagicMock()
    git_ops.worktree_base = project_root / '.worktrees'

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        run_id=run_id,
    )
    return wf


async def _setup(wf: TaskWorkflow, base_sha: str = 'a' * 40) -> None:
    """Drive the create_worktree path of _setup_worktree_and_artifacts."""
    wf.git_ops.create_worktree = AsyncMock(
        return_value=WorktreeInfo(path=wf.config.project_root, base_commit=base_sha, stale_commits=0)
    )
    wf._sync_worktree_venvs = AsyncMock()
    with patch('orchestrator.workflow._run', new=AsyncMock(return_value=(0, '', ''))):
        await wf._setup_worktree_and_artifacts('task/101')


# ---------------------------------------------------------------------------
# step-13/14: dispatch stamp
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_stamps_claimant_run_id_and_heartbeat(tmp_path: Path):
    """The pending->in-progress write stamps claimant_run_id + heartbeat_at."""
    wf = _make_workflow(project_root=tmp_path, task_id='101', run_id='run-abc123')

    fixed_now = datetime(2026, 7, 8, 12, 0, 0, tzinfo=UTC)
    with patch('orchestrator.workflow.datetime') as mock_datetime:
        mock_datetime.now.return_value = fixed_now
        await _setup(wf)

    expected_claimant = compose_claimant_run_id('run-abc123', wf.session_id, os.getpid())
    wf.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
        '101', 'in-progress',
        claimant_run_id=expected_claimant,
        heartbeat_at=fixed_now.isoformat(),
    )

    # _setup started the background heartbeat loop; stop it so the test does
    # not leak a pending task (run()'s finally owns this in production).
    await wf._stop_claimant_heartbeat()


@pytest.mark.asyncio
async def test_dispatch_stamp_embeds_run_id_session_id_and_pid(tmp_path: Path):
    """claimant_run_id verbatim-embeds run_id, session_id, and os.getpid()."""
    wf = _make_workflow(project_root=tmp_path, task_id='202', run_id='run-xyz789')

    await _setup(wf)

    kwargs = wf.scheduler.set_task_status.call_args.kwargs  # type: ignore[attr-defined]
    claimant_run_id = kwargs['claimant_run_id']
    assert 'run-xyz789' in claimant_run_id
    assert wf.session_id in claimant_run_id
    assert f'pid={os.getpid()}' in claimant_run_id

    # _setup started the background heartbeat loop; stop it so the test does
    # not leak a pending task (run()'s finally owns this in production).
    await wf._stop_claimant_heartbeat()


# ---------------------------------------------------------------------------
# step-15/16: heartbeat loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heartbeat_loop_starts_after_dispatch_stamp(tmp_path: Path):
    """A background heartbeat task is running immediately after dispatch."""
    wf = _make_workflow(project_root=tmp_path, claimant_heartbeat_interval_secs=0.01)

    await _setup(wf)

    assert wf._claimant_heartbeat_task is not None
    assert not wf._claimant_heartbeat_task.done()

    await wf._stop_claimant_heartbeat()


@pytest.mark.asyncio
async def test_heartbeat_loop_refreshes_heartbeat_only(tmp_path: Path):
    """The loop calls scheduler.set_task_claimant(task_id, heartbeat_at=...)
    without claimant_run_id (refresh, not restamp)."""
    wf = _make_workflow(project_root=tmp_path, task_id='303', claimant_heartbeat_interval_secs=0.01)

    await _setup(wf)
    # Let the loop tick at least once (interval=0.01s).
    await asyncio.sleep(0.05)
    await wf._stop_claimant_heartbeat()

    assert wf.scheduler.set_task_claimant.await_count >= 1  # type: ignore[attr-defined]
    args, kwargs = wf.scheduler.set_task_claimant.call_args  # type: ignore[attr-defined]
    assert args == ('303',)
    assert 'claimant_run_id' not in kwargs
    assert 'heartbeat_at' in kwargs


@pytest.mark.asyncio
async def test_stop_claimant_heartbeat_halts_further_refreshes(tmp_path: Path):
    """After _stop_claimant_heartbeat, no further set_task_claimant calls occur."""
    wf = _make_workflow(project_root=tmp_path, claimant_heartbeat_interval_secs=0.01)

    await _setup(wf)
    await asyncio.sleep(0.05)
    await wf._stop_claimant_heartbeat()
    count_after_stop = wf.scheduler.set_task_claimant.await_count  # type: ignore[attr-defined]

    # Give the (now-cancelled) loop plenty of time to have ticked again if it
    # were still alive.
    await asyncio.sleep(0.05)

    assert wf.scheduler.set_task_claimant.await_count == count_after_stop  # type: ignore[attr-defined]
    assert wf._claimant_heartbeat_task is None or wf._claimant_heartbeat_task.done()


@pytest.mark.asyncio
async def test_stop_claimant_heartbeat_is_idempotent_noop_when_never_started(tmp_path: Path):
    """Calling _stop_claimant_heartbeat before dispatch (no task started) must not raise."""
    wf = _make_workflow(project_root=tmp_path)
    assert wf._claimant_heartbeat_task is None

    await wf._stop_claimant_heartbeat()  # must not raise
