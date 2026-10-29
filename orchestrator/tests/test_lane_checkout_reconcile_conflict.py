"""A lane whose WIP cannot be snapshotted must not kill orchestrator startup (task 4053).

The escape chain this file pins, one layer per test class:

1. ``GitOps.commit`` refuses to snapshot a lane: it raises
   ``WorktreeConflictError`` on an unmerged index and ``RuntimeError('Commit
   failed: ...')`` when ``git commit`` itself fails (e.g. a rejecting hook).
2. ``GitOps.detach_lane_checkout`` awaits that commit before
   ``git checkout --detach``; its contract is a bool, False meaning "the
   branch was not freed".
3. ``Harness._reconcile_lane_checkouts`` calls it at three sites
   (dup-checkout, deleted-id, terminal) and handles only the False return.
4. ``Harness.run`` startup step 2c2 awaits the sweep; an exception escaping
   it ends the orchestrator process on every restart.

A conflicted lane can never be detached anyway (``git checkout --detach``
refuses on an unmerged index), so the correct outcome at every layer is:
leave the checkout attached, keep or restore the pin, warn, and carry on.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _workflow_helpers import _build_harness, _init_git_repo

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, WorktreeConflictError, _run
from orchestrator.harness import Harness

_BARE_ID = '3965'
_TASK_BRANCH = f'task/{_BARE_ID}'
_HOOK_REJECTION = 'test pre-commit hook: snapshot rejected'


async def _git(cwd: Path, *args: str) -> str:
    rc, out, err = await _run(['git', *args], cwd=cwd)
    assert rc == 0, f'git {" ".join(args)} failed in {cwd}: {err}'
    return out.strip()


async def _pool_with_lane(
    tmp_path: Path,
    *,
    bare_id: str = _BARE_ID,
    lane_name: str = '_lane-5',
    pool_size: int = 6,
) -> tuple[OrchestratorConfig, GitOps, Path]:
    """A pool-enabled GitOps over a real repo with ``task/<bare_id>`` checked out at *lane_name*."""
    repo = tmp_path / 'repo'
    _init_git_repo(repo)
    config = OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=pool_size,
        git=GitConfig(warm_lane_pool=True),
        worktree_identity_guard_enabled=False,
    )
    git_ops = GitOps(config.git, repo, warm_lane_pool_size=pool_size)
    assert git_ops.warm_lane_pool is not None

    lane = git_ops.worktree_base / lane_name
    lane.parent.mkdir(parents=True, exist_ok=True)
    await _git(repo, 'worktree', 'add', str(lane), '-b', f'task/{bare_id}', 'main')
    return config, git_ops, lane


async def _make_index_conflicted(repo: Path, lane: Path) -> None:
    """Leave *lane* mid-merge with README.md unmerged (``UU``)."""
    (lane / 'README.md').write_text('lane side\n')
    await _git(lane, 'commit', '-am', 'lane side of README')

    await _git(repo, 'checkout', '-b', 'other', 'main')
    (repo / 'README.md').write_text('other side\n')
    await _git(repo, 'commit', '-am', 'other side of README')
    await _git(repo, 'checkout', 'main')

    rc, _, _ = await _run(['git', 'merge', 'other'], cwd=lane)
    assert rc != 0, 'the merge was expected to conflict'
    assert 'UU README.md' in await _git(lane, 'status', '--porcelain')


async def _make_commit_hook_reject(repo: Path, lane: Path) -> None:
    """Leave uncommitted WIP in *lane* under a pre-commit hook that rejects every commit."""
    hooks = repo.parent / 'failhooks'
    hooks.mkdir()
    pre_commit = hooks / 'pre-commit'
    pre_commit.write_text(f'#!/bin/sh\necho "{_HOOK_REJECTION}" >&2\nexit 1\n')
    pre_commit.chmod(0o755)
    await _git(repo, 'config', 'core.hooksPath', str(hooks))
    (lane / 'README.md').write_text('uncommitted lane WIP\n')


class TestDetachLaneCheckoutRefusedSnapshot:
    """``detach_lane_checkout`` answers False, never raises, when its WIP snapshot is refused."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('make_refused', 'detail'),
        [
            pytest.param(_make_index_conflicted, 'README.md', id='index_conflicted'),
            pytest.param(_make_commit_hook_reject, _HOOK_REJECTION, id='commit_hook_rejects'),
        ],
    )
    async def test_returns_false_and_leaves_checkout_attached(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        make_refused: Callable[[Path, Path], Awaitable[None]],
        detail: str,
    ):
        _config, git_ops, lane = await _pool_with_lane(tmp_path)
        repo = git_ops.project_root
        await make_refused(repo, lane)
        tip_before = await _git(repo, 'rev-parse', _TASK_BRANCH)

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            detached = await git_ops.detach_lane_checkout(lane, _BARE_ID)

        assert detached is False
        assert await _git(lane, 'symbolic-ref', '-q', 'HEAD') == f'refs/heads/{_TASK_BRANCH}', (
            'a lane whose WIP could not be snapshotted must stay attached to its branch'
        )
        assert await _git(repo, 'rev-parse', _TASK_BRANCH) == tip_before, (
            'no snapshot (and so no conflict markers) may have been committed'
        )
        warnings = [
            record.getMessage() for record in caplog.records
            if record.name == 'orchestrator.git_ops' and record.levelno >= logging.WARNING
        ]
        assert any(str(lane) in message and detail in message for message in warnings), warnings


class TestReconcileLaneCheckoutsConflictedLane:
    """The startup sweep absorbs a conflicted lane at each of its three detach sites."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('statuses', 'trusted_pin_name'),
        [
            pytest.param({_BARE_ID: 'done'}, None, id='terminal'),
            pytest.param({'999': 'pending'}, None, id='deleted_id'),
            pytest.param({_BARE_ID: 'in-progress'}, '_lane-2', id='dup_checkout'),
        ],
    )
    async def test_sweep_completes_and_keeps_the_conflicted_lane_pinned(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        statuses: dict[str, str],
        trusted_pin_name: str | None,
    ):
        config, git_ops, lane = await _pool_with_lane(tmp_path)
        await _make_index_conflicted(git_ops.project_root, lane)
        pool = git_ops.warm_lane_pool
        assert pool is not None
        checkouts = await git_ops.lane_branch_checkouts()
        assert checkouts is not None
        conflicted_lane = checkouts[_BARE_ID]

        harness = _build_harness(config)
        harness.git_ops = git_ops
        harness.scheduler.get_statuses = AsyncMock(return_value=(statuses, None))
        if trusted_pin_name is None:
            expected_pin = conflicted_lane
        else:
            expected_pin = git_ops.worktree_base / trusted_pin_name
            pool.restore_assignment(_BARE_ID, expected_pin)

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await harness._reconcile_lane_checkouts()

        assert await _git(lane, 'symbolic-ref', '-q', 'HEAD') == f'refs/heads/{_TASK_BRANCH}'
        assert pool.assignment_for(_BARE_ID) == expected_pin, (
            'the conflicted lane must stay pinned (or the trusted pin stay put), '
            'never handed to a fresh dispatch'
        )
        assert any(
            'FAILED to detach' in record.getMessage() for record in caplog.records
            if record.name == 'orchestrator.harness' and record.levelno == logging.WARNING
        ), caplog.text


@pytest.fixture
def startup_harness(tmp_path: Path, mock_orch_config: MagicMock) -> Harness:
    """A Harness whose run() exercises only the startup block, then stops at acquire_next."""
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
    h.git_ops.has_dirty_working_tree = AsyncMock(return_value=None)
    h.git_ops.worktree_base = tmp_path / '.worktrees'
    h.git_ops.disable_shared_repo_auto_maintenance = AsyncMock()

    h._start_escalation_server = AsyncMock()
    h._start_merge_worker = AsyncMock()
    h._dismiss_stale_escalations = AsyncMock()
    h._tag_task_modules = AsyncMock()
    h._recover_crashed_tasks = AsyncMock()
    h._reconcile_lane_checkouts = AsyncMock()
    h._reconcile_stranded_in_progress = AsyncMock()
    h._tag_prd_metadata = AsyncMock()

    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[{'id': '99', 'status': 'pending'}])
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.acquire_next = AsyncMock(
        side_effect=RuntimeError('__loop_reached_sentinel__'),
    )
    return h


class TestStartupSurvivesLaneCheckoutReconcileFailure:
    """Startup step 2c2 is non-fatal: whatever escapes the sweep, startup carries on."""

    @pytest.mark.asyncio
    async def test_escaping_sweep_error_is_logged_and_startup_continues(
        self,
        startup_harness: Harness,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        h = startup_harness
        reconcile = AsyncMock(
            side_effect=WorktreeConflictError(tmp_path / '_lane-5', ['README.md']),
        )
        h._reconcile_lane_checkouts = reconcile

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'), \
             pytest.raises(RuntimeError, match='__loop_reached_sentinel__'):
            await h.run(prd_path=None)

        reconcile.assert_awaited_once()
        cast(AsyncMock, h._reconcile_stranded_in_progress).assert_awaited()
        assert any(
            'lane checkouts' in record.getMessage() and 'README.md' in record.getMessage()
            for record in caplog.records
            if record.name == 'orchestrator.harness' and record.levelno == logging.WARNING
        ), caplog.text
