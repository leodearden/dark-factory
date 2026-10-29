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

import pytest
from _workflow_helpers import _init_git_repo

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run

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
