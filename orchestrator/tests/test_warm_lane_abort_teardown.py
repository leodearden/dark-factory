"""Tests for GitOps._abort_lane_acquisition — the unified never-raise teardown
primitive every acquire_warm_lane fault-exit routes through (task 2199).

Self-contained, mirroring test_warm_lane_disk_guard.py: conftest.py provides
no shared `git_repo` fixture, so this module owns its own fixture + stub
scripts rather than depending on test_git_ops.py's module-level helpers.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import (
    GitOps,
    WarmLaneUnavailable,
    WorktreeInfo,
    _run,
)
from orchestrator.warm_lane_pool import LaneState

# ---------------------------------------------------------------------------
# Repo fixture (mirrors test_warm_lane_disk_guard.py / test_git_ops.py)
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Temporary git repo with an initial commit + a resolvable warm base.

    Task 2061: pre-creates the DEFAULT derived warm-lane base
    (<repo>/.worktrees/_merge-verify/target, non-empty) so the
    acquire_warm_lane pre-acquire base-health gate sees WarmBaseHealth.OK.
    Task 2099: also marks `.worktrees/.pool-root` present so the
    create-once mount-presence guard does not false-trip.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    default_base = repo / '.worktrees' / '_merge-verify' / 'target'
    default_base.mkdir(parents=True, exist_ok=True)
    (default_base / '.keep').write_text('warm base sentinel\n')
    (repo / '.worktrees' / '.pool-root').touch()
    return repo


# ---------------------------------------------------------------------------
# Warm-lane stub scripts + config helper
# ---------------------------------------------------------------------------


async def _add_warm_lane_scripts(repo: Path, port: int = 39411) -> None:
    """Commit stub seed-warm-lane.sh + setup-worktree-debug-port.sh into repo."""
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    seed_script = scripts_dir / 'seed-warm-lane.sh'
    seed_script.write_text(
        '#!/usr/bin/env bash\nmkdir -p "$2/target"\necho "seeded" > "$2/target/seeded.bin"\n'
    )
    seed_script.chmod(0o755)
    debug_script = scripts_dir / 'setup-worktree-debug-port.sh'
    debug_script.write_text(f'#!/usr/bin/env bash\necho {port}\n')
    debug_script.chmod(0o755)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add warm-lane scripts'], cwd=repo)


def _warm_config(**overrides) -> GitConfig:
    """Build a GitConfig with the warm-lane pool enabled (canonical settings)."""
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
        **overrides,
    )


async def _get_head(repo: Path) -> str:
    """Return the HEAD commit SHA of the repo."""
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0, f'git rev-parse HEAD failed (rc={rc})'
    return out.strip()


# ---------------------------------------------------------------------------
# _abort_lane_acquisition — detach + pool-release core (steps 1-4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAbortLaneAcquisitionDetachAndRelease:
    """_abort_lane_acquisition: HEAD-detach + pool-release core contract."""

    async def test_abort_remove_worktree_false_detaches_and_frees(
        self, git_repo: Path,
    ):
        """remove_worktree=False: HEAD detaches, pool slot frees, worktree retained."""
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        info = await git_ops.acquire_warm_lane('X', start_ref)
        assert isinstance(info, WorktreeInfo), f'Expected WorktreeInfo; got {info!r}'

        await git_ops._abort_lane_acquisition(info.path, 'X', remove_worktree=False)

        _, branch, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=info.path,
        )
        assert branch.strip() == 'HEAD', (
            f'Expected detached HEAD after abort; got branch={branch.strip()!r}'
        )
        assert git_ops.warm_lane_pool is not None
        assert git_ops.warm_lane_pool.state(info.path) == LaneState.FREE, (
            'Pool slot must be FREE after abort'
        )
        assert await git_ops._is_registered_worktree(info.path), (
            'remove_worktree=False must retain the worktree registration'
        )

    async def test_abort_remove_worktree_true_removes_minted_worktree(
        self, git_repo: Path,
    ):
        """remove_worktree=True: the minted worktree is removed; slot still frees."""
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        info = await git_ops.acquire_warm_lane('X', start_ref)
        assert isinstance(info, WorktreeInfo), f'Expected WorktreeInfo; got {info!r}'

        await git_ops._abort_lane_acquisition(info.path, 'X', remove_worktree=True)

        assert not await git_ops._is_registered_worktree(info.path), (
            'remove_worktree=True must remove the minted worktree'
        )
        assert git_ops.warm_lane_pool is not None
        assert git_ops.warm_lane_pool.state(info.path) == LaneState.FREE, (
            'Pool slot must be FREE after abort'
        )
