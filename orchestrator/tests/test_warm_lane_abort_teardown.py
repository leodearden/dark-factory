"""Tests for GitOps._abort_lane_acquisition — the unified never-raise teardown
primitive every acquire_warm_lane fault-exit routes through (task 2199).

Self-contained, mirroring test_warm_lane_disk_guard.py: conftest.py provides
no shared `git_repo` fixture, so this module owns its own fixture + stub
scripts rather than depending on test_git_ops.py's module-level helpers.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch

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


# ---------------------------------------------------------------------------
# _abort_lane_acquisition — WIP-commit + degenerate-gated retention (steps 5-6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAbortLaneAcquisitionWipAndRetention:
    """_abort_lane_acquisition: commit-WIP-first, then degenerate-gated delete."""

    async def test_abort_commits_wip_before_teardown(self, git_repo: Path):
        """Uncommitted TRACKED WIP is committed onto the branch before teardown."""
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        info = await git_ops.acquire_warm_lane('X', start_ref)
        assert isinstance(info, WorktreeInfo), f'Expected WorktreeInfo; got {info!r}'
        # Scrub the seeded (untracked) target/ dir so the only dirty state
        # left is the tracked-file edit below (mirrors the target/-scrub
        # idiom used elsewhere in this module, e.g. β re-seed).
        shutil.rmtree(info.path / 'target', ignore_errors=True)
        (info.path / 'README.md').write_text('uncommitted WIP\n')

        await git_ops._abort_lane_acquisition(info.path, 'X', remove_worktree=False)

        rc, log, _ = await _run(['git', 'log', '--oneline', 'task/X'], cwd=git_repo)
        assert rc == 0, 'task/X must still exist (commit-bearing, retained)'
        assert 'save WIP before lane-acquire abort' in log, (
            f'Expected a WIP-save commit on task/X; got log:\n{log}'
        )

    async def test_abort_retains_commit_bearing_branch(self, git_repo: Path):
        """Even when warm_lane_ref_is_degenerate=True, a commit-bearing branch is retained."""
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        info = await git_ops.acquire_warm_lane('X', start_ref)
        assert isinstance(info, WorktreeInfo), f'Expected WorktreeInfo; got {info!r}'
        shutil.rmtree(info.path / 'target', ignore_errors=True)
        (info.path / 'work.txt').write_text('real work\n')
        await _run(['git', 'add', '-A'], cwd=info.path)
        await _run(['git', 'commit', '-m', 'real work commit'], cwd=info.path)

        with patch.object(
            git_ops, 'warm_lane_ref_is_degenerate', AsyncMock(return_value=True),
        ):
            await git_ops._abort_lane_acquisition(info.path, 'X', remove_worktree=False)

        rc, _, _ = await _run(['git', 'rev-parse', '--verify', 'task/X'], cwd=git_repo)
        assert rc == 0, (
            'Commit-bearing branch must be RETAINED even when flagged degenerate'
        )

    async def test_abort_deletes_degenerate_ref(self, git_repo: Path):
        """0-commits-beyond-main + degenerate=True ⇒ branch is deleted."""
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        info = await git_ops.acquire_warm_lane('X', start_ref)
        assert isinstance(info, WorktreeInfo), f'Expected WorktreeInfo; got {info!r}'
        # No tracked edits and no seeded target/ dirt — task/X sits exactly
        # at the main tip (0 commits beyond main) going into the abort.
        shutil.rmtree(info.path / 'target', ignore_errors=True)

        with patch.object(
            git_ops, 'warm_lane_ref_is_degenerate', AsyncMock(return_value=True),
        ):
            await git_ops._abort_lane_acquisition(info.path, 'X', remove_worktree=False)

        rc, _, _ = await _run(['git', 'rev-parse', '--verify', 'task/X'], cwd=git_repo)
        assert rc != 0, 'Degenerate 0-commits-beyond-main branch must be DELETED'


# ---------------------------------------------------------------------------
# acquire_warm_lane's top-level except — load-bearing PRD fault-injection
# (task-2062 residual: a commit() fault inside the in-memory reuse path must
# not leak a checked-out branch into the FREE pool) (step 7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAcquireWarmLaneReuseFaultRoutesThroughAbort:
    """acquire_warm_lane's top-level except must detach before releasing."""

    async def test_reuse_path_commit_fault_detaches_and_no_collision(
        self, git_repo: Path, caplog,
    ):
        """A commit() fault inside the in-memory reuse path (_reuse_warm_lane
        -> commit() -> RuntimeError) must be caught by acquire_warm_lane's
        top-level except and routed through _abort_lane_acquisition, so the
        lane's HEAD is detached (not left checked out on task/R) before the
        pool slot is freed — otherwise a follow-up acquire can collide with
        'already used by worktree'.
        """
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        # Create-once: seeds the lane and maps 'R' -> lane in the pool.
        info = await git_ops.acquire_warm_lane('R', start_ref)
        assert isinstance(info, WorktreeInfo), f'Expected WorktreeInfo; got {info!r}'

        # Second acquire of the SAME task, no release between: in-memory
        # reuse -> _reuse_warm_lane -> commit() raises -> propagates to the
        # top-level except.
        with patch.object(
            git_ops, 'commit', side_effect=RuntimeError('Commit failed: injected'),
        ):
            result = await git_ops.acquire_warm_lane('R', start_ref)

        assert result is WarmLaneUnavailable.FAULT, (
            f'Expected FAULT from the injected commit() failure; got {result!r}'
        )

        _, branch, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=info.path,
        )
        assert branch.strip() == 'HEAD', (
            f'Expected detached HEAD after the reuse-path fault; got branch={branch.strip()!r}'
        )
        assert git_ops.warm_lane_pool is not None
        assert git_ops.warm_lane_pool.state(info.path) == LaneState.FREE, (
            'Pool slot must be FREE after the fault teardown'
        )

        # Outside the patch: a follow-up acquire must succeed cleanly, with
        # no 'already used by worktree' collision logged.
        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            followup = await git_ops.acquire_warm_lane('R', start_ref)

        assert isinstance(followup, WorktreeInfo), (
            f'Expected the follow-up acquire to succeed; got {followup!r}'
        )
        assert 'already used by worktree' not in caplog.text, (
            f'Follow-up acquire must not collide with a leaked checkout:\n{caplog.text}'
        )
