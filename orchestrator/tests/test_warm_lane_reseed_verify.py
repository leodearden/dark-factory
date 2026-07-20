"""Warm-lane reseed-contamination verification (task 2854).

A fresh-reseed warm-lane acquire (the ``RECYCLE`` / ``CREATE_ONCE_FRESH``
routes) must GUARANTEE + VERIFY a clean reseed — the lane's checked-out
branch is at the base with ZERO retained prior-occupant commits — BEFORE the
lane is handed out for dispatch. Reify incident 2026-07-20: ``_lane-12`` was
acquired for task 5279 but its ``task/5279`` branch still sat at task 5264's
commits; the rebase-collapse ``BranchResetError`` guard caught it downstream,
but a lane serving another task's tree is a reseed-consistency defect that
must be faulted at acquire time (re-acquire a DIFFERENT lane) rather than
relied on the collapse guard to catch late.

Self-contained, mirroring ``test_lane_lifecycle_gitops.py`` /
``test_warm_lane_abort_teardown.py``: ``conftest.py`` provides no shared
``git_repo`` fixture, so this module owns its own real-git-repo fixture +
stub warm-lane scripts (copied verbatim from ``test_lane_lifecycle_gitops.py``)
rather than depending on another test module's module-level helpers.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import (
    GitOps,
    _run,
)

# ---------------------------------------------------------------------------
# Repo fixture (copied from test_lane_lifecycle_gitops.py — self-contained)
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

    Pre-creates the default derived warm-lane base (task 2061 gate) so the
    acquire_warm_lane pre-acquire base-health gate sees WarmBaseHealth.OK,
    and marks ``.worktrees/.pool-root`` present (task 2099 create-once guard)
    — mirrors test_lane_lifecycle_gitops.py's git_repo fixture.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    default_base = repo / '.worktrees' / '_merge-verify' / 'target'
    default_base.mkdir(parents=True, exist_ok=True)
    (default_base / '.keep').write_text('warm base sentinel\n')
    (repo / '.worktrees' / '.pool-root').touch()
    return repo


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


async def _add_lane_worktree(repo: Path, lane: Path, branch: str, base: str) -> None:
    """Create a lane worktree checked out on *branch* at *base*."""
    rc, _, err = await _run(
        ['git', 'worktree', 'add', '-b', branch, str(lane), base], cwd=repo,
    )
    assert rc == 0, f'git worktree add failed (rc={rc}): {err}'


# ---------------------------------------------------------------------------
# step-1: RED — GitOps._reseed_verified_clean predicate (not yet implemented)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReseedVerifiedCleanPredicate:
    """``_reseed_verified_clean(lane, full_branch, base_ref)`` returns True iff
    the lane's HEAD is on *full_branch* AND carries ZERO commits beyond
    *base_ref*. Fail-closed: any ambiguity (detached HEAD, wrong branch, git
    error) returns False so a lane that cannot be PROVEN clean is never
    dispatched onto."""

    async def test_clean_reseed_at_base_returns_true(
        self, git_repo: Path, tmp_path: Path,
    ):
        base = await _get_head(git_repo)
        lane = tmp_path / 'lane_clean'
        await _add_lane_worktree(git_repo, lane, 'task/NEW', base)
        git_ops = GitOps(_warm_config(), git_repo)

        assert await git_ops._reseed_verified_clean(lane, 'task/NEW', base) is True

    async def test_one_commit_beyond_base_returns_false(
        self, git_repo: Path, tmp_path: Path,
    ):
        """The incident shape: the lane's branch carries one retained commit
        beyond the base it was supposed to be reset to."""
        base = await _get_head(git_repo)
        lane = tmp_path / 'lane_dirty'
        await _add_lane_worktree(git_repo, lane, 'task/NEW', base)
        # One extra commit on task/NEW in the lane → HEAD is 1 beyond base.
        (lane / 'retained.txt').write_text('prior occupant work\n')
        await _run(['git', 'add', '-A'], cwd=lane)
        await _run(['git', 'commit', '-m', 'retained prior commit'], cwd=lane)
        git_ops = GitOps(_warm_config(), git_repo)

        assert await git_ops._reseed_verified_clean(lane, 'task/NEW', base) is False

    async def test_detached_head_returns_false(
        self, git_repo: Path, tmp_path: Path,
    ):
        base = await _get_head(git_repo)
        lane = tmp_path / 'lane_detached'
        await _add_lane_worktree(git_repo, lane, 'task/NEW', base)
        await _run(['git', 'checkout', '--detach'], cwd=lane)
        git_ops = GitOps(_warm_config(), git_repo)

        assert await git_ops._reseed_verified_clean(lane, 'task/NEW', base) is False

    async def test_different_branch_returns_false(
        self, git_repo: Path, tmp_path: Path,
    ):
        """A lane checked out on a branch OTHER than the one the reseed was
        supposed to switch to is not a verified clean reseed."""
        base = await _get_head(git_repo)
        lane = tmp_path / 'lane_other_branch'
        await _add_lane_worktree(git_repo, lane, 'task/OTHER', base)
        git_ops = GitOps(_warm_config(), git_repo)

        assert await git_ops._reseed_verified_clean(lane, 'task/NEW', base) is False

    async def test_bogus_base_ref_fails_closed_false(
        self, git_repo: Path, tmp_path: Path,
    ):
        """Fail-closed: an unresolvable base_ref makes the rev-list error, and
        an error must return False (cannot prove clean → do not dispatch),
        never fail-open to True."""
        base = await _get_head(git_repo)
        lane = tmp_path / 'lane_bogus_base'
        await _add_lane_worktree(git_repo, lane, 'task/NEW', base)
        git_ops = GitOps(_warm_config(), git_repo)

        assert await git_ops._reseed_verified_clean(
            lane, 'task/NEW', 'nonexistent-ref-deadbeefdeadbeef',
        ) is False
