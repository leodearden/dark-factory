"""Tests for GitOps <-> LaneLifecycle integration (W11 gamma):

- acquire_warm_lane / release_warm_lane durable ASSIGNED/RELEASED writes
- the .pool-root sentinel fold (GitOps delegates to LaneLifecycle)
- the .task -> .task-meta relocations (disk-backstop plan.json,
  interactive.json stamp)

Self-contained, mirroring test_warm_lane_abort_teardown.py: conftest.py
provides no shared `git_repo` fixture, so this module owns its own fixture +
stub scripts rather than depending on test_git_ops.py's module-level helpers.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import (
    GitOps,
    WorktreeInfo,
    _run,
)
from orchestrator.lane_lifecycle import LaneState

# ---------------------------------------------------------------------------
# Repo fixture (mirrors test_warm_lane_abort_teardown.py)
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
    and marks `.worktrees/.pool-root` present (task 2099 create-once guard)
    — mirrors test_warm_lane_abort_teardown.py's git_repo fixture.
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


# ---------------------------------------------------------------------------
# acquire_warm_lane -> durable ASSIGNED record
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAcquireWarmLaneWritesAssignedRecord:
    """acquire_warm_lane must write a durable ASSIGNED LaneLifecycle record."""

    async def test_create_once_acquire_writes_assigned_record(self, git_repo: Path):
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        wt = await git_ops.acquire_warm_lane('321', start_ref, expected_title='Fix X')

        assert isinstance(wt, WorktreeInfo), f'Expected WorktreeInfo; got {wt!r}'
        record = git_ops._lane_lifecycle.read(wt.path)
        assert record is not None, 'expected a durable record after acquire'
        assert record.state is LaneState.ASSIGNED, (
            f'expected ASSIGNED, got {record.state!r}'
        )
        assert record.task_id == '321', f'expected task_id==321, got {record.task_id!r}'
        assert record.title == 'Fix X', f'expected title=="Fix X", got {record.title!r}'
        assert record.branch == 'task/321', (
            f'expected branch=="task/321", got {record.branch!r}'
        )

        record_path = git_ops.worktree_base / '.lane-state' / f'{wt.path.name}.json'
        assert record_path.is_file(), (
            f'expected a durable record file to exist at {record_path}'
        )


# ---------------------------------------------------------------------------
# release_warm_lane -> durable RELEASED record
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReleaseWarmLaneWritesReleasedRecord:
    """release_warm_lane must write a durable RELEASED LaneLifecycle record."""

    async def test_release_after_acquire_writes_released_record(self, git_repo: Path):
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        wt = await git_ops.acquire_warm_lane('321', start_ref, expected_title='Fix X')
        assert isinstance(wt, WorktreeInfo), f'Expected WorktreeInfo; got {wt!r}'

        await git_ops.release_warm_lane(wt.path, '321')

        record = git_ops._lane_lifecycle.read(wt.path)
        assert record is not None, 'expected a durable record after release'
        assert record.state is LaneState.RELEASED, (
            f'expected RELEASED, got {record.state!r}'
        )
        assert record.task_id is None, f'expected task_id cleared, got {record.task_id!r}'
        assert record.title is None, f'expected title cleared, got {record.title!r}'

    async def test_release_already_released_lane_is_a_noop(self, git_repo: Path):
        """A second release_warm_lane call on an already-RELEASED lane must
        not raise. RELEASED -> RELEASED is not a legal edge, so
        _lifecycle_note_released must no-op rather than attempt it."""
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        wt = await git_ops.acquire_warm_lane('321', start_ref)
        assert isinstance(wt, WorktreeInfo), f'Expected WorktreeInfo; got {wt!r}'
        await git_ops.release_warm_lane(wt.path, '321')
        before = git_ops._lane_lifecycle.read(wt.path)
        assert before is not None and before.state is LaneState.RELEASED

        await git_ops.release_warm_lane(wt.path, '321')  # must not raise

        after = git_ops._lane_lifecycle.read(wt.path)
        assert after is not None
        assert after.state is LaneState.RELEASED, (
            f'expected RELEASED to remain a legal no-op state, got {after.state!r}'
        )
