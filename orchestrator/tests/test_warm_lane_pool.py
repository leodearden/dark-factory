"""Tests for WarmLanePool — pure state-machine tests (no git I/O)
+ GitOps-wiring tests (step-3).

Step-1: RED — WarmLanePool and LaneState are absent; import fails.
Step-3: RED — GitOps pool-wiring / warm_lane_base_target_path absent.
"""

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.warm_lane_pool import LaneState, WarmLanePool


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_pool(tmp_path: Path, size: int = 3) -> WarmLanePool:
    base = tmp_path / 'worktrees'
    base.mkdir(parents=True, exist_ok=True)
    return WarmLanePool(worktree_base=base, size=size)


# ---------------------------------------------------------------------------
# Construction invariants
# ---------------------------------------------------------------------------


class TestWarmLanePoolConstruction:
    def test_size_is_exposed(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=3)
        assert pool.size == 3

    def test_lane_paths_are_correct(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'
        expected = [base / '_lane-0', base / '_lane-1', base / '_lane-2']
        for k, expected_path in enumerate(expected):
            lane = base / f'_lane-{k}'
            assert pool.state(lane) == LaneState.FREE

    def test_all_lanes_start_free(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=4)
        base = tmp_path / 'worktrees'
        for k in range(4):
            assert pool.state(base / f'_lane-{k}') == LaneState.FREE

    def test_zero_size_pool(self, tmp_path: Path):
        """A pool of size 0 is valid — try_acquire always returns None."""
        pool = _make_pool(tmp_path, size=0)
        assert pool.size == 0


# ---------------------------------------------------------------------------
# try_acquire
# ---------------------------------------------------------------------------


class TestTryAcquire:
    def test_acquire_returns_free_lane(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=3)
        lane = asyncio.run(pool.try_acquire())
        assert lane is not None
        base = tmp_path / 'worktrees'
        assert lane.parent == base
        assert lane.name.startswith('_lane-')

    def test_acquire_flips_to_assigned(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=3)
        lane = asyncio.run(pool.try_acquire())
        assert pool.state(lane) == LaneState.ASSIGNED

    def test_acquire_all_then_exhaustion(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=3)
        acquired = [asyncio.run(pool.try_acquire()) for _ in range(3)]
        assert all(p is not None for p in acquired)
        # All 3 assigned — next must return None
        fourth = asyncio.run(pool.try_acquire())
        assert fourth is None

    def test_acquire_returns_distinct_lanes(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=3)
        lanes = [asyncio.run(pool.try_acquire()) for _ in range(3)]
        assert len(set(lanes)) == 3

    def test_acquire_ordered_first_free(self, tmp_path: Path):
        """try_acquire should hand out _lane-0 before _lane-1, etc."""
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'
        lane = asyncio.run(pool.try_acquire())
        assert lane == base / '_lane-0'

    def test_exhausted_pool_returns_none(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=0)
        lane = asyncio.run(pool.try_acquire())
        assert lane is None


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------


class TestRelease:
    def test_release_flips_assigned_to_free(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=3)
        lane = asyncio.run(pool.try_acquire())
        asyncio.run(pool.release(lane))
        assert pool.state(lane) == LaneState.FREE

    def test_release_makes_lane_reacquirable(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=1)
        lane = asyncio.run(pool.try_acquire())
        assert lane is not None
        asyncio.run(pool.release(lane))
        lane2 = asyncio.run(pool.try_acquire())
        assert lane2 == lane

    def test_release_free_lane_is_idempotent(self, tmp_path: Path):
        """Releasing a lane that is already FREE must not raise."""
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'
        free_lane = base / '_lane-0'
        # Already FREE — should not raise
        asyncio.run(pool.release(free_lane))
        assert pool.state(free_lane) == LaneState.FREE

    def test_release_unknown_lane_is_noop(self, tmp_path: Path):
        """Releasing an unknown path must not raise (no-op)."""
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'
        unknown = base / 'task-99'
        asyncio.run(pool.release(unknown))  # must not raise


# ---------------------------------------------------------------------------
# is_lane
# ---------------------------------------------------------------------------


class TestIsLane:
    def test_known_lane_is_true(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'
        for k in range(3):
            assert pool.is_lane(base / f'_lane-{k}') is True

    def test_non_lane_path_is_false(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'
        assert pool.is_lane(base / 'task-7') is False

    def test_non_lane_path_resolved_is_false(self, tmp_path: Path):
        """Resolved symlink paths not in the pool are still False."""
        pool = _make_pool(tmp_path, size=3)
        other = tmp_path / 'other_dir'
        assert pool.is_lane(other) is False

    def test_is_lane_uses_resolved_path(self, tmp_path: Path):
        """is_lane should match by resolved path so symlinks work."""
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'
        lane_0 = base / '_lane-0'
        # Direct path should match
        assert pool.is_lane(lane_0) is True


# ---------------------------------------------------------------------------
# Concurrency — no duplicate allocations under asyncio.gather
# ---------------------------------------------------------------------------


class TestConcurrentAcquire:
    def test_no_duplicate_lanes_under_gather(self, tmp_path: Path):
        """Concurrent try_acquire calls must never hand the same lane to two callers."""
        pool = _make_pool(tmp_path, size=5)

        async def run():
            tasks = [pool.try_acquire() for _ in range(5)]
            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(run())
        non_none = [r for r in results if r is not None]
        # All 5 should have been handed out without duplicates
        assert len(non_none) == 5
        assert len(set(non_none)) == 5

    def test_no_duplicate_when_over_subscribed(self, tmp_path: Path):
        """If more callers than lanes, extras get None; no duplicates in non-None results."""
        pool = _make_pool(tmp_path, size=3)

        async def run():
            tasks = [pool.try_acquire() for _ in range(6)]
            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(run())
        non_none = [r for r in results if r is not None]
        # At most 3 handed out
        assert len(non_none) <= 3
        # No duplicates
        assert len(set(non_none)) == len(non_none)


# ---------------------------------------------------------------------------
# state() accessor for unknown path
# ---------------------------------------------------------------------------


class TestStateAccessor:
    def test_state_unknown_path_returns_none(self, tmp_path: Path):
        """state() returns None for paths not in the pool."""
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'
        assert pool.state(base / 'task-99') is None

    def test_state_returns_correct_value_after_transitions(self, tmp_path: Path):
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'
        lane = base / '_lane-1'
        # Start FREE
        assert pool.state(lane) == LaneState.FREE
        # Acquire _lane-0 then _lane-1
        asyncio.run(pool.try_acquire())  # lane-0
        asyncio.run(pool.try_acquire())  # lane-1
        assert pool.state(lane) == LaneState.ASSIGNED
        # Release
        asyncio.run(pool.release(lane))
        assert pool.state(lane) == LaneState.FREE


# ===========================================================================
# Step-3: RED — GitOps pool-wiring + warm_lane_base_target_path property
# ===========================================================================


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def wl_git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


@pytest.fixture
def wl_git_config_on() -> GitConfig:
    """GitConfig with warm_lane_pool=True."""
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
    )


@pytest.fixture
def wl_git_config_off() -> GitConfig:
    """GitConfig with warm_lane_pool=False (default)."""
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=False,
    )


class TestGitOpsPoolWiring:
    def test_pool_constructed_when_enabled_with_size(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """GitOps(config(warm_lane_pool=True), repo, warm_lane_pool_size=4).warm_lane_pool
        is a WarmLanePool of size 4 with lanes under <worktree_base>."""
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=4)
        assert git_ops.warm_lane_pool is not None
        assert isinstance(git_ops.warm_lane_pool, WarmLanePool)
        assert git_ops.warm_lane_pool.size == 4
        # Lanes live under worktree_base
        worktree_base = (wl_git_repo / '.worktrees').resolve()
        for k in range(4):
            assert git_ops.warm_lane_pool.is_lane(worktree_base / f'_lane-{k}')

    def test_pool_none_when_size_zero(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """warm_lane_pool is None when warm_lane_pool_size=0 (even with knob on)."""
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=0)
        assert git_ops.warm_lane_pool is None

    def test_pool_none_when_knob_off_with_size(
        self, wl_git_repo: Path, wl_git_config_off: GitConfig,
    ):
        """warm_lane_pool is None when warm_lane_pool=False even with size>0."""
        git_ops = GitOps(wl_git_config_off, wl_git_repo, warm_lane_pool_size=4)
        assert git_ops.warm_lane_pool is None

    def test_pool_none_by_default(self, wl_git_repo: Path, wl_git_config_on: GitConfig):
        """GitOps(config, repo) without warm_lane_pool_size → pool is None."""
        git_ops = GitOps(wl_git_config_on, wl_git_repo)
        assert git_ops.warm_lane_pool is None


class TestGitConfigDefaults:
    def test_warm_lane_pool_default_false(self):
        config = GitConfig()
        assert config.warm_lane_pool is False

    def test_warm_lane_base_target_dir_default_none(self):
        config = GitConfig()
        assert config.warm_lane_base_target_dir is None


class TestWarmLaneBaseTargetPath:
    def test_derived_default_from_persistent_merge_worktree(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Without override, warm_lane_base_target_path derives from
        persistent_merge_worktree_path / reap_build_artifact_dirs[0]."""
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=2)
        expected = git_ops.persistent_merge_worktree_path / 'target'
        assert git_ops.warm_lane_base_target_path == expected

    def test_override_from_config(self, wl_git_repo: Path, tmp_path: Path):
        """When warm_lane_base_target_dir is set, it takes precedence."""
        custom_dir = str(tmp_path / 'custom_target')
        config = GitConfig(
            warm_lane_pool=True,
            warm_lane_base_target_dir=custom_dir,
        )
        git_ops = GitOps(config, wl_git_repo, warm_lane_pool_size=2)
        assert git_ops.warm_lane_base_target_path == Path(custom_dir)

    def test_derived_uses_first_reap_dir(self, wl_git_repo: Path):
        """When reap_build_artifact_dirs is overridden, derived path uses [0]."""
        config = GitConfig(
            warm_lane_pool=True,
            reap_build_artifact_dirs=['build', 'dist'],
        )
        git_ops = GitOps(config, wl_git_repo, warm_lane_pool_size=2)
        expected = git_ops.persistent_merge_worktree_path / 'build'
        assert git_ops.warm_lane_base_target_path == expected

    def test_derived_falls_back_to_target_when_no_reap_dirs(
        self, wl_git_repo: Path,
    ):
        """Empty reap_build_artifact_dirs → derived path uses 'target' as fallback."""
        config = GitConfig(
            warm_lane_pool=True,
            reap_build_artifact_dirs=[],
        )
        git_ops = GitOps(config, wl_git_repo, warm_lane_pool_size=2)
        expected = git_ops.persistent_merge_worktree_path / 'target'
        assert git_ops.warm_lane_base_target_path == expected
