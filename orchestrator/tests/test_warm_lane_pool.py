"""Tests for WarmLanePool — pure state-machine tests (no git I/O)
+ GitOps-wiring tests (step-3)
+ _seed_warm_lane tests (step-5).

Step-1: RED — WarmLanePool and LaneState are absent; import fails.
Step-3: RED — GitOps pool-wiring / warm_lane_base_target_path absent.
Step-5: RED — GitOps._seed_warm_lane absent.
"""

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, WorktreeInfo, _run
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
        for k, _expected_path in enumerate(expected):
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
        assert lane is not None
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
        assert lane is not None
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


@pytest.fixture
def wl_git_config_rolling_base(tmp_path: Path) -> GitConfig:
    """GitConfig with warm_lane_base_target_dir pointing to a SEPARATE rolling-base.

    The rolling base dir is ``tmp_path/rolling_base``.  Since
    ``advancing`` (= ``<worktree_base>/_merge-verify/target``) differs from
    ``base`` (= ``tmp_path/rolling_base``), ``refresh_warm_base()`` will not
    short-circuit the advancing == base guard and the refresh actually fires.

    Tests that need the rolling-base path can compute it as
    ``tmp_path / 'rolling_base'``.
    """
    rolling = tmp_path / 'rolling_base'
    rolling.mkdir(parents=True, exist_ok=True)
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
        warm_lane_base_target_dir=str(rolling),
    )


# ---------------------------------------------------------------------------
# P1 helpers: fake refresh-warm-base.sh recorder scripts
# ---------------------------------------------------------------------------


def _install_refresh_recorder(scripts_dir: Path, recorder_file: Path) -> None:
    """Install a refresh-warm-base.sh that records its args ($1 advancing, $2 base)
    to *recorder_file* and exits 0.

    Mirrors the fake seed-warm-lane.sh pattern used by TestSeedWarmLane.
    """
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / 'refresh-warm-base.sh'
    script.write_text(
        f'#!/usr/bin/env bash\necho "$1 $2" >> {recorder_file}\n'
    )
    script.chmod(0o755)


def _install_refresh_failing(scripts_dir: Path) -> None:
    """Install a refresh-warm-base.sh that exits non-zero (fail-soft test variant)."""
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / 'refresh-warm-base.sh'
    script.write_text('#!/usr/bin/env bash\necho "refresh failed" >&2\nexit 1\n')
    script.chmod(0o755)


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


# ===========================================================================
# Step-5: RED — GitOps._seed_warm_lane
# ===========================================================================


async def _make_lane_dir(repo: Path, lane_name: str = '_lane-0') -> Path:
    """Create a real worktree at <repo>/.worktrees/<lane_name> for seeding tests."""
    worktree_base = repo / '.worktrees'
    worktree_base.mkdir(parents=True, exist_ok=True)
    lane = worktree_base / lane_name
    # Use git worktree add to create the lane as a real registered worktree
    await _run(
        ['git', 'worktree', 'add', '-b', f'task/{lane_name}', str(lane), 'HEAD'],
        cwd=repo,
    )
    return lane


@pytest.mark.asyncio
class TestSeedWarmLane:
    async def test_seed_calls_script_with_correct_args(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """_seed_warm_lane invokes seed-warm-lane.sh with <base_target> <lane> <mode>."""
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        # Create a real lane worktree so the script can live inside it
        lane = await _make_lane_dir(wl_git_repo, '_lane-0')

        # Write a stub seed-warm-lane.sh into the lane's scripts dir.
        # The stub appends its argv (space-joined) to a marker file.
        scripts_dir = lane / 'scripts'
        scripts_dir.mkdir(parents=True, exist_ok=True)
        marker = lane / 'seed_args.txt'
        script = scripts_dir / 'seed-warm-lane.sh'
        script.write_text(
            f'#!/usr/bin/env bash\necho "$@" >> {marker}\n'
        )
        script.chmod(0o755)

        result = await git_ops._seed_warm_lane(lane, '--fresh-checkout')

        assert result is True
        assert marker.exists(), 'seed-warm-lane.sh was not called (marker missing)'
        recorded = marker.read_text().strip()
        base_target = str(git_ops.warm_lane_base_target_path)
        expected = f'{base_target} {lane} --fresh-checkout'
        assert recorded == expected, (
            f'Wrong args: got {recorded!r}, expected {expected!r}'
        )

    async def test_seed_absent_script_returns_false(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Absent seed-warm-lane.sh → returns False (no warm capability), never raises."""
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        lane = await _make_lane_dir(wl_git_repo, '_lane-abs')
        # No script installed in lane/scripts/
        result = await git_ops._seed_warm_lane(lane, '--fresh-checkout')
        assert result is False

    async def test_seed_nonzero_exit_returns_false(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Script that exits non-zero → returns False (fail-soft), never raises."""
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        lane = await _make_lane_dir(wl_git_repo, '_lane-fail')

        scripts_dir = lane / 'scripts'
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script = scripts_dir / 'seed-warm-lane.sh'
        script.write_text('#!/usr/bin/env bash\necho "failure" >&2\nexit 1\n')
        script.chmod(0o755)

        result = await git_ops._seed_warm_lane(lane, '--fresh-checkout')
        assert result is False

    async def test_seed_reset_in_place_mode_passes_correct_flag(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """_seed_warm_lane with '--reset-in-place' passes the correct mode flag."""
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        lane = await _make_lane_dir(wl_git_repo, '_lane-rip')

        scripts_dir = lane / 'scripts'
        scripts_dir.mkdir(parents=True, exist_ok=True)
        marker = lane / 'seed_args_rip.txt'
        script = scripts_dir / 'seed-warm-lane.sh'
        script.write_text(
            f'#!/usr/bin/env bash\necho "$@" >> {marker}\n'
        )
        script.chmod(0o755)

        result = await git_ops._seed_warm_lane(lane, '--reset-in-place')
        assert result is True
        recorded = marker.read_text().strip()
        assert recorded.endswith('--reset-in-place')


# ===========================================================================
# Step-7: RED — GitOps.acquire_warm_lane (create-once / first-time)
# ===========================================================================


async def _add_seed_and_debug_port_scripts(repo: Path, port: int = 39411) -> Path:
    """Commit stub seed-warm-lane.sh + setup-worktree-debug-port.sh into repo main.

    The seed script creates <lane>/target/seeded.bin (simulating a CoW copy)
    and the debug-port script just echoes <port>.
    """
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)

    seed_script = scripts_dir / 'seed-warm-lane.sh'
    seed_script.write_text(
        '#!/usr/bin/env bash\n'
        '# argv: <base_target> <lane_dir> <mode>\n'
        'LANE_DIR="$2"\n'
        'mkdir -p "$LANE_DIR/target"\n'
        'echo "seeded" > "$LANE_DIR/target/seeded.bin"\n'
    )
    seed_script.chmod(0o755)

    debug_script = scripts_dir / 'setup-worktree-debug-port.sh'
    debug_script.write_text(f'#!/usr/bin/env bash\necho {port}\n')
    debug_script.chmod(0o755)

    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add stub seed + debug-port scripts'], cwd=repo)
    return scripts_dir


@pytest.mark.asyncio
class TestAcquireWarmLaneCreateOnce:
    """acquire_warm_lane creates a new worktree+branch and seeds it on first call."""

    async def test_acquire_returns_worktree_info(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """acquire_warm_lane returns WorktreeInfo with lane path, base_commit, debug port."""
        from orchestrator.git_ops import WorktreeInfo
        await _add_seed_and_debug_port_scripts(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=2)

        # Get start_ref (main HEAD)
        _, start_ref, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wl_git_repo)
        start_ref = start_ref.strip()

        info = await git_ops.acquire_warm_lane('task-A', start_ref)

        assert info is not None
        assert isinstance(info, WorktreeInfo)

    async def test_acquire_lane_path_is_pool_lane(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Returned WorktreeInfo.path is _lane-0 (first free lane), not a branch-named dir."""
        await _add_seed_and_debug_port_scripts(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=2)
        _, start_ref, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wl_git_repo)
        start_ref = start_ref.strip()

        info = await git_ops.acquire_warm_lane('task-A', start_ref)

        assert info is not None
        expected_lane = git_ops.worktree_base / '_lane-0'
        assert info.path == expected_lane

    async def test_acquire_lane_is_registered_worktree(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """The lane is a registered git worktree after acquire."""
        await _add_seed_and_debug_port_scripts(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=2)
        _, start_ref, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wl_git_repo)
        start_ref = start_ref.strip()

        info = await git_ops.acquire_warm_lane('task-A', start_ref)
        assert info is not None
        assert await git_ops._is_registered_worktree(info.path)

    async def test_acquire_lane_on_correct_branch(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """The lane's HEAD is on task/task-A at the start_ref SHA."""
        await _add_seed_and_debug_port_scripts(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=2)
        _, start_ref, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wl_git_repo)
        start_ref = start_ref.strip()

        info = await git_ops.acquire_warm_lane('task-A', start_ref)
        assert info is not None

        # Check HEAD SHA matches start_ref
        _, head_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=info.path)
        assert head_sha.strip() == start_ref

        # Check branch name
        _, branch, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=info.path,
        )
        assert branch.strip() == 'task/task-A'

    async def test_acquire_lane_has_base_commit(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """WorktreeInfo.base_commit is set (40-char SHA)."""
        await _add_seed_and_debug_port_scripts(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=2)
        _, start_ref, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wl_git_repo)
        start_ref = start_ref.strip()

        info = await git_ops.acquire_warm_lane('task-A', start_ref)
        assert info is not None
        assert len(info.base_commit) == 40

    async def test_acquire_provides_debug_port(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Debug-port re-provision ran: WorktreeInfo.reify_debug_port == 39411."""
        await _add_seed_and_debug_port_scripts(wl_git_repo, port=39411)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=2)
        _, start_ref, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wl_git_repo)
        start_ref = start_ref.strip()

        info = await git_ops.acquire_warm_lane('task-A', start_ref)
        assert info is not None
        assert info.reify_debug_port == 39411

    async def test_acquire_seed_invoked_with_fresh_checkout(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """seed-warm-lane.sh is called with --fresh-checkout on first acquire."""
        # Replace seed script with one that logs its mode arg
        scripts_dir = wl_git_repo / 'scripts'
        scripts_dir.mkdir(parents=True, exist_ok=True)
        mode_log = wl_git_repo / 'seed_mode.txt'
        seed_script = scripts_dir / 'seed-warm-lane.sh'
        seed_script.write_text(
            f'#!/usr/bin/env bash\n'
            f'echo "$3" >> {mode_log}\n'
            f'mkdir -p "$2/target"\n'
            f'echo seeded > "$2/target/seeded.bin"\n'
        )
        seed_script.chmod(0o755)
        # Add debug-port script
        debug_script = scripts_dir / 'setup-worktree-debug-port.sh'
        debug_script.write_text('#!/usr/bin/env bash\necho 39411\n')
        debug_script.chmod(0o755)
        await _run(['git', 'add', '-A'], cwd=wl_git_repo)
        await _run(['git', 'commit', '-m', 'add mode-logging scripts'], cwd=wl_git_repo)

        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=2)
        _, start_ref, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wl_git_repo)
        start_ref = start_ref.strip()

        info = await git_ops.acquire_warm_lane('task-A', start_ref)
        assert info is not None
        assert mode_log.exists(), 'seed script was not called'
        assert mode_log.read_text().strip() == '--fresh-checkout'

    async def test_acquire_target_dir_is_warm(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """The lane's target/seeded.bin exists (CoW seed ran)."""
        await _add_seed_and_debug_port_scripts(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=2)
        _, start_ref, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wl_git_repo)
        start_ref = start_ref.strip()

        info = await git_ops.acquire_warm_lane('task-A', start_ref)
        assert info is not None
        assert (info.path / 'target' / 'seeded.bin').exists()

    async def test_acquire_marks_pool_lane_assigned(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """After acquire, the pool marks _lane-0 ASSIGNED."""
        from orchestrator.warm_lane_pool import LaneState
        await _add_seed_and_debug_port_scripts(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=2)
        _, start_ref, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wl_git_repo)
        start_ref = start_ref.strip()

        info = await git_ops.acquire_warm_lane('task-A', start_ref)
        assert info is not None
        assert git_ops.warm_lane_pool is not None
        assert git_ops.warm_lane_pool.state(info.path) == LaneState.ASSIGNED

    async def test_acquire_exhausted_returns_none(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Pool exhaustion: acquire returns None (caller should cold-fallback)."""
        await _add_seed_and_debug_port_scripts(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        _, start_ref, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wl_git_repo)
        start_ref = start_ref.strip()

        # Exhaust the pool
        info1 = await git_ops.acquire_warm_lane('task-A', start_ref)
        assert info1 is not None

        # Next acquire should return None (exhausted)
        info2 = await git_ops.acquire_warm_lane('task-B', start_ref)
        assert info2 is None

    async def test_acquire_absent_seed_returns_none(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """No seed-warm-lane.sh → acquire returns None (cold-fallback signal)."""
        # No scripts committed — seed will fail → acquire returns None
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=2)
        _, start_ref, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wl_git_repo)
        start_ref = start_ref.strip()

        info = await git_ops.acquire_warm_lane('task-A', start_ref)
        assert info is None


# ===========================================================================
# Step-9: RED — reset-in-place reuse (reset-determinism + warmth retention)
# ===========================================================================


@pytest.mark.asyncio
class TestAcquireWarmLaneResetInPlace:
    """Second acquire on an already-registered lane: reset-in-place, no re-seed."""

    async def _make_two_commits(self, repo: Path) -> tuple[str, str]:
        """Make two distinct commits in repo; return (sha_A, sha_B)."""
        # Commit A: add file_a.txt
        (repo / 'file_a.txt').write_text('version A\n')
        await _run(['git', 'add', '-A'], cwd=repo)
        await _run(['git', 'commit', '-m', 'commit A'], cwd=repo)
        _, sha_a, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)

        # Commit B: replace file_a.txt with different content
        (repo / 'file_a.txt').write_text('version B\n')
        await _run(['git', 'add', '-A'], cwd=repo)
        await _run(['git', 'commit', '-m', 'commit B'], cwd=repo)
        _, sha_b, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
        return sha_a.strip(), sha_b.strip()

    async def _make_repo_with_scripts_and_two_commits(
        self, repo: Path,
    ) -> tuple[str, str]:
        """Add seed+debug scripts, then make two commits. Returns (sha_A, sha_B)."""
        # Add a seed script that logs calls (not creates target/ — warmth test uses
        # manually written target/ file to simulate a prior warm build)
        scripts_dir = repo / 'scripts'
        scripts_dir.mkdir(parents=True, exist_ok=True)
        seed_marker = repo / 'seed_calls.txt'
        seed_script = scripts_dir / 'seed-warm-lane.sh'
        seed_script.write_text(
            f'#!/usr/bin/env bash\n'
            f'echo "called:$3" >> {seed_marker}\n'
            f'mkdir -p "$2/target"\n'
            f'echo "seeded" > "$2/target/seeded.bin"\n'
        )
        seed_script.chmod(0o755)
        debug_script = scripts_dir / 'setup-worktree-debug-port.sh'
        debug_script.write_text('#!/usr/bin/env bash\necho 39411\n')
        debug_script.chmod(0o755)
        await _run(['git', 'add', '-A'], cwd=repo)
        await _run(['git', 'commit', '-m', 'add scripts'], cwd=repo)

        return await self._make_two_commits(repo)

    async def test_reacquire_returns_same_lane(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Second acquire of a released lane returns the same _lane-0 path."""
        sha_a, sha_b = await self._make_repo_with_scripts_and_two_commits(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info_a = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert info_a is not None
        assert git_ops.warm_lane_pool is not None
        # Release the lane (mark FREE directly — release_warm_lane comes in step-12)
        await git_ops.warm_lane_pool.release(info_a.path)

        info_b = await git_ops.acquire_warm_lane('task-B', sha_b)
        assert info_b is not None
        assert info_b.path == info_a.path  # same _lane-0

    async def test_reacquire_head_is_at_new_commit(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """After reset-in-place, HEAD is at commit B (different from A)."""
        sha_a, sha_b = await self._make_repo_with_scripts_and_two_commits(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info_a = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert info_a is not None
        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(info_a.path)

        info_b = await git_ops.acquire_warm_lane('task-B', sha_b)
        assert info_b is not None

        _, head_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=info_b.path)
        assert head_sha.strip() == sha_b

    async def test_reacquire_source_tree_is_clean(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Source tree is bit-identical to a fresh checkout: stray.txt gone, git status clean."""
        sha_a, sha_b = await self._make_repo_with_scripts_and_two_commits(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info_a = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert info_a is not None

        # Simulate stray work: untracked file and modification of tracked file
        (info_a.path / 'stray.txt').write_text('should be gone\n')
        (info_a.path / 'file_a.txt').write_text('dirty modification\n')
        # Warm artifact must survive
        (info_a.path / 'target').mkdir(exist_ok=True)
        (info_a.path / 'target' / 'cache.bin').write_bytes(b'\x00' * 128)

        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(info_a.path)
        info_b = await git_ops.acquire_warm_lane('task-B', sha_b)
        assert info_b is not None

        # stray.txt must be gone
        assert not (info_b.path / 'stray.txt').exists()

        # git status must be clean (only target/ excluded)
        _, status_out, _ = await _run(
            ['git', 'status', '--porcelain'], cwd=info_b.path,
        )
        # strip target/-related lines from status (target/ is excluded from clean)
        non_target_lines = [
            line for line in status_out.splitlines()
            if not line.strip().startswith('?? target/')
            and not line.strip().startswith('?? .task/')
        ]
        assert non_target_lines == [], f'Unexpected dirty files: {non_target_lines}'

    async def test_reacquire_retains_target_warmth(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Recycled lane's target/ is RE-SEEDED on the recycle path (D10).

        Before D10 this test asserted silent retention of target/cache.bin.
        D10 changes the contract: target/ is always re-seeded from the current
        base on a fresh-recycle acquire, so the assertion is that seed WAS
        invoked (seed_calls.txt gained a new entry), not just that a prior
        artifact survived silently.
        """
        sha_a, sha_b = await self._make_repo_with_scripts_and_two_commits(wl_git_repo)
        seed_marker = wl_git_repo / 'seed_calls.txt'
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info_a = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert info_a is not None
        calls_after_first = (
            seed_marker.read_text() if seed_marker.exists() else ''
        ).splitlines()

        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(info_a.path)
        info_b = await git_ops.acquire_warm_lane('task-B', sha_b)
        assert info_b is not None

        calls_after_second = (
            seed_marker.read_text() if seed_marker.exists() else ''
        ).splitlines()
        # D10: seed MUST have been called on the recycle path
        assert len(calls_after_second) > len(calls_after_first), (
            'seed-warm-lane.sh was NOT called on recycle (D10 violation: no re-seed)'
        )
        # The recycle call must be with --fresh-checkout mode
        assert any('--fresh-checkout' in ln for ln in calls_after_second), (
            f'No --fresh-checkout call found in seed log: {calls_after_second}'
        )

    async def test_reacquire_calls_seed_fresh_checkout(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Second acquire for a DIFFERENT task on a recycled FREE lane MUST invoke
        _seed_warm_lane with mode '--fresh-checkout' (D10 always-re-seed-at-acquire).

        Before D10 this test was named test_reacquire_does_not_call_seed and
        asserted the opposite (no call on warm reacquire).  D10 inverts the
        contract: the fresh-recycle path always re-seeds, so seed_calls.txt
        must gain a new '--fresh-checkout' entry after the second acquire.
        """
        sha_a, sha_b = await self._make_repo_with_scripts_and_two_commits(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        seed_marker = wl_git_repo / 'seed_calls.txt'

        info_a = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert info_a is not None
        # Record marker state after first (create-once) acquire
        calls_after_first = seed_marker.read_text() if seed_marker.exists() else ''

        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(info_a.path)
        info_b = await git_ops.acquire_warm_lane('task-B', sha_b)
        assert info_b is not None

        calls_after_second = seed_marker.read_text() if seed_marker.exists() else ''
        # D10: seed MUST be called on the recycle path (new line in marker)
        assert calls_after_second != calls_after_first, (
            'seed-warm-lane.sh was NOT called on recycle (D10 violation: no re-seed)'
        )
        # The recycle call must use --fresh-checkout mode
        new_calls = calls_after_second[len(calls_after_first):]
        assert '--fresh-checkout' in new_calls, (
            f'Recycle seed call did not use --fresh-checkout: {new_calls!r}'
        )

    async def test_reacquire_single_worktree_registration(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Exactly one _lane-0 registration in git worktree list after reacquire."""
        sha_a, sha_b = await self._make_repo_with_scripts_and_two_commits(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info_a = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert info_a is not None
        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(info_a.path)

        info_b = await git_ops.acquire_warm_lane('task-B', sha_b)
        assert info_b is not None

        _, wt_list, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'], cwd=wl_git_repo,
        )
        lane_registrations = [
            line for line in wt_list.splitlines()
            if line.startswith('worktree ') and '_lane-0' in line
        ]
        assert len(lane_registrations) == 1

    async def test_recycle_reseed_failure_keeps_lane(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Recycle path: failed re-seed is fail-soft — lane kept with degraded warmth.

        When seed-warm-lane.sh exits non-zero on the RECYCLE path,
        acquire_warm_lane must still return a usable WorktreeInfo (lane remains
        registered, worktree NOT removed).  Contrast with create-once where a
        seed failure triggers cold-fallback (returns None).

        Fails against step-2 impl (which treats recycle seed failure as fatal,
        returning None).  Passes after step-4's fail-soft implementation.
        """
        sha_a, sha_b = await self._make_repo_with_scripts_and_two_commits(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        # First acquire (create-once): passing seed → lane registered, target/ warm
        info_a = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert info_a is not None, 'prerequisite: create-once acquire must succeed'

        # Place a retained artifact so we can verify it survives fail-soft recycle
        (info_a.path / 'target').mkdir(exist_ok=True)
        retained_artifact = info_a.path / 'target' / 'prior.bin'
        retained_artifact.write_bytes(b'\xca\xfe' * 32)

        # Overwrite the seed script in the MAIN REPO to exit 1 and commit it.
        # After _reset_warm_lane resets the lane to this commit, the lane's
        # seed script will exit non-zero.
        fail_script = wl_git_repo / 'scripts' / 'seed-warm-lane.sh'
        fail_script.write_text(
            '#!/usr/bin/env bash\necho "forced failure" >&2\nexit 1\n'
        )
        fail_script.chmod(0o755)
        await _run(['git', 'add', '-A'], cwd=wl_git_repo)
        await _run(['git', 'commit', '-m', 'fail: seed script exits 1'], cwd=wl_git_repo)
        _, sha_fail, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wl_git_repo)
        sha_fail = sha_fail.strip()

        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(info_a.path)

        # Second acquire for a DIFFERENT task at sha_fail: seed exits 1.
        # step-2 (fatal): returns None.  step-4 (fail-soft): WorktreeInfo.
        info_b = await git_ops.acquire_warm_lane('task-B', sha_fail)

        # D10 fail-soft: must return a usable lane (NOT None / cold-fallback)
        assert info_b is not None, (
            'Recycle seed failure must be fail-soft: return WorktreeInfo, not None'
        )
        # Worktree must NOT be removed (the retained target/ survives)
        assert await git_ops._is_registered_worktree(info_b.path), (
            'Worktree must not be removed on recycle seed failure'
        )
        # Same recycled lane path
        assert info_b.path == info_a.path


# ===========================================================================
# Step-11: RED — GitOps.release_warm_lane
# ===========================================================================


@pytest.mark.asyncio
class TestReleaseWarmLane:
    """release_warm_lane: FREE pool, delete branch, retain worktree + target/."""

    async def _acquire_lane_a(
        self, repo: Path, config: GitConfig,
    ) -> tuple[GitOps, WorktreeInfo]:
        """Acquire _lane-0 for task-A; return (git_ops, info)."""
        scripts_dir = repo / 'scripts'
        scripts_dir.mkdir(parents=True, exist_ok=True)
        seed_script = scripts_dir / 'seed-warm-lane.sh'
        seed_script.write_text(
            '#!/usr/bin/env bash\nmkdir -p "$2/target"\necho seeded > "$2/target/seeded.bin"\n'
        )
        seed_script.chmod(0o755)
        debug_script = scripts_dir / 'setup-worktree-debug-port.sh'
        debug_script.write_text('#!/usr/bin/env bash\necho 39411\n')
        debug_script.chmod(0o755)
        await _run(['git', 'add', '-A'], cwd=repo)
        await _run(['git', 'commit', '-m', 'add scripts'], cwd=repo)

        git_ops = GitOps(config, repo, warm_lane_pool_size=1)
        _, start_ref, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
        start_ref = start_ref.strip()

        info = await git_ops.acquire_warm_lane('task-A', start_ref)
        assert info is not None
        return git_ops, info

    async def test_release_marks_pool_free(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """After release_warm_lane, pool state for _lane-0 is FREE."""
        from orchestrator.warm_lane_pool import LaneState
        git_ops, info = await self._acquire_lane_a(wl_git_repo, wl_git_config_on)
        assert git_ops.warm_lane_pool is not None
        assert git_ops.warm_lane_pool.state(info.path) == LaneState.ASSIGNED

        await git_ops.release_warm_lane(info.path, 'task-A')

        assert git_ops.warm_lane_pool.state(info.path) == LaneState.FREE

    async def test_release_deletes_task_branch(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Branch task/task-A is deleted after release."""
        git_ops, info = await self._acquire_lane_a(wl_git_repo, wl_git_config_on)

        await git_ops.release_warm_lane(info.path, 'task-A')

        rc, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/task-A'], cwd=wl_git_repo,
        )
        assert rc != 0, 'Branch task/task-A should be deleted after release'

    async def test_release_lane_remains_registered(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """_lane-0 is still a registered git worktree after release."""
        git_ops, info = await self._acquire_lane_a(wl_git_repo, wl_git_config_on)
        lane = info.path

        await git_ops.release_warm_lane(lane, 'task-A')

        assert await git_ops._is_registered_worktree(lane), (
            '_lane-0 must remain registered after release (not removed)'
        )

    async def test_release_retains_target_dir(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """target/ is NOT deleted by release (incidental; next acquire re-seeds).

        D10: release leaves target/ in place incidentally (CoW-cheap, harmless).
        target/ retention is no longer a warmth-contract promise — the next
        acquire_warm_lane always re-seeds from the current base, so the
        released lane's target/ drift is irrelevant.  This test is a regression
        guard: release must not actively delete target/.
        """
        git_ops, info = await self._acquire_lane_a(wl_git_repo, wl_git_config_on)
        target_file = info.path / 'target' / 'seeded.bin'
        assert target_file.exists(), 'prerequisite: seed.bin should exist after acquire'

        await git_ops.release_warm_lane(info.path, 'task-A')

        assert (info.path / 'target').exists(), 'target/ must not be deleted by release'
        assert target_file.exists(), 'target/seeded.bin must not be deleted by release'

    async def test_release_makes_lane_reacquirable(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """After release, _lane-0 can be acquired again."""
        git_ops, info = await self._acquire_lane_a(wl_git_repo, wl_git_config_on)
        lane = info.path

        await git_ops.release_warm_lane(lane, 'task-A')

        _, start_ref, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wl_git_repo)
        info2 = await git_ops.acquire_warm_lane('task-C', start_ref.strip())
        assert info2 is not None
        assert info2.path == lane

    async def test_release_idempotent_when_already_free(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Calling release_warm_lane on a FREE lane does not raise."""
        git_ops, info = await self._acquire_lane_a(wl_git_repo, wl_git_config_on)
        lane = info.path

        await git_ops.release_warm_lane(lane, 'task-A')
        # Second call — pool thinks it's FREE, method must not raise
        await git_ops.release_warm_lane(lane, 'task-A')  # must not raise


# ===========================================================================
# Step-21: RED — acquire_warm_lane REUSE path (live-requeue, same process)
#
# Today acquire_warm_lane calls try_acquire() (size-1 → second acquire returns
# None) → info2 is None → RED.
# After step-22, acquire_for('A') recognises the mapped lane and routes to
# _reuse_warm_lane which preserves .task/plan.json + rebases + retains target/.
# ===========================================================================


async def _make_repo_for_reuse_test(repo: Path) -> tuple[str, str, Path]:
    """Setup a repo for the live-requeue reuse test.

    Creates:
    - .gitignore ignoring target/
    - stub seed + debug-port scripts
    - commit sha_a: adds task_work.txt (WIP target for the task branch)
    - commit sha_main: adds main_advance.txt (different file, no conflict)

    Returns (sha_a, sha_main, seed_marker_path).
    """
    # .gitignore so target/ artifacts stay untracked during WIP commit
    (repo / '.gitignore').write_text('target/\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add .gitignore'], cwd=repo)

    # Add stub seed + debug-port scripts
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    seed_marker = repo / 'seed_calls.txt'
    seed_script = scripts_dir / 'seed-warm-lane.sh'
    seed_script.write_text(
        f'#!/usr/bin/env bash\n'
        f'echo "called:$3" >> {seed_marker}\n'
        f'mkdir -p "$2/target"\n'
        f'echo "seeded" > "$2/target/seeded.bin"\n'
    )
    seed_script.chmod(0o755)
    debug_script = scripts_dir / 'setup-worktree-debug-port.sh'
    debug_script.write_text('#!/usr/bin/env bash\necho 39411\n')
    debug_script.chmod(0o755)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add scripts'], cwd=repo)

    # sha_a: add task_work.txt — the tracked file WIP will modify
    (repo / 'task_work.txt').write_text('original content\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'commit A'], cwd=repo)
    _, sha_a, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)

    # sha_main: advance main with a separate file (no conflict with task_work.txt)
    (repo / 'main_advance.txt').write_text('main advanced\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'advance main'], cwd=repo)
    _, sha_main, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)

    return sha_a.strip(), sha_main.strip(), seed_marker


@pytest.mark.asyncio
class TestAcquireWarmLaneReuse:
    """acquire_warm_lane REUSE path: live-requeue of same branch without releasing."""

    async def test_reuse_returns_not_none(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Second acquire_warm_lane('A') without release returns info2 (not None)."""
        sha_a, sha_main, _marker = await _make_repo_for_reuse_test(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info1 = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert info1 is not None

        # Simulate agent work: WIP on a tracked file + .task/plan.json + target/
        (info1.path / 'task_work.txt').write_text('WIP changes\n')
        (info1.path / '.task').mkdir(exist_ok=True)
        (info1.path / '.task' / 'plan.json').write_text('{"task_id": "task-A"}')
        (info1.path / 'target').mkdir(exist_ok=True)
        (info1.path / 'target' / 'cache.bin').write_bytes(b'\xca\xfe' * 64)

        # WITHOUT releasing, requeue the same task onto new main
        info2 = await git_ops.acquire_warm_lane('task-A', sha_main)

        # TODAY: try_acquire() returns None (exhausted) → info2 is None → RED
        assert info2 is not None

    async def test_reuse_same_lane_path(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Re-acquired lane is the SAME _lane-0 — no new lane consumed."""
        sha_a, sha_main, _marker = await _make_repo_for_reuse_test(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info1 = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert info1 is not None
        (info1.path / 'task_work.txt').write_text('WIP\n')
        (info1.path / '.task').mkdir(exist_ok=True)
        (info1.path / '.task' / 'plan.json').write_text('{"task_id": "task-A"}')

        info2 = await git_ops.acquire_warm_lane('task-A', sha_main)
        assert info2 is not None
        assert info2.path == info1.path

    async def test_reuse_preserves_task_plan_json(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Re-acquired lane still has .task/plan.json with task_id 'task-A'."""
        import json
        sha_a, sha_main, _marker = await _make_repo_for_reuse_test(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info1 = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert info1 is not None
        # Write .task/plan.json (gitignored by .task/.gitignore set up by acquire)
        (info1.path / '.task').mkdir(exist_ok=True)
        plan_file = info1.path / '.task' / 'plan.json'
        plan_file.write_text('{"task_id": "task-A", "title": "my task"}')
        (info1.path / 'task_work.txt').write_text('WIP\n')

        info2 = await git_ops.acquire_warm_lane('task-A', sha_main)
        assert info2 is not None

        assert plan_file.exists(), '.task/plan.json must be preserved on reuse'
        data = json.loads(plan_file.read_text())
        assert data['task_id'] == 'task-A', (
            f'plan.json task_id was overwritten: {data}'
        )

    async def test_reuse_wip_committed_and_rebased(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """WIP commit exists on task/task-A and sha_main is now an ancestor of HEAD."""
        sha_a, sha_main, _marker = await _make_repo_for_reuse_test(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info1 = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert info1 is not None
        # Staged WIP on tracked file
        (info1.path / 'task_work.txt').write_text('WIP changes\n')
        (info1.path / '.task').mkdir(exist_ok=True)
        (info1.path / '.task' / 'plan.json').write_text('{"task_id": "task-A"}')

        info2 = await git_ops.acquire_warm_lane('task-A', sha_main)
        assert info2 is not None

        # A commit with 'save WIP' in message must be reachable from HEAD
        _, log_out, _ = await _run(
            ['git', 'log', '--oneline', '--all', f'{sha_main}..HEAD'],
            cwd=info2.path,
        )
        wip_commits = [
            line for line in log_out.splitlines()
            if 'save WIP before requeue rebase' in line or 'save WIP' in line.lower()
        ]
        assert wip_commits, (
            f'No WIP-save commit found in log:\n{log_out}'
        )

        # sha_main must be an ancestor of HEAD
        rc_ancestor, _, _ = await _run(
            ['git', 'merge-base', '--is-ancestor', sha_main, 'HEAD'],
            cwd=info2.path,
        )
        assert rc_ancestor == 0, (
            f'{sha_main} is not an ancestor of HEAD — rebase did not run'
        )

    async def test_reuse_retains_target_cache(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """target/cache.bin is retained (warmth) after reuse."""
        sha_a, sha_main, _marker = await _make_repo_for_reuse_test(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info1 = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert info1 is not None
        cache_file = info1.path / 'target' / 'cache.bin'
        cache_file.parent.mkdir(exist_ok=True)
        cache_file.write_bytes(b'\xca\xfe' * 64)
        (info1.path / 'task_work.txt').write_text('WIP\n')
        (info1.path / '.task').mkdir(exist_ok=True)
        (info1.path / '.task' / 'plan.json').write_text('{"task_id": "task-A"}')

        info2 = await git_ops.acquire_warm_lane('task-A', sha_main)
        assert info2 is not None

        assert cache_file.exists(), 'target/cache.bin must be retained on reuse'

    async def test_reuse_no_reseed(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """seed-warm-lane.sh is NOT re-invoked on the reuse path."""
        sha_a, sha_main, seed_marker = await _make_repo_for_reuse_test(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info1 = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert info1 is not None
        calls_after_first = seed_marker.read_text() if seed_marker.exists() else ''

        (info1.path / 'task_work.txt').write_text('WIP\n')
        (info1.path / '.task').mkdir(exist_ok=True)
        (info1.path / '.task' / 'plan.json').write_text('{"task_id": "task-A"}')

        info2 = await git_ops.acquire_warm_lane('task-A', sha_main)
        assert info2 is not None

        calls_after_second = seed_marker.read_text() if seed_marker.exists() else ''
        assert calls_after_second == calls_after_first, (
            'seed-warm-lane.sh was re-invoked on reuse (should be skipped)'
        )

    async def test_reuse_assignment_still_set(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """After reuse, assignment_for('task-A') still points to _lane-0."""
        sha_a, sha_main, _marker = await _make_repo_for_reuse_test(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info1 = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert info1 is not None
        (info1.path / 'task_work.txt').write_text('WIP\n')
        (info1.path / '.task').mkdir(exist_ok=True)
        (info1.path / '.task' / 'plan.json').write_text('{"task_id": "task-A"}')

        info2 = await git_ops.acquire_warm_lane('task-A', sha_main)
        assert info2 is not None

        assert git_ops.warm_lane_pool is not None
        assert git_ops.warm_lane_pool.assignment_for('task-A') == info1.path


# ===========================================================================
# Step-23: RED — on-disk backstop: same-task plan.json survives restart
#
# Acquire 'A' fresh, write .task/plan.json with task_id 'A' + tracked WIP.
# Construct a BRAND-NEW GitOps (simulating a process restart — empty
# _assignments).  acquire_warm_lane('A', sha_main) should detect that
# _lane-0's plan.json has task_id == 'A' and route to _reuse_warm_lane
# (preserving .task/plan.json + rebasing).
#
# Contrast (fresh, different task): a new GitOps with the SAME registered
# _lane-0 whose plan.json has task_id 'A' but we acquire for 'Z' → treated
# FRESH → .task/plan.json is gone (git clean removed it) and target/ retained.
#
# Today the fresh-but-registered branch always calls _reset_warm_lane →
# git clean -xfd -e target deletes .task/ even for the same task → RED.
# ===========================================================================


@pytest.mark.asyncio
class TestAcquireWarmLaneOnDiskBackstop:
    """On-disk backstop: plan.json task_id match restores REUSE across restart."""

    async def test_same_task_plan_preserved_after_restart(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """After a restart (new GitOps), same-task acquire preserves .task/plan.json."""
        import json
        sha_a, sha_main, _marker = await _make_repo_for_reuse_test(wl_git_repo)

        # First GitOps: fresh acquire for 'task-A'
        git_ops1 = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        info1 = await git_ops1.acquire_warm_lane('task-A', sha_a)
        assert info1 is not None

        # Simulate agent writing plan.json + tracked WIP
        plan_file = info1.path / '.task' / 'plan.json'
        plan_file.write_text('{"task_id": "task-A", "title": "task A work"}')
        (info1.path / 'task_work.txt').write_text('WIP after restart\n')

        # Simulate process restart: fresh GitOps with empty _assignments
        git_ops2 = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        assert git_ops2.warm_lane_pool is not None
        # Confirm: in-memory map is empty (simulating restart)
        assert git_ops2.warm_lane_pool.assignment_for('task-A') is None

        # Reacquire same task on fresh GitOps — disk backstop should detect REUSE
        info2 = await git_ops2.acquire_warm_lane('task-A', sha_main)

        # TODAY: fresh-but-registered → _reset_warm_lane → git clean → .task/ gone
        # After step-24: reads plan.json, detects task_id match → _reuse_warm_lane
        assert info2 is not None
        assert info2.path == info1.path  # same _lane-0

        assert plan_file.exists(), '.task/plan.json must be preserved by disk backstop'
        data = json.loads(plan_file.read_text())
        assert data['task_id'] == 'task-A', (
            f'plan.json was overwritten; got {data}'
        )

    async def test_same_task_wip_committed_after_restart(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """After restart, same-task reacquire commits tracked WIP + rebases."""
        sha_a, sha_main, _marker = await _make_repo_for_reuse_test(wl_git_repo)

        git_ops1 = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        info1 = await git_ops1.acquire_warm_lane('task-A', sha_a)
        assert info1 is not None
        plan_file = info1.path / '.task' / 'plan.json'
        plan_file.write_text('{"task_id": "task-A"}')
        (info1.path / 'task_work.txt').write_text('WIP work\n')

        # Restart
        git_ops2 = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        info2 = await git_ops2.acquire_warm_lane('task-A', sha_main)
        assert info2 is not None

        # sha_main must be ancestor of HEAD (rebased)
        rc, _, _ = await _run(
            ['git', 'merge-base', '--is-ancestor', sha_main, 'HEAD'],
            cwd=info2.path,
        )
        assert rc == 0, f'{sha_main} not ancestor of HEAD — rebase did not run'

    async def test_different_task_gets_fresh_lane(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """A different task on the same registered lane gets a FRESH reset (plan.json cleared)."""
        sha_a, sha_main, _marker = await _make_repo_for_reuse_test(wl_git_repo)

        # Acquire for 'task-A', write plan.json
        git_ops1 = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        info1 = await git_ops1.acquire_warm_lane('task-A', sha_a)
        assert info1 is not None
        plan_file = info1.path / '.task' / 'plan.json'
        plan_file.write_text('{"task_id": "task-A"}')

        # Release so pool marks the lane FREE (different task can acquire it)
        await git_ops1.release_warm_lane(info1.path, 'task-A')

        # Fresh restart: acquire for 'task-Z' (different branch)
        git_ops2 = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        info2 = await git_ops2.acquire_warm_lane('task-Z', sha_main)
        assert info2 is not None
        assert info2.path == info1.path  # same _lane-0 (pool gave it)

        # .task/plan.json should be GONE (fresh reset cleaned it)
        # (task-Z's plan.json != task-A's, so the disk backstop routes FRESH)
        assert not plan_file.exists(), (
            'Fresh acquire for task-Z should have cleared the prior task-A plan.json'
        )

    async def test_different_task_retains_target_warmth(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Fresh acquire for a different task RE-SEEDS target/ (D10 always-re-seed).

        Before D10 this asserted silent retention of target/cache.bin.  D10
        changes the contract: the fresh-reset path for a new task always
        re-seeds, so we verify that seed-warm-lane.sh was invoked on the
        fresh-recycle acquire (seed_calls.txt must gain a new entry after the
        task-Z acquire).
        """
        sha_a, sha_main, seed_marker = await _make_repo_for_reuse_test(wl_git_repo)

        git_ops1 = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        info1 = await git_ops1.acquire_warm_lane('task-A', sha_a)
        assert info1 is not None
        plan_file = info1.path / '.task' / 'plan.json'
        plan_file.write_text('{"task_id": "task-A"}')
        calls_after_first = (
            seed_marker.read_text() if seed_marker.exists() else ''
        ).splitlines()

        await git_ops1.release_warm_lane(info1.path, 'task-A')

        git_ops2 = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        info2 = await git_ops2.acquire_warm_lane('task-Z', sha_main)
        assert info2 is not None

        calls_after_second = (
            seed_marker.read_text() if seed_marker.exists() else ''
        ).splitlines()
        # D10: seed MUST be called on the fresh-recycle path for a new task
        assert len(calls_after_second) > len(calls_after_first), (
            'seed-warm-lane.sh was NOT called on fresh-recycle for new task (D10 violation)'
        )


# ===========================================================================
# Step-19: RED — WarmLanePool.acquire_for + assignment_for
#   Branch->lane assignment map for in-memory live-requeue detection.
#   acquire_for / assignment_for are absent today → AttributeError → RED.
# ===========================================================================


class TestAcquireFor:
    """Pure state-machine tests — no git I/O."""

    def test_acquire_for_fresh_returns_lane_and_false(self, tmp_path: Path):
        """First acquire_for('A') returns (lane, False) — a fresh allocation."""
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'

        result = asyncio.run(pool.acquire_for('A'))

        assert result is not None
        lane, reused = result
        assert lane == base / '_lane-0'
        assert reused is False

    def test_acquire_for_fresh_marks_lane_assigned(self, tmp_path: Path):
        """After acquire_for('A'), _lane-0 is ASSIGNED."""
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'

        asyncio.run(pool.acquire_for('A'))

        assert pool.state(base / '_lane-0') == LaneState.ASSIGNED

    def test_acquire_for_records_assignment(self, tmp_path: Path):
        """After acquire_for('A'), assignment_for('A') == _lane-0."""
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'

        asyncio.run(pool.acquire_for('A'))

        assert pool.assignment_for('A') == base / '_lane-0'

    def test_acquire_for_reuse_returns_same_lane_and_true(self, tmp_path: Path):
        """Second acquire_for('A') (without release) returns (same_lane, True)."""
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'

        result1 = asyncio.run(pool.acquire_for('A'))
        assert result1 is not None

        result2 = asyncio.run(pool.acquire_for('A'))
        assert result2 is not None
        lane2, reused2 = result2
        assert lane2 == base / '_lane-0'
        assert reused2 is True

    def test_acquire_for_reuse_consumes_no_new_lane(self, tmp_path: Path):
        """Second acquire_for('A') does NOT consume _lane-1 or _lane-2."""
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'

        asyncio.run(pool.acquire_for('A'))
        asyncio.run(pool.acquire_for('A'))  # reuse — must not consume _lane-1

        assert pool.state(base / '_lane-1') == LaneState.FREE
        assert pool.state(base / '_lane-2') == LaneState.FREE

    def test_acquire_for_different_branch_gets_new_lane(self, tmp_path: Path):
        """acquire_for('B') after acquire_for('A') returns _lane-1, not _lane-0."""
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'

        asyncio.run(pool.acquire_for('A'))
        result_b = asyncio.run(pool.acquire_for('B'))

        assert result_b is not None
        lane_b, reused_b = result_b
        assert lane_b == base / '_lane-1'
        assert reused_b is False

    def test_assignment_for_unknown_branch_returns_none(self, tmp_path: Path):
        """assignment_for('Z') returns None when 'Z' was never acquired."""
        pool = _make_pool(tmp_path, size=3)
        assert pool.assignment_for('Z') is None

    def test_release_drops_assignment(self, tmp_path: Path):
        """After release(_lane-0), assignment_for('A') is None and lane is FREE."""
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'
        lane_0 = base / '_lane-0'

        asyncio.run(pool.acquire_for('A'))
        asyncio.run(pool.release(lane_0))

        assert pool.assignment_for('A') is None
        assert pool.state(lane_0) == LaneState.FREE

    def test_release_allows_fresh_reacquire(self, tmp_path: Path):
        """After release, acquire_for('A') returns fresh (reused=False)."""
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'
        lane_0 = base / '_lane-0'

        asyncio.run(pool.acquire_for('A'))
        asyncio.run(pool.release(lane_0))

        result = asyncio.run(pool.acquire_for('A'))
        assert result is not None
        lane, reused = result
        assert reused is False

    def test_exhaustion_returns_none_for_new_branch(self, tmp_path: Path):
        """On a size-1 pool, acquire_for('X') exhausts it; acquire_for('Y') returns None."""
        pool = _make_pool(tmp_path, size=1)

        result_x = asyncio.run(pool.acquire_for('X'))
        assert result_x is not None
        _, reused_x = result_x
        assert reused_x is False

        result_y = asyncio.run(pool.acquire_for('Y'))
        assert result_y is None

    def test_back_compat_try_acquire_unchanged(self, tmp_path: Path):
        """try_acquire() still works as before (does not touch _assignments)."""
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'

        lane = asyncio.run(pool.try_acquire())
        assert lane == base / '_lane-0'
        assert lane is not None
        assert pool.state(lane) == LaneState.ASSIGNED
        # assignment_for is not touched by try_acquire
        assert pool.assignment_for('anything') is None

    def test_back_compat_release_still_works(self, tmp_path: Path):
        """release() still works for lanes acquired via try_acquire (back-compat)."""
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'
        lane_0 = base / '_lane-0'

        asyncio.run(pool.try_acquire())
        asyncio.run(pool.release(lane_0))

        assert pool.state(lane_0) == LaneState.FREE

    def test_note_assignment_records_mapping(self, tmp_path: Path):
        """note_assignment records branch->lane for on-disk backstop use."""
        pool = _make_pool(tmp_path, size=3)
        base = tmp_path / 'worktrees'
        lane_0 = base / '_lane-0'

        pool.note_assignment('A', lane_0)

        assert pool.assignment_for('A') == lane_0

    def test_concurrent_acquire_for_no_duplicates(self, tmp_path: Path):
        """Concurrent acquire_for calls with distinct branches get distinct lanes."""
        pool = _make_pool(tmp_path, size=5)

        async def run():
            branches = ['br-A', 'br-B', 'br-C', 'br-D', 'br-E']
            tasks = [pool.acquire_for(b) for b in branches]
            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(run())
        lanes = [r[0] for r in results if r is not None]
        assert len(lanes) == 5
        assert len(set(lanes)) == 5  # all distinct


# ===========================================================================
# Step-1 RED: WarmLanePool.restore_assignment
# Sets BOTH the assignment map AND lane state to ASSIGNED — the startup
# recovery rebuild that prevents a freed lane from being grabbed by a
# concurrent fresh dispatch before the original task is re-dispatched.
# ===========================================================================


class TestRestoreAssignment:
    def test_restore_sets_lane_assigned(self, tmp_path: Path):
        """After restore_assignment, state(_lane-0) == ASSIGNED."""
        pool = _make_pool(tmp_path, size=2)
        base = tmp_path / 'worktrees'
        lane = base / '_lane-0'

        pool.restore_assignment('42', lane)

        assert pool.state(lane) == LaneState.ASSIGNED

    def test_restore_records_assignment_map(self, tmp_path: Path):
        """After restore_assignment, assignment_for('42') == _lane-0."""
        pool = _make_pool(tmp_path, size=2)
        base = tmp_path / 'worktrees'
        lane = base / '_lane-0'

        pool.restore_assignment('42', lane)

        assert pool.assignment_for('42') == lane

    def test_restore_reserved_lane_skipped_by_try_acquire(self, tmp_path: Path):
        """try_acquire() skips the restored (ASSIGNED) lane and returns _lane-1."""
        pool = _make_pool(tmp_path, size=2)
        base = tmp_path / 'worktrees'
        lane_0 = base / '_lane-0'
        lane_1 = base / '_lane-1'

        pool.restore_assignment('42', lane_0)

        # try_acquire should give lane-1 (lane-0 is reserved ASSIGNED)
        acquired = asyncio.run(pool.try_acquire())
        assert acquired == lane_1

    def test_restore_acquire_for_returns_reused(self, tmp_path: Path):
        """acquire_for('42') after restore returns (lane, reused=True)."""
        pool = _make_pool(tmp_path, size=2)
        base = tmp_path / 'worktrees'
        lane = base / '_lane-0'

        pool.restore_assignment('42', lane)

        result = asyncio.run(pool.acquire_for('42'))
        assert result is not None
        returned_lane, reused = result
        assert returned_lane == lane
        assert reused is True

    def test_restore_idempotent(self, tmp_path: Path):
        """Calling restore_assignment twice is idempotent (no state corruption)."""
        pool = _make_pool(tmp_path, size=2)
        base = tmp_path / 'worktrees'
        lane = base / '_lane-0'

        pool.restore_assignment('42', lane)
        pool.restore_assignment('42', lane)  # second call must not raise or corrupt

        assert pool.state(lane) == LaneState.ASSIGNED
        assert pool.assignment_for('42') == lane

    def test_restore_unknown_lane_is_noop(self, tmp_path: Path):
        """Unknown lane path is silently ignored (never raises)."""
        pool = _make_pool(tmp_path, size=2)
        base = tmp_path / 'worktrees'
        unknown = base / '_lane-99'

        # Must not raise
        pool.restore_assignment('42', unknown)

        # No side effects
        assert pool.assignment_for('42') is None

    def test_restore_does_not_affect_other_lanes(self, tmp_path: Path):
        """restore_assignment for _lane-0 leaves _lane-1 FREE."""
        pool = _make_pool(tmp_path, size=2)
        base = tmp_path / 'worktrees'
        lane_0 = base / '_lane-0'
        lane_1 = base / '_lane-1'

        pool.restore_assignment('42', lane_0)

        assert pool.state(lane_1) == LaneState.FREE


# ===========================================================================
# Step-6: RED — GitOps.refresh_warm_base unit + inv.9 promote-provenance
# ===========================================================================


@pytest.mark.asyncio
class TestRefreshWarmBase:
    """GitOps.refresh_warm_base: invoke <_merge-verify>/scripts/refresh-warm-base.sh."""

    async def _setup_merge_verify(
        self,
        repo: Path,
        recorder_file: Path,
    ) -> Path:
        """Create _merge-verify/target/ and install the recorder script."""
        worktree_base = repo / '.worktrees'
        merge_verify = worktree_base / '_merge-verify'
        merge_verify.mkdir(parents=True, exist_ok=True)
        (merge_verify / 'target').mkdir(exist_ok=True)
        _install_refresh_recorder(merge_verify / 'scripts', recorder_file)
        return merge_verify

    async def test_refresh_invokes_script_with_correct_args(
        self,
        wl_git_repo: Path,
        wl_git_config_rolling_base: GitConfig,
        tmp_path: Path,
    ):
        """refresh_warm_base() invokes the script with <advancing> <base> args."""
        recorder = tmp_path / 'refresh_args.txt'
        await self._setup_merge_verify(wl_git_repo, recorder)
        git_ops = GitOps(wl_git_config_rolling_base, wl_git_repo, warm_lane_pool_size=1)

        result = await git_ops.refresh_warm_base()

        assert result is True, 'refresh_warm_base must return True on success'
        assert recorder.exists(), 'refresh-warm-base.sh was not called (recorder missing)'
        recorded = recorder.read_text().strip()

        advancing = str(git_ops.persistent_merge_worktree_path / 'target')
        base = str(git_ops.warm_lane_base_target_path)
        expected = f'{advancing} {base}'
        assert recorded == expected, (
            f'Wrong args: got {recorded!r}, expected {expected!r}'
        )

    async def test_refresh_advancing_is_merge_verify_not_lane(
        self,
        wl_git_repo: Path,
        wl_git_config_rolling_base: GitConfig,
        tmp_path: Path,
    ):
        """inv.9 promote-provenance: advancing arg is always _merge-verify/target,
        never a task-lane path (_lane-K).

        refresh_warm_base() hardcodes the advancing dir to
        persistent_merge_worktree_path/<artifact_dir> and exposes no caller-
        supplied advancing parameter, so DF can never source un-landed WIP.
        """
        recorder = tmp_path / 'refresh_args.txt'
        await self._setup_merge_verify(wl_git_repo, recorder)
        git_ops = GitOps(wl_git_config_rolling_base, wl_git_repo, warm_lane_pool_size=1)

        await git_ops.refresh_warm_base()

        assert recorder.exists(), 'refresh-warm-base.sh was not called'
        recorded = recorder.read_text().strip()
        advancing_arg = recorded.split(' ')[0]  # first positional arg

        # inv.9: advancing must include '_merge-verify' in the path
        assert '_merge-verify' in advancing_arg, (
            f'Advancing arg must reference _merge-verify, got: {advancing_arg!r}'
        )
        # inv.9: advancing must NEVER be a pool lane path
        assert '_lane-' not in advancing_arg, (
            f'Advancing arg must NOT be a task-lane path, got: {advancing_arg!r}'
        )
