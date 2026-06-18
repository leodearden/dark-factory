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
        """target/cache.bin survives the reset-in-place (warmth retention)."""
        sha_a, sha_b = await self._make_repo_with_scripts_and_two_commits(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info_a = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert info_a is not None
        # Write a warm artifact into target/
        (info_a.path / 'target').mkdir(exist_ok=True)
        cache_file = info_a.path / 'target' / 'cache.bin'
        cache_file.write_bytes(b'\x00' * 128)

        await git_ops.warm_lane_pool.release(info_a.path)
        info_b = await git_ops.acquire_warm_lane('task-B', sha_b)
        assert info_b is not None

        # Warm artifact must still be there
        assert cache_file.exists(), 'target/cache.bin was removed (warmth lost)'

    async def test_reacquire_does_not_call_seed(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """seed-warm-lane.sh is NOT called on warm reacquire (marker unchanged)."""
        sha_a, sha_b = await self._make_repo_with_scripts_and_two_commits(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        seed_marker = wl_git_repo / 'seed_calls.txt'

        info_a = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert info_a is not None
        # Record marker state after first (create-once) acquire
        calls_after_first = seed_marker.read_text() if seed_marker.exists() else ''

        await git_ops.warm_lane_pool.release(info_a.path)
        info_b = await git_ops.acquire_warm_lane('task-B', sha_b)
        assert info_b is not None

        calls_after_second = seed_marker.read_text() if seed_marker.exists() else ''
        assert calls_after_second == calls_after_first, (
            'seed-warm-lane.sh was called on warm reacquire (should be skipped)'
        )

    async def test_reacquire_single_worktree_registration(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Exactly one _lane-0 registration in git worktree list after reacquire."""
        sha_a, sha_b = await self._make_repo_with_scripts_and_two_commits(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info_a = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert info_a is not None
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
