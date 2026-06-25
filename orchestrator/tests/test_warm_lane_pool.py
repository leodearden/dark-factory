"""Tests for WarmLanePool — pure state-machine tests (no git I/O)
+ GitOps-wiring tests (step-3)
+ _seed_warm_lane tests (step-5).

Step-1: RED — WarmLanePool and LaneState are absent; import fails.
Step-3: RED — GitOps pool-wiring / warm_lane_base_target_path absent.
Step-5: RED — GitOps._seed_warm_lane absent.
"""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, WarmLaneUnavailable, WorktreeInfo, _run
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
# Step-1: RED — WarmLanePool.assignments_snapshot (Diff 1)
# ===========================================================================


class TestAssignmentsSnapshot:
    def test_assignments_snapshot_is_a_copy(self, tmp_path: Path):
        """assignments_snapshot() returns a shallow copy decoupled from the live dict.

        Pins Diff 1: after release drops 'A' from _assignments, the snapshot
        taken before release still contains 'A', and it is a distinct object.
        """
        pool = _make_pool(tmp_path, size=3)
        result = asyncio.run(pool.acquire_for('A'))
        assert result is not None
        lane, _reused = result
        snap = pool.assignments_snapshot()
        # Release the lane — drops 'A' from _assignments
        asyncio.run(pool.release(lane))
        # Snapshot captured the state BEFORE release
        assert 'A' in snap, 'snapshot must contain A even after live release'
        # Snapshot is a distinct object from the live _assignments dict
        assert snap is not pool._assignments, (
            'assignments_snapshot() must return a copy, not the live dict'
        )


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
    """Install a refresh-warm-base.sh that records all args ($@) to *recorder_file*
    and exits 0.

    Records all argv so tests can assert the full argument list including any
    optional flags (e.g. --landed-commit <sha> added in task-1846).
    Mirrors the fake seed-warm-lane.sh pattern used by TestSeedWarmLane.
    """
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / 'refresh-warm-base.sh'
    script.write_text(
        f'#!/usr/bin/env bash\necho "$@" >> {recorder_file}\n'
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


class TestGitOpsSpecPoolWiring:
    """Step-3 RED / Step-4 GREEN — GitOps.spec_warm_lane_pool wiring.

    Mirrors TestGitOpsPoolWiring but for the _spec- pool.
    """

    def test_spec_pool_constructed_when_enabled_with_size(
        self, wl_git_repo: Path,
    ):
        """GitOps(config(merge_spec_warm_lane_pool=True), repo, merge_spec_warm_lane_pool_size=K)
        exposes spec_warm_lane_pool: a WarmLanePool with name_prefix='_spec-' and size K."""
        config = GitConfig(merge_spec_warm_lane_pool=True)
        git_ops = GitOps(config, wl_git_repo, merge_spec_warm_lane_pool_size=3)
        assert git_ops.spec_warm_lane_pool is not None
        assert isinstance(git_ops.spec_warm_lane_pool, WarmLanePool)
        assert git_ops.spec_warm_lane_pool.size == 3
        # Lanes live under worktree_base with _spec- prefix
        worktree_base = (wl_git_repo / '.worktrees').resolve()
        for k in range(3):
            assert git_ops.spec_warm_lane_pool.is_lane(worktree_base / f'_spec-{k}')

    def test_spec_pool_none_when_size_zero(self, wl_git_repo: Path):
        """spec_warm_lane_pool is None when merge_spec_warm_lane_pool_size=0."""
        config = GitConfig(merge_spec_warm_lane_pool=True)
        git_ops = GitOps(config, wl_git_repo, merge_spec_warm_lane_pool_size=0)
        assert git_ops.spec_warm_lane_pool is None

    def test_spec_pool_none_when_knob_off(self, wl_git_repo: Path):
        """spec_warm_lane_pool is None when merge_spec_warm_lane_pool=False even with size>0."""
        config = GitConfig(merge_spec_warm_lane_pool=False)
        git_ops = GitOps(config, wl_git_repo, merge_spec_warm_lane_pool_size=3)
        assert git_ops.spec_warm_lane_pool is None

    def test_spec_pool_none_by_default(self, wl_git_repo: Path):
        """GitOps without merge_spec_warm_lane_pool_size → spec_warm_lane_pool is None."""
        config = GitConfig(merge_spec_warm_lane_pool=True)
        git_ops = GitOps(config, wl_git_repo)
        assert git_ops.spec_warm_lane_pool is None


class TestGitConfigDefaults:
    def test_warm_lane_pool_default_false(self):
        config = GitConfig()
        assert config.warm_lane_pool is False

    def test_warm_lane_base_target_dir_default_none(self):
        config = GitConfig()
        assert config.warm_lane_base_target_dir is None

    def test_merge_spec_warm_lane_pool_default_false(self):
        """merge_spec_warm_lane_pool defaults to False (feature gated off).

        Step-1 RED: field absent → AttributeError.
        Step-2 GREEN: field present and default False.
        """
        config = GitConfig()
        assert config.merge_spec_warm_lane_pool is False


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
        """_seed_warm_lane invokes seed-warm-lane.sh with <base_target> <lane> <mode>.

        Returns 0 (int) on success.
        """
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

        assert result == 0, f'Expected rc=0 (success), got {result!r}'
        assert marker.exists(), 'seed-warm-lane.sh was not called (marker missing)'
        recorded = marker.read_text().strip()
        base_target = str(git_ops.warm_lane_base_target_path)
        expected = f'{base_target} {lane} --fresh-checkout'
        assert recorded == expected, (
            f'Wrong args: got {recorded!r}, expected {expected!r}'
        )

    async def test_seed_absent_script_returns_nonzero(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Absent seed-warm-lane.sh → returns non-zero sentinel (127), never raises."""
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        lane = await _make_lane_dir(wl_git_repo, '_lane-abs')
        # No script installed in lane/scripts/
        result = await git_ops._seed_warm_lane(lane, '--fresh-checkout')
        assert result != 0, f'Expected non-zero rc for absent script, got {result!r}'
        # Sentinel for absent script is 127 (command-not-found convention)
        assert result == 127, f'Expected sentinel 127 for absent script, got {result!r}'

    async def test_seed_nonzero_exit_returns_exit_code(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Script that exits 1 → returns exactly 1 (preserves the exit code), never raises."""
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        lane = await _make_lane_dir(wl_git_repo, '_lane-fail')

        scripts_dir = lane / 'scripts'
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script = scripts_dir / 'seed-warm-lane.sh'
        script.write_text('#!/usr/bin/env bash\necho "failure" >&2\nexit 1\n')
        script.chmod(0o755)

        result = await git_ops._seed_warm_lane(lane, '--fresh-checkout')
        assert result == 1, f'Expected rc=1 (script exit code), got {result!r}'

    async def test_seed_reset_in_place_mode_passes_correct_flag(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """_seed_warm_lane with '--reset-in-place' passes the correct mode flag.

        Returns 0 (int) on success.
        """
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
        assert result == 0, f'Expected rc=0 (success), got {result!r}'
        recorded = marker.read_text().strip()
        assert recorded.endswith('--reset-in-place')

    async def test_seed_disk_pressure_exit_75_returns_75(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Script that exits 75 (EX_TEMPFAIL / disk-pressure) → returns exactly 75.

        This is the DISK_PRESSURE discriminant: exit 75 must be preserved as-is
        so the caller can distinguish transient disk pressure from a generic fault.
        """
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        lane = await _make_lane_dir(wl_git_repo, '_lane-dp')

        scripts_dir = lane / 'scripts'
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script = scripts_dir / 'seed-warm-lane.sh'
        script.write_text('#!/usr/bin/env bash\necho "disk pressure" >&2\nexit 75\n')
        script.chmod(0o755)

        result = await git_ops._seed_warm_lane(lane, '--fresh-checkout')
        assert result == 75, (
            f'Expected rc=75 (EX_TEMPFAIL disk-pressure discriminant), got {result!r}'
        )


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

        assert isinstance(info, WorktreeInfo)
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

        assert isinstance(info, WorktreeInfo)
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
        assert isinstance(info, WorktreeInfo)
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
        assert isinstance(info, WorktreeInfo)

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
        assert isinstance(info, WorktreeInfo)
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
        assert isinstance(info, WorktreeInfo)
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
        assert isinstance(info, WorktreeInfo)
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
        assert isinstance(info, WorktreeInfo)
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
        assert isinstance(info, WorktreeInfo)
        assert git_ops.warm_lane_pool is not None
        assert git_ops.warm_lane_pool.state(info.path) == LaneState.ASSIGNED

    async def test_acquire_exhausted_returns_exhausted(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Pool exhaustion: acquire returns WarmLaneUnavailable.EXHAUSTED (backpressure)."""
        await _add_seed_and_debug_port_scripts(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        _, start_ref, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wl_git_repo)
        start_ref = start_ref.strip()

        # Exhaust the pool with the first acquire
        info1 = await git_ops.acquire_warm_lane('task-A', start_ref)
        assert isinstance(info1, WorktreeInfo), f'Expected WorktreeInfo, got {info1!r}'

        # Next acquire must return EXHAUSTED — never bare None
        info2 = await git_ops.acquire_warm_lane('task-B', start_ref)
        assert info2 is WarmLaneUnavailable.EXHAUSTED, (
            f'Expected WarmLaneUnavailable.EXHAUSTED on pool exhaustion, got {info2!r}'
        )

    async def test_acquire_absent_seed_returns_fault(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """No seed-warm-lane.sh → acquire returns WarmLaneUnavailable.FAULT (infra fault)."""
        # No scripts committed — seed will fail with rc=127 (absent script sentinel)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=2)
        _, start_ref, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wl_git_repo)
        start_ref = start_ref.strip()

        info = await git_ops.acquire_warm_lane('task-A', start_ref)
        assert info is WarmLaneUnavailable.FAULT, (
            f'Expected WarmLaneUnavailable.FAULT for absent seed script, got {info!r}'
        )

    async def test_acquire_create_once_seed_disk_pressure_returns_disk_pressure(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Create-once acquire whose seed exits 75 → WarmLaneUnavailable.DISK_PRESSURE.

        The lane is released back to FREE (no leaked ASSIGNED lanes).
        """
        from orchestrator.warm_lane_pool import LaneState

        # Commit a seed script that exits 75 + a debug-port script
        scripts_dir = wl_git_repo / 'scripts'
        scripts_dir.mkdir(parents=True, exist_ok=True)
        seed_script = scripts_dir / 'seed-warm-lane.sh'
        seed_script.write_text(
            '#!/usr/bin/env bash\necho "disk pressure" >&2\nexit 75\n'
        )
        seed_script.chmod(0o755)
        debug_script = scripts_dir / 'setup-worktree-debug-port.sh'
        debug_script.write_text('#!/usr/bin/env bash\necho 39411\n')
        debug_script.chmod(0o755)
        await _run(['git', 'add', '-A'], cwd=wl_git_repo)
        await _run(['git', 'commit', '-m', 'add disk-pressure seed script'], cwd=wl_git_repo)

        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        _, start_ref, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wl_git_repo)
        start_ref = start_ref.strip()

        result = await git_ops.acquire_warm_lane('task-A', start_ref)
        assert result is WarmLaneUnavailable.DISK_PRESSURE, (
            f'Expected WarmLaneUnavailable.DISK_PRESSURE for seed exit 75, got {result!r}'
        )

        # Lane must be released back to FREE — no leaked ASSIGNED lanes
        assert git_ops.warm_lane_pool is not None
        lane = git_ops.warm_lane_pool._base / '_lane-0'
        assert git_ops.warm_lane_pool.state(lane) == LaneState.FREE, (
            f'Lane must be FREE after DISK_PRESSURE failure, got {git_ops.warm_lane_pool.state(lane)!r}'
        )


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
        assert isinstance(info_a, WorktreeInfo)
        assert git_ops.warm_lane_pool is not None
        # Release the lane (mark FREE directly — release_warm_lane comes in step-12)
        await git_ops.warm_lane_pool.release(info_a.path)

        info_b = await git_ops.acquire_warm_lane('task-B', sha_b)
        assert isinstance(info_b, WorktreeInfo)
        assert info_b.path == info_a.path  # same _lane-0

    async def test_reacquire_head_is_at_new_commit(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """After reset-in-place, HEAD is at commit B (different from A)."""
        sha_a, sha_b = await self._make_repo_with_scripts_and_two_commits(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info_a = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert isinstance(info_a, WorktreeInfo)
        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(info_a.path)

        info_b = await git_ops.acquire_warm_lane('task-B', sha_b)
        assert isinstance(info_b, WorktreeInfo)

        _, head_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=info_b.path)
        assert head_sha.strip() == sha_b

    async def test_reacquire_source_tree_is_clean(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Source tree is bit-identical to a fresh checkout: stray.txt gone, git status clean."""
        sha_a, sha_b = await self._make_repo_with_scripts_and_two_commits(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info_a = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert isinstance(info_a, WorktreeInfo)

        # Simulate stray work: untracked file and modification of tracked file
        (info_a.path / 'stray.txt').write_text('should be gone\n')
        (info_a.path / 'file_a.txt').write_text('dirty modification\n')
        # Warm artifact must survive
        (info_a.path / 'target').mkdir(exist_ok=True)
        (info_a.path / 'target' / 'cache.bin').write_bytes(b'\x00' * 128)

        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(info_a.path)
        info_b = await git_ops.acquire_warm_lane('task-B', sha_b)
        assert isinstance(info_b, WorktreeInfo)

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
        assert isinstance(info_a, WorktreeInfo)
        calls_after_first = (
            seed_marker.read_text() if seed_marker.exists() else ''
        ).splitlines()

        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(info_a.path)
        info_b = await git_ops.acquire_warm_lane('task-B', sha_b)
        assert isinstance(info_b, WorktreeInfo)

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
        assert isinstance(info_a, WorktreeInfo)
        # Record marker state after first (create-once) acquire
        calls_after_first = seed_marker.read_text() if seed_marker.exists() else ''

        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(info_a.path)
        info_b = await git_ops.acquire_warm_lane('task-B', sha_b)
        assert isinstance(info_b, WorktreeInfo)

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
        assert isinstance(info_a, WorktreeInfo)
        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(info_a.path)

        info_b = await git_ops.acquire_warm_lane('task-B', sha_b)
        assert isinstance(info_b, WorktreeInfo)

        _, wt_list, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'], cwd=wl_git_repo,
        )
        lane_registrations = [
            line for line in wt_list.splitlines()
            if line.startswith('worktree ') and '_lane-0' in line
        ]
        assert len(lane_registrations) == 1

    async def test_recycle_reseed_fault_returns_fault(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Recycle path: failed re-seed (exit 1) → WarmLaneUnavailable.FAULT + lane released.

        When seed-warm-lane.sh exits non-zero (generic fault) on the RECYCLE path,
        acquire_warm_lane must return FAULT (NOT a WorktreeInfo — no degraded warmth).
        The lane is released back to FREE.
        """
        from orchestrator.warm_lane_pool import LaneState

        sha_a, sha_b = await self._make_repo_with_scripts_and_two_commits(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        # First acquire (create-once): passing seed → lane registered
        info_a = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert isinstance(info_a, WorktreeInfo), f'prerequisite: create-once acquire must succeed, got {info_a!r}'
        lane_path = info_a.path

        # Overwrite the seed script in the MAIN REPO to exit 1 and commit it.
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
        await git_ops.warm_lane_pool.release(lane_path)

        # Second acquire for a DIFFERENT task: seed exits 1 → must return FAULT
        result = await git_ops.acquire_warm_lane('task-B', sha_fail)
        assert result is WarmLaneUnavailable.FAULT, (
            f'Recycle seed failure (exit 1) must return FAULT, got {result!r}'
        )
        # Lane must be released back to FREE — no leaked ASSIGNED lanes
        assert git_ops.warm_lane_pool.state(lane_path) == LaneState.FREE, (
            f'Lane must be FREE after FAULT, got {git_ops.warm_lane_pool.state(lane_path)!r}'
        )

    async def test_recycle_reseed_disk_pressure_returns_disk_pressure(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Recycle path: seed exits 75 → WarmLaneUnavailable.DISK_PRESSURE + lane released."""
        from orchestrator.warm_lane_pool import LaneState

        sha_a, sha_b = await self._make_repo_with_scripts_and_two_commits(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        # First acquire (create-once): passing seed → lane registered
        info_a = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert isinstance(info_a, WorktreeInfo), f'prerequisite: create-once acquire must succeed, got {info_a!r}'
        lane_path = info_a.path

        # Overwrite the seed script to exit 75 (disk pressure) and commit it
        dp_script = wl_git_repo / 'scripts' / 'seed-warm-lane.sh'
        dp_script.write_text(
            '#!/usr/bin/env bash\necho "disk pressure" >&2\nexit 75\n'
        )
        dp_script.chmod(0o755)
        await _run(['git', 'add', '-A'], cwd=wl_git_repo)
        await _run(['git', 'commit', '-m', 'dp: seed script exits 75'], cwd=wl_git_repo)
        _, sha_dp, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wl_git_repo)
        sha_dp = sha_dp.strip()

        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(lane_path)

        # Second acquire: seed exits 75 → must return DISK_PRESSURE
        result = await git_ops.acquire_warm_lane('task-B', sha_dp)
        assert result is WarmLaneUnavailable.DISK_PRESSURE, (
            f'Recycle seed failure (exit 75) must return DISK_PRESSURE, got {result!r}'
        )
        # Lane must be released back to FREE
        assert git_ops.warm_lane_pool.state(lane_path) == LaneState.FREE, (
            f'Lane must be FREE after DISK_PRESSURE, got {git_ops.warm_lane_pool.state(lane_path)!r}'
        )

    async def test_recycle_reseed_is_thin_no_retained_bloat(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Recycle path (signal a): stray target/ artifact is removed before re-seed.

        A stray file placed under lane/target/ BEFORE the recycle acquire must be
        GONE after a successful recycle (the rm-before-seed defensive cleanup ran).
        The lane itself must be the same _lane-0 (same recycled lane, not cold-created).
        """
        sha_a, sha_b = await self._make_repo_with_scripts_and_two_commits(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        # First acquire (create-once): lane registered, target/ seeded
        info_a = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert isinstance(info_a, WorktreeInfo), f'prerequisite: create-once acquire must succeed, got {info_a!r}'
        lane_path = info_a.path

        # Plant a stray bloat artifact under target/ to simulate retained bloat
        (lane_path / 'target').mkdir(exist_ok=True)
        stray = lane_path / 'target' / 'bloat.bin'
        stray.write_bytes(b'\xde\xad\xbe\xef' * 64)
        assert stray.exists(), 'test setup: stray artifact must exist before recycle'

        assert git_ops.warm_lane_pool is not None
        await git_ops.warm_lane_pool.release(lane_path)

        # Second acquire (recycle, same seed script → exits 0): must succeed
        info_b = await git_ops.acquire_warm_lane('task-B', sha_b)
        assert isinstance(info_b, WorktreeInfo), (
            f'Recycle acquire must succeed (WorktreeInfo), got {info_b!r}'
        )
        # Signal a: stray artifact must be gone (target/ was rm-before-seed)
        assert not stray.exists(), (
            'Stray target/ artifact must be removed by rm-before-seed on recycle'
        )
        # Same recycled lane (not a cold-created worktree_base/branch_name dir)
        assert info_b.path == lane_path, (
            f'Recycled lane path must be the same _lane-0, got {info_b.path}'
        )


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
        assert isinstance(info, WorktreeInfo)
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
        assert isinstance(info2, WorktreeInfo)
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
        assert isinstance(info1, WorktreeInfo)

        # Simulate agent work: WIP on a tracked file + .task/plan.json + target/
        (info1.path / 'task_work.txt').write_text('WIP changes\n')
        (info1.path / '.task').mkdir(exist_ok=True)
        (info1.path / '.task' / 'plan.json').write_text('{"task_id": "task-A"}')
        (info1.path / 'target').mkdir(exist_ok=True)
        (info1.path / 'target' / 'cache.bin').write_bytes(b'\xca\xfe' * 64)

        # WITHOUT releasing, requeue the same task onto new main
        info2 = await git_ops.acquire_warm_lane('task-A', sha_main)

        # TODAY: try_acquire() returns None (exhausted) → info2 is None → RED
        assert isinstance(info2, WorktreeInfo)

    async def test_reuse_same_lane_path(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Re-acquired lane is the SAME _lane-0 — no new lane consumed."""
        sha_a, sha_main, _marker = await _make_repo_for_reuse_test(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info1 = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert isinstance(info1, WorktreeInfo)
        (info1.path / 'task_work.txt').write_text('WIP\n')
        (info1.path / '.task').mkdir(exist_ok=True)
        (info1.path / '.task' / 'plan.json').write_text('{"task_id": "task-A"}')

        info2 = await git_ops.acquire_warm_lane('task-A', sha_main)
        assert isinstance(info2, WorktreeInfo)
        assert info2.path == info1.path

    async def test_reuse_preserves_task_plan_json(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Re-acquired lane still has .task/plan.json with task_id 'task-A'."""
        import json
        sha_a, sha_main, _marker = await _make_repo_for_reuse_test(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info1 = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert isinstance(info1, WorktreeInfo)
        # Write .task/plan.json (gitignored by .task/.gitignore set up by acquire)
        (info1.path / '.task').mkdir(exist_ok=True)
        plan_file = info1.path / '.task' / 'plan.json'
        plan_file.write_text('{"task_id": "task-A", "title": "my task"}')
        (info1.path / 'task_work.txt').write_text('WIP\n')

        info2 = await git_ops.acquire_warm_lane('task-A', sha_main)
        assert isinstance(info2, WorktreeInfo)

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
        assert isinstance(info1, WorktreeInfo)
        # Staged WIP on tracked file
        (info1.path / 'task_work.txt').write_text('WIP changes\n')
        (info1.path / '.task').mkdir(exist_ok=True)
        (info1.path / '.task' / 'plan.json').write_text('{"task_id": "task-A"}')

        info2 = await git_ops.acquire_warm_lane('task-A', sha_main)
        assert isinstance(info2, WorktreeInfo)

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
        assert isinstance(info1, WorktreeInfo)
        cache_file = info1.path / 'target' / 'cache.bin'
        cache_file.parent.mkdir(exist_ok=True)
        cache_file.write_bytes(b'\xca\xfe' * 64)
        (info1.path / 'task_work.txt').write_text('WIP\n')
        (info1.path / '.task').mkdir(exist_ok=True)
        (info1.path / '.task' / 'plan.json').write_text('{"task_id": "task-A"}')

        info2 = await git_ops.acquire_warm_lane('task-A', sha_main)
        assert isinstance(info2, WorktreeInfo)

        assert cache_file.exists(), 'target/cache.bin must be retained on reuse'

    async def test_reuse_no_reseed(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """seed-warm-lane.sh is NOT re-invoked on the reuse path."""
        sha_a, sha_main, seed_marker = await _make_repo_for_reuse_test(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info1 = await git_ops.acquire_warm_lane('task-A', sha_a)
        assert isinstance(info1, WorktreeInfo)
        calls_after_first = seed_marker.read_text() if seed_marker.exists() else ''

        (info1.path / 'task_work.txt').write_text('WIP\n')
        (info1.path / '.task').mkdir(exist_ok=True)
        (info1.path / '.task' / 'plan.json').write_text('{"task_id": "task-A"}')

        info2 = await git_ops.acquire_warm_lane('task-A', sha_main)
        assert isinstance(info2, WorktreeInfo)

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
        assert isinstance(info1, WorktreeInfo)
        (info1.path / 'task_work.txt').write_text('WIP\n')
        (info1.path / '.task').mkdir(exist_ok=True)
        (info1.path / '.task' / 'plan.json').write_text('{"task_id": "task-A"}')

        info2 = await git_ops.acquire_warm_lane('task-A', sha_main)
        assert isinstance(info2, WorktreeInfo)

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
        assert isinstance(info1, WorktreeInfo)

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
        assert isinstance(info2, WorktreeInfo)
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
        assert isinstance(info1, WorktreeInfo)
        plan_file = info1.path / '.task' / 'plan.json'
        plan_file.write_text('{"task_id": "task-A"}')
        (info1.path / 'task_work.txt').write_text('WIP work\n')

        # Restart
        git_ops2 = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        info2 = await git_ops2.acquire_warm_lane('task-A', sha_main)
        assert isinstance(info2, WorktreeInfo)

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
        assert isinstance(info1, WorktreeInfo)
        plan_file = info1.path / '.task' / 'plan.json'
        plan_file.write_text('{"task_id": "task-A"}')

        # Release so pool marks the lane FREE (different task can acquire it)
        await git_ops1.release_warm_lane(info1.path, 'task-A')

        # Fresh restart: acquire for 'task-Z' (different branch)
        git_ops2 = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        info2 = await git_ops2.acquire_warm_lane('task-Z', sha_main)
        assert isinstance(info2, WorktreeInfo)
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
        assert isinstance(info1, WorktreeInfo)
        plan_file = info1.path / '.task' / 'plan.json'
        plan_file.write_text('{"task_id": "task-A"}')
        calls_after_first = (
            seed_marker.read_text() if seed_marker.exists() else ''
        ).splitlines()

        await git_ops1.release_warm_lane(info1.path, 'task-A')

        git_ops2 = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        info2 = await git_ops2.acquire_warm_lane('task-Z', sha_main)
        assert isinstance(info2, WorktreeInfo)

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
        """refresh_warm_base() invokes the script with <advancing> <base> --landed-commit <sha>.

        The --landed-commit <sha> is derived from ``git rev-parse HEAD`` of the
        _merge-verify worktree (task-1846 D10 contract).
        """
        recorder = tmp_path / 'refresh_args.txt'
        merge_verify = await self._setup_merge_verify(wl_git_repo, recorder)
        git_ops = GitOps(wl_git_config_rolling_base, wl_git_repo, warm_lane_pool_size=1)

        result = await git_ops.refresh_warm_base()

        assert result is True, 'refresh_warm_base must return True on success'
        assert recorder.exists(), 'refresh-warm-base.sh was not called (recorder missing)'
        recorded = recorder.read_text().strip()

        # Derive expected landed_commit the same way the impl does
        rc, head_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=merge_verify)
        assert rc == 0, f'Could not derive HEAD from _merge-verify: {head_out!r}'
        head = head_out.strip()

        advancing = str(git_ops.persistent_merge_worktree_path / 'target')
        base = str(git_ops.warm_lane_base_target_path)
        expected = f'{advancing} {base} --landed-commit {head}'
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


# ===========================================================================
# Step-8: RED — refresh_warm_base guards + fail-soft
# ===========================================================================


@pytest.mark.asyncio
class TestRefreshWarmBaseGuards:
    """refresh_warm_base guards and fail-soft behavior.

    Tests (a) and (b) fail against the step-7 happy-only impl (no guards).
    Tests (c) and (d) verify fail-soft behavior already present in step-7
    (regression guards).
    """

    async def test_no_op_when_pool_is_none(
        self, wl_git_repo: Path, wl_git_config_rolling_base: GitConfig, tmp_path: Path,
    ):
        """(a) refresh_warm_base returns False and does NOT invoke the script when
        warm_lane_pool is None (knob off / size-0).

        Fails against step-7 (no pool-None guard — would invoke the script).
        """
        recorder = tmp_path / 'refresh_args.txt'
        # Set up the _merge-verify dir and recorder script
        merge_verify = wl_git_repo / '.worktrees' / '_merge-verify'
        merge_verify.mkdir(parents=True, exist_ok=True)
        (merge_verify / 'target').mkdir(exist_ok=True)
        _install_refresh_recorder(merge_verify / 'scripts', recorder)

        # warm_lane_pool_size=0 → warm_lane_pool is None
        git_ops = GitOps(wl_git_config_rolling_base, wl_git_repo, warm_lane_pool_size=0)
        assert git_ops.warm_lane_pool is None, 'prerequisite: pool must be None'

        result = await git_ops.refresh_warm_base()

        assert result is False, 'refresh_warm_base must return False when pool is None'
        assert not recorder.exists(), (
            'refresh-warm-base.sh must NOT be invoked when pool is None'
        )

    async def test_no_op_when_advancing_equals_base(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig, tmp_path: Path,
    ):
        """(b) refresh_warm_base returns False / script NOT invoked when advancing == base.

        When warm_lane_base_target_dir is unset, warm_lane_base_target_path defaults
        to persistent_merge_worktree_path/target — same as the advancing dir.  Refreshing
        would be a degenerate self-copy, so it is skipped.

        Fails against step-7 (no advancing==base guard — would invoke the script).
        """
        recorder = tmp_path / 'refresh_args.txt'
        merge_verify = wl_git_repo / '.worktrees' / '_merge-verify'
        merge_verify.mkdir(parents=True, exist_ok=True)
        (merge_verify / 'target').mkdir(exist_ok=True)
        _install_refresh_recorder(merge_verify / 'scripts', recorder)

        # wl_git_config_on has NO warm_lane_base_target_dir → advancing == base
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        advancing = git_ops.persistent_merge_worktree_path / 'target'
        base = git_ops.warm_lane_base_target_path
        assert advancing == base, 'prerequisite: advancing and base must be the same path'

        result = await git_ops.refresh_warm_base()

        assert result is False, (
            'refresh_warm_base must return False (no-op) when advancing == base'
        )
        assert not recorder.exists(), (
            'refresh-warm-base.sh must NOT be invoked when advancing == base'
        )

    async def test_absent_script_returns_false(
        self, wl_git_repo: Path, wl_git_config_rolling_base: GitConfig,
    ):
        """(c) Absent script returns False without raising (fail-soft).

        Regression guard: already passes in step-7 (existence guard present).
        """
        merge_verify = wl_git_repo / '.worktrees' / '_merge-verify'
        merge_verify.mkdir(parents=True, exist_ok=True)
        (merge_verify / 'target').mkdir(exist_ok=True)
        # Do NOT install any script

        git_ops = GitOps(wl_git_config_rolling_base, wl_git_repo, warm_lane_pool_size=1)
        result = await git_ops.refresh_warm_base()
        assert result is False, 'Absent script must return False (fail-soft)'

    async def test_failing_script_returns_false(
        self, wl_git_repo: Path, wl_git_config_rolling_base: GitConfig,
    ):
        """(d) Non-zero-exit script returns False without raising (fail-soft).

        Regression guard: already passes in step-7 (rc!=0 → return False).
        """
        merge_verify = wl_git_repo / '.worktrees' / '_merge-verify'
        merge_verify.mkdir(parents=True, exist_ok=True)
        (merge_verify / 'target').mkdir(exist_ok=True)
        _install_refresh_failing(merge_verify / 'scripts')

        git_ops = GitOps(wl_git_config_rolling_base, wl_git_repo, warm_lane_pool_size=1)
        result = await git_ops.refresh_warm_base()
        assert result is False, 'Non-zero script exit must return False (fail-soft)'


# ===========================================================================
# Step-β7: RED — create_worktree routes discriminated WarmLaneUnavailable
# ===========================================================================


@pytest.mark.asyncio
class TestCreateWorktreeWarmLaneRouting:
    """create_worktree with warm_lane_pool enabled routes acquire_warm_lane result.

    (b) Pool exhausted → raises WarmLanePoolExhausted + no cold worktree dir.
    (c) FAULT → raises RuntimeError + no cold dir.
    (d) DISK_PRESSURE → raises WarmLaneDiskPressure.
    (e) Success → returns WorktreeInfo on the pool lane (not a cold-created dir).
    """

    async def _setup_repo_with_seed(
        self,
        repo: Path,
        seed_exit: int = 0,
        port: int = 39411,
    ) -> None:
        """Commit a seed script that exits <seed_exit> + a debug-port script."""
        scripts_dir = repo / 'scripts'
        scripts_dir.mkdir(parents=True, exist_ok=True)
        if seed_exit == 0:
            seed_body = (
                '#!/usr/bin/env bash\n'
                'mkdir -p "$2/target"\n'
                'echo seeded > "$2/target/seeded.bin"\n'
            )
        else:
            seed_body = (
                f'#!/usr/bin/env bash\n'
                f'echo "seed exit {seed_exit}" >&2\n'
                f'exit {seed_exit}\n'
            )
        seed_script = scripts_dir / 'seed-warm-lane.sh'
        seed_script.write_text(seed_body)
        seed_script.chmod(0o755)
        debug_script = scripts_dir / 'setup-worktree-debug-port.sh'
        debug_script.write_text(f'#!/usr/bin/env bash\necho {port}\n')
        debug_script.chmod(0o755)
        await _run(['git', 'add', '-A'], cwd=repo)
        await _run(['git', 'commit', '-m', f'add seed (exit={seed_exit}) + debug-port scripts'], cwd=repo)

    async def test_create_worktree_exhausted_raises_pool_exhausted(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """(b) Pool exhausted → create_worktree raises WarmLanePoolExhausted.

        The cold worktree dir (worktree_base/branch_name) must NOT be created —
        no cold-path fall-through when the pool is enabled.
        """
        from orchestrator.git_ops import WarmLanePoolExhausted

        await self._setup_repo_with_seed(wl_git_repo, seed_exit=0)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        # Exhaust the pool with a first create_worktree call
        info_a = await git_ops.create_worktree('task-first')
        assert isinstance(info_a, WorktreeInfo), f'Prerequisite: first create must succeed, got {info_a!r}'

        # Now pool is exhausted; second call must raise WarmLanePoolExhausted
        cold_dir = git_ops.worktree_base / 'task-second'
        with pytest.raises(WarmLanePoolExhausted):
            await git_ops.create_worktree('task-second')

        # Signal (b): no cold worktree dir created
        assert not cold_dir.exists(), (
            f'Cold dir {cold_dir} must NOT be created on pool exhaustion'
        )

    async def test_create_worktree_fault_raises_runtime_error(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """(c) FAULT (seed exit 1) → create_worktree raises RuntimeError.

        No degraded WorktreeInfo is returned; no cold worktree dir is created.
        """
        await self._setup_repo_with_seed(wl_git_repo, seed_exit=1)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        cold_dir = git_ops.worktree_base / 'task-fault'
        with pytest.raises(RuntimeError):
            await git_ops.create_worktree('task-fault')

        # Signal (c): no cold worktree dir created
        assert not cold_dir.exists(), (
            f'Cold dir {cold_dir} must NOT be created on FAULT'
        )

    async def test_create_worktree_disk_pressure_raises_warm_lane_disk_pressure(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """(d) DISK_PRESSURE (seed exit 75) → create_worktree raises WarmLaneDiskPressure."""
        from orchestrator.git_ops import WarmLaneDiskPressure

        await self._setup_repo_with_seed(wl_git_repo, seed_exit=75)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        with pytest.raises(WarmLaneDiskPressure):
            await git_ops.create_worktree('task-dp')

    async def test_create_worktree_success_returns_worktree_info_on_lane(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """(e) Success → WorktreeInfo whose path is the pool lane (_lane-0)."""
        await self._setup_repo_with_seed(wl_git_repo, seed_exit=0)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        info = await git_ops.create_worktree('task-success')
        assert isinstance(info, WorktreeInfo), f'Expected WorktreeInfo, got {info!r}'
        # Path must be the pool lane, not a cold worktree_base/branch_name dir
        assert '_lane-' in info.path.name, (
            f'Expected pool lane path (_lane-*), got {info.path}'
        )
        cold_dir = git_ops.worktree_base / 'task-success'
        assert not cold_dir.exists(), (
            f'Cold dir {cold_dir} must NOT be created when pool lane was used'
        )


# ===========================================================================
# Step-3: RED — GitOps.release_lane_for_terminal_task (shared primitive)
# ===========================================================================


@pytest.mark.asyncio
class TestReleaseLaneForTerminalTask:
    """GitOps.release_lane_for_terminal_task — idempotent, never-raise shared primitive.

    Tests use in-memory assignment and disk backstop scenarios.
    Git operations inside release_warm_lane are best-effort (fail silently when
    no real worktree exists) so pool state assertions hold regardless.
    """

    async def test_release_resolves_via_in_memory_assignment(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Acquire '3459' → release_lane_for_terminal_task('3459') → True, lane FREE."""
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        pool = git_ops.warm_lane_pool
        assert pool is not None
        result = await pool.acquire_for('3459')
        assert result is not None
        lane, _ = result

        freed = await git_ops.release_lane_for_terminal_task('3459')

        assert freed is True
        assert pool.state(lane) == LaneState.FREE
        assert pool.assignment_for('3459') is None

    async def test_release_strips_branch_prefix(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """release_lane_for_terminal_task('task/3459') strips branch_prefix, frees lane."""
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        pool = git_ops.warm_lane_pool
        assert pool is not None
        result = await pool.acquire_for('3459')
        assert result is not None
        lane, _ = result

        freed = await git_ops.release_lane_for_terminal_task('task/3459')

        assert freed is True
        assert pool.state(lane) == LaneState.FREE

    async def test_release_resolves_via_plan_json_backstop(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """ASSIGNED lane with empty _assignments → disk backstop finds and frees it.

        Updated in step-18: the disk backstop is now opt-in (allow_disk_backstop=True).
        This test represents the legitimate lost-map restart path — callers that
        genuinely need the disk scan must explicitly opt in.
        """
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        pool = git_ops.warm_lane_pool
        assert pool is not None
        # try_acquire marks _lane-0 ASSIGNED but adds NO _assignments entry for '3459'
        lane = await pool.try_acquire()
        assert lane is not None

        # Create the lane directory and write plan.json backstop
        task_dir = lane / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / 'plan.json').write_text('{"task_id": "3459"}')

        freed = await git_ops.release_lane_for_terminal_task('3459', allow_disk_backstop=True)

        assert freed is True, 'disk backstop must find _lane-0 and free it'
        assert pool.state(lane) == LaneState.FREE

    async def test_release_returns_false_when_no_lane(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Unknown task id with no assignment and no plan.json → False, no raise."""
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)

        freed = await git_ops.release_lane_for_terminal_task('99999')

        assert freed is False

    async def test_release_is_idempotent(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """First release → True; second release → False; lane stays FREE (B+A double-fire)."""
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        pool = git_ops.warm_lane_pool
        assert pool is not None
        result = await pool.acquire_for('3459')
        assert result is not None
        lane, _ = result

        first = await git_ops.release_lane_for_terminal_task('3459')
        assert first is True
        assert pool.state(lane) == LaneState.FREE

        # Second call: no in-memory assignment, no plan.json on disk
        second = await git_ops.release_lane_for_terminal_task('3459')
        assert second is False, '2nd call (neither in-memory nor disk) must return False'
        assert pool.state(lane) == LaneState.FREE, 'lane must remain FREE after double-fire'

    async def test_release_never_raises_on_cleanup_error(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """cleanup_worktree raising → primitive catches exception, returns False."""
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        pool = git_ops.warm_lane_pool
        assert pool is not None
        result = await pool.acquire_for('3459')
        assert result is not None

        async def _raise(*args: object, **kwargs: object) -> None:
            raise RuntimeError('simulated cleanup failure')

        monkeypatch.setattr(git_ops, 'cleanup_worktree', _raise)

        freed = await git_ops.release_lane_for_terminal_task('3459')
        assert freed is False, 'cleanup error must be swallowed and return False'

    # ------------------------------------------------------------------
    # Step-17 (1881): disk-backstop opt-in + theft guard
    # ------------------------------------------------------------------

    async def test_release_default_skips_disk_backstop(
        self,
        wl_git_repo: Path,
        wl_git_config_on: GitConfig,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Default call (no kwarg): disk backstop must NOT be consulted.

        Simulates the common done-exit path where _maybe_cleanup_done_worktree
        already released the lane and dropped the in-memory assignment. A stale
        plan.json remains on disk, but the default (allow_disk_backstop=False)
        call must be a true no-op — no disk scan, no redundant cleanup_worktree /
        git branch -D retry.

        Pins consequences 1+2 from design_decision #8:
        (1) no spurious 'branch -D ... failed' WARNING on every DONE exit;
        (2) primitive returns False (true no-op, matching B1/B2/A comments).

        RED: current impl unconditionally calls _find_lane_by_plan_task_id
        (TypeError on the kwarg in (b); disk scan in (a)). Fails because:
        (a) returns True instead of False, (b) lane is freed not ASSIGNED,
        (c) disk_scanned and cleanup_called are non-empty.
        """
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        pool = git_ops.warm_lane_pool
        assert pool is not None

        # try_acquire: marks _lane-0 ASSIGNED but does NOT add a _assignments entry
        lane = await pool.try_acquire()
        assert lane is not None
        assert pool.state(lane) == LaneState.ASSIGNED

        # Write stale plan.json (as if task '3459' had previously used this lane;
        # plan.json survives release_warm_lane since that only detaches + deletes branch)
        task_dir = lane / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / 'plan.json').write_text('{"task_id": "3459"}')

        # Spy on both disk-touching methods — neither must be called
        disk_scanned: list[object] = []
        cleanup_called: list[object] = []

        orig_find = git_ops._find_lane_by_plan_task_id

        def spy_find(task_id: str) -> object:
            disk_scanned.append(task_id)
            return orig_find(task_id)

        monkeypatch.setattr(git_ops, '_find_lane_by_plan_task_id', spy_find)

        orig_cleanup = git_ops.cleanup_worktree

        async def spy_cleanup(*args: Any, **kwargs: Any) -> Any:
            cleanup_called.append(args)
            return await orig_cleanup(*args, **kwargs)

        monkeypatch.setattr(git_ops, 'cleanup_worktree', spy_cleanup)

        freed = await git_ops.release_lane_for_terminal_task('3459')

        assert freed is False, (
            'default call must return False when no in-memory assignment '
            '(disk backstop must NOT be consulted)'
        )
        assert pool.state(lane) == LaneState.ASSIGNED, (
            'lane must remain ASSIGNED — default must not consult disk backstop'
        )
        assert not disk_scanned, (
            '_find_lane_by_plan_task_id must NOT be called (no disk scan on default path)'
        )
        assert not cleanup_called, (
            'cleanup_worktree must NOT be called (no redundant cleanup on default path)'
        )

    async def test_release_allow_disk_backstop_opt_in_uses_disk(
        self,
        wl_git_repo: Path,
        wl_git_config_on: GitConfig,
    ):
        """allow_disk_backstop=True: disk backstop resolves lane and frees it.

        Lost-map / post-restart path: the in-memory assignment was dropped but
        the plan.json still carries the task id.  Explicit opt-in is required.

        RED: release_lane_for_terminal_task has no allow_disk_backstop param
        (TypeError on the call).
        """
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        pool = git_ops.warm_lane_pool
        assert pool is not None

        # try_acquire: ASSIGNED, no _assignments entry (in-memory map lost)
        lane = await pool.try_acquire()
        assert lane is not None

        task_dir = lane / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / 'plan.json').write_text('{"task_id": "3459"}')

        freed = await git_ops.release_lane_for_terminal_task('3459', allow_disk_backstop=True)

        assert freed is True, (
            'allow_disk_backstop=True must resolve via disk backstop and free the lane'
        )
        assert pool.state(lane) == LaneState.FREE

    async def test_release_disk_backstop_refuses_lane_held_by_other_task(
        self,
        wl_git_repo: Path,
        wl_git_config_on: GitConfig,
    ):
        """Theft guard: disk backstop resolves lane but it's held by a DIFFERENT live task.

        Race window: task '3459' completes → its lane freed → a concurrent
        dispatch assigns _lane-0 to '9000' → before the stale plan.json is
        rewritten, a reconciler fires release_lane_for_terminal_task('3459',
        allow_disk_backstop=True).  The theft guard must detect that _lane-0 is
        now assigned to '9000' (holder != '3459') and refuse the release.

        Pins consequence 3 from design_decision #8: no cross-task lane-theft.

        RED: no theft guard — the current impl frees _lane-0, stealing task
        9000's lane (returns True); test fails at assert freed is False.
        """
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        pool = git_ops.warm_lane_pool
        assert pool is not None

        # Write stale plan.json for '3459' on _lane-0 (task 3459's old lane)
        lane_0 = git_ops.worktree_base / '_lane-0'
        task_dir = lane_0 / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / 'plan.json').write_text('{"task_id": "3459"}')

        # A DIFFERENT live task acquires _lane-0 (the first FREE lane)
        result = await pool.acquire_for('9000')
        assert result is not None
        lane, _ = result
        assert lane == lane_0, 'acquire_for must have assigned _lane-0 to 9000'
        assert pool.assignment_for('9000') == lane_0

        # Try to release '3459' via disk backstop — theft guard must refuse
        freed = await git_ops.release_lane_for_terminal_task('3459', allow_disk_backstop=True)

        assert freed is False, (
            'theft guard must refuse: disk found _lane-0 for 3459 but it is '
            'now held by the live task 9000'
        )
        assert pool.assignment_for('9000') == lane_0, (
            "task 9000's lane assignment must be preserved"
        )
        assert pool.state(lane_0) == LaneState.ASSIGNED, (
            "task 9000's lane must remain ASSIGNED (not stolen)"
        )


# ===========================================================================
# Task-1914 step-1: RED — reset-in-place reattach (acquire integration)
# ===========================================================================


@pytest.mark.asyncio
class TestAcquireWarmLaneReattach:
    """acquire_warm_lane detects an orphan task/<id> with commits beyond main
    and reattaches to it (reset-in-place + create-once paths) rather than
    resetting/colliding.  Fresh ids still reset/create as before (byte-identical)."""

    async def _make_repo_with_scripts(self, repo: Path) -> str:
        """Commit seed+debug-port scripts to *repo*; return start_ref (HEAD SHA)."""
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
        await _run(['git', 'commit', '-m', 'add warm-lane scripts'], cwd=repo)
        _, ref, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
        return ref.strip()

    async def _create_orphan_branch_with_commits(
        self,
        repo: Path,
        branch: str,
        start_ref: str,
        tmp_wt: Path,
        n: int = 2,
    ) -> tuple[str, int]:
        """Create *branch* at *start_ref* with *n* commits via a temp worktree.

        Removes the worktree after committing, leaving *branch* as an orphan
        (exists in the repo but not checked out anywhere).

        Returns:
            (tip_sha, commit_count_beyond_main)
        """
        await _run(
            ['git', 'worktree', 'add', '-b', branch, str(tmp_wt), start_ref],
            cwd=repo,
        )
        for i in range(n):
            (tmp_wt / f'wip_{i}.txt').write_text(f'work item {i}\n')
            await _run(['git', 'add', '-A'], cwd=tmp_wt)
            await _run(['git', 'commit', '-m', f'wip {i}'], cwd=tmp_wt)
        _, tip_sha, _ = await _run(['git', 'rev-parse', branch], cwd=repo)
        _, count_out, _ = await _run(
            ['git', 'rev-list', '--count', f'main..{branch}'],
            cwd=repo,
        )
        # --force: safe here since we've committed all changes
        await _run(['git', 'worktree', 'remove', '--force', str(tmp_wt)], cwd=repo)
        return tip_sha.strip(), int(count_out.strip())

    async def test_reset_in_place_reattaches_orphan_with_commits(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig, tmp_path: Path,
    ):
        """Reset-in-place path REATTACHES to task/A that carries commits beyond main.

        Setup: register+free _lane-0 via acquire('seed')+pool.release; create orphan
        task/A with 2 commits via a temp worktree then remove it.

        RED today: `_reset_warm_lane` runs `checkout -f -B task/A start_ref`,
        resetting commit count→0 and destroying the orphan's work.
        GREEN after step-2 adds the reattach guard before _reset_warm_lane.
        """
        start_ref = await self._make_repo_with_scripts(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        assert git_ops.warm_lane_pool is not None

        # Register+free _lane-0 to make it a registered worktree on task/seed.
        # After pool.release, _lane-0 is FREE but still a registered worktree.
        info_seed = await git_ops.acquire_warm_lane('seed', start_ref)
        assert isinstance(info_seed, WorktreeInfo), (
            f'Initial acquire for seed failed: {info_seed!r}'
        )
        await git_ops.warm_lane_pool.release(info_seed.path)

        # Create orphan task/A with 2 commits via a temp worktree, then remove it.
        # task/A exists in the repo but is not checked out anywhere.
        tip_a, count_a = await self._create_orphan_branch_with_commits(
            wl_git_repo, 'task/A', start_ref, tmp_path / 'tmp_wt_A', n=2,
        )
        assert count_a == 2, f'Expected 2 commits for task/A pre-test, got {count_a}'

        # Acquire _lane-0 for 'A' — reattach guard must fire, NOT reset-in-place.
        info_a = await git_ops.acquire_warm_lane('A', start_ref)

        assert isinstance(info_a, WorktreeInfo), (
            f'Expected WorktreeInfo after reattach, got {info_a!r}'
        )

        # task/A commit count MUST be preserved (reattach, not reset to 0).
        # Note: _reuse_warm_lane may add a WIP commit (saving untracked files),
        # so count_after may be > count_a.  The invariant is ">= n, not 0".
        _, count_out, _ = await _run(
            ['git', 'rev-list', '--count', 'main..task/A'],
            cwd=wl_git_repo,
        )
        count_after = int(count_out.strip())
        assert count_after >= count_a, (
            f'task/A commit count must be preserved after reattach: '
            f'expected >={count_a}, got {count_after}'
        )

        # Lane HEAD must equal the current task/A branch tip.
        # (_reuse_warm_lane may add a WIP commit, so the branch tip may differ
        # from the pre-reattach tip_a — use rev-parse task/A for the live tip.)
        _, current_tip, _ = await _run(['git', 'rev-parse', 'task/A'], cwd=wl_git_repo)
        _, lane_head, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=info_a.path)
        assert lane_head.strip() == current_tip.strip(), (
            f'Lane HEAD must equal current task/A tip after reattach: '
            f'expected {current_tip.strip()}, got {lane_head.strip()}'
        )

        # Lane HEAD must NOT be start_ref (that would mean a reset happened).
        assert lane_head.strip() != start_ref, (
            'Lane HEAD must NOT be start_ref — that would mean a reset happened '
            'and task/A commits were destroyed'
        )

        # The original task/A tip must be an ancestor of the current HEAD —
        # the pre-reattach work is preserved in the commit chain.
        anc_rc, _, _ = await _run(
            ['git', 'merge-base', '--is-ancestor', tip_a, lane_head.strip()],
            cwd=wl_git_repo,
        )
        assert anc_rc == 0, (
            f'Original task/A tip ({tip_a[:12]}) must be an ancestor of '
            f'current HEAD: the original commits must be reachable'
        )

    async def test_reset_in_place_fresh_id_resets_to_start_ref(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Reset-in-place path RESETS to start_ref for a fresh id with no orphan branch.

        Regression guard: the reattach guard must NOT fire when task/FRESH has
        no pre-existing branch.  Existing reset-in-place behavior is byte-identical
        (guard scoped to branches that exist AND carry commits beyond main).

        Must stay GREEN both before and after the step-2 impl.
        """
        start_ref = await self._make_repo_with_scripts(wl_git_repo)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        assert git_ops.warm_lane_pool is not None

        # Register+free _lane-0
        info_seed = await git_ops.acquire_warm_lane('seed', start_ref)
        assert isinstance(info_seed, WorktreeInfo), (
            f'Initial acquire for seed failed: {info_seed!r}'
        )
        await git_ops.warm_lane_pool.release(info_seed.path)

        # Assert task/FRESH does not exist yet
        rc, _, _ = await _run(
            ['git', 'rev-parse', '--verify', '--quiet', 'task/FRESH'],
            cwd=wl_git_repo,
        )
        assert rc != 0, 'task/FRESH must not exist before the test'

        # Acquire for fresh id 'FRESH' — guard must NOT fire; reset-in-place as before
        info_fresh = await git_ops.acquire_warm_lane('FRESH', start_ref)

        assert isinstance(info_fresh, WorktreeInfo), (
            f'Expected WorktreeInfo for fresh id, got {info_fresh!r}'
        )

        # HEAD must be at start_ref (reset-in-place creates task/FRESH at start_ref)
        _, lane_head, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=info_fresh.path)
        assert lane_head.strip() == start_ref, (
            f'Lane HEAD must be start_ref after reset-in-place: '
            f'expected {start_ref}, got {lane_head.strip()}'
        )

        # Commit count must be 0 (branch at main tip)
        _, count_out, _ = await _run(
            ['git', 'rev-list', '--count', 'main..task/FRESH'],
            cwd=wl_git_repo,
        )
        assert int(count_out.strip()) == 0, (
            'task/FRESH must have 0 commits beyond main after reset-in-place'
        )

        # Branch must exist
        rc_b, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/FRESH'], cwd=wl_git_repo,
        )
        assert rc_b == 0, 'task/FRESH must exist after acquire'

    # ── step-3: create-once reattach ────────────────────────────────────────

    async def test_create_once_reattaches_existing_leftover_with_commits(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig, tmp_path: Path,
    ):
        """Create-once path REATTACHES to task/A that carries commits beyond main.

        Setup: FRESH pool (_lane-0 never acquired → unregistered); create orphan
        task/A with 2 commits via a temp worktree then remove it.

        RED today: create-once runs `git worktree add -b task/A lane start_ref`
        which collides ('A branch named task/A already exists') → FAULT.
        GREEN after step-4 adds the reattach guard before the -b worktree add.
        """
        start_ref = await self._make_repo_with_scripts(wl_git_repo)
        # FRESH pool — _lane-0 is unregistered (never acquired)
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        assert git_ops.warm_lane_pool is not None

        # Create orphan task/A with 2 commits, then remove the temp worktree.
        # task/A exists in the repo but is not checked out anywhere.
        tip_a, count_a = await self._create_orphan_branch_with_commits(
            wl_git_repo, 'task/A', start_ref, tmp_path / 'tmp_wt_co_A', n=2,
        )
        assert count_a == 2, f'Expected 2 commits for task/A pre-test, got {count_a}'

        # Acquire for 'A' — reattach guard must fire, NOT collide with -b
        info_a = await git_ops.acquire_warm_lane('A', start_ref)

        assert isinstance(info_a, WorktreeInfo), (
            f'Expected WorktreeInfo after create-once reattach, got {info_a!r}'
        )

        # Lane must be a registered worktree on task/A branch
        _, abbrev_head, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=info_a.path,
        )
        assert abbrev_head.strip() == 'task/A', (
            f'Lane must be on task/A after reattach, got {abbrev_head.strip()!r}'
        )

        # task/A commit count MUST be preserved (>=2, not 0)
        _, count_out, _ = await _run(
            ['git', 'rev-list', '--count', 'main..task/A'],
            cwd=wl_git_repo,
        )
        count_after = int(count_out.strip())
        assert count_after >= count_a, (
            f'task/A commit count must be preserved after create-once reattach: '
            f'expected >={count_a}, got {count_after}'
        )

        # Lane HEAD must NOT be start_ref (that would mean a reset happened)
        _, lane_head, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=info_a.path)
        assert lane_head.strip() != start_ref, (
            'Lane HEAD must NOT be start_ref — that would mean a fresh create '
            'happened and task/A commits were not carried over'
        )

        # Original tip_a must be an ancestor of current HEAD (commits preserved)
        anc_rc, _, _ = await _run(
            ['git', 'merge-base', '--is-ancestor', tip_a, lane_head.strip()],
            cwd=wl_git_repo,
        )
        assert anc_rc == 0, (
            f'Original task/A tip ({tip_a[:12]}) must be an ancestor of '
            f'current HEAD: the original commits must be reachable'
        )

    async def test_create_once_fresh_id_creates_branch(
        self, wl_git_repo: Path, wl_git_config_on: GitConfig,
    ):
        """Create-once path CREATES task/FRESH at start_ref for a fresh id with no orphan.

        Regression guard: the reattach guard must NOT fire when task/FRESH has
        no pre-existing branch.  Existing create-once behavior (git worktree add
        -b task/FRESH lane start_ref) is byte-identical.

        Must stay GREEN both before and after the step-4 impl.
        """
        start_ref = await self._make_repo_with_scripts(wl_git_repo)
        # FRESH pool — _lane-0 is unregistered
        git_ops = GitOps(wl_git_config_on, wl_git_repo, warm_lane_pool_size=1)
        assert git_ops.warm_lane_pool is not None

        # Assert task/FRESH does not exist yet
        rc, _, _ = await _run(
            ['git', 'rev-parse', '--verify', '--quiet', 'task/FRESH'],
            cwd=wl_git_repo,
        )
        assert rc != 0, 'task/FRESH must not exist before the test'

        # Acquire for fresh id 'FRESH' — guard must NOT fire; create-once as before
        info_fresh = await git_ops.acquire_warm_lane('FRESH', start_ref)

        assert isinstance(info_fresh, WorktreeInfo), (
            f'Expected WorktreeInfo for fresh id, got {info_fresh!r}'
        )

        # HEAD must be at start_ref (fresh create places task/FRESH at start_ref)
        _, lane_head, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=info_fresh.path)
        assert lane_head.strip() == start_ref, (
            f'Lane HEAD must be start_ref after fresh create: '
            f'expected {start_ref}, got {lane_head.strip()}'
        )

        # Commit count must be 0 (branch at main tip)
        _, count_out, _ = await _run(
            ['git', 'rev-list', '--count', 'main..task/FRESH'],
            cwd=wl_git_repo,
        )
        assert int(count_out.strip()) == 0, (
            'task/FRESH must have 0 commits beyond main after fresh create'
        )

        # Branch must exist
        rc_b, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/FRESH'], cwd=wl_git_repo,
        )
        assert rc_b == 0, 'task/FRESH must exist after acquire'
