"""Tests for WarmLanePool — pure state-machine tests (no git I/O).

Step-1: RED — WarmLanePool and LaneState are absent; import fails.
"""

import asyncio
from pathlib import Path

import pytest

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
