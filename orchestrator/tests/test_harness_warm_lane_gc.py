"""Tests for task 1926: Warm-lane auto-GC cadence loop.

Covers:
  step-03 RED  — _run_warm_lane_gc_pass() delegates to
                 git_ops._run_warm_lane_gc_reclaim() and is fail-soft.
  step-05 RED  — Loop lifecycle: _start/_stop_warm_lane_gc kill-switch +
                 dedup + cancel/clear.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.harness import Harness
from orchestrator.run_store import RunStore

# ---------------------------------------------------------------------------
# Test factories
# ---------------------------------------------------------------------------


def _make_harness(tmp_path: Path) -> tuple[Harness, MagicMock]:
    """Bare Harness with a real config and a spy RunStore.

    Mirrors test_harness_no_landings_breaker._make_harness.
    Returns (harness, mock_run_store).
    """
    config = OrchestratorConfig(project_root=tmp_path)
    harness = Harness(config)
    mock_run_store = MagicMock(spec=RunStore)
    harness._run_store = mock_run_store
    harness._run_id = 'run-warm-lane-gc-0001'
    harness.event_store = EventStore(tmp_path / 'events.db', 'run-warm-lane-gc-0001')
    return harness, mock_run_store


# ---------------------------------------------------------------------------
# step-03: _run_warm_lane_gc_pass delegates to _run_warm_lane_gc_reclaim
# ---------------------------------------------------------------------------


class TestWarmLaneGcPass:
    """Harness._run_warm_lane_gc_pass() delegates to git_ops._run_warm_lane_gc_reclaim().

    RED until step-4 GREEN adds _run_warm_lane_gc_pass to Harness.
    """

    @pytest.mark.asyncio
    async def test_pass_delegates_to_reclaim(self, tmp_path: Path) -> None:
        """_run_warm_lane_gc_pass() awaits git_ops._run_warm_lane_gc_reclaim() once."""
        harness, _rs = _make_harness(tmp_path)
        mock_reclaim = AsyncMock(return_value=0)
        harness.git_ops._run_warm_lane_gc_reclaim = mock_reclaim

        await harness._run_warm_lane_gc_pass()

        mock_reclaim.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pass_swallows_nonzero_rc(self, tmp_path: Path) -> None:
        """_run_warm_lane_gc_pass() does not raise when reclaim returns non-zero rc.

        rc=127 is the fail-soft sentinel (script absent); other non-zero values
        indicate a script error. Neither should propagate as an exception.
        """
        harness, _rs = _make_harness(tmp_path)
        for rc in (127, 1, 2):
            mock_reclaim = AsyncMock(return_value=rc)
            harness.git_ops._run_warm_lane_gc_reclaim = mock_reclaim
            # Must not raise
            await harness._run_warm_lane_gc_pass()
            mock_reclaim.assert_awaited_once()


# ---------------------------------------------------------------------------
# step-05: Loop lifecycle
# ---------------------------------------------------------------------------


class TestHarnessWarmLaneGcLifecycle:
    """_start/_stop_warm_lane_gc lifecycle mirrors TestHarnessNoLandingsLifecycle.

    RED until step-6 GREEN adds _start_warm_lane_gc, _stop_warm_lane_gc,
    _warm_lane_gc_loop, and the _warm_lane_gc_task field to Harness.
    """

    @pytest.mark.asyncio
    async def test_start_creates_task_when_enabled(self, tmp_path: Path) -> None:
        """_start_warm_lane_gc() creates a live asyncio.Task when flag=True."""
        harness, _rs = _make_harness(tmp_path)
        harness.config.warm_lane_gc_enabled = True
        # Use a very long interval so the loop never actually runs a pass
        harness.config.warm_lane_gc_interval_secs = 9999.0
        harness._start_warm_lane_gc()
        assert harness._warm_lane_gc_task is not None
        assert not harness._warm_lane_gc_task.done()
        # Cleanup
        await harness._stop_warm_lane_gc()

    @pytest.mark.asyncio
    async def test_start_noop_when_disabled(self, tmp_path: Path) -> None:
        """_start_warm_lane_gc() is a no-op when warm_lane_gc_enabled=False."""
        harness, _rs = _make_harness(tmp_path)
        harness.config.warm_lane_gc_enabled = False
        harness._start_warm_lane_gc()
        assert harness._warm_lane_gc_task is None

    @pytest.mark.asyncio
    async def test_start_dedup_no_duplicate_task(self, tmp_path: Path) -> None:
        """Calling _start_warm_lane_gc() twice does not spawn a duplicate task."""
        harness, _rs = _make_harness(tmp_path)
        harness.config.warm_lane_gc_enabled = True
        harness.config.warm_lane_gc_interval_secs = 9999.0
        harness._start_warm_lane_gc()
        task1 = harness._warm_lane_gc_task
        harness._start_warm_lane_gc()  # second call — should be a no-op
        task2 = harness._warm_lane_gc_task
        assert task1 is task2, 'second _start_warm_lane_gc() must not spawn a new task'
        await harness._stop_warm_lane_gc()

    @pytest.mark.asyncio
    async def test_stop_cancels_and_clears_task(self, tmp_path: Path) -> None:
        """_stop_warm_lane_gc() cancels the task and sets _warm_lane_gc_task to None."""
        harness, _rs = _make_harness(tmp_path)
        harness.config.warm_lane_gc_enabled = True
        harness.config.warm_lane_gc_interval_secs = 9999.0
        harness._start_warm_lane_gc()
        assert harness._warm_lane_gc_task is not None
        await harness._stop_warm_lane_gc()
        assert harness._warm_lane_gc_task is None

    @pytest.mark.asyncio
    async def test_stop_when_no_task_is_noop(self, tmp_path: Path) -> None:
        """_stop_warm_lane_gc() is a no-op when no task was ever started."""
        harness, _rs = _make_harness(tmp_path)
        # Should not raise
        await harness._stop_warm_lane_gc()
        assert harness._warm_lane_gc_task is None
