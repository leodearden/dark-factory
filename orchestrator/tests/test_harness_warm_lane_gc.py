"""Tests for task 1926: Warm-lane auto-GC cadence loop.

Covers:
  step-03 RED  — _run_warm_lane_gc_pass() delegates to
                 git_ops._run_warm_lane_gc_reclaim() and is fail-soft.
  step-05 RED  — Loop lifecycle: _start/_stop_warm_lane_gc kill-switch +
                 dedup + cancel/clear.
  amend-1      — _warm_lane_gc_loop() body: exception path is swallowed and
                 loop survives (failure-resilience contract).
  amend-3      — Startup/shutdown wiring: _warm_lane_gc_task is None before
                 startup, live after _start_warm_lane_gc(), None after
                 _stop_warm_lane_gc() (mirrors run()/shutdown contract).
"""
from __future__ import annotations

import asyncio
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


# ---------------------------------------------------------------------------
# amend-1: Loop body — failure resilience
# ---------------------------------------------------------------------------


class TestWarmLaneGcLoopBody:
    """_warm_lane_gc_loop() body: the exception branch swallows errors and continues.

    RED until _warm_lane_gc_loop() exists (step-6); the tests here verify the
    failure-resilience contract that the lifecycle tests (9999s interval) cannot
    reach — specifically that an exception from _run_warm_lane_gc_pass() is
    caught, logged, and the loop continues rather than dying.
    """

    @pytest.mark.asyncio
    async def test_loop_survives_pass_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_warm_lane_gc_loop() swallows a raising pass and keeps running.

        Drives one full exception iteration:
          sleep(tiny) → _run_warm_lane_gc_pass raises → error logged →
          backoff sleep(tiny) → sleep(tiny) → _run_warm_lane_gc_pass succeeds

        After all that the task must still be alive (not done/cancelled),
        proving the exception branch does not kill the loop.
        """
        import orchestrator.harness as harness_mod

        harness_obj, _rs = _make_harness(tmp_path)
        harness_obj.config.warm_lane_gc_enabled = True
        # Tiny interval so the loop fires almost immediately
        harness_obj.config.warm_lane_gc_interval_secs = 0.01
        # Patch backoff constant so we don't wait 60 s in CI
        monkeypatch.setattr(harness_mod, '_BG_LOOP_FAILURE_BACKOFF_SECS', 0.01)

        call_count = 0

        async def failing_then_ok() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError('simulated pass failure')
            # Second and later calls succeed silently

        harness_obj._run_warm_lane_gc_pass = failing_then_ok

        harness_obj._start_warm_lane_gc()
        loop_task = harness_obj._warm_lane_gc_task
        assert loop_task is not None

        # Wait long enough for: sleep(0.01) → raise → backoff(0.01) → sleep(0.01) → ok
        # 0.3 s is ~10× the total cycle time — generous without being slow.
        await asyncio.sleep(0.3)

        # Task must still be alive — exception was swallowed, not re-raised
        assert not loop_task.done(), (
            'loop task must survive a pass exception; '
            f'done={loop_task.done()}, cancelled={loop_task.cancelled()}'
        )
        # Pass was invoked at least once (the failing call)
        assert call_count >= 1, f'expected at least 1 pass call, got {call_count}'

        # Cleanup
        await harness_obj._stop_warm_lane_gc()


# ---------------------------------------------------------------------------
# amend-3: Startup / shutdown wiring
# ---------------------------------------------------------------------------


class TestWarmLaneGcRunShutdownWiring:
    """Startup/shutdown wiring contract for the warm-lane GC cadence loop.

    Verifies that _warm_lane_gc_task is None before startup, live after
    _start_warm_lane_gc() (mirroring run()'s 1c1f block), and None again after
    _stop_warm_lane_gc() (mirroring the shutdown stop-block), without spinning
    up the full Harness.run() loop.
    """

    @pytest.mark.asyncio
    async def test_task_none_before_startup(self, tmp_path: Path) -> None:
        """_warm_lane_gc_task is None immediately after Harness construction."""
        harness, _rs = _make_harness(tmp_path)
        assert harness._warm_lane_gc_task is None

    @pytest.mark.asyncio
    async def test_task_live_after_startup_none_after_shutdown(
        self, tmp_path: Path
    ) -> None:
        """Startup wiring → task live; shutdown wiring → task None.

        Simulates the run() 1c1f block (_start_warm_lane_gc) and the shutdown
        stop-block (_stop_warm_lane_gc).  Uses a long interval so the loop
        never fires a real pass during the test.
        """
        harness, _rs = _make_harness(tmp_path)
        harness.config.warm_lane_gc_enabled = True
        harness.config.warm_lane_gc_interval_secs = 9999.0

        # Pre-startup state
        assert harness._warm_lane_gc_task is None, 'task must be None before startup'

        # Simulate run() → _start_warm_lane_gc()
        harness._start_warm_lane_gc()
        assert harness._warm_lane_gc_task is not None, 'task must be live after startup'
        assert not harness._warm_lane_gc_task.done(), 'task must not be done on startup'

        # Simulate shutdown → _stop_warm_lane_gc()
        await harness._stop_warm_lane_gc()
        assert harness._warm_lane_gc_task is None, 'task must be None after shutdown'
