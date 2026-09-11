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

    @pytest.mark.asyncio
    async def test_pass_delegates_to_terminal_lane_record_reclaim(
        self, tmp_path: Path
    ) -> None:
        """_run_warm_lane_gc_pass() awaits self._reclaim_terminal_lane_records() once.

        Leaf γ (task 2891): the durable-record terminal-lane reclaim rides the
        existing warm-lane GC cadence tick — no new timer/loop.
        """
        harness, _rs = _make_harness(tmp_path)
        harness.git_ops._run_warm_lane_gc_reclaim = AsyncMock(return_value=0)
        mock_reclaim = AsyncMock(return_value=0)
        harness._reclaim_terminal_lane_records = mock_reclaim

        await harness._run_warm_lane_gc_pass()

        mock_reclaim.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pass_swallows_terminal_lane_reclaim_raise(
        self, tmp_path: Path
    ) -> None:
        """A raise from _reclaim_terminal_lane_records must NOT break the GC cadence.

        Belt-and-suspenders fail-soft, mirroring the interactive-worktree reaper
        delegate: a fault in the reclaim delegate cannot propagate out of
        _run_warm_lane_gc_pass (never-raise contract).
        """
        harness, _rs = _make_harness(tmp_path)
        harness.git_ops._run_warm_lane_gc_reclaim = AsyncMock(return_value=0)
        harness._reclaim_terminal_lane_records = AsyncMock(
            side_effect=RuntimeError('boom')
        )

        # Must not raise
        await harness._run_warm_lane_gc_pass()

        harness._reclaim_terminal_lane_records.assert_awaited_once()

