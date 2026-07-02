"""Tests for Harness's interactive-worktree (``_iact-*``) reaper wiring — task δ (2012).

Covers the thin harness-side cadence/startup wiring around
``git_ops.reap_interactive_worktrees`` (task δ's git-primitive, see
test_interactive_worktree_reaper.py):

  step-09/10 — ``_run_interactive_worktree_reaper_pass()`` delegates to
               ``git_ops.reap_interactive_worktrees()``, logs one INFO line per
               reaped record, and never raises.
  step-11/12 — the pass is folded into the existing warm-lane GC cadence tick
               (``_run_warm_lane_gc_pass()``).
  step-13/14 — an unconditional sweep runs at ``run()`` startup, independent of
               the ``warm_lane_gc_enabled`` kill-switch (crash recovery on boot).
  step-15/16 — end-to-end over a real repo: the I2 user-observable signal (log
               line + absence from ``git worktree list``).
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps, ReapedInteractiveWorktree
from orchestrator.harness import Harness
from orchestrator.run_store import RunStore

# ---------------------------------------------------------------------------
# Test factory (mirrors test_harness_warm_lane_gc._make_harness)
# ---------------------------------------------------------------------------


def _make_harness(tmp_path: Path) -> tuple[Harness, MagicMock]:
    """Bare Harness with a real config and a spy RunStore.

    Mirrors test_harness_warm_lane_gc._make_harness.
    Returns (harness, mock_run_store).
    """
    config = OrchestratorConfig(project_root=tmp_path)
    harness = Harness(config)
    mock_run_store = MagicMock(spec=RunStore)
    harness._run_store = mock_run_store
    harness._run_id = 'run-interactive-reaper-0001'
    harness.event_store = EventStore(tmp_path / 'events.db', 'run-interactive-reaper-0001')
    return harness, mock_run_store


# ---------------------------------------------------------------------------
# Step-09: _run_interactive_worktree_reaper_pass delegates to
# git_ops.reap_interactive_worktrees, logs per-record, and never raises.
# ---------------------------------------------------------------------------


class TestRunInteractiveWorktreeReaperPass:
    """RED until step-10 GREEN adds _run_interactive_worktree_reaper_pass to Harness."""

    @pytest.mark.asyncio
    async def test_pass_delegates_and_logs_one_info_line_per_record(
        self, tmp_path: Path, caplog,
    ) -> None:
        """Awaits reap_interactive_worktrees() once; logs one INFO line per
        reaped record naming slug, branch, and reason."""
        harness, _rs = _make_harness(tmp_path)
        records = [
            ReapedInteractiveWorktree(
                path=Path('/tmp/x/_iact-alpha'), branch='task/alpha',
                slug='alpha', reason='ttl_idle',
            ),
            ReapedInteractiveWorktree(
                path=Path('/tmp/x/_iact-beta'), branch='task/beta',
                slug='beta', reason='landed',
            ),
        ]
        mock_reap = AsyncMock(return_value=records)
        harness.git_ops.reap_interactive_worktrees = mock_reap

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
            await harness._run_interactive_worktree_reaper_pass()

        mock_reap.assert_awaited_once()

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        for rec in records:
            assert any(
                rec.slug in r.getMessage()
                and rec.branch in r.getMessage()
                and rec.reason in r.getMessage()
                for r in info_records
            ), (
                f'expected an INFO line naming slug={rec.slug!r} '
                f'branch={rec.branch!r} reason={rec.reason!r}; '
                f'got: {[r.getMessage() for r in info_records]}'
            )

    @pytest.mark.asyncio
    async def test_pass_swallows_exception_and_logs_error(
        self, tmp_path: Path, caplog,
    ) -> None:
        """A raising reap_interactive_worktrees() does not propagate; an
        error is logged instead."""
        harness, _rs = _make_harness(tmp_path)
        mock_reap = AsyncMock(side_effect=RuntimeError('boom'))
        harness.git_ops.reap_interactive_worktrees = mock_reap

        with caplog.at_level(logging.ERROR, logger='orchestrator.harness'):
            # Must not raise.
            await harness._run_interactive_worktree_reaper_pass()

        mock_reap.assert_awaited_once()
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, 'expected an ERROR log when reap_interactive_worktrees raises'


# ---------------------------------------------------------------------------
# Step-11: the interactive reaper is folded into the existing warm-lane GC
# cadence tick (_run_warm_lane_gc_pass), per PRD δ "folded into the existing
# warm-lane GC cadence" — no new config knob, no new loop.
# ---------------------------------------------------------------------------


class TestWarmLaneGcPassFoldsInInteractiveReaper:
    """_run_warm_lane_gc_pass() also drives the interactive-worktree reaper.

    RED until step-12 GREEN adds the delegation call to
    Harness._run_warm_lane_gc_pass.
    """

    @pytest.mark.asyncio
    async def test_gc_pass_awaits_interactive_reaper_pass_once(
        self, tmp_path: Path,
    ) -> None:
        """Every warm-lane GC cadence tick also runs the interactive reaper."""
        harness, _rs = _make_harness(tmp_path)
        harness.git_ops._run_warm_lane_gc_reclaim = AsyncMock(return_value=0)
        mock_reaper_pass = AsyncMock(return_value=None)
        harness._run_interactive_worktree_reaper_pass = mock_reaper_pass

        await harness._run_warm_lane_gc_pass()

        mock_reaper_pass.assert_awaited_once()
