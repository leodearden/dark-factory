"""Tests for the escalation-watcher-auto subprocess supervisor.

Task 1326 — AFK hardening: escalation-watcher-auto skill + orch subprocess supervisor.

Steps covered by this file:
  step-3: TestWatcherConfig — OrchestratorConfig field presence + defaults
  step-5: TestWatcherSupervisorLifecycle — start/stop/idempotent lifecycle
  step-7: TestRunWatcherRotation — _run_watcher_rotation invoke contract
  step-9: TestWatcherSupervisorLoopClassification — clean/unclean backoff
  step-11: TestWatcherCrashloopTrip — crashloop detection + pause_scheduler
  step-13: TestWatcherSupervisorWiring — __init__ attrs + run() source guard
"""

from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness


# ---------------------------------------------------------------------------
# step-3: Config field presence and defaults
# ---------------------------------------------------------------------------

class TestWatcherConfig:
    """OrchestratorConfig exposes all spec'd watcher_* fields with correct defaults."""

    def test_watcher_supervisor_enabled_default_true(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_supervisor_enabled is True

    def test_watcher_subprocess_restart_backoff_secs_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_subprocess_restart_backoff_secs == 30.0

    def test_watcher_rotation_escalations_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_rotation_escalations == 50

    def test_watcher_rotation_hours_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_rotation_hours == 4.0

    def test_watcher_max_crashloop_restarts_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_max_crashloop_restarts == 5

    def test_watcher_crashloop_window_secs_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_crashloop_window_secs == 600

    # Invocation knobs
    def test_watcher_model_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_model == 'opus'

    def test_watcher_rotation_budget_usd_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_rotation_budget_usd == 40.0

    def test_watcher_max_turns_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_max_turns == 400

    def test_watcher_effort_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_effort == 'high'

    def test_watcher_backend_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_backend == 'claude'


# ---------------------------------------------------------------------------
# step-5: Supervisor lifecycle — start / stop / idempotent
# ---------------------------------------------------------------------------

async def _never_return() -> None:
    """A coroutine that never completes (simulates a running supervisor loop)."""
    await asyncio.sleep(9999)


def _make_lifecycle_harness(tmp_path: Path, *, enabled: bool = True) -> Harness:
    """Build a minimal Harness via __new__ with lifecycle attributes injected."""
    from collections import deque

    h = Harness.__new__(Harness)
    config = OrchestratorConfig(project_root=tmp_path)
    config = config.model_copy(update={'watcher_supervisor_enabled': enabled})
    h.config = config
    h._watcher_supervisor_task = None
    h._watcher_unclean_exits: deque = deque()
    return h


class TestWatcherSupervisorLifecycle:
    """_start/_stop_watcher_supervisor lifecycle."""

    @pytest.mark.asyncio
    async def test_start_noop_when_disabled(self, tmp_path: Path) -> None:
        """When disabled, _start_watcher_supervisor is a no-op."""
        h = _make_lifecycle_harness(tmp_path, enabled=False)
        with patch.object(h, '_watcher_supervisor_loop', side_effect=_never_return):
            h._start_watcher_supervisor()
        assert h._watcher_supervisor_task is None

    @pytest.mark.asyncio
    async def test_start_creates_named_task(self, tmp_path: Path) -> None:
        """When enabled, creates an asyncio.Task named 'watcher-supervisor'."""
        h = _make_lifecycle_harness(tmp_path, enabled=True)
        with patch.object(Harness, '_watcher_supervisor_loop', return_value=_never_return()):
            h._start_watcher_supervisor()
            task = h._watcher_supervisor_task
            assert task is not None
            assert isinstance(task, asyncio.Task)
            assert task.get_name() == 'watcher-supervisor'
            task.cancel()
            with pytest.raises((asyncio.CancelledError, Exception)):
                await task

    @pytest.mark.asyncio
    async def test_start_idempotent(self, tmp_path: Path) -> None:
        """A second call while the task is still alive does not replace it."""
        h = _make_lifecycle_harness(tmp_path, enabled=True)
        with patch.object(Harness, '_watcher_supervisor_loop', return_value=_never_return()):
            h._start_watcher_supervisor()
            first_task = h._watcher_supervisor_task
            h._start_watcher_supervisor()
            assert h._watcher_supervisor_task is first_task
            first_task.cancel()
            with pytest.raises((asyncio.CancelledError, Exception)):
                await first_task

    @pytest.mark.asyncio
    async def test_stop_cancels_and_resets(self, tmp_path: Path) -> None:
        """_stop_watcher_supervisor cancels the task and resets to None."""
        h = _make_lifecycle_harness(tmp_path, enabled=True)
        with patch.object(Harness, '_watcher_supervisor_loop', return_value=_never_return()):
            h._start_watcher_supervisor()
            assert h._watcher_supervisor_task is not None
        await h._stop_watcher_supervisor()
        assert h._watcher_supervisor_task is None

    @pytest.mark.asyncio
    async def test_stop_noop_when_none(self, tmp_path: Path) -> None:
        """_stop_watcher_supervisor with no task is a no-op."""
        h = _make_lifecycle_harness(tmp_path)
        # Should not raise
        await h._stop_watcher_supervisor()
