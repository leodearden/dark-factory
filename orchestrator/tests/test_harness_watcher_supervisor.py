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
