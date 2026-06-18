"""Tests for Harness main-tip integrity sweep — task 1832.

Background: merge-queue verify is SCOPED per task (diff/module), so test-suite-wide
breakage and cross-file regressions can land on main and surface only incidentally
when an unlucky later task runs a broad/__fallback__ verify.  Confirmed instances:
  - task 1829: autouse fixture silently defeated two correctness tests; only task
    1817's broad __fallback__ verify surfaced it (esc-1817-28).
  - esc-1749-16: stale tests poking retired singular fields landed on main.

Fix: a background asyncio task on the Harness wakes every
``config.main_tip_sweep_interval_secs``, and when main has advanced since the last
sweep, runs a FULL unscoped verification (all subprojects: test + lint + typecheck)
against a throwaway detached worktree pinned at the current main SHA — completely off
the serial merge lane, so per-merge latency is untouched.  On drift it files one L1
escalation per distinct bad SHA.

This file covers:
  step-1:  test_config_defaults_main_tip_sweep
  step-7:  TestRunMainTipSweepHarness — _run_main_tip_sweep single-pass tests
  step-9:  TestRunMainTipSweepHarness edge cases (pass, sha-dedup, no-queue)
  step-11: TestMainTipSweepLifecycle — start/stop wiring tests
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import OrchestratorConfig


# ---------------------------------------------------------------------------
# step-1: Config field presence and defaults
# ---------------------------------------------------------------------------


def test_config_defaults_main_tip_sweep() -> None:
    """OrchestratorConfig exposes main_tip_sweep_enabled (True) and
    main_tip_sweep_interval_secs (1800.0) with the correct defaults."""
    config = OrchestratorConfig()
    assert config.main_tip_sweep_enabled is True
    assert config.main_tip_sweep_interval_secs == 1800.0
