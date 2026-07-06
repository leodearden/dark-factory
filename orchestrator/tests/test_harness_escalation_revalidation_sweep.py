"""Tests for the generalized escalation-revalidation sweep — task 2114.

Background: task 2074 built escalation-revalidation machinery scoped to the
blocked-deterministic subset only (Source A/B inside
``_run_deterministic_recon_sweep``).  Motivating evidence (2026-07-06): six L2
escalations were manually ``close_only``'d that a general sweep would have
auto-closed — five because their SUBJECT task became terminal (done or
cancelled), one because the cited defect landed on main (a main-tip-sweep
integrity gate whose fix commits are now ancestors of main).

This file generalizes the revalidation sweep to auto-close ALL stale open L2
escalations using only POSITIVE, fail-safe evidence, split across two homes:

  Source C — terminal-subject closure (criterion a, general, all
  categories/task_kinds): hosted inside the EXISTING
  ``_run_deterministic_recon_sweep`` pass.  For each pending L2 escalation,
  reads the subject task's status via ``scheduler.get_statuses`` and closes
  when the status is terminal (``done`` or ``cancelled``).

  Main-tip-sweep self-heal (criterion b, the main-tip integrity cluster):
  hosted inside ``_run_main_tip_sweep``'s ``vr.passed`` branch.  Closes any
  pending ``orchestrator-main-sweep`` escalation whose swept SHA is a strict
  ancestor of the just-verified clean tip.

This file covers:
  step-1: test_config_defaults_escalation_revalidation
  step-3: TestRevalidateOpenL2
  step-5: TestReconSweepSourceC
  step-7: TestCloseSupersededMainSweepEscalations
  step-9: TestMainTipSweepSelfHealWiring
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import _init_harness_state_for_test
from escalation.models import Escalation

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness

# ---------------------------------------------------------------------------
# step-1: Config field presence and default
# ---------------------------------------------------------------------------


def test_config_defaults_escalation_revalidation() -> None:
    """OrchestratorConfig exposes escalation_revalidation_enabled (True) —
    the single operator kill-switch for both new auto-closure paths."""
    config = OrchestratorConfig()
    assert config.escalation_revalidation_enabled is True
