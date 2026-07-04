"""Tests for the periodic deterministic-strand reconciliation sweep — task 2074.

Background: task 2066 hardened DeterministicRunner against NEW strands but
explicitly scoped OUT recovery of tasks already stranded by a PAST occurrence
(e.g. task 2059): a deterministic gate/deploy task left BLOCKED with
``before_done_ran_at`` stamped but ``before_done_verified_at`` /
``gate_escalated_at`` / ``done_provenance`` all absent, and an EMPTY pending
escalation queue.  Root cause: the cross-unit deploy severed the
orchestrator's own fused-memory connection, so the runner could neither write
back success nor file an escalation.

This file adds a new, cross-cutting background sweep (mirroring
``_main_tip_sweep_loop`` / ``_stranded_reconcile_loop``) that:

  Source A — detects ABSENT-escalation strands (blocked deterministic task +
  empty pending queue) and RE-FILES an L1 escalation whose category depends
  on live systemd unit health (never flips status itself).

  Source B — re-validates OPEN deterministic-deploy ``infra_issue``
  escalations against live unit health, auto-resolving when the stated
  failure is now contradicted by a healthy unit.

This file covers:
  step-1:  test_config_defaults_deterministic_recon_sweep
  step-3:  TestDeterministicDeployHealthVerdict / TestRevalidateDeployHealth
  step-5:  TestIsStrandedDeterministicShape
  step-7:  TestRecoverStrandedDeterministicTask
  step-9:  TestRevalidateOpenDeterministicEscalation
  step-11: TestRunDeterministicReconSweep
  step-13: TestDeterministicReconSweepLifecycle
"""

from __future__ import annotations

from orchestrator.config import OrchestratorConfig

# ---------------------------------------------------------------------------
# step-1: Config field presence and defaults
# ---------------------------------------------------------------------------


def test_config_defaults_deterministic_recon_sweep() -> None:
    """OrchestratorConfig exposes deterministic_recon_sweep_enabled (True) and
    deterministic_recon_sweep_interval_secs (900.0) with the correct defaults."""
    config = OrchestratorConfig()
    assert config.deterministic_recon_sweep_enabled is True
    assert config.deterministic_recon_sweep_interval_secs == 900.0
