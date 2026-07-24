"""Tests for task 2988 — EXHAUSTED requeue-cap flip + structural-exhaustion
pool-level L2 (PRD ε / W3, plans/warm-lane-exhaustion-hardening-prd.md).

Two coupled mechanisms closing both failure poles of the 2026-07-22 incident:
  (1) Cap-semantics: a WarmLanePoolExhausted requeue no longer counts against
      the per-task requeue cap (threaded table-driven from the disposition
      flip through TerminalReport -> TaskReport -> Scheduler.record_requeue).
  (2) Structural-exhaustion L2: a pool-GLOBAL consecutive-EXHAUSTED counter on
      GitOps fires a harness-installed born-at-L2 escalation after N trips.

Test classes (added across steps 3/7/9/11 — the record_requeue route unit
test lives in test_retry_cap.py, step-5):
  step-03: TestStructuralExhaustionConfigKnob — the GitConfig threshold field
  step-07: test_warm_lane_pool_exhausted_does_not_count — end-to-end retry-cap
  step-09: TestConsecutiveExhaustedCounter — git_ops counter mechanics
  step-11: TestFileStructuralExhaustionL2 / TestStructuralExhaustionWiring —
           the harness filer, dedup, no-op, and callback install
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# step-03: GitConfig.warm_lane_structural_exhaustion_l2_threshold knob
# ---------------------------------------------------------------------------


class TestStructuralExhaustionConfigKnob:
    """GitConfig.warm_lane_structural_exhaustion_l2_threshold (task 2988):
    green-tier/hot-reloadable, default 5 (PRD Open Q1), ge=1.  Mirrors
    warm_lane_drift_l2_threshold's Field conventions; defaults.yaml
    deliberately omits it (the Field default is the single source of truth).
    """

    def test_default_is_five(self):
        from orchestrator.config import GitConfig
        assert GitConfig().warm_lane_structural_exhaustion_l2_threshold == 5

    def test_zero_is_rejected_ge_one(self):
        from pydantic import ValidationError

        from orchestrator.config import GitConfig
        with pytest.raises(ValidationError):
            GitConfig(warm_lane_structural_exhaustion_l2_threshold=0)

    def test_negative_is_rejected_ge_one(self):
        from pydantic import ValidationError

        from orchestrator.config import GitConfig
        with pytest.raises(ValidationError):
            GitConfig(warm_lane_structural_exhaustion_l2_threshold=-1)

    def test_accepts_explicit_positive_override(self):
        from orchestrator.config import GitConfig
        cfg = GitConfig(warm_lane_structural_exhaustion_l2_threshold=2)
        assert cfg.warm_lane_structural_exhaustion_l2_threshold == 2
