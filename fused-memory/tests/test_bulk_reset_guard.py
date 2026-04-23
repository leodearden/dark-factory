"""Tests for BulkResetGuard — defence-in-depth circuit-breaker for bulk
task-status reversals (task 918).

Covers:
  - ReconciliationConfig default field values (step-1)
  - BulkResetGuard.observe_attempt under-threshold behaviour (step-3)
  - Threshold tripping within the window (step-5)
  - Non-reversal transitions are ignored (step-7)
  - Window-expiry drains old entries (step-9)
  - Per-project isolation (step-11)
  - L1 escalation file written on trip (step-13)
  - Escalation rate-limiting (step-15)
  - Guard disabled short-circuits everything (step-17)
  - BulkResetVerdict.to_error_dict() shape (step-19)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import pytest_asyncio

from fused_memory.reconciliation.bulk_reset_guard import BulkResetGuard, BulkResetVerdict


# ---------------------------------------------------------------------------
# step-1: ReconciliationConfig defaults
# ---------------------------------------------------------------------------

def test_reconciliation_config_bulk_reset_guard_defaults():
    """ReconciliationConfig() must carry the four new bulk-reset-guard fields
    with their specified default values."""
    from fused_memory.config.schema import ReconciliationConfig

    cfg = ReconciliationConfig()
    assert cfg.bulk_reset_guard_enabled is True
    assert cfg.bulk_reset_guard_threshold == 10
    assert cfg.bulk_reset_guard_window_seconds == 60.0
    assert cfg.bulk_reset_guard_escalation_rate_limit_seconds == 900.0


# ---------------------------------------------------------------------------
# step-3: observe_attempt returns ok for all calls under threshold
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_observe_attempt_returns_ok_under_threshold(tmp_path):
    """Nine done→pending attempts (below threshold of 10) all return ok."""
    clock = [1000.0]

    def fake_clock() -> float:
        return clock[0]

    guard = BulkResetGuard(
        threshold=10,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        time_provider=fake_clock,
        escalations_fallback_dir=tmp_path,
    )

    for i in range(9):
        clock[0] += 1.0
        verdict = await guard.observe_attempt(
            project_id='proj-a',
            task_id=str(i),
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
        assert verdict.outcome == 'ok', f'attempt {i}: expected ok, got {verdict.outcome}'
        assert verdict.is_rejection is False
