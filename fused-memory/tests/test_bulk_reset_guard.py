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
