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


# ---------------------------------------------------------------------------
# step-5: threshold-trip rejects the N-th and subsequent attempts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_observe_attempt_rejects_at_threshold_within_window(tmp_path):
    """With threshold=3, attempts 1-3 are ok; attempt 4 trips the guard."""
    clock = [1000.0]

    def fake_clock() -> float:
        return clock[0]

    guard = BulkResetGuard(
        threshold=3,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        time_provider=fake_clock,
        escalations_fallback_dir=tmp_path,
    )

    task_ids = ['task-a', 'task-b', 'task-c', 'task-d']
    verdicts = []
    for tid in task_ids:
        clock[0] += 1.0
        v = await guard.observe_attempt(
            project_id='proj-a',
            task_id=tid,
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
        verdicts.append(v)

    # First three: ok
    for i in range(3):
        assert verdicts[i].outcome == 'ok', f'attempt {i}: expected ok'

    # Fourth: tripped
    v4 = verdicts[3]
    assert v4.outcome in {'rejection', 'escalated'}, (
        f'attempt 4: expected rejection/escalated, got {v4.outcome}'
    )
    assert v4.is_rejection is True
    assert v4.error_type == 'BulkResetGuardTripped'
    assert v4.threshold == 3
    assert v4.window_seconds == 60.0
    # All four task IDs should be in affected_task_ids
    assert set(v4.affected_task_ids) == set(task_ids)
    assert len(v4.triggering_timestamps) == 4


# ---------------------------------------------------------------------------
# step-7: non-reversal transitions do not consume window slots
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_observe_attempt_ignores_non_reversal_transitions(tmp_path):
    """Non-reversal transitions never count; reversals still trip at threshold."""
    clock = [1000.0]

    def fake_clock() -> float:
        return clock[0]

    guard = BulkResetGuard(
        threshold=3,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        time_provider=fake_clock,
        escalations_fallback_dir=tmp_path,
    )

    non_reversals = [
        ('pending', 'in-progress'),
        ('in-progress', 'done'),
        ('blocked', 'pending'),
        ('pending', 'done'),
        ('done', 'blocked'),
    ]
    for old, new in non_reversals:
        clock[0] += 1.0
        v = await guard.observe_attempt(
            project_id='proj-a',
            task_id=f'{old}->{new}',
            old_status=old,
            new_status=new,
            project_root=str(tmp_path),
        )
        assert v.outcome == 'ok', f'{old}→{new}: expected ok, got {v.outcome}'

    # Now fire three reversal attempts — should all be ok (threshold=3 allows 3)
    for i in range(3):
        clock[0] += 1.0
        v = await guard.observe_attempt(
            project_id='proj-a',
            task_id=f'rev-{i}',
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
        assert v.outcome == 'ok', f'reversal {i}: expected ok, got {v.outcome}'

    # Fourth reversal should trip (non-reversals did not consume window slots)
    clock[0] += 1.0
    v4 = await guard.observe_attempt(
        project_id='proj-a',
        task_id='rev-3',
        old_status='in-progress',
        new_status='pending',
        project_root=str(tmp_path),
    )
    assert v4.is_rejection is True, (
        f'Expected rejection on 4th reversal, got {v4.outcome}'
    )
