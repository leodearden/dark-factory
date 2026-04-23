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


# ---------------------------------------------------------------------------
# step-9: window expiry drains old entries
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_window_drains_after_expiry(tmp_path):
    """Entries older than window_seconds are pruned; the guard resets.

    With threshold=3 and window=60:
      - Three reversals at t=1000,1001,1002 are all ok.
      - Advance clock past the window so all three original entries expire.
      - Three new reversals are ok again (window is fresh).
      - The fourth new reversal (threshold+1 in the refreshed window) trips.
      - The rejection's triggering_timestamps contain ONLY the fresh timestamps.
    """
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

    # Fire three reversals in the initial window (all ok at threshold=3).
    original_task_ids = ['orig-0', 'orig-1', 'orig-2']
    for tid in original_task_ids:
        clock[0] += 1.0
        v = await guard.observe_attempt(
            project_id='proj-a',
            task_id=tid,
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
        assert v.outcome == 'ok', f'{tid}: expected ok, got {v.outcome}'

    # After firing three the guard is at the limit (len=3 == threshold, ok).
    # Advance clock so that ALL original entries (t=1001,1002,1003) have aged out.
    # Cutoff must be > 1003: set t=1064 (cutoff=1004 > 1003 ✓).
    clock[0] = 1064.0

    # First three fresh reversals should all be ok (window is reset).
    fresh_task_ids = ['fresh-0', 'fresh-1', 'fresh-2', 'fresh-3']
    for tid in fresh_task_ids[:3]:
        clock[0] += 1.0
        v = await guard.observe_attempt(
            project_id='proj-a',
            task_id=tid,
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
        assert v.outcome == 'ok', f'{tid}: expected ok, got {v.outcome}'

    # The fourth fresh reversal (threshold+1 in the refreshed window) should trip.
    clock[0] += 1.0
    v_trip = await guard.observe_attempt(
        project_id='proj-a',
        task_id='fresh-3',
        old_status='done',
        new_status='pending',
        project_root=str(tmp_path),
    )
    assert v_trip.is_rejection is True, (
        f'Expected rejection, got {v_trip.outcome}'
    )
    # Only the fresh timestamps should appear — expired entries must not be included.
    trip_ts_set = set(v_trip.triggering_timestamps)
    # None of the original timestamps should be in the trip
    for orig in v_trip.triggering_timestamps:
        # All timestamps must be >= t=1065 (the first fresh attempt)
        from datetime import datetime, timezone
        ts_val = datetime.fromisoformat(orig).timestamp()
        assert ts_val >= 1065.0, (
            f'triggering_timestamps contains stale entry at {orig} (ts={ts_val})'
        )
    assert len(v_trip.triggering_timestamps) == 4, (
        f'Expected 4 triggering timestamps (threshold+1), got '
        f'{len(v_trip.triggering_timestamps)}: {v_trip.triggering_timestamps}'
    )


# ---------------------------------------------------------------------------
# step-11: per-project isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_per_project_isolation(tmp_path):
    """Each project's reversal count is tracked independently."""
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

    # Interleave three reversals on proj-a and three on proj-b (all ok).
    for i in range(3):
        clock[0] += 1.0
        va = await guard.observe_attempt(
            project_id='proj-a',
            task_id=f'a-{i}',
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
        assert va.outcome == 'ok', f'proj-a attempt {i}: expected ok'

        clock[0] += 1.0
        vb = await guard.observe_attempt(
            project_id='proj-b',
            task_id=f'b-{i}',
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
        assert vb.outcome == 'ok', f'proj-b attempt {i}: expected ok'

    # Fourth on proj-a trips proj-a independently.
    clock[0] += 1.0
    va4 = await guard.observe_attempt(
        project_id='proj-a',
        task_id='a-3',
        old_status='done',
        new_status='pending',
        project_root=str(tmp_path),
    )
    assert va4.is_rejection is True, (
        f'proj-a 4th: expected rejection, got {va4.outcome}'
    )
    assert va4.project_id == 'proj-a'

    # Fourth on proj-b trips proj-b independently.
    clock[0] += 1.0
    vb4 = await guard.observe_attempt(
        project_id='proj-b',
        task_id='b-3',
        old_status='done',
        new_status='pending',
        project_root=str(tmp_path),
    )
    assert vb4.is_rejection is True, (
        f'proj-b 4th: expected rejection, got {vb4.outcome}'
    )
    assert vb4.project_id == 'proj-b'


# ---------------------------------------------------------------------------
# step-13: L1 escalation file written on trip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emits_l1_escalation_on_trip(tmp_path):
    """When the guard trips, an L1 escalation JSON is written and the verdict
    carries outcome=='escalated' with escalation_path pointing to the file."""
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

    task_ids = ['t1', 't2', 't3', 't4']

    # First three: ok
    for tid in task_ids[:3]:
        clock[0] += 1.0
        v = await guard.observe_attempt(
            project_id='proj-esc',
            task_id=tid,
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
        assert v.outcome == 'ok', f'{tid}: expected ok'

    # Fourth: should escalate
    clock[0] += 1.0
    v4 = await guard.observe_attempt(
        project_id='proj-esc',
        task_id='t4',
        old_status='done',
        new_status='pending',
        project_root=str(tmp_path),
    )
    assert v4.outcome == 'escalated', f'Expected escalated, got {v4.outcome}'
    assert v4.escalation_path is not None

    # File must exist
    esc_file = Path(v4.escalation_path)
    assert esc_file.exists(), f'Escalation file not found at {v4.escalation_path}'
    assert str(esc_file).startswith(str(tmp_path / 'data' / 'escalations'))

    # Parse and validate contents
    data = json.loads(esc_file.read_text(encoding='utf-8'))
    assert data['id'].startswith('esc-bulk-reset-')
    assert data['task_id'] is None
    assert data['agent_role'] == 'fused-memory'
    assert data['severity'] == 'blocking'
    assert data['category'] == 'infra_issue'
    assert data['level'] == 1
    assert data['status'] == 'pending'
    assert data['workflow_state'] == 'infra'

    # Guard-specific fields
    assert set(data['affected_task_ids']) == set(task_ids)
    assert len(data['affected_task_ids']) == 4
    assert len(data['triggering_timestamps']) == 4
    # Each triggering timestamp must be a parseable ISO string
    from datetime import datetime
    for ts_str in data['triggering_timestamps']:
        datetime.fromisoformat(ts_str)  # raises if not parseable
    assert data['threshold'] == 3
    assert data['window_seconds'] == 60.0
    assert data['project_id'] == 'proj-esc'


# ---------------------------------------------------------------------------
# step-15: escalation rate-limiting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_escalation_rate_limited(tmp_path):
    """A second trip within the rate-limit window returns 'rejection', not 'escalated',
    and does NOT write a new escalation file."""
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

    esc_dir = tmp_path / 'data' / 'escalations'

    # Fire four reversals — fourth trips and writes escalation X.
    for i in range(4):
        clock[0] += 1.0
        v = await guard.observe_attempt(
            project_id='proj-rl',
            task_id=f'rl-{i}',
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
    # v is now the fourth verdict (escalated)
    assert v.outcome == 'escalated'
    first_esc_path = v.escalation_path
    assert first_esc_path is not None

    # Advance clock by 1s (well within both the 60s window and 900s rate-limit).
    # Advancing by 60s would prune most entries from the window (making the
    # guard return ok rather than reaching the rate-limit check), so we use 1s
    # to keep entries in the window while exercising the rate-limit path.
    clock[0] += 1.0

    # Fire another reversal — still inside the window AND inside the rate-limit.
    v5 = await guard.observe_attempt(
        project_id='proj-rl',
        task_id='rl-4',
        old_status='done',
        new_status='pending',
        project_root=str(tmp_path),
    )
    assert v5.outcome == 'rejection', (
        f'Expected rejection (rate-limited), got {v5.outcome}'
    )
    assert v5.is_rejection is True

    # Only one escalation file should exist (not a second one).
    esc_files = list(esc_dir.glob('*.json'))
    assert len(esc_files) == 1, (
        f'Expected 1 escalation file, found {len(esc_files)}: {esc_files}'
    )

    # Advance clock past both the rate-limit AND the window so a fresh trip fires.
    # Rate-limit is 900s; window is 60s. Move to t = 1000 + 1000 = 2000 to be safe.
    clock[0] = 2000.0

    # Seed three new reversals (all ok at threshold=3).
    for i in range(3):
        clock[0] += 1.0
        v = await guard.observe_attempt(
            project_id='proj-rl',
            task_id=f'new-{i}',
            old_status='in-progress',
            new_status='pending',
            project_root=str(tmp_path),
        )
        assert v.outcome == 'ok'

    # Fourth new reversal — should write a SECOND escalation file.
    clock[0] += 1.0
    v_new = await guard.observe_attempt(
        project_id='proj-rl',
        task_id='new-3',
        old_status='in-progress',
        new_status='pending',
        project_root=str(tmp_path),
    )
    assert v_new.outcome == 'escalated', (
        f'Expected escalated (rate-limit reset), got {v_new.outcome}'
    )
    esc_files = list(esc_dir.glob('*.json'))
    assert len(esc_files) == 2, (
        f'Expected 2 escalation files, found {len(esc_files)}'
    )


# ---------------------------------------------------------------------------
# step-17: disabled guard always returns ok
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_guard_disabled_returns_ok(tmp_path):
    """With enabled=False the guard is a no-op: all attempts return ok,
    no escalation dir is created, and no internal state is accumulated."""
    clock = [1000.0]

    def fake_clock() -> float:
        return clock[0]

    guard = BulkResetGuard(
        enabled=False,
        threshold=1,  # Would trip on the first attempt if enabled
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        time_provider=fake_clock,
        escalations_fallback_dir=tmp_path,
    )

    for i in range(5):
        clock[0] += 1.0
        v = await guard.observe_attempt(
            project_id='proj-disabled',
            task_id=f'task-{i}',
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
        assert v.outcome == 'ok', f'attempt {i}: expected ok, got {v.outcome}'
        assert v.is_rejection is False

    # No escalation dir should have been created
    esc_dir = tmp_path / 'data' / 'escalations'
    assert not esc_dir.exists(), 'escalations dir should not exist when guard is disabled'

    # Internal state deque should be empty (no entries accumulated)
    assert not guard._state, 'guard state should be empty when disabled'


# ---------------------------------------------------------------------------
# step-19: BulkResetVerdict.to_error_dict() shape
# ---------------------------------------------------------------------------

def test_verdict_to_error_dict_ok():
    """outcome='ok' produces an empty dict."""
    v = BulkResetVerdict(outcome='ok', project_id='p')
    assert v.to_error_dict() == {}


def test_verdict_to_error_dict_rejection():
    """outcome='rejection' produces a structured error payload."""
    v = BulkResetVerdict(
        outcome='rejection',
        affected_task_ids=('5', '6', '7'),
        triggering_timestamps=(
            '2026-04-23T00:00:00+00:00',
            '2026-04-23T00:00:01+00:00',
            '2026-04-23T00:00:02+00:00',
        ),
        threshold=10,
        window_seconds=60.0,
        project_id='proj',
        error_type='BulkResetGuardTripped',
    )
    d = v.to_error_dict()
    assert d['success'] is False
    assert d['error_type'] == 'BulkResetGuardTripped'
    assert 'BulkResetGuardTripped' in d['error']
    assert str(d['threshold']) in d['error'] or d['threshold'] == 10
    assert str(d['window_seconds']) in d['error'] or d['window_seconds'] == 60.0
    assert d['affected_task_ids'] == ['5', '6', '7']
    assert d['triggering_timestamps'] == [
        '2026-04-23T00:00:00+00:00',
        '2026-04-23T00:00:01+00:00',
        '2026-04-23T00:00:02+00:00',
    ]
    assert d['threshold'] == 10
    assert d['window_seconds'] == 60.0
    assert d['project_id'] == 'proj'
    assert 'hint' in d
    # No escalation_path for plain rejection
    assert 'escalation_path' not in d


def test_verdict_to_error_dict_escalated():
    """outcome='escalated' produces the same rejection payload PLUS escalation_path."""
    v = BulkResetVerdict(
        outcome='escalated',
        affected_task_ids=('t1', 't2'),
        triggering_timestamps=('2026-04-23T00:00:00+00:00', '2026-04-23T00:00:01+00:00'),
        threshold=3,
        window_seconds=60.0,
        project_id='proj',
        error_type='BulkResetGuardTripped',
        escalation_path='/tmp/esc-bulk-reset-xyz.json',
    )
    d = v.to_error_dict()
    assert d['success'] is False
    assert d['error_type'] == 'BulkResetGuardTripped'
    assert d['escalation_path'] == '/tmp/esc-bulk-reset-xyz.json'
    assert d['project_id'] == 'proj'
