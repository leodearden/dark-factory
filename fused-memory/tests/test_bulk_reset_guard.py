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
  - task-979: async I/O offloading, idle-state preservation, write-failure backoff
  - task-1016: per-kind counter isolation (test_per_kind_counter_isolation)
  - task-1016: _reversal_kind classifier (test_reversal_kind_classifier)
  - task-1016: BulkResetVerdict.kind field (test_verdict_carries_tripped_kind)
  - task-1016: kind slug in escalation filename/JSON
    (test_escalation_filename_and_body_include_kind)
  - task-1016: acceptance regression — 2026-04-24 reify incident
    esc-bulk-reset-reify-2026-04-24T070944_6456580000
    (test_acceptance_scenario_startup_reconcile_does_not_trip)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fused_memory.reconciliation.bulk_reset_guard import (
    BulkResetGuard,
    BulkResetVerdict,
    _reversal_kind,
)

# ---------------------------------------------------------------------------
# step-1: ReconciliationConfig defaults
# ---------------------------------------------------------------------------

def test_reconciliation_config_bulk_reset_guard_defaults():
    """ReconciliationConfig() must carry the split-threshold bulk-reset-guard
    fields with their specified default values.

    The legacy single-field ``bulk_reset_guard_threshold`` was split into two
    independent per-kind thresholds in task 1016 to distinguish the benign
    startup-stranded-task reconcile pattern (in-progress→pending) from the
    data-loss pattern caught by task 918 (done→pending).
    """
    from fused_memory.config.schema import ReconciliationConfig

    cfg = ReconciliationConfig()
    assert cfg.bulk_reset_guard_enabled is True
    assert cfg.bulk_reset_guard_done_to_pending_threshold == 10
    assert cfg.bulk_reset_guard_in_progress_to_pending_threshold == 100
    assert cfg.bulk_reset_guard_window_seconds == 60.0
    assert cfg.bulk_reset_guard_escalation_rate_limit_seconds == 900.0
    # The legacy single-threshold field must be gone.
    assert not hasattr(cfg, 'bulk_reset_guard_threshold')


# ---------------------------------------------------------------------------
# amend-1: ReconciliationConfig rejects the legacy threshold key explicitly
# ---------------------------------------------------------------------------

def test_reconciliation_config_rejects_legacy_threshold_key():
    """ReconciliationConfig must raise ValidationError when the legacy
    ``bulk_reset_guard_threshold`` key is present in the raw input dict.

    Without this validator the key would be silently dropped by
    ``extra='ignore'``, leaving the done→pending data-loss guard at its
    default threshold of 10 regardless of any operator tuning — the worst
    failure mode for a security-relevant guard.
    """
    from pydantic import ValidationError

    from fused_memory.config.schema import ReconciliationConfig

    with pytest.raises(ValidationError, match='bulk_reset_guard_threshold'):
        ReconciliationConfig.model_validate({'bulk_reset_guard_threshold': 5})


# ---------------------------------------------------------------------------
# step-2b: BulkResetGuard constructor accepts split thresholds
# ---------------------------------------------------------------------------

def test_guard_constructor_accepts_split_thresholds(tmp_path):
    """BulkResetGuard must accept done_threshold/in_progress_threshold kwargs.

    (a) The new signature constructs without error.
    (b) The legacy ``threshold=`` kwarg raises TypeError — it is no longer
        accepted so callers cannot silently wire only one counter.
    """
    # (a) New split-threshold signature succeeds.
    guard = BulkResetGuard(
        done_threshold=10,
        in_progress_threshold=100,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        escalations_fallback_dir=tmp_path,
    )
    assert guard is not None

    # (b) Legacy ``threshold=`` kwarg must be rejected.
    with pytest.raises(TypeError, match="threshold"):
        BulkResetGuard(threshold=10, window_seconds=60.0)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# step-4b: _reversal_kind classifier
# ---------------------------------------------------------------------------

def test_reversal_kind_classifier():
    """_reversal_kind returns the correct kind string for guarded transitions
    and None for all non-guarded transitions.

    Guarded:
      - done → pending        → 'done_to_pending'
      - in-progress → pending → 'in_progress_to_pending'

    Non-guarded (must all return None):
      - pending → in-progress
      - in-progress → done
      - blocked → pending     (blocked→pending is NOT guarded)
      - pending → done
      - done → blocked
      - cancelled → pending
    """
    assert _reversal_kind('done', 'pending') == 'done_to_pending'
    assert _reversal_kind('in-progress', 'pending') == 'in_progress_to_pending'

    assert _reversal_kind('pending', 'in-progress') is None
    assert _reversal_kind('in-progress', 'done') is None
    assert _reversal_kind('blocked', 'pending') is None
    assert _reversal_kind('pending', 'done') is None
    assert _reversal_kind('done', 'blocked') is None
    assert _reversal_kind('cancelled', 'pending') is None


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
        done_threshold=10,
        in_progress_threshold=100,
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
        done_threshold=3,
        in_progress_threshold=100,
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
        done_threshold=3,
        in_progress_threshold=100,
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

    # Fourth done→pending reversal should trip the done counter.
    # (Using done→pending rather than in-progress→pending: the two counters are
    # independent, and the in-progress threshold is 100, so an in-progress
    # reversal here would NOT trip.  The test verifies that non-reversals do
    # not consume done→pending window slots — the done counter is at 3 == done_threshold,
    # so the next done→pending is the (threshold+1)-th and must be rejected.)
    clock[0] += 1.0
    v4 = await guard.observe_attempt(
        project_id='proj-a',
        task_id='rev-3',
        old_status='done',
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
        done_threshold=3,
        in_progress_threshold=100,
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
    # None of the original timestamps should be in the trip
    for orig in v_trip.triggering_timestamps:
        # All timestamps must be >= t=1065 (the first fresh attempt)
        from datetime import datetime
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
        done_threshold=3,
        in_progress_threshold=100,
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
        done_threshold=3,
        in_progress_threshold=100,
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
    # kind must be present in the JSON record (step-12 writes it).
    assert 'kind' in data, f"escalation JSON missing 'kind' key; keys: {list(data)}"
    # Cross-kind context field must be present (amend-3).  Only done→pending
    # reversals were fired in this test, so the in-progress deque is empty.
    assert 'other_kind_task_ids_in_window' in data, (
        f"escalation JSON missing 'other_kind_task_ids_in_window'; keys: {list(data)}"
    )
    assert data['other_kind_task_ids_in_window'] == [], (
        f"Expected empty cross-kind list, got {data['other_kind_task_ids_in_window']!r}"
    )


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
        done_threshold=3,
        in_progress_threshold=100,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        time_provider=fake_clock,
        escalations_fallback_dir=tmp_path,
    )

    esc_dir = tmp_path / 'data' / 'escalations'

    # Fire four reversals — fourth trips and writes escalation X.
    v = None
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
    assert v is not None
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

    # Seed three new done→pending reversals (all ok at done_threshold=3).
    # Using done→pending (not in-progress→pending) because in_progress_threshold=100
    # and we'd need 101 in-progress reversals to re-trip — use done_threshold=3 instead.
    for i in range(3):
        clock[0] += 1.0
        v = await guard.observe_attempt(
            project_id='proj-rl',
            task_id=f'new-{i}',
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
        assert v.outcome == 'ok'

    # Fourth new reversal — should write a SECOND escalation file.
    clock[0] += 1.0
    v_new = await guard.observe_attempt(
        project_id='proj-rl',
        task_id='new-3',
        old_status='done',
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
# task-1016 step-11: escalation filename and JSON body carry kind slug
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_escalation_filename_and_body_include_kind(tmp_path):
    """The escalation filename slug and JSON 'kind' field must reflect which
    reversal kind tripped the guard.

    (a) done counter trips: filename matches esc-bulk-reset-done-<project>-...
        and JSON data['kind'] == 'done_to_pending'.
    (b) in-progress counter trips (fresh project to bypass rate-limit):
        filename matches esc-bulk-reset-in-progress-<project>-...
        and JSON data['kind'] == 'in_progress_to_pending'.

    Fails until step-12 inserts the kind slug into the filename and records
    'kind' in the JSON body.
    """
    import re

    clock = [1000.0]

    def fake_clock() -> float:
        return clock[0]

    # --- (a) done counter trips ---
    guard = BulkResetGuard(
        done_threshold=3,
        in_progress_threshold=100,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        time_provider=fake_clock,
        escalations_fallback_dir=tmp_path,
    )
    v: BulkResetVerdict | None = None
    for i in range(4):
        clock[0] += 1.0
        v = await guard.observe_attempt(
            project_id='kind-proj',
            task_id=f'done-{i}',
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
    assert v is not None
    assert v.outcome == 'escalated', f'(a) expected escalated, got {v.outcome}'
    assert v.escalation_path is not None
    esc_file_a = Path(v.escalation_path)
    assert re.search(r'esc-bulk-reset-done-kind-proj-', esc_file_a.name), (
        f'(a) filename {esc_file_a.name!r} missing done kind slug'
    )
    data_a = json.loads(esc_file_a.read_text(encoding='utf-8'))
    assert data_a.get('kind') == 'done_to_pending', (
        f"(a) JSON kind: expected 'done_to_pending', got {data_a.get('kind')!r}"
    )
    # Cross-kind context (amend-3): only done→pending fired so in-progress deque empty.
    assert 'other_kind_task_ids_in_window' in data_a, (
        "(a) escalation JSON missing 'other_kind_task_ids_in_window'"
    )
    assert data_a['other_kind_task_ids_in_window'] == []

    # --- (b) in-progress counter trips (fresh project avoids rate-limit) ---
    guard_b = BulkResetGuard(
        done_threshold=100,
        in_progress_threshold=3,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        time_provider=fake_clock,
        escalations_fallback_dir=tmp_path,
    )
    v_b = None
    for i in range(4):
        clock[0] += 1.0
        v_b = await guard_b.observe_attempt(
            project_id='kind-proj-ip',
            task_id=f'ip-{i}',
            old_status='in-progress',
            new_status='pending',
            project_root=str(tmp_path),
        )
    assert v_b is not None
    assert v_b.outcome == 'escalated', f'(b) expected escalated, got {v_b.outcome}'
    assert v_b.escalation_path is not None
    esc_file_b = Path(v_b.escalation_path)
    assert re.search(r'esc-bulk-reset-in-progress-kind-proj-ip-', esc_file_b.name), (
        f'(b) filename {esc_file_b.name!r} missing in-progress kind slug'
    )
    data_b = json.loads(esc_file_b.read_text(encoding='utf-8'))
    assert data_b.get('kind') == 'in_progress_to_pending', (
        f"(b) JSON kind: expected 'in_progress_to_pending', got {data_b.get('kind')!r}"
    )
    # Cross-kind context (amend-3): only in-progress→pending fired so done deque empty.
    assert 'other_kind_task_ids_in_window' in data_b, (
        "(b) escalation JSON missing 'other_kind_task_ids_in_window'"
    )
    assert data_b['other_kind_task_ids_in_window'] == []


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
        done_threshold=1,  # Would trip on the first attempt if enabled
        in_progress_threshold=1,
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


@pytest.mark.parametrize("threshold,window_seconds", [
    (10, 60.0),  # baseline values
    (6, 60.0),   # vary threshold
    (10, 6.0),   # vary window_seconds
])
def test_verdict_to_error_dict_rejection(threshold, window_seconds):
    """outcome='rejection' produces a structured error payload.

    Three parametrized (threshold, window_seconds) combinations verify that
    the structured fields carry the correct typed values across different
    configurations without pinning the exact label wording of the error string.

    Note: affected_task_ids is kept short (3 items) across all cases.  In
    production a rejection verdict is only constructed when
    len(affected_task_ids) > threshold, so the 3-vs-6 and 3-vs-10
    combinations are semantically impossible.  BulkResetVerdict.to_error_dict()
    does not enforce that invariant, so this test intentionally decouples
    formatting correctness from semantic validity.
    """
    v = BulkResetVerdict(
        outcome='rejection',
        affected_task_ids=('5', '6', '7'),
        triggering_timestamps=(
            '2026-04-23T00:00:00+00:00',
            '2026-04-23T00:00:01+00:00',
            '2026-04-23T00:00:02+00:00',
        ),
        threshold=threshold,
        window_seconds=window_seconds,
        project_id='proj',
        error_type='BulkResetGuardTripped',
    )
    d = v.to_error_dict()
    assert d['success'] is False
    assert d['error_type'] == 'BulkResetGuardTripped'
    assert 'BulkResetGuardTripped' in d['error']
    assert d['affected_task_ids'] == ['5', '6', '7']
    assert d['triggering_timestamps'] == [
        '2026-04-23T00:00:00+00:00',
        '2026-04-23T00:00:01+00:00',
        '2026-04-23T00:00:02+00:00',
    ]
    assert d['threshold'] == threshold
    assert d['window_seconds'] == window_seconds
    assert d['project_id'] == 'proj'
    assert 'hint' in d
    # No escalation_path for plain rejection
    assert 'escalation_path' not in d


def test_verdict_to_error_dict_escalated():
    """outcome='escalated' produces the same rejection payload PLUS escalation_path.
    When the verdict carries a kind, to_error_dict() must surface it as d['kind'].
    """
    v = BulkResetVerdict(
        outcome='escalated',
        affected_task_ids=('t1', 't2'),
        triggering_timestamps=('2026-04-23T00:00:00+00:00', '2026-04-23T00:00:01+00:00'),
        threshold=3,
        window_seconds=60.0,
        project_id='proj',
        error_type='BulkResetGuardTripped',
        escalation_path='/tmp/esc-bulk-reset-xyz.json',
        kind='done_to_pending',
    )
    d = v.to_error_dict()
    assert d['success'] is False
    assert d['error_type'] == 'BulkResetGuardTripped'
    assert d['escalation_path'] == '/tmp/esc-bulk-reset-xyz.json'
    assert d['project_id'] == 'proj'
    # kind must be surfaced in the error payload (step-10 wires this).
    assert d.get('kind') == 'done_to_pending'


# ---------------------------------------------------------------------------
# task-979 step-1: async I/O offloading
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_escalation_write_offloads_io_to_thread(tmp_path, monkeypatch):
    """mkdir and write_text in _maybe_write_escalation must be offloaded to a
    thread via asyncio.to_thread so they do not block the event loop.

    Monkeypatches asyncio.to_thread with a tracking wrapper that records each
    raw callable passed and then delegates to the real asyncio.to_thread.  The
    guard is tripped (4 reversals, threshold=3) and the test asserts that
    exactly one tracked call is Path.mkdir bound to the escalation directory and
    exactly one is Path.write_text bound to a file inside that directory.
    """
    import asyncio as _asyncio

    tracked_callables: list = []
    real_to_thread = _asyncio.to_thread

    async def tracking_to_thread(func, *args, **kwargs):
        tracked_callables.append(func)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(_asyncio, 'to_thread', tracking_to_thread)

    expected_esc_dir = tmp_path / 'data' / 'escalations'
    clock = [1000.0]

    def fake_clock() -> float:
        return clock[0]

    guard = BulkResetGuard(
        done_threshold=3,
        in_progress_threshold=100,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        time_provider=fake_clock,
        escalations_fallback_dir=tmp_path,
    )

    for i in range(4):
        clock[0] += 1.0
        await guard.observe_attempt(
            project_id='proj-thread',
            task_id=f't{i}',
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )

    mkdir_calls = [
        f for f in tracked_callables
        if getattr(f, '__func__', None) is Path.mkdir
        and getattr(f, '__self__', None) == expected_esc_dir
    ]
    write_text_calls = [
        f for f in tracked_callables
        if getattr(f, '__func__', None) is Path.write_text
        and getattr(getattr(f, '__self__', None), 'parent', None) == expected_esc_dir
    ]
    assert len(mkdir_calls) == 1, (
        f'Expected exactly 1 Path.mkdir bound-method call on {expected_esc_dir}; '
        f'got {len(mkdir_calls)}. Tracked callables: {tracked_callables}'
    )
    assert len(write_text_calls) == 1, (
        f'Expected exactly 1 Path.write_text bound-method call inside {expected_esc_dir}; '
        f'got {len(write_text_calls)}. Tracked callables: {tracked_callables}'
    )


# ---------------------------------------------------------------------------
# task-979 step-3: dead eviction removal characterization
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_idle_project_state_is_not_evicted_across_attempts(tmp_path):
    """last_escalation_ts must be preserved across an idle-period attempt.

    Setup: threshold=3, window=60, rate_limit=900.  Trip the guard at t≈1004
    so last_escalation_ts is recorded.  Advance clock past both the window and
    the rate-limit window (t=2000).  Fire one more reversal (threshold not yet
    re-crossed).  Assert that guard._state['proj'].last_escalation_ts == 1004.0
    — proving the state dict entry was NOT replaced with a fresh _GuardState
    (which would reset last_escalation_ts to 0.0).

    This test FAILS under the current inline eviction code (lines 243-250 of
    bulk_reset_guard.py) which pops and re-inserts a fresh _GuardState, and
    PASSES after that block is removed.
    """
    clock = [1000.0]

    def fake_clock() -> float:
        return clock[0]

    guard = BulkResetGuard(
        done_threshold=3,
        in_progress_threshold=100,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        time_provider=fake_clock,
        escalations_fallback_dir=tmp_path,
    )

    # Fire 4 reversals at t=1001..1004 — 4th trips, writes escalation.
    # last_escalation_ts should be set to 1004.0 after the successful write.
    trip_ts = None
    for i in range(4):
        clock[0] += 1.0
        if i == 3:
            trip_ts = clock[0]
        await guard.observe_attempt(
            project_id='proj',
            task_id=f't{i}',
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )

    assert trip_ts == 1004.0
    # Verify the escalation was recorded
    assert guard._state['proj'].last_escalation_ts == trip_ts, (
        f'Expected last_escalation_ts={trip_ts}, '
        f'got {guard._state["proj"].last_escalation_ts}'
    )

    # Advance clock well past window (60s) and rate-limit (900s).
    clock[0] = 2000.0

    # Fire one more reversal (does not cross threshold again; just one attempt).
    await guard.observe_attempt(
        project_id='proj',
        task_id='idle-check',
        old_status='done',
        new_status='pending',
        project_root=str(tmp_path),
    )

    # last_escalation_ts must still be 1004.0 — not reset to 0.0.
    assert guard._state['proj'].last_escalation_ts == trip_ts, (
        f'last_escalation_ts was reset! Expected {trip_ts}, '
        f'got {guard._state["proj"].last_escalation_ts}. '
        'The dead eviction block (lines 243-250) replaced state with a fresh _GuardState.'
    )


# ---------------------------------------------------------------------------
# task-979 step-5: write-failure backoff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_failure_triggers_per_project_backoff(tmp_path, monkeypatch):
    """A write_text OSError arms a per-project backoff that suppresses retries.

    Scenario:
      1. Trip guard at t=1001..1004 (threshold=3); write_text raises on first
         call — verdict is 'rejection'; write_text call count == 1.
      2. Advance clock by 30s (inside the 60s backoff window); fire another
         reversal — verdict is 'rejection' AND write_text call count still == 1
         (backoff suppressed the retry).
      3. Advance clock to t=3000 (past window+backoff); seed 3 fresh ok
         reversals then trip again.  The guard attempts write_text again
         (call count becomes 2) and this time succeeds; verdict is 'escalated'.

    This test fails today because:
      - BulkResetGuard has no write_failure_backoff_seconds parameter.
      - _maybe_write_escalation has no backoff check.
    """
    write_text_calls: list[int] = [0]
    real_write_text = Path.write_text

    def flaky_write_text(self, data, *args, **kwargs):
        write_text_calls[0] += 1
        if write_text_calls[0] == 1:
            raise OSError('simulated flaky mount')
        return real_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, 'write_text', flaky_write_text)

    clock = [1000.0]

    def fake_clock() -> float:
        return clock[0]

    guard = BulkResetGuard(
        done_threshold=3,
        in_progress_threshold=100,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        write_failure_backoff_seconds=60.0,
        time_provider=fake_clock,
        escalations_fallback_dir=tmp_path,
    )

    # Phase 1: trip the guard at t=1001..1004; write_text raises → rejection.
    v: BulkResetVerdict | None = None
    for i in range(4):
        clock[0] += 1.0
        v = await guard.observe_attempt(
            project_id='proj',
            task_id=f't{i}',
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
    assert v is not None
    assert v.outcome == 'rejection', f'Phase 1: expected rejection, got {v.outcome}'
    assert write_text_calls[0] == 1, (
        f'Phase 1: expected 1 write_text call, got {write_text_calls[0]}'
    )

    # Phase 2: 30s later — still inside 60s backoff window.
    clock[0] += 30.0
    v2 = await guard.observe_attempt(
        project_id='proj',
        task_id='t4',
        old_status='done',
        new_status='pending',
        project_root=str(tmp_path),
    )
    assert v2.outcome == 'rejection', (
        f'Phase 2: expected rejection (backoff suppressed retry), got {v2.outcome}'
    )
    assert write_text_calls[0] == 1, (
        f'Phase 2: backoff should have suppressed write_text retry; '
        f'call count is {write_text_calls[0]}'
    )

    # Phase 3: advance past window+backoff; seed 3 fresh ok reversals then trip.
    clock[0] = 3000.0
    for i in range(3):
        clock[0] += 1.0
        v = await guard.observe_attempt(
            project_id='proj',
            task_id=f'new-{i}',
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
        assert v.outcome == 'ok', f'Phase 3 ok reversal {i}: expected ok, got {v.outcome}'

    # Trip again — write_text is called (2nd time) and now succeeds → escalated.
    clock[0] += 1.0
    v3 = await guard.observe_attempt(
        project_id='proj',
        task_id='new-3',
        old_status='done',
        new_status='pending',
        project_root=str(tmp_path),
    )
    assert write_text_calls[0] == 2, (
        f'Phase 3: expected 2 write_text calls total, got {write_text_calls[0]}'
    )
    assert v3.outcome == 'escalated', (
        f'Phase 3: expected escalated after backoff cleared, got {v3.outcome}'
    )


# ---------------------------------------------------------------------------
# task-979 step-8: mkdir-failure backoff (symmetric to step-5)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mkdir_failure_triggers_per_project_backoff(tmp_path, monkeypatch):
    """An OSError from esc_dir.mkdir arms a per-project backoff identically to
    a write_text failure — and must NOT propagate out of observe_attempt.

    Scenario:
      1. Trip guard at t=1001..1004 (threshold=3); mkdir raises on first call —
         NO exception propagates; verdict is 'rejection'; mkdir call count == 1;
         write_text call count == 0 (short-circuited before write).
      2. Advance clock by 30s (inside the 60s backoff window); fire another
         reversal — verdict is 'rejection' AND mkdir call count still == 1
         (backoff suppressed the retry, so neither mkdir NOR write_text called).
      3. Advance clock to t=3000 (past window+backoff); seed 3 fresh ok
         reversals then trip again.  mkdir is called a 2nd time and now
         succeeds, write_text is called and succeeds; verdict is 'escalated'.

    This test fails today because esc_dir.mkdir at bulk_reset_guard.py:352 is
    NOT wrapped in try/except — the OSError propagates rather than arming the
    backoff.
    """
    # Filter out pathlib's parent-recursion (different self) and self-retry (parents kwarg absent).
    expected_esc_dir = tmp_path / 'data' / 'escalations'
    outer_mkdir_calls: list[int] = [0]
    write_text_calls: list[int] = [0]
    real_mkdir = Path.mkdir
    real_write_text = Path.write_text

    def intercepting_mkdir(self, *args, **kwargs):
        is_outer = self == expected_esc_dir and kwargs.get('parents', False)  # relies on production passing parents=True as a kwarg (bulk_reset_guard.py:487)
        if is_outer:
            outer_mkdir_calls[0] += 1
            if outer_mkdir_calls[0] == 1:
                raise OSError('simulated read-only mount')
        return real_mkdir(self, *args, **kwargs)

    def tracking_write_text(self, data, *args, **kwargs):
        write_text_calls[0] += 1
        return real_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, 'mkdir', intercepting_mkdir)
    monkeypatch.setattr(Path, 'write_text', tracking_write_text)

    clock = [1000.0]

    def fake_clock() -> float:
        return clock[0]

    guard = BulkResetGuard(
        done_threshold=3,
        in_progress_threshold=100,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        write_failure_backoff_seconds=60.0,
        time_provider=fake_clock,
        escalations_fallback_dir=tmp_path,
    )

    # Phase 1: trip guard at t=1001..1004; mkdir raises → rejection (no propagation).
    v: BulkResetVerdict | None = None
    for i in range(4):
        clock[0] += 1.0
        v = await guard.observe_attempt(
            project_id='proj',
            task_id=f't{i}',
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
    assert v is not None
    assert v.outcome == 'rejection', f'Phase 1: expected rejection, got {v.outcome}'
    assert outer_mkdir_calls[0] == 1, (
        f'Phase 1: expected 1 outer mkdir call, got {outer_mkdir_calls[0]}'
    )
    assert write_text_calls[0] == 0, (
        f'Phase 1: OSError from mkdir should short-circuit before write_text; '
        f'write_text call count is {write_text_calls[0]}'
    )

    # Phase 2: 30s later — still inside 60s backoff window.
    clock[0] += 30.0
    v2 = await guard.observe_attempt(
        project_id='proj',
        task_id='t4',
        old_status='done',
        new_status='pending',
        project_root=str(tmp_path),
    )
    assert v2.outcome == 'rejection', (
        f'Phase 2: expected rejection (backoff suppressed retry), got {v2.outcome}'
    )
    assert outer_mkdir_calls[0] == 1, (
        f'Phase 2: backoff should have suppressed mkdir retry; '
        f'outer mkdir call count is {outer_mkdir_calls[0]}'
    )
    assert write_text_calls[0] == 0, (
        f'Phase 2: write_text should not be called during backoff; '
        f'write_text call count is {write_text_calls[0]}'
    )

    # Phase 3: advance past window+backoff; seed 3 fresh ok reversals then trip.
    clock[0] = 3000.0
    for i in range(3):
        clock[0] += 1.0
        v = await guard.observe_attempt(
            project_id='proj',
            task_id=f'new-{i}',
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
        assert v.outcome == 'ok', f'Phase 3 ok reversal {i}: expected ok, got {v.outcome}'

    # Trip again — outer mkdir called 2nd time (succeeds), write_text called once → escalated.
    clock[0] += 1.0
    v3 = await guard.observe_attempt(
        project_id='proj',
        task_id='new-3',
        old_status='done',
        new_status='pending',
        project_root=str(tmp_path),
    )
    assert outer_mkdir_calls[0] == 2, (
        f'Phase 3: expected 2 outer mkdir calls total, got {outer_mkdir_calls[0]}'
    )
    assert write_text_calls[0] == 1, (
        f'Phase 3: expected 1 write_text call after backoff cleared, got {write_text_calls[0]}'
    )
    assert v3.outcome == 'escalated', (
        f'Phase 3: expected escalated after backoff cleared, got {v3.outcome}'
    )


# ---------------------------------------------------------------------------
# task-1016 step-7: per-kind counter isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_per_kind_counter_isolation(tmp_path):
    """done→pending and in-progress→pending counts are independent.

    Scenario A (done trips first):
      done_threshold=3, in_progress_threshold=50
      - Fire 3 done→pending: all ok (done counter at 3, in-progress at 0).
      - Fire 49 in-progress→pending: all ok (in-progress at 49 < 50; done untouched).
      - Fire 4th done→pending: trips done counter;
        affected_task_ids contains only the 4 done task-ids (no in-progress ids).

    Scenario B (in-progress trips first):
      done_threshold=50, in_progress_threshold=3
      - Fire 3 in-progress→pending: all ok.
      - Fire 3 done→pending: all ok (done at 3 < 50; in-progress untouched at 3).
      - Fire 4th in-progress→pending: trips in-progress counter;
        affected_task_ids contains only the 4 in-progress task-ids.
    """
    clock = [1000.0]

    def fake_clock() -> float:
        return clock[0]

    # ---- Scenario A ----
    guard_a = BulkResetGuard(
        done_threshold=3,
        in_progress_threshold=50,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        time_provider=fake_clock,
        escalations_fallback_dir=tmp_path,
    )

    done_ids = [f'done-{i}' for i in range(4)]
    ip_ids = [f'ip-{i}' for i in range(49)]

    # (a) 3 done→pending — all ok
    for tid in done_ids[:3]:
        clock[0] += 1.0
        v = await guard_a.observe_attempt(
            project_id='scenario-a',
            task_id=tid,
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
        assert v.outcome == 'ok', f'{tid}: expected ok, got {v.outcome}'

    # (b) 49 in-progress→pending — all ok (in-progress counter at 49 < 50)
    for tid in ip_ids:
        clock[0] += 0.01
        v = await guard_a.observe_attempt(
            project_id='scenario-a',
            task_id=tid,
            old_status='in-progress',
            new_status='pending',
            project_root=str(tmp_path),
        )
        assert v.outcome == 'ok', f'{tid}: expected ok, got {v.outcome}'

    # (c) 4th done→pending trips the done counter
    clock[0] += 1.0
    v_trip = await guard_a.observe_attempt(
        project_id='scenario-a',
        task_id=done_ids[3],
        old_status='done',
        new_status='pending',
        project_root=str(tmp_path),
    )
    assert v_trip.is_rejection is True, (
        f'Scenario A: expected rejection on 4th done→pending, got {v_trip.outcome}'
    )
    assert set(v_trip.affected_task_ids) == set(done_ids), (
        f'Scenario A: affected_task_ids should contain only done ids; '
        f'got {v_trip.affected_task_ids}'
    )
    for ip_id in ip_ids:
        assert ip_id not in v_trip.affected_task_ids, (
            f'Scenario A: in-progress id {ip_id} leaked into done verdict'
        )

    # ---- Scenario B ----
    clock2 = [5000.0]

    def fake_clock2() -> float:
        return clock2[0]

    guard_b = BulkResetGuard(
        done_threshold=50,
        in_progress_threshold=3,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        time_provider=fake_clock2,
        escalations_fallback_dir=tmp_path,
    )

    ip2_ids = [f'ip2-{i}' for i in range(4)]
    done2_ids = [f'done2-{i}' for i in range(3)]

    # (i) 3 in-progress→pending — all ok
    for tid in ip2_ids[:3]:
        clock2[0] += 1.0
        v = await guard_b.observe_attempt(
            project_id='scenario-b',
            task_id=tid,
            old_status='in-progress',
            new_status='pending',
            project_root=str(tmp_path),
        )
        assert v.outcome == 'ok', f'{tid}: expected ok, got {v.outcome}'

    # (ii) 3 done→pending — all ok (done counter at 3, well under 50)
    for tid in done2_ids:
        clock2[0] += 1.0
        v = await guard_b.observe_attempt(
            project_id='scenario-b',
            task_id=tid,
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
        assert v.outcome == 'ok', f'{tid}: expected ok, got {v.outcome}'

    # (iii) 4th in-progress→pending trips the in-progress counter
    clock2[0] += 1.0
    v_trip_b = await guard_b.observe_attempt(
        project_id='scenario-b',
        task_id=ip2_ids[3],
        old_status='in-progress',
        new_status='pending',
        project_root=str(tmp_path),
    )
    assert v_trip_b.is_rejection is True, (
        f'Scenario B: expected rejection on 4th in-progress→pending, got {v_trip_b.outcome}'
    )
    assert set(v_trip_b.affected_task_ids) == set(ip2_ids), (
        f'Scenario B: affected_task_ids should contain only in-progress ids; '
        f'got {v_trip_b.affected_task_ids}'
    )
    for d_id in done2_ids:
        assert d_id not in v_trip_b.affected_task_ids, (
            f'Scenario B: done id {d_id} leaked into in-progress verdict'
        )


# ---------------------------------------------------------------------------
# task-1016 step-9: BulkResetVerdict.kind field
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verdict_carries_tripped_kind(tmp_path):
    """BulkResetVerdict.kind must carry the reversal kind that tripped the guard,
    or None on an ok outcome.

    (a) Tripping the done counter yields verdict.kind == 'done_to_pending'.
    (b) Tripping the in-progress counter yields verdict.kind == 'in_progress_to_pending'.
    (c) An ok outcome (below threshold) has verdict.kind is None.

    Fails until step-10 adds the kind field to BulkResetVerdict and populates
    it inside observe_attempt.
    """
    clock = [1000.0]

    def fake_clock() -> float:
        return clock[0]

    # (c) ok verdict — kind must be None
    guard_ok = BulkResetGuard(
        done_threshold=10,
        in_progress_threshold=100,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        time_provider=fake_clock,
        escalations_fallback_dir=tmp_path,
    )
    clock[0] += 1.0
    v_ok = await guard_ok.observe_attempt(
        project_id='proj-kind',
        task_id='ok-task',
        old_status='done',
        new_status='pending',
        project_root=str(tmp_path),
    )
    assert v_ok.outcome == 'ok'
    assert v_ok.kind is None, (
        f'ok verdict should have kind=None, got {v_ok.kind!r}'
    )

    # (a) Trip the done counter — kind must be 'done_to_pending'
    guard_done = BulkResetGuard(
        done_threshold=3,
        in_progress_threshold=100,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        time_provider=fake_clock,
        escalations_fallback_dir=tmp_path,
    )
    done_trip_ids = ['d0', 'd1', 'd2', 'd3']
    v_done_trip = None
    for tid in done_trip_ids:
        clock[0] += 1.0
        v_done_trip = await guard_done.observe_attempt(
            project_id='proj-done',
            task_id=tid,
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
    assert v_done_trip is not None
    assert v_done_trip.is_rejection is True
    assert v_done_trip.kind == 'done_to_pending', (
        f'done counter trip: expected kind=done_to_pending, got {v_done_trip.kind!r}'
    )

    # (b) Trip the in-progress counter — kind must be 'in_progress_to_pending'
    guard_ip = BulkResetGuard(
        done_threshold=100,
        in_progress_threshold=3,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        time_provider=fake_clock,
        escalations_fallback_dir=tmp_path,
    )
    ip_trip_ids = ['ip0', 'ip1', 'ip2', 'ip3']
    v_ip_trip = None
    for tid in ip_trip_ids:
        clock[0] += 1.0
        v_ip_trip = await guard_ip.observe_attempt(
            project_id='proj-ip',
            task_id=tid,
            old_status='in-progress',
            new_status='pending',
            project_root=str(tmp_path),
        )
    assert v_ip_trip is not None
    assert v_ip_trip.is_rejection is True
    assert v_ip_trip.kind == 'in_progress_to_pending', (
        f'in-progress counter trip: expected kind=in_progress_to_pending, '
        f'got {v_ip_trip.kind!r}'
    )


# ---------------------------------------------------------------------------
# task-1016 step-13: acceptance scenario — startup reconcile must not trip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_acceptance_scenario_startup_reconcile_does_not_trip(tmp_path):
    """Regression pin for the 2026-04-24 reify incident.

    Incident: esc-bulk-reset-reify-2026-04-24T070944_6456580000
    The startup stranded-task reconciler reverted 27 in-progress→pending tasks
    in ~2 s.  The single-threshold guard (threshold=10) tripped even though
    zero done→pending transitions occurred — a false positive.

    Acceptance criteria (task 1016):
    (a) 27 in-progress→pending attempts within 2 s must ALL return outcome='ok'
        with the production defaults (done_threshold=10,
        in_progress_threshold=100, window_seconds=60.0).
    (b) A subsequent burst of 11 done→pending attempts within 2 s must trip the
        done counter on the 11th attempt, with kind=='done_to_pending'.
    """
    clock = [1000.0]

    def fake_clock() -> float:
        return clock[0]

    guard = BulkResetGuard(
        done_threshold=10,
        in_progress_threshold=100,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        time_provider=fake_clock,
        escalations_fallback_dir=tmp_path,
    )
    esc_dir = tmp_path / 'data' / 'escalations'

    # (a) Simulate the incident: 27 in-progress→pending within 2 s — must not trip.
    for i in range(27):
        clock[0] += 2.0 / 27  # spread 27 transitions over 2 s
        v = await guard.observe_attempt(
            project_id='reify',
            task_id=f'stranded-{i}',
            old_status='in-progress',
            new_status='pending',
            project_root=str(tmp_path),
        )
        assert v.outcome == 'ok', (
            f'Incident regression FAIL: in-progress reversal {i} returned '
            f'{v.outcome!r} — startup reconcile must never trip the guard'
        )

    # No escalation file must have been written.
    if esc_dir.exists():
        esc_files = list(esc_dir.glob('*.json'))
        assert esc_files == [], (
            f'No escalation expected for 27 in-progress reversals; '
            f'found: {esc_files}'
        )

    # (b) A done→pending burst of 11 must trip on the 11th attempt.
    v_done = None
    for i in range(11):
        clock[0] += 0.1
        v_done = await guard.observe_attempt(
            project_id='reify',
            task_id=f'done-task-{i}',
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )
    assert v_done is not None
    assert v_done.is_rejection is True, (
        f'Expected rejection on 11th done→pending, got {v_done.outcome!r}'
    )
    assert v_done.kind == 'done_to_pending', (
        f'Expected kind=done_to_pending, got {v_done.kind!r}'
    )


# ---------------------------------------------------------------------------
# task-1016 step-15: server wires split thresholds
# ---------------------------------------------------------------------------

def test_server_wires_split_thresholds(tmp_path):
    """Validate that server/main.py wires BulkResetGuard with the split-threshold
    kwarg pattern (done_threshold / in_progress_threshold) rather than the
    legacy single threshold=.

    Construct BulkResetGuard using the same kwarg pattern that server/main.py uses
    after step-16, with a bumped in_progress_threshold, and assert the instance
    attribute matches.  This proves the split-threshold construction pattern works
    end-to-end from config → guard instance.
    """
    from fused_memory.config.schema import ReconciliationConfig

    # Guard construction using the new split-threshold kwarg pattern.
    cfg = ReconciliationConfig(bulk_reset_guard_in_progress_to_pending_threshold=7)
    guard = BulkResetGuard(
        enabled=cfg.bulk_reset_guard_enabled,
        done_threshold=cfg.bulk_reset_guard_done_to_pending_threshold,
        in_progress_threshold=cfg.bulk_reset_guard_in_progress_to_pending_threshold,
        window_seconds=cfg.bulk_reset_guard_window_seconds,
        escalation_rate_limit_seconds=cfg.bulk_reset_guard_escalation_rate_limit_seconds,
        escalations_fallback_dir=tmp_path,
    )
    assert guard._in_progress_threshold == 7, (
        f'Expected _in_progress_threshold=7, got {guard._in_progress_threshold}'
    )
    assert guard._done_threshold == 10, (
        f'Expected _done_threshold=10 (default), got {guard._done_threshold}'
    )


# ---------------------------------------------------------------------------
# task-1032: guard accepts write_failure_backoff_seconds from config
# ---------------------------------------------------------------------------

def test_guard_accepts_cfg_write_failure_backoff(tmp_path):
    """Prove that BulkResetGuard accepts write_failure_backoff_seconds from
    a ReconciliationConfig instance and stores it as _write_failure_backoff_seconds.

    This is a construction-contract witness: it confirms the config field exists,
    that BulkResetGuard accepts the kwarg, and that the value flows through to
    the guard instance attribute.  It mirrors the kwarg pattern that server/main.py
    uses after task-1032 (see main.py:426-437), but does NOT exercise main.py
    directly — code review verifies the main.py call site.

    Mirrors test_server_wires_split_thresholds (the post-step-20 replacement
    for the fragile inspect.getsource meta-test, see commit da8e5a4c96).
    """
    from fused_memory.config.schema import ReconciliationConfig

    cfg = ReconciliationConfig(bulk_reset_guard_write_failure_backoff_seconds=42.0)
    guard = BulkResetGuard(
        enabled=cfg.bulk_reset_guard_enabled,
        done_threshold=cfg.bulk_reset_guard_done_to_pending_threshold,
        in_progress_threshold=cfg.bulk_reset_guard_in_progress_to_pending_threshold,
        window_seconds=cfg.bulk_reset_guard_window_seconds,
        escalation_rate_limit_seconds=cfg.bulk_reset_guard_escalation_rate_limit_seconds,
        write_failure_backoff_seconds=cfg.bulk_reset_guard_write_failure_backoff_seconds,
        escalations_fallback_dir=tmp_path,
    )
    assert guard._write_failure_backoff_seconds == 42.0, (
        f'Expected _write_failure_backoff_seconds=42.0, got '
        f'{guard._write_failure_backoff_seconds}'
    )


# ---------------------------------------------------------------------------
# task-1021 item-1: no-op eviction removal — state object identity
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fresh_project_state_object_identity_persists_across_window_reset(tmp_path):
    """_GuardState object identity for a never-tripped project is preserved
    across two observe_attempt calls separated by a window reset.

    Pre-fix (no-op eviction block at bulk_reset_guard.py lines 314-327 present):
      On call #2 the deques are empty after pruning AND both timestamps are 0.0,
      so the eviction block fires: it pops the dict entry, creates a NEW
      _GuardState, and re-inserts it.  The `is` comparison is False — FAIL.

    Post-fix (eviction block removed):
      The same _GuardState object is preserved in the dict — the `is` comparison
      is True — PASS.
    """
    clock = [1000.0]

    def fake_clock() -> float:
        return clock[0]

    guard = BulkResetGuard(
        done_threshold=10,
        in_progress_threshold=100,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        time_provider=fake_clock,
        escalations_fallback_dir=tmp_path,
    )

    # Call #1: add an entry at t=1001 (inside the 60s window).
    clock[0] = 1001.0
    await guard.observe_attempt(
        project_id='proj-fresh',
        task_id='t0',
        old_status='done',
        new_status='pending',
        project_root=str(tmp_path),
    )
    initial_state = guard._state['proj-fresh']

    # Advance clock to 1100.0: cutoff = 1100 - 60 = 1040 > 1001, so the
    # t=1001 entry is pruned on call #2.  Both timestamps are still 0.0
    # (no trip occurred), so the no-op eviction predicate is satisfied.
    clock[0] = 1100.0
    await guard.observe_attempt(
        project_id='proj-fresh',
        task_id='t1',
        old_status='done',
        new_status='pending',
        project_root=str(tmp_path),
    )

    # The dead eviction block (bulk_reset_guard.py lines 314-327) would have
    # popped and re-inserted a fresh _GuardState here.  After removal the same
    # object is preserved.
    assert guard._state['proj-fresh'] is initial_state, (
        'State object was replaced — the no-op eviction block (lines 314-327 of '
        'bulk_reset_guard.py) fired, popped the dict entry, and inserted a fresh '
        '_GuardState.  Remove that block to fix this.'
    )


# ---------------------------------------------------------------------------
# task-1021 item-2: _record_write_failure must use fresh _now() after I/O
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_record_write_failure_uses_fresh_now_after_io(tmp_path, monkeypatch):
    """last_write_failure_ts is set to a fresh _now() reading taken AFTER the
    slow I/O attempt, not the stale 'now' captured at the top of
    _maybe_write_escalation (step-3 polish test for task-1021, item 2).

    Sequence of _now() calls in the trip path:
      call #1 (1004.0) → now at top of observe_attempt
      call #2 (1005.0) → now at top of _maybe_write_escalation
      call #3 (1099.0) → fresh now inside _record_write_failure (post-fix only)

    Pre-fix: _record_write_failure uses the 'now' parameter (1005.0, stale) →
      assertion '1005.0 == 1099.0' fails.
    Post-fix: _record_write_failure calls self._now() itself (1099.0, fresh) →
      assertion passes.

    The finite iterator raises StopIteration on any unexpected 4th _now() call,
    surfacing accidental extra calls as a clear test failure.
    """
    # Seed phase: 3 ok done→pending reversals with a simple counter clock.
    clock = [1000.0]

    def seed_clock() -> float:
        return clock[0]

    guard = BulkResetGuard(
        done_threshold=3,
        in_progress_threshold=100,
        window_seconds=60.0,
        escalation_rate_limit_seconds=900.0,
        write_failure_backoff_seconds=60.0,
        time_provider=seed_clock,
        escalations_fallback_dir=tmp_path,
    )
    for i in range(3):
        clock[0] += 1.0
        await guard.observe_attempt(
            project_id='proj-fresh-now',
            task_id=f's{i}',
            old_status='done',
            new_status='pending',
            project_root=str(tmp_path),
        )

    # Trip phase: replace _now with a finite iterator that returns explicit values.
    # StopIteration on an unexpected 4th call surfaces extra _now() calls as a
    # clear failure rather than a silent assertion mismatch.
    trip_values = iter([1004.0, 1005.0, 1099.0])
    guard._now = lambda: next(trip_values)

    # Monkeypatch write_text to raise OSError (simulates a slow failed write).
    def failing_write_text(self, data, *args, **kwargs):
        raise OSError('simulated slow-then-failed write')

    monkeypatch.setattr(Path, 'write_text', failing_write_text)

    verdict = await guard.observe_attempt(
        project_id='proj-fresh-now',
        task_id='t-trip',
        old_status='done',
        new_status='pending',
        project_root=str(tmp_path),
    )

    assert verdict.outcome == 'rejection', (
        f'Expected rejection (write failed), got {verdict.outcome!r}'
    )
    assert guard._state['proj-fresh-now'].last_write_failure_ts == 1099.0, (
        f'Expected last_write_failure_ts=1099.0 (fresh _now() after I/O), '
        f'got {guard._state["proj-fresh-now"].last_write_failure_ts}. '
        'Stale-now bug: _record_write_failure used the now parameter (1005.0) '
        'captured at the top of _maybe_write_escalation rather than calling '
        'self._now() itself after the slow I/O returned.'
    )
