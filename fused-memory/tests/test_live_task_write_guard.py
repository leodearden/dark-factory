"""Tests for the code-enforced before/after live-task-write self-check (task 2624).

Incident: dark_factory task 2588 un-claim (mem0 5bde6829) — the "read
status/claimant_run_id/heartbeat_at via get_task immediately BEFORE and
AFTER a write to a live in-progress task, and self-file a finding on
unexpected divergence" rule was only an LLM prompt convention, skippable by
an inattentive/budget-constrained Stage-2 recon run. This module makes it
code-enforced.

Layout mirrors ``test_recon_write_policy.py``: pure-unit tests for the
detector core first, then the async ``guarded_recon_task_write`` wrapper
tests, then interceptor-boundary tests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.middleware import live_task_write_guard as guard_mod
from fused_memory.middleware.live_task_write_guard import (
    LIFECYCLE_RESET_FLAG_TYPE,
    LifecycleResetFinding,
    LiveTaskSnapshot,
    detect_lifecycle_reset,
    extract_live_snapshot,
    has_live_claimant,
)

LIVE_CLAIMANT = 'run-1/session-1/pid=123'


def _flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT, heartbeat_at='2026-07-15T12:00:00+00:00', **extra):
    payload = {
        'id': '1',
        'status': status,
        'claimant_run_id': claimant_run_id,
        'heartbeat_at': heartbeat_at,
    }
    payload.update(extra)
    return payload


def _nested(status='in-progress', claimant_run_id=LIVE_CLAIMANT, heartbeat_at='2026-07-15T12:00:00+00:00', **extra):
    return {'data': _flat(status, claimant_run_id, heartbeat_at, **extra)}


# ---------------------------------------------------------------------------
# (a) extract_live_snapshot — flat / nested shape handling
# ---------------------------------------------------------------------------


class TestExtractLiveSnapshot:
    def test_flat_shape(self):
        snapshot = extract_live_snapshot(_flat())
        assert snapshot == LiveTaskSnapshot(
            status='in-progress',
            claimant_run_id=LIVE_CLAIMANT,
            heartbeat_at='2026-07-15T12:00:00+00:00',
        )

    def test_nested_data_shape(self):
        snapshot = extract_live_snapshot(_nested())
        assert snapshot == LiveTaskSnapshot(
            status='in-progress',
            claimant_run_id=LIVE_CLAIMANT,
            heartbeat_at='2026-07-15T12:00:00+00:00',
        )

    def test_flat_shape_wins_when_status_key_present(self):
        """Mirrors _extract_status's discriminator: presence of a top-level
        'status' key means flat, even if a 'data' key also happens to be
        present alongside it."""
        task_data = _flat()
        task_data['data'] = {'status': 'done', 'claimant_run_id': None, 'heartbeat_at': None}
        snapshot = extract_live_snapshot(task_data)
        assert snapshot.status == 'in-progress'
        assert snapshot.claimant_run_id == LIVE_CLAIMANT

    def test_missing_fields_default_safely(self):
        snapshot = extract_live_snapshot({'id': '1'})
        assert snapshot.status == 'unknown'
        assert snapshot.claimant_run_id is None
        assert snapshot.heartbeat_at is None

    def test_non_mapping_input_defaults_safely(self):
        snapshot = extract_live_snapshot(None)
        assert snapshot.status == 'unknown'
        assert snapshot.claimant_run_id is None
        assert snapshot.heartbeat_at is None


# ---------------------------------------------------------------------------
# (b) has_live_claimant
# ---------------------------------------------------------------------------


class TestHasLiveClaimant:
    def test_true_for_in_progress_with_claimant(self):
        assert has_live_claimant(_flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT)) is True

    def test_true_for_in_progress_with_claimant_nested(self):
        assert has_live_claimant(_nested(status='in-progress', claimant_run_id=LIVE_CLAIMANT)) is True

    def test_false_when_not_in_progress(self):
        assert has_live_claimant(_flat(status='pending', claimant_run_id=LIVE_CLAIMANT)) is False
        assert has_live_claimant(_flat(status='done', claimant_run_id=LIVE_CLAIMANT)) is False

    def test_false_when_claimant_none(self):
        assert has_live_claimant(_flat(status='in-progress', claimant_run_id=None)) is False

    def test_false_when_claimant_blank(self):
        assert has_live_claimant(_flat(status='in-progress', claimant_run_id='   ')) is False

    def test_false_when_claimant_empty_string(self):
        assert has_live_claimant(_flat(status='in-progress', claimant_run_id='')) is False


# ---------------------------------------------------------------------------
# (c)-(f) detect_lifecycle_reset
# ---------------------------------------------------------------------------


def _detect(before, after, *, op='update_task', task_id='1', project_id='dark_factory',
            requested_status=None, requested_claimant_write=False, requested_heartbeat_write=False):
    return detect_lifecycle_reset(
        before,
        after,
        op=op,
        task_id=task_id,
        project_id=project_id,
        requested_status=requested_status,
        requested_claimant_write=requested_claimant_write,
        requested_heartbeat_write=requested_heartbeat_write,
    )


class TestDetectLifecycleReset:
    def test_unrequested_claimant_reset_fires(self):
        """(c) claimant_run_id resets to None un-requested during update_task."""
        before = _flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT)
        after = _flat(status='in-progress', claimant_run_id=None)

        finding = _detect(
            before, after, op='update_task', task_id='42', project_id='dark_factory',
            requested_status=None, requested_claimant_write=False,
        )

        assert finding is not None
        assert isinstance(finding, LifecycleResetFinding)
        assert finding.flag_type == LIFECYCLE_RESET_FLAG_TYPE
        assert finding.flag_type == 'task_lifecycle_reset_detected'
        assert finding.task_id == '42'
        assert finding.project_id == 'dark_factory'
        assert finding.op == 'update_task'
        assert 'claimant_run_id' in finding.diverged_fields

    def test_claimant_unchanged_is_benign(self):
        before = _flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT)
        after = _flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT)

        assert _detect(before, after) is None

    def test_requested_claimant_write_clearing_is_benign(self):
        """The write itself explicitly cleared the claimant — not unexpected."""
        before = _flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT)
        after = _flat(status='in-progress', claimant_run_id=None)

        assert _detect(before, after, requested_claimant_write=True) is None

    def test_before_not_in_progress_is_dormant(self):
        before = _flat(status='pending', claimant_run_id=None)
        after = _flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT)

        assert _detect(before, after) is None

    def test_before_in_progress_without_live_claimant_is_dormant(self):
        before = _flat(status='in-progress', claimant_run_id=None)
        after = _flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT)

        assert _detect(before, after) is None

    def test_status_changing_to_exactly_requested_status_is_benign(self):
        before = _flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT)
        after = _flat(status='done', claimant_run_id=LIVE_CLAIMANT)

        finding = _detect(
            before, after, op='set_task_status', requested_status='done',
            requested_claimant_write=False,
        )

        assert finding is None

    def test_unrequested_status_revert_fires(self):
        """(e) requested_status='done' but after status=='pending'."""
        before = _flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT)
        after = _flat(status='pending', claimant_run_id=LIVE_CLAIMANT)

        finding = _detect(
            before, after, op='set_task_status', requested_status='done',
            requested_claimant_write=False,
        )

        assert finding is not None
        assert 'status' in finding.diverged_fields

    def test_heartbeat_only_change_does_not_trigger(self):
        """(f) claimant + status unchanged, only heartbeat_at differs (a
        legitimate concurrent heartbeat refresh) — must not false-positive."""
        before = _flat(
            status='in-progress', claimant_run_id=LIVE_CLAIMANT,
            heartbeat_at='2026-07-15T12:00:00+00:00',
        )
        after = _flat(
            status='in-progress', claimant_run_id=LIVE_CLAIMANT,
            heartbeat_at='2026-07-15T12:05:00+00:00',
        )

        assert _detect(before, after) is None

    def test_heartbeat_recorded_as_corroboration_when_finding_fires(self):
        """When a real finding fires, an un-requested heartbeat divergence is
        included in diverged_fields as corroborating evidence."""
        before = _flat(
            status='in-progress', claimant_run_id=LIVE_CLAIMANT,
            heartbeat_at='2026-07-15T12:00:00+00:00',
        )
        after = _flat(
            status='in-progress', claimant_run_id=None,
            heartbeat_at=None,
        )

        finding = _detect(before, after, op='update_task', requested_claimant_write=False)

        assert finding is not None
        assert 'claimant_run_id' in finding.diverged_fields
        assert 'heartbeat_at' in finding.diverged_fields

    def test_requested_heartbeat_write_suppresses_heartbeat_corroboration(self):
        before = _flat(
            status='in-progress', claimant_run_id=LIVE_CLAIMANT,
            heartbeat_at='2026-07-15T12:00:00+00:00',
        )
        after = _flat(
            status='in-progress', claimant_run_id=None,
            heartbeat_at='2026-07-15T12:05:00+00:00',
        )

        finding = _detect(
            before, after, op='update_task',
            requested_claimant_write=False, requested_heartbeat_write=True,
        )

        assert finding is not None
        assert 'heartbeat_at' not in finding.diverged_fields
