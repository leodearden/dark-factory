"""Tests for orchestrator.digest — pure digest helpers and EWA math.

Task 1327 — AFK hardening: Per-N-escalation digest + EWA escalation/done trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import orchestrator.digest as digest


class TestUpdateEwa:
    """Unit tests for digest.update_ewa(prev_ewa, escalations_in_step, done_in_step, alpha)."""

    def test_standard_case(self) -> None:
        """Standard EWA update: prev=0.5, esc=10, done=5, alpha=0.3 → 0.3*(10/5)+0.7*0.5=0.95."""
        result = digest.update_ewa(prev_ewa=0.5, escalations_in_step=10, done_in_step=5, alpha=0.3)
        assert result == pytest.approx(0.95), f"Expected 0.95; got {result}"

    def test_done_zero_uses_denominator_one(self) -> None:
        """done==0 guard: denominator treated as 1; esc=4, done=0, alpha=0.3, prev=0.0 → 1.2.

        Proves zero-done pushes EWA up (not crash, not suppression).
        0.3*(4/1) + 0.7*0.0 = 1.2
        """
        result = digest.update_ewa(prev_ewa=0.0, escalations_in_step=4, done_in_step=0, alpha=0.3)
        assert result == pytest.approx(1.2), f"Expected 1.2; got {result}"

    def test_esc_zero_pulls_toward_zero(self) -> None:
        """esc==0 with done>0 pulls EWA toward zero: esc=0, done=5, prev=1.0, alpha=0.3 → 0.7."""
        result = digest.update_ewa(prev_ewa=1.0, escalations_in_step=0, done_in_step=5, alpha=0.3)
        assert result == pytest.approx(0.7), f"Expected 0.7; got {result}"

    def test_alpha_zero_returns_prev_unchanged(self) -> None:
        """alpha=0.0: EWA is never updated — returns prev unchanged."""
        result = digest.update_ewa(prev_ewa=3.5, escalations_in_step=100, done_in_step=1, alpha=0.0)
        assert result == pytest.approx(3.5), f"Expected 3.5 (prev unchanged); got {result}"

    def test_alpha_one_returns_raw_ratio(self) -> None:
        """alpha=1.0: EWA collapses to the raw ratio (esc/max(done,1))."""
        # esc=8, done=4, alpha=1.0 → 8/4 = 2.0
        result = digest.update_ewa(prev_ewa=99.0, escalations_in_step=8, done_in_step=4, alpha=1.0)
        assert result == pytest.approx(2.0), f"Expected 2.0; got {result}"


# ---------------------------------------------------------------------------
# Helpers for seeding escalation JSON files
# ---------------------------------------------------------------------------

def _write_esc(path: Path, esc_id: str, **kwargs: object) -> None:
    """Write a minimal esc-*.json at path."""
    data: dict = {
        'id': esc_id,
        'task_id': '1',
        'agent_role': 'orchestrator',
        'severity': 'info',
        'category': kwargs.get('category', 'infra_issue'),
        'summary': 's',
        'detail': '',
        'suggested_action': '',
        'timestamp': kwargs.get('timestamp', '2026-05-10T10:00:00+00:00'),
        'status': kwargs.get('status', 'pending'),
        'resolution': None,
        'worktree': None,
        'workflow_state': None,
        'level': kwargs.get('level', 0),
        'resolved_at': kwargs.get('resolved_at', None),
        'resolved_by': None,
        'resolution_turns': None,
        'dedupe_count': 0,
        'dedupe_children': kwargs.get('dedupe_children', []),
    }
    path.write_text(json.dumps(data))


class TestAggregateEscalations:
    """Tests for digest.aggregate_escalations(escalations_dir, window_start_iso, window_end_iso)."""

    def test_basic_aggregation(self, tmp_path: Path) -> None:
        """Counts by category × level × status, first/last ts, dedupe totals, pending total."""
        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()

        # In-window escalations
        _write_esc(esc_dir / 'esc-1-1.json', 'esc-1-1',
                   category='infra_issue', level=0, status='pending',
                   timestamp='2026-05-10T08:00:00+00:00', dedupe_children=['esc-1-x'])
        _write_esc(esc_dir / 'esc-1-2.json', 'esc-1-2',
                   category='infra_issue', level=0, status='resolved',
                   timestamp='2026-05-10T09:00:00+00:00')
        _write_esc(esc_dir / 'esc-1-3.json', 'esc-1-3',
                   category='scope_violation', level=1, status='dismissed',
                   timestamp='2026-05-10T10:00:00+00:00')
        # Out-of-window (before window_start)
        _write_esc(esc_dir / 'esc-1-4.json', 'esc-1-4',
                   category='infra_issue', level=0, status='pending',
                   timestamp='2026-05-09T07:59:59+00:00')

        window_start = '2026-05-10T00:00:00+00:00'
        window_end = '2026-05-10T23:59:59+00:00'
        stats = digest.aggregate_escalations(esc_dir, window_start, window_end)

        # Total counts
        assert stats.pending_total == 1, f"Expected 1 pending in window; got {stats.pending_total}"
        assert stats.dedupe_children_total == 1, (
            f"Expected 1 dedupe child; got {stats.dedupe_children_total}"
        )

        # category × level × status counts
        key_pending = ('infra_issue', 0, 'pending')
        key_resolved = ('infra_issue', 0, 'resolved')
        key_dismissed = ('scope_violation', 1, 'dismissed')
        assert stats.category_level_status_counts.get(key_pending, 0) == 1
        assert stats.category_level_status_counts.get(key_resolved, 0) == 1
        assert stats.category_level_status_counts.get(key_dismissed, 0) == 1

        # first_ts and last_ts are the earliest/latest in-window timestamps
        assert stats.first_ts == '2026-05-10T08:00:00+00:00'
        assert stats.last_ts == '2026-05-10T10:00:00+00:00'

    def test_archive_deduplication(self, tmp_path: Path) -> None:
        """An id in both root and archive is counted only once (root wins)."""
        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()
        archive_sub = esc_dir / 'archive' / '2026-05-10'
        archive_sub.mkdir(parents=True)

        # Root copy (takes precedence)
        _write_esc(esc_dir / 'esc-dup-1.json', 'esc-dup-1',
                   status='resolved', timestamp='2026-05-10T12:00:00+00:00')
        # Archive copy of same id — should be ignored
        _write_esc(archive_sub / 'esc-dup-1.json', 'esc-dup-1',
                   status='pending', timestamp='2026-05-10T12:00:00+00:00')
        # Archive-only entry (not in root)
        _write_esc(archive_sub / 'esc-arc-1.json', 'esc-arc-1',
                   status='pending', timestamp='2026-05-10T13:00:00+00:00')

        stats = digest.aggregate_escalations(
            esc_dir,
            '2026-05-10T00:00:00+00:00',
            '2026-05-10T23:59:59+00:00',
        )

        # 2 unique escalations total in window
        total = sum(stats.category_level_status_counts.values())
        assert total == 2, f"Expected 2 unique in-window escalations; got {total}"
        # esc-dup-1 is 'resolved' (root wins, not archive's 'pending')
        key_resolved = ('infra_issue', 0, 'resolved')
        assert stats.category_level_status_counts.get(key_resolved, 0) == 1
        # pending_total is 1 (only esc-arc-1)
        assert stats.pending_total == 1

    def test_window_filtering_excludes_outside(self, tmp_path: Path) -> None:
        """Escalations outside [window_start, window_end] are excluded."""
        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()
        _write_esc(esc_dir / 'esc-before.json', 'esc-before',
                   timestamp='2026-05-09T23:59:58+00:00')
        _write_esc(esc_dir / 'esc-after.json', 'esc-after',
                   timestamp='2026-05-11T00:00:01+00:00')

        stats = digest.aggregate_escalations(
            esc_dir,
            '2026-05-10T00:00:00+00:00',
            '2026-05-10T23:59:59+00:00',
        )
        total = sum(stats.category_level_status_counts.values())
        assert total == 0, f"Expected 0 in-window; got {total}"
        assert stats.first_ts is None
        assert stats.last_ts is None
