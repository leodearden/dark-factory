"""Tests for orchestrator.digest — pure digest helpers and EWA math.

Task 1327 — AFK hardening: Per-N-escalation digest + EWA escalation/done trip.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Callable

import pytest
import pytest_asyncio
from shared.cost_store import CostStore

import orchestrator.digest as digest
from orchestrator.event_store import EventStore, EventType


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


def _seed_events_db_direct(db_path: Path) -> None:
    """Seed an events DB (via EventStore schema + direct SQLite) with task_completed rows."""
    # Create schema via EventStore
    store = EventStore(db_path, 'run-test')
    del store  # schema is created in __init__

    rows = [
        # (timestamp, task_id, outcome)
        ('2026-05-10T08:00:00+00:00', 't1', 'done'),       # in-window done
        ('2026-05-10T10:00:00+00:00', 't2', 'done'),       # in-window done
        ('2026-05-10T11:00:00+00:00', 't3', 'requeued'),   # in-window non-done
        ('2026-05-10T12:00:00+00:00', 't4', 'blocked'),    # in-window non-done
        ('2026-05-09T23:59:59+00:00', 't5', 'done'),       # before window
        ('2026-05-11T00:00:01+00:00', 't6', 'done'),       # after window
    ]
    conn = sqlite3.connect(str(db_path))
    try:
        for ts, tid, outcome in rows:
            conn.execute(
                'INSERT INTO events (timestamp, run_id, task_id, event_type, data) '
                'VALUES (?, ?, ?, ?, ?)',
                (ts, 'run-test', tid, 'task_completed', json.dumps({'outcome': outcome})),
            )
        conn.commit()
    finally:
        conn.close()


class TestCountDoneInWindow:
    """Tests for digest.count_done_in_window(events_db_path, start_iso, end_iso)."""

    def test_counts_done_in_window(self, tmp_path: Path) -> None:
        """Returns count of task_completed rows with outcome='done' inside window."""
        db_path = tmp_path / 'events.db'
        _seed_events_db_direct(db_path)

        count = digest.count_done_in_window(
            db_path,
            '2026-05-10T00:00:00+00:00',
            '2026-05-10T23:59:59+00:00',
        )
        assert count == 2, f"Expected 2 done in window; got {count}"

    def test_missing_db_returns_zero(self, tmp_path: Path) -> None:
        """Non-existent DB returns 0 (fail-open)."""
        count = digest.count_done_in_window(
            tmp_path / 'no-such.db',
            '2026-05-10T00:00:00+00:00',
            '2026-05-10T23:59:59+00:00',
        )
        assert count == 0, f"Expected 0 for missing DB; got {count}"

    def test_empty_window_returns_zero(self, tmp_path: Path) -> None:
        """Window with no matching rows returns 0."""
        db_path = tmp_path / 'events.db'
        _seed_events_db_direct(db_path)

        count = digest.count_done_in_window(
            db_path,
            '2026-05-01T00:00:00+00:00',
            '2026-05-01T23:59:59+00:00',
        )
        assert count == 0, f"Expected 0 for empty window; got {count}"


# ---------------------------------------------------------------------------
# Fixtures for async CostStore tests
# ---------------------------------------------------------------------------

from datetime import UTC, datetime as _datetime


def _iso_offset(delta: timedelta) -> str:
    """Return an ISO timestamp offset from now by delta."""
    return (_datetime.now(UTC) + delta).isoformat()


@pytest_asyncio.fixture
async def _cost_store(tmp_path: Path):
    """Fixture: open a CostStore, yield it, close it."""
    store = CostStore(tmp_path / 'cost.db')
    await store.open()
    yield store
    await store.close()


class TestCostInWindow:
    """Tests for digest.cost_in_window(cost_store, window_start_iso, window_end_iso)."""

    @pytest.mark.asyncio
    async def test_window_cost_sums_correctly(
        self, tmp_path: Path, _cost_store: CostStore
    ) -> None:
        """watcher_cost_in_window and total_cost_in_window are correct."""
        store = _cost_store
        # In-window watcher
        await store.save_invocation(
            run_id='r1', task_id='t1', project_id='p', account_name='a',
            model='opus', role='escalation-watcher-auto',
            cost_usd=5.0, input_tokens=None, output_tokens=None,
            cache_read_tokens=None, cache_create_tokens=None, duration_ms=100, capped=False,
            started_at='2026-05-10T08:00:00+00:00',
            completed_at='2026-05-10T08:00:00+00:00',
        )
        # In-window non-watcher
        await store.save_invocation(
            run_id='r2', task_id='t2', project_id='p', account_name='a',
            model='sonnet', role='orchestrator',
            cost_usd=2.0, input_tokens=None, output_tokens=None,
            cache_read_tokens=None, cache_create_tokens=None, duration_ms=100, capped=False,
            started_at='2026-05-10T09:00:00+00:00',
            completed_at='2026-05-10T09:00:00+00:00',
        )
        # Out-of-window
        await store.save_invocation(
            run_id='r3', task_id='t3', project_id='p', account_name='a',
            model='sonnet', role='escalation-watcher-auto',
            cost_usd=99.0, input_tokens=None, output_tokens=None,
            cache_read_tokens=None, cache_create_tokens=None, duration_ms=100, capped=False,
            started_at='2026-05-09T00:00:00+00:00',
            completed_at='2026-05-09T00:00:00+00:00',
        )

        stats = await digest.cost_in_window(
            store,
            '2026-05-10T00:00:00+00:00',
            '2026-05-10T23:59:59+00:00',
        )
        assert stats.watcher_cost_in_window == pytest.approx(5.0)
        assert stats.total_cost_in_window == pytest.approx(7.0)

    @pytest.mark.asyncio
    async def test_none_store_returns_zeros(self) -> None:
        """cost_store=None returns all zeros (fail-open)."""
        stats = await digest.cost_in_window(
            None,
            '2026-05-10T00:00:00+00:00',
            '2026-05-10T23:59:59+00:00',
        )
        assert stats.watcher_cost_in_window == pytest.approx(0.0)
        assert stats.total_cost_in_window == pytest.approx(0.0)
        assert stats.watcher_cost_24h == pytest.approx(0.0)
        assert stats.total_cost_24h == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_trailing_24h_cost_present(
        self, tmp_path: Path, _cost_store: CostStore
    ) -> None:
        """watcher_cost_24h and total_cost_24h include trailing-24h rows."""
        store = _cost_store
        # Within 24h
        await store.save_invocation(
            run_id='r1', task_id='t1', project_id='p', account_name='a',
            model='opus', role='escalation-watcher-auto',
            cost_usd=3.0, input_tokens=None, output_tokens=None,
            cache_read_tokens=None, cache_create_tokens=None, duration_ms=100, capped=False,
            started_at=_iso_offset(timedelta(hours=-1)),
            completed_at=_iso_offset(timedelta(hours=-1)),
        )
        await store.save_invocation(
            run_id='r2', task_id='t2', project_id='p', account_name='a',
            model='sonnet', role='orchestrator',
            cost_usd=1.5, input_tokens=None, output_tokens=None,
            cache_read_tokens=None, cache_create_tokens=None, duration_ms=100, capped=False,
            started_at=_iso_offset(timedelta(hours=-2)),
            completed_at=_iso_offset(timedelta(hours=-2)),
        )
        # Older than 24h — not in 24h window
        await store.save_invocation(
            run_id='r3', task_id='t3', project_id='p', account_name='a',
            model='sonnet', role='escalation-watcher-auto',
            cost_usd=50.0, input_tokens=None, output_tokens=None,
            cache_read_tokens=None, cache_create_tokens=None, duration_ms=100, capped=False,
            started_at=_iso_offset(timedelta(hours=-25)),
            completed_at=_iso_offset(timedelta(hours=-25)),
        )

        stats = await digest.cost_in_window(
            store,
            '2026-05-10T00:00:00+00:00',  # window_start does not match recent rows
            '2026-05-10T23:59:59+00:00',
        )
        # 24h figures are anchored to now, not the digest window
        assert stats.watcher_cost_24h == pytest.approx(3.0)
        assert stats.total_cost_24h == pytest.approx(4.5)
