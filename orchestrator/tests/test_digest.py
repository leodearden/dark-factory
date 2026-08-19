"""Tests for orchestrator.digest — pure digest helpers and EWA math.

Task 1327 — AFK hardening: Per-N-escalation digest + EWA escalation/done trip.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import UTC, timedelta
from datetime import datetime as _datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from shared.cost_store import _SCHEMA as _COST_SCHEMA
from shared.cost_store import CostStore

import orchestrator.digest as digest
from orchestrator.event_store import EventStore
from orchestrator.run_store import RunStore


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

    def test_esc_zero_raises_value_error(self) -> None:
        """esc==0 raises ValueError — caller contract violation.

        The digest gate guarantees escalations_in_step >= N >= 1 before
        calling update_ewa, so esc=0 is always a caller error.  Raising keeps
        the unreachable path loud rather than silently returning a stale EWA.
        """
        with pytest.raises(ValueError, match='escalations_in_step must be > 0'):
            digest.update_ewa(prev_ewa=1.0, escalations_in_step=0, done_in_step=5, alpha=0.3)

    def test_esc_zero_done_zero_raises_value_error(self) -> None:
        """(esc=0, done=0) edge case raises ValueError — not a divide-by-zero path.

        max(done, 1) would handle (esc=0, done=0) mathematically but the
        ValueError guard fires first, keeping the caller contract explicit.
        """
        with pytest.raises(ValueError, match='escalations_in_step must be > 0'):
            digest.update_ewa(prev_ewa=0.42, escalations_in_step=0, done_in_step=0, alpha=0.3)

    def test_alpha_zero_returns_prev_unchanged(self) -> None:
        """alpha=0.0: EWA is never updated — returns prev unchanged.

        NOTE: alpha=0.0 is rejected by OrchestratorConfig (Field constraint gt=0.0),
        so this case cannot be reached through normal config paths.  It is kept as a
        mathematical edge-case test to verify the formula itself, not a production
        regression guard.  For reachable near-zero alpha, see test_standard_case
        (alpha=0.3) which exercises the same code path with a valid config value.
        """
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
        'resolved_at': kwargs.get('resolved_at'),
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

    def test_missing_db_returns_zero(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Non-existent DB returns 0 (fail-open) and does NOT create a stub file."""
        db_path = tmp_path / 'no-such.db'
        with caplog.at_level(logging.WARNING, logger='orchestrator.digest'):
            count = digest.count_done_in_window(
                db_path,
                '2026-05-10T00:00:00+00:00',
                '2026-05-10T23:59:59+00:00',
            )
        assert count == 0, f"Expected 0 for missing DB; got {count}"
        assert not db_path.exists(), "count_done_in_window must not create a stub DB file"
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and r.name == 'orchestrator.digest']
        assert not warnings, f"Missing DB must not WARNING; got: {[r.message for r in warnings]}"

    def test_schema_missing_returns_zero(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """DB file exists but has no 'events' table — returns 0 (fail-open) with WARNING."""
        db_path = tmp_path / 'schema-less.db'
        # Create a real but schema-less SQLite file (no EventStore setup).
        conn = sqlite3.connect(str(db_path))
        conn.close()
        assert db_path.exists(), "pre-condition: file must exist"

        with caplog.at_level(logging.WARNING, logger='orchestrator.digest'):
            count = digest.count_done_in_window(
                db_path,
                '2026-05-10T00:00:00+00:00',
                '2026-05-10T23:59:59+00:00',
            )
        assert count == 0, f"Expected 0 for schema-less DB; got {count}"
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and r.name == 'orchestrator.digest']
        assert len(warnings) == 1, f"Schema-missing must WARNING exactly once; got {len(warnings)}: {[r.message for r in warnings]}"

    def test_missing_db_independent_of_sqlite_message_wording(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Missing-DB detection must not rely on SQLite error wording.

        Simulates a future SQLite version where the error message is different
        (e.g. 'could not open database' instead of 'unable to open database file').
        The structural Path.exists() check must short-circuit before connect is
        ever called, so the non-matching wording never reaches the brittle match.
        """
        db_path = tmp_path / 'no-such.db'
        assert not db_path.exists(), "pre-condition: file must not exist"

        # side_effect is a tripwire if the Path.exists() short-circuit regresses;
        # assert_not_called() below is the primary contract.
        mock_connect = MagicMock(side_effect=sqlite3.OperationalError('could not open database'))
        monkeypatch.setattr(sqlite3, 'connect', mock_connect)

        with caplog.at_level(logging.WARNING, logger='orchestrator.digest'):
            count = digest.count_done_in_window(
                db_path,
                '2026-05-10T00:00:00+00:00',
                '2026-05-10T23:59:59+00:00',
            )
        assert count == 0, f"Expected 0 for missing DB; got {count}"
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and r.name == 'orchestrator.digest']
        assert not warnings, f"Missing-DB detection must not depend on SQLite error wording; got: {[r.message for r in warnings]}"
        mock_connect.assert_not_called()  # verifies upfront Path.exists() short-circuit prevents connect from being reached

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

    def test_relative_path_resolves_to_absolute(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Relative path with existing DB resolves correctly — no WARNING, correct count."""
        db_path = tmp_path / 'events.db'
        _seed_events_db_direct(db_path)
        monkeypatch.chdir(tmp_path)

        with caplog.at_level(logging.WARNING, logger='orchestrator.digest'):
            count = digest.count_done_in_window(
                Path('events.db'),
                '2026-05-10T00:00:00+00:00',
                '2026-05-10T23:59:59+00:00',
            )
        assert count == 2, f"Expected 2 done in window via relative path; got {count}"
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and r.name == 'orchestrator.digest']
        assert not warnings, f"Relative path with existing DB must not WARNING; got: {[r.message for r in warnings]}"


# ---------------------------------------------------------------------------
# Merge-disposition counts from EventStore (task 2384 γ, boundary row 7)
# ---------------------------------------------------------------------------


def _seed_merge_attempt_events_db_direct(
    db_path: Path, rows: list[tuple[str, str, dict]],
) -> None:
    """Seed an events DB (via EventStore schema + direct SQLite) with
    merge_attempt rows, modeled on ``_seed_events_db_direct`` above.

    *rows* is a list of ``(timestamp, task_id, data)`` — *data* is written
    verbatim as the event's JSON payload, so callers control the
    ``'disposition'`` key (present/absent/value) directly.
    """
    store = EventStore(db_path, 'run-test')
    del store  # schema is created in __init__

    conn = sqlite3.connect(str(db_path))
    try:
        for ts, task_id, data in rows:
            conn.execute(
                'INSERT INTO events (timestamp, run_id, task_id, event_type, data) '
                'VALUES (?, ?, ?, ?, ?)',
                (ts, 'run-test', task_id, 'merge_attempt', json.dumps(data)),
            )
        conn.commit()
    finally:
        conn.close()


class TestMergeDispositionCounts:
    """Tests for digest.merge_disposition_counts(events_db_path, start_iso, end_iso)."""

    def test_counts_by_disposition_excludes_out_of_window_and_null(
        self, tmp_path: Path,
    ) -> None:
        """GROUPs BY disposition within the window; excludes out-of-window and
        null-disposition rows — proving integration_skew is separable from
        branch_bug/indeterminate/flakes (the γ stats-separation goal)."""
        db_path = tmp_path / 'events.db'
        _seed_merge_attempt_events_db_direct(db_path, [
            # In-window, disposition set
            ('2026-05-10T08:00:00+00:00', 't1',
             {'outcome': 'verify_failed', 'disposition': 'integration_skew'}),
            ('2026-05-10T09:00:00+00:00', 't2',
             {'outcome': 'verify_failed', 'disposition': 'integration_skew'}),
            ('2026-05-10T10:00:00+00:00', 't3',
             {'outcome': 'verify_failed', 'disposition': 'branch_bug'}),
            ('2026-05-10T11:00:00+00:00', 't4',
             {'outcome': 'verify_failed', 'disposition': 'indeterminate'}),
            # In-window, but no disposition key at all (pre-β / non-orchestrator
            # submit paths) — must be excluded, not counted as a phantom bucket.
            ('2026-05-10T12:00:00+00:00', 't5', {'outcome': 'verify_failed'}),
            # Out-of-window (before window_start), disposition set — must be excluded.
            ('2026-05-09T23:59:59+00:00', 't6',
             {'outcome': 'verify_failed', 'disposition': 'integration_skew'}),
        ])

        counts = digest.merge_disposition_counts(
            db_path,
            '2026-05-10T00:00:00+00:00',
            '2026-05-10T23:59:59+00:00',
        )
        assert counts == {'integration_skew': 2, 'branch_bug': 1, 'indeterminate': 1}, (
            f"counts={counts}"
        )

    def test_missing_db_returns_empty_dict(self, tmp_path: Path) -> None:
        """Non-existent DB returns {} (fail-open) and does NOT create a stub file."""
        db_path = tmp_path / 'no-such.db'
        counts = digest.merge_disposition_counts(
            db_path,
            '2026-05-10T00:00:00+00:00',
            '2026-05-10T23:59:59+00:00',
        )
        assert counts == {}, f"Expected {{}} for missing DB; got {counts}"
        assert not db_path.exists(), "merge_disposition_counts must not create a stub DB file"

    def test_schema_missing_returns_empty_dict(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """DB file exists but has no 'events' table — returns {} (fail-open) with WARNING.

        This is the corrupt/unreadable-DB branch of the fail-open contract
        (digest.py's `_query_events_ro` except-clause, WARNING path) — the one
        most worth asserting since the whole point of fail-open stats
        collection is that it never raises even when the DB is present but
        unreadable in the expected shape. Mirrors
        TestCountDoneInWindow.test_schema_missing_returns_zero.
        """
        db_path = tmp_path / 'schema-less.db'
        # Create a real but schema-less SQLite file (no EventStore setup).
        conn = sqlite3.connect(str(db_path))
        conn.close()
        assert db_path.exists(), "pre-condition: file must exist"

        with caplog.at_level(logging.WARNING, logger='orchestrator.digest'):
            counts = digest.merge_disposition_counts(
                db_path,
                '2026-05-10T00:00:00+00:00',
                '2026-05-10T23:59:59+00:00',
            )
        assert counts == {}, f"Expected {{}} for schema-less DB; got {counts}"
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and r.name == 'orchestrator.digest']
        assert len(warnings) == 1, f"Schema-missing must WARNING exactly once; got {len(warnings)}: {[r.message for r in warnings]}"


# ---------------------------------------------------------------------------
# Fixtures for async CostStore tests
# ---------------------------------------------------------------------------

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

    @pytest.mark.asyncio
    async def test_delegates_to_cost_totals_in_window(self, tmp_path: Path) -> None:
        """cost_in_window delegates to CostStore.cost_totals_in_window (not direct SQL).

        Monkeypatches cost_totals_in_window to a recording stub that returns
        known (total, watcher) tuples.  Asserts:
        - exactly two calls were made
        - first call passes window_start_iso and window_end_iso (positional or keyword)
        - second call passes cutoff ≈ now-24h and end ≈ now (within 10-second tolerance)
        - returned CostStats carries stubbed values into the correct fields
        """
        store = CostStore(tmp_path / 'cost.db')
        await store.open()
        try:
            window_start = '2026-05-10T00:00:00+00:00'
            window_end = '2026-05-10T23:59:59+00:00'

            # Stub returns different values per call so we can distinguish them.
            stub = AsyncMock(side_effect=[
                (7.0, 5.0),   # first call: window → (total_win, watcher_win)
                (9.0, 3.0),   # second call: 24h → (total_24h, watcher_24h)
            ])
            store.cost_totals_in_window = stub

            before = _datetime.now(UTC)
            stats = await digest.cost_in_window(store, window_start, window_end)
            after = _datetime.now(UTC)

            # Two calls were made.
            assert stub.call_count == 2, f'Expected 2 calls, got {stub.call_count}'

            # First call passes the window bounds (positional or keyword).
            first_call = stub.call_args_list[0]
            first_start = (first_call.args[0] if first_call.args
                           else first_call.kwargs.get('start_iso'))
            first_end = (first_call.args[1] if len(first_call.args) > 1
                         else first_call.kwargs.get('end_iso'))
            assert first_start == window_start, f'First call start wrong: {first_start!r}'
            assert first_end == window_end, f'First call end wrong: {first_end!r}'

            # Second call: cutoff ≈ now-24h, end ≈ now (positional or keyword).
            second_call = stub.call_args_list[1]
            cutoff_iso = (second_call.args[0] if second_call.args
                          else second_call.kwargs.get('start_iso'))
            now_iso = (second_call.args[1] if len(second_call.args) > 1
                       else second_call.kwargs.get('end_iso'))
            assert cutoff_iso is not None, 'start_iso arg missing from cost_totals_in_window call'
            assert now_iso is not None, 'end_iso arg missing from cost_totals_in_window call'
            cutoff_dt = _datetime.fromisoformat(cutoff_iso)
            now_dt = _datetime.fromisoformat(now_iso)
            expected_cutoff_lo = before - timedelta(hours=24) - timedelta(seconds=10)
            expected_cutoff_hi = after - timedelta(hours=24) + timedelta(seconds=10)
            assert expected_cutoff_lo <= cutoff_dt <= expected_cutoff_hi, (
                f'cutoff_dt {cutoff_dt} not within 10s of (now-24h)'
            )
            assert before - timedelta(seconds=10) <= now_dt <= after + timedelta(seconds=10), (
                f'now_dt {now_dt} not within 10s of now'
            )

            # Returned CostStats uses the stubbed values.
            assert stats.total_cost_in_window == pytest.approx(7.0)
            assert stats.watcher_cost_in_window == pytest.approx(5.0)
            assert stats.total_cost_24h == pytest.approx(9.0)
            assert stats.watcher_cost_24h == pytest.approx(3.0)
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# TestRenderDigestMarkdown (step-11)
# ---------------------------------------------------------------------------


def _make_digest_inputs(*, tripped: bool = False) -> digest.DigestInputs:
    """Build a concrete DigestInputs for render tests."""
    esc_stats = digest.EscalationStats(
        category_level_status_counts={
            ('infra_issue', 0, 'resolved'): 3,
            ('scope_violation', 1, 'pending'): 2,
            ('risk_identified', 0, 'dismissed'): 1,
        },
        first_ts='2026-05-10T08:00:00+00:00',
        last_ts='2026-05-10T20:00:00+00:00',
        dedupe_children_total=4,
        pending_total=2,
    )
    cost_stats = digest.CostStats(
        watcher_cost_in_window=5.25,
        total_cost_in_window=12.50,
        watcher_cost_24h=8.00,
        total_cost_24h=20.00,
    )
    return digest.DigestInputs(
        window_start_iso='2026-05-10T00:00:00+00:00',
        window_end_iso='2026-05-10T23:59:59+00:00',
        escalation_stats=esc_stats,
        done_count=5,
        cost_stats=cost_stats,
        parked_live=3,
        parked_window_churn=7,
        ewa_value=2.5,
        ewa_threshold=3.0,
        tripped=tripped,
        anomaly_flags={
            'cost_spike': True,
            'park_spike': False,
            'ewa_above_threshold': tripped,
            'infra_dedupe_active': True,
        },
        watcher_clusters=['infra × scope cross-pattern (3 events)', 'repeated timeout cluster'],
        dry_run_proposals=['unblock task-42: bump timeout', 'retry task-99 after requeue'],
    )


class TestRenderDigestMarkdown:
    """Tests for digest.render_digest_markdown(inputs) -> str."""

    def test_header_and_window_section(self) -> None:
        """Rendered markdown contains # Digest header and ## Window section."""
        md = digest.render_digest_markdown(_make_digest_inputs())
        assert '# Digest' in md
        assert '## Window' in md
        assert '2026-05-10T00:00:00+00:00' in md
        assert '2026-05-10T23:59:59+00:00' in md

    def test_escalation_outcomes_table(self) -> None:
        """Escalation outcomes section with category × level rows."""
        md = digest.render_digest_markdown(_make_digest_inputs())
        assert '## Escalation outcomes' in md
        assert 'infra_issue' in md
        assert 'scope_violation' in md
        assert 'risk_identified' in md
        # Status labels appear
        assert 'resolved' in md
        assert 'pending' in md
        assert 'dismissed' in md

    def test_tasks_done_section(self) -> None:
        """Tasks done in window count is rendered."""
        md = digest.render_digest_markdown(_make_digest_inputs())
        assert '## Tasks done in window' in md
        assert '## Tasks done in window\n5\n' in md  # done_count anchored to section header

    def test_cost_section(self) -> None:
        """Cost section includes watcher and total figures for window and 24h."""
        md = digest.render_digest_markdown(_make_digest_inputs())
        assert '## Cost' in md
        assert '$5.25' in md   # watcher_cost_in_window
        assert '$12.50' in md  # total_cost_in_window
        assert '$8.00' in md   # watcher_cost_24h
        assert '$20.00' in md  # total_cost_24h

    def test_parked_tasks_section(self) -> None:
        """Parked tasks section shows live count and window churn."""
        md = digest.render_digest_markdown(_make_digest_inputs())
        assert '## Parked tasks' in md
        assert re.search(r'- Live parked:\s+3\b', md)    # parked_live anchored to label (whitespace-flexible)
        assert re.search(r'- Window churn:\s+7\b', md)  # parked_window_churn anchored to label (whitespace-flexible)

    def test_ewa_section_not_tripped(self) -> None:
        """EWA line shows value/threshold; no TRIPPED marker when not tripped."""
        md = digest.render_digest_markdown(_make_digest_inputs(tripped=False))
        assert '## EWA' in md
        assert '- Value / threshold: 2.5000 / 3.0000\n' in md  # ewa_value / ewa_threshold rendered with :.4f, no TRIPPED on this path
        assert 'TRIPPED' not in md

    def test_ewa_section_tripped(self) -> None:
        """TRIPPED marker appears in EWA section when tripped=True."""
        md = digest.render_digest_markdown(_make_digest_inputs(tripped=True))
        assert '## EWA' in md
        assert 'TRIPPED' in md

    def test_anomalies_section_only_true_flags(self) -> None:
        """Anomalies section lists only flags set to True; False flags omitted."""
        md = digest.render_digest_markdown(_make_digest_inputs(tripped=True))
        assert '## Anomalies' in md
        assert 'cost_spike' in md
        assert 'ewa_above_threshold' in md
        assert 'infra_dedupe_active' in md
        # park_spike is False — must NOT appear
        assert 'park_spike' not in md

    def test_watcher_clusters_section(self) -> None:
        """Cross-escalation patterns section lists clusters."""
        md = digest.render_digest_markdown(_make_digest_inputs())
        assert '## Cross-escalation patterns' in md
        assert 'infra × scope cross-pattern' in md
        assert 'repeated timeout cluster' in md

    def test_watcher_clusters_empty(self) -> None:
        """When no clusters, renders 'none detected'."""
        inputs = _make_digest_inputs()
        inputs.watcher_clusters = []
        md = digest.render_digest_markdown(inputs)
        assert '## Cross-escalation patterns' in md
        assert 'none detected' in md

    def test_dry_run_proposals_section(self) -> None:
        """Dry-run proposals section shows count and first-line summaries."""
        md = digest.render_digest_markdown(_make_digest_inputs())
        assert '## Dry-run proposals' in md
        assert 'unblock task-42' in md
        assert 'retry task-99' in md

    def test_dry_run_proposals_empty(self) -> None:
        """When no dry-run proposals, renders 'none queued'."""
        inputs = _make_digest_inputs()
        inputs.dry_run_proposals = []
        md = digest.render_digest_markdown(inputs)
        assert '## Dry-run proposals' in md
        assert 'none queued' in md

    def test_stale_lane_census_defaults_empty(self) -> None:
        """DigestInputs constructs WITHOUT stale_lane_census (defaults to []).

        Leaf γ: the new field is defaulted so every existing constructor/call
        site (including _make_digest_inputs above) stays valid.
        """
        inputs = _make_digest_inputs()
        assert inputs.stale_lane_census == []

    def test_stale_lane_census_section_renders_lines(self) -> None:
        """Stale lane assignments section lists each census line (leaf γ)."""
        inputs = _make_digest_inputs()
        inputs.stale_lane_census = [
            '_lane-3 -> task 5260 (blocked), stale 10.2d',
            '_lane-9 -> task 4211 (in-progress), stale 8.0d',
        ]
        md = digest.render_digest_markdown(inputs)
        assert '## Stale lane assignments' in md
        assert '_lane-3 -> task 5260 (blocked), stale 10.2d' in md
        assert '_lane-9 -> task 4211 (in-progress), stale 8.0d' in md

    def test_stale_lane_census_section_empty(self) -> None:
        """When no stale lanes, the section renders _none_."""
        inputs = _make_digest_inputs()
        inputs.stale_lane_census = []
        md = digest.render_digest_markdown(inputs)
        assert '## Stale lane assignments\n_none_\n' in md


# ---------------------------------------------------------------------------
# TestWriteDigestEntry (step-13)
# ---------------------------------------------------------------------------


class TestWriteDigestEntry:
    """Tests for digest.write_digest_entry(digest_dir, inputs) -> DigestResult."""

    def test_single_call_creates_one_file(self, tmp_path: Path) -> None:
        """write_digest_entry writes exactly one .md file in digest_dir."""
        digest_dir = tmp_path / 'digests'
        result = digest.write_digest_entry(digest_dir, _make_digest_inputs())

        md_files = list(digest_dir.glob('digest-*.md'))
        assert len(md_files) == 1, f"Expected 1 .md file; got {len(md_files)}"
        assert result.path == md_files[0]
        assert result.path is not None

    def test_result_contains_rendered_markdown(self, tmp_path: Path) -> None:
        """The written file contains the rendered markdown (# Digest header)."""
        digest_dir = tmp_path / 'digests'
        result = digest.write_digest_entry(digest_dir, _make_digest_inputs())
        assert result.path is not None
        content = result.path.read_text()
        assert '# Digest' in content

    def test_filename_has_timestamp_prefix(self, tmp_path: Path) -> None:
        """File is named digest-<timestamp>.md."""
        digest_dir = tmp_path / 'digests'
        result = digest.write_digest_entry(digest_dir, _make_digest_inputs())
        assert result.path is not None
        assert result.path.name.startswith('digest-')
        assert result.path.suffix == '.md'

    def test_two_calls_create_two_files(self, tmp_path: Path) -> None:
        """Append-only: two calls create two separate files (never overwrites)."""
        import time
        digest_dir = tmp_path / 'digests'
        digest.write_digest_entry(digest_dir, _make_digest_inputs())
        time.sleep(0.01)  # ensure microsecond suffix differs
        digest.write_digest_entry(digest_dir, _make_digest_inputs())

        md_files = list(digest_dir.glob('digest-*.md'))
        assert len(md_files) == 2, f"Expected 2 files (append-only); got {len(md_files)}"

    def test_digest_dir_auto_created(self, tmp_path: Path) -> None:
        """digest_dir is created if it does not exist."""
        digest_dir = tmp_path / 'nested' / 'digests'
        assert not digest_dir.exists()
        digest.write_digest_entry(digest_dir, _make_digest_inputs())
        assert digest_dir.is_dir()

    def test_result_tripped_and_ewa_value(self, tmp_path: Path) -> None:
        """DigestResult.tripped and .ewa_value mirror inputs."""
        inputs = _make_digest_inputs(tripped=True)
        result = digest.write_digest_entry(tmp_path / 'digests', inputs)
        assert result.tripped is True
        assert result.ewa_value == pytest.approx(inputs.ewa_value)

    def test_read_only_dir_never_raises(self, tmp_path: Path) -> None:
        """write_digest_entry returns DigestResult(path=None) for an unwritable dir.

        Never raises — digest is best-effort observability.
        """
        import os
        digest_dir = tmp_path / 'ro_digests'
        digest_dir.mkdir()
        os.chmod(digest_dir, 0o500)
        try:
            result = digest.write_digest_entry(digest_dir, _make_digest_inputs())
            assert result.path is None, "Expected path=None on write failure"
        finally:
            os.chmod(digest_dir, 0o700)  # restore so tmp_path cleanup works


# ---------------------------------------------------------------------------
# Model×role outcome rollup (task 2534 δ, plans/adaptive-model-routing-prd.md,
# boundary test 12) — digest-side rollup: invocations × task_results (outcome
# rates, cap-hit rate, $/done) plus per-role turn-cap saturation from events.
# ---------------------------------------------------------------------------


def _seed_rollup_runs_db(db_path: Path, run_id: str = 'run-test') -> None:
    """Create a runs.db with events + task_results + invocations tables (schema only).

    Mirrors production layout (harness.py:1373-1434): EventStore, CostStore,
    and RunStore all share ONE runs.db file. Applies EventStore's and
    RunStore's own schema-creation path (both sync in __init__) plus
    CostStore's raw DDL directly (CostStore itself is async-only, so its
    schema is applied via executescript rather than via open()).
    """
    store = EventStore(db_path, run_id)
    del store  # schema created in __init__
    run_store = RunStore(db_path)
    del run_store  # schema created in __init__
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_COST_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _insert_invocation(
    db_path: Path,
    *,
    run_id: str,
    task_id: str | None,
    model: str,
    role: str,
    cost_usd: float,
    capped: bool,
    completed_at: str,
    project_id: str = 'p',
    account_name: str = 'a',
) -> None:
    """Insert one row into the invocations table (direct SQLite — CostStore is async-only)."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, duration_ms, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (run_id, task_id, project_id, account_name, model, role,
             cost_usd, 100, int(capped), completed_at, completed_at),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_task_result(
    db_path: Path,
    *,
    run_id: str,
    task_id: str,
    outcome: str,
    project_id: str = 'p',
    completed_at: str = '2026-05-10T00:00:00+00:00',
) -> None:
    """Insert one row into the task_results table."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            'INSERT INTO task_results (run_id, task_id, project_id, outcome, completed_at) '
            'VALUES (?, ?, ?, ?, ?)',
            (run_id, task_id, project_id, outcome, completed_at),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_event(
    db_path: Path,
    *,
    run_id: str,
    task_id: str | None,
    event_type: str,
    role: str,
    data: dict,
    timestamp: str,
) -> None:
    """Insert one row into the events table (mirrors _seed_events_db_direct above)."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            'INSERT INTO events (timestamp, run_id, task_id, event_type, role, data) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (timestamp, run_id, task_id, event_type, role, json.dumps(data)),
        )
        conn.commit()
    finally:
        conn.close()


class TestModelRoleRollupCore:
    """Tests for digest.model_role_rollup(runs_db, window_start_iso, window_end_iso) — core rollup (step-1/2)."""

    def test_rollup_rates_and_cost_per_done(self, tmp_path: Path) -> None:
        """Per-(model,role) invocation_count/done/blocked rates, cap_hit_rate,
        total_cost_usd, and cost_per_done (None when 0 done) — including
        steward/triage rows and a NULL-task_id module_tagger row."""
        db_path = tmp_path / 'runs.db'
        _seed_rollup_runs_db(db_path)
        run_id = 'run-test'
        window_start = '2026-05-10T00:00:00+00:00'
        window_end = '2026-05-10T23:59:59+00:00'

        _insert_task_result(db_path, run_id=run_id, task_id='t1', outcome='done')
        _insert_task_result(db_path, run_id=run_id, task_id='t2', outcome='blocked')
        _insert_task_result(db_path, run_id=run_id, task_id='t3', outcome='done')

        # (sonnet, implementer): 2 invocations across 2 tasks — t1 done, t2
        # blocked; the t2 invocation is capped.
        _insert_invocation(
            db_path, run_id=run_id, task_id='t1', model='sonnet', role='implementer',
            cost_usd=2.0, capped=False, completed_at='2026-05-10T08:00:00+00:00',
        )
        _insert_invocation(
            db_path, run_id=run_id, task_id='t2', model='sonnet', role='implementer',
            cost_usd=3.0, capped=True, completed_at='2026-05-10T09:00:00+00:00',
        )

        # (haiku, implementer): 1 done invocation.
        _insert_invocation(
            db_path, run_id=run_id, task_id='t3', model='haiku', role='implementer',
            cost_usd=1.0, capped=False, completed_at='2026-05-10T10:00:00+00:00',
        )

        # steward + triage rows (dep 2461's role values) — task t4 has no
        # task_results row, so both LEFT JOIN to a NULL outcome.
        _insert_invocation(
            db_path, run_id=run_id, task_id='t4', model='opus', role='steward',
            cost_usd=0.5, capped=False, completed_at='2026-05-10T11:00:00+00:00',
        )
        _insert_invocation(
            db_path, run_id=run_id, task_id='t4', model='opus', role='triage',
            cost_usd=0.2, capped=False, completed_at='2026-05-10T11:05:00+00:00',
        )

        # module_tagger row: task_id=NULL (harness.py ~2036 threading, step-10).
        _insert_invocation(
            db_path, run_id=run_id, task_id=None, model='haiku', role='module_tagger',
            cost_usd=0.1, capped=False, completed_at='2026-05-10T12:00:00+00:00',
        )

        # Out-of-window row — must be excluded from every aggregate.
        _insert_invocation(
            db_path, run_id=run_id, task_id='t5', model='sonnet', role='implementer',
            cost_usd=99.0, capped=False, completed_at='2026-05-09T00:00:00+00:00',
        )

        rollup = digest.model_role_rollup(db_path, window_start, window_end)
        by_key = {(r.model, r.role): r for r in rollup.rows}

        sonnet_impl = by_key[('sonnet', 'implementer')]
        assert sonnet_impl.invocation_count == 2
        assert sonnet_impl.done_count == 1
        assert sonnet_impl.blocked_count == 1
        assert sonnet_impl.done_rate == pytest.approx(0.5)
        assert sonnet_impl.blocked_rate == pytest.approx(0.5)
        assert sonnet_impl.capped_count == 1
        assert sonnet_impl.cap_hit_rate == pytest.approx(0.5)
        # Out-of-window 99.0 row excluded — total is 2.0 + 3.0, not + 99.0.
        assert sonnet_impl.total_cost_usd == pytest.approx(5.0)
        assert sonnet_impl.cost_per_done == pytest.approx(5.0)  # 5.0 / 1 distinct done task

        haiku_impl = by_key[('haiku', 'implementer')]
        assert haiku_impl.invocation_count == 1
        assert haiku_impl.done_count == 1
        assert haiku_impl.cost_per_done == pytest.approx(1.0)

        steward_row = by_key[('opus', 'steward')]
        assert steward_row.invocation_count == 1
        assert steward_row.done_count == 0
        assert steward_row.blocked_count == 0
        assert steward_row.cost_per_done is None

        triage_row = by_key[('opus', 'triage')]
        assert triage_row.invocation_count == 1
        assert triage_row.done_count == 0

        module_tagger_row = by_key[('haiku', 'module_tagger')]
        assert module_tagger_row.invocation_count == 1
        assert module_tagger_row.done_count == 0
        assert module_tagger_row.blocked_count == 0
        assert module_tagger_row.cost_per_done is None

    def test_missing_db_returns_empty_rollup(self, tmp_path: Path) -> None:
        """Non-existent DB returns an empty ModelRoleRollup (fail-open)."""
        db_path = tmp_path / 'no-such.db'
        rollup = digest.model_role_rollup(
            db_path, '2026-05-10T00:00:00+00:00', '2026-05-10T23:59:59+00:00',
        )
        assert rollup.rows == []


class TestModelRoleRollupTurnCapSaturation:
    """Tests for digest.model_role_rollup(...).turn_cap_saturation (step-3/4)."""

    def test_saturation_per_role(self, tmp_path: Path) -> None:
        """Per-role saturation = fraction of invocation_end turns >= that
        role's MAX(routing_decision.max_turns) in the window; roles with no
        routing_decision max_turns report None (fail-open, not 0/1)."""
        db_path = tmp_path / 'runs.db'
        _seed_rollup_runs_db(db_path)
        run_id = 'run-test'
        window_start = '2026-05-10T00:00:00+00:00'
        window_end = '2026-05-10T23:59:59+00:00'

        # simple_task: max_turns=30, invocation_end turns=30 and turns=12 -> 1/2 = 0.5
        _insert_event(
            db_path, run_id=run_id, task_id='t1', event_type='routing_decision',
            role='simple_task', data={'max_turns': 30}, timestamp='2026-05-10T08:00:00+00:00',
        )
        _insert_event(
            db_path, run_id=run_id, task_id='t1', event_type='invocation_end',
            role='simple_task', data={'turns': 30}, timestamp='2026-05-10T08:01:00+00:00',
        )
        _insert_event(
            db_path, run_id=run_id, task_id='t2', event_type='invocation_end',
            role='simple_task', data={'turns': 12}, timestamp='2026-05-10T08:02:00+00:00',
        )

        # architect: max_turns=60, invocation_end turns=20 -> 0/1 = 0.0
        _insert_event(
            db_path, run_id=run_id, task_id='t3', event_type='routing_decision',
            role='architect', data={'max_turns': 60}, timestamp='2026-05-10T09:00:00+00:00',
        )
        _insert_event(
            db_path, run_id=run_id, task_id='t3', event_type='invocation_end',
            role='architect', data={'turns': 20}, timestamp='2026-05-10T09:01:00+00:00',
        )

        # no_max_turns_role: invocation_end rows but no routing_decision -> None
        _insert_event(
            db_path, run_id=run_id, task_id='t4', event_type='invocation_end',
            role='no_max_turns_role', data={'turns': 5}, timestamp='2026-05-10T10:00:00+00:00',
        )

        # Out-of-window invocation_end — must be excluded.
        _insert_event(
            db_path, run_id=run_id, task_id='t5', event_type='invocation_end',
            role='simple_task', data={'turns': 999}, timestamp='2026-05-09T00:00:00+00:00',
        )

        rollup = digest.model_role_rollup(db_path, window_start, window_end)

        assert rollup.turn_cap_saturation['simple_task'] == pytest.approx(0.5)
        assert rollup.turn_cap_saturation['architect'] == pytest.approx(0.0)
        assert rollup.turn_cap_saturation['no_max_turns_role'] is None


def _make_model_role_rollup() -> digest.ModelRoleRollup:
    """Build a concrete ModelRoleRollup for render tests (step-5)."""
    return digest.ModelRoleRollup(
        rows=[
            digest.ModelRoleRow(
                model='sonnet', role='implementer',
                invocation_count=4, done_count=2, blocked_count=1,
                done_rate=0.5, blocked_rate=0.25,
                capped_count=1, cap_hit_rate=0.25,
                total_cost_usd=8.0, cost_per_done=4.0,
            ),
            digest.ModelRoleRow(
                model='opus', role='steward',
                invocation_count=1, done_count=0, blocked_count=0,
                done_rate=0.0, blocked_rate=0.0,
                capped_count=0, cap_hit_rate=0.0,
                total_cost_usd=0.5, cost_per_done=None,
            ),
        ],
        turn_cap_saturation={
            'simple_task': 0.5,
            'architect': 0.0,
            'no_max_turns_role': None,
        },
    )


class TestRenderDigestRollupSection:
    """Tests for the '## Per-(model×role) rollup' section of render_digest_markdown (step-5/6)."""

    def test_rollup_table_renders_known_cell(self) -> None:
        inputs = _make_digest_inputs()
        inputs.model_role_rollup = _make_model_role_rollup()
        md = digest.render_digest_markdown(inputs)
        assert '## Per-(model×role) rollup' in md
        assert re.search(
            r'\| sonnet \| implementer \| 4 \| 50\.0% \| 25\.0% \| 25\.0% \| \$4\.00 \|', md,
        ), md

    def test_none_cost_per_done_renders_placeholder(self) -> None:
        inputs = _make_digest_inputs()
        inputs.model_role_rollup = _make_model_role_rollup()
        md = digest.render_digest_markdown(inputs)
        assert re.search(r'\| opus \| steward \| 1 \| 0\.0% \| 0\.0% \| 0\.0% \| — \|', md), md

    def test_turn_cap_saturation_subsection(self) -> None:
        inputs = _make_digest_inputs()
        inputs.model_role_rollup = _make_model_role_rollup()
        md = digest.render_digest_markdown(inputs)
        assert re.search(r'- simple_task: 50\.0%', md)
        assert re.search(r'- architect: 0\.0%', md)
        assert re.search(r'- no_max_turns_role: n/a', md)

    def test_empty_rollup_renders_none_placeholder_no_raise(self) -> None:
        inputs = _make_digest_inputs()
        inputs.model_role_rollup = digest.ModelRoleRollup()
        md = digest.render_digest_markdown(inputs)  # must not raise
        assert '## Per-(model×role) rollup' in md
        assert '_none_' in md
