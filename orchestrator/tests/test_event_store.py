"""Tests for the event store."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from orchestrator.event_store import EventStore, EventType


def _query_all(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT * FROM events ORDER BY id').fetchall()
    conn.close()
    return [dict(r) for r in rows]


class TestEventStore:
    def test_emit_writes_event(self, tmp_path: Path) -> None:
        db_path = tmp_path / 'test.db'
        store = EventStore(db_path, 'run-abc123')
        store.emit(
            EventType.invocation_end,
            task_id='42',
            phase='execute',
            role='implementer',
            cost_usd=0.35,
            duration_ms=12000,
            data={'turns': 8, 'model': 'opus', 'success': True},
        )

        rows = _query_all(db_path)
        assert len(rows) == 1
        row = rows[0]
        assert row['run_id'] == 'run-abc123'
        assert row['task_id'] == '42'
        assert row['event_type'] == 'invocation_end'
        assert row['phase'] == 'execute'
        assert row['role'] == 'implementer'
        assert row['cost_usd'] == 0.35
        assert row['duration_ms'] == 12000

        data = json.loads(row['data'])
        assert data['turns'] == 8
        assert data['model'] == 'opus'
        assert data['success'] is True

    def test_emit_multiple_events(self, tmp_path: Path) -> None:
        db_path = tmp_path / 'test.db'
        store = EventStore(db_path, 'run-xyz')

        store.emit(EventType.task_started, task_id='10')
        store.emit(EventType.phase_enter, task_id='10', phase='plan')
        store.emit(
            EventType.invocation_end,
            task_id='10', phase='plan', role='architect',
            cost_usd=0.12,
        )
        store.emit(EventType.phase_exit, task_id='10', phase='plan', cost_usd=0.12)
        store.emit(EventType.task_completed, task_id='10', data={'outcome': 'done'})

        rows = _query_all(db_path)
        assert len(rows) == 5
        types = [r['event_type'] for r in rows]
        assert types == [
            'task_started', 'phase_enter', 'invocation_end',
            'phase_exit', 'task_completed',
        ]

    def test_emit_with_no_data(self, tmp_path: Path) -> None:
        db_path = tmp_path / 'test.db'
        store = EventStore(db_path, 'run-minimal')
        store.emit(EventType.waste_detected, task_id='5')

        rows = _query_all(db_path)
        assert len(rows) == 1
        assert json.loads(rows[0]['data']) == {}
        assert rows[0]['cost_usd'] is None
        assert rows[0]['duration_ms'] is None

    def test_fire_and_forget_bad_path(self, tmp_path: Path) -> None:
        """Emit on a broken DB path should not raise."""
        bad_path = tmp_path / 'nonexistent' / 'deep' / 'path' / 'db.sqlite'
        store = EventStore.__new__(EventStore)
        store.db_path = bad_path
        store.run_id = 'run-broken'

        # Should not raise
        store.emit(EventType.cap_hit, task_id='99', data={'test': True})

    def test_schema_idempotent(self, tmp_path: Path) -> None:
        """Creating EventStore twice on the same DB should not fail."""
        db_path = tmp_path / 'test.db'
        store1 = EventStore(db_path, 'run-1')
        store1.emit(EventType.task_started, task_id='1')

        store2 = EventStore(db_path, 'run-2')
        store2.emit(EventType.task_started, task_id='2')

        rows = _query_all(db_path)
        assert len(rows) == 2
        assert rows[0]['run_id'] == 'run-1'
        assert rows[1]['run_id'] == 'run-2'

    def test_all_event_types_valid(self) -> None:
        """Ensure all EventType values are strings."""
        for et in EventType:
            assert isinstance(et.value, str)
            assert et.value == et.name

    def test_event_type_includes_merge_queued_and_dequeued(self, tmp_path: Path) -> None:
        """merge_queued and merge_dequeued EventType members must exist and round-trip."""
        # Members exist
        assert EventType.merge_queued == 'merge_queued'
        assert EventType.merge_dequeued == 'merge_dequeued'

        # value == name (matching project convention)
        assert EventType.merge_queued.value == EventType.merge_queued.name
        assert EventType.merge_dequeued.value == EventType.merge_dequeued.name

        # Round-trip: emit and verify the row
        db_path = tmp_path / 'test.db'
        store = EventStore(db_path, 'run-1')
        store.emit(
            EventType.merge_queued,
            task_id='42',
            phase='merge',
            data={'branch': 'task/42'},
        )

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, task_id, json_extract(data, '$.branch') AS branch "
            "FROM events WHERE event_type = 'merge_queued'"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == 'merge_queued'
        assert rows[0][1] == '42'
        assert rows[0][2] == 'task/42'

    def test_query_by_event_type(self, tmp_path: Path) -> None:
        """Verify SQL queries against the events table work."""
        db_path = tmp_path / 'test.db'
        store = EventStore(db_path, 'run-q')

        store.emit(EventType.invocation_end, task_id='1', role='architect', cost_usd=0.10)
        store.emit(EventType.invocation_end, task_id='1', role='implementer', cost_usd=0.50)
        store.emit(EventType.invocation_end, task_id='1', role='reviewer', cost_usd=0.15)
        store.emit(EventType.waste_detected, task_id='1', data={'waste_type': 'replan'})

        conn = sqlite3.connect(str(db_path))
        # Aggregate cost by role
        rows = conn.execute(
            "SELECT role, sum(cost_usd) as total "
            "FROM events WHERE event_type = 'invocation_end' "
            "GROUP BY role ORDER BY total DESC"
        ).fetchall()
        conn.close()

        assert len(rows) == 3
        assert rows[0][0] == 'implementer'
        assert rows[0][1] == 0.50
