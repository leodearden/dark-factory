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


class TestMergeFinalizedEventType:
    """EventType.merge_finalized must exist and round-trip through the event store."""

    def test_merge_finalized_member_exists_and_name_equals_value(self) -> None:
        """merge_finalized must exist with value == name (project convention)."""
        assert EventType.merge_finalized == 'merge_finalized'
        assert EventType.merge_finalized.value == EventType.merge_finalized.name

    def test_merge_finalized_round_trip(self, tmp_path: Path) -> None:
        """Emitting merge_finalized writes a row whose data round-trips key fields."""
        db_path = tmp_path / 'mf_test.db'
        store = EventStore(db_path, 'run-mf')
        store.emit(
            EventType.merge_finalized,
            task_id='42',
            phase='merge',
            data={
                'request_id': 'mr-abcd1234',
                'branch': 'task/42',
                'state': 'done',
                'merge_sha': 'deadbeef',
            },
        )

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, task_id, "
            "json_extract(data, '$.request_id') AS request_id, "
            "json_extract(data, '$.branch') AS branch, "
            "json_extract(data, '$.state') AS state "
            "FROM events WHERE event_type = 'merge_finalized'"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == 'merge_finalized'
        assert rows[0][1] == '42'
        assert rows[0][2] == 'mr-abcd1234'
        assert rows[0][3] == 'task/42'
        assert rows[0][4] == 'done'


class TestSchedulerOverrideEventTypes:
    """Assert the five priority-override event types exist on EventType.

    Collapsed from five separate assertions into a single membership check.
    The scheduler integration tests (TestOverrideEventEmission) exercise the
    actual emission behavior end-to-end; this test only guards against the
    enum members being accidentally removed.
    """

    def test_override_event_types_are_present(self) -> None:
        expected = {
            'priority_override_set',
            'priority_override_cleared',
            'task_pinned',
            'task_unpinned',
            'pin_queue_reordered',
        }
        actual = {m.name for m in EventType}
        assert expected <= actual, (
            f'Missing EventType members: {expected - actual}'
        )


class TestSchedulerPauseEventTypes:
    """Assert the two scheduler pause event types exist and can round-trip through EventStore."""

    def test_scheduler_pause_event_types_defined(self) -> None:
        """scheduler_paused and scheduler_resumed must exist as EventType members
        with the expected string values."""
        assert EventType.scheduler_paused.value == 'scheduler_paused', (
            f'Expected scheduler_paused, got {EventType.scheduler_paused.value!r}'
        )
        assert EventType.scheduler_resumed.value == 'scheduler_resumed', (
            f'Expected scheduler_resumed, got {EventType.scheduler_resumed.value!r}'
        )

    def test_scheduler_paused_event_writes_and_reads_back(self, tmp_path: Path) -> None:
        """Emitting scheduler_paused writes a row with correct event_type and data JSON."""
        db_path = tmp_path / 'events.db'
        store = EventStore(db_path, 'run-pause-test')
        store.emit(
            EventType.scheduler_paused,
            data={'reason': 'park-stop: 5 tasks blocked', 'threshold': 5, 'window_hours': 1.0},
        )

        rows = _query_all(db_path)
        assert len(rows) == 1
        row = rows[0]
        assert row['event_type'] == 'scheduler_paused'
        data = json.loads(row['data'])
        assert data['reason'] == 'park-stop: 5 tasks blocked'
        assert data['threshold'] == 5
        assert data['window_hours'] == 1.0

    def test_scheduler_pause_restored_event_type_defined(self) -> None:
        """scheduler_pause_restored must exist as an EventType member with the expected value."""
        assert EventType.scheduler_pause_restored.value == 'scheduler_pause_restored', (
            f'Expected scheduler_pause_restored, '
            f'got {EventType.scheduler_pause_restored.value!r}'
        )

    def test_scheduler_pause_restored_event_writes_and_reads_back(
        self, tmp_path: Path
    ) -> None:
        """Emitting scheduler_pause_restored writes a row with the correct payload."""
        db_path = tmp_path / 'events.db'
        store = EventStore(db_path, 'run-restored-test')
        store.emit(
            EventType.scheduler_pause_restored,
            data={
                'reason': 'park-stop: 5 tasks blocked',
                'pause_at': '2026-05-13T22:00:00+00:00',
                'restored_from_run_id': 'prior-run-001',
            },
        )

        rows = _query_all(db_path)
        assert len(rows) == 1
        row = rows[0]
        assert row['event_type'] == 'scheduler_pause_restored'
        data = json.loads(row['data'])
        assert data['reason'] == 'park-stop: 5 tasks blocked'
        assert data['pause_at'] == '2026-05-13T22:00:00+00:00'
        assert data['restored_from_run_id'] == 'prior-run-001'


class TestTrainEventTypes:
    """Assert the four train_* event types exist and can round-trip through EventStore."""

    def test_train_started_value(self) -> None:
        assert EventType.train_started.value == 'train_started'
        assert EventType.train_started.name == EventType.train_started.value

    def test_train_member_deferred_value(self) -> None:
        assert EventType.train_member_deferred.value == 'train_member_deferred'
        assert EventType.train_member_deferred.name == EventType.train_member_deferred.value

    def test_train_merged_value(self) -> None:
        assert EventType.train_merged.value == 'train_merged'
        assert EventType.train_merged.name == EventType.train_merged.value

    def test_train_derailed_value(self) -> None:
        assert EventType.train_derailed.value == 'train_derailed'
        assert EventType.train_derailed.name == EventType.train_derailed.value

    def test_all_four_train_events_round_trip(self, tmp_path: Path) -> None:
        """Emit all four train_* events with documented payloads and verify rows exist."""
        db_path = tmp_path / 'events.db'
        store = EventStore(db_path, 'run-train-test')

        store.emit(
            EventType.train_started,
            task_id='trn-a',
            phase='merge',
            data={
                'train_id': 'train-001',
                'member_task_ids': ['trn-a', 'trn-b', 'trn-c'],
                'member_count': 3,
                'base_sha': 'abc123',
            },
        )
        store.emit(
            EventType.train_member_deferred,
            task_id='trn-b',
            phase='merge',
            data={
                'train_id': 'train-001',
                'deferred_task_id': 'trn-b',
                'deferred_reason': 'status is planning, expected merge-deferred',
                'remaining_members': ['trn-a', 'trn-c'],
            },
        )
        store.emit(
            EventType.train_merged,
            task_id='trn-a',
            phase='merge',
            data={
                'train_id': 'train-001',
                'member_task_ids': ['trn-a', 'trn-b', 'trn-c'],
                'merge_commit_sha': 'def456',
                'base_sha': 'abc123',
            },
        )
        store.emit(
            EventType.train_derailed,
            task_id='trn-a',
            phase='merge',
            data={
                'train_id': 'train-001',
                'member_task_ids': ['trn-a', 'trn-b', 'trn-c'],
                'derail_reason': 'verify failed: test suite red',
            },
        )

        rows = _query_all(db_path)
        assert len(rows) == 4

        types = [r['event_type'] for r in rows]
        assert types == [
            'train_started',
            'train_member_deferred',
            'train_merged',
            'train_derailed',
        ]

        started_data = json.loads(rows[0]['data'])
        assert started_data['train_id'] == 'train-001'
        assert started_data['member_count'] == 3
        assert started_data['base_sha'] == 'abc123'

        deferred_data = json.loads(rows[1]['data'])
        assert deferred_data['deferred_task_id'] == 'trn-b'
        assert 'planning' in deferred_data['deferred_reason']
        assert deferred_data['remaining_members'] == ['trn-a', 'trn-c']

        merged_data = json.loads(rows[2]['data'])
        assert merged_data['merge_commit_sha'] == 'def456'
        assert merged_data['base_sha'] == 'abc123'

        derailed_data = json.loads(rows[3]['data'])
        assert 'verify' in derailed_data['derail_reason']
