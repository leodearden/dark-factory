"""Tests for the ParkEvictionRequestStore SQLite persistence layer.

Mirrors test_overrides.py TestSchema pattern for schema/WAL introspection.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from orchestrator.config import OrchestratorConfig


class TestSchema:
    def test_creates_table_and_wal_on_init(self, tmp_path: Path) -> None:
        """Constructing the store auto-creates parent dirs, the DB file,
        the park_eviction_requests table, and enables WAL journal mode."""
        from orchestrator.park_eviction_requests import ParkEvictionRequestStore

        db_path = tmp_path / 'sub' / 'park_eviction_requests.db'
        _store = ParkEvictionRequestStore(db_path)

        # Parent dirs must have been auto-created.
        assert db_path.parent.exists()
        assert db_path.exists()

        conn = sqlite3.connect(str(db_path))
        try:
            # Table exists.
            names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert 'park_eviction_requests' in names

            # WAL mode.
            mode = conn.execute('PRAGMA journal_mode').fetchone()[0]
            assert mode == 'wal'
        finally:
            conn.close()

    def test_table_columns(self, tmp_path: Path) -> None:
        """The table has exactly the columns task_id, project_root, requested_at."""
        from orchestrator.park_eviction_requests import ParkEvictionRequestStore

        db_path = tmp_path / 'park_eviction_requests.db'
        _store = ParkEvictionRequestStore(db_path)

        conn = sqlite3.connect(str(db_path))
        try:
            cols = {
                row[1]
                for row in conn.execute(
                    'PRAGMA table_info(park_eviction_requests)'
                ).fetchall()
            }
        finally:
            conn.close()

        assert cols == {'task_id', 'project_root', 'requested_at'}

    def test_from_config_uses_canonical_path(self, tmp_path: Path) -> None:
        """from_config builds the store at config.park_eviction_requests_db_path."""
        from orchestrator.park_eviction_requests import ParkEvictionRequestStore

        config = OrchestratorConfig(project_root=tmp_path)
        store = ParkEvictionRequestStore.from_config(config)
        expected = tmp_path / 'data' / 'orchestrator' / 'park_eviction_requests.db'
        assert store.db_path == expected


class TestEnqueue:
    def test_enqueue_inserts_well_formed_row(self, tmp_path: Path) -> None:
        """enqueue(task_id, project_root) inserts one row with non-empty requested_at."""
        from orchestrator.park_eviction_requests import ParkEvictionRequestStore

        store = ParkEvictionRequestStore(tmp_path / 'park_eviction_requests.db')
        store.enqueue('task-1', '/proj/root')

        conn = sqlite3.connect(str(store.db_path))
        try:
            rows = conn.execute(
                'SELECT task_id, project_root, requested_at FROM park_eviction_requests'
            ).fetchall()
        finally:
            conn.close()

        assert len(rows) == 1
        task_id, project_root, requested_at = rows[0]
        assert task_id == 'task-1'
        assert project_root  # non-empty (normalized canonical path)
        assert requested_at  # non-empty ISO string

    def test_enqueue_with_explicit_requested_at_round_trips(self, tmp_path: Path) -> None:
        """Explicit requested_at is stored verbatim and returned in the stored row."""
        from orchestrator.park_eviction_requests import ParkEvictionRequestStore

        store = ParkEvictionRequestStore(tmp_path / 'park_eviction_requests.db')
        explicit_ts = '2026-01-15T09:30:00+00:00'
        store.enqueue('task-x', '/proj/root', requested_at=explicit_ts)

        conn = sqlite3.connect(str(store.db_path))
        try:
            rows = conn.execute(
                'SELECT task_id, requested_at FROM park_eviction_requests'
            ).fetchall()
        finally:
            conn.close()

        assert len(rows) == 1
        assert rows[0][0] == 'task-x'
        assert rows[0][1] == explicit_ts

    def test_enqueue_same_task_id_twice_both_rows_stored(self, tmp_path: Path) -> None:
        """Enqueueing the same task_id twice creates two rows (no uniqueness constraint)."""
        from orchestrator.park_eviction_requests import ParkEvictionRequestStore

        store = ParkEvictionRequestStore(tmp_path / 'park_eviction_requests.db')
        store.enqueue('dup', '/proj/root')
        store.enqueue('dup', '/proj/root')

        result = store.drain('/proj/root')
        # Both rows must be returned; force_clear is idempotent on the second.
        assert result.count('dup') == 2
        # Table is fully drained.
        assert store.drain('/proj/root') == []


class TestDrain:
    def test_drain_returns_queued_task_ids_and_deletes_them(
        self, tmp_path: Path
    ) -> None:
        """drain(project_root) returns the queued task_ids AND deletes them."""
        from orchestrator.park_eviction_requests import ParkEvictionRequestStore

        store = ParkEvictionRequestStore(tmp_path / 'park_eviction_requests.db')
        store.enqueue('T1', '/proj')
        store.enqueue('T2', '/proj')

        result = store.drain('/proj')
        assert set(result) == {'T1', 'T2'}

        # Second drain is empty — one-shot.
        second = store.drain('/proj')
        assert second == []

    def test_drain_is_project_root_scoped(self, tmp_path: Path) -> None:
        """drain(A) leaves rows for project B untouched."""
        from orchestrator.park_eviction_requests import ParkEvictionRequestStore

        store = ParkEvictionRequestStore(tmp_path / 'park_eviction_requests.db')
        store.enqueue('TA', '/proj/A')
        store.enqueue('TB', '/proj/B')

        result_a = store.drain('/proj/A')
        assert result_a == ['TA']

        # B's row is still there.
        result_b = store.drain('/proj/B')
        assert result_b == ['TB']

    def test_drain_matches_enqueue_after_normalization(self, tmp_path: Path) -> None:
        """drain() matches enqueue() even when path forms differ (normalization contract).

        Pins the canonical-form contract: if the fused-memory ε-layer enqueues
        with the resolved absolute path and the scheduler drains with the same
        resolved path, the WHERE project_root=? filter must match.
        """
        from orchestrator.park_eviction_requests import ParkEvictionRequestStore, _canonical_root

        store = ParkEvictionRequestStore(tmp_path / 'park_eviction_requests.db')
        # Use the resolved path (as config.project_root produces) for enqueue.
        resolved = _canonical_root(str(tmp_path / 'myproject'))
        store.enqueue('T1', resolved)

        # drain() with the same canonical form must find the row.
        result = store.drain(resolved)
        assert result == ['T1']
        assert store.drain(resolved) == []
