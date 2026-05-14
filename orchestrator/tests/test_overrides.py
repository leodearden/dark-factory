"""Tests for the OverrideStore SQLite persistence layer."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


class TestSchema:
    def test_creates_overrides_table_and_index_with_wal(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        db_path = tmp_path / 'sub' / 'scheduler_overrides.db'
        _store = OverrideStore(db_path)

        # Parent dirs must have been auto-created
        assert db_path.parent.exists()
        assert db_path.exists()

        conn = sqlite3.connect(str(db_path))
        try:
            # Table exists
            names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert 'overrides' in names

            # Index exists
            index_names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            }
            assert 'idx_overrides_pinned' in index_names

            # WAL mode
            mode = conn.execute('PRAGMA journal_mode').fetchone()[0]
            assert mode == 'wal'
        finally:
            conn.close()


class TestSetGet:
    def test_set_boost_then_get_overrides_returns_row(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideRow, OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'task-A', boost_tier='high')

        result = store.get_overrides('proj')
        assert result == {
            'task-A': OverrideRow(
                boost_tier='high',
                pinned=False,
                pin_order=None,
                reserve_now=False,
                ttl_until=None,
            )
        }

    def test_get_overrides_does_not_return_other_projects(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj-A', 'task-1', boost_tier='low')
        store.set_override('proj-B', 'task-2', boost_tier='critical')

        result_a = store.get_overrides('proj-A')
        assert 'task-1' in result_a
        assert 'task-2' not in result_a

        result_b = store.get_overrides('proj-B')
        assert 'task-2' in result_b
        assert 'task-1' not in result_b

    def test_set_override_rejects_unknown_boost_tier(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        with pytest.raises(ValueError, match='boost_tier'):
            store.set_override('proj', 'A', boost_tier='urgent')

    def test_set_override_accepts_valid_boost_tiers(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        for tier in ('critical', 'high', 'medium', 'low', 'polish'):
            store.set_override('proj', f'task-{tier}', boost_tier=tier)
            result = store.get_overrides('proj')
            assert result[f'task-{tier}'].boost_tier == tier


class TestPinning:
    def test_pin_auto_assigns_pin_order_max_plus_one(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')

        # Pin A and B without explicit pin_order
        store.set_override('proj', 'A', pinned=True)
        store.set_override('proj', 'B', pinned=True)

        overrides = store.get_overrides('proj')
        assert overrides['A'].pin_order == 1
        assert overrides['B'].pin_order == 2

        # Pin C with explicit pin_order=5
        store.set_override('proj', 'C', pinned=True, pin_order=5)
        overrides = store.get_overrides('proj')
        assert overrides['C'].pin_order == 5

        # Pin D without explicit pin_order — should get MAX(5)+1 = 6
        store.set_override('proj', 'D', pinned=True)
        overrides = store.get_overrides('proj')
        assert overrides['D'].pin_order == 6

    def test_pin_order_collision_raises_with_both_ids(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore, PinOrderCollision

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'A', pinned=True, pin_order=3)

        with pytest.raises(PinOrderCollision) as exc_info:
            store.set_override('proj', 'B', pinned=True, pin_order=3)

        msg = str(exc_info.value)
        assert 'A' in msg
        assert 'B' in msg
        assert '3' in msg

        # Store must be left untouched — B was not written
        overrides = store.get_overrides('proj')
        assert 'B' not in overrides


class TestPinQueue:
    def test_get_pin_queue_returns_pinned_rows_in_pin_order_asc(
        self, tmp_path: Path
    ) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'A', pinned=True, pin_order=3)
        store.set_override('proj', 'B', pinned=True, pin_order=1)
        store.set_override('proj', 'C', pinned=True, pin_order=2)
        # D is boost-only, not pinned
        store.set_override('proj', 'D', boost_tier='high')

        queue = store.get_pin_queue('proj')
        task_ids = [tid for tid, _ in queue]
        assert task_ids == ['B', 'C', 'A']

        # D must not appear in the pin queue
        assert all(tid != 'D' for tid, _ in queue)


class TestClear:
    def test_clear_override_none_deletes_row(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'A', boost_tier='high', pinned=True)

        result = store.clear_override('proj', 'A')
        assert result is True
        assert 'A' not in store.get_overrides('proj')

    def test_clear_override_field_boost_tier_keeps_row(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'A', boost_tier='high', pinned=True)

        result = store.clear_override('proj', 'A', field='boost_tier')
        assert result is True

        overrides = store.get_overrides('proj')
        assert 'A' in overrides
        assert overrides['A'].boost_tier is None
        assert overrides['A'].pinned is True

    def test_clear_override_field_pinned_clears_pin_order_too(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'A', pinned=True, pin_order=2)

        store.clear_override('proj', 'A', field='pinned')

        overrides = store.get_overrides('proj')
        assert 'A' in overrides
        assert overrides['A'].pinned is False
        assert overrides['A'].pin_order is None

    def test_clear_override_missing_row_returns_false(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        result = store.clear_override('proj', 'nonexistent')
        assert result is False


class TestReorder:
    def test_reorder_pin_queue_rewrites_pin_order(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'A', pinned=True, pin_order=1)
        store.set_override('proj', 'B', pinned=True, pin_order=2)
        store.set_override('proj', 'C', pinned=True, pin_order=3)

        store.reorder_pin_queue('proj', ['C', 'A', 'B'])

        queue = store.get_pin_queue('proj')
        task_ids = [tid for tid, _ in queue]
        assert task_ids == ['C', 'A', 'B']

    def test_reorder_pin_queue_accepts_csv_string(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'A', pinned=True, pin_order=1)
        store.set_override('proj', 'B', pinned=True, pin_order=2)
        store.set_override('proj', 'C', pinned=True, pin_order=3)

        store.reorder_pin_queue('proj', 'C,A,B')

        queue = store.get_pin_queue('proj')
        task_ids = [tid for tid, _ in queue]
        assert task_ids == ['C', 'A', 'B']

    def test_reorder_pin_queue_rejects_missing_or_extra_ids(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'A', pinned=True, pin_order=1)
        store.set_override('proj', 'B', pinned=True, pin_order=2)
        store.set_override('proj', 'C', pinned=True, pin_order=3)

        # Omitting B
        with pytest.raises(ValueError):
            store.reorder_pin_queue('proj', ['C', 'A'])

        # Including a non-pinned / unknown task
        with pytest.raises(ValueError):
            store.reorder_pin_queue('proj', ['C', 'A', 'B', 'D'])


class TestSweeps:
    def test_clear_terminal_deletes_named_owners(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'A', boost_tier='high')
        store.set_override('proj', 'B', boost_tier='medium')
        store.set_override('proj', 'C', boost_tier='low')

        cleared = store.clear_terminal('proj', {'A', 'C'})
        assert sorted(cleared) == ['A', 'C']

        remaining = store.get_overrides('proj')
        assert list(remaining.keys()) == ['B']

    def test_clear_expired_deletes_past_ttl(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        now = datetime(2026, 5, 14, tzinfo=UTC)
        past = now - timedelta(hours=1)
        future = now + timedelta(hours=1)

        store.set_override('proj', 'A', ttl_until=past)
        store.set_override('proj', 'B', ttl_until=future)
        store.set_override('proj', 'C')  # no TTL

        cleared = store.clear_expired('proj', now)
        assert cleared == ['A']

        remaining = store.get_overrides('proj')
        assert 'A' not in remaining
        assert 'B' in remaining
        assert 'C' in remaining


class TestSeparateDB:
    def test_override_survives_simulated_set_task_status_cycle(
        self, tmp_path: Path
    ) -> None:
        """Regression guard: override rows survive unrelated DB writes.

        This test encodes the invariant that overrides live in a *separate*
        SQLite file from taskmaster state.  If someone ever consolidates the
        storage, this test will fail and force a conscious decision.
        """
        from orchestrator.overrides import OverrideRow, OverrideStore

        db_path = tmp_path / 'scheduler_overrides.db'
        store = OverrideStore(db_path)
        store.set_override('proj', 'A', boost_tier='high', pinned=True)

        # Simulate what set_task_status / a reify cycle does: create or
        # overwrite an unrelated SQLite file in the same directory.
        taskmaster_db = tmp_path / 'taskmaster.db'
        import sqlite3

        conn = sqlite3.connect(str(taskmaster_db))
        conn.execute('CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT)')
        conn.execute("INSERT INTO tasks VALUES ('A', 'in_progress')")
        conn.commit()
        conn.close()

        # The override must still be intact.  Re-open from the same path to
        # exercise the connect-per-call pattern rather than any cached state.
        store2 = OverrideStore(db_path)
        overrides = store2.get_overrides('proj')

        assert 'A' in overrides
        row = overrides['A']
        assert row == OverrideRow(
            boost_tier='high',
            pinned=True,
            pin_order=1,
            reserve_now=False,
            ttl_until=None,
        )
