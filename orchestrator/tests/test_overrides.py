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
