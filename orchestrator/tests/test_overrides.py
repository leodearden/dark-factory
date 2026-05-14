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
