"""Tests for sqlite_sync_base: apply_full_durability_pragmas_sync and CheckpointResult.

Mirrors the layout of shared/tests/test_async_sqlite_base.py::TestApplyFullDurabilityPragmas
for the sync variant.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock


class TestApplyFullDurabilityPragmasSync:
    """apply_full_durability_pragmas_sync(conn, busy_timeout_ms) sets all five PRAGMAs."""

    def test_applies_full_pragma_triad(self, tmp_path: Path) -> None:
        """A single call sets journal_mode, busy_timeout, synchronous,
        wal_autocheckpoint, and journal_size_limit — all five PRAGMAs.

        Uses busy_timeout_ms=12345 so we can distinguish the configured value
        from any SQLite default (5000 is a plausible default; 12345 is not).
        """
        from shared.sqlite_sync_base import apply_full_durability_pragmas_sync

        db_path = tmp_path / 'test.db'
        conn = sqlite3.connect(str(db_path))
        try:
            apply_full_durability_pragmas_sync(conn, busy_timeout_ms=12345)

            journal_row = conn.execute('PRAGMA journal_mode').fetchone()
            timeout_row = conn.execute('PRAGMA busy_timeout').fetchone()
            sync_row = conn.execute('PRAGMA synchronous').fetchone()
            checkpoint_row = conn.execute('PRAGMA wal_autocheckpoint').fetchone()
            size_row = conn.execute('PRAGMA journal_size_limit').fetchone()
        finally:
            conn.close()

        assert journal_row is not None and journal_row[0] == 'wal'
        assert timeout_row is not None and timeout_row[0] == 12345
        assert sync_row is not None and sync_row[0] == 2  # 2 == FULL
        assert checkpoint_row is not None and checkpoint_row[0] == 100
        assert size_row is not None and size_row[0] == 67108864

    def test_raises_runtime_error_on_non_wal_journal_mode(self) -> None:
        """Raises RuntimeError when PRAGMA journal_mode returns non-WAL result
        (e.g. 'delete' on a filesystem that doesn't support WAL)."""
        from shared.sqlite_sync_base import apply_full_durability_pragmas_sync

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ('delete',)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_cursor

        import pytest
        with pytest.raises(RuntimeError, match='WAL'):
            apply_full_durability_pragmas_sync(mock_conn, busy_timeout_ms=5000)


class TestCheckpointResultReexport:
    """CheckpointResult is re-exported from sqlite_sync_base (not a duplicate type)."""

    def test_checkpoint_result_importable(self) -> None:
        """CheckpointResult is importable from shared.sqlite_sync_base."""
        from shared.sqlite_sync_base import CheckpointResult
        assert CheckpointResult is not None

    def test_checkpoint_result_is_same_type_as_async_base(self) -> None:
        """CheckpointResult from sync module is the SAME object as from async_sqlite_base.

        This ensures it's a re-export, not a duplicate NamedTuple definition.
        """
        from shared.async_sqlite_base import CheckpointResult as AsyncCheckpointResult
        from shared.sqlite_sync_base import CheckpointResult as SyncCheckpointResult

        assert SyncCheckpointResult is AsyncCheckpointResult

    def test_checkpoint_result_namedtuple_fields(self) -> None:
        """CheckpointResult has (busy, log, checkpointed) fields."""
        from shared.sqlite_sync_base import CheckpointResult

        result = CheckpointResult(busy=0, log=5, checkpointed=5)
        assert result.busy == 0
        assert result.log == 5
        assert result.checkpointed == 5
