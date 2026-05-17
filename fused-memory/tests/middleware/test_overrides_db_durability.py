"""Tests for scheduler_overrides.db pragma triad and checkpoint helper.

Mirrors the assertion style from:
- shared/tests/test_async_sqlite_base.py::TestApplyFullDurabilityPragmas
- orchestrator/tests/test_sqlite_pragmas.py::TestRunStorePragmas

Two test classes:
1. TestOpenOverridesDbPragmas — verifies _open_overrides_db applies all five
   Phase-3 durability pragmas on every returned connection.
2. TestCheckpointOverridesDb — verifies _checkpoint_overrides_db returns
   CheckpointResult, truncates the WAL, and is idempotent.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from shared.async_sqlite_base import CheckpointResult

from fused_memory.server.tools import _checkpoint_overrides_db, _open_overrides_db

# ---------------------------------------------------------------------------
# TestOpenOverridesDbPragmas
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOpenOverridesDbPragmas:
    """_open_overrides_db applies the full Phase-3 durability pragma triad."""

    @pytest.mark.parametrize(
        'autocommit,expected_busy',
        [
            (False, 5000),
            (True, 30000),
        ],
        ids=['autocommit=False', 'autocommit=True'],
    )
    async def test_open_overrides_db_applies_full_triad(
        self,
        tmp_path: Path,
        autocommit: bool,
        expected_busy: int,
    ) -> None:
        """_open_overrides_db sets all five Phase-3 pragmas on the returned connection.

        Per-connection pragmas (busy_timeout, synchronous, wal_autocheckpoint,
        journal_size_limit) are not persisted in the DB file — they must be
        verified on the connection that set them, which is exactly what this
        test does by holding the returned connection open.

        journal_mode persists in the DB file header and is also verified here
        for completeness.
        """
        project_root = str(tmp_path / 'proj')
        db = await _open_overrides_db(project_root, autocommit=autocommit)
        try:
            async with db.execute('PRAGMA journal_mode') as cur:
                journal_row = await cur.fetchone()
            async with db.execute('PRAGMA busy_timeout') as cur:
                timeout_row = await cur.fetchone()
            async with db.execute('PRAGMA synchronous') as cur:
                sync_row = await cur.fetchone()
            async with db.execute('PRAGMA wal_autocheckpoint') as cur:
                checkpoint_row = await cur.fetchone()
            async with db.execute('PRAGMA journal_size_limit') as cur:
                size_row = await cur.fetchone()
        finally:
            await db.close()

        assert journal_row is not None and journal_row[0] == 'wal'
        assert timeout_row is not None and timeout_row[0] == expected_busy
        assert sync_row is not None and sync_row[0] == 2  # 2 == FULL
        assert checkpoint_row is not None and checkpoint_row[0] == 100
        assert size_row is not None and size_row[0] == 67108864


# ---------------------------------------------------------------------------
# TestCheckpointOverridesDb
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCheckpointOverridesDb:
    """_checkpoint_overrides_db returns CheckpointResult, truncates WAL, and is idempotent."""

    async def test_returns_checkpoint_result_with_busy_zero(
        self,
        tmp_path: Path,
    ) -> None:
        """checkpoint on a fresh DB returns CheckpointResult with busy==0."""
        project_root = str(tmp_path / 'proj')
        result = await _checkpoint_overrides_db(project_root)
        assert isinstance(result, CheckpointResult)
        assert result.busy == 0

    async def test_truncates_wal_after_writes(
        self,
        tmp_path: Path,
    ) -> None:
        """After writes that grow the WAL, checkpoint truncates WAL to ≤32 bytes.

        Uses a long-lived raw sqlite3.connect writer that stays open across all
        commits (preventing the PASSIVE close-checkpoint from firing between
        writes).  WAL size is verified while the writer is open, then
        checkpoint() is called after closing the writer.

        Mirrors orchestrator/tests/test_sqlite_pragmas.py:121-165.
        """
        project_root = str(tmp_path / 'proj')
        db_path = (
            tmp_path / 'proj' / 'data' / 'orchestrator' / 'scheduler_overrides.db'
        )
        wal_path = Path(str(db_path) + '-wal')

        # Create schema via the helper so the DB file and WAL header exist.
        init_db = await _open_overrides_db(project_root)
        await init_db.close()

        # Long-lived writer: keeps the connection open across commits so the WAL
        # accumulates without the PASSIVE close-checkpoint truncating it.
        writer = sqlite3.connect(str(db_path))
        try:
            writer.execute('PRAGMA journal_mode=WAL')
            for i in range(5):
                writer.execute(
                    'INSERT OR REPLACE INTO overrides '
                    '(project_root, task_id, created_at, updated_at) '
                    'VALUES (?, ?, ?, ?)',
                    (project_root, f'task-wal-{i}', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'),
                )
            writer.commit()

            # Check WAL while the connection is still open — no close-checkpoint yet.
            before_size = wal_path.stat().st_size if wal_path.exists() else 0
            assert before_size > 0, (
                'Expected WAL frames to exist while writer connection is still open'
            )
        finally:
            writer.close()

        # After writer closes, a PASSIVE checkpoint fires (moves frames to DB but
        # does not necessarily truncate the physical WAL file).  Regardless,
        # _checkpoint_overrides_db must produce a TRUNCATE result with busy==0
        # and leave the WAL file at ≤32 bytes.
        result = await _checkpoint_overrides_db(project_root)
        assert result.busy == 0
        after_size = wal_path.stat().st_size if wal_path.exists() else 0
        assert after_size <= 32, (
            f'WAL file should be ≤32 bytes after TRUNCATE checkpoint; got {after_size}'
        )

    async def test_is_idempotent(
        self,
        tmp_path: Path,
    ) -> None:
        """Two back-to-back checkpoint calls both succeed and return CheckpointResult."""
        project_root = str(tmp_path / 'proj')
        result1 = await _checkpoint_overrides_db(project_root)
        result2 = await _checkpoint_overrides_db(project_root)
        assert isinstance(result1, CheckpointResult)
        assert isinstance(result2, CheckpointResult)
        assert result1.busy == 0
        assert result2.busy == 0
