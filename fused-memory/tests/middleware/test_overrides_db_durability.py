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

from fused_memory.server.tools import _open_overrides_db


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
