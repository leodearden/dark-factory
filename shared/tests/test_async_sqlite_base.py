"""Tests for AsyncSqliteBase base class and apply_wal_pragmas utility."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiosqlite
import pytest

# ---------------------------------------------------------------------------
# Step-1: apply_wal_pragmas
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestApplyWalPragmas:
    """apply_wal_pragmas(conn, busy_timeout_ms) sets WAL mode and busy_timeout."""

    async def test_sets_journal_mode_wal(self, tmp_path: Path):
        """After apply_wal_pragmas, PRAGMA journal_mode returns 'wal'."""
        from shared.async_sqlite_base import apply_wal_pragmas

        db_path = tmp_path / 'test.db'
        async with aiosqlite.connect(str(db_path)) as conn:
            await apply_wal_pragmas(conn, busy_timeout_ms=5000)
            async with conn.execute('PRAGMA journal_mode') as cur:
                row = await cur.fetchone()
        assert row[0] == 'wal'

    async def test_sets_busy_timeout(self, tmp_path: Path):
        """After apply_wal_pragmas, PRAGMA busy_timeout returns the configured value."""
        from shared.async_sqlite_base import apply_wal_pragmas

        db_path = tmp_path / 'test.db'
        async with aiosqlite.connect(str(db_path)) as conn:
            await apply_wal_pragmas(conn, busy_timeout_ms=12345)
            async with conn.execute('PRAGMA busy_timeout') as cur:
                row = await cur.fetchone()
        assert row[0] == 12345

    async def test_zero_busy_timeout_skips_pragma(self, tmp_path: Path):
        """busy_timeout_ms=0 means skip the PRAGMA busy_timeout entirely (not set to 0)."""
        from shared.async_sqlite_base import apply_wal_pragmas

        db_path = tmp_path / 'test.db'
        async with aiosqlite.connect(str(db_path)) as conn:
            # Set a non-zero value first so we can confirm it was NOT changed
            await conn.execute('PRAGMA busy_timeout=9999')
            await apply_wal_pragmas(conn, busy_timeout_ms=0)
            async with conn.execute('PRAGMA busy_timeout') as cur:
                row = await cur.fetchone()
        # busy_timeout=0 → skip pragma → previous value 9999 should be unchanged
        assert row[0] == 9999

    async def test_default_busy_timeout_is_set(self, tmp_path: Path):
        """apply_wal_pragmas with busy_timeout_ms=5000 sets the timeout."""
        from shared.async_sqlite_base import apply_wal_pragmas

        db_path = tmp_path / 'test.db'
        async with aiosqlite.connect(str(db_path)) as conn:
            await apply_wal_pragmas(conn, busy_timeout_ms=5000)
            async with conn.execute('PRAGMA busy_timeout') as cur:
                row = await cur.fetchone()
        assert row[0] == 5000
