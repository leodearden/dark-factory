"""Tests for CostStore persistent connection + async context manager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from shared.cost_store import CostStore


@pytest.mark.asyncio
class TestCostStoreInit:
    """step-1: CostStore.__init__ stores db_path and sets _conn to None."""

    async def test_init_stores_db_path(self, tmp_path: Path):
        db_path = tmp_path / 'costs.db'
        store = CostStore(db_path)
        assert store.db_path == db_path

    async def test_init_conn_is_none(self, tmp_path: Path):
        db_path = tmp_path / 'costs.db'
        store = CostStore(db_path)
        assert store._conn is None


@pytest.mark.asyncio
class TestCostStoreOpenClose:
    """step-3: open() creates persistent WAL connection; close() is idempotent."""

    async def test_open_sets_conn(self, tmp_path: Path):
        store = CostStore(tmp_path / 'costs.db')
        await store.open()
        try:
            assert store._conn is not None
        finally:
            await store.close()

    async def test_open_sets_wal_mode(self, tmp_path: Path):
        store = CostStore(tmp_path / 'costs.db')
        await store.open()
        try:
            async with store._conn.execute('PRAGMA journal_mode') as cur:
                row = await cur.fetchone()
            assert row[0] == 'wal'
        finally:
            await store.close()

    async def test_open_sets_busy_timeout(self, tmp_path: Path):
        store = CostStore(tmp_path / 'costs.db')
        await store.open()
        try:
            async with store._conn.execute('PRAGMA busy_timeout') as cur:
                row = await cur.fetchone()
            assert row[0] == 30000
        finally:
            await store.close()

    async def test_close_sets_conn_to_none(self, tmp_path: Path):
        store = CostStore(tmp_path / 'costs.db')
        await store.open()
        await store.close()
        assert store._conn is None

    async def test_close_is_idempotent(self, tmp_path: Path):
        """Double-close should not raise."""
        store = CostStore(tmp_path / 'costs.db')
        await store.open()
        await store.close()
        await store.close()  # should not raise
        assert store._conn is None

    async def test_open_creates_parent_dirs(self, tmp_path: Path):
        nested = tmp_path / 'a' / 'b' / 'c' / 'costs.db'
        store = CostStore(nested)
        await store.open()
        try:
            assert nested.exists()
        finally:
            await store.close()
