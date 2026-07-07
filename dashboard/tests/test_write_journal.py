"""Tests for write_journal data queries (memory graphs)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import aiosqlite
import pytest

WRITE_OPS_SCHEMA = """
CREATE TABLE IF NOT EXISTS write_ops (
    id TEXT PRIMARY KEY,
    causation_id TEXT,
    source TEXT,
    provenance TEXT DEFAULT 'original',
    operation TEXT,
    project_id TEXT,
    agent_id TEXT,
    params TEXT DEFAULT '{}',
    result_summary TEXT,
    success INTEGER DEFAULT 1,
    error TEXT,
    created_at TEXT NOT NULL,
    session_id TEXT,
    kind TEXT NOT NULL DEFAULT 'write'
);
"""


@pytest.fixture()
def journal_db(tmp_path):
    """Create a write_journal DB with sample data spanning several hours."""
    db_path = tmp_path / 'write_journal.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(WRITE_OPS_SCHEMA)

    now = datetime.now(UTC)
    rows = [
        # Recent reads
        ('op-1', 'search', 'dark_factory', 'claude-interactive', 'read',
         (now - timedelta(hours=1)).isoformat()),
        ('op-2', 'search', 'dark_factory', 'claude-interactive', 'read',
         (now - timedelta(hours=1, minutes=30)).isoformat()),
        ('op-3', 'get_entity', 'dark_factory', 'claude-interactive', 'read',
         (now - timedelta(hours=2)).isoformat()),
        # Recent writes
        ('op-4', 'add_memory', 'dark_factory', 'claude-interactive', 'write',
         (now - timedelta(hours=1)).isoformat()),
        ('op-5', 'add_memory', 'dark_factory', 'recon-stage-consolidator', 'write',
         (now - timedelta(hours=3)).isoformat()),
        ('op-6', 'delete_memory', 'dark_factory', 'recon-stage-consolidator', 'write',
         (now - timedelta(hours=3)).isoformat()),
        # Old data (>24h) — should be excluded
        ('op-7', 'search', 'dark_factory', 'claude-interactive', 'read',
         (now - timedelta(hours=25)).isoformat()),
    ]
    for op_id, operation, project_id, agent_id, kind, created_at in rows:
        conn.execute(
            'INSERT INTO write_ops (id, operation, project_id, agent_id, kind, created_at)'
            ' VALUES (?, ?, ?, ?, ?, ?)',
            (op_id, operation, project_id, agent_id, kind, created_at),
        )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def empty_journal_db(tmp_path):
    """Write journal DB with schema but no data."""
    db_path = tmp_path / 'write_journal.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(WRITE_OPS_SCHEMA)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
async def journal_conn(journal_db):
    async with aiosqlite.connect(str(journal_db)) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


@pytest.fixture()
async def empty_journal_conn(empty_journal_db):
    async with aiosqlite.connect(str(empty_journal_db)) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


class TestGetMemoryTimeseries:
    @pytest.mark.asyncio
    async def test_returns_24_buckets(self, journal_conn):
        from dashboard.data.write_journal import get_memory_timeseries

        result = await get_memory_timeseries(journal_conn)
        assert len(result['labels']) == 24
        assert len(result['reads']) == 24
        assert len(result['writes']) == 24

    @pytest.mark.asyncio
    async def test_labels_are_hhmm_format(self, journal_conn):
        from dashboard.data.write_journal import get_memory_timeseries

        result = await get_memory_timeseries(journal_conn)
        for label in result['labels']:
            assert len(label) == 5
            assert label[2] == ':'

    @pytest.mark.asyncio
    async def test_excludes_old_data(self, journal_conn):
        from dashboard.data.write_journal import get_memory_timeseries

        result = await get_memory_timeseries(journal_conn)
        # op-7 is >24h old — total reads should be 3, not 4
        assert sum(result['reads']) == 3

    @pytest.mark.asyncio
    async def test_counts_reads_and_writes(self, journal_conn):
        from dashboard.data.write_journal import get_memory_timeseries

        result = await get_memory_timeseries(journal_conn)
        assert sum(result['reads']) == 3
        assert sum(result['writes']) == 3

    @pytest.mark.asyncio
    async def test_empty_db_returns_zeros(self, empty_journal_conn):
        from dashboard.data.write_journal import get_memory_timeseries

        result = await get_memory_timeseries(empty_journal_conn)
        assert sum(result['reads']) == 0
        assert sum(result['writes']) == 0
        assert len(result['labels']) == 24

    @pytest.mark.asyncio
    async def test_missing_db_returns_zeros(self):
        from dashboard.data.write_journal import get_memory_timeseries

        result = await get_memory_timeseries(None)
        assert len(result['labels']) == 24
        assert sum(result['reads']) == 0
        assert sum(result['writes']) == 0


class TestGetOperationsBreakdown:
    @pytest.mark.asyncio
    async def test_returns_all_operations(self, journal_conn):
        from dashboard.data.write_journal import get_operations_breakdown

        result = await get_operations_breakdown(journal_conn)
        assert set(result['labels']) == {'search', 'get_entity', 'add_memory', 'delete_memory'}

    @pytest.mark.asyncio
    async def test_sorted_by_count_desc(self, journal_conn):
        from dashboard.data.write_journal import get_operations_breakdown

        result = await get_operations_breakdown(journal_conn)
        assert result['values'] == sorted(result['values'], reverse=True)

    @pytest.mark.asyncio
    async def test_excludes_old_data(self, journal_conn):
        from dashboard.data.write_journal import get_operations_breakdown

        result = await get_operations_breakdown(journal_conn)
        assert sum(result['values']) == 6  # not 7

    @pytest.mark.asyncio
    async def test_empty_db(self, empty_journal_conn):
        from dashboard.data.write_journal import get_operations_breakdown

        result = await get_operations_breakdown(empty_journal_conn)
        assert result == {'labels': [], 'values': []}

    @pytest.mark.asyncio
    async def test_missing_db(self):
        from dashboard.data.write_journal import get_operations_breakdown

        result = await get_operations_breakdown(None)
        assert result == {'labels': [], 'values': []}


class TestGetAgentBreakdown:
    @pytest.mark.asyncio
    async def test_returns_all_agents(self, journal_conn):
        from dashboard.data.write_journal import get_agent_breakdown

        result = await get_agent_breakdown(journal_conn)
        assert set(result['labels']) == {'claude-interactive', 'recon-stage-consolidator'}

    @pytest.mark.asyncio
    async def test_sorted_by_count_desc(self, journal_conn):
        from dashboard.data.write_journal import get_agent_breakdown

        result = await get_agent_breakdown(journal_conn)
        assert result['values'] == sorted(result['values'], reverse=True)

    @pytest.mark.asyncio
    async def test_excludes_old_data(self, journal_conn):
        from dashboard.data.write_journal import get_agent_breakdown

        result = await get_agent_breakdown(journal_conn)
        assert sum(result['values']) == 6

    @pytest.mark.asyncio
    async def test_empty_db(self, empty_journal_conn):
        from dashboard.data.write_journal import get_agent_breakdown

        result = await get_agent_breakdown(empty_journal_conn)
        assert result == {'labels': [], 'values': []}

    @pytest.mark.asyncio
    async def test_missing_db(self):
        from dashboard.data.write_journal import get_agent_breakdown

        result = await get_agent_breakdown(None)
        assert result == {'labels': [], 'values': []}


@pytest.fixture()
async def no_table_conn(tmp_path):
    """Connection to an empty DB with no write_ops table."""
    db_path = tmp_path / 'empty_notables.db'
    sqlite3.connect(str(db_path)).close()  # empty, no tables
    async with aiosqlite.connect(str(db_path)) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


class TestDataLayerErrorHandling:
    """Verify all 3 write_journal functions handle errors at data layer."""

    @pytest.mark.asyncio
    async def test_timeseries_returns_default_on_none_db(self):
        from dashboard.data.write_journal import get_memory_timeseries
        result = await get_memory_timeseries(None)
        assert len(result['labels']) == 24
        assert sum(result['reads']) == 0
        assert sum(result['writes']) == 0

    @pytest.mark.asyncio
    async def test_operations_returns_default_on_none_db(self):
        from dashboard.data.write_journal import get_operations_breakdown
        result = await get_operations_breakdown(None)
        assert result == {'labels': [], 'values': []}

    @pytest.mark.asyncio
    async def test_agents_returns_default_on_none_db(self):
        from dashboard.data.write_journal import get_agent_breakdown
        result = await get_agent_breakdown(None)
        assert result == {'labels': [], 'values': []}

    @pytest.mark.asyncio
    async def test_timeseries_returns_default_on_operational_error(self, no_table_conn):
        from dashboard.data.write_journal import get_memory_timeseries
        result = await get_memory_timeseries(no_table_conn)
        assert len(result['labels']) == 24
        assert sum(result['reads']) == 0
        assert sum(result['writes']) == 0

    @pytest.mark.asyncio
    async def test_operations_returns_default_on_operational_error(self, no_table_conn):
        from dashboard.data.write_journal import get_operations_breakdown
        result = await get_operations_breakdown(no_table_conn)
        assert result == {'labels': [], 'values': []}

    @pytest.mark.asyncio
    async def test_agents_returns_default_on_operational_error(self, no_table_conn):
        from dashboard.data.write_journal import get_agent_breakdown
        result = await get_agent_breakdown(no_table_conn)
        assert result == {'labels': [], 'values': []}

    @pytest.mark.asyncio
    async def test_timeseries_returns_default_on_os_error(self, tmp_path):
        from dashboard.data.write_journal import get_memory_timeseries
        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()
        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            mock_cursor = AsyncMock()
            mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
            mock_cursor.__aexit__ = AsyncMock(return_value=False)
            mock_cursor.fetchall = AsyncMock(side_effect=OSError('disk I/O error'))
            with patch.object(conn, 'execute', return_value=mock_cursor):
                result = await get_memory_timeseries(conn)
        assert len(result['labels']) == 24
        assert sum(result['reads']) == 0
        assert sum(result['writes']) == 0

    @pytest.mark.asyncio
    async def test_operations_returns_default_on_os_error(self, tmp_path):
        from dashboard.data.write_journal import get_operations_breakdown
        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()
        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            mock_cursor = AsyncMock()
            mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
            mock_cursor.__aexit__ = AsyncMock(return_value=False)
            mock_cursor.fetchall = AsyncMock(side_effect=OSError('disk I/O error'))
            with patch.object(conn, 'execute', return_value=mock_cursor):
                result = await get_operations_breakdown(conn)
        assert result == {'labels': [], 'values': []}

    @pytest.mark.asyncio
    async def test_agents_returns_default_on_os_error(self, tmp_path):
        from dashboard.data.write_journal import get_agent_breakdown
        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()
        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            mock_cursor = AsyncMock()
            mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
            mock_cursor.__aexit__ = AsyncMock(return_value=False)
            mock_cursor.fetchall = AsyncMock(side_effect=OSError('disk I/O error'))
            with patch.object(conn, 'execute', return_value=mock_cursor):
                result = await get_agent_breakdown(conn)
        assert result == {'labels': [], 'values': []}


def _seed_write_ops(db_path, rows):
    """Create a write_ops SQLite DB at *db_path* seeded with *rows*.

    Each row is ``(op_id, operation, project_id, agent_id, kind, created_at)``,
    matching the column order used throughout this module's fixtures.
    """
    conn = sqlite3.connect(str(db_path))
    conn.executescript(WRITE_OPS_SCHEMA)
    for op_id, operation, project_id, agent_id, kind, created_at in rows:
        conn.execute(
            'INSERT INTO write_ops (id, operation, project_id, agent_id, kind, created_at)'
            ' VALUES (?, ?, ?, ?, ?, ?)',
            (op_id, operation, project_id, agent_id, kind, created_at),
        )
    conn.commit()
    conn.close()


class TestNowThreading:
    """now-threading: each function accepts now=fixed and derives its cutoff from it.

    Mirrors ``Test_Cutoff`` in test_costs_data.py: a fixed-now determinism test
    per function plus one no-now bracket test, rather than relying on the
    live clock for every assertion.
    """

    FIXED_NOW = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_memory_timeseries_uses_provided_now(self, tmp_path):
        """get_memory_timeseries(now=fixed) buckets rows against fixed, not the live clock.

        One row 1h before FIXED_NOW (well inside the 24h window relative to
        FIXED_NOW) and one row 25h before FIXED_NOW (outside it). FIXED_NOW is
        an arbitrary historical instant unrelated to the real current time, so
        this only passes if the function actually threads `now` through.
        """
        from dashboard.data.write_journal import get_memory_timeseries

        db_path = tmp_path / 'timeseries_fixed_now.db'
        inside = self.FIXED_NOW - timedelta(hours=1)
        outside = self.FIXED_NOW - timedelta(hours=25)
        _seed_write_ops(db_path, [
            ('in-1', 'search', 'dark_factory', 'agent-a', 'read', inside.isoformat()),
            ('out-1', 'search', 'dark_factory', 'agent-a', 'read', outside.isoformat()),
        ])

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await get_memory_timeseries(conn, now=self.FIXED_NOW)
        assert sum(result['reads']) == 1

    @pytest.mark.asyncio
    async def test_memory_timeseries_no_now_resolves_via_clock(self, tmp_path):
        """Without now, get_memory_timeseries still buckets against the live clock.

        Same shape as the fixed-now test above, but rows are anchored to the
        real ``datetime.now(UTC)`` and no ``now`` kwarg is passed — regression
        coverage for the default (no-now) path once resolve_now is threaded in.
        """
        from dashboard.data.write_journal import get_memory_timeseries

        db_path = tmp_path / 'timeseries_live_clock.db'
        real_now = datetime.now(UTC)
        inside = real_now - timedelta(hours=1)
        outside = real_now - timedelta(hours=25)
        _seed_write_ops(db_path, [
            ('in-1', 'search', 'dark_factory', 'agent-a', 'read', inside.isoformat()),
            ('out-1', 'search', 'dark_factory', 'agent-a', 'read', outside.isoformat()),
        ])

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await get_memory_timeseries(conn)
        assert sum(result['reads']) == 1

    @pytest.mark.asyncio
    async def test_operations_breakdown_uses_provided_now_at_minute_boundary(self, tmp_path):
        """get_operations_breakdown(now=fixed): fixed-24h+1min is counted, fixed-24h-1min is not.

        Unlike get_memory_timeseries, this function's cutoff is a plain
        ``created_at >= since`` comparison (no hour-bucketing), so a tight
        1-minute boundary deterministically separates included/excluded rows.
        """
        from dashboard.data.write_journal import get_operations_breakdown

        db_path = tmp_path / 'ops_fixed_now_boundary.db'
        cutoff = self.FIXED_NOW - timedelta(hours=24)
        just_inside = cutoff + timedelta(minutes=1)
        just_outside = cutoff - timedelta(minutes=1)
        _seed_write_ops(db_path, [
            ('in-1', 'search', 'dark_factory', 'agent-a', 'read', just_inside.isoformat()),
            ('out-1', 'search', 'dark_factory', 'agent-a', 'read', just_outside.isoformat()),
        ])

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await get_operations_breakdown(conn, now=self.FIXED_NOW)
        assert sum(result['values']) == 1

    @pytest.mark.asyncio
    async def test_agent_breakdown_uses_provided_now_at_minute_boundary(self, tmp_path):
        """get_agent_breakdown(now=fixed): fixed-24h+1min is counted, fixed-24h-1min is not."""
        from dashboard.data.write_journal import get_agent_breakdown

        db_path = tmp_path / 'agents_fixed_now_boundary.db'
        cutoff = self.FIXED_NOW - timedelta(hours=24)
        just_inside = cutoff + timedelta(minutes=1)
        just_outside = cutoff - timedelta(minutes=1)
        _seed_write_ops(db_path, [
            ('in-1', 'search', 'dark_factory', 'agent-a', 'read', just_inside.isoformat()),
            ('out-1', 'search', 'dark_factory', 'agent-a', 'read', just_outside.isoformat()),
        ])

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await get_agent_breakdown(conn, now=self.FIXED_NOW)
        assert sum(result['values']) == 1
