"""Tests for write_journal data queries (memory graphs)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

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
