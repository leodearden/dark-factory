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
-- Mirrors the real six-index set fused-memory creates on write_ops, so
-- query-plan assertions against this fixture (see
-- TestOperationsBreakdownQueryPlan) exercise the same planner choices as the
-- live journal. Copied from the `SCHEMA_SQL` constant in
-- fused_memory/services/write_journal.py (cited by FILE + CONSTANT NAME, not
-- line number: fused-memory/tests/test_write_journal.py:718-724 records that a
-- line-number cross-package citation already went stale once).
CREATE INDEX IF NOT EXISTS idx_wo_causation ON write_ops(causation_id);
CREATE INDEX IF NOT EXISTS idx_wo_project_time ON write_ops(project_id, created_at);
CREATE INDEX IF NOT EXISTS idx_wo_operation ON write_ops(operation);
CREATE INDEX IF NOT EXISTS idx_wo_kind_time ON write_ops(kind, created_at);
CREATE INDEX IF NOT EXISTS idx_wo_agent_time ON write_ops(agent_id, created_at);
CREATE INDEX IF NOT EXISTS idx_wo_created ON write_ops(created_at);
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


def _seed_ops_breakdown_plan_fixture(db_path):
    """Seed 500 rows / 5 operations spread over ~20 days for plan assertions.

    Matches the architect's validated fixture shape exactly (500 rows, 5
    distinct operations, sqlite 3.50.4, real six-index schema): reproducing
    it here is what makes the acceptance strings in
    TestOperationsBreakdownQueryPlan reliable rather than guessed. Returns
    the ``since`` ISO timestamp to use (the fixture's start instant), which
    is what makes the plan meaningful — EXPLAIN QUERY PLAN reasons about the
    predicate shape, not matched-row counts.
    """
    base = datetime(2026, 4, 1, tzinfo=UTC)
    rows = [
        (
            f'op-{i}',
            f'operation-{i % 5}',
            'dark_factory',
            'agent-a',
            'read' if i % 2 == 0 else 'write',
            (base + timedelta(minutes=i * 57.6)).isoformat(),
        )
        for i in range(500)
    ]
    _seed_write_ops(db_path, rows)
    return base.isoformat()


class TestOperationsBreakdownQueryPlan:
    """Pins get_operations_breakdown's query plan to a range-seek on idx_wo_created.

    Reproduces the architect's validated finding (500-row / 5-operation
    fixture, sqlite 3.50.4, real six-index schema, task 3519): unhinted,
    ``GROUP BY operation`` is satisfied in-order by walking
    ``idx_wo_operation`` (a full ``SCAN``); an explicit
    ``INDEXED BY idx_wo_created`` flips the plan to a range ``SEARCH`` on
    ``created_at``. Both plans are stable across ``ANALYZE``, so this is
    schema-shape-determined, not statistics-dependent, and needs no live DB.

    Imports ``OPS_BREAKDOWN_SQL`` from the real module rather than copying the
    SQL into the test — closing the drift gap
    fused-memory/tests/test_write_journal.py:726-732 filed as follow-up (its
    own plan tests hold verbatim SQL copies that cannot detect the dashboard
    changing its query).
    """

    @pytest.mark.asyncio
    async def test_plan_is_search_on_idx_wo_created(self, tmp_path):
        from dashboard.data.write_journal import OPS_BREAKDOWN_SQL

        db_path = tmp_path / 'ops_plan.db'
        since = _seed_ops_breakdown_plan_fixture(db_path)

        async with (
            aiosqlite.connect(str(db_path)) as conn,
            conn.execute(f'EXPLAIN QUERY PLAN {OPS_BREAKDOWN_SQL}', (since,)) as cursor,
        ):
            plan = ' '.join(row[3] for row in await cursor.fetchall())

        assert 'SEARCH' in plan, f'expected a range seek, got: {plan}'
        assert 'idx_wo_created' in plan, f'expected idx_wo_created in the plan, got: {plan}'
        assert 'created_at>?' in plan, (
            f'created_at must be the seek constraint, got: {plan}'
        )
        assert 'SCAN' not in plan, f'still full-scanning write_ops: {plan}'
