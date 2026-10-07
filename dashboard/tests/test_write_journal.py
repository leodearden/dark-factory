"""Tests for write_journal data queries (memory graphs)."""

from __future__ import annotations

import logging
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
-- TestMemoryOpsQueryPlan) exercise the same planner choices as the
-- live journal. Two sources, both in fused_memory/services/write_journal.py
-- (cited by FILE + CONSTANT/METHOD NAME, not line number:
-- fused-memory/tests/test_write_journal.py:718-724 records that a
-- line-number cross-package citation already went stale once):
-- idx_wo_causation, idx_wo_project_time, idx_wo_operation and idx_wo_created
-- come from the `SCHEMA_SQL` constant; idx_wo_kind_time and idx_wo_agent_time
-- are NOT in SCHEMA_SQL — they're created separately inside
-- `WriteJournal._migrate()` (they depend on the `kind` column that migration
-- adds). Keep both sources in sync when mirroring future index changes here.
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


_OPS_NOW = datetime(2026, 10, 3, 12, 30, tzinfo=UTC)
_OPS_FLOOR = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)
_OPS_WINDOW_START = _OPS_FLOOR - timedelta(hours=23)

# The 24 'HH:00' keys floor(now)-23h .. floor(now), oldest first: the same
# keys the Memory chart has always shown for a 24h window.
_OPS_LABELS = tuple(
    (_OPS_WINDOW_START + timedelta(hours=i)).strftime('%H:00') for i in range(24)
)


def _ops_bucket(at):
    """Index of *at*'s hour among _OPS_LABELS (the window is hour-aligned)."""
    return int((at - _OPS_WINDOW_START).total_seconds() // 3600)


_OPS_ROWS = [
    ('r-1', 'search', 'dark_factory', 'agent-a', 'read',
     datetime(2026, 10, 3, 11, 10, tzinfo=UTC)),
    ('r-2', 'search', 'dark_factory', 'agent-a', 'read',
     datetime(2026, 10, 3, 11, 50, tzinfo=UTC)),
    ('r-3', 'get_entity', 'dark_factory', 'agent-a', 'read',
     datetime(2026, 10, 3, 12, 20, tzinfo=UTC)),
    ('w-1', 'add_memory', 'dark_factory', 'agent-b', 'write',
     datetime(2026, 10, 3, 9, 5, tzinfo=UTC)),
    ('w-2', 'add_memory', 'dark_factory', 'agent-b', 'write',
     datetime(2026, 10, 3, 9, 6, tzinfo=UTC)),
    ('w-3', 'delete_memory', 'dark_factory', 'agent-b', 'write',
     datetime(2026, 10, 2, 20, 0, tzinfo=UTC)),
    # A kind outside read/write: the old timeseries dropped it while the
    # breakdown counted it.
    ('m-1', 'compact', 'dark_factory', 'agent-c', 'maintenance',
     datetime(2026, 10, 3, 6, 15, tzinfo=UTC)),
    ('n-1', None, 'dark_factory', 'agent-b', 'write',
     datetime(2026, 10, 3, 7, 0, tzinfo=UTC)),
    # Exactly the window start: the first bucket.
    ('start', 'search', 'dark_factory', 'agent-a', 'read', _OPS_WINDOW_START),
    # One minute before the window start: the old timeseries dropped this
    # partial hour while the breakdown counted it. Now in no view.
    ('before', 'search', 'dark_factory', 'agent-a', 'read',
     _OPS_WINDOW_START - timedelta(minutes=1)),
    # floor(now)+1h: future-dated, in no view.
    ('future', 'add_memory', 'dark_factory', 'agent-b', 'write',
     _OPS_FLOOR + timedelta(hours=1)),
]


@pytest.fixture()
async def ops_conn(tmp_path):
    db_path = tmp_path / 'memory_ops.db'
    _seed_write_ops(db_path, [
        (op_id, operation, project, agent, kind, at.isoformat())
        for op_id, operation, project, agent, kind, at in _OPS_ROWS
    ])
    async with aiosqlite.connect(str(db_path)) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


async def _ops_value(conn, **kwargs):
    """The MemoryOps a successful get_memory_ops read measured."""
    from dashboard.data.write_journal import get_memory_ops

    result = await get_memory_ops(conn, **kwargs)
    assert result.value is not None, f'expected a measured window, got: {result}'
    return result.value


def _empty_ops():
    from dashboard.data.write_journal import MemoryOps

    return MemoryOps(
        labels=_OPS_LABELS,
        reads=(0,) * 24,
        writes=(0,) * 24,
        other=(0,) * 24,
        by_operation=(),
    )


class TestGetMemoryOps:
    """One query, one reduction: every counted row lands once in the hourly
    series and once in by_operation (PRD sketch #11)."""

    @pytest.mark.asyncio
    async def test_series_and_breakdown_reconcile(self, ops_conn):
        ops = await _ops_value(ops_conn, hours=24, now=_OPS_NOW)

        series_total = sum(ops.reads) + sum(ops.writes) + sum(ops.other)
        assert series_total == sum(count for _, count in ops.by_operation)
        assert series_total == 9

    @pytest.mark.asyncio
    async def test_kind_outside_read_write_counts_as_other(self, ops_conn):
        ops = await _ops_value(ops_conn, hours=24, now=_OPS_NOW)

        maintenance_at = datetime(2026, 10, 3, 6, 15, tzinfo=UTC)
        assert ops.other[_ops_bucket(maintenance_at)] == 1
        assert sum(ops.other) == 1
        assert dict(ops.by_operation)['compact'] == 1

    @pytest.mark.asyncio
    async def test_labels_are_the_24_hour_keys_oldest_first(self, ops_conn):
        ops = await _ops_value(ops_conn, hours=24, now=_OPS_NOW)

        assert ops.labels == _OPS_LABELS
        assert ops.labels[0] == '13:00'
        assert ops.labels[-1] == '12:00'
        assert len(ops.reads) == len(ops.writes) == len(ops.other) == 24

    @pytest.mark.asyncio
    async def test_window_is_hour_aligned_and_bounded_above(self, ops_conn):
        ops = await _ops_value(ops_conn, hours=24, now=_OPS_NOW)

        # The window-start row is the first bucket's only read; the row a
        # minute before it and the future row are in no view.
        assert ops.reads[0] == 1
        assert sum(ops.reads) == 4
        assert sum(ops.writes) == 4
        assert dict(ops.by_operation)['search'] == 3
        assert dict(ops.by_operation)['add_memory'] == 2

    @pytest.mark.asyncio
    async def test_hourly_buckets_hold_their_rows(self, ops_conn):
        ops = await _ops_value(ops_conn, hours=24, now=_OPS_NOW)

        assert ops.reads[_ops_bucket(datetime(2026, 10, 3, 11, tzinfo=UTC))] == 2
        assert ops.reads[-1] == 1
        assert ops.writes[_ops_bucket(datetime(2026, 10, 3, 9, tzinfo=UTC))] == 2
        assert ops.writes[_ops_bucket(datetime(2026, 10, 2, 20, tzinfo=UTC))] == 1

    @pytest.mark.asyncio
    async def test_by_operation_count_desc_ties_by_label_and_null_is_unknown(
        self, ops_conn,
    ):
        ops = await _ops_value(ops_conn, hours=24, now=_OPS_NOW)

        assert ops.by_operation == (
            ('search', 3),
            ('add_memory', 2),
            ('compact', 1),
            ('delete_memory', 1),
            ('get_entity', 1),
            ('unknown', 1),
        )

    @pytest.mark.asyncio
    async def test_a_readable_empty_journal_is_a_fresh_quiet_window(
        self, empty_journal_conn,
    ):
        """A journal that answers with no rows is a measured quiet day, kept
        distinct from a dead journal (TestDataLayerErrorHandling)."""
        from dashboard.data import write_journal
        from dashboard.data.datum import DatumState, validate_datum

        result = await write_journal.get_memory_ops(
            empty_journal_conn, hours=24, now=_OPS_NOW,
        )

        assert result.state is DatumState.FRESH
        assert result.as_of == _OPS_NOW
        assert result.reason is None
        assert (
            result.freshness_bound_seconds
            == write_journal.MEMORY_OPS_FRESHNESS_BOUND_SECONDS
        )
        assert result.value == _empty_ops()
        validate_datum(result, served_at=_OPS_NOW)

    @pytest.mark.asyncio
    async def test_a_populated_read_is_fresh_as_of_the_passed_now(self, ops_conn):
        from dashboard.data import write_journal
        from dashboard.data.datum import DatumState, validate_datum

        result = await write_journal.get_memory_ops(ops_conn, hours=24, now=_OPS_NOW)

        assert result.state is DatumState.FRESH
        assert result.as_of == _OPS_NOW
        assert result.reason is None
        assert (
            result.freshness_bound_seconds
            == write_journal.MEMORY_OPS_FRESHNESS_BOUND_SECONDS
        )
        assert result.value is not None
        assert sum(result.value.reads) == 4
        validate_datum(result, served_at=_OPS_NOW)


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


def _assert_unknown_memory_ops(result, *reason_fragments):
    """*result* is an unmeasured window: an unknown Datum naming why, never a
    zero window that would read as a quiet day."""
    from dashboard.data import write_journal
    from dashboard.data.datum import DatumState, validate_datum

    assert result.state is DatumState.UNKNOWN
    assert result.value is None
    assert result.as_of is None
    assert result.reason and result.reason.strip()
    for fragment in reason_fragments:
        assert fragment in result.reason, (fragment, result.reason)
    assert (
        result.freshness_bound_seconds
        == write_journal.MEMORY_OPS_FRESHNESS_BOUND_SECONDS
    )
    validate_datum(result, served_at=_OPS_NOW)


class TestDataLayerErrorHandling:
    """Verify both write_journal functions handle errors at data layer."""

    @pytest.mark.asyncio
    async def test_memory_ops_is_unknown_on_none_db(self):
        from dashboard.data.write_journal import get_memory_ops
        result = await get_memory_ops(None, now=_OPS_NOW)
        _assert_unknown_memory_ops(result, 'write journal')

    @pytest.mark.asyncio
    async def test_agents_returns_default_on_none_db(self):
        from dashboard.data.write_journal import get_agent_breakdown
        result = await get_agent_breakdown(None)
        assert result == {'labels': [], 'values': []}

    @pytest.mark.asyncio
    async def test_memory_ops_is_unknown_naming_the_operational_error(
        self, no_table_conn,
    ):
        from dashboard.data.write_journal import get_memory_ops
        result = await get_memory_ops(no_table_conn, now=_OPS_NOW)
        _assert_unknown_memory_ops(result, 'OperationalError', 'no such table')

    @pytest.mark.asyncio
    async def test_agents_returns_default_on_operational_error(self, no_table_conn):
        from dashboard.data.write_journal import get_agent_breakdown
        result = await get_agent_breakdown(no_table_conn)
        assert result == {'labels': [], 'values': []}

    @pytest.mark.asyncio
    async def test_memory_ops_is_unknown_naming_the_os_error(self, tmp_path):
        from dashboard.data.write_journal import get_memory_ops
        db_path = tmp_path / 'test.db'
        sqlite3.connect(str(db_path)).close()
        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            mock_cursor = AsyncMock()
            mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
            mock_cursor.__aexit__ = AsyncMock(return_value=False)
            mock_cursor.fetchall = AsyncMock(side_effect=OSError('disk I/O error'))
            with patch.object(conn, 'execute', return_value=mock_cursor):
                result = await get_memory_ops(conn, now=_OPS_NOW)
        _assert_unknown_memory_ops(result, 'OSError', 'disk I/O error')

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
    async def test_memory_ops_uses_provided_now(self, tmp_path):
        """get_memory_ops(now=fixed) windows rows against fixed, not the live clock.

        One row 1h before FIXED_NOW (inside its window) and one 25h before it
        (outside). FIXED_NOW is an arbitrary historical instant unrelated to
        the real current time, so this only passes if `now` is threaded through.
        """
        db_path = tmp_path / 'memory_ops_fixed_now.db'
        inside = self.FIXED_NOW - timedelta(hours=1)
        outside = self.FIXED_NOW - timedelta(hours=25)
        _seed_write_ops(db_path, [
            ('in-1', 'search', 'dark_factory', 'agent-a', 'read', inside.isoformat()),
            ('out-1', 'search', 'dark_factory', 'agent-a', 'read', outside.isoformat()),
        ])

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await _ops_value(conn, now=self.FIXED_NOW)
        assert sum(result.reads) == 1
        assert result.by_operation == (('search', 1),)

    @pytest.mark.asyncio
    async def test_memory_ops_no_now_resolves_via_clock(self, tmp_path):
        """Without now, get_memory_ops still windows against the live clock."""
        db_path = tmp_path / 'memory_ops_live_clock.db'
        real_now = datetime.now(UTC)
        inside = real_now - timedelta(hours=1)
        outside = real_now - timedelta(hours=25)
        _seed_write_ops(db_path, [
            ('in-1', 'search', 'dark_factory', 'agent-a', 'read', inside.isoformat()),
            ('out-1', 'search', 'dark_factory', 'agent-a', 'read', outside.isoformat()),
        ])

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await _ops_value(conn)
        assert sum(result.reads) == 1

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


# Deliberately UNEQUAL per-operation weights (sum to 500, matching the
# architect's validated row count / date spread). An even `i % 5` split ties
# every group's count at 100, which makes a count-descending assertion
# vacuously true for ANY ordering (reviewer_comprehensive finding, task 3519
# amendment). The weighted split makes that assertion real; see
# TestMemoryOpsResultEquivalence's `test_hinted_and_unhinted_agree_on_ties`
# for dedicated tied-count coverage.
_OPS_BREAKDOWN_WEIGHTS = {
    'operation-0': 200,
    'operation-1': 120,
    'operation-2': 90,
    'operation-3': 60,
    'operation-4': 30,
}

_PLAN_FIXTURE_BASE = datetime(2026, 4, 1, tzinfo=UTC)
# Rows span base .. base+479.04h; this window [base-23h, base+481h) holds all.
_PLAN_FIXTURE_NOW = _PLAN_FIXTURE_BASE + timedelta(hours=480)
_PLAN_FIXTURE_HOURS = 504


def _seed_ops_breakdown_plan_fixture(db_path):
    """Seed 500 rows / 5 operations spread over ~20 days for plan assertions.

    Matches the architect's validated fixture shape (500 rows, 5 distinct
    operations, sqlite 3.50.4, real six-index schema) for the query-plan
    acceptance strings in TestMemoryOpsQueryPlan, which are determined by
    row/operation counts and date spread, not by the per-operation split.
    Per-operation counts follow `_OPS_BREAKDOWN_WEIGHTS` rather than an even
    split — see that constant's comment. Every row falls inside
    ``get_memory_ops(hours=_PLAN_FIXTURE_HOURS, now=_PLAN_FIXTURE_NOW)``.
    """
    operations = [
        operation
        for operation, count in _OPS_BREAKDOWN_WEIGHTS.items()
        for _ in range(count)
    ]
    assert len(operations) == 500, f'weights must sum to 500 rows, got {len(operations)}'
    rows = [
        (
            f'op-{i}',
            operations[i],
            'dark_factory',
            'agent-a',
            'read' if i % 2 == 0 else 'write',
            (_PLAN_FIXTURE_BASE + timedelta(minutes=i * 57.6)).isoformat(),
        )
        for i in range(500)
    ]
    _seed_write_ops(db_path, rows)


class TestMemoryOpsQueryPlan:
    """Pins get_memory_ops' query plan to a range-seek on idx_wo_created.

    Unhinted, the old operations-breakdown query walked ``idx_wo_operation``
    in full to satisfy ``GROUP BY operation`` in order (task 3519); the
    explicit ``INDEXED BY idx_wo_created`` keeps the combined query a range
    ``SEARCH`` on ``created_at`` whatever the planner's statistics say. The
    plan is schema-shape-determined, so this needs no live DB.

    Imports ``MEMORY_OPS_SQL`` from the real module rather than copying the
    SQL into the test, so a change to the query cannot slip past this pin.
    """

    @pytest.mark.asyncio
    async def test_plan_is_search_on_idx_wo_created(self, tmp_path):
        from dashboard.data.write_journal import MEMORY_OPS_SQL

        db_path = tmp_path / 'ops_plan.db'
        _seed_ops_breakdown_plan_fixture(db_path)
        window = (
            _PLAN_FIXTURE_BASE.isoformat(),
            (_PLAN_FIXTURE_BASE + timedelta(days=21)).isoformat(),
        )

        async with (
            aiosqlite.connect(str(db_path)) as conn,
            conn.execute(f'EXPLAIN QUERY PLAN {MEMORY_OPS_SQL}', window) as cursor,
        ):
            plan = ' '.join(row[3] for row in await cursor.fetchall())

        assert 'SEARCH' in plan, f'expected a range seek, got: {plan}'
        assert 'idx_wo_created' in plan, f'expected idx_wo_created in the plan, got: {plan}'
        assert 'created_at>?' in plan, (
            f'created_at must be the seek constraint, got: {plan}'
        )
        assert 'SCAN' not in plan, f'still full-scanning write_ops: {plan}'


async def _memory_ops_with_sql(conn, sql, monkeypatch, **kwargs):
    """get_memory_ops with its query swapped for *sql* (hinted vs unhinted)."""
    from dashboard.data import write_journal

    monkeypatch.setattr(write_journal, 'MEMORY_OPS_SQL', sql)
    return await _ops_value(conn, **kwargs)


class TestMemoryOpsResultEquivalence:
    """Characterization guard: the INDEXED BY hint must not change results.

    Passes by construction today, because "the results are unchanged" IS the
    property under test; it goes red if a future edit makes the hinted and
    unhinted SQL disagree. Compares whole MemoryOps values, so the series,
    the labels and the by_operation ORDER (count desc, ties by label) all
    have to agree — including on tied counts, where SQL's own row order is
    unspecified and the two plans walk different indexes.
    """

    @pytest.mark.asyncio
    async def test_hinted_and_unhinted_agree(self, tmp_path, monkeypatch):
        from dashboard.data.write_journal import MEMORY_OPS_SQL, MEMORY_OPS_SQL_UNHINTED

        db_path = tmp_path / 'ops_equivalence.db'
        _seed_ops_breakdown_plan_fixture(db_path)
        window = {'hours': _PLAN_FIXTURE_HOURS, 'now': _PLAN_FIXTURE_NOW}

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            hinted = await _memory_ops_with_sql(conn, MEMORY_OPS_SQL, monkeypatch, **window)
            unhinted = await _memory_ops_with_sql(
                conn, MEMORY_OPS_SQL_UNHINTED, monkeypatch, **window,
            )

        assert hinted == unhinted
        assert hinted.by_operation == tuple(_OPS_BREAKDOWN_WEIGHTS.items())
        assert sum(hinted.reads) + sum(hinted.writes) == 500

    @pytest.mark.asyncio
    async def test_hinted_and_unhinted_agree_on_ties(self, tmp_path, monkeypatch):
        """Tied counts come back in label order from both plans."""
        from dashboard.data.write_journal import MEMORY_OPS_SQL, MEMORY_OPS_SQL_UNHINTED

        db_path = tmp_path / 'ops_equivalence_ties.db'
        rows = [
            (
                f'tie-{i}',
                f'operation-{i % 5}',
                'dark_factory',
                'agent-a',
                'read' if i % 2 == 0 else 'write',
                (_PLAN_FIXTURE_BASE + timedelta(minutes=i * 57.6)).isoformat(),
            )
            for i in range(250)
        ]
        _seed_write_ops(db_path, rows)
        window = {'hours': _PLAN_FIXTURE_HOURS, 'now': _PLAN_FIXTURE_NOW}

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            hinted = await _memory_ops_with_sql(conn, MEMORY_OPS_SQL, monkeypatch, **window)
            unhinted = await _memory_ops_with_sql(
                conn, MEMORY_OPS_SQL_UNHINTED, monkeypatch, **window,
            )

        assert hinted == unhinted
        assert hinted.by_operation == tuple((f'operation-{i}', 50) for i in range(5))


class TestMemoryOpsMissingIndexFallback:
    """Pins the observable when idx_wo_created goes missing.

    ``INDEXED BY`` makes the index a hard constraint, so SQLite raises
    ``no such index`` rather than scanning. get_memory_ops catches exactly
    that error and retries unhinted: the chart stays correct (just slow) and
    an ERROR naming the index is logged — never a 500 and never a silently
    empty chart (task 3519 review pass).
    """

    @pytest.mark.asyncio
    async def test_missing_index_falls_back_to_unhinted_with_error_log(
        self, tmp_path, caplog,
    ):
        db_path = tmp_path / 'ops_missing_index.db'
        now = datetime(2026, 4, 11, 12, 30, tzinfo=UTC)
        rows = [
            ('op-1', 'search', 'dark_factory', 'agent-a', 'read',
             (now - timedelta(hours=1)).isoformat()),
            ('op-2', 'add_memory', 'dark_factory', 'agent-a', 'write',
             (now - timedelta(hours=2)).isoformat()),
        ]
        _seed_write_ops(db_path, rows)

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            indexed = await _ops_value(conn, now=now)

        # WRITE_OPS_SCHEMA minus idx_wo_created — the coupling breaks here.
        setup_conn = sqlite3.connect(str(db_path))
        setup_conn.execute('DROP INDEX idx_wo_created')
        setup_conn.commit()
        setup_conn.close()

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            with caplog.at_level(logging.ERROR, logger='dashboard.data.write_journal'):
                fallback = await _ops_value(conn, now=now)

        assert fallback == indexed
        assert dict(fallback.by_operation) == {'search': 1, 'add_memory': 1}, (
            f'expected the fallback to still return the real rows, got: {fallback}'
        )
        assert any(
            record.name == 'dashboard.data.write_journal'
            and record.levelno == logging.ERROR
            and 'idx_wo_created' in record.getMessage()
            for record in caplog.records
        ), f'expected an ERROR naming idx_wo_created, got: {caplog.record_tuples}'
