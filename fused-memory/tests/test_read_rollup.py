"""Tests for the hourly read-count rollup (task 5405).

Everything runs on real sqlite3 connections built from the module's own
public SQL constants; nothing is mocked.
"""

import sqlite3

import pytest

from fused_memory.services.read_rollup import (
    COUNT_READ_SQL,
    READ_OP_COUNTS_SCHEMA_SQL,
)


def _rollup_db() -> sqlite3.Connection:
    conn = sqlite3.connect(':memory:')
    conn.executescript(READ_OP_COUNTS_SCHEMA_SQL)
    return conn


def _count(
    conn: sqlite3.Connection,
    created_at: str,
    project_id: str | None = 'p',
    agent_id: str | None = 'a',
    operation: str | None = 'get_task',
    n: int = 1,
) -> None:
    conn.execute(COUNT_READ_SQL, (created_at, project_id, agent_id, operation, n))


def _rows(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute(
        'SELECT hour_start, project_id, agent_id, operation, count '
        'FROM read_op_counts ORDER BY hour_start, project_id, agent_id, operation'
    ).fetchall()


class TestSchema:
    def test_schema_is_idempotent_and_creates_read_op_counts(self):
        conn = sqlite3.connect(':memory:')
        conn.executescript(READ_OP_COUNTS_SCHEMA_SQL)
        conn.executescript(READ_OP_COUNTS_SCHEMA_SQL)
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert 'read_op_counts' in tables

    def test_table_rejects_non_positive_count(self):
        conn = _rollup_db()
        for bad in (0, -1):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    'INSERT INTO read_op_counts '
                    '(hour_start, project_id, agent_id, operation, count) '
                    "VALUES ('2026-10-11T21:00:00+00:00', 'p', 'a', 'get_task', ?)",
                    (bad,),
                )


class TestCountReadHourBucket:
    def test_mid_hour_timestamp_lands_at_its_hour_start(self):
        conn = _rollup_db()
        _count(conn, '2026-10-11T21:35:12.123456+00:00')
        assert _rows(conn) == [('2026-10-11T21:00:00+00:00', 'p', 'a', 'get_task', 1)]

    def test_hour_edges_share_one_bucket(self):
        conn = _rollup_db()
        _count(conn, '2026-10-11T21:00:00+00:00')
        _count(conn, '2026-10-11T21:59:59.999999+00:00')
        assert _rows(conn) == [('2026-10-11T21:00:00+00:00', 'p', 'a', 'get_task', 2)]

    def test_next_hour_starts_a_new_bucket(self):
        conn = _rollup_db()
        _count(conn, '2026-10-11T21:59:59.999999+00:00')
        _count(conn, '2026-10-11T22:00:00+00:00')
        assert _rows(conn) == [
            ('2026-10-11T21:00:00+00:00', 'p', 'a', 'get_task', 1),
            ('2026-10-11T22:00:00+00:00', 'p', 'a', 'get_task', 1),
        ]


class TestCountReadUpsert:
    def test_same_key_sums_n_into_one_row(self):
        conn = _rollup_db()
        _count(conn, '2026-10-11T21:05:00+00:00', n=2)
        _count(conn, '2026-10-11T21:45:00+00:00', n=5)
        assert _rows(conn) == [('2026-10-11T21:00:00+00:00', 'p', 'a', 'get_task', 7)]

    @pytest.mark.parametrize(
        'override',
        [
            {'project_id': 'q'},
            {'agent_id': 'b'},
            {'operation': 'get_statuses'},
        ],
    )
    def test_distinct_grain_column_gives_distinct_row(self, override):
        conn = _rollup_db()
        _count(conn, '2026-10-11T21:05:00+00:00')
        _count(conn, '2026-10-11T21:06:00+00:00', **override)
        rows = _rows(conn)
        assert len(rows) == 2
        assert [row[4] for row in rows] == [1, 1]

    def test_null_grain_columns_are_stored_as_empty_string_and_share_a_key(self):
        conn = _rollup_db()
        _count(conn, '2026-10-11T21:05:00+00:00', project_id=None, agent_id=None, operation=None)
        _count(conn, '2026-10-11T21:06:00+00:00', project_id='', agent_id='', operation='')
        _count(conn, '2026-10-11T21:07:00+00:00', project_id=None, agent_id=None, operation=None)
        assert _rows(conn) == [('2026-10-11T21:00:00+00:00', '', '', '', 3)]
