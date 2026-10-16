"""Tests for the hourly read-count rollup (task 5405).

Everything runs on real sqlite3 connections built from the module's own
public SQL constants; nothing is mocked.
"""

import itertools
import sqlite3
import uuid
from collections import Counter

import pytest

from fused_memory.services.read_rollup import (
    COUNT_ONLY_READ_PREDICATE,
    COUNT_READ_SQL,
    DELETE_COUNT_ONLY_READS_SQL,
    RAW_READS_BY_ROLLUP_KEY_SQL,
    READ_OP_COUNTS_SCHEMA_SQL,
    keeps_read_row,
)
from fused_memory.services.write_journal import SCHEMA_SQL as WRITE_OPS_SCHEMA_SQL


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


class TestKeepsReadRow:
    @pytest.mark.parametrize(
        ('operation', 'success', 'expected'),
        [
            ('search', True, True),
            ('search', False, True),
            ('get_task', True, False),
            ('get_task', False, True),
            ('get_entity', False, True),
            (None, True, False),
        ],
    )
    def test_truth_table(self, operation, success, expected):
        assert keeps_read_row(operation=operation, success=success) is expected


def _insert_write_op(
    conn: sqlite3.Connection,
    *,
    kind: str,
    operation: str | None,
    success: int | None,
) -> str:
    row_id = str(uuid.uuid4())
    conn.execute(
        'INSERT INTO write_ops (id, operation, kind, success, created_at) '
        "VALUES (?, ?, ?, ?, '2026-10-11T21:00:00+00:00')",
        (row_id, operation, kind, success),
    )
    return row_id


class TestCountOnlyPredicateAgreesWithKeepsReadRow:
    def test_sql_predicate_selects_exactly_the_reads_python_would_count(self):
        conn = sqlite3.connect(':memory:')
        conn.executescript(WRITE_OPS_SCHEMA_SQL)
        operations = ['search', 'get_task', 'get_statuses', 'get_entity', None]
        expected_count_only: set[str] = set()
        for operation, success in itertools.product(operations, (1, 0)):
            read_id = _insert_write_op(conn, kind='read', operation=operation, success=success)
            if not keeps_read_row(operation=operation, success=bool(success)):
                expected_count_only.add(read_id)
            _insert_write_op(conn, kind='write', operation=operation, success=success)
        _insert_write_op(conn, kind='read', operation='get_task', success=None)

        selected = {
            row[0]
            for row in conn.execute(f'SELECT id FROM write_ops WHERE {COUNT_ONLY_READ_PREDICATE}')
        }

        assert selected == expected_count_only
        assert expected_count_only


def _journal_db() -> sqlite3.Connection:
    conn = sqlite3.connect(':memory:')
    conn.executescript(WRITE_OPS_SCHEMA_SQL)
    conn.executescript(READ_OP_COUNTS_SCHEMA_SQL)
    return conn


_HOURS = ('2026-10-11T19', '2026-10-11T20', '2026-10-11T21')
_COUNT_ONLY_OPERATIONS = ('get_task', 'get_statuses', None)


def _seed_history(conn: sqlite3.Connection) -> list[dict]:
    """Interleave count-only reads with search, failed and write rows by rowid.

    Returns the seeded rows, rowid included, in insertion order.
    """
    seeded = []
    for i in range(60):
        at = f'{_HOURS[(i // 20) % 3]}:{(i * 7) % 60:02d}:{i % 60:02d}.{i:06d}+00:00'
        shape = i % 5
        if shape == 4:
            kind, operation, success = 'write', 'add_memory', 1
        elif shape == 3:
            kind, operation, success = 'read', 'search', 1
        elif shape == 2 and i % 10 == 2:
            kind, operation, success = 'read', 'get_entity', 0
        else:
            kind, operation, success = 'read', _COUNT_ONLY_OPERATIONS[i % 3], 1
        row = {
            'id': f'op-{i}',
            'operation': operation,
            'project_id': ('dark_factory', 'reify')[(i // 2) % 2],
            'agent_id': ('agent-a', None)[(i // 3) % 2],
            'kind': kind,
            'success': success,
            'error': None if success else 'boom',
            'params': '{"query": "q"}',
            'created_at': at,
        }
        cursor = conn.execute(
            'INSERT INTO write_ops (id, operation, project_id, agent_id, kind, success, error,'
            ' params, created_at) VALUES (:id, :operation, :project_id, :agent_id, :kind,'
            ' :success, :error, :params, :created_at)',
            row,
        )
        seeded.append({**row, 'rowid': cursor.lastrowid})
    return seeded


def _is_count_only(row: dict) -> bool:
    return row['kind'] == 'read' and not keeps_read_row(
        operation=row['operation'], success=bool(row['success'])
    )


def _full_rows(conn: sqlite3.Connection) -> dict[int, tuple]:
    return {row[0]: row[1:] for row in conn.execute('SELECT rowid, * FROM write_ops')}


def _live_path_counts(rows: list[dict]) -> Counter:
    """Key -> count, produced by the LIVE upsert applied once per row."""
    conn = sqlite3.connect(':memory:')
    conn.executescript(READ_OP_COUNTS_SCHEMA_SQL)
    for row in rows:
        conn.execute(
            COUNT_READ_SQL,
            (row['created_at'], row['project_id'], row['agent_id'], row['operation'], 1),
        )
    return Counter({
        tuple(key): count
        for *key, count in conn.execute(
            'SELECT hour_start, project_id, agent_id, operation, count FROM read_op_counts'
        )
    })


class TestMigrationChunkSql:
    _LO, _HI = 8, 47

    def _window_count_only(self, seeded: list[dict]) -> list[dict]:
        return [
            row for row in seeded
            if self._LO <= row['rowid'] < self._HI and _is_count_only(row)
        ]

    def test_seed_covers_every_shape(self):
        seeded = _seed_history(_journal_db())
        count_only = [row for row in seeded if _is_count_only(row)]
        assert {row['created_at'][:13] for row in count_only} == set(_HOURS)
        assert {row['agent_id'] for row in count_only} == {'agent-a', None}
        assert {row['project_id'] for row in count_only} == {'dark_factory', 'reify'}
        assert {row['operation'] for row in seeded if not _is_count_only(row)} == {
            'add_memory', 'search', 'get_entity',
        }

    def test_preview_matches_the_live_upsert_for_count_only_rows_in_the_window(self):
        conn = _journal_db()
        seeded = _seed_history(conn)

        preview = conn.execute(RAW_READS_BY_ROLLUP_KEY_SQL, (self._LO, self._HI)).fetchall()

        expected = _live_path_counts(self._window_count_only(seeded))
        assert Counter({tuple(key): n for *key, n in preview}) == expected
        assert sum(n for *_, n in preview) == len(self._window_count_only(seeded))
        assert len(preview) == len(expected)

    def test_preview_rows_fed_back_through_the_live_upsert_keep_their_keys(self):
        conn = _journal_db()
        _seed_history(conn)
        preview = conn.execute(RAW_READS_BY_ROLLUP_KEY_SQL, (self._LO, self._HI)).fetchall()

        conn.executemany(COUNT_READ_SQL, preview)

        stored = conn.execute(
            'SELECT hour_start, project_id, agent_id, operation, count FROM read_op_counts'
        ).fetchall()
        assert sorted(stored) == sorted(preview)

    def test_delete_removes_exactly_the_window_count_only_rows(self):
        conn = _journal_db()
        seeded = _seed_history(conn)
        before = _full_rows(conn)
        preview = conn.execute(RAW_READS_BY_ROLLUP_KEY_SQL, (self._LO, self._HI)).fetchall()

        deleted = conn.execute(DELETE_COUNT_ONLY_READS_SQL, (self._LO, self._HI)).rowcount

        removed = {row['rowid'] for row in self._window_count_only(seeded)}
        assert deleted == sum(n for *_, n in preview) == len(removed)
        after = _full_rows(conn)
        assert set(after) == set(before) - removed
        assert all(after[rowid] == before[rowid] for rowid in after)

    def test_rows_outside_the_window_are_untouched(self):
        conn = _journal_db()
        seeded = _seed_history(conn)
        before = _full_rows(conn)

        conn.execute(DELETE_COUNT_ONLY_READS_SQL, (self._LO, self._HI))

        outside = [row for row in seeded if not self._LO <= row['rowid'] < self._HI]
        assert any(_is_count_only(row) for row in outside)
        after = _full_rows(conn)
        assert all(after[row['rowid']] == before[row['rowid']] for row in outside)
