"""Tests for the flake ledger (plans/flake-ledger-prd.md, task α).

Schema assertions deliberately bypass the module under test and read the
on-disk truth with raw ``sqlite3`` (the ``test_event_store.py`` / ``test_run_store.py``
convention), so a reader mapping bug can never make a schema test pass.

STRUCTURAL CONSTRAINT — sync and async tests live in strictly separate classes.
``orchestrator/pyproject.toml`` sets no ``asyncio_mode`` (pytest-asyncio STRICT) and
promotes "marked with '@pytest.mark.asyncio' but it is not an async function" to an
ERROR-level filterwarning, so a sync ``def test_`` inside an ``@pytest.mark.asyncio``
class is a collection FAILURE, not a warning.  The ledger API is deliberately
mixed-colour (sync writer/readers, async debt functions); the split is structural.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

# Column order is pinned, not merely presence: run_store.py:53-58 establishes
# physical-column-order parity between a freshly-created DB and an ALTER-migrated
# one as this repo's migration convention (an operator forensic `SELECT *` is
# positional).  These lists are PRD §5.3 verbatim.
_OCCURRENCE_COLUMNS = [
    'id',
    'observed_at',
    'test_id',
    'project_id',
    'verdict',
    'call_site',
    'runner',
    'merge_sha',
    'task_id',
    'psi_cpu_some10',
    'detail',
]

_DEBT_COLUMNS = [
    'test_id',
    'project_id',
    'opened_at',
    'resolved_at',
    'owner_task_id',
    'open_count',
    'prior_resolved_at',
    'prior_resolving_commit',
    'last_occurrence_at',
]


def _rows(db_path: Path, sql: str, params: tuple = ()) -> list[dict]:
    """Read rows with raw sqlite3 — never through the module under test."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _table_names(db_path: Path) -> list[str]:
    return [r['name'] for r in _rows(db_path, "SELECT name FROM sqlite_master WHERE type='table'")]


def _column_names(db_path: Path, table: str) -> list[str]:
    return [r['name'] for r in _rows(db_path, f'PRAGMA table_info({table})')]


def _table_sql(db_path: Path, name: str) -> str:
    rows = _rows(db_path, 'SELECT sql FROM sqlite_master WHERE name = ?', (name,))
    assert rows, f'no sqlite_master entry for {name!r}'
    return rows[0]['sql'] or ''


class TestEnsureSchema:
    """``ensure_schema`` creates both PRD §5.3 tables additively on an existing runs.db."""

    def test_creates_both_tables(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        names = _table_names(db_path)
        assert 'flake_occurrence' in names
        assert 'flake_debt' in names

    def test_flake_occurrence_column_order(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        assert _column_names(db_path, 'flake_occurrence') == _OCCURRENCE_COLUMNS

    def test_flake_debt_column_order(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        assert _column_names(db_path, 'flake_debt') == _DEBT_COLUMNS

    def test_flake_debt_test_id_is_primary_key(self, tmp_path: Path) -> None:
        """§5.3: runs.db is per-project, so test_id ALONE is the key — which is what
        makes η's recurrence trigger a PK lookup rather than a scan."""
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        pk_columns = [r['name'] for r in _rows(db_path, 'PRAGMA table_info(flake_debt)') if r['pk']]
        assert pk_columns == ['test_id']

    def test_flake_occurrence_id_is_autoincrement(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        assert 'AUTOINCREMENT' in _table_sql(db_path, 'flake_occurrence').upper()

        pk_columns = [
            r['name'] for r in _rows(db_path, 'PRAGMA table_info(flake_occurrence)') if r['pk']
        ]
        assert pk_columns == ['id']

        # AUTOINCREMENT (as opposed to plain INTEGER PRIMARY KEY) materialises
        # sqlite_sequence on first insert — monotonic ids, never reused.
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                'INSERT INTO flake_occurrence '
                '(observed_at, test_id, project_id, verdict, call_site) '
                "VALUES ('2026-08-06T12:00:00+00:00', 't', 'p', 'unconfirmable', 'merge_gate')"
            )
            conn.commit()
        finally:
            conn.close()
        assert 'sqlite_sequence' in _table_names(db_path)

    def test_dedup_index_is_unique_on_the_idempotency_triple(self, tmp_path: Path) -> None:
        """§8.3's idempotency key ``(test_id, observed_at, call_site)`` is enforced
        declaratively by a UNIQUE index, not by swallowing an IntegrityError."""
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        indexes = _rows(db_path, "SELECT name, sql FROM sqlite_master WHERE type='index'")
        dedup = [
            r
            for r in indexes
            if r['sql']
            and 'flake_occurrence' in r['sql']
            and 'test_id' in r['sql']
            and 'observed_at' in r['sql']
            and 'call_site' in r['sql']
        ]
        assert len(dedup) == 1, f'expected exactly one dedup index, got {indexes}'
        assert 'unique' in dedup[0]['sql'].lower()

    def test_column_defaults(self, tmp_path: Path) -> None:
        """``open_count`` NOT NULL DEFAULT 1 and ``detail`` DEFAULT '{}', verified by
        inserting a minimal row and reading the defaults back."""
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                'INSERT INTO flake_occurrence '
                '(observed_at, test_id, project_id, verdict, call_site) '
                "VALUES ('2026-08-06T12:00:00+00:00', 't', 'p', 'unconfirmable', 'merge_gate')"
            )
            conn.execute(
                'INSERT INTO flake_debt (test_id, project_id, opened_at, last_occurrence_at) '
                "VALUES ('t', 'p', '2026-08-06T12:00:00+00:00', '2026-08-06T12:00:00+00:00')"
            )
            conn.commit()
        finally:
            conn.close()

        assert _rows(db_path, 'SELECT detail FROM flake_occurrence')[0]['detail'] == '{}'
        assert _rows(db_path, 'SELECT open_count FROM flake_debt')[0]['open_count'] == 1

        # NOT NULL, so an explicit NULL is rejected rather than silently stored.
        conn = sqlite3.connect(str(db_path))
        try:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    'INSERT INTO flake_debt '
                    '(test_id, project_id, opened_at, last_occurrence_at, open_count) '
                    "VALUES ('t2', 'p', '2026-08-06T12:00:00+00:00', "
                    "'2026-08-06T12:00:00+00:00', NULL)"
                )
        finally:
            conn.close()

    def test_ensure_schema_is_idempotent(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)
        ensure_schema(db_path)

        names = _table_names(db_path)
        assert names.count('flake_occurrence') == 1
        assert names.count('flake_debt') == 1

    def test_additive_on_an_existing_runs_db(self, tmp_path: Path) -> None:
        """The ledger is a FIFTH owner of runs.db — its tables must appear on every
        existing DB on first write, with no migration to sequence.

        The "existing" DB is built by the REAL ``EventStore``/``RunStore`` writing real
        rows, not a hand-copied DDL replica, so additivity is proven against the actual
        other owners' schemas and cannot silently drift from them.
        """
        from orchestrator.event_store import EventStore, EventType
        from orchestrator.flake_ledger import ensure_schema
        from orchestrator.run_store import RunStore

        db_path = tmp_path / 'runs.db'
        EventStore(db_path, 'run-x').emit(EventType.task_started, task_id='42')
        RunStore(db_path).start_run('run-x', 'dark_factory', '2026-08-06T12:00:00+00:00')

        events_before = _rows(db_path, 'SELECT * FROM events ORDER BY id')
        runs_before = _rows(db_path, 'SELECT * FROM runs ORDER BY run_id')
        assert len(events_before) == 1
        assert len(runs_before) == 1

        ensure_schema(db_path)

        # The other owners' rows are untouched...
        assert _rows(db_path, 'SELECT * FROM events ORDER BY id') == events_before
        assert _rows(db_path, 'SELECT * FROM runs ORDER BY run_id') == runs_before
        # ...and the two new tables now exist alongside them.
        names = _table_names(db_path)
        assert 'flake_occurrence' in names
        assert 'flake_debt' in names
        assert 'events' in names
        assert 'runs' in names

    def test_ledger_db_path(self, tmp_path: Path) -> None:
        """One spelling of harness.py:2119's hand-built runs.db literal, for the four
        downstream consumers (β, ε, ζ, ι)."""
        from orchestrator.flake_ledger import ledger_db_path

        assert ledger_db_path(tmp_path) == tmp_path / 'data' / 'orchestrator' / 'runs.db'
