"""ReconciliationJournal.initialize() against a ``runs`` table that predates
its five later columns, on a real SQLite file."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from fused_memory.reconciliation.journal import ReconciliationJournal

ADDED_RUNS_COLUMNS = ('triggered_by', 'instance_id', 'session_id', 'stage_cursor', 'attempt')

LEGACY_RUNS_SQL = """
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    run_type TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    events_processed INTEGER DEFAULT 0,
    stage_reports TEXT DEFAULT '{}',
    status TEXT DEFAULT 'running'
)
"""

LEGACY_RUN_ID = 'legacy-run'


def _write_legacy_journal(data_dir: Path) -> Path:
    data_dir.mkdir()
    db_path = data_dir / 'reconciliation.db'
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(LEGACY_RUNS_SQL)
        conn.execute(
            'INSERT INTO runs (id, project_id, run_type, trigger_reason, started_at, status)'
            " VALUES (?, 'p', 'full', 'legacy', '2026-01-01T00:00:00+00:00', 'completed')",
            (LEGACY_RUN_ID,),
        )
        conn.commit()
    return db_path


def _runs_column_names(db_path: Path) -> list[str]:
    with closing(sqlite3.connect(db_path)) as conn:
        return [row[1] for row in conn.execute('PRAGMA table_info(runs)')]


async def _initialize_and_close(data_dir: Path) -> None:
    journal = ReconciliationJournal(data_dir)
    await journal.initialize()
    await journal.close()


@pytest.mark.asyncio
async def test_initialize_adds_every_later_runs_column_to_a_legacy_journal(tmp_path):
    db_path = _write_legacy_journal(tmp_path / 'recon')
    assert not set(ADDED_RUNS_COLUMNS) & set(_runs_column_names(db_path))

    journal = ReconciliationJournal(tmp_path / 'recon')
    await journal.initialize()
    try:
        legacy_run = await journal.get_run(LEGACY_RUN_ID)
    finally:
        await journal.close()

    assert legacy_run is not None
    assert (
        legacy_run.triggered_by,
        legacy_run.instance_id,
        legacy_run.session_id,
        legacy_run.stage_cursor,
    ) == (None, None, None, None)
    assert set(ADDED_RUNS_COLUMNS) <= set(_runs_column_names(db_path))
    with closing(sqlite3.connect(db_path)) as conn:
        stored_attempt = conn.execute(
            'SELECT attempt FROM runs WHERE id = ?', (LEGACY_RUN_ID,)
        ).fetchone()[0]
    assert stored_attempt == 0


@pytest.mark.asyncio
async def test_reinitializing_a_migrated_legacy_journal_leaves_its_schema_unchanged(tmp_path):
    db_path = _write_legacy_journal(tmp_path / 'recon')
    await _initialize_and_close(tmp_path / 'recon')
    migrated_columns = _runs_column_names(db_path)

    await _initialize_and_close(tmp_path / 'recon')

    assert _runs_column_names(db_path) == migrated_columns
