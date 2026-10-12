"""Add a SQLite column if it is missing, tolerating a concurrent adder."""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

__all__ = ['add_column_if_missing', 'add_column_if_missing_sync']

logger = logging.getLogger(__name__)

_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')

_COLUMN_PRESENT_SQL = 'SELECT 1 FROM pragma_table_info(?) WHERE name = ?'


def _alter_sql(table: str, column: str, column_ddl: str) -> str:
    for argument, value in (('table', table), ('column', column)):
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError(
                f'{argument} {value!r} is not a bare SQL identifier '
                f'({_IDENTIFIER.pattern}); it is interpolated into ALTER TABLE, '
                'which cannot bind identifiers'
            )
    return f'ALTER TABLE {table} ADD COLUMN {column} {column_ddl}'


async def _column_present(db: aiosqlite.Connection, table: str, column: str) -> bool:
    return bool(await db.execute_fetchall(_COLUMN_PRESENT_SQL, (table, column)))


def _column_present_sync(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return conn.execute(_COLUMN_PRESENT_SQL, (table, column)).fetchone() is not None


async def add_column_if_missing(
    db: aiosqlite.Connection, table: str, column: str, column_ddl: str
) -> bool:
    """Add ``table.column`` with ``column_ddl`` unless it already exists.

    Returns True only when THIS call's ALTER added the column, so a caller can
    log its own line and run column-specific follow-ups (an index, a backfill)
    exactly once. ``column_ddl`` is the type-and-constraints tail after the
    column name, e.g. ``"TEXT NOT NULL DEFAULT 'write'"``; it is a trusted code
    literal and is not parsed. ``table`` and ``column`` must be bare
    identifiers, or ValueError is raised before any SQL runs.

    SQLite's ALTER TABLE has no IF NOT EXISTS, and two connections opening one
    file can both probe the column as missing; the loser's ALTER then fails.
    Tolerance is structural: after a failed ALTER the column is probed again,
    and the failure is swallowed only if the column now exists, because the
    migration's goal is then met. The error text is never inspected. Every
    other failure (no such table, database is locked, disk I/O) re-raises the
    original error, since a silently half-migrated schema surfaces much later
    as a confusing "no such column", which is worse than a loud open failure.

    Never commits: the caller owns the transaction. The additive-only,
    no-user_version rule these migrations follow is stated at
    ``orchestrator/src/orchestrator/run_store.py::RunStore._migrate_task_results_block_context``.
    """
    statement = _alter_sql(table, column, column_ddl)
    if await _column_present(db, table, column):
        return False
    try:
        await db.execute(statement)
    except sqlite3.OperationalError:
        if not await _column_present(db, table, column):
            raise
        logger.debug('%s.%s was added concurrently by another connection', table, column)
        return False
    return True


def add_column_if_missing_sync(
    conn: sqlite3.Connection, table: str, column: str, column_ddl: str
) -> bool:
    """Synchronous twin of :func:`add_column_if_missing`; same contract."""
    statement = _alter_sql(table, column, column_ddl)
    if _column_present_sync(conn, table, column):
        return False
    try:
        conn.execute(statement)
    except sqlite3.OperationalError:
        if not _column_present_sync(conn, table, column):
            raise
        logger.debug('%s.%s was added concurrently by another connection', table, column)
        return False
    return True
