"""Sync SQLite utilities for WAL-mode connections.

Provides:
- apply_full_durability_pragmas_sync(conn, busy_timeout_ms): WAL + busy_timeout + Phase 3 triad
- CheckpointResult: re-exported from shared.async_sqlite_base (same NamedTuple, single source)

This module is the synchronous counterpart to shared.async_sqlite_base, which uses
aiosqlite.  The three orchestrator stores (run_store, event_store, overrides) all use the
connect-per-call pattern rather than persistent connections, so they use this module-level
helper rather than inheriting from AsyncSqliteBase.

See ``docs/task-recovery-2026-05-13/`` for the production incident that drove this
convention across all fused-memory SQLite stores.
"""

from __future__ import annotations

import sqlite3

# Re-export CheckpointResult from the async sibling — single definition, no duplication.
from shared.async_sqlite_base import CheckpointResult
from shared.sqlite_durability import DURABILITY_TRIAD, require_wal_journal_mode

__all__ = ['apply_full_durability_pragmas_sync', 'CheckpointResult']


def apply_full_durability_pragmas_sync(
    conn: sqlite3.Connection,
    *,
    busy_timeout_ms: int,
) -> None:
    """Configure WAL mode, busy_timeout, and the Phase 3 durability triad on a sync connection.

    Sets, in order:

    1. ``PRAGMA journal_mode=WAL`` — enable WAL journal mode; raises ``RuntimeError``
       if the filesystem cannot support WAL (result != 'wal').
    2. ``PRAGMA busy_timeout={busy_timeout_ms}`` — milliseconds to wait for a locked DB.
    3. ``shared/src/shared/sqlite_durability.py::DURABILITY_TRIAD``, in order.

    Args:
        conn: An open sqlite3.Connection.
        busy_timeout_ms: Milliseconds to wait for a locked database.

    Raises:
        RuntimeError: If ``PRAGMA journal_mode=WAL`` returns a non-WAL result
            (e.g. 'delete' on a filesystem that doesn't support WAL).
    """
    require_wal_journal_mode(conn.execute('PRAGMA journal_mode=WAL').fetchone())
    conn.execute(f'PRAGMA busy_timeout={busy_timeout_ms}')
    for pragma in DURABILITY_TRIAD:
        conn.execute(pragma.statement)
