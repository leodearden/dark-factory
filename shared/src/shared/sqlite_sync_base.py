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
from shared.async_sqlite_base import CheckpointResult  # noqa: F401

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
    3. ``PRAGMA synchronous=FULL`` (2) — fsync per-commit; eliminates corruption on
       unexpected shutdown without relying on WAL-checkpoint timing.
    4. ``PRAGMA wal_autocheckpoint=100`` — auto-checkpoint after every 100 WAL pages to
       bound WAL growth under normal load.
    5. ``PRAGMA journal_size_limit=67108864`` (64 MiB) — caps the WAL file size to
       prevent unbounded disk use during high-write bursts.

    See ``docs/task-recovery-2026-05-13/`` for the production incident that drove this
    convention across all fused-memory SQLite stores.

    Args:
        conn: An open sqlite3.Connection.
        busy_timeout_ms: Milliseconds to wait for a locked database.

    Raises:
        RuntimeError: If ``PRAGMA journal_mode=WAL`` returns a non-WAL result
            (e.g. 'delete' on a filesystem that doesn't support WAL).
    """
    row = conn.execute('PRAGMA journal_mode=WAL').fetchone()
    if row is None or row[0] != 'wal':
        got = row[0] if row is not None else None
        raise RuntimeError(f'Failed to enable WAL journal mode (got {got!r})')
    conn.execute(f'PRAGMA busy_timeout={busy_timeout_ms}')
    # synchronous=FULL: per-commit fsync. Cost is ~1-5ms/commit; the
    # win is crash durability without relying on WAL checkpoints. See
    # docs/task-recovery-2026-05-13/ for the prod incident that drove
    # this change across all fused-memory SQLite stores.
    conn.execute('PRAGMA synchronous=FULL')
    conn.execute('PRAGMA wal_autocheckpoint=100')
    conn.execute('PRAGMA journal_size_limit=67108864')
