"""Async SQLite base class and utilities for WAL-mode persistent connections.

Provides:
- apply_wal_pragmas(conn, busy_timeout_ms): standalone utility to configure WAL + busy_timeout
- AsyncSqliteBase: ABC with lifecycle management (open/close/context-manager/guard)
"""

from __future__ import annotations

import abc
from pathlib import Path

import aiosqlite

__all__ = ['apply_wal_pragmas', 'AsyncSqliteBase']


async def apply_wal_pragmas(conn: aiosqlite.Connection, *, busy_timeout_ms: int) -> None:
    """Configure WAL journal mode and optional busy_timeout on an open aiosqlite connection.

    Args:
        conn: An open aiosqlite connection.
        busy_timeout_ms: Milliseconds to wait for a locked database.
            Pass 0 to skip setting the busy_timeout pragma entirely.
    """
    await conn.execute('PRAGMA journal_mode=WAL')
    if busy_timeout_ms != 0:
        await conn.execute(f'PRAGMA busy_timeout={busy_timeout_ms}')
