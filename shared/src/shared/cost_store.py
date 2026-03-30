"""Async aiosqlite-backed store for per-invocation cost records and account events.

Uses a persistent connection opened via open()/close() or the async context manager::

    async with CostStore(path) as store:
        await store.save_invocation(...)
        await store.save_account_event(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import aiosqlite

__all__ = ['CostStore']

# Schema without PRAGMA — pragmas are set once on the persistent connection.
_SCHEMA = """\
CREATE TABLE IF NOT EXISTS invocations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL,
    task_id             TEXT,
    project_id          TEXT NOT NULL,
    account_name        TEXT NOT NULL,
    model               TEXT NOT NULL,
    role                TEXT NOT NULL,
    cost_usd            REAL NOT NULL DEFAULT 0.0,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    cache_read_tokens   INTEGER,
    cache_create_tokens INTEGER,
    duration_ms         INTEGER NOT NULL DEFAULT 0,
    capped              INTEGER NOT NULL DEFAULT 0,
    started_at          TEXT NOT NULL,
    completed_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_inv_project
    ON invocations(project_id);

CREATE INDEX IF NOT EXISTS idx_inv_account
    ON invocations(account_name);

CREATE INDEX IF NOT EXISTS idx_inv_run
    ON invocations(run_id);

CREATE TABLE IF NOT EXISTS account_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_name TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    project_id   TEXT,
    run_id       TEXT,
    details      TEXT,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_acct_evt_account
    ON account_events(account_name);
"""


class CostStore:
    """Persistent-connection SQLite writer for cost records and account events.

    Lifecycle::

        store = CostStore(path)
        await store.open()
        try:
            await store.save_invocation(...)
        finally:
            await store.close()

    Or via async context manager::

        async with CostStore(path) as store:
            await store.save_invocation(...)
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    # -- lifecycle ------------------------------------------------------------

    async def open(self) -> None:
        """Open persistent connection, set WAL + busy_timeout, ensure schema."""
        if self._conn is not None:
            raise RuntimeError('CostStore already opened')
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(self.db_path))
        try:
            await conn.execute('PRAGMA journal_mode=WAL')
            await conn.execute('PRAGMA busy_timeout=30000')
            await conn.executescript(_SCHEMA)
        except BaseException:
            await conn.close()
            raise
        self._conn = conn

    async def close(self) -> None:
        """Close the connection. Idempotent — safe to call when already closed."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # -- async context manager ------------------------------------------------

    async def __aenter__(self) -> CostStore:
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.close()

    # -- internal helpers -----------------------------------------------------

    def _require_conn(self) -> aiosqlite.Connection:
        """Return the open connection or raise RuntimeError."""
        if self._conn is None:
            raise RuntimeError('CostStore not opened')
        return self._conn

    async def _execute(self, sql: str, params: tuple[Any, ...]) -> None:
        """Execute a single statement and commit."""
        conn = self._require_conn()
        await conn.execute(sql, params)
        await conn.commit()

    # -- public write API -----------------------------------------------------

    async def save_invocation(
        self,
        *,
        run_id: str,
        task_id: str | None,
        project_id: str,
        account_name: str,
        model: str,
        role: str,
        cost_usd: float,
        input_tokens: int | None,
        output_tokens: int | None,
        cache_read_tokens: int | None,
        cache_create_tokens: int | None,
        duration_ms: int,
        capped: bool,
        started_at: str,
        completed_at: str,
    ) -> None:
        """Insert one row into the invocations table."""
        await self._execute(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, input_tokens, output_tokens, cache_read_tokens, '
            ' cache_create_tokens, duration_ms, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                run_id,
                task_id,
                project_id,
                account_name,
                model,
                role,
                cost_usd,
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_create_tokens,
                duration_ms,
                int(capped),
                started_at,
                completed_at,
            ),
        )

    async def save_account_event(
        self,
        *,
        account_name: str,
        event_type: str,
        project_id: str | None,
        run_id: str | None,
        details: str | None,
        created_at: str,
    ) -> None:
        """Insert one row into the account_events table."""
        await self._execute(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (account_name, event_type, project_id, run_id, details, created_at),
        )
