"""Async aiosqlite-backed store for per-invocation cost records and account events."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

__all__ = ['CostStore']

_SCHEMA = """\
PRAGMA journal_mode=WAL;

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
    """Async SQLite writer for per-invocation cost records and account events.

    Use the async classmethod ``open()`` to create an instance — plain
    ``__init__`` cannot run async setup.

    Example::

        store = await CostStore.open(Path('data/orchestrator/runs.db'))
        await store.save_invocation(...)
        await store.save_account_event(...)
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    @classmethod
    async def open(cls, db_path: Path) -> CostStore:
        """Create a CostStore, ensuring the schema and WAL mode are set up."""
        instance = cls(db_path)
        instance.db_path.parent.mkdir(parents=True, exist_ok=True)
        await instance._ensure_schema()
        return instance

    async def _ensure_schema(self) -> None:
        """Run CREATE TABLE IF NOT EXISTS for both tables and set WAL mode."""
        async with aiosqlite.connect(str(self.db_path)) as conn:
            await conn.executescript(_SCHEMA)

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
        async with aiosqlite.connect(str(self.db_path)) as conn:
            await conn.execute(
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
            await conn.commit()

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
        async with aiosqlite.connect(str(self.db_path)) as conn:
            await conn.execute(
                'INSERT INTO account_events '
                '(account_name, event_type, project_id, run_id, details, created_at) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (account_name, event_type, project_id, run_id, details, created_at),
            )
            await conn.commit()
