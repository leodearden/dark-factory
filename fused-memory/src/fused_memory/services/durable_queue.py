"""SQLite-backed durable write queue for Graphiti operations.

Replaces the in-memory QueueService with crash-safe persistence, retry with
exponential backoff, dead-lettering, and per-group worker pools.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

# -- Schema ------------------------------------------------------------------

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS write_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id    TEXT    NOT NULL,
    operation   TEXT    NOT NULL,
    payload     TEXT    NOT NULL,
    callback_type TEXT,
    status      TEXT    NOT NULL DEFAULT 'pending',
    attempts    INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 5,
    next_retry_at REAL  NOT NULL DEFAULT 0,
    created_at  REAL    NOT NULL,
    completed_at REAL,
    error       TEXT
);
"""

_CREATE_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_wq_status_group
    ON write_queue (status, group_id, next_retry_at);
"""


# -- Data class ---------------------------------------------------------------

class QueueItem:
    """Lightweight representation of a row."""

    __slots__ = (
        'id', 'group_id', 'operation', 'payload', 'callback_type',
        'status', 'attempts', 'max_attempts', 'next_retry_at',
        'created_at', 'completed_at', 'error',
    )

    def __init__(self, row: aiosqlite.Row | tuple):
        (
            self.id, self.group_id, self.operation, self.payload,
            self.callback_type, self.status, self.attempts, self.max_attempts,
            self.next_retry_at, self.created_at, self.completed_at, self.error,
        ) = row

    def parsed_payload(self) -> dict[str, Any]:
        return json.loads(self.payload)


# -- Queue --------------------------------------------------------------------

CallbackFn = Callable[[str, Any, dict[str, Any]], Coroutine[Any, Any, None]]


class DurableWriteQueue:
    """SQLite WAL-backed write queue with per-group workers and global semaphore."""

    def __init__(
        self,
        *,
        data_dir: str | Path,
        execute_write: Callable[..., Coroutine[Any, Any, Any]],
        workers_per_group: int = 3,
        semaphore_limit: int = 20,
        max_attempts: int = 5,
        retry_base_seconds: float = 5.0,
        retry_max_delay_seconds: float = 300.0,
        write_timeout_seconds: float = 120.0,
    ):
        self._data_dir = Path(data_dir)
        self._execute_write = execute_write
        self._workers_per_group = workers_per_group
        self._max_attempts = max_attempts
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_delay_seconds = retry_max_delay_seconds
        self._write_timeout_seconds = write_timeout_seconds

        self._semaphore = asyncio.Semaphore(semaphore_limit)
        self._db: aiosqlite.Connection | None = None
        self._callbacks: dict[str, CallbackFn] = {}
        self._group_events: dict[str, asyncio.Event] = {}
        self._group_locks: dict[str, asyncio.Lock] = {}
        self._worker_tasks: dict[str, list[asyncio.Task]] = {}
        self._closed = False

    # -- lifecycle ------------------------------------------------------------

    async def initialize(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        db_path = self._data_dir / 'write_queue.db'
        self._db = await aiosqlite.connect(str(db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute('PRAGMA journal_mode=WAL')
        await self._db.execute('PRAGMA busy_timeout=5000')
        await self._db.execute(_CREATE_TABLE)
        await self._db.execute(_CREATE_INDEX)
        await self._db.commit()
        # Recover any items left in_flight from a previous crash
        await self._recover_in_flight()
        # Spin up workers for groups that have pending work
        await self._start_workers_for_pending_groups()
        logger.info('DurableWriteQueue initialized at %s', db_path)

    async def close(self) -> None:
        self._closed = True
        # Cancel all workers
        for tasks in self._worker_tasks.values():
            for t in tasks:
                t.cancel()
        # Wait for them to finish
        all_tasks = [t for tasks in self._worker_tasks.values() for t in tasks]
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        if self._db:
            await self._db.close()
            self._db = None
        logger.info('DurableWriteQueue closed')

    # -- callbacks ------------------------------------------------------------

    def register_callback(self, name: str, fn: CallbackFn) -> None:
        self._callbacks[name] = fn

    # -- enqueue --------------------------------------------------------------

    async def enqueue(
        self,
        group_id: str,
        operation: str,
        payload: dict[str, Any],
        callback_type: str | None = None,
    ) -> int:
        """Persist a write item and signal workers. Returns item id."""
        assert self._db is not None
        now = time.time()
        cursor = await self._db.execute(
            'INSERT INTO write_queue '
            '(group_id, operation, payload, callback_type, status, attempts, '
            ' max_attempts, next_retry_at, created_at) '
            'VALUES (?, ?, ?, ?, ?, 0, ?, 0, ?)',
            (group_id, operation, json.dumps(payload), callback_type,
             'pending', self._max_attempts, now),
        )
        await self._db.commit()
        item_id = cursor.lastrowid
        self._ensure_workers(group_id)
        self._signal_group(group_id)
        return item_id  # type: ignore[return-value]

    async def enqueue_batch(
        self, items: list[dict[str, Any]]
    ) -> list[int]:
        """Bulk insert in a single transaction. Each dict needs group_id,
        operation, payload, and optionally callback_type."""
        assert self._db is not None
        now = time.time()
        ids: list[int] = []
        groups_seen: set[str] = set()
        await self._db.execute('BEGIN')
        try:
            for item in items:
                cursor = await self._db.execute(
                    'INSERT INTO write_queue '
                    '(group_id, operation, payload, callback_type, status, attempts, '
                    ' max_attempts, next_retry_at, created_at) '
                    'VALUES (?, ?, ?, ?, ?, 0, ?, 0, ?)',
                    (item['group_id'], item['operation'],
                     json.dumps(item['payload']),
                     item.get('callback_type'),
                     'pending', self._max_attempts, now),
                )
                ids.append(cursor.lastrowid)  # type: ignore[arg-type]
                groups_seen.add(item['group_id'])
            await self._db.commit()
        except Exception:
            await self._db.rollback()
            raise
        for g in groups_seen:
            self._ensure_workers(g)
            self._signal_group(g)
        return ids

    # -- worker pool ----------------------------------------------------------

    def _ensure_workers(self, group_id: str) -> None:
        """Spawn workers for group_id if not already running."""
        if self._closed:
            return
        if group_id not in self._group_events:
            self._group_events[group_id] = asyncio.Event()
        if group_id not in self._group_locks:
            self._group_locks[group_id] = asyncio.Lock()
        existing = self._worker_tasks.get(group_id, [])
        # Clean up completed tasks
        alive = [t for t in existing if not t.done()]
        needed = self._workers_per_group - len(alive)
        for _ in range(needed):
            task = asyncio.create_task(
                self._worker_loop(group_id), name=f'dq-worker-{group_id}'
            )
            alive.append(task)
        self._worker_tasks[group_id] = alive

    def _signal_group(self, group_id: str) -> None:
        ev = self._group_events.get(group_id)
        if ev:
            ev.set()

    async def _worker_loop(self, group_id: str) -> None:
        event = self._group_events[group_id]
        while not self._closed:
            item = await self._claim_next(group_id)
            if item is None:
                event.clear()
                # Short poll interval so retries with backoff are picked up promptly
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(event.wait(), timeout=0.5)
                continue
            await self._process_item(item)

    async def _claim_next(self, group_id: str) -> QueueItem | None:
        """Claim the next pending/retry item for this group.

        Uses a per-group asyncio.Lock so only one worker claims at a time.
        """
        assert self._db is not None
        lock = self._group_locks[group_id]
        async with lock:
            now = time.time()
            cursor = await self._db.execute(
                "SELECT * FROM write_queue "
                "WHERE group_id = ? AND status IN ('pending', 'retry') "
                "  AND next_retry_at <= ? "
                "ORDER BY id ASC LIMIT 1",
                (group_id, now),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            item = QueueItem(tuple(row))
            await self._db.execute(
                "UPDATE write_queue SET status = 'in_flight' WHERE id = ?",
                (item.id,),
            )
            await self._db.commit()
            return item

    async def _process_item(self, item: QueueItem) -> None:
        """Execute the write, handle success/failure.

        Callbacks run *before* marking completed so that a callback
        failure triggers retry instead of being silently lost.
        """
        async with self._semaphore:
            try:
                result = await asyncio.wait_for(
                    self._execute_write(item.operation, item.parsed_payload()),
                    timeout=self._write_timeout_seconds,
                )
                # Fire callback before marking completed — failure retries item
                if item.callback_type and item.callback_type in self._callbacks:
                    await self._callbacks[item.callback_type](
                        item.callback_type, result, item.parsed_payload()
                    )
                await self._mark_completed(item)
            except Exception as exc:
                await self._handle_failure(item, exc)

    async def _mark_completed(self, item: QueueItem) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE write_queue SET status = 'completed', completed_at = ?, "
            "attempts = attempts + 1 WHERE id = ?",
            (time.time(), item.id),
        )
        await self._db.commit()

    async def _handle_failure(self, item: QueueItem, exc: Exception) -> None:
        assert self._db is not None
        new_attempts = item.attempts + 1
        error_msg = f'{type(exc).__name__}: {exc}'
        if new_attempts >= item.max_attempts:
            await self._db.execute(
                "UPDATE write_queue SET status = 'dead', attempts = ?, error = ? "
                "WHERE id = ?",
                (new_attempts, error_msg, item.id),
            )
            logger.warning(
                'Item %d dead-lettered after %d attempts: %s',
                item.id, new_attempts, error_msg,
            )
        else:
            delay = min(
                self._retry_base_seconds * (2 ** (new_attempts - 1))
                + random.uniform(0, self._retry_base_seconds),
                self._retry_max_delay_seconds,
            )
            next_retry = time.time() + delay
            await self._db.execute(
                "UPDATE write_queue SET status = 'retry', attempts = ?, "
                "next_retry_at = ?, error = ? WHERE id = ?",
                (new_attempts, next_retry, error_msg, item.id),
            )
            logger.info(
                'Item %d retry %d/%d in %.1fs: %s',
                item.id, new_attempts, item.max_attempts, delay, error_msg,
            )
        await self._db.commit()

    # -- recovery -------------------------------------------------------------

    async def _recover_in_flight(self) -> None:
        """Reset items left in_flight (crashed mid-write) back to pending."""
        assert self._db is not None
        cursor = await self._db.execute(
            "UPDATE write_queue SET status = 'pending' WHERE status = 'in_flight'"
        )
        await self._db.commit()
        if cursor.rowcount:
            logger.info('Recovered %d in-flight items to pending', cursor.rowcount)

    async def _start_workers_for_pending_groups(self) -> None:
        """Start workers for any groups that have pending/retry items."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT DISTINCT group_id FROM write_queue "
            "WHERE status IN ('pending', 'retry')"
        )
        rows = await cursor.fetchall()
        for row in rows:
            group_id = row[0] if isinstance(row, tuple) else row['group_id']
            self._ensure_workers(group_id)
            self._signal_group(group_id)

    # -- management -----------------------------------------------------------

    async def replay_dead(self, group_id: str | None = None) -> int:
        """Reset dead items to pending for retry. Returns count reset."""
        assert self._db is not None
        if group_id:
            cursor = await self._db.execute(
                "UPDATE write_queue SET status = 'pending', attempts = 0, "
                "next_retry_at = 0, error = NULL "
                "WHERE status = 'dead' AND group_id = ?",
                (group_id,),
            )
        else:
            cursor = await self._db.execute(
                "UPDATE write_queue SET status = 'pending', attempts = 0, "
                "next_retry_at = 0, error = NULL "
                "WHERE status = 'dead'",
            )
        await self._db.commit()
        count = cursor.rowcount or 0
        if count:
            if group_id:
                self._ensure_workers(group_id)
                self._signal_group(group_id)
            else:
                await self._start_workers_for_pending_groups()
        return count

    async def get_stats(self) -> dict[str, Any]:
        """Return counts by status and oldest pending age."""
        assert self._db is not None
        cursor = await self._db.execute(
            'SELECT status, COUNT(*) as cnt FROM write_queue GROUP BY status'
        )
        rows = await cursor.fetchall()
        counts = {
            row[0] if isinstance(row, tuple) else row['status']:
            row[1] if isinstance(row, tuple) else row['cnt']
            for row in rows
        }

        oldest_pending_age = None
        cursor = await self._db.execute(
            "SELECT MIN(created_at) FROM write_queue "
            "WHERE status IN ('pending', 'retry')"
        )
        row = await cursor.fetchone()
        if row:
            min_created = row[0] if isinstance(row, tuple) else row[0]
            if min_created is not None:
                oldest_pending_age = time.time() - min_created

        return {
            'counts': counts,
            'oldest_pending_age_seconds': oldest_pending_age,
        }

    async def purge_dead(
        self,
        *,
        group_id: str | None = None,
        error_pattern: str | None = None,
        ids: list[int] | None = None,
        confirm_purge_all: bool = False,
    ) -> int:
        """Permanently delete dead-lettered items. Returns count deleted.

        Requires at least one filter (group_id, error_pattern, or ids) unless
        confirm_purge_all=True is explicitly passed (safety rail against mass-delete).

        Args:
            group_id: Delete only dead items with this group_id.
            error_pattern: Delete only dead items whose error matches this SQL
                LIKE pattern (e.g. 'NodeNotFoundError%').
            ids: Delete only dead items with these specific row ids.
            confirm_purge_all: When True, bypass the filter requirement and
                delete ALL dead items.

        Returns:
            Number of rows deleted.

        Raises:
            ValueError: When no filter is supplied and confirm_purge_all is False.
        """
        assert self._db is not None
        ids = ids or None  # normalise empty list to None so it contributes no filter
        has_filter = group_id is not None or error_pattern is not None or ids is not None
        if not has_filter and not confirm_purge_all:
            raise ValueError(
                'purge_dead requires at least one filter or confirm_purge_all=True'
            )

        sql = "DELETE FROM write_queue WHERE status = 'dead'"
        params: list[Any] = []

        if group_id is not None:
            sql += ' AND group_id = ?'
            params.append(group_id)
        if error_pattern is not None:
            sql += ' AND error LIKE ?'
            params.append(error_pattern)
        if ids is not None:
            placeholders = ','.join('?' * len(ids))
            sql += f' AND id IN ({placeholders})'
            params.extend(ids)

        cursor = await self._db.execute(sql, params)
        await self._db.commit()
        count = cursor.rowcount or 0
        logger.info(
            'purge_dead deleted %d item(s) [group_id=%r, error_pattern=%r, ids=%r]',
            count, group_id, error_pattern, ids,
        )
        return count

    async def get_dead_items(self, group_id: str | None = None) -> list[dict[str, Any]]:
        """Return dead-lettered items."""
        assert self._db is not None
        if group_id:
            cursor = await self._db.execute(
                "SELECT * FROM write_queue WHERE status = 'dead' AND group_id = ?",
                (group_id,),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM write_queue WHERE status = 'dead'"
            )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            item = QueueItem(tuple(row))
            results.append({
                'id': item.id,
                'group_id': item.group_id,
                'operation': item.operation,
                'payload': item.parsed_payload(),
                'attempts': item.attempts,
                'error': item.error,
                'created_at': item.created_at,
            })
        return results
