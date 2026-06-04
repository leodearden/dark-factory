"""In-process SQLite-backed task backend.

Per-project DB file at ``<project_root>/.taskmaster/tasks/tasks.db``.
WAL mode handles concurrent readers natively; mutations are serialised
per project_root by an :class:`asyncio.Lock`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite
from shared.async_sqlite_base import apply_full_durability_pragmas, connect_daemon

from fused_memory.backends.task_backend_errors import TaskmasterError
from fused_memory.backends.task_backend_types import (
    AddTaskResult,
    DependencyResult,
    GetTasksResult,
    RemoveTaskResult,
    SetTaskStatusResult,
    UpdateTaskResult,
    ValidateDependenciesResult,
)
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.models.scope import resolve_project_id

logger = logging.getLogger(__name__)


# ``parent_id = 0`` is a sentinel for top-level tasks in the still-present
# schema column.  All tasks are top-level after DF-D; this constant keeps every
# INSERT/WHERE clause from changing until the schema-drop migration in step-8.
_TOP_LEVEL_SENTINEL = 0

# Per-process dedup set for the malformed-metadata WARNING below.  `_row_to_task`
# is invoked once per row on every `get_tasks` / `get_task` call, so a project
# DB with many corrupted rows would otherwise flood the log with duplicate
# WARNINGs on every read.  Keyed by ``(project_root, tag, id)`` because a single
# SqliteTaskBackend instance services all project_roots, and the default first
# task in every project is ``(master, 1)`` — without project_root in the key, a
# second project DB with the same corrupted row silently swallows its WARN.
# Growth is bounded by the number of distinct (project_root, tag, id) triples
# across all project DBs opened in this process.  No eviction needed; restart
# re-emits.
_warned_malformed_task_ids: set[tuple[str, str, int]] = set()


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    tag           TEXT NOT NULL DEFAULT 'master',
    id            INTEGER NOT NULL,
    parent_id     INTEGER NOT NULL DEFAULT 0,
    title         TEXT NOT NULL,
    description   TEXT,
    details       TEXT,
    test_strategy TEXT,
    status        TEXT NOT NULL,
    priority      TEXT,
    metadata      TEXT,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (tag, parent_id, id)
);

CREATE INDEX IF NOT EXISTS ix_tasks_status ON tasks (tag, status);
CREATE INDEX IF NOT EXISTS ix_tasks_parent ON tasks (tag, parent_id);

CREATE TABLE IF NOT EXISTS dependencies (
    tag        TEXT NOT NULL DEFAULT 'master',
    task_id    INTEGER NOT NULL,
    parent_id  INTEGER NOT NULL DEFAULT 0,
    depends_on INTEGER NOT NULL,
    PRIMARY KEY (tag, parent_id, task_id, depends_on)
);

-- Monotonic id high-water marks per (tag, parent_id) sequence.  ``add_task``
-- allocates ``max(MAX(tasks.id), max_id) + 1`` so a deleted
-- top-level id is NEVER reissued (the counter holds the line after the row is
-- gone).  Applied idempotently via executescript on every connection open; old
-- code that ignores this table still reads/writes ``tasks`` correctly, so the
-- change is backward-compatible.  ``parent_id`` mirrors the tasks sentinel:
-- ``_TOP_LEVEL_SENTINEL`` (0) for the top-level sequence, the parent's int id
-- for a per-parent subtask sequence.
CREATE TABLE IF NOT EXISTS id_counters (
    tag        TEXT NOT NULL DEFAULT 'master',
    parent_id  INTEGER NOT NULL DEFAULT 0,
    max_id     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (tag, parent_id)
);
"""

DEFAULT_TAG = 'master'


def _now() -> str:
    """ISO-8601 UTC timestamp matching the Taskmaster ``updatedAt`` format."""
    return datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%S.') + (
        f'{datetime.now(UTC).microsecond // 1000:03d}Z'
    )


def _parse_task_id(raw: str | int) -> int:
    """Parse ``"292"`` or ``292`` into a bare integer task id.

    Raises ``TaskmasterError`` with code ``INVALID_TASK_ID`` for any dotted
    (e.g. ``"292.1"``) or non-numeric input.
    """
    s = str(raw).strip()
    if not s:
        raise TaskmasterError('INVALID_TASK_ID', f'empty task id: {raw!r}')
    if '.' in s:
        raise TaskmasterError(
            'INVALID_TASK_ID', f'dotted task ids are not supported: {raw!r}',
        )
    try:
        return int(s)
    except ValueError as exc:
        raise TaskmasterError(
            'INVALID_TASK_ID', f'non-numeric task id: {raw!r}',
        ) from exc


def _parse_qualified_dep(depends_on: str) -> tuple[str, int]:
    """Parse a cross-project dependency string of the form ``"project_id:task_id"``.

    Assumes the caller has already detected that ``':'`` is present.
    Strips whitespace, normalises ``'-'`` to ``'_'`` in the project_id portion,
    and rejects any malformed input with a ``TaskmasterError`` whose code is
    ``TASKMASTER_TOOL_ERROR``.

    Returns:
        ``(normalized_project_id, dep_int)`` where ``dep_int > 0``.

    Raises:
        :class:`TaskmasterError` with ``TASKMASTER_TOOL_ERROR`` on any of:
        empty project_id, empty task_id, extra colons, non-numeric task_id,
        dotted (subtask) task_id, or non-positive task_id.
    """
    _MALFORMED = (
        'TASKMASTER_TOOL_ERROR',
        f'add_dependency: malformed cross-project dependency {depends_on!r};'
        ' expected "project_id:task_id"',
    )
    parts = depends_on.split(':')
    if len(parts) != 2:
        raise TaskmasterError(*_MALFORMED)
    raw_pid, raw_tid = parts[0].strip(), parts[1].strip()
    if not raw_pid:
        raise TaskmasterError(*_MALFORMED)
    if not raw_tid:
        raise TaskmasterError(*_MALFORMED)
    # Reject dotted subtask ids (e.g. "5.1") and non-numeric ids.
    if '.' in raw_tid or not raw_tid.lstrip('-').isdigit():
        raise TaskmasterError(*_MALFORMED)
    try:
        dep_int = int(raw_tid)
    except ValueError as err:
        raise TaskmasterError(*_MALFORMED) from err
    if dep_int <= 0:
        raise TaskmasterError(*_MALFORMED)
    norm_pid = raw_pid.lower().replace('-', '_')
    return norm_pid, dep_int


def _format_task_id(task_id: int) -> str:
    return str(task_id)


def _row_to_task(row: aiosqlite.Row, dependencies: list[int], *, project_root: str) -> dict[str, Any]:
    """Convert a tasks-table row into the get_tasks/get_task wire dict.

    All tasks are top-level after DF-D. Ids surface as strings (matches live
    get_tasks wire shape). get_task converts to int after the call.
    """
    metadata_raw = row['metadata']
    metadata: Any = None
    if metadata_raw:
        try:
            metadata = json.loads(metadata_raw)
        except (TypeError, ValueError):
            # Malformed legacy row: discard and surface {} so downstream
            # `(task.get('metadata') or {}).get(...)` callers never see a str.
            # WARN once per (project_root, tag, id) per process so a corrupted-row
            # batch doesn't fan out to one log line per row per get_tasks call.
            dedup_key = (project_root, row['tag'], row['id'])
            if dedup_key not in _warned_malformed_task_ids:
                _warned_malformed_task_ids.add(dedup_key)
                logger.warning(
                    'sqlite_task_backend: malformed metadata JSON — project_root=%s'
                    ' tag=%s id=%s metadata_raw=%s; coerced to {}',
                    project_root,
                    row['tag'],
                    row['id'],
                    repr(metadata_raw)[:80],
                )
            metadata = {}

    return {
        'id': str(row['id']),
        'title': row['title'],
        'description': row['description'] or '',
        'details': row['details'] or '',
        'testStrategy': row['test_strategy'] or '',
        'status': row['status'],
        'dependencies': dependencies,
        'priority': row['priority'] or 'medium',
        'subtasks': [],
        'updatedAt': row['updated_at'],
        'metadata': metadata if metadata is not None else {},
    }


class SqliteTaskBackend:
    """Implements :class:`TaskBackendProtocol` against per-project SQLite files.

    A single backend instance services all projects fused-memory has been
    asked about. Connections are opened lazily on first use of each
    ``project_root`` and kept open for the lifetime of the backend; close()
    drains all of them.
    """

    def __init__(self, config: TaskmasterConfig | None = None) -> None:
        self.config = config
        self._connections: dict[str, aiosqlite.Connection] = {}
        # Guards the connection map AND each project's first-access bring-up
        # (schema + WAL pragmas). Held briefly during open; released before
        # any user-visible call runs.
        self._connect_locks: dict[str, asyncio.Lock] = {}
        self._connect_locks_lock = asyncio.Lock()
        # Per-project write serialisation (mirrors the interceptor's
        # ``_write_lock`` pattern). WAL allows concurrent readers natively.
        self._write_locks: dict[str, asyncio.Lock] = {}
        self._closed = False
        self._started = False
        # SQLite connections don't restart, so the counter is pinned at 1
        # once start() is called (matches "session up" semantics for
        # downstream callers).
        self._restart_count = 0

    # ── Lifecycle ──────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._started and not self._closed

    @property
    def restart_count(self) -> int:
        return self._restart_count

    async def start(self) -> None:
        """No-op connect — connections open lazily on first project access."""
        if self._started:
            return
        self._closed = False
        self._started = True
        self._restart_count = 1
        logger.info('SqliteTaskBackend ready (lazy per-project connections)')

    async def initialize(self) -> None:
        """Alias for :meth:`start` — preserved for back-compat callers."""
        await self.start()

    async def ensure_connected(self) -> None:
        if self._closed:
            raise RuntimeError('SqliteTaskBackend is closed')
        if not self._started:
            await self.start()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._started = False
        # Snapshot under the lock so concurrent open() calls don't observe
        # a half-empty map mid-tear-down.
        async with self._connect_locks_lock:
            connection_items = list(self._connections.items())
            self._connections.clear()
        # Final TRUNCATE checkpoint on each connection so a clean shutdown
        # leaves the WAL empty and the main DB up to date — the prod
        # recovery path on next-open then has nothing to replay. Best-effort:
        # failures here don't block the close.
        for root, conn in connection_items:
            with contextlib.suppress(Exception):
                await conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            with contextlib.suppress(Exception):
                await conn.close()
            logger.debug('SqliteTaskBackend final-checkpointed and closed %s', root)
        logger.info('SqliteTaskBackend closed (%d connection(s))', len(connection_items))

    async def is_alive(self) -> tuple[bool, str | None]:
        if self._closed or not self._started:
            return False, 'not started'
        return True, None

    # ── WAL maintenance ────────────────────────────────────────────────

    async def checkpoint_all(self) -> dict[str, dict[str, int]]:
        """Run ``PRAGMA wal_checkpoint(TRUNCATE)`` on every open project DB.

        Returns ``{project_root: {'busy': int, 'log': int, 'checkpointed': int}}``
        — the three values SQLite reports for each checkpoint. A non-zero
        ``busy`` means active readers/writers blocked the truncate from
        completing fully; the routine still copies what it can but the WAL
        will not shrink.

        Called by the periodic checkpoint task in ``server/main.py`` to
        bound the un-flushed-WAL window and advance the main DB file on a
        known cadence. Independent of the per-project write lock — the
        checkpoint pragma itself is what serialises against writers in
        SQLite.
        """
        results: dict[str, dict[str, int]] = {}
        async with self._connect_locks_lock:
            roots = list(self._connections.keys())
        for root in roots:
            conn = self._connections.get(root)
            if conn is None:
                continue
            try:
                cursor = await conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                row = await cursor.fetchone()
                # PRAGMA wal_checkpoint returns (busy, log, checkpointed):
                # busy=0 means truncate succeeded; busy=1 means readers blocked.
                if row is None:
                    results[root] = {'busy': -1, 'log': -1, 'checkpointed': -1}
                else:
                    results[root] = {
                        'busy': int(row[0]),
                        'log': int(row[1]),
                        'checkpointed': int(row[2]),
                    }
            except Exception as exc:
                logger.warning('checkpoint failed for %s: %s', root, exc)
                results[root] = {'busy': -1, 'log': -1, 'checkpointed': -1}
        return results

    # ── Connection management ──────────────────────────────────────────

    @staticmethod
    def _db_path(project_root: str) -> Path:
        return Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'

    async def _get_connection(self, project_root: str) -> aiosqlite.Connection:
        """Return an open, schema-applied connection for ``project_root``.

        First call for a given project opens the file (creating parent
        directories), applies WAL/busy-timeout pragmas, and runs the schema.
        Subsequent calls reuse the cached connection.
        """
        if self._closed:
            raise RuntimeError('SqliteTaskBackend is closed')
        if project_root in self._connections:
            return self._connections[project_root]

        async with self._connect_locks_lock:
            lock = self._connect_locks.setdefault(project_root, asyncio.Lock())

        async with lock:
            # Re-check after acquiring lock — another caller may have raced us.
            conn = self._connections.get(project_root)
            if conn is not None:
                return conn

            db_path = self._db_path(project_root)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = await connect_daemon(str(db_path))
            conn.row_factory = aiosqlite.Row
            await apply_full_durability_pragmas(conn, busy_timeout_ms=5000)
            await conn.execute('PRAGMA foreign_keys=OFF')
            await conn.executescript(_SCHEMA_SQL)
            await conn.commit()
            self._connections[project_root] = conn
            logger.info('SqliteTaskBackend opened %s', db_path)
            return conn

    def _write_lock(self, project_root: str) -> asyncio.Lock:
        return self._write_locks.setdefault(project_root, asyncio.Lock())

    @contextlib.asynccontextmanager
    async def _txn(self, project_root: str):
        """Explicit transaction wrapper: commit on success, rollback otherwise.

        Cancellation hardening (soak fix):

        * ``commit()`` and ``rollback()`` are wrapped in ``asyncio.shield`` so
          an outer cancellation arriving mid-flush can't tear the transaction
          across the wire and leave the connection in a half-committed
          ``BEGIN`` state.

        * ``contextlib.suppress(BaseException)`` (was ``Exception``) — the
          previous form let ``CancelledError`` (a ``BaseException``, not an
          ``Exception``) escape past the rollback, so cancellation-during-
          rollback could leave the connection mid-transaction *and* take the
          original exception with it.
        """
        conn = await self._get_connection(project_root)
        try:
            yield conn
            await asyncio.shield(conn.commit())
        except BaseException:
            with contextlib.suppress(BaseException):
                await asyncio.shield(conn.rollback())
            raise

    # ── Read helpers ───────────────────────────────────────────────────

    async def _fetch_dependencies(
        self, conn: aiosqlite.Connection, tag: str,
    ) -> dict[int, list[int]]:
        """Return ``{task_id: [depends_on, ...]}`` for *tag*."""
        cursor = await conn.execute(
            'SELECT task_id, depends_on FROM dependencies WHERE tag = ?',
            (tag,),
        )
        rows = await cursor.fetchall()
        out: dict[int, list[int]] = {}
        for row in rows:
            out.setdefault(row['task_id'], []).append(row['depends_on'])
        for deps in out.values():
            deps.sort()
        return out

    async def _get_tasks_internal(
        self, project_root: str, tag: str,
    ) -> list[dict[str, Any]]:
        conn = await self._get_connection(project_root)
        cursor = await conn.execute(
            'SELECT * FROM tasks WHERE tag = ? AND parent_id = ? ORDER BY id',
            (tag, _TOP_LEVEL_SENTINEL),
        )
        rows = await cursor.fetchall()
        deps = await self._fetch_dependencies(conn, tag)
        return [
            _row_to_task(row, deps.get(row['id'], []), project_root=project_root)
            for row in rows
        ]

    # ── Public surface ─────────────────────────────────────────────────

    async def get_tasks(
        self, project_root: str, tag: str | None = None,
    ) -> GetTasksResult:
        await self.ensure_connected()
        tag = tag or DEFAULT_TAG
        tasks = await self._get_tasks_internal(project_root, tag)
        return {'tasks': tasks}

    async def get_task(
        self, task_id: str, project_root: str, tag: str | None = None,
    ) -> dict:
        await self.ensure_connected()
        tag = tag or DEFAULT_TAG
        tid = _parse_task_id(task_id)
        conn = await self._get_connection(project_root)

        cursor = await conn.execute(
            'SELECT * FROM tasks WHERE tag = ? AND id = ? AND parent_id = ?',
            (tag, tid, _TOP_LEVEL_SENTINEL),
        )
        row = await cursor.fetchone()
        if row is None:
            raise TaskmasterError(
                'TASKMASTER_TOOL_ERROR', f'No tasks found for ID(s): {task_id}',
            )
        deps = await self._fetch_dependencies(conn, tag)

        out = _row_to_task(row, deps.get(row['id'], []), project_root=project_root)
        # get_task surfaces a single task — Taskmaster returns int id here
        # (asymmetric with get_tasks; mirror that quirk to keep wire-compat).
        with contextlib.suppress(TypeError, ValueError):
            out['id'] = int(out['id'])
        return out

    async def set_task_status(
        self,
        task_id: str,
        status: str,
        project_root: str,
        tag: str | None = None,
    ) -> SetTaskStatusResult:
        await self.ensure_connected()
        tag = tag or DEFAULT_TAG
        tid = _parse_task_id(task_id)
        async with self._write_lock(project_root), self._txn(project_root) as conn:
            cursor = await conn.execute(
                'SELECT status FROM tasks WHERE tag = ? AND id = ? AND parent_id = ?',
                (tag, tid, _TOP_LEVEL_SENTINEL),
            )
            row = await cursor.fetchone()
            if row is None:
                raise TaskmasterError(
                    'TASKMASTER_TOOL_ERROR',
                    f'No tasks found for ID(s): {task_id}',
                )
            old_status = row['status']
            await conn.execute(
                'UPDATE tasks SET status = ?, updated_at = ? '
                'WHERE tag = ? AND id = ? AND parent_id = ?',
                (status, _now(), tag, tid, _TOP_LEVEL_SENTINEL),
            )
        return {
            'message': f'Successfully updated 1 task(s) to "{status}"',
            'tasks': [{
                'taskId': task_id,
                'oldStatus': old_status,
                'newStatus': status,
            }],
        }

    async def add_task(
        self,
        project_root: str,
        prompt: str | None = None,
        title: str | None = None,
        description: str | None = None,
        details: str | None = None,
        dependencies: str | None = None,
        priority: str | None = None,
        metadata: str | None = None,
        tag: str | None = None,
        status: str = 'pending',
    ) -> AddTaskResult:
        await self.ensure_connected()
        tag = tag or DEFAULT_TAG

        # SqliteTaskBackend has no LLM — derive title/description from prompt
        # when the caller only supplied a prompt. The first non-empty line is
        # the title; the full text is the description.
        if not title and prompt:
            for line in prompt.splitlines():
                stripped = line.strip()
                if stripped:
                    title = stripped[:200]
                    break
            description = description or prompt
        if not title:
            raise TaskmasterError(
                'TASKMASTER_TOOL_ERROR',
                'Either the prompt parameter or both title and description are required',
            )

        deps_list = _parse_dependency_list(dependencies)

        async with self._write_lock(project_root), self._txn(project_root) as conn:
            # High-water across BOTH live rows and the persisted counter, so a
            # deleted top-level id is never reissued (see id_counters in the
            # schema).  max(MAX(tasks.id), stored)+1 self-heals legacy DBs that
            # predate the counter: the existing row high-water is honoured on
            # the first post-upgrade alloc, then the counter holds the line.
            cursor = await conn.execute(
                """
                SELECT MAX(highwater) FROM (
                    SELECT COALESCE(MAX(id), 0) AS highwater FROM tasks
                        WHERE tag = ? AND parent_id = ?
                    UNION ALL
                    SELECT COALESCE(max_id, 0) AS highwater FROM id_counters
                        WHERE tag = ? AND parent_id = ?
                )
                """,
                (tag, _TOP_LEVEL_SENTINEL, tag, _TOP_LEVEL_SENTINEL),
            )
            _max_row = await cursor.fetchone()
            assert _max_row is not None  # aggregate MAX always returns one row
            next_id = (_max_row[0] or 0) + 1
            await conn.execute(
                """
                    INSERT INTO tasks (tag, id, parent_id, title, description,
                                       details, test_strategy, status, priority,
                                       metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?)
                    """,
                (
                    tag, next_id, _TOP_LEVEL_SENTINEL, title,
                    description or '', details or '',
                    status, priority or 'medium', metadata, _now(),
                ),
            )
            for dep in deps_list:
                await conn.execute(
                    'INSERT OR IGNORE INTO dependencies '
                    '(tag, task_id, parent_id, depends_on) VALUES (?, ?, ?, ?)',
                    (tag, next_id, _TOP_LEVEL_SENTINEL, dep),
                )
            await conn.execute(
                """
                INSERT INTO id_counters (tag, parent_id, max_id) VALUES (?, ?, ?)
                    ON CONFLICT(tag, parent_id) DO UPDATE SET max_id = excluded.max_id
                        WHERE excluded.max_id > id_counters.max_id
                """,
                (tag, _TOP_LEVEL_SENTINEL, next_id),
            )
        return {
            'id': str(next_id),
            'message': f'Successfully added new task #{next_id}',
        }

    async def update_task(
        self,
        task_id: str,
        project_root: str,
        prompt: str | None = None,
        metadata: str | None = None,
        append: bool = False,
        tag: str | None = None,
        *,
        title: str | None = None,
        description: str | None = None,
        details: str | None = None,
        priority: str | None = None,
        status: str | None = None,
        dependencies: list[str] | None = None,
    ) -> UpdateTaskResult:
        # Structured fields (title/description/details/priority/status/dependencies)
        # land deterministically — any non-None value overrides the current row.
        # ``prompt`` is kept for backward compatibility: when no explicit
        # ``details`` is passed it feeds the details path (replace, or append
        # when ``append=True``). ``metadata`` retains the merge-or-replace
        # semantics keyed off ``append``.
        await self.ensure_connected()
        tag = tag or DEFAULT_TAG
        tid = _parse_task_id(task_id)

        async with self._write_lock(project_root), self._txn(project_root) as conn:
            cursor = await conn.execute(
                'SELECT * FROM tasks WHERE tag = ? AND id = ? AND parent_id = ?',
                (tag, tid, _TOP_LEVEL_SENTINEL),
            )
            row = await cursor.fetchone()
            if row is None:
                raise TaskmasterError(
                    'TASKMASTER_TOOL_ERROR',
                    f'No tasks found for ID(s): {task_id}',
                )

            # Build the SET clause from non-None structured fields plus
            # the prompt-derived details path.
            set_columns: list[str] = []
            set_values: list[Any] = []

            if title is not None:
                set_columns.append('title = ?')
                set_values.append(title)
            if description is not None:
                set_columns.append('description = ?')
                set_values.append(description)
            if priority is not None:
                set_columns.append('priority = ?')
                set_values.append(priority)
            if status is not None:
                set_columns.append('status = ?')
                set_values.append(status)

            # details: explicit param wins over prompt. Both honor ``append``.
            existing_details = row['details'] or ''
            if details is not None:
                new_details = (
                    f'{existing_details}\n\n{details}'
                    if (append and existing_details) else details
                )
                set_columns.append('details = ?')
                set_values.append(new_details)
            elif prompt is not None:
                new_details = (
                    f'{existing_details}\n\n{prompt}'
                    if (append and existing_details) else prompt
                )
                set_columns.append('details = ?')
                set_values.append(new_details)

            if metadata is not None:
                new_metadata = _merge_metadata(
                    row['metadata'], metadata, append=append,
                )
                set_columns.append('metadata = ?')
                set_values.append(new_metadata)

            # updated_at always advances, even on a no-op write — matches
            # the original behaviour and avoids surprising "stale" reads.
            set_columns.append('updated_at = ?')
            set_values.append(_now())

            set_clause = ', '.join(set_columns)
            set_values.extend([tag, tid, _TOP_LEVEL_SENTINEL])
            await conn.execute(
                f'UPDATE tasks SET {set_clause} '
                f'WHERE tag = ? AND id = ? AND parent_id = ?',
                set_values,
            )

            # Dependencies: replace-mode only. Empty list clears all deps.
            if dependencies is not None:
                parsed_deps: list[int] = [_parse_task_id(raw) for raw in dependencies]
                await conn.execute(
                    'DELETE FROM dependencies WHERE tag = ? AND task_id = ? AND parent_id = ?',
                    (tag, tid, _TOP_LEVEL_SENTINEL),
                )
                for dep in parsed_deps:
                    await conn.execute(
                        'INSERT OR IGNORE INTO dependencies '
                        '(tag, task_id, parent_id, depends_on) VALUES (?, ?, ?, ?)',
                        (tag, tid, _TOP_LEVEL_SENTINEL, dep),
                    )

            refreshed_cursor = await conn.execute(
                'SELECT * FROM tasks WHERE tag = ? AND id = ? AND parent_id = ?',
                (tag, tid, _TOP_LEVEL_SENTINEL),
            )
            refreshed = await refreshed_cursor.fetchone()
        deps = (
            await self._fetch_dependencies(
                await self._get_connection(project_root), tag,
            )
        )
        updated_task = (
            _row_to_task(refreshed, deps.get(refreshed['id'], []), project_root=project_root)
            if refreshed is not None else None
        )
        return {
            'id': task_id,
            'message': f'Task {task_id} updated',
            'updated': True,
            'updated_task': updated_task,
        }

    async def remove_tasks(
        self,
        ids: list[str],
        project_root: str,
        tag: str | None = None,
    ) -> RemoveTaskResult:
        await self.ensure_connected()
        tag = tag or DEFAULT_TAG

        if not ids:
            return {
                'successful': 0,
                'failed': 0,
                'removed_ids': [],
                'message': 'no ids supplied',
            }

        # Parse all upfront — a single bad id fails the whole batch (caller
        # sent garbage; no partial-success on malformed input).
        parsed: list[int] = [_parse_task_id(raw) for raw in ids]

        async with self._write_lock(project_root), self._txn(project_root) as conn:
            # One SELECT to identify which requested rows exist.
            id_placeholders = ','.join('?' for _ in parsed)
            cursor = await conn.execute(
                f'SELECT id FROM tasks WHERE tag = ? AND parent_id = ? AND id IN ({id_placeholders})',
                [tag, _TOP_LEVEL_SENTINEL, *parsed],
            )
            existing_ids: set[int] = {row['id'] for row in await cursor.fetchall()}

            # Classify into existing (to remove) vs missing.
            removed_ids: list[int] = []
            removed_display: list[str] = []
            failed_display: list[str] = []
            seen_removed: set[int] = set()
            seen_failed: set[str] = set()

            for tid in parsed:
                disp = _format_task_id(tid)
                if tid not in existing_ids:
                    if disp not in seen_failed:
                        failed_display.append(disp)
                        seen_failed.add(disp)
                    continue
                if tid not in seen_removed:
                    removed_ids.append(tid)
                    removed_display.append(disp)
                    seen_removed.add(tid)

            # Two batch DELETEs — tasks then their owning dependencies.
            # Cross-task deps pointing AT removed ids stay dangling on
            # purpose (lets validate_dependencies surface them).
            if removed_ids:
                rm_placeholders = ','.join('?' for _ in removed_ids)
                await conn.execute(
                    f'DELETE FROM tasks WHERE tag = ? AND parent_id = ? AND id IN ({rm_placeholders})',
                    [tag, _TOP_LEVEL_SENTINEL, *removed_ids],
                )
                await conn.execute(
                    f'DELETE FROM dependencies WHERE tag = ? AND parent_id = ? AND task_id IN ({rm_placeholders})',
                    [tag, _TOP_LEVEL_SENTINEL, *removed_ids],
                )

        successful = len(removed_display)
        failed = len(failed_display)
        if failed_display:
            msg = (
                f'Removed {successful} task(s); '
                f'{failed} not found: {", ".join(failed_display)}'
            )
        else:
            msg = f'Removed {successful} task(s)'
        return {
            'successful': successful,
            'failed': failed,
            'removed_ids': removed_display,
            'message': msg,
        }

    async def add_dependency(
        self,
        task_id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ) -> DependencyResult:
        """Add a dependency to a task.

        ``depends_on`` accepts two forms:

        * **Bare integer** (e.g. ``"3"``): the traditional integer-table path.
          Both the dependent task and the target must exist in this project.
        * **Qualified** (e.g. ``"dark_factory:13"``): cross-project dependency
          stored in the dependent task's ``metadata.external_deps`` list.
          The foreign target is **not** verified at write time (lenient write —
          the target may be filed later or live in another project).
          ``'-'`` in the project_id portion is normalised to ``'_'`` before
          storing.
        """
        await self.ensure_connected()
        tag = tag or DEFAULT_TAG

        # ── Qualified (cross-project) path ─────────────────────────────────
        if ':' in str(depends_on):
            tid = _parse_task_id(task_id)
            norm_pid, dep_int = _parse_qualified_dep(depends_on)
            canonical = f'{norm_pid}:{dep_int}'

            # Self-check: reject when both project_id and task_id match this task.
            if norm_pid == resolve_project_id(project_root) and dep_int == tid:
                raise TaskmasterError(
                    'TASKMASTER_TOOL_ERROR',
                    'add_dependency: task cannot depend on itself',
                )

            async with self._write_lock(project_root), self._txn(project_root) as conn:
                cursor = await conn.execute(
                    'SELECT metadata FROM tasks WHERE tag = ? AND id = ? AND parent_id = ?',
                    (tag, tid, _TOP_LEVEL_SENTINEL),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise TaskmasterError(
                        'TASKMASTER_TOOL_ERROR',
                        f'No tasks found for ID(s): {tid}',
                    )
                new_meta = _merge_metadata(
                    row['metadata'],
                    json.dumps({'external_deps': [canonical]}),
                    append=True,
                )
                await conn.execute(
                    'UPDATE tasks SET metadata = ?, updated_at = ? '
                    'WHERE tag = ? AND id = ? AND parent_id = ?',
                    (new_meta, _now(), tag, tid, _TOP_LEVEL_SENTINEL),
                )
            return {
                'id': str(tid),
                'dependency_id': canonical,
                'message': f'Added external dependency: {tid} now depends on {canonical}',
            }

        # ── Bare-integer (same-project) path ───────────────────────────────
        tid = _parse_task_id(task_id)
        dep_tid = _parse_task_id(depends_on)

        async with self._write_lock(project_root), self._txn(project_root) as conn:
            # Verify both endpoints exist before inserting.
            for tid_check in (tid, dep_tid):
                cursor = await conn.execute(
                    'SELECT id FROM tasks WHERE tag = ? AND id = ? AND parent_id = ?',
                    (tag, tid_check, _TOP_LEVEL_SENTINEL),
                )
                if (await cursor.fetchone()) is None:
                    raise TaskmasterError(
                        'TASKMASTER_TOOL_ERROR',
                        f'No tasks found for ID(s): {tid_check}',
                    )
            if tid == dep_tid:
                raise TaskmasterError(
                    'TASKMASTER_TOOL_ERROR',
                    'add_dependency: task cannot depend on itself',
                )
            await conn.execute(
                'INSERT OR IGNORE INTO dependencies '
                '(tag, task_id, parent_id, depends_on) VALUES (?, ?, ?, ?)',
                (tag, tid, _TOP_LEVEL_SENTINEL, dep_tid),
            )
        return {
            'id': str(tid),
            'dependency_id': str(dep_tid),
            'message': f'Added dependency: {tid} now depends on {dep_tid}',
        }

    async def remove_dependency(
        self,
        task_id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ) -> DependencyResult:
        """Remove a dependency from a task.

        ``depends_on`` accepts two forms:

        * **Bare integer** (e.g. ``"3"``): deletes from the integer dependencies
          table. Idempotent — no error if the dependency is absent.
        * **Qualified** (e.g. ``"dark_factory:13"``): removes the canonical entry
          from ``metadata.external_deps`` via an atomic read-modify-write inside
          the backend's ``_txn``. Idempotent — no error if the row or dep is absent.
          ``'-'`` in the project_id portion is normalised to ``'_'``.
        """
        await self.ensure_connected()
        tag = tag or DEFAULT_TAG

        # ── Qualified (cross-project) path ─────────────────────────────────
        if ':' in str(depends_on):
            norm_pid, dep_int = _parse_qualified_dep(depends_on)
            canonical = f'{norm_pid}:{dep_int}'
            tid = _parse_task_id(task_id)

            async with self._write_lock(project_root), self._txn(project_root) as conn:
                cursor = await conn.execute(
                    'SELECT metadata FROM tasks WHERE tag = ? AND id = ? AND parent_id = ?',
                    (tag, tid, _TOP_LEVEL_SENTINEL),
                )
                row = await cursor.fetchone()
                if row is not None:
                    try:
                        meta = json.loads(row['metadata'] or '{}')
                    except (TypeError, ValueError):
                        meta = {}
                    existing = meta.get('external_deps', [])
                    if canonical in existing:
                        updated = [e for e in existing if e != canonical]
                        meta['external_deps'] = updated
                        await conn.execute(
                            'UPDATE tasks SET metadata = ?, updated_at = ? '
                            'WHERE tag = ? AND id = ? AND parent_id = ?',
                            (json.dumps(meta), _now(), tag, tid, _TOP_LEVEL_SENTINEL),
                        )
            return {
                'id': str(tid),
                'dependency_id': canonical,
                'message': f'Removed external dependency: {tid} no longer depends on {canonical}',
            }

        # ── Bare-integer (same-project) path ───────────────────────────────
        tid = _parse_task_id(task_id)
        dep_tid = _parse_task_id(depends_on)
        async with self._write_lock(project_root), self._txn(project_root) as conn:
            await conn.execute(
                'DELETE FROM dependencies WHERE tag = ? AND task_id = ? '
                'AND parent_id = ? AND depends_on = ?',
                (tag, tid, _TOP_LEVEL_SENTINEL, dep_tid),
            )
        return {
            'id': str(tid),
            'dependency_id': str(dep_tid),
            'message': f'Removed dependency: {tid} no longer depends on {dep_tid}',
        }

    async def validate_dependencies(
        self, project_root: str, tag: str | None = None,
    ) -> ValidateDependenciesResult:
        await self.ensure_connected()
        tag = tag or DEFAULT_TAG
        conn = await self._get_connection(project_root)
        # Detect any dependency whose target doesn't exist; surface as a
        # message line per dangling reference. Mirrors Taskmaster's "OK"-or-
        # "list of issues" message-only DTO.
        cursor = await conn.execute(
            """
            SELECT d.task_id, d.depends_on
            FROM dependencies d
            LEFT JOIN tasks t ON t.tag = d.tag AND t.id = d.depends_on
                                 AND t.parent_id = ?
            WHERE d.tag = ? AND t.id IS NULL
            """,
            (_TOP_LEVEL_SENTINEL, tag),
        )
        dangling = await cursor.fetchall()
        if not dangling:
            return {'message': 'Dependencies validated successfully'}
        parts = '; '.join(f'{r["task_id"]} -> {r["depends_on"]}' for r in dangling)
        return {'message': f'Dangling dependencies: {parts}'}


def _parse_dependency_list(raw: str | None) -> list[int]:
    """Accept Taskmaster's ``"1,2,3"`` comma-string and return ``[1,2,3]``."""
    if not raw:
        return []
    out: list[int] = []
    for part in raw.split(','):
        s = part.strip()
        if not s:
            continue
        try:
            out.append(int(s))
        except ValueError:
            raise TaskmasterError(
                'TASKMASTER_TOOL_ERROR',
                f'add_task: non-numeric dependency id in {raw!r}',
            ) from None
    return out


def _merge_values(old: object, new: object) -> object:
    """Recursively merge *new* into *old* using additive semantics.

    Rules (applied depth-first):
    * Both values are **lists** — concatenate, then deduplicate hashable items
      in stable old-then-new order.  If any item is unhashable, fall back to
      plain concatenation (no dedup).
    * Both values are **dicts** — recurse over the union of keys; keys present
      in only one side pass through unchanged; collisions recurse.
    * All other cases (scalar collision, type mismatch) — **OLD wins**,
      preserving the audit-field protection of the original implementation.
    """
    if isinstance(old, list) and isinstance(new, list):
        combined = old + new
        try:
            seen: set = set()
            deduped = []
            for item in combined:
                if item not in seen:
                    seen.add(item)
                    deduped.append(item)
            return deduped
        except TypeError:
            return combined
    if isinstance(old, dict) and isinstance(new, dict):
        merged: dict = {}
        for key in old.keys() | new.keys():
            if key in old and key in new:
                merged[key] = _merge_values(old[key], new[key])
            elif key in old:
                merged[key] = old[key]
            else:
                merged[key] = new[key]
        return merged
    # Scalar collision or type mismatch — OLD wins.
    return old


def _normalize_legacy_memory_hints_value(value: object) -> object:
    """Coerce a legacy list-of-dicts ``memory_hints`` value to canonical dict shape.

    The canonical shape is ``{"entities": [...], "queries": [...]}``.
    The legacy shape is ``[{"entity": <str>, "query": <str>}, ...]``.

    When *value* is a list each element that is a dict is inspected:
    * ``entity`` is appended to ``entities`` only when it is a non-empty string.
    * ``query`` is appended to ``queries`` only when it is a non-empty string.
    * Non-dict items and entries with missing / empty / non-string keys are skipped.

    Any non-list input (dict, scalar, None) is returned unchanged — the function is
    a no-op for already-canonical values and for any shape it does not recognise.
    """
    if not isinstance(value, list):
        return value
    entities: list[str] = []
    queries: list[str] = []
    seen_entities: set[str] = set()
    seen_queries: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        entity = item.get("entity")
        if isinstance(entity, str) and entity and entity not in seen_entities:
            entities.append(entity)
            seen_entities.add(entity)
        query = item.get("query")
        if isinstance(query, str) and query and query not in seen_queries:
            queries.append(query)
            seen_queries.add(query)
    return {"entities": entities, "queries": queries}


def _merge_metadata(existing_raw: str | None, incoming: str, *, append: bool) -> str:
    """Merge ``incoming`` metadata JSON into ``existing_raw``.

    When ``append=True`` the merge is additive and recursive:
    * List-valued keys are concatenated and deduplicated (hashable items only)
      in stable old-then-new order.
    * Dict-valued keys are merged recursively with the same rules.
    * Scalar collisions and type-mismatched collisions resolve to OLD-wins,
      preserving audit fields such as ``prd`` and ``spawned_from``.

    When ``append=False`` (or when ``existing_raw`` is ``None``) the incoming
    value replaces whatever was there — last-write-wins.

    If either side fails to JSON-decode, the new value replaces the old
    verbatim — matches Taskmaster's "last write wins" fallback.

    Special case — ``memory_hints``:  when **both** sides carry a
    ``memory_hints`` key and either side's value is the legacy list-of-dicts
    shape (``[{"entity": ..., "query": ...}, ...]``), that side is normalised to
    the canonical dict shape before ``_merge_values`` runs, so the merge falls
    into the dict-vs-dict recursive union path rather than the type-mismatch
    OLD-wins path.  Normalization is scoped to the merge-collision path: a write
    that does not include ``memory_hints`` leaves a stored legacy-shape value
    unchanged.  All other keys are unaffected.
    """
    if existing_raw is None or not append:
        return incoming
    try:
        old = json.loads(existing_raw)
        new = json.loads(incoming)
    except (TypeError, ValueError):
        return incoming
    if not isinstance(old, dict) or not isinstance(new, dict):
        return incoming
    # Normalize legacy memory_hints list shape on both sides before merging.
    # Only fires when BOTH sides carry memory_hints — i.e. on the merge collision
    # path — so a write that omits memory_hints does not silently migrate the
    # stored legacy shape.
    if "memory_hints" in old and "memory_hints" in new:
        old = {**old, "memory_hints": _normalize_legacy_memory_hints_value(old["memory_hints"])}
        new = {**new, "memory_hints": _normalize_legacy_memory_hints_value(new["memory_hints"])}
    try:
        merged = _merge_values(old, new)
    except RecursionError:
        # Pathologically deep metadata; fall back to last-write-wins.
        return incoming
    return json.dumps(merged)
