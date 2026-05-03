"""In-process SQLite-backed task backend (drop-in for ``TaskmasterBackend``).

Per-project DB file at ``<project_root>/.taskmaster/tasks/tasks.db``.
Mirrors the existing tasks.json layout — rollback is "stop using db, the
JSON file is still there." WAL mode handles concurrent readers natively;
mutations are serialised per project_root by an :class:`asyncio.Lock`.

Wire shapes are kept identical to the Taskmaster MCP wrapper outputs in
:mod:`fused_memory.backends.taskmaster_client` so the
:class:`fused_memory.backends.dual_compare_backend.DualCompareBackend`
soak runs cleanly during cutover.

Subtasks live as their own rows with ``parent_id`` set to their parent's
top-level id; the dotted display form (``"292.1"``) is composed at read
time and parsed on write.
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

from fused_memory.backends.taskmaster_types import (
    AddSubtaskResult,
    AddTaskResult,
    DependencyResult,
    GetTasksResult,
    RemoveTaskResult,
    SetTaskStatusResult,
    TaskmasterError,
    UpdateTaskResult,
    ValidateDependenciesResult,
)
from fused_memory.config.schema import TaskmasterConfig

logger = logging.getLogger(__name__)


# ``parent_id = 0`` is a sentinel for top-level tasks; subtask rows store the
# parent's int id. Avoiding NULL keeps the PRIMARY KEY simple — SQLite cannot
# use COALESCE(...) inside a PRIMARY KEY column list, and NULLs in a UNIQUE
# index are treated as distinct, which would let duplicate top-levels slip in.
_TOP_LEVEL_SENTINEL = 0


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
"""

DEFAULT_TAG = 'master'


def _now() -> str:
    """ISO-8601 UTC timestamp matching the Taskmaster ``updatedAt`` format."""
    return datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%S.') + (
        f'{datetime.now(UTC).microsecond // 1000:03d}Z'
    )


def _parse_task_id(raw: str | int) -> tuple[int, int | None]:
    """Parse ``"292"`` or ``"292.1"`` into ``(id, parent_id)``.

    Top-level ids return ``parent_id=None``. Raises ``TaskmasterError`` with
    code ``INVALID_TASK_ID`` when the input is not a parseable id —
    matching how Taskmaster surfaces malformed ids.
    """
    s = str(raw).strip()
    if not s:
        raise TaskmasterError('INVALID_TASK_ID', f'empty task id: {raw!r}')
    if '.' in s:
        parent_str, child_str = s.split('.', 1)
        if '.' in child_str:
            raise TaskmasterError(
                'INVALID_TASK_ID', f'nested subtask ids not supported: {raw!r}',
            )
        try:
            return int(child_str), int(parent_str)
        except ValueError as exc:
            raise TaskmasterError(
                'INVALID_TASK_ID', f'non-numeric task id components: {raw!r}',
            ) from exc
    try:
        return int(s), None
    except ValueError as exc:
        raise TaskmasterError(
            'INVALID_TASK_ID', f'non-numeric task id: {raw!r}',
        ) from exc


def _format_task_id(task_id: int, parent_id: int | None) -> str:
    return f'{parent_id}.{task_id}' if parent_id is not None else str(task_id)


def _row_to_task(row: aiosqlite.Row, dependencies: list[int]) -> dict[str, Any]:
    """Convert a tasks-table row into the get_tasks/get_task wire dict.

    Top-level tasks emit string ``id`` ("292") and an empty ``subtasks``
    list (filled in later by ``_get_tasks_internal``). Subtasks emit a
    short integer ``id`` plus ``parentTaskId`` mirroring Taskmaster's
    actual file layout — see ``project_root/.taskmaster/tasks/tasks.json``
    for an example.
    """
    parent_id_db = row['parent_id']
    parent_id: int | None = parent_id_db if parent_id_db != _TOP_LEVEL_SENTINEL else None
    metadata_raw = row['metadata']
    metadata: Any = None
    if metadata_raw:
        try:
            metadata = json.loads(metadata_raw)
        except (TypeError, ValueError):
            metadata = metadata_raw

    if parent_id is None:
        # Top-level: ids surface as strings (matches live get_tasks wire shape
        # — see test_get_tasks_returns_flat_dto in test_taskmaster_client_contract.py).
        out: dict[str, Any] = {
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
        return out

    # Subtask: short integer id + parentTaskId, no testStrategy/priority/metadata
    # in the Taskmaster file format.
    return {
        'id': row['id'],
        'title': row['title'],
        'description': row['description'] or '',
        'details': row['details'] or '',
        'status': row['status'],
        'dependencies': dependencies,
        'parentTaskId': parent_id,
        'parentId': 'undefined',
        'updatedAt': row['updated_at'],
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
        # Keep wire-compatible with TaskmasterBackend's restart-count surface;
        # SQLite connections don't restart, so it's pinned at 1 once start()
        # is called (matches "session up" semantics for downstream callers).
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
            connections = list(self._connections.values())
            self._connections.clear()
        for conn in connections:
            with contextlib.suppress(Exception):
                await conn.close()
        logger.info('SqliteTaskBackend closed (%d connection(s))', len(connections))

    async def is_alive(self) -> tuple[bool, str | None]:
        if self._closed or not self._started:
            return False, 'not started'
        return True, None

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
            conn = await aiosqlite.connect(str(db_path))
            conn.row_factory = aiosqlite.Row
            await conn.execute('PRAGMA journal_mode=WAL')
            await conn.execute('PRAGMA busy_timeout=5000')
            await conn.execute('PRAGMA synchronous=NORMAL')
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
    ) -> dict[tuple[int, int], list[int]]:
        """Return ``{(task_id, parent_id_or_0): [depends_on, ...]}`` for *tag*."""
        cursor = await conn.execute(
            'SELECT task_id, parent_id, depends_on FROM dependencies WHERE tag = ?',
            (tag,),
        )
        rows = await cursor.fetchall()
        out: dict[tuple[int, int], list[int]] = {}
        for row in rows:
            key = (row['task_id'], row['parent_id'])
            out.setdefault(key, []).append(row['depends_on'])
        for deps in out.values():
            deps.sort()
        return out

    async def _get_tasks_internal(
        self, project_root: str, tag: str,
    ) -> list[dict[str, Any]]:
        conn = await self._get_connection(project_root)
        # Order: top-levels first (parent_id=0), then subtasks. Within each,
        # by id ascending — matches Taskmaster's file ordering.
        cursor = await conn.execute(
            'SELECT * FROM tasks WHERE tag = ? ORDER BY '
            'CASE WHEN parent_id = ? THEN id ELSE parent_id END, parent_id, id',
            (tag, _TOP_LEVEL_SENTINEL),
        )
        rows = await cursor.fetchall()
        deps = await self._fetch_dependencies(conn, tag)

        # Build top-levels first, then attach subtasks under each.
        top_by_id: dict[int, dict[str, Any]] = {}
        for row in rows:
            if row['parent_id'] == _TOP_LEVEL_SENTINEL:
                key = (row['id'], _TOP_LEVEL_SENTINEL)
                top_by_id[row['id']] = _row_to_task(row, deps.get(key, []))

        for row in rows:
            if row['parent_id'] != _TOP_LEVEL_SENTINEL:
                key = (row['id'], row['parent_id'])
                parent = top_by_id.get(row['parent_id'])
                if parent is None:
                    # Orphan subtask — surface as top-level so it isn't lost.
                    top_by_id[-row['parent_id']] = _row_to_task(
                        row, deps.get(key, []),
                    )
                else:
                    parent['subtasks'].append(_row_to_task(row, deps.get(key, [])))

        return [top_by_id[k] for k in sorted(top_by_id)]

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
        tid, parent_id = _parse_task_id(task_id)
        parent_db = parent_id if parent_id is not None else _TOP_LEVEL_SENTINEL
        conn = await self._get_connection(project_root)

        cursor = await conn.execute(
            'SELECT * FROM tasks WHERE tag = ? AND id = ? AND parent_id = ?',
            (tag, tid, parent_db),
        )
        row = await cursor.fetchone()
        if row is None:
            raise TaskmasterError(
                'TASKMASTER_TOOL_ERROR', f'No tasks found for ID(s): {task_id}',
            )
        deps = await self._fetch_dependencies(conn, tag)

        out = _row_to_task(row, deps.get((row['id'], row['parent_id']), []))
        # get_task surfaces a single task — Taskmaster returns int id here
        # (asymmetric with get_tasks; mirror that quirk to keep wire-compat).
        if parent_id is None:
            try:
                out['id'] = int(out['id'])
            except (TypeError, ValueError):
                pass
            # Walk subtasks under this top-level for completeness.
            subtask_cursor = await conn.execute(
                'SELECT * FROM tasks WHERE tag = ? AND parent_id = ? ORDER BY id',
                (tag, tid),
            )
            sub_rows = await subtask_cursor.fetchall()
            out['subtasks'] = [
                _row_to_task(r, deps.get((r['id'], r['parent_id']), []))
                for r in sub_rows
            ]
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
        tid, parent_id = _parse_task_id(task_id)
        parent_db = parent_id if parent_id is not None else _TOP_LEVEL_SENTINEL
        async with self._write_lock(project_root):
            async with self._txn(project_root) as conn:
                cursor = await conn.execute(
                    'SELECT status FROM tasks WHERE tag = ? AND id = ? AND parent_id = ?',
                    (tag, tid, parent_db),
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
                    (status, _now(), tag, tid, parent_db),
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

        async with self._write_lock(project_root):
            async with self._txn(project_root) as conn:
                cursor = await conn.execute(
                    'SELECT COALESCE(MAX(id), 0) FROM tasks WHERE tag = ? AND parent_id = ?',
                    (tag, _TOP_LEVEL_SENTINEL),
                )
                next_id = (await cursor.fetchone())[0] + 1
                await conn.execute(
                    """
                    INSERT INTO tasks (tag, id, parent_id, title, description,
                                       details, test_strategy, status, priority,
                                       metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, '', 'pending', ?, ?, ?)
                    """,
                    (
                        tag, next_id, _TOP_LEVEL_SENTINEL, title,
                        description or '', details or '',
                        priority or 'medium', metadata, _now(),
                    ),
                )
                for dep in deps_list:
                    await conn.execute(
                        'INSERT OR IGNORE INTO dependencies '
                        '(tag, task_id, parent_id, depends_on) VALUES (?, ?, ?, ?)',
                        (tag, next_id, _TOP_LEVEL_SENTINEL, dep),
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
        tid, parent_id = _parse_task_id(task_id)
        parent_db = parent_id if parent_id is not None else _TOP_LEVEL_SENTINEL

        async with self._write_lock(project_root):
            async with self._txn(project_root) as conn:
                cursor = await conn.execute(
                    'SELECT * FROM tasks WHERE tag = ? AND id = ? AND parent_id = ?',
                    (tag, tid, parent_db),
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
                set_values.extend([tag, tid, parent_db])
                await conn.execute(
                    f'UPDATE tasks SET {set_clause} '
                    f'WHERE tag = ? AND id = ? AND parent_id = ?',
                    set_values,
                )

                # Dependencies: replace-mode only. Subtask deps are unsupported
                # (matches add_dependency); structured callers pass top-level
                # ids only. Empty list clears all deps.
                if dependencies is not None:
                    if parent_id is not None:
                        raise TaskmasterError(
                            'TASKMASTER_TOOL_ERROR',
                            'update_task: subtask dependencies are not supported',
                        )
                    parsed_deps: list[int] = []
                    for raw in dependencies:
                        dep_tid, dep_parent = _parse_task_id(raw)
                        if dep_parent is not None:
                            raise TaskmasterError(
                                'TASKMASTER_TOOL_ERROR',
                                'update_task: subtask dependencies are not supported',
                            )
                        parsed_deps.append(dep_tid)
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
                    (tag, tid, parent_db),
                )
                refreshed = await refreshed_cursor.fetchone()
        deps = (
            await self._fetch_dependencies(
                await self._get_connection(project_root), tag,
            )
        )
        updated_task = (
            _row_to_task(refreshed, deps.get((refreshed['id'], refreshed['parent_id']), []))
            if refreshed is not None else None
        )
        return {
            'id': task_id,
            'message': f'Task {task_id} updated',
            'updated': True,
            'updated_task': updated_task,
        }

    async def add_subtask(
        self,
        parent_id: str,
        project_root: str,
        title: str | None = None,
        description: str | None = None,
        details: str | None = None,
        tag: str | None = None,
    ) -> AddSubtaskResult:
        await self.ensure_connected()
        tag = tag or DEFAULT_TAG
        parent_int = _parse_task_id(parent_id)
        if parent_int[1] is not None:
            raise TaskmasterError(
                'INVALID_TASK_ID',
                f'add_subtask: nested subtask ids not supported: {parent_id!r}',
            )
        parent_tid = parent_int[0]
        if not title:
            raise TaskmasterError(
                'TASKMASTER_TOOL_ERROR',
                'add_subtask: title is required',
            )

        async with self._write_lock(project_root):
            async with self._txn(project_root) as conn:
                cursor = await conn.execute(
                    'SELECT id FROM tasks WHERE tag = ? AND id = ? AND parent_id = ?',
                    (tag, parent_tid, _TOP_LEVEL_SENTINEL),
                )
                if (await cursor.fetchone()) is None:
                    raise TaskmasterError(
                        'TASKMASTER_TOOL_ERROR',
                        f'Parent task not found: {parent_id}',
                    )

                max_cursor = await conn.execute(
                    'SELECT COALESCE(MAX(id), 0) FROM tasks WHERE tag = ? AND parent_id = ?',
                    (tag, parent_tid),
                )
                next_id = (await max_cursor.fetchone())[0] + 1
                await conn.execute(
                    """
                    INSERT INTO tasks (tag, id, parent_id, title, description,
                                       details, test_strategy, status, priority,
                                       metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, '', 'pending', NULL, NULL, ?)
                    """,
                    (
                        tag, next_id, parent_tid, title, description or '',
                        details or '', _now(),
                    ),
                )

                refreshed_cursor = await conn.execute(
                    'SELECT * FROM tasks WHERE tag = ? AND id = ? AND parent_id = ?',
                    (tag, next_id, parent_tid),
                )
                refreshed = await refreshed_cursor.fetchone()

        subtask_dict = (
            _row_to_task(refreshed, []) if refreshed is not None else {}
        )
        formatted_id = _format_task_id(next_id, parent_tid)
        return {
            'id': formatted_id,
            'parent_id': str(parent_tid),
            'message': f'New subtask {formatted_id} successfully created',
            'subtask': subtask_dict,
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
        # sent garbage; no partial-success on malformed input). Order is
        # preserved so the reported removed_ids and missing list mirror the
        # caller's input order.
        parsed: list[tuple[int, int | None]] = [_parse_task_id(raw) for raw in ids]

        async with self._write_lock(project_root):
            async with self._txn(project_root) as conn:
                # One SELECT to identify which requested rows exist. SQLite's
                # default SQL parameter limit (999) bounds batch size; realistic
                # callers stay well under it.
                where_pairs = ' OR '.join(
                    '(id = ? AND parent_id = ?)' for _ in parsed
                )
                params: list[Any] = [tag]
                for tid, parent_id in parsed:
                    parent_db = (
                        parent_id if parent_id is not None
                        else _TOP_LEVEL_SENTINEL
                    )
                    params.extend([tid, parent_db])
                cursor = await conn.execute(
                    f'SELECT id, parent_id FROM tasks '
                    f'WHERE tag = ? AND ({where_pairs})',
                    params,
                )
                existing_keys: set[tuple[int, int]] = {
                    (row['id'], row['parent_id'])
                    for row in await cursor.fetchall()
                }

                # Classify into existing (to remove) vs missing. Dedupe by
                # (id, parent_id) so duplicate caller input doesn't double-count.
                removed_keys: set[tuple[int, int]] = set()
                removed_display: list[str] = []
                existing_top_tids: set[int] = set()
                failed_display: list[str] = []
                failed_seen: set[str] = set()

                for tid, parent_id in parsed:
                    parent_db = (
                        parent_id if parent_id is not None
                        else _TOP_LEVEL_SENTINEL
                    )
                    key = (tid, parent_db)
                    disp = _format_task_id(tid, parent_id)
                    if key not in existing_keys:
                        if disp not in failed_seen:
                            failed_display.append(disp)
                            failed_seen.add(disp)
                        continue
                    if key not in removed_keys:
                        removed_keys.add(key)
                        removed_display.append(disp)
                    if parent_id is None:
                        existing_top_tids.add(tid)

                # Cascade: every existing top-level pulls in its subtasks.
                # Skip subtasks already explicitly listed by the caller so
                # they aren't reported twice.
                if existing_top_tids:
                    top_list = list(existing_top_tids)
                    top_placeholders = ','.join('?' for _ in top_list)
                    sub_cursor = await conn.execute(
                        f'SELECT id, parent_id FROM tasks '
                        f'WHERE tag = ? AND parent_id IN ({top_placeholders})',
                        [tag, *top_list],
                    )
                    sub_rows = sorted(
                        await sub_cursor.fetchall(),
                        key=lambda r: (r['parent_id'], r['id']),
                    )
                    for sub_row in sub_rows:
                        sub_key = (sub_row['id'], sub_row['parent_id'])
                        if sub_key in removed_keys:
                            continue
                        removed_keys.add(sub_key)
                        removed_display.append(
                            _format_task_id(sub_row['id'], sub_row['parent_id']),
                        )

                # Two batch DELETEs — tasks then their owning dependencies.
                # Cross-task deps pointing AT removed ids stay dangling on
                # purpose (matches the original single-id behaviour and lets
                # validate_dependencies surface them).
                if removed_keys:
                    keys_list = list(removed_keys)
                    task_pairs = ' OR '.join(
                        '(id = ? AND parent_id = ?)' for _ in keys_list
                    )
                    task_params: list[Any] = [tag]
                    for tid, pdb in keys_list:
                        task_params.extend([tid, pdb])
                    await conn.execute(
                        f'DELETE FROM tasks WHERE tag = ? AND ({task_pairs})',
                        task_params,
                    )
                    dep_pairs = ' OR '.join(
                        '(task_id = ? AND parent_id = ?)' for _ in keys_list
                    )
                    dep_params: list[Any] = [tag]
                    for tid, pdb in keys_list:
                        dep_params.extend([tid, pdb])
                    await conn.execute(
                        f'DELETE FROM dependencies WHERE tag = ? AND ({dep_pairs})',
                        dep_params,
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
        await self.ensure_connected()
        tag = tag or DEFAULT_TAG
        tid, parent_id = _parse_task_id(task_id)
        dep_tid, dep_parent_id = _parse_task_id(depends_on)
        if parent_id is not None or dep_parent_id is not None:
            # Taskmaster's tasks.json schema only persists top-level dependencies;
            # subtask deps are an undocumented edge that we explicitly reject so
            # callers can't silently lose state across the SQLite cutover.
            raise TaskmasterError(
                'TASKMASTER_TOOL_ERROR',
                'add_dependency: subtask dependencies are not supported',
            )

        async with self._write_lock(project_root):
            async with self._txn(project_root) as conn:
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
        await self.ensure_connected()
        tag = tag or DEFAULT_TAG
        tid, _ = _parse_task_id(task_id)
        dep_tid, _ = _parse_task_id(depends_on)
        async with self._write_lock(project_root):
            async with self._txn(project_root) as conn:
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


def _merge_metadata(existing_raw: str | None, incoming: str, *, append: bool) -> str:
    """Merge ``incoming`` metadata JSON into ``existing_raw``.

    Both halves are JSON-decoded and merged shallowly; ``append=True``
    preserves existing keys when there is a collision, otherwise the new
    value wins. If either side fails to decode, the new value replaces the
    old verbatim — matches Taskmaster's "last write wins" behaviour.
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
    merged = {**new, **old} if append else {**old, **new}
    return json.dumps(merged)
