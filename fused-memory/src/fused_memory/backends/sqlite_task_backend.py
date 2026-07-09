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
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite
from shared.async_sqlite_base import apply_full_durability_pragmas, connect_daemon
from shared.task_metadata import SchemaWarning, apply_migrations, parse_metadata
from shared.task_statuses import TaskStatus

from fused_memory.backends.task_backend_errors import DuplicateCandidateKeyError, TaskmasterError
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
from fused_memory.middleware.candidate_key import compute_candidate_key
from fused_memory.middleware.candidate_key_escalation import (
    emit_residual_candidate_key_escalation,
)
from fused_memory.models.scope import resolve_project_id

logger = logging.getLogger(__name__)


# Incremented whenever the DB schema changes shape.  Stored in the SQLite
# user_version header; read by ``_migrate`` at connection-open time.
#   v1: flat schema (parent_id removed).
#   v2: + claimant_run_id/heartbeat_at columns (task 2182, PRD
#       plans/task-status-authority-prd.md C4/D4). See ``_migrate_v1_to_v2``.
#   v3: + candidate_key column (fm-task-dedup W8 task A1) — computed on every
#       insert going forward; backfilled for non-cancelled rows on migration.
#       See ``_migrate_v2_to_v3``.
#   v4: + partial UNIQUE index ux_tasks_candidate_key over (tag,
#       candidate_key) WHERE candidate_key IS NOT NULL AND status !=
#       'cancelled' (fm-task-dedup W8 task A2) — self-gating: re-audits for
#       residual non-cancelled duplicates at connection-open and SKIPS the
#       index build (leaving user_version at 3) when any remain, so the next
#       open lands it once residuals are cleaned up. See ``_migrate_v3_to_v4``.
_SCHEMA_VERSION = 4

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
    tag             TEXT NOT NULL DEFAULT 'master',
    id              INTEGER NOT NULL,
    title           TEXT NOT NULL,
    description     TEXT,
    details         TEXT,
    test_strategy   TEXT,
    status          TEXT NOT NULL,
    priority        TEXT,
    metadata        TEXT,
    updated_at      TEXT NOT NULL,
    claimant_run_id TEXT,
    heartbeat_at    TEXT,
    candidate_key   TEXT,
    PRIMARY KEY (tag, id)
);

CREATE INDEX IF NOT EXISTS ix_tasks_status ON tasks (tag, status);

CREATE TABLE IF NOT EXISTS dependencies (
    tag        TEXT NOT NULL DEFAULT 'master',
    task_id    INTEGER NOT NULL,
    depends_on INTEGER NOT NULL,
    PRIMARY KEY (tag, task_id, depends_on)
);

-- Monotonic id high-water mark per tag sequence.  ``add_task`` allocates
-- ``max(MAX(tasks.id), max_id) + 1`` so a deleted id is NEVER reissued.
-- Applied idempotently via executescript on every connection open.
CREATE TABLE IF NOT EXISTS id_counters (
    tag    TEXT NOT NULL DEFAULT 'master',
    max_id INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (tag)
);
"""

DEFAULT_TAG = 'master'

# Sentinel distinguishing "leave the claimant column untouched" (default) from
# "clear it to NULL" (explicit None). Module-level and private — never
# compared across processes, only used as an in-process default marker.
_UNSET = object()

# Store-level vocabulary floor (PRD task-status-authority C2, finding 6.4):
# set_task_status/add_task reject any status outside this set instead of
# writing it verbatim. Single source of truth is shared.task_statuses —
# frozenset(TaskStatus) rather than a hardcoded literal, so a new status
# added there is accepted here automatically.
_VALID_STATUSES: frozenset[TaskStatus] = frozenset(TaskStatus)


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


def _files_for_key(metadata_raw: str | None) -> list[Any]:
    """Extract the files list used for ``candidate_key`` computation from a
    metadata JSON string.

    Single owner of the "parse metadata JSON → prefer ``files``, fall back
    to ``files_to_modify``, else ``[]``" precedence (Open Q #5: different
    producers use either key). Every candidate_key computation site —
    ``add_task`` / ``update_task`` (in-flight call args) and
    ``_migrate_v2_to_v3`` (already-persisted rows) — calls this instead of
    hand-mirroring the block, so a future change to the precedence lands in
    exactly one place instead of three.

    Defensive: a missing/``None``/malformed-JSON/non-dict ``metadata_raw``,
    or a non-list value under either key, all yield ``[]`` rather than
    raising — key computation must never be the reason an insert/update/
    migration fails. (``compute_candidate_key`` further filters non-``str``
    entries out of whatever list this returns.)
    """
    if not metadata_raw:
        return []
    try:
        parsed = json.loads(metadata_raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, dict):
        return []
    raw_files = parsed.get('files')
    if raw_files is None:
        raw_files = parsed.get('files_to_modify')
    return raw_files if isinstance(raw_files, list) else []


async def _migrate(
    conn: aiosqlite.Connection,
    *,
    project_root: str | None = None,
    residual_dup_escalation_cb: Any = None,
) -> None:
    """Cumulative, idempotent, version-gated schema migration.

    Gated on ``PRAGMA user_version``; a no-op once it reaches
    ``_SCHEMA_VERSION``. Otherwise runs whichever of the following steps the
    DB's current version still needs, in order, ending with the version
    stamped to ``_SCHEMA_VERSION``:

    * v0 → v1: parent_id schema → flat schema. When ``tasks`` still has a
      ``parent_id`` column: rebuild all three tables without it (straggler
      subtask rows with parent_id != 0 are silently dropped — by soak + DF-B
      there are none). If parent_id is already absent (fresh DB opened with
      the new ``_SCHEMA_SQL``), this step just stamps version 1 — no rebuild
      needed.
    * v1 → v2: see :func:`_migrate_v1_to_v2` — add ``claimant_run_id`` /
      ``heartbeat_at`` (task 2182, PRD plans/task-status-authority-prd.md).
    * v2 → v3: see :func:`_migrate_v2_to_v3` — add + backfill
      ``candidate_key`` (fm-task-dedup W8 task A1).
    * v3 → v4: see :func:`_migrate_v3_to_v4` — self-gating partial UNIQUE
      index over ``candidate_key`` (fm-task-dedup W8 task A2).

    Each ALTER step is column-presence-guarded, so a fresh DB whose
    ``_SCHEMA_SQL`` already created every column runs all steps as no-op
    ALTERs and only advances the ``user_version`` stamps.
    """
    row = await (await conn.execute('PRAGMA user_version')).fetchone()
    version = row[0] if row else 0
    if version >= _SCHEMA_VERSION:
        return

    if version < 1:
        info_rows = await (await conn.execute('PRAGMA table_info(tasks)')).fetchall()
        col_names = {r[1] for r in info_rows}
        if 'parent_id' in col_names:
            # Full rebuild: parent_id column is still present in all three
            # tables. Rows with parent_id != 0 (straggler subtasks) are
            # dropped by the INSERT...SELECT WHERE parent_id = 0 — no prior
            # cancellation needed. This step always lands at exactly v1 (NOT
            # _SCHEMA_VERSION) — the claimant and candidate_key columns are
            # added on top of this rebuilt table by the v1->v2 / v2->v3 steps
            # below, in the same _migrate() call.
            await conn.executescript("""
                BEGIN;

                CREATE TABLE tasks_new (
                    tag           TEXT NOT NULL DEFAULT 'master',
                    id            INTEGER NOT NULL,
                    title         TEXT NOT NULL,
                    description   TEXT,
                    details       TEXT,
                    test_strategy TEXT,
                    status        TEXT NOT NULL,
                    priority      TEXT,
                    metadata      TEXT,
                    updated_at    TEXT NOT NULL,
                    PRIMARY KEY (tag, id)
                );
                INSERT INTO tasks_new
                    (tag, id, title, description, details, test_strategy,
                     status, priority, metadata, updated_at)
                SELECT tag, id, title, description, details, test_strategy,
                       status, priority, metadata, COALESCE(updated_at, '')
                FROM tasks WHERE parent_id = 0;
                DROP TABLE tasks;
                ALTER TABLE tasks_new RENAME TO tasks;
                CREATE INDEX IF NOT EXISTS ix_tasks_status ON tasks (tag, status);

                CREATE TABLE dependencies_new (
                    tag        TEXT NOT NULL DEFAULT 'master',
                    task_id    INTEGER NOT NULL,
                    depends_on INTEGER NOT NULL,
                    PRIMARY KEY (tag, task_id, depends_on)
                );
                INSERT OR IGNORE INTO dependencies_new (tag, task_id, depends_on)
                SELECT tag, task_id, depends_on FROM dependencies WHERE parent_id = 0;
                DROP TABLE dependencies;
                ALTER TABLE dependencies_new RENAME TO dependencies;

                CREATE TABLE id_counters_new (
                    tag    TEXT NOT NULL DEFAULT 'master',
                    max_id INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (tag)
                );
                INSERT INTO id_counters_new (tag, max_id)
                SELECT tag, MAX(max_id) FROM id_counters WHERE parent_id = 0 GROUP BY tag;
                DROP TABLE id_counters;
                ALTER TABLE id_counters_new RENAME TO id_counters;

                PRAGMA user_version = 1;
                COMMIT;
            """)
        else:
            await conn.execute('PRAGMA user_version = 1')
            await conn.commit()
        version = 1

    if version < 2:
        await _migrate_v1_to_v2(conn)
        version = 2

    if version < 3:
        await _migrate_v2_to_v3(conn)
        version = 3

    if version < 4:
        await _migrate_v3_to_v4(
            conn,
            project_root=project_root,
            residual_dup_escalation_cb=residual_dup_escalation_cb,
        )


async def _migrate_v1_to_v2(conn: aiosqlite.Connection) -> None:
    """v1 -> v2 (task 2182, PRD plans/task-status-authority-prd.md C4/D4):
    add the ``claimant_run_id`` / ``heartbeat_at`` columns.

    Feature-detects each column via ``PRAGMA table_info`` and
    ``ALTER TABLE ADD COLUMN``s whichever is missing (idempotent -- a fresh DB
    created by ``_SCHEMA_SQL`` already has both, so neither ALTER fires), then
    stamps ``user_version = 2``. Reached only after the v0->v1 step, so
    ``parent_id`` is already gone by the time this runs.

    IMPORTANT: never short-circuit to "parent_id absent -> just stamp the
    version". ``_SCHEMA_SQL`` uses ``CREATE TABLE IF NOT EXISTS`` and never
    adds columns to an existing table, so an already-migrated v1 production DB
    (the common case) would silently keep missing columns forever if this step
    only stamped the version without also running the ALTERs.
    """
    info_rows = await (await conn.execute('PRAGMA table_info(tasks)')).fetchall()
    col_names = {r[1] for r in info_rows}
    if 'claimant_run_id' not in col_names:
        await conn.execute('ALTER TABLE tasks ADD COLUMN claimant_run_id TEXT')
    if 'heartbeat_at' not in col_names:
        await conn.execute('ALTER TABLE tasks ADD COLUMN heartbeat_at TEXT')
    await conn.execute('PRAGMA user_version = 2')
    await conn.commit()


async def _migrate_v2_to_v3(conn: aiosqlite.Connection) -> None:
    """v2 -> v3 (fm-task-dedup W8 task A1): add + backfill ``candidate_key``.

    Adds the nullable ``candidate_key`` column when ``tasks`` doesn't already
    have it (fresh DBs get it straight from ``_SCHEMA_SQL``, so this is a
    no-op there), then backfills it for every NON-cancelled row from that
    row's title + metadata files. Cancelled rows are deliberately left NULL:
    a cancelled task's work may be legitimately re-filed later, so a
    cancelled row should neither backfill nor count toward duplicates.

    Emits exactly one report-only audit log line naming the number of
    duplicate ``candidate_key`` groups (same tag, same key, >1 non-cancelled
    row) -- WARNING when > 0, INFO when 0. NEVER deletes a row and NEVER
    creates an index: the UNIQUE index is task A2's job, gated on this audit
    coming back clean.
    """
    info_rows = await (await conn.execute('PRAGMA table_info(tasks)')).fetchall()
    col_names = {r[1] for r in info_rows}
    if 'candidate_key' not in col_names:
        await conn.execute('ALTER TABLE tasks ADD COLUMN candidate_key TEXT')

    cursor = await conn.execute(
        "SELECT tag, id, title, metadata FROM tasks WHERE status != 'cancelled'",
    )
    # Accumulate then apply as a single executemany rather than one UPDATE
    # per row inside the loop: this is a one-shot migration, but on a large
    # legacy tasks table N sequential round-trips would otherwise delay the
    # first read after connection-open.
    updates: list[tuple[str, str, int]] = []
    for row in await cursor.fetchall():
        candidate_key = compute_candidate_key(row['title'], _files_for_key(row['metadata']))
        if candidate_key is not None:
            updates.append((candidate_key, row['tag'], row['id']))
    if updates:
        await conn.executemany(
            'UPDATE tasks SET candidate_key = ? WHERE tag = ? AND id = ?',
            updates,
        )

    dup_cursor = await conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT tag, candidate_key FROM tasks
            WHERE candidate_key IS NOT NULL AND status != 'cancelled'
            GROUP BY tag, candidate_key
            HAVING COUNT(*) > 1
        )
        """,
    )
    dup_row = await dup_cursor.fetchone()
    dup_groups = dup_row[0] if dup_row else 0

    log_fn = logger.warning if dup_groups > 0 else logger.info
    log_fn(
        'sqlite_task_backend: schema v2->v3 migration -- candidate_key '
        'backfilled for non-cancelled rows (cancelled rows left NULL); '
        'duplicate_groups=%d among non-cancelled rows (report-only -- no '
        'rows modified/deleted, no index created; see fm-task-dedup task A2)',
        dup_groups,
    )

    await conn.execute('PRAGMA user_version = 3')
    await conn.commit()


async def _migrate_v3_to_v4(
    conn: aiosqlite.Connection,
    *,
    project_root: str | None = None,
    residual_dup_escalation_cb: Any = None,
) -> None:
    """v3 -> v4 (fm-task-dedup W8 task A2): self-gating partial UNIQUE index.

    Re-runs the same residual non-cancelled duplicate ``candidate_key``
    audit ``_migrate_v2_to_v3`` performed (report-only there), extended with
    ``GROUP_CONCAT(id ORDER BY id)`` to name the offending rows in a
    deterministic (ascending id) order — SQLite does not otherwise guarantee
    ``GROUP_CONCAT`` row order, and both the ERROR log token and the
    escalation payload's ``task_ids`` list depend on a stable order:

    * **Residuals found** — log a loud ERROR naming the groups (via the
      ``residual_group_count=`` token, deliberately distinct from v2->v3's
      ``duplicate_groups=`` token so the two audits' log-scraping assertions
      never collide), invoke ``residual_dup_escalation_cb(project_root,
      residual_groups)`` when provided (best-effort — a raising callback is
      caught and logged, never propagated), and SKIP the index build.
      ``user_version`` is left at 3 (NOT stamped to 4): a later
      connection-open — after an operator cleans up the residuals — re-runs
      this step and lands the index then (PRD decision #4: "the next deploy
      lands the index").
    * **Clean** — build ``ux_tasks_candidate_key``, a PARTIAL UNIQUE index
      over ``(tag, candidate_key)`` excluding NULL keys and cancelled rows,
      then stamp ``user_version = 4``.

    FAIL-SAFE: this step NEVER raises. ``_get_connection`` only caches the
    connection AFTER ``_migrate`` returns, so a raising migration would
    crash-loop fused-memory on every connection-open. The index CREATE is
    additionally guarded against ``sqlite3.IntegrityError`` (a residual
    duplicate slipping past the audit, e.g. a race) -- skip, don't raise --
    and the whole step is wrapped so nothing unexpected propagates either.

    Deliberately NOT added to ``_SCHEMA_SQL``: a fresh-schema ``executescript``
    run at every connection-open would raise ``IntegrityError`` building the
    index against a DB that still holds residual duplicates, defeating the
    self-gating fail-safe. Fresh DBs still get the index via the full
    v0->v1->v2->v3->v4 chain, where this audit is trivially clean.
    """
    try:
        dup_cursor = await conn.execute(
            """
            SELECT tag, candidate_key, GROUP_CONCAT(id ORDER BY id) AS ids,
                   COUNT(*) AS n
            FROM tasks
            WHERE candidate_key IS NOT NULL AND status != 'cancelled'
            GROUP BY tag, candidate_key
            HAVING COUNT(*) > 1
            """,
        )
        # aiosqlite types fetchall() as Iterable[Row] (not Sized); materialize
        # to a list so len() below type-checks (it's already a list at runtime).
        residual_rows = list(await dup_cursor.fetchall())

        if residual_rows:
            groups_desc = '; '.join(
                f'tag={row["tag"]!r} candidate_key={row["candidate_key"]!r} '
                f'ids=[{row["ids"]}]'
                for row in residual_rows
            )
            logger.error(
                'sqlite_task_backend: schema v3->v4 migration SKIPPED -- '
                'residual_group_count=%d non-cancelled duplicate candidate_key '
                'group(s) still present; UNIQUE index NOT created, '
                'user_version stays at 3. Clean up the residual duplicates '
                '(cancel or merge the extras) and the next connection-open '
                'will land the index. Groups: %s',
                len(residual_rows), groups_desc,
            )

            residual_groups = [
                {
                    'tag': row['tag'],
                    'candidate_key': row['candidate_key'],
                    'task_ids': row['ids'].split(','),
                    'count': row['n'],
                }
                for row in residual_rows
            ]
            if residual_dup_escalation_cb is not None:
                try:
                    residual_dup_escalation_cb(project_root, residual_groups)
                except Exception:
                    # A broken/misbehaving callback must never crash
                    # connection-open — the skip above has already happened;
                    # escalation is purely additive.
                    logger.exception(
                        'sqlite_task_backend: residual_dup_escalation_cb '
                        'raised while escalating %d residual duplicate '
                        'candidate_key group(s) for project_root=%r',
                        len(residual_groups), project_root,
                    )
            return

        try:
            await conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_tasks_candidate_key "
                "ON tasks(tag, candidate_key) "
                "WHERE candidate_key IS NOT NULL AND status != 'cancelled'",
            )
        except sqlite3.IntegrityError:
            logger.error(
                'sqlite_task_backend: schema v3->v4 migration -- CREATE UNIQUE '
                'INDEX ux_tasks_candidate_key raised IntegrityError despite a '
                'clean audit (race?); skipping index build, user_version '
                'stays at 3.',
            )
            return

        await conn.execute('PRAGMA user_version = 4')
        await conn.commit()
        logger.info(
            'sqlite_task_backend: schema v3->v4 migration -- residual audit '
            'clean; built partial UNIQUE index ux_tasks_candidate_key over '
            '(tag, candidate_key) and advanced user_version to 4 '
            '(fm-task-dedup task A2).',
        )
    except Exception:
        # Defensive backstop -- this step must NEVER raise at connection-open
        # (see docstring): a raising migration would crash-loop fused-memory.
        logger.exception(
            'sqlite_task_backend: schema v3->v4 migration failed unexpectedly; '
            'skipping (user_version stays below 4, retried on next open)',
        )


async def _claimant_columns_present(conn: aiosqlite.Connection) -> bool:
    """True iff both ``claimant_run_id`` and ``heartbeat_at`` exist on ``tasks``.

    Feature-detection backing the claimant-writing paths' (``set_task_claimant``,
    the claimant kwargs on ``set_task_status``) fail-safe behaviour — rather
    than error — when a connection somehow predates the v1->v2 ALTER (e.g. a
    routine orchestrator restart racing ahead of the fused-memory deploy
    that ships this migration).

    Called exactly once per project_root, from ``_get_connection`` right
    after ``_migrate`` runs; the result is cached on
    ``SqliteTaskBackend._claimant_columns_cache`` and reused by the write
    paths so a hot claimant-write loop (e.g. heartbeat refresh) doesn't
    re-run ``PRAGMA table_info`` on every call — schema shape is immutable
    for the life of a connection, so a single check at open time suffices.
    """
    info_rows = await (await conn.execute('PRAGMA table_info(tasks)')).fetchall()
    col_names = {r[1] for r in info_rows}
    return 'claimant_run_id' in col_names and 'heartbeat_at' in col_names


def _warn_malformed_metadata_once(
    project_root: str,
    tag: str,
    task_id: int,
    metadata_raw: str,
    *,
    resolution: str,
) -> None:
    """Emit a deduped WARNING for a malformed metadata JSON blob.

    Owned by the module-level ``_warned_malformed_task_ids`` set, keyed by
    ``(project_root, tag, task_id)``.  Warns at most once per distinct triple
    per process, regardless of whether the access came from the read path
    (``_row_to_task``) or a write path (``_merge_metadata``).

    ``resolution`` is appended after the semicolon; ``'coerced to {}'``
    reproduces the pre-existing ``_row_to_task`` message verbatim so the
    four covering WARNING-assertion tests remain green across the extraction.
    """
    dedup_key = (project_root, tag, task_id)
    if dedup_key not in _warned_malformed_task_ids:
        _warned_malformed_task_ids.add(dedup_key)
        logger.warning(
            'sqlite_task_backend: malformed metadata JSON — project_root=%s'
            ' tag=%s id=%s metadata_raw=%s; %s',
            project_root,
            tag,
            task_id,
            repr(metadata_raw)[:80],
            resolution,
        )


def _emit_schema_warning(task_id: int, warning: SchemaWarning) -> None:
    """Emit the write-boundary census line for one :class:`SchemaWarning`.

    One WARNING line per warning ``parse_metadata`` returns — this is not
    scoped to warn-mode. In warn-mode every violation surfaces this way
    (``parse_metadata`` never raises there). In enforce-mode, only non-fatal
    warnings that ``parse_metadata`` returns *without* raising still reach
    here (e.g. ``unknown_key`` on an otherwise-valid blob); violations that
    raise (invalid fields, invariant breaches, unparseable JSON) never do,
    because the caller's ``_txn`` rolls back before this is called. The
    literal token ``task_metadata.schema_warning`` is what the enforce-gate
    census greps for in the fused-memory journal (PRD §1/§5) — it is
    deliberately distinct from the ``_warn_malformed_metadata_once``
    read-path token (``'malformed metadata'``) so the two censuses never
    conflate.
    """
    logger.warning(
        'task_metadata.schema_warning task_id=%s field=%s error=%s',
        task_id, warning.field, warning.message,
    )


def _row_to_task(row: aiosqlite.Row, dependencies: list[int], *, project_root: str) -> dict[str, Any]:
    """Convert a tasks-table row into the get_tasks/get_task wire dict.

    All tasks are top-level after DF-D. Ids surface as strings (matches live
    get_tasks wire shape). get_task converts to int after the call.
    """
    metadata_raw = row['metadata']
    metadata: Any = None
    if metadata_raw:
        # Detects the same two malformed cases parse_metadata(direction='read')
        # would flag via its 'unparseable_json'/'not_an_object' SchemaWarning
        # codes (shared/src/shared/task_metadata.py) — both are raised by that
        # function's own json.loads/isinstance(dict) guard, before
        # apply_migrations or any TaskMetadata construction runs. Checking
        # directly here — one json.loads, one isinstance — skips building
        # (and discarding) a full pydantic model per row on the get_tasks hot
        # path, and avoids parsing metadata_raw a second time for the happy
        # case.
        try:
            parsed_raw = json.loads(metadata_raw)
        except ValueError:
            parsed_raw = None
        if isinstance(parsed_raw, dict):
            # Raw shape preserved — never parse_metadata(...).model_dump():
            # unknown keys, absent schema_version, etc. round-trip
            # byte-for-value (I1) rather than gaining typed-field defaults.
            metadata = parsed_raw
        else:
            # Malformed legacy row (unparseable JSON, or valid JSON that
            # isn't an object): discard and surface {} so downstream
            # `(task.get('metadata') or {}).get(...)` callers never see a str
            # or a non-dict. WARN once per (project_root, tag, id) per process
            # so a corrupted-row batch doesn't fan out to one log line per row
            # per get_tasks call.
            _warn_malformed_metadata_once(
                project_root, row['tag'], row['id'], metadata_raw,
                resolution='coerced to {}',
            )
            metadata = {}

    row_keys = row.keys()
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
        # Guarded access (task 2182 / fm-task-dedup A1): a row from a
        # not-yet-migrated connection (pre-ALTER window) simply surfaces None
        # rather than raising.
        'claimant_run_id': row['claimant_run_id'] if 'claimant_run_id' in row_keys else None,
        'heartbeat_at': row['heartbeat_at'] if 'heartbeat_at' in row_keys else None,
        'candidate_key': row['candidate_key'] if 'candidate_key' in row_keys else None,
    }


class SqliteTaskBackend:
    """Implements :class:`TaskBackendProtocol` against per-project SQLite files.

    A single backend instance services all projects fused-memory has been
    asked about. Connections are opened lazily on first use of each
    ``project_root`` and kept open for the lifetime of the backend; close()
    drains all of them.
    """

    def __init__(
        self,
        config: TaskmasterConfig | None = None,
        *,
        task_metadata_enforce: bool = False,
        residual_dup_escalation_cb: Any = None,
    ) -> None:
        self.config = config
        # RED-TIER / restart-only (task 2162, W3-β): False (default) is
        # warn-mode — a write-boundary schema violation emits a
        # task_metadata.schema_warning census line and the write proceeds.
        # True is enforce-mode — the same violation raises and the write is
        # rolled back. See config.schema.TaskMetadataConfig.
        self._task_metadata_enforce = task_metadata_enforce
        # Injectable escalation seam for the v3->v4 migration's residual-dup
        # skip path (fm-task-dedup W8 task A2) — defaults to the production
        # helper so escalations fire without any server wiring; tests inject
        # a recording stub. See _migrate_v3_to_v4.
        self._residual_dup_escalation_cb = (
            residual_dup_escalation_cb or emit_residual_candidate_key_escalation
        )
        # Test-only fault-injection seam (fm-task-dedup W8 task A3, BT-A3):
        # when set to a callable, add_task invokes it immediately after the
        # tasks INSERT and before the txn commit, to simulate a crash
        # between INSERT and COMMIT. NOT part of TaskBackendProtocol;
        # production code paths never set this — it stays None.
        self._after_insert_fault_hook: Any = None
        self._connections: dict[str, aiosqlite.Connection] = {}
        # Guards the connection map AND each project's first-access bring-up
        # (schema + WAL pragmas). Held briefly during open; released before
        # any user-visible call runs.
        self._connect_locks: dict[str, asyncio.Lock] = {}
        self._connect_locks_lock = asyncio.Lock()
        # Per-project write serialisation (mirrors the interceptor's
        # ``_write_lock`` pattern). WAL allows concurrent readers natively.
        self._write_locks: dict[str, asyncio.Lock] = {}
        # Cached result of `_claimant_columns_present` per project_root,
        # populated once in `_get_connection` right after `_migrate` runs.
        # Column presence is immutable for the life of a connection (the only
        # writer of schema shape is `_migrate`, which only runs once per
        # connection-open), so re-querying `PRAGMA table_info` on every
        # claimant write (the heartbeat-refresh hot path) is unnecessary.
        # Absent entries default to False (fail-safe) in the write paths.
        self._claimant_columns_cache: dict[str, bool] = {}
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
            await _migrate(
                conn,
                project_root=project_root,
                residual_dup_escalation_cb=self._residual_dup_escalation_cb,
            )
            # Cache once per project_root — see the field docstring in
            # __init__ for why this is safe to compute a single time here
            # rather than on every claimant write.
            self._claimant_columns_cache[project_root] = await _claimant_columns_present(conn)
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
        statuses: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if statuses is not None and not statuses:
            return []
        conn = await self._get_connection(project_root)
        if statuses is None:
            cursor = await conn.execute(
                'SELECT * FROM tasks WHERE tag = ? ORDER BY id',
                (tag,),
            )
        else:
            placeholders = ','.join('?' * len(statuses))
            cursor = await conn.execute(
                f'SELECT * FROM tasks WHERE tag = ? AND status IN ({placeholders}) ORDER BY id',
                (tag, *statuses),
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
        statuses: list[str] | None = None,
    ) -> GetTasksResult:
        await self.ensure_connected()
        tag = tag or DEFAULT_TAG
        tasks = await self._get_tasks_internal(project_root, tag, statuses=statuses)
        return {'tasks': tasks}

    async def get_task(
        self, task_id: str, project_root: str, tag: str | None = None,
    ) -> dict:
        await self.ensure_connected()
        tag = tag or DEFAULT_TAG
        tid = _parse_task_id(task_id)
        conn = await self._get_connection(project_root)

        cursor = await conn.execute(
            'SELECT * FROM tasks WHERE tag = ? AND id = ?',
            (tag, tid),
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

    async def get_statuses_raw(
        self,
        project_root: str,
        tag: str | None = None,
        ids: list[str] | None = None,
    ) -> dict[str, str]:
        """Return ``{id_str: status_str}`` for tasks, reading ONLY ``id`` and ``status``.

        This is the O(K) status-only path — it never calls
        ``_get_tasks_internal``, ``_row_to_task``, or ``json.loads``,
        so metadata columns are never decoded.

        Args:
            project_root: Absolute path to the project root.
            tag: Tag context; defaults to ``DEFAULT_TAG`` when ``None``.
            ids: When given, only return entries for these task ids (as strings;
                 cast to int for the SQL IN clause; non-numeric ids silently
                 omitted).  ``None`` returns all tasks.  ``[]`` returns ``{}``.

        Returns:
            ``{str(id): status}`` mapping.  Unknown ids are silently omitted.
            A ``NULL`` status (defensive; unreachable via normal writes) maps to
            ``'unknown'``.
        """
        # ensure_connected is idempotent; the double-call here (interceptor's
        # _ensure_taskmaster also calls it) is harmless and keeps the backend
        # safely callable in isolation without relying on the caller to connect first.
        await self.ensure_connected()
        tag = tag or DEFAULT_TAG
        conn = await self._get_connection(project_root)

        if ids is not None:
            if not ids:
                return {}
            int_ids: list[int] = []
            for raw_id in ids:
                with contextlib.suppress(ValueError, TypeError):
                    int_ids.append(int(raw_id))
            if not int_ids:
                return {}
            placeholders = ','.join('?' * len(int_ids))
            cursor = await conn.execute(
                f'SELECT id, status FROM tasks WHERE tag = ? AND id IN ({placeholders})',
                (tag, *int_ids),
            )
        else:
            cursor = await conn.execute(
                'SELECT id, status FROM tasks WHERE tag = ?',
                (tag,),
            )

        rows = await cursor.fetchall()
        return {
            str(row['id']): (row['status'] if row['status'] is not None else 'unknown')
            for row in rows
        }

    async def get_statuses(
        self,
        project_root: str,
        ids: list[str] | None = None,
        tag: str | None = None,
    ) -> dict[str, str]:
        """Return ``{id_str: status_str}`` for tasks in *project_root*.

        Higher-level counterpart to :meth:`get_statuses_raw`.  For the SQLite
        backend the ``NULL`` → ``'unknown'`` coercion already happens in
        ``get_statuses_raw``, so this simply delegates — keeping a single
        SQL/coercion path while still satisfying ``TaskBackendProtocol`` (the
        reconciliation harness calls ``taskmaster.get_statuses`` directly).
        """
        return await self.get_statuses_raw(project_root, tag=tag, ids=ids)

    async def set_task_status(
        self,
        task_id: str,
        status: str,
        project_root: str,
        tag: str | None = None,
        *,
        claimant_run_id: str | None = _UNSET,  # type: ignore[assignment]
        heartbeat_at: str | None = _UNSET,  # type: ignore[assignment]
    ) -> SetTaskStatusResult:
        """Update ``status``, optionally stamping/clearing the claimant columns.

        ``claimant_run_id``/``heartbeat_at`` are tri-state (task 2182, PRD
        C4/D4): a string stamps the column, explicit ``None`` clears it to
        NULL (release), and the default ``_UNSET`` leaves it untouched — a
        plain status change must never wipe a live claimant. Fails safe
        (WARNING, status-only write, no error) when the claimant columns are
        absent from a not-yet-migrated connection.

        Raises :class:`DuplicateCandidateKeyError` (fm-task-dedup W8 task A2
        review amendment) instead of letting a raw ``sqlite3.IntegrityError``
        escape on the narrow un-cancel collision case: moving a row's status
        OFF ``'cancelled'`` makes it visible to the partial UNIQUE index on
        ``(tag, candidate_key)``, and if another non-cancelled row already
        holds the same key, this UPDATE is rejected rather than silently
        reactivating a duplicate.
        """
        await self.ensure_connected()
        tag = tag or DEFAULT_TAG
        tid = _parse_task_id(task_id)
        if status not in _VALID_STATUSES:
            raise TaskmasterError(
                'TASKMASTER_TOOL_ERROR',
                f'Invalid status {status!r}. Must be one of '
                f'{sorted(s.value for s in _VALID_STATUSES)}.',
            )
        async with self._write_lock(project_root), self._txn(project_root) as conn:
            cursor = await conn.execute(
                'SELECT status, candidate_key FROM tasks WHERE tag = ? AND id = ?',
                (tag, tid),
            )
            row = await cursor.fetchone()
            if row is None:
                raise TaskmasterError(
                    'TASKMASTER_TOOL_ERROR',
                    f'No tasks found for ID(s): {task_id}',
                )
            old_status = row['status']
            row_candidate_key = row['candidate_key']

            set_columns = ['status = ?', 'updated_at = ?']
            set_values: list[Any] = [status, _now()]
            if claimant_run_id is not _UNSET or heartbeat_at is not _UNSET:
                if self._claimant_columns_cache.get(project_root, False):
                    if claimant_run_id is not _UNSET:
                        set_columns.append('claimant_run_id = ?')
                        set_values.append(claimant_run_id)
                    if heartbeat_at is not _UNSET:
                        set_columns.append('heartbeat_at = ?')
                        set_values.append(heartbeat_at)
                else:
                    logger.warning(
                        'set_task_status: claimant_run_id/heartbeat_at columns absent '
                        '(pre-migration connection) — writing status only for '
                        'task_id=%s project_root=%s',
                        task_id, project_root,
                    )

            set_values.extend([tag, tid])
            try:
                await conn.execute(
                    f'UPDATE tasks SET {", ".join(set_columns)} '
                    'WHERE tag = ? AND id = ?',
                    set_values,
                )
            except sqlite3.IntegrityError as exc:
                # Only the candidate_key partial UNIQUE index is mapped to a
                # typed collision (mirrors add_task's collision mapping); any
                # other integrity violation is unrelated and re-raised
                # untouched. Reachable via the narrow un-cancel path (see the
                # docstring above). Nothing in this transaction has been
                # written yet — this is the first write statement — so this
                # survivor lookup sees the same state a post-rollback read
                # would (this row's own candidate_key/status are unaffected,
                # having never been applied).
                if row_candidate_key is None or 'candidate_key' not in str(exc):
                    raise
                survivor_cursor = await conn.execute(
                    "SELECT id, status FROM tasks WHERE tag = ? AND candidate_key = ? "
                    "AND status != 'cancelled' ORDER BY id LIMIT 1",
                    (tag, row_candidate_key),
                )
                survivor = await survivor_cursor.fetchone()
                raise DuplicateCandidateKeyError(
                    existing_id=survivor['id'] if survivor is not None else None,
                    existing_status=survivor['status'] if survivor is not None else None,
                    tag=tag,
                    candidate_key=row_candidate_key,
                ) from exc
        return {
            'message': f'Successfully updated 1 task(s) to "{status}"',
            'tasks': [{
                'taskId': task_id,
                'oldStatus': old_status,
                'newStatus': status,
            }],
        }

    async def set_task_claimant(
        self,
        task_id: str,
        project_root: str,
        *,
        claimant_run_id: str | None = _UNSET,  # type: ignore[assignment]
        heartbeat_at: str | None = _UNSET,  # type: ignore[assignment]
        tag: str | None = None,
    ) -> dict:
        """Stamp or clear claimant_run_id/heartbeat_at without touching status.

        Dedicated write path for the heartbeat-refresh/clear cycle (task 2182,
        PRD ``plans/task-status-authority-prd.md`` C4/D4) — kept separate from
        ``set_task_status``, which owns the status-FSM gates. Each param is
        independently tri-state: a string stamps the column, explicit
        ``None`` clears it to NULL, and the default ``_UNSET`` leaves it
        untouched. Fails safe (WARNING, no write, no error) when the
        claimant columns are absent from a not-yet-migrated connection.
        """
        await self.ensure_connected()
        tag = tag or DEFAULT_TAG
        tid = _parse_task_id(task_id)
        async with self._write_lock(project_root), self._txn(project_root) as conn:
            cursor = await conn.execute(
                'SELECT id FROM tasks WHERE tag = ? AND id = ?',
                (tag, tid),
            )
            if (await cursor.fetchone()) is None:
                raise TaskmasterError(
                    'TASKMASTER_TOOL_ERROR',
                    f'No tasks found for ID(s): {task_id}',
                )

            set_columns: list[str] = []
            set_values: list[Any] = []
            if claimant_run_id is not _UNSET:
                set_columns.append('claimant_run_id')
                set_values.append(claimant_run_id)
            if heartbeat_at is not _UNSET:
                set_columns.append('heartbeat_at')
                set_values.append(heartbeat_at)

            if not set_columns:
                return {'id': task_id, 'message': f'No claimant changes supplied for task {task_id}'}

            if not self._claimant_columns_cache.get(project_root, False):
                logger.warning(
                    'set_task_claimant: claimant_run_id/heartbeat_at columns absent '
                    '(pre-migration connection) — skipping claimant write for '
                    'task_id=%s project_root=%s',
                    task_id, project_root,
                )
                return {'id': task_id, 'message': f'Claimant columns unavailable; no write for task {task_id}'}

            set_clause = ', '.join(f'{c} = ?' for c in set_columns)
            set_values.extend([tag, tid])
            await conn.execute(
                f'UPDATE tasks SET {set_clause} WHERE tag = ? AND id = ?',
                set_values,
            )
        return {
            'id': task_id,
            'message': f'Updated claimant fields for task {task_id}',
        }

    # ── Write-boundary validation (task 2162, W3-β) ────────────────────

    async def _validate_metadata_on_write(
        self,
        metadata: str | None,
        *,
        project_root: str,
        tag: str,
        task_id: int,
    ) -> None:
        """Validate a ``metadata`` JSON blob at the add_task/update_task write boundary.

        Delegates to the shared ``parse_metadata`` (direction='write',
        ``enforce=self._task_metadata_enforce``). In warn-mode (the default)
        every returned :class:`SchemaWarning` is logged as one
        ``task_metadata.schema_warning`` census line and the write proceeds
        unchanged. In enforce-mode, a malformed blob's raise
        (``ValidationError`` / ``ValueError`` / ``TypeError``) propagates
        uncaught — the caller's ``_txn`` rolls back.

        ``project_root``/``tag`` are accepted but not yet read by this method
        — the census line only carries ``task_id``/field/error. They mirror
        ``_warn_malformed_metadata_once``'s ``(project_root, tag, task_id)``
        triple so a future write-side census enrichment (e.g. scoping/dedup
        by project) can use them without changing either call site's
        signature.
        """
        _, warnings = parse_metadata(
            metadata, direction='write', enforce=self._task_metadata_enforce,
        )
        for warning in warnings:
            _emit_schema_warning(task_id, warning)

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
        if status not in _VALID_STATUSES:
            raise TaskmasterError(
                'TASKMASTER_TOOL_ERROR',
                f'Invalid status {status!r}. Must be one of '
                f'{sorted(s.value for s in _VALID_STATUSES)}.',
            )

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

        # candidate_key (fm-task-dedup W8 task A1): computed from title +
        # metadata['files'] (via the shared ``_files_for_key`` extraction,
        # which falls back to metadata['files_to_modify'] — Open Q #5:
        # different producers use either key) on every insert. Defensive —
        # malformed/non-dict metadata yields an empty file list rather than
        # raising; the key must never be the reason an insert fails.
        candidate_key = compute_candidate_key(title, _files_for_key(metadata))

        try:
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
                            WHERE tag = ?
                        UNION ALL
                        SELECT COALESCE(max_id, 0) AS highwater FROM id_counters
                            WHERE tag = ?
                    )
                    """,
                    (tag, tag),
                )
                _max_row = await cursor.fetchone()
                assert _max_row is not None  # aggregate MAX always returns one row
                next_id = (_max_row[0] or 0) + 1
                await self._validate_metadata_on_write(
                    metadata, project_root=project_root, tag=tag, task_id=next_id,
                )
                await conn.execute(
                    """
                        INSERT INTO tasks (tag, id, title, description,
                                           details, test_strategy, status, priority,
                                           metadata, updated_at, candidate_key)
                        VALUES (?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?)
                        """,
                    (
                        tag, next_id, title,
                        description or '', details or '',
                        status, priority or 'medium', metadata, _now(),
                        candidate_key,
                    ),
                )
                if self._after_insert_fault_hook is not None:
                    # BT-A3 crash seam: raises to simulate a crash between
                    # INSERT and COMMIT; _txn rolls back on any BaseException
                    # and re-raises, so this propagates untouched (it is not
                    # a sqlite3.IntegrityError, so the except clause below
                    # does not intercept it).
                    self._after_insert_fault_hook()
                for dep in deps_list:
                    await conn.execute(
                        'INSERT OR IGNORE INTO dependencies '
                        '(tag, task_id, depends_on) VALUES (?, ?, ?)',
                        (tag, next_id, dep),
                    )
                await conn.execute(
                    """
                    INSERT INTO id_counters (tag, max_id) VALUES (?, ?)
                        ON CONFLICT(tag) DO UPDATE SET max_id = excluded.max_id
                            WHERE excluded.max_id > id_counters.max_id
                    """,
                    (tag, next_id),
                )
            return {
                'id': str(next_id),
                'message': f'Successfully added new task #{next_id}',
            }
        except sqlite3.IntegrityError as exc:
            # Only the partial UNIQUE index on candidate_key is mapped to a
            # typed collision — any other integrity violation (e.g. a PRIMARY
            # KEY clash, which the id high-water allocation above should make
            # unreachable) is unrelated and re-raised untouched. SQLite names
            # the violated columns in the message ('UNIQUE constraint failed:
            # tasks.tag, tasks.candidate_key'), and this is the only index
            # that references candidate_key, so the substring check is
            # unambiguous. A NULL candidate_key can never trip this index
            # (the partial predicate excludes it), so it's checked first.
            if candidate_key is None or 'candidate_key' not in str(exc):
                raise
            # `_txn` above has already rolled back the failed INSERT (zero
            # orphan rows) by the time this except runs. Look up the
            # surviving non-cancelled row with this (tag, candidate_key) on
            # a fresh read over the same cached connection.
            conn = await self._get_connection(project_root)
            survivor_cursor = await conn.execute(
                "SELECT id, status FROM tasks WHERE tag = ? AND candidate_key = ? "
                "AND status != 'cancelled' ORDER BY id LIMIT 1",
                (tag, candidate_key),
            )
            survivor = await survivor_cursor.fetchone()
            raise DuplicateCandidateKeyError(
                existing_id=survivor['id'] if survivor is not None else None,
                existing_status=survivor['status'] if survivor is not None else None,
                tag=tag,
                candidate_key=candidate_key,
            ) from exc

    async def update_task(
        self,
        task_id: str,
        project_root: str,
        prompt: str | None = None,
        metadata: str | None = None,
        append: bool | None = None,
        tag: str | None = None,
        *,
        metadata_mode: str | None = None,
        title: str | None = None,
        description: str | None = None,
        details: str | None = None,
        priority: str | None = None,
        status: str | None = None,
        dependencies: list[str] | None = None,
    ) -> UpdateTaskResult:
        # Backend floor mirroring the server/tools.py + interceptor ceiling
        # (2026-05-08 forensics). set_task_status is the only sanctioned status
        # writer — it enforces the terminal-exit, phantom-done, and
        # done-provenance gates. Reject unconditionally, before ensure_connected()
        # and the task SELECT, so status rejection takes precedence over any
        # existence or connection error.
        if status is not None:
            raise TaskmasterError(
                'TASKMASTER_TOOL_ERROR',
                'update_task is metadata-only and cannot write status. '
                'Use set_task_status(status=…) instead — it enforces the '
                'terminal-exit, phantom-done, and done-provenance gates.',
            )
        # Structured fields (title/description/details/priority/dependencies)
        # land deterministically — any non-None value overrides the current row.
        # ``prompt`` is kept for backward compatibility: when no explicit
        # ``details`` is passed it feeds the details path (replace, or append
        # when ``append=True``). ``metadata`` retains the merge-or-replace
        # semantics keyed off ``append``.
        # Validate metadata_mode unconditionally — a bad value should always
        # raise immediately, even if no metadata is supplied in this call.
        # Resolution is stored so the metadata block can reuse it without a
        # second call (and to keep the single call-site clear).
        resolved_mode = _resolve_metadata_mode(metadata_mode, append)

        await self.ensure_connected()
        tag = tag or DEFAULT_TAG
        tid = _parse_task_id(task_id)

        async with self._write_lock(project_root), self._txn(project_root) as conn:
            cursor = await conn.execute(
                'SELECT * FROM tasks WHERE tag = ? AND id = ?',
                (tag, tid),
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
            # NOTE: status is intentionally absent — the guard at the top of
            # this method unconditionally raises before reaching this point.

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

            new_metadata: str | None = None
            if metadata is not None:
                # Behavior note: on merge/additive, _merge_metadata RAISES
                # TaskmasterError if the stored blob is corrupt — preventing a
                # silent clobber.  The _txn wrapper rolls back, leaving the
                # original bytes intact.  To repair a corrupt row, pass
                # metadata_mode='replace' (bypasses the guard intentionally).
                new_metadata = _merge_metadata(
                    row['metadata'], metadata,
                    mode=resolved_mode,
                    project_root=project_root, tag=tag, task_id=tid,
                )
                # Validate the POST-MERGE blob so the deterministic
                # cross-field invariant (I3) is caught on update, not only
                # on submit. Warn-mode logs the census line and proceeds;
                # enforce-mode's raise propagates out of this `async with
                # self._txn(...)`, rolling back the UPDATE below.
                await self._validate_metadata_on_write(
                    new_metadata, project_root=project_root, tag=tag, task_id=tid,
                )
                set_columns.append('metadata = ?')
                set_values.append(new_metadata)

            # candidate_key (fm-task-dedup W8 task A1 amendment): recompute
            # whenever this call touches title and/or metadata — both feed
            # the key, and a stale key (still reflecting the pre-update
            # title/files) would silently break the future A2 dedup index.
            # An update touching neither is left alone: the existing value
            # still correctly describes the unchanged row.
            new_candidate_key: str | None = None
            if title is not None or metadata is not None:
                final_title = title if title is not None else row['title']
                final_metadata_raw = new_metadata if metadata is not None else row['metadata']
                new_candidate_key = compute_candidate_key(
                    final_title, _files_for_key(final_metadata_raw),
                )
                set_columns.append('candidate_key = ?')
                set_values.append(new_candidate_key)

            # updated_at always advances, even on a no-op write — matches
            # the original behaviour and avoids surprising "stale" reads.
            set_columns.append('updated_at = ?')
            set_values.append(_now())

            set_clause = ', '.join(set_columns)
            set_values.extend([tag, tid])
            try:
                await conn.execute(
                    f'UPDATE tasks SET {set_clause} '
                    f'WHERE tag = ? AND id = ?',
                    set_values,
                )
            except sqlite3.IntegrityError as exc:
                # Only the candidate_key partial UNIQUE index is mapped to a
                # typed collision (mirrors add_task; fm-task-dedup W8 task A2
                # review amendment); any other integrity violation is
                # unrelated and re-raised untouched. Reachable only when this
                # call recomputed candidate_key above (title/metadata
                # touched) and the new key collides with another
                # non-cancelled row of the same tag. Nothing in this
                # transaction has been written yet — this is the first write
                # statement — so this survivor lookup sees the same state a
                # post-rollback read would.
                if new_candidate_key is None or 'candidate_key' not in str(exc):
                    raise
                survivor_cursor = await conn.execute(
                    "SELECT id, status FROM tasks WHERE tag = ? AND candidate_key = ? "
                    "AND status != 'cancelled' ORDER BY id LIMIT 1",
                    (tag, new_candidate_key),
                )
                survivor = await survivor_cursor.fetchone()
                raise DuplicateCandidateKeyError(
                    existing_id=survivor['id'] if survivor is not None else None,
                    existing_status=survivor['status'] if survivor is not None else None,
                    tag=tag,
                    candidate_key=new_candidate_key,
                ) from exc

            # Dependencies: replace-mode only. Empty list clears all deps.
            if dependencies is not None:
                parsed_deps: list[int] = [_parse_task_id(raw) for raw in dependencies]
                await conn.execute(
                    'DELETE FROM dependencies WHERE tag = ? AND task_id = ?',
                    (tag, tid),
                )
                for dep in parsed_deps:
                    await conn.execute(
                        'INSERT OR IGNORE INTO dependencies '
                        '(tag, task_id, depends_on) VALUES (?, ?, ?)',
                        (tag, tid, dep),
                    )

            refreshed_cursor = await conn.execute(
                'SELECT * FROM tasks WHERE tag = ? AND id = ?',
                (tag, tid),
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
                f'SELECT id FROM tasks WHERE tag = ? AND id IN ({id_placeholders})',
                [tag, *parsed],
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
                    f'DELETE FROM tasks WHERE tag = ? AND id IN ({rm_placeholders})',
                    [tag, *removed_ids],
                )
                await conn.execute(
                    f'DELETE FROM dependencies WHERE tag = ? AND task_id IN ({rm_placeholders})',
                    [tag, *removed_ids],
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
                    'SELECT metadata FROM tasks WHERE tag = ? AND id = ?',
                    (tag, tid),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise TaskmasterError(
                        'TASKMASTER_TOOL_ERROR',
                        f'No tasks found for ID(s): {tid}',
                    )
                # If the stored blob is corrupt, _merge_metadata raises
                # TaskmasterError (propagates through _txn → rollback → no
                # write).  This is intentionally ASYMMETRIC with
                # remove_dependency, which returns a non-removal message
                # rather than raising: a merge into a corrupt blob would
                # clobber (must refuse), while a remove is idempotent and
                # the dep is provably absent in an unparseable blob (no
                # write needed, accurate message suffices).
                new_meta = _merge_metadata(
                    row['metadata'],
                    json.dumps({'external_deps': [canonical]}),
                    mode='additive',
                    project_root=project_root, tag=tag, task_id=tid,
                )
                await conn.execute(
                    'UPDATE tasks SET metadata = ?, updated_at = ? '
                    'WHERE tag = ? AND id = ?',
                    (new_meta, _now(), tag, tid),
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
                    'SELECT id FROM tasks WHERE tag = ? AND id = ?',
                    (tag, tid_check),
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
                '(tag, task_id, depends_on) VALUES (?, ?, ?)',
                (tag, tid, dep_tid),
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
                    'SELECT metadata FROM tasks WHERE tag = ? AND id = ?',
                    (tag, tid),
                )
                row = await cursor.fetchone()
                if row is not None:
                    try:
                        meta = json.loads(row['metadata'] or '{}')
                    except (TypeError, ValueError):
                        # Corrupt stored blob: warn once via the shared gate and
                        # return an accurate message — do NOT claim removal, do
                        # NOT write (blob left intact).
                        #
                        # Intentionally does NOT raise (contrast with
                        # add_dependency, which raises on a corrupt blob):
                        # remove is idempotent — the dep is provably absent
                        # in an unparseable blob, so no write is needed and
                        # an accurate return value is the right signal.
                        # add_dependency must refuse because a merge into a
                        # corrupt blob would clobber external_deps.
                        _warn_malformed_metadata_once(
                            project_root, tag, tid, row['metadata'] or '',
                            resolution='remove_dependency skipped — corrupt blob left intact',
                        )
                        return {
                            'id': str(tid),
                            'dependency_id': canonical,
                            'message': (
                                f'Could not remove external dependency {canonical} from task {tid}:'
                                f' metadata blob is corrupt and was left intact.'
                            ),
                        }
                    existing = meta.get('external_deps', [])
                    if canonical in existing:
                        updated = [e for e in existing if e != canonical]
                        meta['external_deps'] = updated
                        await conn.execute(
                            'UPDATE tasks SET metadata = ?, updated_at = ? '
                            'WHERE tag = ? AND id = ?',
                            (json.dumps(meta), _now(), tag, tid),
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
                'AND depends_on = ?',
                (tag, tid, dep_tid),
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
            WHERE d.tag = ? AND t.id IS NULL
            """,
            (tag,),
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


_METADATA_MODES: frozenset[str] = frozenset({'merge', 'additive', 'replace'})


def _resolve_metadata_mode(
    metadata_mode: str | None,
    append: bool | None,
) -> str:
    """Resolve the effective metadata merge mode from the two input signals.

    Precedence (high → low):
    1. ``metadata_mode`` — explicit tri-state wins unconditionally.
    2. ``append`` legacy shim — True → 'additive', False → 'replace'.
    3. Default — 'merge' (shallow last-write-wins) when both are None.

    Raises :class:`TaskmasterError` (``TASKMASTER_TOOL_ERROR``) for an
    unrecognised ``metadata_mode`` value (loud over silent).
    """
    if metadata_mode is not None:
        if metadata_mode not in _METADATA_MODES:
            raise TaskmasterError(
                'TASKMASTER_TOOL_ERROR',
                f"Invalid metadata_mode {metadata_mode!r}; "
                f"must be one of {sorted(_METADATA_MODES)}.",
            )
        return metadata_mode
    if append is not None:
        return 'additive' if append else 'replace'
    return 'merge'


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


def _merge_metadata(
    existing_raw: str | None,
    incoming: str,
    *,
    mode: str,
    project_root: str | None = None,
    tag: str | None = None,
    task_id: int | None = None,
) -> str:
    """Merge ``incoming`` metadata JSON into ``existing_raw`` using ``mode``.

    ``mode`` must be one of the three values produced by
    :func:`_resolve_metadata_mode`:

    * ``'replace'`` — return ``incoming`` verbatim; bypasses the corrupt-blob
      guard (the sanctioned path to repair a corrupt row).
    * ``'merge'`` — shallow last-write-wins: ``{**existing, **incoming}``.
      Omitted keys are preserved; every supplied key (scalar **or** list)
      overwrites wholesale.  Falls back to ``incoming`` when either side is
      valid JSON but not a dict.
    * ``'additive'`` — existing recursive ``_merge_values`` semantics: list
      union+dedup, dict-recursive, scalar/type-collision OLD-wins.  Keeps the
      legacy ``memory_hints`` list-shape normalisation before the dict union.

    For ``'merge'`` and ``'additive'``, a corrupt *existing* blob raises
    :class:`TaskmasterError` (``TASKMASTER_TOOL_ERROR``) — refusing to clobber
    it.  The ``_txn`` rollback leaves the original bytes intact.  Pass
    ``project_root``/``tag``/``task_id`` to emit a deduplicated WARNING.

    A ``None`` ``existing_raw`` is treated as empty: ``incoming`` is returned
    for all three modes (nothing to preserve or merge into).
    """
    # replace (or no existing data): return incoming verbatim, no guard.
    if existing_raw is None or mode == 'replace':
        return incoming
    # merge and additive: parse the existing blob first so a corrupt EXISTING
    # blob is distinguishable from a corrupt incoming blob.
    try:
        old = json.loads(existing_raw)
    except (TypeError, ValueError) as err:
        # Corrupt EXISTING blob: refuse to clobber on merge/additive.
        # Warn once through the shared dedup gate when context is available.
        if project_root is not None and tag is not None and task_id is not None:
            _warn_malformed_metadata_once(
                project_root, tag, task_id, existing_raw,
                resolution='refused metadata merge — original bytes preserved',
            )
        raise TaskmasterError(
            'TASKMASTER_TOOL_ERROR',
            f'Task {task_id} has a corrupt metadata blob; refusing to overwrite it '
            f'(original bytes preserved).',
            raw=existing_raw,
        ) from err
    # Corrupt INCOMING blob: intentionally last-write-wins (pre-existing
    # Taskmaster fallback behaviour).
    try:
        new = json.loads(incoming)
    except (TypeError, ValueError):
        return incoming
    if not isinstance(old, dict) or not isinstance(new, dict):
        return incoming
    if mode == 'merge':
        # Shallow last-write-wins: every supplied key overwrites wholesale.
        return json.dumps({**old, **new})
    # mode == 'additive': recursive _merge_values with memory_hints normalization.
    # Normalize legacy memory_hints list shape on both sides before merging.
    # Only fires when BOTH sides carry memory_hints — i.e. on the merge collision
    # path — so a write that omits memory_hints does not silently migrate the
    # stored legacy shape.
    if "memory_hints" in old and "memory_hints" in new:
        old = {**old, "memory_hints": apply_migrations({"memory_hints": old["memory_hints"]})["memory_hints"]}
        new = {**new, "memory_hints": apply_migrations({"memory_hints": new["memory_hints"]})["memory_hints"]}
    try:
        merged = _merge_values(old, new)
    except RecursionError:
        # Pathologically deep metadata; fall back to last-write-wins.
        return incoming
    return json.dumps(merged)
