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
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite
from shared.async_sqlite_base import (
    apply_full_durability_pragmas,
    apply_wal_pragmas,
    connect_daemon,
)
from shared.task_metadata import (
    _WHOLE_METADATA_FIELD,
    SchemaWarning,
    apply_migrations,
    parse_metadata,
)
from shared.task_statuses import TaskStatus

from fused_memory.backends.task_backend_errors import (
    DoneProvenanceWriteAuthorityError,
    DuplicateCandidateKeyError,
    StatusWriteAuthorityError,
    TaskmasterError,
)
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

# `SchemaWarning.code` values that `_validate_metadata_on_write`'s enforce-mode
# gate (below) must NOT treat as fatal when deciding whether an incoming-key
# warning re-triggers whole-blob validation. `unknown_key` is the ONLY code
# `parse_metadata(..., enforce=True)` never actually raises on: `TaskMetadata`
# has `model_config = ConfigDict(extra='allow')`, so an unrecognised top-level
# key is accepted into the model and its `unknown_key` warning is emitted only
# AFTER construction already succeeded (shared/task_metadata.py
# `parse_metadata`). Every other code (unparseable_json, not_an_object,
# invalid_submodel, invalid_field, invalid_metadata) corresponds to a path
# that DOES raise under enforce=True. Blacklisting this one code — rather
# than whitelisting the fatal ones — is fail-closed: a hypothetical future
# warning code defaults to fatal rather than silently under-blocking a
# genuinely-bad write (task 2405).
_NON_FATAL_WRITE_WARNING_CODES: frozenset[str] = frozenset({'unknown_key'})


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


def _classify_residual_group(
    rows: list[Any],
) -> tuple[str, int, list[int]] | tuple[str, str]:
    """Classify one residual (tag, candidate_key) duplicate group of
    non-cancelled rows for the v3->v4 self-heal migration step (fm-task-dedup
    W8 amendment, reify incident esc-candidate-key-migration-2).

    ``rows`` are dict-likes (``aiosqlite.Row`` or plain ``dict``), each
    carrying at least ``id``, ``title``, ``status``, ``metadata`` (raw JSON
    text or ``None``), and ``candidate_key`` (the STORED value shared by
    every row in the group -- that's why the caller's audit query grouped
    them together in the first place).

    Returns:
        ``('heal', canonical_id, cancel_ids)`` when every row recomputes to
        the SAME candidate_key as what's stored (guards a stale/legacy
        stored key -- the "verified same candidate_key" check) AND no row is
        ``done`` (``shared.task_statuses.TaskStatus.DONE``) -- a genuine
        content-duplicate, safe to auto-collapse. ``canonical_id`` is the
        lowest id among ``in-progress`` rows if any, else the lowest id
        overall; ``cancel_ids`` is the ascending-sorted remainder.

        ``('flag', reason)`` otherwise, with ``reason`` one of:

        * ``'title_divergent'`` -- some row's fresh ``(title, files)``
          recompute disagrees with the group's stored candidate_key, so the
          group is not actually a content match (a stale/legacy stored key
          coincidentally collided). Checked FIRST: a group that fails this
          "verified same candidate_key" check is never safe to reason about
          via status alone.
        * ``'mixed_status'`` -- a verified-same group that also contains a
          ``done`` row. Cancelling completed work needs a human even though
          the content genuinely matches.
    """
    stored_key = rows[0]['candidate_key']
    if any(
        compute_candidate_key(row['title'], _files_for_key(row['metadata'])) != stored_key
        for row in rows
    ):
        return ('flag', 'title_divergent')
    if any(row['status'] == TaskStatus.DONE for row in rows):
        return ('flag', 'mixed_status')
    in_progress_ids = [row['id'] for row in rows if row['status'] == TaskStatus.IN_PROGRESS]
    canonical_id = min(in_progress_ids) if in_progress_ids else min(row['id'] for row in rows)
    cancel_ids = sorted(row['id'] for row in rows if row['id'] != canonical_id)
    return ('heal', canonical_id, cancel_ids)


async def _migrate_v3_to_v4(
    conn: aiosqlite.Connection,
    *,
    project_root: str | None = None,
    residual_dup_escalation_cb: Any = None,
) -> dict[str, Any]:
    """v3 -> v4 (fm-task-dedup W8 task A2, self-heal amendment): self-gating
    partial UNIQUE index, now with an intermediate self-heal pass.

    Runs the same residual non-cancelled duplicate ``candidate_key``
    audit ``_migrate_v2_to_v3`` performed (report-only there), extended with
    ``GROUP_CONCAT(id ORDER BY id)`` to name the offending rows in a
    deterministic (ascending id) order — SQLite does not otherwise guarantee
    ``GROUP_CONCAT`` row order, and both the ERROR log token and the
    escalation payload's ``task_ids`` list depend on a stable order.

    Every residual group is classified via ``_classify_residual_group``:

    * **Heal** — a genuine content-duplicate (every row verified to
      recompute the SAME candidate_key; no row ``done``): the non-canonical
      rows are cancelled directly (``UPDATE ... SET status = 'cancelled'``),
      each stamped with a durable ``auto_cancelled_by_self_heal`` metadata
      provenance marker (canonical id + candidate_key) merged onto its
      existing metadata, plus a loud per-group WARNING log. Fixes reify
      incident esc-candidate-key-migration-2 (37 dup groups / 58 rows that
      previously required a manual ``set_task_status`` cancel per row).
    * **Flag** — ambiguous (``reason`` is ``'mixed_status'`` or
      ``'title_divergent'``): left untouched and collected — with its
      ``reason`` — into ``flagged_groups`` for escalation below.

    * **Any group ends up flagged** — log a loud ERROR naming the flagged
      groups and their reasons (via the ``residual_group_count=`` token,
      deliberately distinct from v2->v3's ``duplicate_groups=`` token so
      the two audits' log-scraping assertions never collide; also names
      ``healed_group_count=`` so an operator sees both halves of the
      pass), invoke ``residual_dup_escalation_cb(project_root,
      flagged_groups)`` when provided — ONLY the flagged groups, never the
      auto-healed ones — (best-effort — a raising callback is caught and
      logged, never propagated), and SKIP the index build. ``user_version``
      is left at 3 (NOT stamped to 4): a later connection-open — after an
      operator resolves the flagged residuals — re-runs this step and
      lands the index then (PRD decision #4 amendment: genuine duplicates
      now self-heal automatically; only ambiguous residuals still require
      a human before the index lands).
    * **Nothing flagged** (whether or not anything was healed) — build
      ``ux_tasks_candidate_key``, a PARTIAL UNIQUE index over ``(tag,
      candidate_key)`` excluding NULL keys and cancelled rows, then stamp
      ``user_version = 4`` — in the SAME connection-open that performed the
      heal, no restart required.

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

    Returns a result dict ``{'index_built': bool, 'healed': [...],
    'flagged': [...]}`` at every exit -- ``healed``/``flagged`` are lists of
    per-group descriptor dicts (``flagged`` entries are the same shape fed to
    ``residual_dup_escalation_cb``; ``healed`` entries carry ``tag``/
    ``candidate_key``/``canonical_id``/``cancelled_ids``). The connection-open
    call site (``_migrate``) ignores this return value -- it exists so
    ``SqliteTaskBackend.reaudit_candidate_key_index`` (the live, on-demand
    re-run) can share this single implementation and report an accurate
    status without a server restart.
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

        healed_groups: list[dict[str, Any]] = []
        flagged_groups: list[dict[str, Any]] = []
        for group in residual_rows:
            rows_cursor = await conn.execute(
                "SELECT id, title, status, metadata, candidate_key FROM tasks "
                "WHERE tag = ? AND candidate_key = ? AND status != 'cancelled' "
                "ORDER BY id",
                (group['tag'], group['candidate_key']),
            )
            group_rows = list(await rows_cursor.fetchall())
            classification = _classify_residual_group(group_rows)
            if len(classification) != 3:
                _, reason = classification
                flagged_groups.append({
                    'tag': group['tag'],
                    'candidate_key': group['candidate_key'],
                    'task_ids': [str(row['id']) for row in group_rows],
                    'count': len(group_rows),
                    'reason': reason,
                })
                continue

            _, canonical_id, cancel_ids = classification
            by_id = {row['id']: row for row in group_rows}
            now = _now()
            for cancel_id in cancel_ids:
                stamp = json.dumps({
                    'auto_cancelled_by_self_heal': {
                        'canonical_id': canonical_id,
                        'candidate_key': group['candidate_key'],
                        'migration': 'v3_v4',
                    },
                })
                new_metadata = _merge_metadata(
                    by_id[cancel_id]['metadata'], stamp,
                    mode='merge',
                    project_root=project_root, tag=group['tag'], task_id=cancel_id,
                )
                await conn.execute(
                    "UPDATE tasks SET status = 'cancelled', updated_at = ?, "
                    "metadata = ? WHERE tag = ? AND id = ?",
                    (now, new_metadata, group['tag'], cancel_id),
                )
            healed_groups.append({
                'tag': group['tag'],
                'candidate_key': group['candidate_key'],
                'canonical_id': canonical_id,
                'cancelled_ids': cancel_ids,
            })
            logger.warning(
                'sqlite_task_backend: schema v3->v4 self-heal -- auto-cancelled '
                '%d redundant duplicate candidate_key row(s) for tag=%r '
                'candidate_key=%r; canonical survivor id=%d retained, '
                'cancelled_ids=%s (genuine content-duplicate group -- verified '
                'same candidate_key, no done row -- safe to auto-collapse; see '
                'fm-task-dedup self-heal amendment, reify incident '
                'esc-candidate-key-migration-2)',
                len(cancel_ids), group['tag'], group['candidate_key'],
                canonical_id, cancel_ids,
            )

        if healed_groups:
            # Commit unconditionally here (not deferred to the clean-build
            # commit below): a still-flagged group below causes an early
            # `return`, and healed cancels must survive that skip rather
            # than riding on a commit that may never happen.
            await conn.commit()

        if flagged_groups:
            groups_desc = '; '.join(
                f'tag={g["tag"]!r} candidate_key={g["candidate_key"]!r} '
                f'ids=[{",".join(g["task_ids"])}] reason={g["reason"]!r}'
                for g in flagged_groups
            )
            logger.error(
                'sqlite_task_backend: schema v3->v4 migration SKIPPED -- '
                'residual_group_count=%d ambiguous duplicate candidate_key '
                'group(s) still flagged for human review '
                '(healed_group_count=%d genuine duplicate group(s) '
                'auto-resolved this pass); UNIQUE index NOT created, '
                'user_version stays at 3. Resolve the flagged group(s) '
                '(cancel or merge the extras) and the next connection-open '
                'will land the index. Groups: %s',
                len(flagged_groups), len(healed_groups), groups_desc,
            )

            if residual_dup_escalation_cb is not None:
                try:
                    residual_dup_escalation_cb(project_root, flagged_groups)
                except Exception:
                    # A broken/misbehaving callback must never crash
                    # connection-open — the skip above has already happened;
                    # escalation is purely additive.
                    logger.exception(
                        'sqlite_task_backend: residual_dup_escalation_cb '
                        'raised while escalating %d residual duplicate '
                        'candidate_key group(s) for project_root=%r',
                        len(flagged_groups), project_root,
                    )
            return {'index_built': False, 'healed': healed_groups, 'flagged': flagged_groups}

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
            return {'index_built': False, 'healed': healed_groups, 'flagged': []}

        await conn.execute('PRAGMA user_version = 4')
        await conn.commit()
        logger.info(
            'sqlite_task_backend: schema v3->v4 migration -- residual audit '
            'clean; built partial UNIQUE index ux_tasks_candidate_key over '
            '(tag, candidate_key) and advanced user_version to 4 '
            '(fm-task-dedup task A2).',
        )
        return {'index_built': True, 'healed': healed_groups, 'flagged': []}
    except Exception:
        # Defensive backstop -- this step must NEVER raise at connection-open
        # (see docstring): a raising migration would crash-loop fused-memory.
        logger.exception(
            'sqlite_task_backend: schema v3->v4 migration failed unexpectedly; '
            'skipping (user_version stays below 4, retried on next open)',
        )
        return {'index_built': False, 'healed': [], 'flagged': []}


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


async def _candidate_key_index_present(conn: aiosqlite.Connection) -> bool:
    """True iff the ``ux_tasks_candidate_key`` partial UNIQUE index exists on ``tasks``.

    Feature-detection backing the add_task/update_task pre-write dedup
    guards' hot-path skip (fm-task-dedup self-heal amendment, reviewer
    follow-up): once the index is present it alone enforces (tag,
    candidate_key) uniqueness for non-cancelled rows (backstopped by the
    existing post-write ``sqlite3.IntegrityError`` mapping), so the extra
    guard SELECT those write paths run as an index-INDEPENDENT backstop is
    only needed while the index is absent -- the v3->v4 migration's
    self-gated window (a flagged residual leaves it absent indefinitely,
    reify incident esc-candidate-key-migration-2).

    Called once per project_root, from ``_get_connection`` right after
    ``_migrate`` runs and again by ``reaudit_candidate_key_index`` after a
    live rebuild; the result is cached on
    ``SqliteTaskBackend._candidate_key_index_cache`` and reused by the write
    paths so a hot add_task/update_task loop doesn't re-run ``PRAGMA
    index_list`` on every call.
    """
    index_rows = await (await conn.execute('PRAGMA index_list(tasks)')).fetchall()
    return any(row[1] == 'ux_tasks_candidate_key' for row in index_rows)


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
        self._after_insert_fault_hook: Callable[[], None] | None = None
        self._connections: dict[str, aiosqlite.Connection] = {}
        # Cached per-project AUTOCOMMIT (isolation_level=None) read
        # connections for the hot get_statuses/get_statuses_raw path (task
        # 2455). Distinct from self._connections (the write connection,
        # opened in Python sqlite3's legacy deferred-transaction mode): an
        # autocommit connection never holds a read transaction open across
        # statements, so it can never be pinned to a stale WAL snapshot the
        # way the write connection can be. See _get_read_connection.
        self._read_connections: dict[str, aiosqlite.Connection] = {}
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
        # Cached result of `_candidate_key_index_present` per project_root
        # (reviewer follow-up to fm-task-dedup self-heal amendment):
        # populated in `_get_connection` right after `_migrate` runs, and
        # refreshed by `reaudit_candidate_key_index`. Lets add_task/
        # update_task skip their index-independent pre-write dedup guard
        # SELECT once the partial UNIQUE index is confirmed present -- the
        # common steady state, where the index (backstopped by the existing
        # post-write IntegrityError mapping) alone enforces uniqueness.
        # Absent entries default to False (fail-safe: run the guard) so an
        # unmigrated/absent cache entry never silently skips protection.
        self._candidate_key_index_cache: dict[str, bool] = {}
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
            read_connection_items = list(self._read_connections.items())
            self._read_connections.clear()
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
        # Read connections (task 2455) are autocommit and never write, so
        # there's no WAL checkpoint to run — just close them, best-effort.
        for root, read_conn in read_connection_items:
            with contextlib.suppress(Exception):
                await read_conn.close()
            logger.debug('SqliteTaskBackend closed read connection for %s', root)
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
            self._candidate_key_index_cache[project_root] = await _candidate_key_index_present(conn)
            self._connections[project_root] = conn
            logger.info('SqliteTaskBackend opened %s', db_path)
            return conn

    async def _get_read_connection(self, project_root: str) -> aiosqlite.Connection:
        """Return a cached per-project AUTOCOMMIT connection for hot status reads.

        Used by :meth:`get_statuses_raw` (task 2455). Unlike
        :meth:`_get_connection`'s cached connection — opened in Python
        sqlite3's legacy deferred-transaction mode, so a read transaction
        left open on it pins a stale WAL snapshot (task 2388) — this
        connection is opened with ``isolation_level=None`` (autocommit), so
        it never holds a transaction open across statements and can never
        be pinned. It uses the exact same open recipe as
        :meth:`get_statuses_fresh` (``connect_daemon(..., isolation_level=
        None)`` + ``apply_wal_pragmas``), but caches the result per
        ``project_root`` so repeated hot-path calls don't pay a per-call
        connection-open cost.
        """
        if self._closed:
            raise RuntimeError('SqliteTaskBackend is closed')
        # Ensure the DB file + schema + migrations exist before opening our
        # own connection onto the same file.
        await self._get_connection(project_root)
        if project_root in self._read_connections:
            return self._read_connections[project_root]

        async with self._connect_locks_lock:
            # Re-check after acquiring lock — another caller may have raced us.
            conn = self._read_connections.get(project_root)
            if conn is not None:
                return conn

            conn = await connect_daemon(str(self._db_path(project_root)), isolation_level=None)
            await apply_wal_pragmas(conn, busy_timeout_ms=5000)
            conn.row_factory = aiosqlite.Row
            self._read_connections[project_root] = conn
            logger.info('SqliteTaskBackend opened read connection for %s', project_root)
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

    async def _statuses_from_conn(
        self,
        conn: aiosqlite.Connection,
        tag: str,
        ids: list[str] | None,
    ) -> dict[str, str]:
        """Run the id/status SELECT on *conn* and coerce to ``{id_str: status_str}``.

        Shared body for :meth:`get_statuses_raw` (cached per-project
        connection) and :meth:`get_statuses_fresh` (dedicated short-lived
        connection) — a single source of truth for the ``ids`` filtering
        and the ``NULL`` → ``'unknown'`` coercion rule, so both read paths
        stay identical apart from which connection they run on.

        Args:
            conn: An open connection with ``row_factory`` already set to
                :class:`aiosqlite.Row` (callers own connection setup).
            tag: Tag context — callers apply the ``tag or DEFAULT_TAG``
                default before calling this helper.
            ids: When given, only return entries for these task ids (as strings;
                 cast to int for the SQL IN clause; non-numeric ids silently
                 omitted).  ``None`` returns all tasks.  ``[]`` returns ``{}``.

        Returns:
            ``{str(id): status}`` mapping.  Unknown ids are silently omitted.
            A ``NULL`` status (defensive; unreachable via normal writes) maps to
            ``'unknown'``.
        """
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

        Reads via the cached per-project AUTOCOMMIT connection returned by
        :meth:`_get_read_connection` (task 2455) rather than the cached
        WRITE connection (:meth:`_get_connection`) that :meth:`get_task`/
        :meth:`get_tasks` use — see :meth:`_get_read_connection` and
        :meth:`get_statuses_fresh` for why a pinnable connection can go
        stale here.

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
        conn = await self._get_read_connection(project_root)
        return await self._statuses_from_conn(conn, tag, ids)

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

    async def get_statuses_fresh(
        self,
        project_root: str,
        ids: list[str] | None = None,
        tag: str | None = None,
    ) -> dict[str, str]:
        """Return a snapshot-fresh ``{id_str: status_str}`` census for *project_root*.

        Unlike :meth:`get_statuses`/:meth:`get_statuses_raw`, which read via
        the cached per-project connection returned by :meth:`_get_connection`,
        this method opens a DEDICATED short-lived connection with
        ``isolation_level=None`` (autocommit) for every call, so its SELECT
        always sees the latest committed WAL state and can never be pinned.

        Why this exists (task 2388): ``_get_connection``'s cached connection
        is opened via ``connect_daemon(str(db_path))`` *without*
        ``isolation_level=None`` — Python sqlite3's legacy deferred
        transaction mode. If a read transaction is ever left open on that
        cached connection, every subsequent read on it — including
        ``get_statuses``/``get_statuses_raw`` and the ``get_tasks`` tree
        read, since they all share the same cached connection — is pinned
        to that transaction's WAL snapshot and silently returns stale data,
        even after other connections/processes have committed newer writes.
        Because the tree read and the census read went stale *together*,
        they still agreed with each other, so
        ``cross_verify_task_counts`` (``reconciliation/task_filter.py``)
        reported a false ``consistent: true`` instead of surfacing the
        drift. This method gives the reconciliation harness's authoritative
        census (``_fetch_task_count_census``) a read that cannot be pinned;
        the hot compact-status-map callers should keep using
        ``get_statuses``/``get_statuses_raw`` unchanged.

        Fails open to ``{}`` on any error (including a not-yet-created DB
        file) — this is a best-effort freshness upgrade for a cross-check,
        never a reason to raise into the reconciliation cycle.

        Args:
            project_root: Absolute path to the project root.
            ids: When given, only return entries for these task ids (as
                 strings).  ``None`` returns all tasks.  ``[]`` returns ``{}``.
            tag: Tag context; defaults to ``DEFAULT_TAG`` when ``None``.

        Returns:
            ``{str(id): status}`` mapping; ``{}`` if the DB file does not
            exist yet or the read fails for any reason.
        """
        db_path = self._db_path(project_root)
        if not db_path.exists():
            return {}
        # conn is opened *inside* the try (not before it) so that a failure
        # to even open the dedicated connection — e.g. a permission error,
        # disk I/O failure, or corrupt file — also fails open to {} per the
        # "any error" contract documented above, instead of propagating.
        conn: aiosqlite.Connection | None = None
        try:
            conn = await connect_daemon(str(db_path), isolation_level=None)
            await apply_wal_pragmas(conn, busy_timeout_ms=5000)
            conn.row_factory = aiosqlite.Row
            return await self._statuses_from_conn(conn, tag or DEFAULT_TAG, ids)
        except Exception:
            logger.warning(
                "get_statuses_fresh: failing open to {} for project_root=%r "
                "after error reading fresh status census",
                project_root,
                exc_info=True,
            )
            return {}
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    await conn.close()

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

    async def reaudit_candidate_key_index(self, project_root: str) -> dict[str, Any]:
        """Idempotently re-run the v3->v4 self-heal migration on the LIVE
        cached connection for ``project_root`` -- no server restart required.

        ``_migrate_v3_to_v4`` only ever runs at connection-open:
        ``_get_connection`` short-circuits to the cached connection on every
        later call, so a running server that already holds a pre-audit
        connection never re-lands the partial UNIQUE index after an operator
        resolves a previously-flagged residual (the reify incident's second
        failure mode, esc-candidate-key-migration-2). This method closes
        that gap by re-running the SAME classify/heal/audit/build logic
        connection-open uses, sharing the implementation via
        ``_migrate_v3_to_v4``'s result dict (fm-task-dedup self-heal
        amendment).

        Short-circuits to ``{'index_built': True, 'already_at_v4': True,
        'user_version': 4}`` without touching the write lock when the
        connection is already at v4 (the common case once the index has
        landed) — a plain read, so no locking is needed for the check
        itself. Otherwise acquires the write lock (excluding concurrent
        writers for the duration of the re-audit, same as any other mutating
        method) and re-reads ``PRAGMA user_version`` a SECOND time, now that
        the lock is held: a concurrent caller (another ``reaudit_...`` call,
        or a racing connection-open) may have landed the index between the
        first unlocked read and lock acquisition, and re-running
        ``_migrate_v3_to_v4`` against an already-v4 connection is merely
        wasteful (idempotent, but logs a confusing "clean audit (race?)"
        line) rather than unsafe — this second check avoids that. Returns
        ``_migrate_v3_to_v4``'s result dict merged with the final
        ``user_version`` when the migration does run.
        """
        await self.ensure_connected()
        conn = await self._get_connection(project_root)
        version_cursor = await conn.execute('PRAGMA user_version')
        version_row = await version_cursor.fetchone()
        current_version = version_row[0] if version_row is not None else 0
        if current_version >= 4:
            self._candidate_key_index_cache[project_root] = True
            return {'index_built': True, 'already_at_v4': True, 'user_version': 4}

        async with self._write_lock(project_root):
            conn = await self._get_connection(project_root)
            # Re-check under the lock (see docstring) — another writer may
            # have already landed the index while we were waiting for it.
            version_cursor = await conn.execute('PRAGMA user_version')
            version_row = await version_cursor.fetchone()
            current_version = version_row[0] if version_row is not None else current_version
            if current_version >= 4:
                self._candidate_key_index_cache[project_root] = True
                return {'index_built': True, 'already_at_v4': True, 'user_version': 4}

            result = await _migrate_v3_to_v4(
                conn,
                project_root=project_root,
                residual_dup_escalation_cb=self._residual_dup_escalation_cb,
            )
            version_cursor = await conn.execute('PRAGMA user_version')
            version_row = await version_cursor.fetchone()
            final_version = version_row[0] if version_row is not None else current_version
            self._candidate_key_index_cache[project_root] = final_version >= 4

        return {**result, 'user_version': final_version}

    # ── Write-boundary validation (task 2162, W3-β) ────────────────────

    async def _validate_metadata_on_write(
        self,
        metadata: str | None,
        *,
        project_root: str,
        tag: str,
        task_id: int,
        incoming_keys: set[str] | None = None,
    ) -> None:
        """Validate a ``metadata`` JSON blob at the add_task/update_task write boundary.

        Always parses once in warn-mode (``enforce=False``) so every
        :class:`SchemaWarning` the blob produces — including ones that would
        otherwise raise — is logged as one ``task_metadata.schema_warning``
        census line (unchanged census behaviour).

        In enforce-mode, a second, authoritative pass decides whether to
        raise. ``incoming_keys`` — the top-level keys the CURRENT write
        actually supplied (``None`` means "unknown/whole-blob", e.g.
        ``add_task``, which enforces every field) — scopes the raise to
        findings the write is responsible for (task 2401): a warning whose
        ``field`` is in ``incoming_keys``, or the whole-blob sentinel
        (:data:`_WHOLE_METADATA_FIELD`, always fatal), re-triggers validation
        with ``enforce=True`` so the authentic ``ValidationError`` /
        ``ValueError`` / ``TypeError`` propagates uncaught — the caller's
        ``_txn`` rolls back. This tolerates an untouched legacy field (e.g. a
        pre-existing ``done_provenance`` missing the now-required ``kind``)
        rather than blocking every future write to the row. An incoming-key
        warning only opens this gate when its code is genuinely fatal (i.e.
        not in :data:`_NON_FATAL_WRITE_WARNING_CODES`): an ``unknown_key``
        warning is never fatal under ``enforce=True`` (``TaskMetadata``
        accepts unrecognised top-level keys via ``extra='allow'``), so a
        patch whose OWN keys are themselves unrecognised (e.g. a
        reconciliation sidecar patch) must not, by itself, re-trigger
        whole-blob validation and trip over an untouched fatal field
        elsewhere in the row (task 2405).

        ``project_root``/``tag`` are accepted but not yet read by this method
        — the census line only carries ``task_id``/field/error. They mirror
        ``_warn_malformed_metadata_once``'s ``(project_root, tag, task_id)``
        triple so a future write-side census enrichment (e.g. scoping/dedup
        by project) can use them without changing either call site's
        signature.
        """
        _, warnings = parse_metadata(metadata, direction='write', enforce=False)
        for warning in warnings:
            _emit_schema_warning(task_id, warning)
        if self._task_metadata_enforce and warnings:
            if incoming_keys is None:
                should_reraise = True
            else:
                should_reraise = any(
                    (w.field == _WHOLE_METADATA_FIELD or w.field in incoming_keys)
                    and w.code not in _NON_FATAL_WRITE_WARNING_CODES
                    for w in warnings
                )
            if should_reraise:
                parse_metadata(metadata, direction='write', enforce=True)

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

                # Index-independent dedup guard (fm-task-dedup self-heal
                # amendment): the v3->v4 partial UNIQUE index is
                # self-gating — a flagged residual leaves it ABSENT
                # indefinitely (reify incident esc-candidate-key-migration-2)
                # — so this SELECT is the only backstop during that window.
                # Same typed error the post-INSERT IntegrityError mapping
                # below raises when the index IS present, so callers (the
                # interceptor's create-dispatch combined resolution) need no
                # changes regardless of which path catches the collision.
                # Gated on `_candidate_key_index_cache` (reviewer follow-up):
                # once the index is confirmed present, IT alone enforces
                # uniqueness (via the IntegrityError mapping below), so this
                # extra SELECT is skipped on that hot-path steady state and
                # only runs during the degraded index-absent window. A
                # missing cache entry defaults to "absent" (fail-safe).
                if candidate_key is not None and not self._candidate_key_index_cache.get(
                    project_root, False,
                ):
                    guard_cursor = await conn.execute(
                        "SELECT id, status FROM tasks WHERE tag = ? AND candidate_key = ? "
                        "AND status != 'cancelled' ORDER BY id LIMIT 1",
                        (tag, candidate_key),
                    )
                    guard_row = await guard_cursor.fetchone()
                    if guard_row is not None:
                        raise DuplicateCandidateKeyError(
                            existing_id=guard_row['id'],
                            existing_status=guard_row['status'],
                            tag=tag,
                            candidate_key=candidate_key,
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
        # Write-authority floors mirroring the server/tools.py + interceptor
        # ceiling (2026-05-08 forensics). set_task_status is the only
        # sanctioned writer for status AND metadata.done_provenance — it
        # enforces the terminal-exit, phantom-done, and done-provenance
        # gates. Both floors reject unconditionally, before ensure_connected()
        # and the task SELECT, so a write-authority rejection takes
        # precedence over any existence or connection error.
        if status is not None:
            raise StatusWriteAuthorityError(task_id, status)
        if metadata is not None:
            # Mirror the interceptor's _reject_done_provenance_in_update_metadata:
            # accept an already-parsed dict directly before falling back to
            # json.loads. A caller that bypasses the documented ``str | None``
            # signature and passes a dict would otherwise hit
            # ``json.loads(dict)`` -> TypeError -> parsed_metadata=None,
            # silently permitting a done_provenance write past this floor.
            if isinstance(metadata, dict):
                parsed_metadata: dict | None = metadata
            else:
                try:
                    parsed_metadata = json.loads(metadata)
                except (ValueError, TypeError):
                    parsed_metadata = None
            if isinstance(parsed_metadata, dict) and 'done_provenance' in parsed_metadata:
                raise DoneProvenanceWriteAuthorityError(task_id)
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
                #
                # incoming_keys (task 2401, fix b): the top-level keys of THIS
                # write's own payload — not the post-merge blob — so enforce
                # mode only blocks on findings this write is responsible for.
                # An untouched legacy field (e.g. a pre-existing
                # done_provenance missing the now-required kind) is tolerated
                # rather than permanently blocking every future patch to the
                # row. Falls back to None (enforce-all) when the payload
                # isn't a JSON object — _merge_metadata already treats a
                # non-dict incoming blob as last-write-wins, so there is no
                # narrower "responsibility" to scope to here.
                incoming_keys: set[str] | None = None
                try:
                    _loaded_incoming = json.loads(metadata)
                except (TypeError, ValueError):
                    pass
                else:
                    if isinstance(_loaded_incoming, dict):
                        incoming_keys = set(_loaded_incoming.keys())
                await self._validate_metadata_on_write(
                    new_metadata, project_root=project_root, tag=tag, task_id=tid,
                    incoming_keys=incoming_keys,
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

                # Index-independent dedup guard (fm-task-dedup self-heal
                # amendment): mirrors add_task's pre-INSERT guard. The v3->v4
                # partial UNIQUE index is self-gating — a flagged residual
                # leaves it ABSENT indefinitely (reify incident
                # esc-candidate-key-migration-2) — so this SELECT is the
                # only backstop against reactivating the recomputed key onto
                # another non-cancelled row during that window. Same typed
                # error the post-UPDATE IntegrityError mapping below raises
                # when the index IS present, so callers need no changes
                # regardless of which path catches the collision. Excludes
                # this row's own id — an update that recomputes to the key
                # it already holds is not a collision. Gated on
                # `_candidate_key_index_cache` (reviewer follow-up) — see
                # add_task's identical guard for why this SELECT is skipped
                # once the index is confirmed present.
                if new_candidate_key is not None and not self._candidate_key_index_cache.get(
                    project_root, False,
                ):
                    guard_cursor = await conn.execute(
                        "SELECT id, status FROM tasks WHERE tag = ? AND candidate_key = ? "
                        "AND status != 'cancelled' AND id != ? ORDER BY id LIMIT 1",
                        (tag, new_candidate_key, tid),
                    )
                    guard_row = await guard_cursor.fetchone()
                    if guard_row is not None:
                        raise DuplicateCandidateKeyError(
                            existing_id=guard_row['id'],
                            existing_status=guard_row['status'],
                            tag=tag,
                            candidate_key=new_candidate_key,
                        )

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

    async def stamp_audit_metadata(
        self,
        task_id: str,
        project_root: str,
        fields: dict,
        tag: str | None = None,
    ) -> dict:
        """Privileged, non-protocol writer of done_provenance/reopen_* audit fields.

        Reachable only from :class:`TaskInterceptor` (PRD C-C) — ``update_task``
        remains the sole PUBLIC metadata writer and unconditionally rejects
        ``metadata.done_provenance`` (see the floor above). Performs a
        read-modify-write merge under the same write-lock + txn pattern as
        ``update_task``/``set_task_claimant``: last-write-wins on the supplied
        keys, preserving every omitted sibling key (``memory_hints``, ``files``,
        ``external_deps``, ...) via ``_merge_metadata(mode='merge')``.

        Deliberately NOT declared on :class:`TaskBackendProtocol` — keeping it
        off the 12-method contract prevents other callers from treating it as
        sanctioned public surface.
        """
        await self.ensure_connected()
        tag = tag or DEFAULT_TAG
        tid = _parse_task_id(task_id)
        async with self._write_lock(project_root), self._txn(project_root) as conn:
            cursor = await conn.execute(
                'SELECT metadata FROM tasks WHERE tag = ? AND id = ?',
                (tag, tid),
            )
            row = await cursor.fetchone()
            if row is None:
                raise TaskmasterError(
                    'TASKMASTER_TOOL_ERROR',
                    f'No tasks found for ID(s): {task_id}',
                )
            new_metadata = _merge_metadata(
                row['metadata'], json.dumps(fields),
                mode='merge',
                project_root=project_root, tag=tag, task_id=tid,
            )
            await conn.execute(
                'UPDATE tasks SET metadata = ?, updated_at = ? WHERE tag = ? AND id = ?',
                (new_metadata, _now(), tag, tid),
            )
        return {
            'id': task_id,
            'message': f'Stamped audit metadata for task {task_id}',
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
