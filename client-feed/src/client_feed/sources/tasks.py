"""The task store's allowlisted projection, the rule for whom an unfinished task waits on, and the carriers of this project's client defect reports.

Only the allowlisted columns and metadata keys are selected; the rest of a task never enters the process.
The carrier read adds exactly the two fields of plans/client-feed-prd.md §14.3: each `metadata.defects[].ref`
and `metadata.x_client_defect_gate_for` that begins with this project's portal ref prefix. Every other ref,
of another origin or another project, is never selected.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from pydantic import TypeAdapter, ValidationError
from shared.project_layout import tasks_db

from client_feed.contract import CommitSha, RepoPath, Source, TaskStatus, WaitingOn
from client_feed.defect_ref import PortalReport, parse_portal_ref, portal_ref_prefix
from client_feed.redact import redact_paths
from client_feed.sources.escalations import OpenEscalations
from client_feed.sources.store import read_only

TASK_TAG: Final = 'master'
LEO_LEVEL: Final = 2

_FINISHED: Final = frozenset({TaskStatus.DONE, TaskStatus.CANCELLED})


def waiting_on(status: TaskStatus, levels: Sequence[int]) -> WaitingOn | None:
    """Leo on an open level-2 escalation; the factory on any other, or on a block; no one once finished."""
    if status in _FINISHED:
        return None
    if LEO_LEVEL in levels:
        return WaitingOn.LEO
    if levels or status is TaskStatus.BLOCKED:
        return WaitingOn.FACTORY
    return None


class TaskSkipReason(StrEnum):
    BAD_UPDATED_AT = 'bad_updated_at'
    UNKNOWN_STATUS = 'unknown_status'


@dataclass(frozen=True)
class TaskRow:
    task_id: int
    title: str
    status: TaskStatus
    waiting_on: WaitingOn | None
    updated_at: datetime
    depends_on: tuple[int, ...]
    prd_path: str | None
    prd_task_label: str | None
    landed_commit: str | None


@dataclass(frozen=True)
class SkippedTask:
    task_id: int
    reason: TaskSkipReason


@dataclass(frozen=True)
class TasksRead:
    """The rows updated after a cursor; `cursor` is the last projected row's updated_at, as the store spells it.

    A skipped row never moves the cursor, so it is read again until a projected row passes it.
    """

    rows: tuple[TaskRow, ...]
    skipped: tuple[SkippedTask, ...]
    cursor: str | None


def _json_at(document: str, path: str, *, json_type: str, guard: str) -> str:
    """The value at `path` in `document`, NULL unless `guard` holds and the value is of `json_type`."""
    return (
        f'CASE WHEN {guard} THEN'
        f" CASE WHEN json_type({document}, '{path}') = '{json_type}' THEN json_extract({document}, '{path}') END"
        ' END'
    )


def _metadata_at(path: str, *, json_type: str = 'text') -> str:
    """The metadata value at `path`, NULL when metadata is not JSON or the value is not of `json_type`."""
    return _json_at('metadata', path, json_type=json_type, guard='json_valid(metadata)')


_UPDATED_AFTER: Final = 'tag = ? AND (? IS NULL OR updated_at > ?)'

_TASKS_SQL: Final = (
    'SELECT id, title, status, updated_at,'
    f" {_metadata_at('$.prd_path')},"
    f" {_metadata_at('$.prd_task_label')},"
    f" {_metadata_at('$.done_provenance.commit')}"
    f' FROM tasks WHERE {_UPDATED_AFTER} ORDER BY updated_at, id'
)

_DEPENDENCIES_SQL: Final = (
    'SELECT task_id, depends_on FROM dependencies'
    f' WHERE tag = ? AND task_id IN (SELECT id FROM tasks WHERE {_UPDATED_AFTER})'
    ' ORDER BY task_id, depends_on'
)

_REPO_PATH: Final = TypeAdapter(RepoPath)
_COMMIT: Final = TypeAdapter(CommitSha)


def _utc_instant(text: str) -> datetime | None:
    """The instant a UTC ISO timestamp names; any other spelling, offset or none, is refused."""
    try:
        value = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    offset = value.utcoffset()
    return value.astimezone(UTC) if offset is not None and not offset else None


def _validated(adapter: TypeAdapter[str], value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return adapter.validate_python(value)
    except ValidationError:
        return None


def _repo_path(root: Path, value: str | None) -> str | None:
    """A repo-relative path; an absolute one is made relative when it lies under the root, else dropped."""
    if value is not None and Path(value).is_absolute():
        path = Path(value)
        value = path.relative_to(root).as_posix() if path.is_relative_to(root) else None
    return _validated(_REPO_PATH, value)


@dataclass(frozen=True)
class _Context:
    root: Path
    depends_on: Mapping[int, tuple[int, ...]]
    escalations: OpenEscalations


def _projected(raw: tuple[Any, ...], context: _Context) -> TaskRow | SkippedTask:
    task_id, title, status, updated_at, prd_path, prd_task_label, landed_commit = raw
    when = _utc_instant(updated_at)
    if when is None:
        return SkippedTask(task_id=task_id, reason=TaskSkipReason.BAD_UPDATED_AT)
    try:
        task_status = TaskStatus(status)
    except ValueError:
        return SkippedTask(task_id=task_id, reason=TaskSkipReason.UNKNOWN_STATUS)
    return TaskRow(
        task_id=task_id,
        title=redact_paths(title),
        status=task_status,
        waiting_on=waiting_on(task_status, context.escalations.levels(task_id)),
        updated_at=when,
        depends_on=context.depends_on.get(task_id, ()),
        prd_path=_repo_path(context.root, prd_path),
        prd_task_label=None if prd_task_label is None else redact_paths(prd_task_label),
        landed_commit=_validated(_COMMIT, landed_commit),
    )


def _grouped(pairs: list[tuple[int, int]]) -> dict[int, tuple[int, ...]]:
    grouped: dict[int, tuple[int, ...]] = {}
    for task_id, dependency in pairs:
        grouped[task_id] = (*grouped.get(task_id, ()), dependency)
    return grouped


def _cursor(
    rows: list[tuple[Any, ...]], projected: list[TaskRow | SkippedTask], since: str | None
) -> str | None:
    """The raw updated_at of the last row projected; a skipped row's value is not trusted to order."""
    return next(
        (
            updated_at
            for (_, _, _, updated_at, *_), item in zip(reversed(rows), reversed(projected), strict=True)
            if isinstance(item, TaskRow)
        ),
        since,
    )


def read_tasks(root: Path, *, since: str | None, escalations: OpenEscalations) -> TasksRead:
    """Every task of the pinned tag updated after `since`, projected to its allowlisted fields."""
    window = (TASK_TAG, since, since)
    with read_only(root, tasks_db(root), source=Source.TASKS) as conn:
        rows = conn.execute(_TASKS_SQL, window).fetchall()
        dependencies = conn.execute(_DEPENDENCIES_SQL, (TASK_TAG, *window)).fetchall()
    context = _Context(root=root, depends_on=_grouped(dependencies), escalations=escalations)
    projected = [_projected(row, context) for row in rows]
    return TasksRead(
        rows=tuple(item for item in projected if isinstance(item, TaskRow)),
        skipped=tuple(item for item in projected if isinstance(item, SkippedTask)),
        cursor=_cursor(rows, projected, since),
    )


@dataclass(frozen=True)
class DefectCarrier:
    """The task carrying a client defect report of this project, and whether a human gate holds it."""

    ref: str
    block: int
    report: int
    task_id: int
    gated: bool


_PREFIXED: Final = 'substr(ref, 1, length(:prefix)) = :prefix'

_DEFECTS_ARRAY: Final = _metadata_at('$.defects', json_type='array')

_ENTRY_REF: Final = _json_at('entry.value', '$.ref', json_type='text', guard="entry.type = 'object'")

_HELD_REFS_SQL: Final = (
    'SELECT id, status, ref FROM ('
    f'SELECT tasks.id AS id, tasks.status AS status, {_ENTRY_REF} AS ref'
    f' FROM tasks, json_each({_DEFECTS_ARRAY}) AS entry WHERE tasks.tag = :tag'
    f') WHERE {_PREFIXED}'
)

_AMONG_TASK_IDS: Final = 'IN (SELECT value FROM json_each(:task_ids))'

_CARRIER_DEPENDENCIES_SQL: Final = (
    f'SELECT task_id, depends_on FROM dependencies WHERE tag = :tag AND task_id {_AMONG_TASK_IDS}'
)

_GATES_SQL: Final = (
    f"SELECT id, ref FROM (SELECT id, {_metadata_at('$.x_client_defect_gate_for')} AS ref"
    f' FROM tasks WHERE tag = :tag AND id {_AMONG_TASK_IDS}) WHERE {_PREFIXED}'
)


def _carries(status: str) -> bool:
    """A task carries its reports in any status but deferred; an unknown status carries nothing."""
    try:
        return TaskStatus(status) is not TaskStatus.DEFERRED
    except ValueError:
        return False


def _lowest_holders(
    held: list[tuple[int, str, str]], *, project_id: str
) -> dict[str, tuple[int, PortalReport]]:
    """Each ref of `project_id` a carrying task holds, with the lowest such task id: the first filing."""
    lowest: dict[str, tuple[int, PortalReport]] = {}
    for task_id, status, ref in held:
        named = parse_portal_ref(ref, project_id=project_id)
        if named is not None and _carries(status):
            lowest[ref] = min(lowest.get(ref, (task_id, named)), (task_id, named))
    return lowest


def _by_task_id(
    conn: sqlite3.Connection, sql: str, task_ids: Iterable[int], **params: str
) -> list[tuple[Any, ...]]:
    """The rows `sql` selects for `task_ids` of the pinned tag, bound as one JSON array however many ids."""
    bound = {'tag': TASK_TAG, 'task_ids': json.dumps(sorted(set(task_ids))), **params}
    return conn.execute(sql, bound).fetchall()


def read_defect_carriers(root: Path, *, project_id: str) -> tuple[DefectCarrier, ...]:
    """Every client defect report of `project_id` some task carries, sorted by ref.

    Read in full every pass: a same-project add_dependency inserts the gate edge without bumping the
    task's updated_at, so no updated_at cursor would see `gated` turn true. Only the held refs scan
    every task's metadata; the carriers' edges and their gates are looked up by task id.
    """
    prefix = portal_ref_prefix(project_id)
    with read_only(root, tasks_db(root), source=Source.TASKS) as conn:
        held = conn.execute(_HELD_REFS_SQL, {'tag': TASK_TAG, 'prefix': prefix}).fetchall()
        lowest = _lowest_holders(held, project_id=project_id)
        carriers = (task_id for task_id, _ in lowest.values())
        depends_on = _grouped(_by_task_id(conn, _CARRIER_DEPENDENCIES_SQL, carriers))
        dependencies = (dependency for edges in depends_on.values() for dependency in edges)
        gate_for = dict(_by_task_id(conn, _GATES_SQL, dependencies, prefix=prefix))
    return tuple(
        DefectCarrier(
            ref=ref,
            block=named.block,
            report=named.report,
            task_id=task_id,
            gated=any(gate_for.get(dependency) == ref for dependency in depends_on.get(task_id, ())),
        )
        for ref, (task_id, named) in sorted(lowest.items())
    )
