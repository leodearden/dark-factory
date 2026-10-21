"""The task store's allowlisted projection, and the rule for whom an unfinished task waits on.

Only the allowlisted columns and metadata keys are selected; the rest of a task never enters the process.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from pydantic import TypeAdapter, ValidationError
from shared.project_layout import tasks_db

from client_feed.contract import CommitSha, RepoPath, Source, TaskStatus, WaitingOn
from client_feed.redact import redact_paths
from client_feed.sources.events import OpenEscalations
from client_feed.sources.store import read_only

TASK_TAG: Final = 'master'
LEO_LEVEL: Final = 2

_FINISHED: Final = frozenset({TaskStatus.DONE, TaskStatus.CANCELLED})


def waiting_on(status: TaskStatus, levels: Sequence[int | None]) -> WaitingOn | None:
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
    """The rows updated after a cursor; `cursor` is the greatest updated_at read, as the store spells it."""

    rows: tuple[TaskRow, ...]
    skipped: tuple[SkippedTask, ...]
    cursor: str | None


def _metadata_text(path: str) -> str:
    """The metadata string at `path`, NULL when metadata is not JSON or the value is not a string."""
    return (
        'CASE WHEN json_valid(metadata) THEN'
        f" CASE WHEN json_type(metadata, '{path}') = 'text' THEN json_extract(metadata, '{path}') END"
        ' END'
    )


_UPDATED_AFTER: Final = 'tag = ? AND (? IS NULL OR updated_at > ?)'

_TASKS_SQL: Final = (
    'SELECT id, title, status, updated_at,'
    f" {_metadata_text('$.prd_path')},"
    f" {_metadata_text('$.prd_task_label')},"
    f" {_metadata_text('$.done_provenance.commit')}"
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
        cursor=rows[-1][3] if rows else since,
    )
