"""The row a dispatch acts on: the claimed winner re-read from the store after its claim."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from shared.task_fields import dependency_ids
from shared.task_read import TaskAbsent, TaskFound, TaskRead, TaskReadFailed

__all__ = ['FreshRow', 'RowSource', 'refresh_dispatch_row']


@dataclass(frozen=True)
class FreshRow:
    """The store's current row for a claimed task and the statuses of its local dependencies."""

    task: dict[str, Any]
    dep_statuses: Mapping[str, str]


class RowSource(Protocol):
    async def read_task(self, task_id: str, /) -> TaskRead: ...

    async def get_statuses(
        self, ids: list[str] | None = ...
    ) -> tuple[dict[str, str], Exception | None]: ...


async def refresh_dispatch_row(
    source: RowSource, row: Mapping[str, Any],
) -> FreshRow | TaskAbsent | TaskReadFailed:
    task_id = str(row['id'])
    read = await source.read_task(task_id)
    if not isinstance(read, TaskFound):
        return read
    deps = dependency_ids(read.task)
    if not deps:
        return FreshRow(read.task, {})
    statuses, err = await source.get_statuses(deps)
    if err is not None:
        return TaskReadFailed(
            f'statuses of task {task_id} dependencies {deps} unreadable: '
            f'{type(err).__name__}: {err}'
        )
    return FreshRow(read.task, statuses)
