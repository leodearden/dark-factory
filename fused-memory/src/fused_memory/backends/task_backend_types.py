"""Typed DTOs returned by every task backend's wrapper methods.

Each result is a flat ``TypedDict`` — callers read fields directly with
no defensive ``.get('data')`` unwrapping. Id-bearing fields surface as
``str`` regardless of how the underlying storage layer represents them.
"""

from __future__ import annotations

from typing import TypedDict


class AddTaskResult(TypedDict):
    id: str
    message: str


class UpdateTaskResult(TypedDict):
    id: str
    message: str
    updated: bool
    updated_task: dict | None


class SetTaskStatusResult(TypedDict):
    """DTO for ``set_task_status``.

    The wire shape is ``{message, tasks: [<per-id result>]}`` where each
    per-id result carries ``taskId`` and ``newStatus`` fields. The
    per-id list is surfaced as-is for callers that care; most callers
    only use ``message`` as a sanity check.
    """

    message: str
    tasks: list[dict]


class AddSubtaskResult(TypedDict):
    id: str
    parent_id: str
    message: str
    subtask: dict


class RemoveTaskResult(TypedDict):
    successful: int
    failed: int
    removed_ids: list[str]
    message: str


class DependencyResult(TypedDict):
    """DTO for both ``add_dependency`` and ``remove_dependency``."""

    id: str
    dependency_id: str
    message: str


class ValidateDependenciesResult(TypedDict):
    message: str


class GetTasksResult(TypedDict):
    """DTO for ``get_tasks`` — the task-tree snapshot.

    Task objects in ``tasks`` have their top-level ``id``, ``title``,
    ``status``, etc. already flattened — no further unwrapping needed.
    """

    tasks: list[dict]


__all__ = [
    'AddSubtaskResult',
    'AddTaskResult',
    'DependencyResult',
    'GetTasksResult',
    'RemoveTaskResult',
    'SetTaskStatusResult',
    'UpdateTaskResult',
    'ValidateDependenciesResult',
]
