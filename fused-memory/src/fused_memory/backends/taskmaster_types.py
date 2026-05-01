"""Typed DTOs for the Taskmaster MCP adapter.

Every wrapper method on :class:`TaskmasterBackend` maps the Taskmaster MCP
wire envelope (``{data: {...}, version, tag}``) into one of the flat
``TypedDict`` DTOs declared here. Callers read these fields directly —
no defensive ``.get('data')`` unwrapping.

Id-bearing fields surface as ``str`` regardless of the underlying wire
type (Taskmaster sometimes emits ints for task ids). Fields that
Taskmaster's ``processMCPResponseData`` strips (``details``,
``testStrategy``) are not available on the task dicts returned by
``get_task``/``get_tasks`` — callers must not rely on them.
"""

from __future__ import annotations

from typing import Any, TypedDict


class TaskmasterError(Exception):
    """Raised when a Taskmaster MCP tool call fails or returns an unexpected shape.

    ``code`` mirrors Taskmaster's error-code convention
    (e.g. ``MISSING_ARGUMENT``, ``INPUT_VALIDATION_ERROR``) when the
    wire error response carries one, or one of the adapter-level codes
    ``TASKMASTER_TOOL_ERROR`` / ``UNEXPECTED_RESPONSE_SHAPE``.

    ``raw`` preserves the underlying response for post-mortem diagnosis.
    """

    def __init__(self, code: str, message: str, raw: Any = None) -> None:
        super().__init__(f'{code}: {message}')
        self.code = code
        self.message = message
        self.raw = raw


# ── DTOs ────────────────────────────────────────────────────────────


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

    The TS tool emits ``{message, tasks: [<per-id result>]}`` where each
    per-id result carries ``taskId`` and ``newStatus`` fields. We surface
    the per-id list as-is for callers that care; most callers only use
    ``message`` as a sanity check.
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
    ``status``, etc. already flattened by Taskmaster — no further
    unwrapping needed.
    """

    tasks: list[dict]
