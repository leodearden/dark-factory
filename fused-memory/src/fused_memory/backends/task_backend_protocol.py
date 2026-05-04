"""Protocol that codifies the task-backend surface used by ``TaskInterceptor``.

:class:`fused_memory.backends.sqlite_task_backend.SqliteTaskBackend`
implements this protocol; new task backends slot in by satisfying the same
surface so the interceptor can be typed against a single shape.

``parse_prd`` and ``expand_task`` are intentionally absent —
orchestrator-side PRD decomposition goes through ``planning_mode`` + the
curator instead.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from fused_memory.backends.task_backend_types import (
    AddSubtaskResult,
    AddTaskResult,
    DependencyResult,
    GetTasksResult,
    RemoveTaskResult,
    SetTaskStatusResult,
    UpdateTaskResult,
    ValidateDependenciesResult,
)


@runtime_checkable
class TaskBackendProtocol(Protocol):
    """The 11-method + lifecycle surface every task backend must implement."""

    # ── Lifecycle ──────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        """Whether the backend is currently usable."""
        ...

    @property
    def restart_count(self) -> int:
        """Number of successful (re)connects. SQLite backends may pin this at 1."""
        ...

    async def start(self) -> None:
        """Bring the backend up. Idempotent — repeated calls are no-ops."""
        ...

    async def close(self) -> None:
        """Tear the backend down. Idempotent."""
        ...

    async def ensure_connected(self) -> None:
        """Wait briefly until the backend is usable, or raise."""
        ...

    async def is_alive(self) -> tuple[bool, str | None]:
        """``(alive, error_message)`` — read-only health probe."""
        ...

    # ── Reads ──────────────────────────────────────────────────────────

    async def get_tasks(
        self, project_root: str, tag: str | None = None
    ) -> GetTasksResult: ...

    async def get_task(
        self, task_id: str, project_root: str, tag: str | None = None
    ) -> dict: ...

    # ── Mutations ──────────────────────────────────────────────────────

    async def set_task_status(
        self,
        task_id: str,
        status: str,
        project_root: str,
        tag: str | None = None,
    ) -> SetTaskStatusResult: ...

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
    ) -> AddTaskResult: ...

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
    ) -> UpdateTaskResult: ...

    async def add_subtask(
        self,
        parent_id: str,
        project_root: str,
        title: str | None = None,
        description: str | None = None,
        details: str | None = None,
        tag: str | None = None,
    ) -> AddSubtaskResult: ...

    async def remove_tasks(
        self,
        ids: list[str],
        project_root: str,
        tag: str | None = None,
    ) -> RemoveTaskResult: ...

    async def add_dependency(
        self,
        task_id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ) -> DependencyResult: ...

    async def remove_dependency(
        self,
        task_id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ) -> DependencyResult: ...

    async def validate_dependencies(
        self, project_root: str, tag: str | None = None
    ) -> ValidateDependenciesResult: ...


__all__ = ['TaskBackendProtocol']
