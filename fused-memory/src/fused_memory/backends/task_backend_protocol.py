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
    """The 12-method + lifecycle surface every task backend must implement."""

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

    async def get_statuses_raw(
        self,
        project_root: str,
        tag: str | None = None,
        ids: list[str] | None = None,
    ) -> dict[str, str]:
        """Return ``{id_str: status_str}`` without decoding metadata columns."""
        ...

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
        status: str = 'pending',
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
    ) -> UpdateTaskResult:
        """Update task metadata fields (title, description, details, priority, metadata, dependencies).

        **Implementations MUST reject a non-None ``status`` by raising** (e.g.
        ``TaskmasterError('TASKMASTER_TOOL_ERROR', …)``).  ``set_task_status``
        is the only sanctioned status writer — it enforces the terminal-exit,
        phantom-done, and done-provenance gates.  Accepting ``status`` here
        would silently bypass all three.

        The ``status`` param is kept in the signature as a **reject-trap**: it
        preserves the ``status=None`` passthrough that ``server/tools.py`` and
        ``task_interceptor.py`` forward via ``**kwargs``, and it makes the
        reject-only contract explicit for future backend authors.
        """
        ...

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
