"""Structural protocol types for the orchestrator's injectable dependencies.

Using Protocol here (instead of concrete base classes) lets tests inject
lightweight fakes without subclassing the real implementations, which
require expensive initialization (OrchestratorConfig, HTTP connections, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class SchedulerProtocol(Protocol):
    """Minimal scheduler interface used by TaskWorkflow."""

    async def set_task_status(self, task_id: str, status: str) -> None: ...

    async def handle_blast_radius_expansion(
        self,
        task_id: str,
        current: list[str],
        needed: list[str],
    ) -> bool: ...


@runtime_checkable
class BriefingProtocol(Protocol):
    """Minimal briefing-assembler interface used by TaskWorkflow."""

    async def build_architect_prompt(
        self,
        task: dict,
        worktree: Path | None = None,
        context: str | None = None,
    ) -> str: ...

    async def build_implementer_prompt(
        self,
        plan: dict,
        iteration_log: list,
        context: str | None = None,
    ) -> str: ...

    async def build_debugger_prompt(
        self,
        failures: str,
        plan: dict,
        context: str | None = None,
    ) -> str: ...

    async def build_reviewer_prompt(
        self,
        reviewer_type: str,
        diff: str,
        context: str | None = None,
    ) -> str: ...

    async def build_merger_prompt(
        self,
        conflicts: str,
        task_intent: str,
        context: str | None = None,
    ) -> str: ...

    async def build_resume_prompt(
        self,
        task: dict,
        plan: dict,
        escalation_summary: str,
        resolution: str,
        worktree: Path | None = None,
    ) -> str: ...


@runtime_checkable
class McpProtocol(Protocol):
    """Minimal MCP-lifecycle interface used by TaskWorkflow."""

    @property
    def url(self) -> str: ...

    def mcp_config_json(self, escalation_url: str | None = None) -> dict: ...
