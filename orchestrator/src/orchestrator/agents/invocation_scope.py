"""The isolation one role's agent invocation carries under one config.

Shared by ``TaskWorkflow._invoke`` and the shadow isolation probe, so both read one decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.agents.roles import AgentRole
    from orchestrator.config import OrchestratorConfig

__all__ = ['InvocationScope', 'invocation_scope']


@dataclass(frozen=True)
class InvocationScope:
    strict_mcp_config: bool
    sandboxed: bool

    @property
    def sandbox_modules(self) -> list[str] | None:
        """``[]`` is the whole-worktree sandbox-on gate (PRD os-sandbox D1); None runs unsandboxed."""
        return [] if self.sandboxed else None


def invocation_scope(config: OrchestratorConfig, role: AgentRole) -> InvocationScope:
    return InvocationScope(
        strict_mcp_config=config.strict_mcp_config or role.strict_mcp_scope,
        sandboxed=config.sandbox.enabled and role.sandboxed,
    )
