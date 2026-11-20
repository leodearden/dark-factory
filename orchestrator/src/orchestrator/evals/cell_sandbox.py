"""Binds an eval workflow cell's sandbox backend from its own config.

An eval process has no ``harness.py::Harness`` to bind the process's sandbox
backend at startup, so a workflow cell binds it itself. S8 κ1 (task 9232)
moves the backend onto the invocation and retires this module.
"""

from __future__ import annotations

from orchestrator.agents import sandbox_dispatch
from orchestrator.agents.sandbox_dispatch import Backend
from orchestrator.config import OrchestratorConfig


def bind_cell_sandbox_backend(config: OrchestratorConfig) -> Backend:
    """Bind *config*'s backend as ``Harness`` does; return the backend its sandboxed agents run under."""
    sandbox_dispatch.set_backend(config.sandbox.backend)
    if not config.sandbox.enabled:
        return 'none'
    return sandbox_dispatch.resolve_active_backend()
