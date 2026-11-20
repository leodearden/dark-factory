"""Binds an eval workflow cell's sandbox backend from its own config.

An eval process has no ``harness.py::Harness`` to bind the process's sandbox
backend at startup, so a workflow cell binds it itself. S8 κ1 (task 9232)
moves the backend onto the invocation and retires this module.
"""

from __future__ import annotations

from typing import Literal

from orchestrator.agents import sandbox_dispatch
from orchestrator.agents.sandbox_dispatch import Backend
from orchestrator.config import OrchestratorConfig

CellSandbox = Backend | Literal['refused']
"""What a workflow cell's sandboxed agents ran under.

A resolved backend when they ran confined; ``'none'`` when they ran unconfined,
because the sandbox is disabled or its backend is the ``none`` escape hatch;
``'refused'`` when the sandbox is enabled but its configured backend is
unavailable on the host, so ``workflow.py::TaskWorkflow._guard_sandbox``
refused them fail-closed and they never ran.
"""


def bind_cell_sandbox_backend(config: OrchestratorConfig) -> CellSandbox:
    """Bind *config*'s backend as ``Harness`` does; return what the cell's sandboxed agents run under."""
    sandbox_dispatch.set_backend(config.sandbox.backend)
    if not config.sandbox.enabled or config.sandbox.backend == 'none':
        return 'none'
    resolved = sandbox_dispatch.resolve_active_backend()
    return 'refused' if resolved == 'none' else resolved
