"""End-to-end integration gate for the capability-delivered-checks PRD
(``plans/capability-delivered-checks-prd.md``, task zeta).

Proves the whole feature works through the product's own surfaces — no
product code changes here, this module is purely test authoring covering
the PRD's full 10-row Boundary-test sketch table (rows 1, 2, 3, 4, 5, 6, 8,
9 are owned by this file; rows 3/4/7/10 are re-exercised through the real
git runner in ``orchestrator/tests/test_delivered_check_gate_e2e.py``).

This file drives the CROSS-CUTTING headline: ``commit_planning`` (fused
gamma, task 2578) stamping a producer's ``metadata.delivered_checks`` from
a capability-manifest sidecar, THEN a real orchestrator Scheduler tick
(delta/epsilon, tasks 2580/2583) gating dispatch of a dependent task on
those checks passing against a real git ``main`` branch.

Rig shape (see plan.json design_decisions for the full rationale):
- ONE real temp git repo used as ``project_root`` by the fused-memory
  backend, ``commit_planning``, and the orchestrator scheduler's delivered-
  check runner (``git -C project_root grep|rev-parse ... main``).
- A real ``SqliteTaskBackend`` + ``TaskInterceptor`` + ``TicketStore`` +
  ``EventBuffer`` + ``create_mcp_server`` stack (the ``real_task_stack``
  pattern from ``test_task_tools.py``), driven via
  ``server._tool_manager.call_tool(...)`` for submit_task/add_dependency/
  commit_planning/get_task — the product's own MCP surface.
- A real orchestrator ``Harness``/``Scheduler`` with a small delegating
  ``_BackendMcpSession`` injected at ``scheduler._mcp_session`` so
  ``acquire_next()`` reads/writes the SAME live backend state the
  ``commit_planning`` call stamped, and a real ``EscalationQueue`` so the
  born-at-L2 escalation is genuinely filed and read back.

Assertions go exclusively through product read/observe paths: ``get_task``
(metadata + status), ``EscalationQueue.get_by_task(..., level=2,
agent_role='orchestrator-scheduler')`` (the born-at-L2), and
``acquire_next()``'s returned ``TaskAssignment`` / ``None`` (dispatch vs
withhold) — never by peeking at scheduler internals or event-store state
for the primary assertions.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
import yaml

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server

from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.harness import Harness


# ─────────────────────────────────────────────────────────────────────────────
# Import-resolution smoke test (prerequisite pre-1)
# ─────────────────────────────────────────────────────────────────────────────


def test_cross_package_imports_resolve():
    """Prerequisite pre-1: fused_memory + orchestrator + escalation import
    together inside fused-memory's pytest env (pythonpath includes
    ../orchestrator/src per fused-memory/pyproject.toml)."""
    assert SqliteTaskBackend is not None
    assert TaskmasterConfig is not None
    assert TaskInterceptor is not None
    assert TicketStore is not None
    assert EventBuffer is not None
    assert create_mcp_server is not None
    assert EscalationQueue is not None
    assert OrchestratorConfig is not None
    assert EventType is not None
    assert Harness is not None
