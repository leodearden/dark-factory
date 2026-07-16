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
import re
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


# ─────────────────────────────────────────────────────────────────────────────
# Headline fixture constants
# ─────────────────────────────────────────────────────────────────────────────

_PRD_PATH = 'plans/e2e-fixture-prd.md'
_PRODUCER_LABEL = 'producer'
_CAPABILITY_NAME = 'zeta_cap'
_CAPABILITY_TOKEN = 'ZETA_CAPABILITY_TOKEN_V1'
_MARKER_REL_PATH = 'src/marker.py'
_DEPENDENT_REL_PATH = 'src/dependent_target.py'


# ─────────────────────────────────────────────────────────────────────────────
# TestHeadline — rows 1, 4, 5, 6, 3: stamp -> withhold -> escalate -> heal -> dispatch
# ─────────────────────────────────────────────────────────────────────────────


class TestHeadline:
    """The closed-loop headline scenario, built up incrementally across
    steps 1/3/5 (each step appends more of the scenario to this same test
    method — see plan.json step descriptions)."""

    @pytest.mark.asyncio
    async def test_headline_stamp_withhold_escalate_heal_dispatch(self, backend_stack):
        """Headline part A (step-1): row 1 — commit_planning stamps the
        producer's real task_id into the sidecar and copies the sidecar's
        mechanical delivered_check into the producer's
        metadata.delivered_checks, visible via get_task."""
        server, _interceptor, project_root = backend_stack

        _write_sidecar(
            project_root,
            prd_path=_PRD_PATH,
            label=_PRODUCER_LABEL,
            capability_name=_CAPABILITY_NAME,
            pattern=_CAPABILITY_TOKEN,
            paths=[_MARKER_REL_PATH],
        )

        producer_id, dependent_id = await _file_planning_batch(
            server, project_root, prd_path=_PRD_PATH,
        )

        # --- row 1: commit_planning stamps the sidecar + copies delivered_checks ---
        result = await _commit_planning(server, project_root, [producer_id, dependent_id])

        expected_sidecar_rel = re.sub(r'\.md$', '', _PRD_PATH) + '.capability-manifest.yaml'
        assert result['manifest_stamping'] == {
            'path': expected_sidecar_rel,
            'stamped': [_PRODUCER_LABEL],
            'missing_labels': [],
            'errors': [],
        }

        sidecar_path = project_root / expected_sidecar_rel
        reloaded = yaml.safe_load(sidecar_path.read_text(encoding='utf-8'))
        assert reloaded['tasks'][0]['task_id'] == int(producer_id)

        producer_task = await _get_task(server, project_root, producer_id)
        checks = producer_task['metadata']['delivered_checks']
        assert len(checks) == 1
        assert checks[0]['name'] == _CAPABILITY_NAME
        assert checks[0]['kind'] == 'grep'
        assert checks[0]['pattern'] == _CAPABILITY_TOKEN
        assert checks[0]['expect'] == 'present'
        assert checks[0]['paths'] == [_MARKER_REL_PATH]

        # Status flip landed too (target_status defaults to 'pending').
        assert producer_task['status'] == 'pending'
        dependent_task = await _get_task(server, project_root, dependent_id)
        assert dependent_task['status'] == 'pending'
