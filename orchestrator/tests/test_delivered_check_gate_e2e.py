"""End-to-end integration gate for the delivered-check dep-gate
(capability-delivered-checks PRD, ``plans/capability-delivered-checks-prd.md``,
task zeta) driven through the REAL (unmocked) git runner.

Complements ``fused-memory/tests/test_delivered_checks_e2e.py`` (the
cross-cutting stamp -> gate headline, which needs the fused-memory backend).
This file cannot import ``fused_memory`` (orchestrator/pyproject.toml does
not add ../fused-memory/src to pythonpath) — it drives delta/epsilon
(``orchestrator.scheduler``/``orchestrator.harness``) directly against a
real temp git repo with hand-authored producer metadata (the same
``metadata.delivered_checks`` shape gamma's ``commit_planning`` stamps),
using the UNMOCKED ``orchestrator.delivered_checks.run_delivered_check`` +
``Scheduler._resolve_main_sha`` (real ``git grep``/``git rev-parse``
subprocess calls) — new coverage vs. ``test_delivered_check_gate.py``,
whose unit tests fake both.

Boundary rows owned by this file (see the PRD's Boundary-test sketch):
- rows 3 + 4 (real git grep: token absent -> withhold + held event; token
  committed to main -> dispatch)
- row 7 (script-kind runner ERROR via a real missing-script OSError ->
  fail-safe wait, then real recovery once the script exists)
- row 10 (a CANCELLED producer carrying a failing check is gated exactly
  like a done one)

Rig pattern adapted from
``orchestrator/tests/test_cross_project_dispatch_integration.py``: a
backend-free MCP session double (``_LocalDepMcpSession``) serving
``get_tasks``/``set_task_status`` over an in-memory local-dep task list,
a real ``Harness`` with a real ``EscalationQueue``, and dispatch observed
exclusively through ``acquire_next()``'s returned ``TaskAssignment`` / None
-- never by reading scheduler internals for the primary assertions.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.harness import Harness


# ─────────────────────────────────────────────────────────────────────────────
# Import-resolution smoke test (prerequisite pre-1)
# ─────────────────────────────────────────────────────────────────────────────


def test_imports_resolve_without_fused_memory():
    """Prerequisite pre-1: this file imports orchestrator + escalation only
    (no fused_memory — orchestrator/pyproject.toml has no path to it)."""
    assert EscalationQueue is not None
    assert OrchestratorConfig is not None
    assert EventType is not None
    assert Harness is not None
