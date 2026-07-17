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


# ─────────────────────────────────────────────────────────────────────────────
# Row 3+4 fixture constants (real `git grep` gate)
# ─────────────────────────────────────────────────────────────────────────────

_CAP_NAME_34 = 'row34_cap'
_CAP_TOKEN_34 = 'ROW34_CAPABILITY_TOKEN_V1'
_MARKER_REL_PATH_34 = 'src/row34_marker.py'


# ─────────────────────────────────────────────────────────────────────────────
# TestRealGitGrepGate — rows 3+4: real `git grep` withhold, then dispatch
# ─────────────────────────────────────────────────────────────────────────────


class TestRealGitGrepGate:
    """Rows 3+4, driven through the REAL (unmocked) ``run_delivered_check``
    + ``Scheduler._resolve_main_sha`` — new coverage vs.
    ``test_delivered_check_gate.py``, whose unit tests fake both.

    A 'done' producer carries a grep-kind ``metadata.delivered_checks``
    entry (the same shape gamma's ``commit_planning`` stamps) whose token
    is initially absent from branch main; a pending dependent depends on
    it via a LOCAL dependency. Row 4: while the token is absent, a real
    ``git grep`` finds no match (rc==1 -> FAILED) so ``acquire_next()``
    withholds dispatch and the hold is visible (event + streak). Row 3:
    once the token is committed to main, the very next tick dispatches
    the dependent (real ``git grep`` rc==0 -> DELIVERED).
    """

    @pytest.mark.asyncio
    async def test_token_absent_withholds_then_landed_token_dispatches(
        self, tmp_path: Path
    ) -> None:
        project_root = _init_git_repo(tmp_path, marker_rel_path=_MARKER_REL_PATH_34)
        session = _LocalDepMcpSession()
        _register_producer(
            session, 'P34', status='done',
            checks=[_grep_check(_CAP_NAME_34, _CAP_TOKEN_34, [_MARKER_REL_PATH_34])],
        )
        _register_dependent(session, 'D34', dep_id='P34')

        harness = _build_harness(project_root, session, tmp_path / 'escalations')

        # --- row 4: token absent from main -> withhold, held event/streak ---
        result = await _run_tick(harness)
        assert result is None, (
            f'row 4: token absent from main must withhold dispatch of the '
            f'dependent; got {result!r}'
        )
        assert _held_events(harness), (
            'row 4: a delivered_check_gate_held event must be recorded for '
            'the withheld dependent'
        )
        assert harness.scheduler._streak_delivered_hold.value('D34') >= 1, (
            'row 4: _streak_delivered_hold must bump for the withheld dependent'
        )

        # --- row 3: land the token on main -> dispatch the very next tick ---
        _commit_marker(project_root, _MARKER_REL_PATH_34, _CAP_TOKEN_34)

        result = await _run_tick(harness)
        assert result == 'D34', (
            f'row 3: once the real `git grep` finds the token on main, the '
            f'dependent must dispatch; got {result!r}'
        )
