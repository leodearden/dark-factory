"""Warm-lane session-resume INTEGRATION GATE (ω, task 2775).

The G2 user-observable, two-way integration gate for the warm-lane
session-resume feature (plans/warm-lane-session-resume-prd.md §8 boundary
matrix B1–B11). Its dependencies are all landed on this worktree:

  α (2771) sidecar v2 + preserve-on-cancel,
  β (2772) recovery adopts sessions,
  γ (2774) guarded injection + config + events + prompt note.

Where the α/β/γ UNIT tests in test_crash_recovery.py each assert ONE side of a
seam, ω proves the COMPOSED, two-way chain: it drives the REAL
``_recover_crashed_tasks`` to populate ``_recovered_sessions`` /
``_recovered_session_config_dirs`` / ``_recovered_plans`` (β's output), then
feeds that SAME recovered state through the REAL ``_run_slot`` γ guard (and,
for B1/B2/B11, the REAL ``_invoke`` α path) — proving β's output is exactly
γ's input, and that the resumed invocation carries the same session id, the
architect is skipped, and the WIP commit survives.

Fakes live ONLY at the true CLI boundary (``build_workflow``,
``invoke_with_cap_retry``); ω never runs a full ``TaskWorkflow.run()`` (that
would invoke real Claude agents and merge machinery). "Architect skipped" is
proven at the ``_run_slot``→``build_workflow`` seam via the injected
``initial_plan`` (the ratified complete-plan-skips-architect invariant), not by
observing zero live architect invocations.

Fixtures/helpers are module-local (no shared conftest) per the established
convention; only the repo-wide ``mock_orch_config`` conftest fixture and the
``_workflow_helpers`` fakes are reused. The autouse
``_derive_meta_root_like_production`` fixture is deliberately NOT imported so
the invoke-driver's ``TaskArtifacts(cwd)`` uses the legacy ``.task`` root that
``_config_dir`` and the sidecar read/write target.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler
from shared.config_dir import TaskConfigDir

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import ARCHITECT, IMPLEMENTER, SIMPLE_TASK
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig, SessionResumeConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps, _run
from orchestrator.harness import Harness
from orchestrator.lane_lifecycle import LaneLifecycle
from orchestrator.lane_lifecycle import LaneState as DurableLaneState
from orchestrator.warm_lane_pool import WarmLanePool
from orchestrator.workflow import TaskWorkflow


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config):
    """A Harness with mocked internals for driving REAL recovery + dispatch.

    Adapted verbatim from test_crash_recovery.py's ``harness`` fixture: heavy
    constructors (McpLifecycle/Scheduler/BriefingAssembler) are patched out,
    ``git_ops`` keeps a REAL worktree_base under ``tmp/.worktrees`` (so
    ``_recover_crashed_tasks`` and ``_run_slot`` run for real), and the
    scheduler/event_store are mocks. ``_lane_lifecycle`` is rebound onto the
    reassigned worktree_base so the record-driven recovery path and the
    ``_seed_lane_record`` helper target the same ``.lane-state`` dir.
    """
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[])
    h.scheduler.set_task_status = AsyncMock()
    # Title-less dict → non-None (no defer) + identities_match fails open (adopt).
    h.scheduler.get_task = AsyncMock(return_value={})
    # None status → transient/None → fall through to restore (never terminal-release).
    h.scheduler.get_status = AsyncMock(return_value=None)
    h.scheduler._dispatched = set()
    # is_deterministic is a sync @staticmethod predicate at the top of _run_slot.
    h.scheduler.is_deterministic = MagicMock(return_value=False)

    h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()
    # Mount-presence guard: mark storage present so recovery does not defer.
    h.git_ops.mark_pool_storage_present()
    h.git_ops.cleanup_worktree = AsyncMock()
    h.git_ops.quarantine_worktree = AsyncMock(return_value=None)
    # Rebind lane lifecycle onto the reassigned worktree_base (W11 delta).
    h.git_ops._lane_lifecycle = LaneLifecycle(
        h.git_ops.worktree_base, quarantine_worktree=h.git_ops.quarantine_worktree,
    )
    # Default "still registered" so fabricated lanes take the restore path.
    h.git_ops._is_registered_worktree = AsyncMock(return_value=True)
    # Exercise the best-effort event emits without a real store.
    h.event_store = MagicMock()

    return h


def _make_plan(
    steps_done: int,
    steps_total: int,
    task_id: str = 'test',
    *,
    session_id: str | None = None,
) -> dict:
    """Build a plan dict with the given step-completion counts.

    Mirrors test_crash_recovery.py's ``_make_plan``: when ``session_id`` is
    given the plan is provenance-stamped (signals the recovery path that the
    architect already produced this plan). Recovery keys a plan into
    ``_recovered_plans`` when it has ≥1 ``done`` step.
    """
    steps = []
    for i in range(steps_total):
        steps.append({
            'id': f'step-{i + 1}',
            'description': f'Step {i + 1}',
            'status': 'done' if i < steps_done else 'pending',
            'commit': f'abc{i}' if i < steps_done else None,
        })
    plan: dict = {
        'task_id': task_id,
        'title': 'Test Task',
        'steps': steps,
    }
    if session_id is not None:
        plan['_session_id'] = session_id
    return plan


# ── B1: clean SIGTERM mid-implementer, warm lane ─────────────────────────────
@pytest.mark.asyncio
async def test_b1_warm_lane_adopts_then_injects_same_session(harness: Harness):
    """B1 — the flagship two-way seam: REAL recovery adopts the session/plan,
    then that SAME recovered state flows through REAL ``_run_slot`` as
    ``resume_session_id`` + ``initial_plan`` with one ``session_resume`` event.

    Chains β's output (``_recovered_sessions`` / ``_recovered_session_config_dirs``
    / ``_recovered_plans``) straight into γ's input, proving they are the same
    dict (``is`` identity) — not merely equal — across the boot→dispatch seam.
    """
    task_id, session_id = '42', 'uuid-b1-implementer'
    _setup_warm_lane_session(harness, task_id, session_id, role='implementer')
    harness.config.session_resume = SessionResumeConfig()

    await harness._recover_crashed_tasks()

    # ── ADOPT side (β) ──
    assert task_id in harness._recovered_sessions
    assert harness._recovered_sessions[task_id]['session_id'] == session_id
    assert task_id in harness._recovered_session_config_dirs
    assert task_id in harness._recovered_plans
    harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    adopted = harness._recovered_sessions[task_id]
    recovered_plan = harness._recovered_plans[task_id]

    # ── INJECT side (γ) ──
    cap = await _dispatch_capture(harness, task_id)
    assert cap.resume_session_id is adopted
    assert cap.resume_session_id['session_id'] == session_id
    assert cap.initial_plan is recovered_plan
    assert [et for et, _ in cap.emits] == [EventType.session_resume]
