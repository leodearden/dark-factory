"""Tests for ``TaskWorkflow._handle_false_premise_report`` and category threading.

Covers:
1. Category threading through ``_mark_blocked`` / ``_ensure_l1_escalation_for_blocked``
   (steps 3-4)
2. ``_handle_false_premise_report`` handler — escalates design_concern to human,
   stops steward, clears artifact, handles malformed artifact (steps 5-6)
3. Wiring: ``_plan`` and ``_run_simple_task`` route false_premise.json to the handler
   PRE-IMPLEMENTATION (steps 7-8)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


# ---------------------------------------------------------------------------
# Shared harness
# ---------------------------------------------------------------------------


@dataclass
class _Fixture:
    wf: TaskWorkflow
    artifacts: TaskArtifacts
    mark_blocked: AsyncMock


def _make(
    *,
    worktree: Path,
    project_root: Path,
    task_id: str = '50',
    steward: object | None = None,
    real_mark_blocked: bool = False,
) -> _Fixture:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = project_root

    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()

    git_ops = MagicMock()

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )

    worktree.mkdir(parents=True, exist_ok=True)
    artifacts = TaskArtifacts(worktree)
    artifacts.init(task_id, 'T', 'd', base_commit='oldbase')
    wf.artifacts = artifacts
    wf.worktree = worktree
    wf._steward = steward

    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    if not real_mark_blocked:
        wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    return _Fixture(wf=wf, artifacts=artifacts, mark_blocked=mark_blocked)


def _make_with_escalation_queue(
    *,
    worktree: Path,
    project_root: Path,
    task_id: str = '50',
) -> tuple[_Fixture, list]:
    """Return a fixture with a real escalation_queue stub that records submits."""
    submitted: list = []

    fake_queue = MagicMock()
    fake_queue.has_open_l1.return_value = False
    fake_queue.make_id.return_value = f'esc-{task_id}-1'
    fake_queue.submit.side_effect = submitted.append

    f = _make(
        worktree=worktree,
        project_root=project_root,
        task_id=task_id,
        real_mark_blocked=True,
    )
    f.wf.escalation_queue = fake_queue
    f.wf.event_store = None
    return f, submitted


# ---------------------------------------------------------------------------
# Step-3 RED: category threading through _mark_blocked → _ensure_l1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalate_to_human_uses_category(tmp_path: Path):
    """_mark_blocked threads category='design_concern' into the L1 escalation."""
    f, submitted = _make_with_escalation_queue(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj'
    )

    await f.wf._mark_blocked(
        'r', detail='d', escalate_to_human=True, category='design_concern'
    )

    l1_escs = [e for e in submitted if getattr(e, 'level', None) == 1]
    assert l1_escs, 'Expected at least one L1 escalation to be submitted'
    assert all(e.category == 'design_concern' for e in l1_escs)


@pytest.mark.asyncio
async def test_default_category_is_task_failure(tmp_path: Path):
    """Default category remains 'task_failure' — regression guard for existing callers."""
    f, submitted = _make_with_escalation_queue(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj'
    )

    await f.wf._mark_blocked('r', detail='d', escalate_to_human=True)

    l1_escs = [e for e in submitted if getattr(e, 'level', None) == 1]
    assert l1_escs, 'Expected at least one L1 escalation to be submitted'
    assert all(e.category == 'task_failure' for e in l1_escs)


# ---------------------------------------------------------------------------
# Step-5 RED: _handle_false_premise_report handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalates_design_concern_to_human(tmp_path: Path):
    """Handler routes to _mark_blocked with escalate_to_human=True, category='design_concern'."""
    f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
    f.artifacts.write_false_premise(
        classification='dependency_capability',
        premise='task produces a Mesh',
        evidence='Mesh is delivered by a task that DEPENDS ON this one',
        proposed_resolution='move signal to the producing downstream leaf',
    )

    outcome = await f.wf._handle_false_premise_report()

    assert outcome == WorkflowOutcome.BLOCKED
    f.mark_blocked.assert_awaited_once()
    args, kwargs = f.mark_blocked.await_args
    assert 'task produces a Mesh' in args[0]
    assert kwargs['escalate_to_human'] is True
    assert kwargs.get('category') == 'design_concern'
    assert 'dependency_capability' in kwargs['detail']
    assert 'move signal to the producing downstream leaf' in kwargs['detail']
    # Artifact must be cleared so a re-run doesn't see a stale report.
    assert f.artifacts.read_false_premise() is None


@pytest.mark.asyncio
async def test_stops_steward_early_when_running(tmp_path: Path):
    """Defense-in-depth: handler stops a running steward before submitting L1."""
    steward = MagicMock()
    steward.stop = AsyncMock()
    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
        steward=steward,
    )
    f.artifacts.write_false_premise(
        classification='exactness',
        premise='p',
        evidence='e',
        proposed_resolution='r',
    )

    outcome = await f.wf._handle_false_premise_report()

    assert outcome == WorkflowOutcome.BLOCKED
    steward.stop.assert_awaited_once()
    assert f.wf._steward is None
    f.mark_blocked.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_steward_present_does_not_raise(tmp_path: Path):
    """At PLAN time the steward hasn't been started — handler must no-op."""
    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
        steward=None,
    )
    f.artifacts.write_false_premise(
        classification='numeric_bound', premise='p', evidence='e', proposed_resolution='r'
    )

    outcome = await f.wf._handle_false_premise_report()
    assert outcome == WorkflowOutcome.BLOCKED


@pytest.mark.asyncio
async def test_missing_premise_still_escalates_design_concern(tmp_path: Path):
    """Malformed artifact (missing premise): still design_concern L1."""
    f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
    (f.artifacts.root / 'false_premise.json').write_text(
        '{"classification": "exactness", "evidence": "e"}\n'
    )

    outcome = await f.wf._handle_false_premise_report()

    assert outcome == WorkflowOutcome.BLOCKED
    f.mark_blocked.assert_awaited_once()
    args, kwargs = f.mark_blocked.await_args
    assert 'malformed' in args[0].lower()
    assert kwargs['escalate_to_human'] is True
    assert kwargs.get('category') == 'design_concern'
    assert f.artifacts.read_false_premise() is None


# ---------------------------------------------------------------------------
# Step-7 RED: _plan / _run_simple_task routing wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_routes_false_premise_to_handler(tmp_path: Path):
    """_plan() detects false_premise.json and routes to _handle_false_premise_report."""
    from orchestrator.agent_runner import AgentResult

    wt = tmp_path / 'wt'
    f = _make(worktree=wt, project_root=tmp_path / 'proj')

    handle_mock = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    f.wf._handle_false_premise_report = handle_mock  # type: ignore[method-assign]

    # Stub the briefing and invoke so _plan() calls the architect but no plan is written.
    f.wf.briefing.build_architect_prompt = AsyncMock(return_value='P')

    def _invoke_writes_false_premise(*args, **kwargs):
        f.artifacts.write_false_premise(
            classification='exactness',
            premise='p',
            evidence='e',
            proposed_resolution='r',
        )
        return AgentResult(success=True, cost_usd=0.5, turns=3)

    f.wf._invoke = AsyncMock(side_effect=_invoke_writes_false_premise)

    # Ensure no plan.json, no plan.lock, no _old_plan_base so revalidation is skipped.
    (wt / 'plan.json').unlink(missing_ok=True)
    (wt / 'plan.lock').unlink(missing_ok=True)
    if hasattr(f.wf, '_old_plan_base'):
        del f.wf._old_plan_base

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.BLOCKED
    handle_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_simple_task_routes_false_premise_to_handler(tmp_path: Path):
    """_run_simple_task() detects false_premise.json and routes to handler."""
    from orchestrator.agent_runner import AgentResult

    wt = tmp_path / 'wt'
    f = _make(worktree=wt, project_root=tmp_path / 'proj')

    handle_mock = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    f.wf._handle_false_premise_report = handle_mock  # type: ignore[method-assign]

    f.wf.briefing.build_simple_task_prompt = AsyncMock(return_value='P')

    def _invoke_writes_false_premise(*args, **kwargs):
        f.artifacts.write_false_premise(
            classification='numeric_bound',
            premise='p',
            evidence='e',
            proposed_resolution='r',
        )
        return AgentResult(success=True, cost_usd=0.5, turns=3)

    f.wf._invoke = AsyncMock(side_effect=_invoke_writes_false_premise)

    outcome = await f.wf._run_simple_task()

    assert outcome == WorkflowOutcome.BLOCKED
    handle_mock.assert_awaited_once()
