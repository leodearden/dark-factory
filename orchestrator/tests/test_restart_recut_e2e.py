"""Acceptance for task 5516: a recut restart re-plans with the architect, on real git.

A task blocked by the plan-touch gate keeps a prior incarnation that Lever B
(``workflow.py::TaskWorkflow._apply_revalidation_skip``) will happily reuse: a
recent finalized plan whose files main has not touched. A plain restart
re-pends onto that incarnation and the architect never runs. A restart
resolved with ``recut=True`` issues a re-cut ruling, the task-3154 dispatch
gate parks the prior incarnation (nothing deleted), and the next run plans
fresh with the architect.

Driven through the real ``resolve_issue`` tool, a real ``EscalationQueue``,
the harness resolve callback, ``recut_ruling.enforce_recut_ruling`` (exactly
as ``harness.py::Harness._run_recut_ruling_gate`` calls it) and
``TaskWorkflow.run``; asserted on emitted events and agent invocations.

The next dispatch is NOT driven through ``Harness._run_slot``, so this test
does not prove the slot's wiring: entering the gate only for a task that
carries a ruling, dropping the recovered plan, session and config dir when
the gate executes, and blocking when it refuses. That wiring is covered by
``test_recut_ruling_dispatch.py::TestRunSlotRecutGate``.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _task_store_doubles import TaskStoreDouble
from _workflow_helpers import (
    PLAN,
    AgentStub,
    _build_harness,
    _build_workflow_with_escalation,
    _derive_meta_root_like_production,
    _init_repo,
)
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.server import create_server
from fastmcp.tools.function_tool import FunctionTool
from shared.task_text_fingerprint import task_text_fingerprint

from orchestrator.artifacts import PLAN_SCHEMA_VERSION, TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig, SandboxConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps, _run
from orchestrator.harness import Harness
from orchestrator.merge_lane.gates import PLAN_FILES_NOT_TOUCHED_REASON_PREFIX
from orchestrator.recut_ruling import (
    RECUT_EXECUTED_KEY,
    RECUT_RULING_KEY,
    GateVerdict,
    RecutGateResult,
    enforce_recut_ruling,
)
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult

# Autouse; named here so the agent-side doubles resolve the meta_root production does.
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.usefixtures(_derive_meta_root_like_production.__name__),
]

TASK_ID = '42'
TITLE = 'Add farewell function'
DESCRIPTION = 'Add a farewell(name) function to lib.py with tests'
LEVER_B_REASON = 'revalidation_skipped_no_overlap'


class _EventRecorder:
    run_id = 'test-run'

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def emit(self, event_type, **kwargs) -> None:
        self.calls.append((event_type, kwargs))

    def lever_b_skips(self) -> list[dict[str, Any]]:
        return [
            kwargs for event_type, kwargs in self.calls
            if event_type == EventType.phase_skipped
            and (kwargs.get('data') or {}).get('reason') == LEVER_B_REASON
        ]


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _init_repo(repo)
    return repo


@pytest.fixture
def config(git_repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=git_repo,
        max_concurrent_tasks=1,
        max_execute_iterations=5,
        max_verify_attempts=3,
        max_review_cycles=2,
        judge_after_each_iteration=False,
        max_amendment_rounds=1,
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees',
        ),
        sandbox=SandboxConfig(enabled=False),
    )


@pytest.fixture
def git_ops(config: OrchestratorConfig) -> GitOps:
    return GitOps(config.git, config.project_root)


@pytest.fixture
def store() -> TaskStoreDouble:
    store = TaskStoreDouble()
    store.add(
        TASK_ID, metadata={'files': ['lib']},
        title=TITLE, description=DESCRIPTION, dependencies=[],
    )
    return store


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'harness-esc')


@pytest.fixture
def harness(
    config: OrchestratorConfig, store: TaskStoreDouble, queue: EscalationQueue,
    monkeypatch: pytest.MonkeyPatch,
) -> Harness:
    h = _build_harness(config)
    store.wire_into(h.scheduler)
    monkeypatch.setattr(h, 'is_workflow_active', lambda task_id: False)
    h._escalation_queue = queue
    queue.set_resolve_callback(h._on_escalation_resolved)
    return h


@pytest.fixture
def agents(monkeypatch: pytest.MonkeyPatch) -> AgentStub:
    stub = AgentStub()
    monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
    monkeypatch.setattr(
        'orchestrator.workflow.run_scoped_verification',
        AsyncMock(return_value=VerifyResult(
            passed=True, test_output='', lint_output='', type_output='',
            summary='All checks passed',
        )),
    )
    return stub


async def _seed_prior_incarnation(git_ops: GitOps) -> str:
    """A Lever-B-eligible, plan-files-not-touched prior incarnation; its tip sha."""
    main_sha = await git_ops.get_main_sha()
    worktree = (await git_ops.create_worktree(TASK_ID)).path
    (worktree / 'NOTES.md').write_text('prior incarnation: touched no plan file\n')
    await _run(['git', 'add', 'NOTES.md'], cwd=worktree)
    await _run(['git', 'commit', '-m', 'prior incarnation marker'], cwd=worktree)
    _, tip, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
    tip = tip.strip()

    artifacts = TaskArtifacts.for_task(worktree, TASK_ID)
    artifacts.init(TASK_ID, TITLE, DESCRIPTION, base_commit=main_sha)
    plan = copy.deepcopy(PLAN)
    for step in plan['steps']:
        step.update(status='done', commit=tip)
    plan.update(
        _session_id='prior-session',
        _created_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        _schema_version=PLAN_SCHEMA_VERSION,
        _task_text_fp=task_text_fingerprint(DESCRIPTION, None),
    )
    assert all((worktree / f).exists() for f in plan['files'])
    artifacts.artifact_path('plan.json').write_text(json.dumps(plan))
    return tip


async def _resolve_plan_touch_block(
    harness: Harness, queue: EscalationQueue, **kwargs: Any,
) -> Escalation:
    esc = Escalation(
        id=queue.make_id(TASK_ID),
        task_id=TASK_ID,
        agent_role='orchestrator',
        severity='blocking',
        category='design_concern',
        summary=f'{PLAN_FILES_NOT_TOUCHED_REASON_PREFIX}: lib.py, test_lib.py.',
        level=1,
    )
    queue.submit(esc)
    tool = await create_server(queue, startup_sweep=False).get_tool('resolve_issue')
    assert isinstance(tool, FunctionTool)
    result = tool.fn(
        escalation_id=esc.id, resolution='re-plan on current main', **kwargs,
    )
    assert 'error' not in result, result
    while harness._background_tasks:
        await asyncio.gather(*list(harness._background_tasks))
    return esc


async def _dispatch(
    config: OrchestratorConfig, git_ops: GitOps, store: TaskStoreDouble,
    agents: AgentStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[RecutGateResult, _EventRecorder]:
    """The next dispatch: the re-cut gate on the stored task, then the workflow."""
    gate = await enforce_recut_ruling(
        await store.get_task(TASK_ID) or {},
        ops=git_ops,
        persist=functools.partial(store.update_task, TASK_ID),
        now=datetime.now(UTC),
    )
    assignment = TaskAssignment(
        task_id=TASK_ID, task=await store.get_task(TASK_ID) or {}, modules=['lib'],
    )
    workflow, _, _ = _build_workflow_with_escalation(
        config, git_ops, assignment, agents, tmp_path,
    )
    events = _EventRecorder()
    monkeypatch.setattr(workflow, 'event_store', events)
    await workflow.run()
    return gate, events


async def test_plain_restart_reuses_the_prior_plan_without_the_architect(
    config, git_ops, store, queue, harness, agents, tmp_path, monkeypatch,
):
    await _seed_prior_incarnation(git_ops)
    await _resolve_plan_touch_block(harness, queue, action='restart')
    assert store.status(TASK_ID) == 'pending'
    assert RECUT_RULING_KEY not in store.metadata(TASK_ID)

    gate, events = await _dispatch(config, git_ops, store, agents, tmp_path, monkeypatch)

    assert gate.verdict is GateVerdict.PROCEED
    assert events.lever_b_skips(), 'the seed must be Lever-B eligible'
    first_review = agents.calls.index('reviewer_comprehensive')
    assert 'architect' not in agents.calls[:first_review], (
        'the stale plan reached review without the architect re-planning it'
    )


async def test_recut_restart_parks_the_prior_incarnation_and_runs_the_architect(
    config, git_ops, store, queue, harness, agents, tmp_path, monkeypatch,
):
    prior_tip = await _seed_prior_incarnation(git_ops)
    esc = await _resolve_plan_touch_block(harness, queue, action='restart', recut=True)
    assert store.status(TASK_ID) == 'pending'
    assert store.metadata(TASK_ID)[RECUT_RULING_KEY]['ruling_id'] == esc.id

    gate, events = await _dispatch(config, git_ops, store, agents, tmp_path, monkeypatch)

    assert gate.verdict is GateVerdict.EXECUTED, gate.reason
    assert agents.calls[0] == 'architect'
    assert events.lever_b_skips() == []
    executed = store.metadata(TASK_ID)[RECUT_EXECUTED_KEY]
    assert executed['ruling_id'] == esc.id
    parked_branch = executed['parked_branch']
    assert parked_branch.startswith(f'task/{TASK_ID}-')
    _, parked_sha, _ = await _run(
        ['git', 'rev-parse', parked_branch], cwd=config.project_root,
    )
    assert parked_sha.strip() == prior_tip
    parked_plans = list(git_ops.quarantine_base.rglob('plan.json'))
    assert len(parked_plans) == 2, (
        f'expected the parked task root and the parking snapshot, got {parked_plans}'
    )
    assert all(not plan.is_symlink() for plan in parked_plans)
    assert {json.loads(plan.read_text()).get('_session_id') for plan in parked_plans} == {
        'prior-session',
    }
