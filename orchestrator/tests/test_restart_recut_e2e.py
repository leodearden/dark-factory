"""Acceptance for task 5516: a recut restart re-plans with the architect, on real git.

A task blocked by the plan-touch gate keeps a prior incarnation that Lever B
(``workflow.py::TaskWorkflow._apply_revalidation_skip``) will happily reuse: a
recent finalized plan whose files main has not touched. A plain restart
re-pends onto that incarnation and the architect never runs. A restart
resolved with ``recut=True`` issues a re-cut ruling, the task-3154 dispatch
gate parks the prior incarnation (nothing deleted), and the next run plans
fresh with the architect.

Two specimens of stale reuse are covered: Lever B reusing the on-disk plan
(task 5027), and boot recovery's plan snapshot handed to the slot as
``initial_plan`` (task 3310).

Driven through the real ``resolve_issue`` tool, a real ``EscalationQueue``,
the harness resolve callback and ``Harness._run_slot``, the dispatch slot that
runs the re-cut gate and decides the recovered plan. Only its
``build_workflow`` seam is replaced, by a real ``TaskWorkflow`` built on the
``_workflow_helpers`` fakes; the slot's own ``initial_plan`` is handed through.
``_run_slot`` is private: no public seam dispatches one task. Asserted on the
task store, emitted events and agent invocations.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _harness_escalation_link import drain_background_tasks, link_escalation_queue
from _task_store_doubles import TaskStoreDouble
from _workflow_helpers import (
    PLAN,
    AgentStub,
    _build_harness,
    _build_workflow_with_escalation,
    _derive_meta_root_like_production,  # noqa: F401 — autouse fixture, imported to activate
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
from orchestrator.harness import Harness, TaskReport
from orchestrator.merge_lane.gates import PLAN_FILES_NOT_TOUCHED_REASON_PREFIX
from orchestrator.recut_ruling import RECUT_EXECUTED_KEY, RECUT_RULING_KEY
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

pytestmark = pytest.mark.asyncio

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
    config: OrchestratorConfig, git_ops: GitOps, store: TaskStoreDouble, queue: EscalationQueue,
) -> Harness:
    h = _build_harness(config)
    store.wire_into(h.scheduler)
    h.scheduler.is_deterministic = MagicMock(return_value=False)  # type: ignore[method-assign]
    h.git_ops = git_ops
    h.is_workflow_active = lambda task_id: False  # type: ignore[method-assign]
    link_escalation_queue(h, queue)
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


@dataclass(frozen=True)
class _PriorIncarnation:
    tip: str
    plan: dict[str, Any]


async def _seed_prior_incarnation(git_ops: GitOps) -> _PriorIncarnation:
    """A Lever-B-eligible, plan-files-not-touched prior incarnation."""
    main_sha = await git_ops.get_main_sha()
    worktree = (await git_ops.create_worktree(TASK_ID)).path
    (worktree / 'NOTES.md').write_text('prior incarnation: touched no plan file\n')
    await _run(['git', 'add', 'NOTES.md'], cwd=worktree)
    await _run(['git', 'commit', '-m', 'prior incarnation marker'], cwd=worktree)
    _, tip, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
    tip = tip.strip()

    artifacts = TaskArtifacts(
        worktree, TaskArtifacts.meta_root_for(worktree.parent, worktree.name),
    )
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
    (artifacts.root / 'plan.json').write_text(json.dumps(plan))
    return _PriorIncarnation(tip, plan)


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
    await drain_background_tasks(harness)
    return esc


@dataclass(frozen=True)
class _Dispatch:
    report: TaskReport | None
    initial_plan: dict[str, Any] | None
    events: _EventRecorder


async def _dispatch(
    harness: Harness, config: OrchestratorConfig, git_ops: GitOps,
    store: TaskStoreDouble, agents: AgentStub, tmp_path: Path,
) -> _Dispatch:
    """The next dispatch, through the harness slot, of the task as stored."""
    events = _EventRecorder()
    built: dict[str, Any] = {}

    def _build_real_workflow(**kwargs: Any) -> TaskWorkflow:
        built.update(kwargs)
        workflow, _, _ = _build_workflow_with_escalation(
            config, git_ops, kwargs['assignment'], agents, tmp_path,
        )
        workflow.initial_plan = kwargs['initial_plan']
        workflow.event_store = events  # type: ignore[assignment]
        return workflow

    assignment = TaskAssignment(
        task_id=TASK_ID, task=await store.get_task(TASK_ID) or {}, modules=['lib'],
    )
    with patch('orchestrator.harness.build_workflow', side_effect=_build_real_workflow):
        report = await harness._run_slot(assignment, MagicMock())
    assert built, f'the slot never built the workflow: {report}'
    return _Dispatch(report, built['initial_plan'], events)


def _stage_recovered_plan(harness: Harness, plan: dict[str, Any]) -> None:
    """What boot recovery leaves for the slot when it snapshots a task's plan."""
    harness._recovered_plans[TASK_ID] = plan


def _architect_ran_before_review(agents: AgentStub) -> bool:
    first_review = agents.calls.index('reviewer_comprehensive')
    return 'architect' in agents.calls[:first_review]


async def test_plain_restart_reuses_the_prior_plan_through_lever_b(
    config, git_ops, store, queue, harness, agents, tmp_path,
):
    await _seed_prior_incarnation(git_ops)
    await _resolve_plan_touch_block(harness, queue, action='restart')
    assert store.status(TASK_ID) == 'pending'
    assert RECUT_RULING_KEY not in store.metadata(TASK_ID)

    dispatch = await _dispatch(harness, config, git_ops, store, agents, tmp_path)

    assert RECUT_EXECUTED_KEY not in store.metadata(TASK_ID)
    assert dispatch.initial_plan is None
    assert dispatch.events.lever_b_skips(), 'the seed must be Lever-B eligible'
    assert not _architect_ran_before_review(agents), (
        'the stale plan reached review without the architect re-planning it'
    )


async def test_plain_restart_hands_the_recovered_plan_to_the_workflow(
    config, git_ops, store, queue, harness, agents, tmp_path,
):
    prior = await _seed_prior_incarnation(git_ops)
    _stage_recovered_plan(harness, prior.plan)
    await _resolve_plan_touch_block(harness, queue, action='restart')

    dispatch = await _dispatch(harness, config, git_ops, store, agents, tmp_path)

    assert dispatch.initial_plan == prior.plan
    assert not _architect_ran_before_review(agents)


@pytest.mark.parametrize('recovered', [False, True], ids=['on-disk-plan', 'recovered-plan'])
async def test_recut_restart_parks_the_prior_incarnation_and_runs_the_architect(
    config, git_ops, store, queue, harness, agents, tmp_path, recovered,
):
    prior = await _seed_prior_incarnation(git_ops)
    if recovered:
        _stage_recovered_plan(harness, prior.plan)
    esc = await _resolve_plan_touch_block(harness, queue, action='restart', recut=True)
    assert store.status(TASK_ID) == 'pending'
    assert store.metadata(TASK_ID)[RECUT_RULING_KEY]['ruling_id'] == esc.id

    dispatch = await _dispatch(harness, config, git_ops, store, agents, tmp_path)

    assert dispatch.report is not None
    assert dispatch.report.outcome is not WorkflowOutcome.BLOCKED, dispatch.report
    assert dispatch.initial_plan is None
    assert agents.calls[0] == 'architect'
    assert dispatch.events.lever_b_skips() == []
    executed = store.metadata(TASK_ID)[RECUT_EXECUTED_KEY]
    assert executed['ruling_id'] == esc.id
    parked_branch = executed['parked_branch']
    assert parked_branch.startswith(f'task/{TASK_ID}-')
    _, parked_sha, _ = await _run(
        ['git', 'rev-parse', parked_branch], cwd=config.project_root,
    )
    assert parked_sha.strip() == prior.tip
    parked_plans = [
        json.loads(p.read_text()) for p in git_ops.quarantine_base.rglob('plan.json')
    ]
    assert [p.get('_session_id') for p in parked_plans] == ['prior-session']
