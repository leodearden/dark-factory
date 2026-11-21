"""A dependent dispatched on a predecessor's branch (task 10473, early start T2a).

Drives the public ``TaskWorkflow.run()`` over a real GitOps and repo. The
workflow is built by the shared
``_workflow_helpers.py::_build_workflow_with_escalation`` builder with a stack
target; ``build_workflow``'s pass-through of ``stack_target`` is pinned in
``test_workflow_factory.py``.
Task D's worktree is created, or re-stacked, by the workflow's own
``create_worktree``; the scenes are the ones ``_stacked_repo`` builds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from _stacked_repo import (
    RestackScene,
    add_worktree,
    build_restack_scene,
    commit_files,
    fail_git_calls_starting,
    git,
    ledger_of,
    make_repo,
    rev_parse,
)
from _workflow_helpers import (
    AgentStub,
    FakeScheduler,
    _build_workflow_with_escalation,
    _make,
    agent_side_artifacts,
)
from escalation.queue import EscalationQueue

from orchestrator.branch_stack import StackRecordState, StackTarget
from orchestrator.config import GitConfig, OrchestratorConfig, SandboxConfig
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome
from orchestrator.workflow_types import TerminalReport

_GUARDED_REBASE_ABORT = '-c rerere.enabled=false rebase --abort'


class _StopAfterSetup(RuntimeError):
    """Raised by the first agent the workflow invokes, once setup is done."""


def _config(repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=1,
        git=GitConfig(main_branch='main', branch_prefix='task/', remote='origin', worktree_dir='.worktrees'),
        sandbox=SandboxConfig(enabled=False),
    )


def _assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id='D',
        task={
            'id': 'D',
            'title': 'Build on P',
            'description': "Add d.txt on top of P's work",
            'status': 'pending',
            'metadata': {'files': ['d.txt']},
            'dependencies': [],
        },
        modules=[],
    )


def _workflow(
    git_ops: GitOps, target: StackTarget, tmp_path: Path, *, escalation_queue: EscalationQueue | None = None,
) -> tuple[TaskWorkflow, FakeScheduler]:
    workflow, scheduler, _ = _build_workflow_with_escalation(
        _config(git_ops.project_root), git_ops, _assignment(), AgentStub(), tmp_path,
        spawn_merge_worker=False, queue=escalation_queue, stack_target=target,
    )
    return workflow, scheduler


class _FirstAgentCall:
    """Stands in for ``invoke_agent``: notes the base commit the first agent
    would see, then stops the run."""

    def __init__(self) -> None:
        self.base_commit: str | None = None

    async def __call__(self, prompt: str, system_prompt: str, cwd: Path, *args: Any, **kwargs: Any) -> Any:
        self.base_commit = agent_side_artifacts(cwd).read_base_commit()
        raise _StopAfterSetup('setup is done')


async def _run(workflow: TaskWorkflow, monkeypatch: pytest.MonkeyPatch) -> tuple[TerminalReport, _FirstAgentCall]:
    first_agent = _FirstAgentCall()
    monkeypatch.setattr('orchestrator.workflow.invoke_agent', first_agent)
    return await workflow.run(), first_agent


async def _assert_left_as_it_was(scene: RestackScene) -> None:
    assert await rev_parse(scene.root, 'task/D') == scene.d_tip
    assert await scene.ledger.base_of('task/D') == scene.p_old


@pytest.mark.asyncio
class TestStackedDispatch:
    async def test_a_fresh_dispatch_starts_on_the_target_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        git_ops = await make_repo(tmp_path / 'repo')
        p_wt = await add_worktree(git_ops, 'P', 'main')
        p_tip = await commit_files(p_wt, {'p.txt': 'p\n'}, 'p1: write p.txt')
        workflow, _ = _workflow(git_ops, StackTarget('task/P', p_tip), tmp_path)

        _, first_agent = await _run(workflow, monkeypatch)

        assert workflow.worktree is not None
        assert (await git(workflow.worktree, 'symbolic-ref', 'HEAD')).strip() == 'refs/heads/task/D'
        assert await rev_parse(workflow.worktree, 'HEAD') == p_tip
        read = await ledger_of(git_ops).read('task/D')
        assert (read.state, read.base) == (StackRecordState.RECORDED, p_tip)
        assert first_agent.base_commit == p_tip

    async def test_a_conflicting_restack_requeues_uncounted_and_moves_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scene = await build_restack_scene(await make_repo(tmp_path / 'repo'), conflicting=True)
        workflow, scheduler = _workflow(scene.git_ops, StackTarget('task/P', scene.p_new), tmp_path)

        report, first_agent = await _run(workflow, monkeypatch)

        assert report.outcome == WorkflowOutcome.REQUEUED
        assert report.counts_against_requeue_cap is False
        assert report.reason.startswith('restack_conflict')
        assert scheduler.statuses['D'][-1] == 'pending'
        assert first_agent.base_commit is None
        await _assert_left_as_it_was(scene)

    async def test_a_failed_abort_blocks_with_the_recovery_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scene = await build_restack_scene(await make_repo(tmp_path / 'repo'), conflicting=True)
        fail_git_calls_starting(monkeypatch, tmp_path / 'shim', _GUARDED_REBASE_ABORT)
        queue = EscalationQueue(tmp_path / 'escalation_queue')
        workflow, _ = _workflow(
            scene.git_ops, StackTarget('task/P', scene.p_new), tmp_path, escalation_queue=queue,
        )

        report, first_agent = await _run(workflow, monkeypatch)

        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.reason.startswith('Re-stack blocked')
        assert first_agent.base_commit is None
        [escalation] = queue.get_by_task('D', level=0)
        assert 'task/D' in escalation.detail
        assert 'rebase --abort' in escalation.detail
        assert 'orchestrator.rebase_recovery preflight' in escalation.detail


# ---------------------------------------------------------------------------
# PRD row 28: on a stacked branch, own work is the commits above the stack
# base, before the predecessor lands and after it lands either way.
# ---------------------------------------------------------------------------


class Landing(Enum):
    NOT_YET = 'not_yet'
    MERGED = 'merged'
    REAPPLIED = 'reapplied'


@dataclass(frozen=True)
class _OwnWorkScene:
    git_ops: GitOps
    d_worktree: Path
    d_base: str


async def _own_work_scene(root: Path, landing: Landing) -> _OwnWorkScene:
    """task/P adds p.py; task/D is stacked on P's tip and adds d.py; then P
    lands on main as *landing* says."""
    git_ops = await make_repo(root)
    p_wt = await add_worktree(git_ops, 'P', 'main')
    p_tip = await commit_files(p_wt, {'p.py': 'p = 1\n'}, 'p1: add p.py')
    info = await git_ops.create_worktree('D', stack_on=StackTarget('task/P', p_tip))
    await commit_files(info.path, {'d.py': 'd = 1\n'}, 'd1: add d.py')
    if landing is Landing.MERGED:
        await git(root, 'merge', '--no-ff', '-m', 'Merge task/P into main', p_tip)
    elif landing is Landing.REAPPLIED:
        await commit_files(root, {'m1.txt': 'm1\n'}, 'M1: main advances')
        await git(root, 'cherry-pick', p_tip)
    return _OwnWorkScene(git_ops=git_ops, d_worktree=info.path, d_base=info.base_commit)


class _PassedTheGate(Exception):
    """The plan-files gate let the branch through to the enqueue."""


def _own_work_workflow(
    scene: _OwnWorkScene, plan_files: list[str], monkeypatch: pytest.MonkeyPatch,
) -> tuple[TaskWorkflow, AsyncMock]:
    f = _make(worktree=scene.d_worktree, project_root=scene.git_ops.project_root, task_id='D')
    wf = f.wf
    wf.git_ops = scene.git_ops
    f.artifacts.update_base_commit(scene.d_base)
    wf._base_commit = scene.d_base
    wf.plan = {'files': plan_files}
    wf.merge_queue = MagicMock()
    wf.config.review.briefing_path = 'df-10473-no-such-briefing.yaml'
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    monkeypatch.setattr(wf, '_mark_blocked', mark_blocked)
    monkeypatch.setattr(wf, '_try_narrow_plan', AsyncMock(return_value=False))
    monkeypatch.setattr(scene.git_ops, 'rebind_branch_to_head', AsyncMock(side_effect=_PassedTheGate))
    return wf, mark_blocked


@pytest.mark.asyncio
@pytest.mark.parametrize('landing', list(Landing))
class TestOwnWorkOnAStackedBranch:
    async def test_a_file_only_the_base_touched_is_reported_not_touched(
        self, tmp_path: Path, landing: Landing, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scene = await _own_work_scene(tmp_path / 'repo', landing)
        wf, mark_blocked = _own_work_workflow(scene, ['d.py', 'p.py'], monkeypatch)

        await wf._submit_to_merge_queue('D')

        mark_blocked.assert_awaited_once()
        args, kwargs = mark_blocked.call_args
        reason = args[0] if args else kwargs['reason']
        assert 'no commit on the branch touched them: p.py.' in reason

    async def test_the_own_file_alone_passes_the_gate(
        self, tmp_path: Path, landing: Landing, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scene = await _own_work_scene(tmp_path / 'repo', landing)
        wf, mark_blocked = _own_work_workflow(scene, ['d.py'], monkeypatch)

        with pytest.raises(_PassedTheGate):
            await wf._submit_to_merge_queue('D')

        mark_blocked.assert_not_awaited()

    async def test_the_review_subject_is_the_own_work_alone(
        self, tmp_path: Path, landing: Landing, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scene = await _own_work_scene(tmp_path / 'repo', landing)
        wf, _ = _own_work_workflow(scene, ['d.py'], monkeypatch)

        subject = await wf._context_complete_review_subject()

        assert {changed.path for changed in subject.change.files} == {'d.py'}
