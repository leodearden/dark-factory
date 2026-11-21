"""A dependent dispatched on a predecessor's branch (task 10473, early start T2a).

Drives the public ``TaskWorkflow.run()`` over a real GitOps and repo, built
through ``build_workflow`` with a stack target, as the harness's launch will.
Task D's worktree is created, or re-stacked, by the workflow's own
``create_worktree``; the scenes are the ones ``_stacked_repo`` builds.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

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
from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler, agent_side_artifacts
from escalation.queue import EscalationQueue

from orchestrator.branch_stack import StackRecordState, StackTarget
from orchestrator.config import GitConfig, OrchestratorConfig, SandboxConfig
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome, build_workflow
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
    git_ops: GitOps, target: StackTarget, *, escalation_queue: EscalationQueue | None = None,
) -> tuple[TaskWorkflow, FakeScheduler]:
    scheduler = FakeScheduler()
    workflow = build_workflow(
        assignment=_assignment(),
        config=_config(git_ops.project_root),
        git_ops=git_ops,
        scheduler=scheduler,  # type: ignore[arg-type]
        briefing=FakeBriefing(),  # type: ignore[arg-type]
        mcp=FakeMcp(),  # type: ignore[arg-type]
        escalation_queue=escalation_queue,
        merge_queue=asyncio.Queue(),
        stack_target=target,
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
        workflow, _ = _workflow(git_ops, StackTarget('task/P', p_tip))

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
        workflow, scheduler = _workflow(scene.git_ops, StackTarget('task/P', scene.p_new))

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
        workflow, _ = _workflow(scene.git_ops, StackTarget('task/P', scene.p_new), escalation_queue=queue)

        report, first_agent = await _run(workflow, monkeypatch)

        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.reason.startswith('Re-stack blocked')
        assert first_agent.base_commit is None
        [escalation] = queue.get_by_task('D')
        assert 'task/D' in escalation.detail
        assert 'rebase --abort' in escalation.detail
        assert 'orchestrator.rebase_recovery preflight' in escalation.detail
