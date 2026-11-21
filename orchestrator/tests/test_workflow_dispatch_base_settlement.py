"""A resumed branch that conflicts with main resolves or escalates before planning (task 7736).

Drives the public ``TaskWorkflow.run()`` over a real GitOps and repo: task 42's
worktree is pre-created with a commit that conflicts with a later commit on
main, so the workflow's own ``create_worktree`` takes the real reuse path,
whose rebase fails and aborts exactly as in production. The same scene pins
the train and external-worktree skips and the fail-open probe.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _dispatch_base_scene import (
    GREET_FILE,
    FailFirst,
    commit_file,
    git_stdout,
    greeting,
    has_merge_head,
    is_ancestor,
    make_repo,
)
from _git_fault_shim import fail_git
from _recording_event_store import _RecordingEventStore
from _workflow_helpers import AgentStub, _build_workflow, _make_git_ops, agent_side_artifacts

from orchestrator.agents.invoke import AgentResult
from orchestrator.branch_stack import StackBaseLedger, StackTarget
from orchestrator.config import GitConfig, OrchestratorConfig, SandboxConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.mcp.verdict_tools import _envelope
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import WorkflowOutcome
from orchestrator.workflow_types import TerminalReport

EVENT = 'dispatch_base_conflict'


def _config(repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=1,
        max_execute_iterations=5,
        max_verify_attempts=3,
        max_review_cycles=2,
        judge_after_each_iteration=False,
        max_amendment_rounds=1,
        git=GitConfig(main_branch='main', branch_prefix='task/', remote='origin', worktree_dir='.worktrees'),
        sandbox=SandboxConfig(enabled=False),
    )


def _assignment(extra_metadata: dict[str, Any]) -> TaskAssignment:
    return TaskAssignment(
        task_id='42',
        task={
            'id': '42',
            'title': 'Add farewell function',
            'description': 'Add a farewell(name) function to lib.py with tests',
            'status': 'pending',
            'metadata': {'files': ['lib'], **extra_metadata},
            'dependencies': [],
        },
        modules=['lib'],
    )


class _Scene:
    """Task 42's resumed worktree, its tip, and main's tip, before the dispatch."""

    def __init__(self, repo: Path, worktree: Path, head_sha: str, main_sha: str) -> None:
        self.repo = repo
        self.worktree = worktree
        self.head_sha = head_sha
        self.main_sha = main_sha


async def _scene(git_ops: GitOps, *, main_file: str) -> _Scene:
    worktree = (await git_ops.create_worktree('42')).path
    head_sha = commit_file(worktree, GREET_FILE, greeting('Hi'), 'task: greet with Hi')
    main_content = greeting('Hey') if main_file == GREET_FILE else 'main notes\n'
    main_sha = commit_file(git_ops.project_root, main_file, main_content, 'main: advance')
    return _Scene(git_ops.project_root, worktree, head_sha, main_sha)


class MergerAct(Enum):
    """What the stub merger does with the dispatch-start merge of main."""

    RESOLVES = 'resolves'
    DECLINES = 'declines'
    WRITES_NO_VERDICT = 'writes_no_verdict'


class _SettlingStub(AgentStub):
    """An AgentStub whose merger acts on the dispatch-start merge of main as *act* says."""

    def __init__(self, act: MergerAct, *, main_sha: str) -> None:
        super().__init__()
        self._act = act
        self._main_sha = main_sha
        self.architect_saw_main: bool | None = None
        self.architect_base_commit: str | None = None

    async def invoke_agent(self, prompt: str, system_prompt: str, cwd: Path, *args: Any, **kwargs: Any) -> AgentResult:
        role = self._detect_role(system_prompt)
        if role == 'merger':
            self.calls.append(role)
            return self._settle(cwd)
        if role == 'architect' and self.architect_saw_main is None:
            self.architect_saw_main = is_ancestor(cwd, self._main_sha, 'HEAD')
            self.architect_base_commit = agent_side_artifacts(cwd).read_base_commit()
        return await super().invoke_agent(prompt, system_prompt, cwd, *args, **kwargs)

    def _settle(self, cwd: Path) -> AgentResult:
        if self._act is MergerAct.WRITES_NO_VERDICT:
            return AgentResult(success=True, output='merger lost its sandbox', cost_usd=0.2)
        if self._act is MergerAct.RESOLVES:
            (cwd / GREET_FILE).write_text(greeting('Hey'))
            git_stdout(cwd, 'add', GREET_FILE)
            git_stdout(cwd, 'commit', '--no-edit')
            verdict = {'blocked': False, 'reason': ''}
        else:
            verdict = {'blocked': True, 'reason': 'cannot combine'}
        agent_side_artifacts(cwd).write_verdict('merger', _envelope('merger', 'sid', verdict))
        return AgentResult(success=True, output='merger done', cost_usd=0.2)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path)


@pytest.fixture
def git_ops(repo: Path) -> GitOps:
    return _make_git_ops(repo)


async def _run(
    repo: Path, git_ops: GitOps, stub: AgentStub, monkeypatch: pytest.MonkeyPatch,
    *, extra_metadata: dict[str, Any] | None = None, external_worktree: Path | None = None,
) -> tuple[TerminalReport, _RecordingEventStore]:
    workflow, _ = _build_workflow(_config(repo), git_ops, _assignment(extra_metadata or {}), stub)
    workflow.worktree = external_worktree
    store = _RecordingEventStore()
    monkeypatch.setattr(workflow, 'event_store', store)
    monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
    monkeypatch.setattr(
        'orchestrator.workflow.run_scoped_verification',
        AsyncMock(return_value=VerifyResult(
            passed=True, test_output='OK', lint_output='', type_output='', summary='All checks passed',
        )),
    )
    return await workflow.run(), store


def _events(store: _RecordingEventStore, name: str) -> list[dict]:
    return [payload['data'] for event, payload in store.events if event == name]


@pytest.mark.asyncio
class TestResumedConflictingBranch:
    async def test_conflict_is_resolved_before_planning(self, repo, git_ops, monkeypatch):
        scene = await _scene(git_ops, main_file=GREET_FILE)
        stub = _SettlingStub(MergerAct.RESOLVES, main_sha=scene.main_sha)

        report, store = await _run(repo, git_ops, stub, monkeypatch)

        assert stub.calls[0] == 'merger'
        assert 'architect' in stub.calls[1:]
        assert stub.architect_saw_main is True
        assert stub.architect_base_commit == scene.main_sha
        [data] = _events(store, EVENT)
        assert data['conflicted_paths'] == [GREET_FILE]
        assert data['conflicted_paths_total'] == 1
        assert data['counterpart_task_ids'] == []
        assert data['main_sha'] == scene.main_sha
        assert data['head_sha'] == scene.head_sha
        assert report.outcome == WorkflowOutcome.DONE

    async def test_declined_conflict_escalates_before_planning(self, repo, git_ops, monkeypatch):
        scene = await _scene(git_ops, main_file=GREET_FILE)
        stub = _SettlingStub(MergerAct.DECLINES, main_sha=scene.main_sha)

        report, _ = await _run(repo, git_ops, stub, monkeypatch)

        assert stub.calls == ['merger']
        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.block_category == 'merge_needs_rebase'
        assert git_stdout(scene.worktree, 'rev-parse', 'HEAD') == scene.head_sha
        assert not has_merge_head(scene.worktree)

    async def test_merger_without_a_verdict_escalates_before_planning(self, repo, git_ops, monkeypatch):
        scene = await _scene(git_ops, main_file=GREET_FILE)
        stub = _SettlingStub(MergerAct.WRITES_NO_VERDICT, main_sha=scene.main_sha)

        report, _ = await _run(repo, git_ops, stub, monkeypatch)

        assert stub.calls == ['merger']
        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.block_category == 'merge_needs_rebase'
        assert 'merger lost its sandbox' in report.detail
        assert git_stdout(scene.worktree, 'rev-parse', 'HEAD') == scene.head_sha
        assert not has_merge_head(scene.worktree)

    async def test_clean_resume_is_untouched(self, repo, git_ops, monkeypatch):
        scene = await _scene(git_ops, main_file='notes.txt')
        stub = _SettlingStub(MergerAct.RESOLVES, main_sha=scene.main_sha)

        _, store = await _run(repo, git_ops, stub, monkeypatch)

        assert stub.calls[0] == 'architect'
        assert _events(store, EVENT) == []


@pytest.mark.asyncio
class TestSettlementGuards:
    """Each guard leaves a conflicting resumed branch to plan unsettled."""

    async def test_train_member_is_skipped(self, repo, git_ops, monkeypatch):
        scene = await _scene(git_ops, main_file=GREET_FILE)
        stub = _SettlingStub(MergerAct.DECLINES, main_sha=scene.main_sha)

        _, store = await _run(
            repo, git_ops, stub, monkeypatch, extra_metadata={'train': {'id': 't1', 'order': 0}},
        )

        assert stub.calls[0] == 'architect'
        assert stub.architect_saw_main is False
        assert _events(store, EVENT) == []

    async def test_external_worktree_is_skipped(self, repo, git_ops, monkeypatch):
        scene = await _scene(git_ops, main_file=GREET_FILE)
        stub = _SettlingStub(MergerAct.DECLINES, main_sha=scene.main_sha)

        _, store = await _run(repo, git_ops, stub, monkeypatch, external_worktree=scene.worktree)

        assert stub.calls[0] == 'architect'
        assert stub.architect_saw_main is False
        assert _events(store, EVENT) == []

    async def test_probe_failure_fails_open_into_planning(self, repo, git_ops, monkeypatch):
        scene = await _scene(git_ops, main_file=GREET_FILE)
        stub = _SettlingStub(MergerAct.DECLINES, main_sha=scene.main_sha)
        probe = FailFirst(git_ops.merge_tree_conflicts, RuntimeError('boom'))
        monkeypatch.setattr(git_ops, 'merge_tree_conflicts', probe)

        _, store = await _run(repo, git_ops, stub, monkeypatch)

        assert probe.calls >= 1
        assert stub.calls[0] == 'architect'
        assert stub.architect_saw_main is False
        assert _events(store, EVENT) == []


class _StopsAtPlanning(RuntimeError):
    """The architect was reached: the dispatch-start settle is behind the run."""


class _StopsAtArchitect(_SettlingStub):
    """A _SettlingStub whose architect notes whether it sees main, then stops the run."""

    async def invoke_agent(self, prompt: str, system_prompt: str, cwd: Path, *args: Any, **kwargs: Any) -> AgentResult:
        if self._detect_role(system_prompt) == 'architect':
            self.calls.append('architect')
            self.architect_saw_main = is_ancestor(cwd, self._main_sha, 'HEAD')
            raise _StopsAtPlanning('planning reached')
        return await super().invoke_agent(prompt, system_prompt, cwd, *args, **kwargs)


class _StackedScene(_Scene):
    def __init__(self, scene: _Scene, p_tip: str) -> None:
        super().__init__(scene.repo, scene.worktree, scene.head_sha, scene.main_sha)
        self.p_tip = p_tip


async def _stacked_scene(git_ops: GitOps) -> _StackedScene:
    """Task 42 stacked on task/P's unlanded tip; 42's own commit conflicts
    with a later main commit, so the reuse route's un-stack stops."""
    p_worktree = (await git_ops.create_worktree('P')).path
    p_tip = commit_file(p_worktree, 'p.txt', 'p\n', 'p: add p.txt')
    worktree = (await git_ops.create_worktree('42', stack_on=StackTarget('task/P', p_tip))).path
    head_sha = commit_file(worktree, GREET_FILE, greeting('Hi'), 'task: greet with Hi')
    main_sha = commit_file(git_ops.project_root, GREET_FILE, greeting('Hey'), 'main: advance')
    return _StackedScene(_Scene(git_ops.project_root, worktree, head_sha, main_sha), p_tip)


def _assert_main_not_merged_in(scene: _Scene) -> None:
    subjects = git_stdout(scene.repo, 'log', '--format=%s', 'main..task/42').splitlines()
    assert not [s for s in subjects if s.startswith('Merge main into')]
    assert not is_ancestor(scene.repo, scene.main_sha, 'task/42')
    assert not has_merge_head(scene.worktree)


@pytest.mark.asyncio
class TestStackedBranchIsNotSettled:
    """Main is never merged into a branch that is, or may be, stacked on an
    unlanded predecessor (task 10473); the ordinary conflicting branch in
    TestResumedConflictingBranch is settled, as the contrast."""

    async def test_a_recorded_stack_base_skips_the_settle(self, repo, git_ops, monkeypatch):
        scene = await _stacked_scene(git_ops)
        stub = _StopsAtArchitect(MergerAct.RESOLVES, main_sha=scene.main_sha)

        _, store = await _run(repo, git_ops, stub, monkeypatch)

        assert stub.calls[0] == 'architect'
        assert 'merger' not in stub.calls
        assert stub.architect_saw_main is False
        assert _events(store, EVENT) == []
        _assert_main_not_merged_in(scene)
        assert await StackBaseLedger(repo, _run).base_of('task/42') == scene.p_tip

    async def test_an_unreadable_ledger_skips_the_settle(self, repo, git_ops, monkeypatch, tmp_path):
        scene = await _stacked_scene(git_ops)
        stub = _StopsAtArchitect(MergerAct.RESOLVES, main_sha=scene.main_sha)
        fail_git(monkeypatch, tmp_path / 'shim' / 'bin', 'for-each-ref *refs/dark-factory/stack-base/*')

        _, store = await _run(repo, git_ops, stub, monkeypatch)

        assert (tmp_path / 'shim' / 'bin' / 'matched').exists()
        assert stub.calls[0] == 'architect'
        assert 'merger' not in stub.calls
        assert stub.architect_saw_main is False
        assert _events(store, EVENT) == []
        _assert_main_not_merged_in(scene)
