"""A resumed branch that conflicts with main resolves or escalates before planning (task 7736).

Drives the public ``TaskWorkflow.run()`` over a real GitOps and repo: task 42's
worktree is pre-created with a commit that conflicts with a later commit on
main, so the workflow's own ``create_worktree`` takes the real reuse path,
whose rebase fails and aborts exactly as in production. The same scene pins
the train and external-worktree skips and the fail-open probe.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _recording_event_store import _RecordingEventStore
from _workflow_helpers import AgentStub, _build_workflow, _init_repo, agent_side_artifacts

from orchestrator.agents.invoke import AgentResult
from orchestrator.config import GitConfig, OrchestratorConfig, SandboxConfig
from orchestrator.git_ops import GitOps
from orchestrator.mcp.verdict_tools import _envelope
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import WorkflowOutcome
from orchestrator.workflow_types import TerminalReport

GREET_FILE = 'lib.py'
EVENT = 'dispatch_base_conflict'


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def _is_ancestor(cwd: Path, ancestor: str, descendant: str) -> bool:
    probe = subprocess.run(['git', 'merge-base', '--is-ancestor', ancestor, descendant], cwd=cwd)
    return probe.returncode == 0


def _has_merge_head(cwd: Path) -> bool:
    probe = subprocess.run(
        ['git', 'rev-parse', '-q', '--verify', 'MERGE_HEAD'], cwd=cwd, capture_output=True,
    )
    return probe.returncode == 0


def _commit_file(cwd: Path, name: str, content: str, message: str) -> str:
    (cwd / name).write_text(content)
    _git(cwd, 'add', name)
    _git(cwd, 'commit', '-m', message)
    return _git(cwd, 'rev-parse', 'HEAD')


def _greeting(word: str) -> str:
    return f'def greet(name: str) -> str:\n    return f"{word}, {{name}}"\n'


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
    head_sha = _commit_file(worktree, GREET_FILE, _greeting('Hi'), 'task: greet with Hi')
    main_content = _greeting('Hey') if main_file == GREET_FILE else 'main notes\n'
    main_sha = _commit_file(git_ops.project_root, main_file, main_content, 'main: advance')
    return _Scene(git_ops.project_root, worktree, head_sha, main_sha)


class _SettlingStub(AgentStub):
    """An AgentStub whose merger concludes or declines the dispatch-start merge of main."""

    def __init__(self, *, resolve: bool, main_sha: str) -> None:
        super().__init__()
        self._resolve = resolve
        self._main_sha = main_sha
        self.architect_saw_main: bool | None = None

    async def invoke_agent(self, prompt: str, system_prompt: str, cwd: Path, *args: Any, **kwargs: Any) -> AgentResult:
        role = self._detect_role(system_prompt)
        if role == 'merger':
            self.calls.append(role)
            return self._settle(cwd)
        if role == 'architect' and self.architect_saw_main is None:
            self.architect_saw_main = _is_ancestor(cwd, self._main_sha, 'HEAD')
        return await super().invoke_agent(prompt, system_prompt, cwd, *args, **kwargs)

    def _settle(self, cwd: Path) -> AgentResult:
        if self._resolve:
            (cwd / GREET_FILE).write_text(_greeting('Hey'))
            _git(cwd, 'add', GREET_FILE)
            _git(cwd, 'commit', '--no-edit')
            verdict = {'blocked': False, 'reason': ''}
        else:
            verdict = {'blocked': True, 'reason': 'cannot combine'}
        agent_side_artifacts(cwd).write_verdict('merger', _envelope('merger', 'sid', verdict))
        return AgentResult(success=True, output='merger done', cost_usd=0.2)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / 'repo'
    path.mkdir()
    _init_repo(path)
    return path


@pytest.fixture
def git_ops(repo: Path) -> GitOps:
    return GitOps(_config(repo).git, repo)


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
        stub = _SettlingStub(resolve=True, main_sha=scene.main_sha)

        report, store = await _run(repo, git_ops, stub, monkeypatch)

        assert stub.calls[0] == 'merger'
        assert 'architect' in stub.calls[1:]
        assert stub.architect_saw_main is True
        [data] = _events(store, EVENT)
        assert data['conflicted_paths'] == [GREET_FILE]
        assert data['conflicted_paths_total'] == 1
        assert data['counterpart_task_ids'] == []
        assert data['main_sha'] == scene.main_sha
        assert data['head_sha'] == scene.head_sha
        assert report.outcome == WorkflowOutcome.DONE

    async def test_declined_conflict_escalates_before_planning(self, repo, git_ops, monkeypatch):
        scene = await _scene(git_ops, main_file=GREET_FILE)
        stub = _SettlingStub(resolve=False, main_sha=scene.main_sha)

        report, _ = await _run(repo, git_ops, stub, monkeypatch)

        assert stub.calls == ['merger']
        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.block_category == 'merge_needs_rebase'
        assert _git(scene.worktree, 'rev-parse', 'HEAD') == scene.head_sha
        assert not _has_merge_head(scene.worktree)

    async def test_clean_resume_is_untouched(self, repo, git_ops, monkeypatch):
        scene = await _scene(git_ops, main_file='notes.txt')
        stub = _SettlingStub(resolve=True, main_sha=scene.main_sha)

        _, store = await _run(repo, git_ops, stub, monkeypatch)

        assert stub.calls[0] == 'architect'
        assert _events(store, EVENT) == []


class _FailFirstProbe:
    """GitOps.merge_tree_conflicts that raises on its first call, then answers for real."""

    def __init__(self, git_ops: GitOps) -> None:
        self._real = git_ops.merge_tree_conflicts
        self.calls = 0

    async def __call__(self, base_tip: str, branch_head: str) -> Any:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError('boom')
        return await self._real(base_tip, branch_head)


@pytest.mark.asyncio
class TestSettlementGuards:
    """Each guard leaves a conflicting resumed branch to plan unsettled."""

    async def test_train_member_is_skipped(self, repo, git_ops, monkeypatch):
        scene = await _scene(git_ops, main_file=GREET_FILE)
        stub = _SettlingStub(resolve=False, main_sha=scene.main_sha)

        _, store = await _run(
            repo, git_ops, stub, monkeypatch, extra_metadata={'train': {'id': 't1', 'order': 0}},
        )

        assert stub.calls[0] == 'architect'
        assert stub.architect_saw_main is False
        assert _events(store, EVENT) == []

    async def test_external_worktree_is_skipped(self, repo, git_ops, monkeypatch):
        scene = await _scene(git_ops, main_file=GREET_FILE)
        stub = _SettlingStub(resolve=False, main_sha=scene.main_sha)

        _, store = await _run(repo, git_ops, stub, monkeypatch, external_worktree=scene.worktree)

        assert stub.calls[0] == 'architect'
        assert stub.architect_saw_main is False
        assert _events(store, EVENT) == []

    async def test_probe_failure_fails_open_into_planning(self, repo, git_ops, monkeypatch):
        scene = await _scene(git_ops, main_file=GREET_FILE)
        stub = _SettlingStub(resolve=False, main_sha=scene.main_sha)
        probe = _FailFirstProbe(git_ops)
        monkeypatch.setattr(git_ops, 'merge_tree_conflicts', probe)

        _, store = await _run(repo, git_ops, stub, monkeypatch)

        assert probe.calls >= 1
        assert stub.calls[0] == 'architect'
        assert stub.architect_saw_main is False
        assert _events(store, EVENT) == []
