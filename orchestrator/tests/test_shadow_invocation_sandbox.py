"""S6: a shadow cell's sandboxed legs reach the landlock wrapper, a corpus cell's do not (task 5386).

The dispatch path is the real one end to end: ``TaskWorkflow._invoke`` in a
real detached eval worktree records the ``invoke_agent`` kwargs, and the REAL
``invoke_agent`` turns them into the argv a subprocess would be spawned with.
Only the spawn itself is faked.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from _git_fixtures import seed_repo
from _workflow_helpers import FakeBriefing, FakeScheduler, _make_git_ops, fixed_prompt
from shared.cli_invoke import AgentResult, _SubprocessResult

from orchestrator.agents import sandbox_dispatch
from orchestrator.agents.invocation_scope import invocation_scope
from orchestrator.agents.invoke import invoke_agent
from orchestrator.agents.roles import ARCHITECT, IMPLEMENTER, AgentRole
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.live_fixture import build_live_fixture
from orchestrator.evals.runner import build_eval_orch_config, eval_mcp_lifecycle
from orchestrator.evals.shadow_invocation import build_shadow_orch_config
from orchestrator.evals.snapshots import create_eval_worktree
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import build_workflow

_IMPL = EvalConfig('probe', 'claude', 'sonnet', 'high')


class _EvalCell:
    def __init__(self, repo: Path, fixture: dict, worktree: Path) -> None:
        self.repo = repo
        self.fixture = fixture
        self.worktree = worktree

    def base_config(self) -> OrchestratorConfig:
        return OrchestratorConfig(
            project_root=self.repo, max_concurrent_tasks=1,
            git=GitConfig(main_branch='main', branch_prefix='task/', remote='origin',
                          worktree_dir='.worktrees'),
        )


@pytest.fixture
def landlock_host(monkeypatch, tmp_path: Path) -> None:
    sandbox_dispatch.set_backend('landlock')
    monkeypatch.setattr('orchestrator.agents.landlock.is_landlock_available', lambda: True)
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))


@pytest.fixture
def pinned_claude(monkeypatch, tmp_path: Path) -> str:
    exec_file = tmp_path / 'claude'
    exec_file.write_text('#!/bin/sh\nexit 0\n')
    exec_file.chmod(0o755)
    monkeypatch.setenv('CLAUDE_BINARY', str(exec_file))
    return str(exec_file)


@pytest_asyncio.fixture
async def eval_cell(tmp_path: Path) -> _EvalCell:
    repo = seed_repo(tmp_path / 'repo')
    sha = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()
    fixture = build_live_fixture(
        {'id': '4242', 'title': 'T', 'metadata': {}},
        base_sha=sha,
        project_root=repo,
        plan={'steps': [{'id': 'step-1', 'type': 'test', 'description': 'd', 'status': 'pending'}]},
        verify_commands={'test': 'true', 'lint': 'true', 'typecheck': 'true'},
        shape='implementer',
        cell_id='0123456789abcdef0123456789abcdef',
    )
    worktree, _run_id = await create_eval_worktree(repo, fixture['id'], sha)
    return _EvalCell(repo, fixture, worktree)


async def _dispatched_kwargs(
    cell: _EvalCell, orch: OrchestratorConfig, role: AgentRole, monkeypatch,
) -> dict[str, Any]:
    """Drive the real ``TaskWorkflow._invoke`` once; return the kwargs ``invoke_agent`` got."""
    recorded: list[dict[str, Any]] = []

    async def record(**kwargs: Any) -> AgentResult:
        recorded.append(kwargs)
        return AgentResult(success=True, output='')

    monkeypatch.setattr('orchestrator.workflow.invoke_agent', record)
    fixture = cell.fixture
    workflow = build_workflow(
        assignment=TaskAssignment(
            task_id=fixture['id'], task=fixture['task_definition'], modules=[],
        ),
        config=orch,
        git_ops=_make_git_ops(cell.repo),
        scheduler=FakeScheduler(),
        briefing=FakeBriefing(),
        mcp=eval_mcp_lifecycle(orch),
        initial_plan=fixture['plan'],
    )
    workflow.worktree = cell.worktree

    await workflow._invoke(role, fixed_prompt('P'), cell.worktree)

    assert len(recorded) == 1, f'expected one invoke_agent dispatch, got {len(recorded)}'
    return recorded[0]


async def _spawned_argv(kwargs: dict[str, Any], monkeypatch) -> list[str]:
    """Run the REAL ``invoke_agent`` on *kwargs*; return the argv it would spawn."""
    captured: list[list[str]] = []

    async def fake_run_subprocess(cmd: list[str], *args: Any, **kw: Any) -> _SubprocessResult:
        captured.append(list(cmd))
        return _SubprocessResult(
            stdout=json.dumps({
                'result': 'ok', 'is_error': False, 'subtype': 'success',
                'cost_usd': 0.01, 'duration_ms': 100, 'num_turns': 1,
                'session_id': 'test-session',
            }),
            stderr='', returncode=0, duration_ms=100, timed_out=False,
        )

    monkeypatch.setattr('orchestrator.agents.invoke._run_subprocess', fake_run_subprocess)
    monkeypatch.setattr('shared.cli_invoke._run_subprocess', fake_run_subprocess)

    await invoke_agent(**kwargs)

    assert len(captured) == 1, f'expected one spawn, got {captured!r}'
    return captured[0]


@pytest.mark.asyncio
@pytest.mark.usefixtures('landlock_host')
async def test_shadow_implementer_is_strict_and_landlock_wrapped(
    eval_cell: _EvalCell, pinned_claude: str, monkeypatch,
) -> None:
    orch = build_shadow_orch_config(_IMPL, eval_cell.fixture, eval_cell.base_config())
    worktree = str(eval_cell.worktree.resolve())

    kwargs = await _dispatched_kwargs(eval_cell, orch, IMPLEMENTER, monkeypatch)

    scope = invocation_scope(orch, IMPLEMENTER)
    assert kwargs['strict_mcp_config'] is True
    assert kwargs['sandbox_modules'] == []
    assert worktree in kwargs['sandbox_extras']
    assert kwargs['cwd'] == eval_cell.worktree
    assert kwargs['strict_mcp_config'] == scope.strict_mcp_config
    assert kwargs['sandbox_modules'] == scope.sandbox_modules

    argv = await _spawned_argv(kwargs, monkeypatch)

    assert argv[1].endswith('landlock_exec.py')
    assert ('--writable', worktree) in zip(argv, argv[1:], strict=False)
    assert '--strict-mcp-config' in argv[argv.index('--') + 1:]


@pytest.mark.asyncio
@pytest.mark.usefixtures('landlock_host')
async def test_corpus_eval_implementer_stays_unsandboxed(
    eval_cell: _EvalCell, pinned_claude: str, monkeypatch,
) -> None:
    orch = build_eval_orch_config(_IMPL, eval_cell.fixture, eval_cell.base_config())

    kwargs = await _dispatched_kwargs(eval_cell, orch, IMPLEMENTER, monkeypatch)

    assert kwargs['sandbox_modules'] is None

    argv = await _spawned_argv(kwargs, monkeypatch)

    assert argv[0] == pinned_claude
    assert not any(token.endswith('landlock_exec.py') for token in argv)


@pytest.mark.asyncio
@pytest.mark.usefixtures('landlock_host')
async def test_shadow_architect_is_strict_and_unsandboxed(
    eval_cell: _EvalCell, monkeypatch,
) -> None:
    orch = build_shadow_orch_config(_IMPL, eval_cell.fixture, eval_cell.base_config())

    kwargs = await _dispatched_kwargs(eval_cell, orch, ARCHITECT, monkeypatch)

    assert kwargs['strict_mcp_config'] is True
    assert kwargs['sandbox_modules'] is None
