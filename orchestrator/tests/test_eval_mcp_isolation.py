"""Isolation tests for the MCP servers an eval agent can reach (task 4757).

The companion of ``test_eval_memory_isolation.py``. That module pins the
HARNESS's own fused-memory client off production; this one pins the AGENT's.

Every eval worktree is a checkout of the target repo, so it carries the
tracked ``.mcp.json`` at the agent's cwd, and that file declares the live
production ``escalation`` (http://127.0.0.1:8102/mcp) and ``fused-memory``
(http://127.0.0.1:8002/mcp) servers. Unless an invocation passes
``--strict-mcp-config``, the Claude CLI merges that file into whatever
``--mcp-config`` the caller supplied, and the eval agent can write into the
live memory store, escalation queue and task DB. The fable-trial-v2 δ tranche-1
cells did exactly that (``plans/fable-architect-trial-v2-decision-2026-08-30.md``
§"A side-finding"): 36 ``add_memory`` calls, 7 ``escalate_info`` filings and
3 ``submit_task`` calls into production, none of them eval-tagged.

These tests are hermetic (no LLM, no network). They assert at the argv
boundary: what each eval spawn site hands ``invoke_agent``, and what the real
``build_claude_argv`` turns that into.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _git_fixtures import seed_repo
from _workflow_helpers import FakeBriefing, FakeScheduler, fixed_prompt
from shared.cli_invoke import AgentResult, build_claude_argv

from orchestrator.agents.roles import ARCHITECT, AgentRole
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.evals import runner
from orchestrator.evals.architect_eval import ArchitectEvalDeps
from orchestrator.evals.compare import Synthesis, _assess_task, compare_models
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.judge import judge_plan_quality, run_judge
from orchestrator.evals.results import EvalResult, ResultStore
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow

_FIXTURE_ID = 'df_task_9001'


@pytest.fixture
def pinned_claude(monkeypatch, tmp_path: Path) -> str:
    """Pin ``CLAUDE_BINARY`` at a real executable so argv assembly is deterministic."""
    exec_file = tmp_path / 'claude'
    exec_file.write_text('#!/bin/sh\nexit 0\n')
    exec_file.chmod(0o755)
    monkeypatch.setenv('CLAUDE_BINARY', str(exec_file))
    return str(exec_file)


def _emitted_mcp_scope(mcp_config: dict | None, strict_mcp_config: bool) -> tuple[list[str], set[str]]:
    """Run the REAL argv builder; return the argv tail from ``--mcp-config`` and the emitted server names."""
    cmd, temp_files = build_claude_argv(
        model='opus',
        max_budget_usd=1.0,
        system_prompt='s',
        max_turns=1,
        permission_mode='bypassPermissions',
        allowed_tools=None,
        disallowed_tools=None,
        mcp_config=mcp_config,
        output_schema=None,
        effort=None,
        resume_session_id=None,
        session_id=None,
        strict_mcp_config=strict_mcp_config,
    )
    try:
        assert '--mcp-config' in cmd, f'no --mcp-config emitted: {cmd!r}'
        at = cmd.index('--mcp-config')
        servers = set(json.loads(Path(cmd[at + 1]).read_text())['mcpServers'])
        return cmd[at:at + 3], servers
    finally:
        for path in temp_files:
            Path(path).unlink(missing_ok=True)


def _write_fixture(tmp_path: Path) -> Path:
    project_root = tmp_path / 'proj'
    project_root.mkdir()
    task_path = tmp_path / f'{_FIXTURE_ID}.json'
    task_path.write_text(json.dumps({
        'id': _FIXTURE_ID,
        'name': 'mcp isolation probe',
        'project_root': str(project_root),
        'pre_task_commit': 'a' * 40,
        'reference': {'post_task_commit': 'b' * 40},
        'task_definition': {'title': 'Probe', 'description': 'probe the architect argv'},
        'timeout_minutes': 1,
    }))
    return task_path


async def _drive_architect_eval(tmp_path: Path) -> dict:
    """Run the REAL ``run_architect_eval`` up to its invoke; return the kwargs its injected invoke received."""
    worktree = tmp_path / 'wt'
    worktree.mkdir()
    mock_invoke = AsyncMock(return_value=AgentResult(success=True, output=''))
    deps = ArchitectEvalDeps(
        store=MagicMock(spec=ResultStore),
        create_worktree=AsyncMock(return_value=(worktree, 'run-abc')),
        cleanup_worktree=AsyncMock(),
        build_prompt=AsyncMock(return_value='P'),
        invoke=mock_invoke,
        read_reference_diff=AsyncMock(return_value=''),
    )

    cfg = EvalConfig(name='architect-probe', backend='claude', model='opus',
                     effort='high', role='architect')
    await runner.run_architect_eval(
        _write_fixture(tmp_path), cfg,
        base_config=OrchestratorConfig(project_root=tmp_path / 'proj'),
        usage_gate=None,
        deps=deps,
    )
    assert mock_invoke.await_args is not None, 'the architect was never invoked'
    return dict(mock_invoke.await_args.kwargs)


@pytest.mark.asyncio
async def test_architect_eval_invoke_is_strictly_scoped_to_plan_tools(
    tmp_path: Path, pinned_claude: str,
) -> None:
    """The eval architect reaches plan-tools and nothing else, through the real cap-retry forwarding."""
    kw = await _drive_architect_eval(tmp_path)

    assert kw.get('strict_mcp_config') is True
    assert set(kw['mcp_config']['mcpServers']) == {'plan-tools'}
    assert kw['mcp_config']['mcpServers']['plan-tools']

    tail, servers = _emitted_mcp_scope(kw['mcp_config'], kw['strict_mcp_config'])
    assert tail[0] == '--mcp-config'
    assert tail[2] == '--strict-mcp-config', f'flag does not follow the pair: {tail!r}'
    assert servers == {'plan-tools'}


# ---------------------------------------------------------------------------
# The /tmp-cwd eval LLM calls: two judges, two compare assessors. None needs
# any MCP server: each answers through its output schema.
# ---------------------------------------------------------------------------

_ASSESSMENT = {
    'winner': 'A', 'confidence': 0.9, 'summary': 's',
    'strengths_a': [], 'weaknesses_a': [], 'strengths_b': [], 'weaknesses_b': [],
    'circumstances': '',
}
_SYNTHESIS = {
    'overall_winner': 'A', 'executive_summary': 'e', 'circumstantial_analysis': 'c',
    'cost_quality_tradeoff': 't', 'workflow_recommendations': 'w',
}
_TASK = {
    'id': _FIXTURE_ID,
    'name': 'mcp isolation probe',
    'pre_task_commit': 'a' * 40,
    'task_definition': {'description': 'd'},
}


def _recording_invoke(structured_output: dict) -> tuple[list[dict], object]:
    """An ``invoke_agent`` double that records every call's kwargs and answers *structured_output*."""
    calls: list[dict] = []

    async def fake_invoke_agent(**kwargs: object) -> AgentResult:
        calls.append(kwargs)
        return AgentResult(success=True, output=json.dumps(structured_output),
                           structured_output=structured_output, cost_usd=0.0)

    return calls, fake_invoke_agent


def _assert_zero_mcp_strict(kwargs: dict) -> None:
    # Truthiness is load-bearing: --strict-mcp-config is emitted only inside `if mcp_config:`.
    assert kwargs.get('mcp_config') == {'mcpServers': {}}
    assert bool(kwargs['mcp_config']) is True
    assert kwargs.get('strict_mcp_config') is True


def _eval_result(config_name: str, worktree: Path) -> EvalResult:
    worktree.mkdir()
    return EvalResult(task_id=_FIXTURE_ID, config_name=config_name, outcome='done',
                      metrics={}, worktree_path=str(worktree))


@pytest.mark.asyncio
async def test_pairwise_judge_carries_no_mcp_servers(monkeypatch) -> None:
    calls, fake = _recording_invoke({'winner': 'A', 'confidence': 0.9, 'reasoning': 'r'})
    monkeypatch.setattr('orchestrator.evals.judge.invoke_agent', fake)

    await run_judge({'diff': '+a', 'config_name': 'A'}, {'diff': '+b', 'config_name': 'B'}, _TASK)

    assert len(calls) == 1
    _assert_zero_mcp_strict(calls[0])


@pytest.mark.asyncio
async def test_plan_quality_judge_carries_no_mcp_servers(monkeypatch) -> None:
    calls, fake = _recording_invoke({'plan_quality': 0.5, 'reasoning': 'r'})
    monkeypatch.setattr('orchestrator.evals.judge.invoke_agent', fake)
    plan = {'steps': [{'id': 'step-1', 'type': 'test', 'description': 'write the test'}]}

    await judge_plan_quality(plan, '+landed', _TASK)

    assert len(calls) == 1
    _assert_zero_mcp_strict(calls[0])


@pytest.mark.asyncio
async def test_compare_assessor_carries_no_mcp_servers(monkeypatch, tmp_path: Path) -> None:
    calls, fake = _recording_invoke(_ASSESSMENT)
    monkeypatch.setattr('orchestrator.evals.compare.invoke_agent', fake)
    monkeypatch.setattr('orchestrator.evals.compare.get_diff', AsyncMock(return_value='+x'))

    await _assess_task(_TASK, _eval_result('A', tmp_path / 'wt_a'),
                       _eval_result('B', tmp_path / 'wt_b'), 'A', 'B')

    assert len(calls) == 1
    _assert_zero_mcp_strict(calls[0])


@pytest.mark.asyncio
async def test_compare_synthesizer_carries_no_mcp_servers(monkeypatch, tmp_path: Path) -> None:
    calls, fake = _recording_invoke({**_ASSESSMENT, **_SYNTHESIS})
    monkeypatch.setattr('orchestrator.evals.compare.invoke_agent', fake)
    monkeypatch.setattr('orchestrator.evals.compare.get_diff', AsyncMock(return_value='+x'))

    report = await compare_models(
        [_eval_result('A', tmp_path / 'wt_a'), _eval_result('B', tmp_path / 'wt_b')],
        ['A'], ['B'], {_FIXTURE_ID: _TASK},
    )

    assert isinstance(report.synthesis, Synthesis), 'the synthesizer was never reached'
    assert report.synthesis.executive_summary == 'e'
    assert len(calls) == 2
    for call in calls:
        _assert_zero_mcp_strict(call)


# ---------------------------------------------------------------------------
# TaskWorkflow._invoke: OrchestratorConfig.strict_mcp_config (off in
# production, on under EVAL_PROFILE: boundary test B10) is forwarded, and never
# swallowed by a role whose families compose no config.
# ---------------------------------------------------------------------------


class _FalsyMcpStub:
    """The worst case the guard must handle: ``runner._EvalMcpStub`` answers a FALSY bare ``{}``."""

    url = 'http://127.0.0.1:1'

    def mcp_config_json(self, escalation_url: str | None = None) -> dict:
        return {}


_ORCH_ONLY = AgentRole(name='probe_orch_only', system_prompt='x', allowed_tools=[],
                       mcp_families=frozenset({'orchestrator'}))
_NO_MCP = AgentRole(name='probe_no_mcp', system_prompt='x', allowed_tools=[],
                    mcp_families=frozenset())


async def _invoke_probe(tmp_path: Path, role: AgentRole, *, strict: bool) -> dict:
    """Drive the REAL ``TaskWorkflow._invoke`` once; return the kwargs ``invoke_with_cap_retry`` received."""
    repo = seed_repo(tmp_path / 'repo')

    config = OrchestratorConfig(
        project_root=repo, max_concurrent_tasks=1, strict_mcp_config=strict,
        git=GitConfig(main_branch='main', branch_prefix='task/', remote='origin',
                      worktree_dir='.worktrees'),
    )
    git_ops = GitOps(config.git, config.project_root)
    assignment = TaskAssignment(
        task_id='42',
        task={'id': '42', 'title': 'X', 'description': 'Y', 'status': 'pending',
              'metadata': {'files': ['lib']}, 'dependencies': []},
        modules=['lib'],
    )
    cwd = (await git_ops.create_worktree(assignment.task_id)).path
    workflow = TaskWorkflow(
        assignment=assignment, config=config, git_ops=git_ops,
        scheduler=FakeScheduler(),
        briefing=FakeBriefing(),
        mcp=_FalsyMcpStub(),
    )
    workflow.artifacts = None

    with patch('orchestrator.workflow.invoke_with_cap_retry', new_callable=AsyncMock,
               return_value=AgentResult(success=True, output='')) as mock_cap_retry:
        await workflow._invoke(role, fixed_prompt('P'), cwd)

    assert mock_cap_retry.await_args is not None, '_invoke never reached invoke_with_cap_retry'
    return dict(mock_cap_retry.await_args.kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(('role', 'expected_servers'), [
    (ARCHITECT, {'plan-tools'}),
    (_ORCH_ONLY, set()),
    (_NO_MCP, set()),
], ids=['plan-tools-composed', 'falsy-orchestrator-config', 'no-families'])
async def test_invoke_strict_config_scopes_every_role(
    tmp_path: Path, role: AgentRole, expected_servers: set[str],
) -> None:
    kw = await _invoke_probe(tmp_path, role, strict=True)

    assert kw.get('strict_mcp_config') is True
    assert bool(kw['mcp_config']) is True, 'a falsy config would make --strict-mcp-config a no-op'
    servers = set(kw['mcp_config']['mcpServers'])
    assert servers.isdisjoint({'escalation', 'fused-memory'})
    assert servers == expected_servers


@pytest.mark.asyncio
async def test_invoke_production_default_keeps_the_composed_plan_tools_config(tmp_path: Path) -> None:
    kw = await _invoke_probe(tmp_path, ARCHITECT, strict=False)

    assert kw.get('strict_mcp_config') is False
    assert kw['mcp_config'].keys() == {'mcpServers'}
    assert kw['mcp_config']['mcpServers'].keys() == {'plan-tools'}


@pytest.mark.asyncio
@pytest.mark.parametrize(('role', 'composed'), [
    (_ORCH_ONLY, {}),
    (_NO_MCP, None),
], ids=['falsy-orchestrator-config', 'no-families'])
async def test_invoke_production_default_passes_a_falsy_config_through(
    tmp_path: Path, role: AgentRole, composed: dict | None,
) -> None:
    kw = await _invoke_probe(tmp_path, role, strict=False)

    assert kw.get('strict_mcp_config') is False
    assert kw['mcp_config'] == composed
