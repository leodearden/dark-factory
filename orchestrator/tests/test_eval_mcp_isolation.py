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
from unittest.mock import AsyncMock, MagicMock

import pytest
from shared.cli_invoke import AgentResult, build_claude_argv

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.config import OrchestratorConfig
from orchestrator.evals import runner
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.results import ResultStore

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


async def _drive_architect_eval(monkeypatch, tmp_path: Path) -> dict:
    """Run the REAL ``run_architect_eval`` up to its invoke; return the kwargs ``invoke_agent`` received."""
    from orchestrator.evals import snapshots

    worktree = tmp_path / 'wt'
    worktree.mkdir()
    monkeypatch.setattr(snapshots, 'create_eval_worktree',
                        AsyncMock(return_value=(worktree, 'run-abc')))
    monkeypatch.setattr(snapshots, 'cleanup_eval_worktree', AsyncMock())
    monkeypatch.setattr(snapshots, 'get_diff_between_commits', AsyncMock(return_value=''))
    monkeypatch.setattr(BriefingAssembler, 'build_architect_prompt',
                        AsyncMock(return_value='P'))
    mock_invoke = AsyncMock(return_value=AgentResult(success=True, output=''))
    monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', mock_invoke)

    cfg = EvalConfig(name='architect-probe', backend='claude', model='opus',
                     effort='high', role='architect')
    await runner.run_architect_eval(
        _write_fixture(tmp_path), cfg,
        base_config=OrchestratorConfig(project_root=tmp_path / 'proj'),
        usage_gate=None,
        store=MagicMock(spec=ResultStore),
    )
    assert mock_invoke.await_args is not None, 'the architect was never invoked'
    return mock_invoke.await_args.kwargs


@pytest.mark.asyncio
async def test_architect_eval_invoke_is_strictly_scoped_to_plan_tools(
    monkeypatch, tmp_path: Path, pinned_claude: str,
) -> None:
    """The eval architect reaches plan-tools and nothing else, through the real cap-retry forwarding."""
    kw = await _drive_architect_eval(monkeypatch, tmp_path)

    assert kw.get('strict_mcp_config') is True
    assert set(kw['mcp_config']['mcpServers']) == {'plan-tools'}
    assert kw['mcp_config']['mcpServers']['plan-tools']

    tail, servers = _emitted_mcp_scope(kw['mcp_config'], kw['strict_mcp_config'])
    assert tail[0] == '--mcp-config'
    assert tail[2] == '--strict-mcp-config', f'flag does not follow the pair: {tail!r}'
    assert servers == {'plan-tools'}
