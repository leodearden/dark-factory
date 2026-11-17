"""What every reconciliation CLI call loads, and what it is denied (task 10492).

Every recon CLI call (the stages, the judge and the verify loop) reads no
settings file, loads no CLAUDE.md and no auto-memory index, keeps
auto-compaction off, and keeps its transcript. Each switch has one job:

- Empty setting sources keep out the settings files: hooks, permission rules,
  settings env and ``.mcp.json`` approvals.
- ``CLAUDE_CODE_DISABLE_CLAUDE_MDS`` keeps out every CLAUDE.md, including the
  ancestor-directory walk, which ``--setting-sources ''`` does not cover for the
  user file reached through /home/leo.
- ``CLAUDE_CODE_DISABLE_AUTO_MEMORY`` keeps out MEMORY.md, which
  ``--setting-sources ''`` was measured not to stop.
- ``DISABLE_AUTO_COMPACT`` keeps compaction off, as ``autoCompactEnabled`` in
  ~/.claude/settings.json did before the settings files were dropped.
- ``CLAUDE_CODE_FORCE_SESSION_PERSISTENCE`` keeps transcripts if a child-session
  marker is ever inherited.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _git_root_helper import make_git_root
from shared.cli_invoke import AgentResult
from shared.testing import make_gate_mock

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import ReconciliationRun, RunStatus, RunType
from fused_memory.reconciliation import (
    RECONCILIATION_CLI_ENV_OVERRIDES,
    RECONCILIATION_CLI_SETTING_SOURCES,
)
from fused_memory.reconciliation.cli_stage_runner import (
    DISALLOW_BUILTIN,
    STAGE1_DISALLOWED,
    STAGE2_DISALLOWED,
    STAGE3_DISALLOWED,
)
from fused_memory.reconciliation.judge import Judge
from fused_memory.reconciliation.verify import CodebaseVerifier
from reconciliation.stubbed_stage_cli import run_stage, run_stage_answered_by

_STAGE_REPORT = AgentResult(success=True, output='{"summary": "s"}')

_BUILTINS_NO_STAGE_CALLS = (
    'Agent', 'ListAgents', 'ReportFindings', 'ScheduleWakeup', 'Skill', 'Workflow',
    'CronCreate', 'CronDelete', 'CronList', 'DesignSync', 'EnterWorktree', 'ExitWorktree',
    'Monitor', 'PushNotification', 'RemoteTrigger', 'SendMessage', 'TaskStop', 'WebFetch',
    'WebSearch',
)
_WRITE_BUILTINS = ('Bash', 'Edit', 'Write', 'NotebookEdit')
_BUILTINS_STAGES_NEED = (
    'Read', 'Grep', 'Glob', 'ToolSearch', 'StructuredOutput',
    'ListMcpResourcesTool', 'ReadMcpResourceDirTool', 'ReadMcpResourceTool',
)


def test_recon_cli_context_constants_name_one_rule():
    assert list(RECONCILIATION_CLI_SETTING_SOURCES) == []
    assert dict(RECONCILIATION_CLI_ENV_OVERRIDES) == {
        'CLAUDE_CODE_DISABLE_CLAUDE_MDS': '1',
        'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '1',
        'DISABLE_AUTO_COMPACT': '1',
        'CLAUDE_CODE_FORCE_SESSION_PERSISTENCE': '1',
    }
    assert not hasattr(RECONCILIATION_CLI_ENV_OVERRIDES, '__setitem__')


@pytest.mark.asyncio
async def test_stage_runner_loads_no_settings_claude_md_or_memory_index(tmp_path):
    _, invoke = await run_stage_answered_by(_STAGE_REPORT, cwd=tmp_path)

    kwargs = invoke.call_args.kwargs
    assert kwargs['setting_sources'] == []
    assert kwargs['env_overrides'] == dict(RECONCILIATION_CLI_ENV_OVERRIDES)
    assert isinstance(kwargs['env_overrides'], dict)
    assert kwargs['strict_mcp_config'] is True
    assert kwargs.get('available_tools') is None
    assert kwargs['cwd'] == tmp_path
    assert kwargs['permission_mode'] == 'bypassPermissions'


@pytest.mark.asyncio
async def test_stage_runner_context_kwargs_survive_forwarding_to_invoke_claude_agent(tmp_path):
    """``usage_gate=None`` runs invoke_with_cap_retry's real forwarding layer, and
    the autospec'd mock rejects any keyword the real invoke_claude_agent lacks."""
    with patch(
        'shared.cli_invoke.invoke_claude_agent', autospec=True, return_value=_STAGE_REPORT,
    ) as mock_agent:
        await run_stage(ReconciliationConfig(sandbox_recon_agents=False), tmp_path)

    mock_agent.assert_called_once()
    kwargs = mock_agent.call_args.kwargs
    assert kwargs['setting_sources'] == []
    assert kwargs['env_overrides'] == dict(RECONCILIATION_CLI_ENV_OVERRIDES)
    assert kwargs['strict_mcp_config'] is True


def test_stages_are_denied_the_builtins_they_never_call():
    assert set(_BUILTINS_NO_STAGE_CALLS + _WRITE_BUILTINS) <= set(DISALLOW_BUILTIN)
    for stage_disallowed in (STAGE1_DISALLOWED, STAGE2_DISALLOWED, STAGE3_DISALLOWED):
        assert set(_BUILTINS_NO_STAGE_CALLS) <= set(stage_disallowed)
    assert not set(_BUILTINS_STAGES_NEED) & set(DISALLOW_BUILTIN)


def _journal_with_one_completed_run(run_id: str) -> MagicMock:
    journal = MagicMock()
    journal.get_run = AsyncMock(return_value=ReconciliationRun(
        id=run_id,
        project_id='p',
        run_type=RunType.full,
        trigger_reason='t',
        started_at=datetime.now(UTC),
        status=RunStatus.completed,
    ))
    journal.get_run_actions_combined = AsyncMock(return_value=[])
    journal.get_recent_verdicts = AsyncMock(return_value=[])
    journal.add_verdict = AsyncMock(return_value=None)
    journal.record_judge_invocation = AsyncMock(return_value=None)
    return journal


@pytest.mark.asyncio
async def test_judge_loads_no_settings_claude_md_or_memory_index(tmp_path, monkeypatch):
    monkeypatch.setenv('CLAUDE_CONFIG_DIR', str(tmp_path / 'claude-config'))
    config = ReconciliationConfig(
        judge_llm_provider='claude_cli',
        judge_llm_model='claude-sonnet-4-5',
        explore_codebase_root=str(tmp_path),
    )
    judge = Judge(
        config=config,
        journal=_journal_with_one_completed_run('run-ctx'),
        usage_gate=make_gate_mock(),
    )

    with patch(
        'fused_memory.reconciliation.judge.invoke_with_cap_retry', new_callable=AsyncMock,
    ) as invoke:
        invoke.return_value = AgentResult(
            success=True, output='', structured_output={'severity': 'ok', 'findings': []},
        )
        await judge.review_run('run-ctx')

    invoke.assert_awaited_once()
    kwargs = invoke.call_args.kwargs
    assert kwargs['setting_sources'] == []
    assert kwargs['env_overrides'] == dict(RECONCILIATION_CLI_ENV_OVERRIDES)
    assert kwargs['cwd'] == tmp_path
    assert kwargs['strict_mcp_config'] is True


@pytest.mark.asyncio
async def test_verify_loop_loads_no_settings_claude_md_or_memory_index(tmp_path):
    git_root = make_git_root(tmp_path)

    with patch(
        'fused_memory.reconciliation.agent_loop.invoke_with_cap_retry', new_callable=AsyncMock,
    ) as invoke:
        invoke.return_value = AgentResult(
            success=True,
            output='',
            structured_output={
                'verdict': 'confirmed', 'confidence': 1.0, 'evidence': [], 'summary': 's',
            },
        )
        await CodebaseVerifier(ReconciliationConfig()).verify(
            claim='x.py defines x', codebase_root=git_root,
        )

    invoke.assert_awaited_once()
    kwargs = invoke.call_args.kwargs
    assert kwargs['setting_sources'] == []
    assert kwargs['env_overrides'] == dict(RECONCILIATION_CLI_ENV_OVERRIDES)
    assert kwargs['cwd'] == git_root.resolve()
    assert kwargs['permission_mode'] == 'dontAsk'
    assert kwargs['available_tools'] == ['Read', 'Grep', 'Glob']
