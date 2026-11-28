"""The verify -> CLI invocation contract (task 4344).

On the claude_cli provider ``CodebaseVerifier.verify`` reaches the model
through exactly one CLI invocation: the CLI runs its own read-only built-in
tools and returns the verdict through ``--json-schema``.  These tests drive the
REAL verifier and patch only the CLI seam,
``fused_memory.reconciliation.agent_loop.invoke_with_cap_retry``.
"""

from unittest.mock import AsyncMock, patch

import pytest
from _git_root_helper import make_git_root
from _invocation_ledger import invocation_rows
from _invocation_tokens import LEDGER_TOKENS, agent_result_with_tokens
from shared.cli_invoke import AgentResult
from shared.cost_store import CostStore
from shared.testing import make_gate_mock

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.invocation_ledger import InvocationLedger, LedgerRole
from fused_memory.reconciliation.verify import CodebaseVerifier


@pytest.fixture
def git_root(tmp_path):
    return make_git_root(tmp_path)


@pytest.mark.asyncio
async def test_a_cli_verdict_becomes_the_verification_result(git_root):
    verdict = {
        'verdict': 'confirmed',
        'confidence': 0.9,
        'evidence': [
            {'file_path': 'a.py', 'line_range': '1-2', 'snippet': 'x', 'relevance': 'r'},
        ],
        'summary': 'a.py defines x',
    }
    with patch(
        'fused_memory.reconciliation.agent_loop.invoke_with_cap_retry',
        new_callable=AsyncMock,
    ) as mock_invoke:
        mock_invoke.return_value = AgentResult(
            success=True, output='', structured_output=verdict,
        )
        result = await CodebaseVerifier(ReconciliationConfig()).verify(
            claim='a.py defines x', codebase_root=git_root,
        )

    mock_invoke.assert_called_once()
    assert result.verdict == 'confirmed'
    assert result.confidence == 0.9
    assert result.evidence == verdict['evidence']
    assert result.summary == 'a.py defines x'
    assert result.agent_failed is False
    assert result.failure_token == ''


_IN_PROCESS_TOOL_NAMES = (
    'read_file', 'glob_search', 'grep_search', 'git_log', 'git_show',
    'verification_complete',
)


async def _verify_cli_kwargs(git_root) -> dict:
    """Run the REAL verifier once; return the kwargs of its CLI invocation."""
    with patch(
        'fused_memory.reconciliation.agent_loop.invoke_with_cap_retry',
        new_callable=AsyncMock,
    ) as mock_invoke:
        mock_invoke.return_value = AgentResult(
            success=True,
            output='',
            structured_output={
                'verdict': 'confirmed', 'confidence': 1.0, 'evidence': [], 'summary': 's',
            },
        )
        await CodebaseVerifier(ReconciliationConfig()).verify(
            claim='x.py defines x',
            context='task context',
            scope_hints=['x.py'],
            codebase_root=git_root,
        )
    mock_invoke.assert_called_once()
    return dict(mock_invoke.call_args.kwargs)


@pytest.mark.asyncio
async def test_the_cli_never_sees_an_in_process_tool_name(git_root):
    """Every name the CLI sees in a prompt is one the model may call natively,
    and the CLI rejects each such call.  The measured native-call rates are in
    fused-memory/scripts/probe_schema_max_turns.py's docstring.
    """
    kwargs = await _verify_cli_kwargs(git_root)
    for name in _IN_PROCESS_TOOL_NAMES:
        assert name not in kwargs['system_prompt'], f'{name!r} in the system prompt'
        assert name not in kwargs['prompt'], f'{name!r} in the user prompt'


@pytest.mark.asyncio
async def test_the_cli_gets_read_grep_glob_and_the_verdict_schema(git_root):
    """The verdict schema's property set is also task 6022's pin: a required
    reasoning field got verify refused by the API's reasoning_extraction
    classifier, so no reasoning or 'thinking' property may be added without
    re-probing with fused-memory/scripts/probe_schema_max_turns.py.
    """
    kwargs = await _verify_cli_kwargs(git_root)
    assert kwargs['available_tools'] == ['Read', 'Grep', 'Glob']
    assert kwargs['permission_mode'] == 'dontAsk'

    schema = kwargs['output_schema']
    assert schema['required'] == ['verdict', 'confidence', 'evidence', 'summary']
    assert schema['properties']['verdict']['enum'] == ['confirmed', 'contradicted', 'inconclusive']
    assert set(schema['properties']) == {
        'verdict', 'confidence', 'evidence', 'summary', 'git_context',
    }
    assert 'explain your reasoning' not in kwargs['system_prompt']
    assert '"thinking"' not in kwargs['system_prompt']
    assert kwargs['cwd'] == git_root.resolve()


@pytest.mark.asyncio
async def test_no_ambient_settings_file_can_widen_the_read_confinement(git_root):
    """dontAsk confines Read/Grep/Glob to cwd only while NO allow rule applies,
    and the CLI otherwise merges allow rules from the user's settings and from
    the target project's own .claude/settings*.json.  The call therefore reads
    no settings file at all.  The measured widening is recorded in
    fused-memory/scripts/probe_schema_max_turns.py's docstring.
    """
    kwargs = await _verify_cli_kwargs(git_root)
    assert kwargs['setting_sources'] == []
    assert not kwargs.get('allowed_tools')


async def _verify_with_ledger(tmp_path, git_root, served: AgentResult):
    config = ReconciliationConfig(agent_llm_provider='claude_cli')
    db_path = tmp_path / 'curator_events.db'
    async with CostStore(db_path) as store:
        ledger = InvocationLedger(store, LedgerRole.RECON_VERIFY, 'run-v', 'proj-v')
        with patch('shared.cli_invoke.invoke_claude_agent', new=AsyncMock(return_value=served)):
            result = await CodebaseVerifier(config).verify(
                claim='a.py defines x', codebase_root=git_root, ledger=ledger,
            )
    return config, result, await invocation_rows(db_path)


@pytest.mark.asyncio
async def test_a_verify_lands_one_invocations_row(tmp_path, git_root):
    served = agent_result_with_tokens(
        success=True,
        structured_output={'verdict': 'confirmed', 'confidence': 0.9, 'evidence': [], 'summary': 's'},
        cost_usd=0.11,
    )

    config, result, rows = await _verify_with_ledger(tmp_path, git_root, served)

    assert result.verdict == 'confirmed'
    assert len(rows) == 1
    row = rows[0]
    assert row['role'] == 'recon_verify'
    assert row['run_id'] == 'run-v'
    assert row['project_id'] == 'proj-v'
    assert row['model'] == config.agent_llm_model
    assert {key: row[key] for key in LEDGER_TOKENS} == LEDGER_TOKENS


@pytest.mark.asyncio
async def test_an_audited_max_turns_failure_still_lands_a_row(tmp_path, git_root):
    exhausted = agent_result_with_tokens(success=False, subtype='error_max_turns')

    _config, result, rows = await _verify_with_ledger(tmp_path, git_root, exhausted)

    assert len(rows) == 1
    assert result.agent_failed is True


@pytest.mark.asyncio
async def test_a_verifier_given_a_usage_gate_dispatches_on_the_pooled_account_token(git_root):
    """Only invoke_claude_agent is patched, so invoke_with_cap_retry's real
    gated branch runs: the CLI must authenticate on the gate's pooled account,
    never on the ambient ~/.claude login (task 11783)."""
    gate = make_gate_mock(before_invoke=AsyncMock(return_value='pooled-token'))
    served = AgentResult(
        success=True,
        output='',
        structured_output={'verdict': 'confirmed', 'confidence': 1.0, 'evidence': [], 'summary': 's'},
    )
    with patch('shared.cli_invoke.invoke_claude_agent', autospec=True, return_value=served) as mock_agent:
        result = await CodebaseVerifier(ReconciliationConfig(), usage_gate=gate).verify(
            claim='x.py defines x', codebase_root=git_root,
        )

    mock_agent.assert_called_once()
    assert mock_agent.call_args.kwargs['oauth_token'] == 'pooled-token'
    gate.confirm_account_ok.assert_called_once_with('pooled-token')
    assert result.verdict == 'confirmed'
    assert result.agent_failed is False
