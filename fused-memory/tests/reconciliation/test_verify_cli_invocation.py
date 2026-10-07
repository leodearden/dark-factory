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
from shared.cli_invoke import AgentResult

from fused_memory.config.schema import ReconciliationConfig
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
