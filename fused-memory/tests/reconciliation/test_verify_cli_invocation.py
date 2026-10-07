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
