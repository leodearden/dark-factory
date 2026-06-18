"""RED/GREEN tests for CodebaseVerifier.verify() agent-failure handling (site 22).

Step-1 (RED): Failure-path tests confirm that when AgentLoop.run() returns a
warning/no-verdict shape, verify() emits a WARNING and sets
summary='agent-failed:<token>' rather than silently laundering the failure as
an empty-summary inconclusive.

Step-2 (GREEN): Implement the fix in verify.py (extract_agent_verdict).
The success-path test guards behavior-preservation across the refactor.
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.reconciliation.verify import CodebaseVerifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_config() -> ReconciliationConfig:
    """ReconciliationConfig with all defaults — suffices for unit tests."""
    return ReconciliationConfig()


# ---------------------------------------------------------------------------
# FAILURE-PATH tests (RED in step-1, GREEN after step-2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    'agent_return',
    [
        ({'warning': 'no_tool_calls', 'text': 'gave up'}, []),
        ({'warning': 'max_steps_reached'}, []),
    ],
    ids=['no_tool_calls', 'max_steps_reached'],
)
async def test_verify_agent_failure_emits_warning_and_sentinel_summary(
    agent_return, caplog
):
    """When AgentLoop.run() returns a warning shape, verify() must:

    - return verdict='inconclusive' (VerificationVerdict constraint)
    - return summary='agent-failed:<token>' (NOT empty)
    - return confidence=0.0, evidence=[], git_context=None
    - emit at least one WARNING record containing the failure token
    """
    result_dict, journal = agent_return
    expected_token = result_dict['warning']

    with patch('fused_memory.reconciliation.verify.AgentLoop') as MockAgentLoop:
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run = AsyncMock(return_value=(result_dict, journal))
        MockAgentLoop.return_value = mock_agent_instance

        verifier = CodebaseVerifier(_default_config())

        with caplog.at_level(logging.WARNING):
            result = await verifier.verify(claim='Task X completed')

    assert result.verdict == 'inconclusive', (
        f'Expected inconclusive but got {result.verdict!r}'
    )
    assert result.summary == f'agent-failed:{expected_token}', (
        f'Expected agent-failed sentinel but got {result.summary!r}'
    )
    assert result.confidence == 0.0
    assert result.evidence == []
    assert result.git_context is None

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, 'Expected at least one WARNING record, got none'
    assert any(expected_token in r.message for r in warning_records), (
        f'Expected a WARNING message containing {expected_token!r}, '
        f'got: {[r.message for r in warning_records]}'
    )


# ---------------------------------------------------------------------------
# SUCCESS-PATH test (written now; must stay green before AND after step-2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_success_path_preserved(caplog):
    """When AgentLoop.run() returns a terminal verification_complete dict,
    verify() must preserve all fields and emit NO WARNING.

    This characterizes current behavior and guards against regressions in
    the step-2 refactor.
    """
    terminal_result = {
        'verdict': 'confirmed',
        'confidence': 0.9,
        'evidence': [{'file_path': 'a.py', 'line_range': '1-10', 'snippet': 'x = 1', 'relevance': 'direct'}],
        'summary': 'looks good',
        'git_context': {'author': 'x', 'latest_relevant_commit': 'abc123'},
    }
    journal: list = []

    with patch('fused_memory.reconciliation.verify.AgentLoop') as MockAgentLoop:
        mock_agent_instance = AsyncMock()
        mock_agent_instance.run = AsyncMock(return_value=(terminal_result, journal))
        MockAgentLoop.return_value = mock_agent_instance

        verifier = CodebaseVerifier(_default_config())

        with caplog.at_level(logging.WARNING):
            result = await verifier.verify(claim='Feature X was shipped')

    assert result.verdict == 'confirmed'
    assert result.summary == 'looks good'
    assert result.confidence == 0.9
    assert result.evidence == [{'file_path': 'a.py', 'line_range': '1-10', 'snippet': 'x = 1', 'relevance': 'direct'}]
    assert result.git_context == {'author': 'x', 'latest_relevant_commit': 'abc123'}

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not warning_records, (
        f'Expected no WARNING records on success path, got: {[r.message for r in warning_records]}'
    )
