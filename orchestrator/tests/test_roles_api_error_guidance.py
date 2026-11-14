"""Anchor test for the steward's API_ERROR guidance (task 3319, PRD
``plans/server-side-api-error-handling-prd.md`` task ε).

``classify_agent_failure`` files both a provider 5xx (including a watchdog-
killed run whose flushed result carried a 529) and a request- or account-side
4xx under the one kind ``AgentFailureKind.API_ERROR``, so the steward's single
entry for that kind must tell the two apart: failover helps neither a provider
outage nor a deterministic 4xx.  Assertions follow the ``test_roles_*.py``
family rule stated in ``test_roles_merge_halt_safety.py``'s module docstring.
"""

from __future__ import annotations

import pytest
from shared.cli_invoke import AgentFailureKind, classify_agent_failure

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import STEWARD, STEWARD_API_ERROR_GUIDANCE

_KIND_LABEL = f'**{AgentFailureKind.API_ERROR.name}**'


def test_api_error_guidance_reaches_the_steward() -> None:
    assert STEWARD_API_ERROR_GUIDANCE in STEWARD.system_prompt


def test_guidance_is_the_steward_prompts_only_api_error_entry() -> None:
    assert STEWARD.system_prompt.count(_KIND_LABEL) == 1
    assert _KIND_LABEL in STEWARD_API_ERROR_GUIDANCE


@pytest.mark.parametrize(
    'result',
    [
        pytest.param(
            AgentResult(
                success=False,
                output='',
                timed_out=True,
                turns=0,
                cost_usd=0.0,
                transcript_turns=0,
                api_error_status=529,
            ),
            id='sigterm-flushed-5xx',
        ),
        pytest.param(
            AgentResult(
                success=False,
                output='',
                timed_out=False,
                api_error_status=400,
            ),
            id='non-timed-out-4xx',
        ),
    ],
)
def test_server_and_client_errors_both_reach_this_entry(result: AgentResult) -> None:
    assert classify_agent_failure(result).kind is AgentFailureKind.API_ERROR
    assert AgentFailureKind.API_ERROR.name in STEWARD_API_ERROR_GUIDANCE
