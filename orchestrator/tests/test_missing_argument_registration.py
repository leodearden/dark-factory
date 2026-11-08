"""The orchestrator's fastmcp servers refuse a missing required argument structurally.

Pins that plan-tools and verdict-tools register
``shared.mcp_missing_arguments.MissingArgumentMiddleware``. Every behavioural
call goes through ``fastmcp.Client``, because ``tool.fn`` / ``tool.run``
bypass middleware. The escalation server's counterpart is
``escalation/tests/test_missing_argument_registration.py``.
"""
from __future__ import annotations

import json
from typing import Any

import pytest
from _markup_helpers import closer, param_opener
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from shared.mcp_missing_arguments import MISSING_ARGUMENT_CODE

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import plan_tools, verdict_tools


@pytest.fixture
def artifacts(tmp_path):
    a = TaskArtifacts(tmp_path)
    a.init('test-1', 'Test task', 'A test')
    return a


async def refusal(server: FastMCP, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    with pytest.raises(ToolError) as excinfo:
        async with Client(server) as client:
            await client.call_tool(tool, arguments)
    message = str(excinfo.value)
    try:
        return json.loads(message)
    except json.JSONDecodeError:
        pytest.fail(f'{tool} was refused with unstructured text: {message}')


# --- plan-tools ---


@pytest.mark.asyncio
async def test_create_plan_without_analysis_is_refused_structurally(artifacts):
    payload = await refusal(
        plan_tools.create_server(artifacts),
        'create_plan',
        {'task_id': 'test-1', 'title': 'T', 'files': ['x.py']},
    )

    assert payload['code'] == MISSING_ARGUMENT_CODE
    assert payload['tool'] == 'create_plan'
    assert [m['name'] for m in payload['missing']] == ['analysis']
    assert artifacts.read_plan() == {}


@pytest.mark.asyncio
async def test_create_plan_without_files_keeps_its_in_body_answer(artifacts):
    async with Client(plan_tools.create_server(artifacts)) as client:
        result = await client.call_tool(
            'create_plan', {'task_id': 'test-1', 'title': 'T', 'analysis': 'A'}
        )

    assert result.structured_content is not None
    assert result.structured_content['status'] == 'error'
    assert artifacts.read_plan() == {}


# --- verdict-tools ---

REVIEWER = 'test_analyst'

ROLE_CALLS: dict[str, tuple[str, dict[str, Any], str]] = {
    'judge': (
        'submit_completion_verdict',
        {'complete': True, 'reasoning': 'r', 'uncovered_plan_steps': []},
        'substantive_work',
    ),
    'triage': (
        'submit_triage',
        {'accepted': [], 'skipped': []},
        'proposed_task_groups',
    ),
    'merger': (
        'submit_merge_disposition',
        {'blocked': False},
        'reason',
    ),
    REVIEWER: (
        'submit_review_verdict',
        {'reviewer': REVIEWER, 'verdict': 'PASS', 'summary': 's'},
        'issues',
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize('role', list(ROLE_CALLS))
async def test_a_missing_argument_is_refused_structurally_on_every_branch(artifacts, role):
    tool, arguments, omitted = ROLE_CALLS[role]

    payload = await refusal(verdict_tools.create_server(artifacts, role), tool, arguments)

    assert payload['code'] == MISSING_ARGUMENT_CODE
    assert payload['tool'] == tool
    assert [m['name'] for m in payload['missing']] == [omitted]
    assert artifacts.read_verdict(role) is None


@pytest.mark.asyncio
async def test_a_partially_repaired_leak_is_refused_for_what_is_still_missing(artifacts):
    leaked_summary = (
        'The change is covered.'
        + closer('parameter')
        + '\n'
        + param_opener('issues')
        + '[{"severity": "low", "suggested_fix": "add a test"}]'
    )

    payload = await refusal(
        verdict_tools.create_server(artifacts, REVIEWER),
        'submit_review_verdict',
        {'reviewer': REVIEWER, 'summary': leaked_summary},
    )

    assert payload['code'] == MISSING_ARGUMENT_CODE
    assert [m['name'] for m in payload['missing']] == ['verdict']
    assert 'issues' in payload['provided']
    assert artifacts.read_verdict(REVIEWER) is None
