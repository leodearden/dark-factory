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
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from shared.mcp_missing_arguments import MISSING_ARGUMENT_CODE, MissingArgumentMiddleware

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import plan_tools


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


def guards_on(server: FastMCP) -> list[MissingArgumentMiddleware]:
    return [m for m in server.middleware if isinstance(m, MissingArgumentMiddleware)]


# --- plan-tools ---


def test_plan_tools_registers_the_guard_exactly_once(artifacts):
    assert len(guards_on(plan_tools.create_server(artifacts))) == 1


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
