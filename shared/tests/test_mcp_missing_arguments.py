"""Tests for ``shared.mcp_missing_arguments`` — the missing-argument refusal.

Every call goes through ``fastmcp.Client``: ``tool.fn`` / ``tool.run`` bypass
middleware (``shared/src/shared/mcp_markup_middleware.py`` substrate fact 3),
so a test written that way would pass without running the guard at all.
"""
from __future__ import annotations

import json
from typing import Annotated, Any

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

from shared.mcp_missing_arguments import MISSING_ARGUMENT_CODE, MissingArgumentMiddleware


class _NeedsField(BaseModel):
    required_inner_field: str


class Harness:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.mcp = FastMCP('missing-argument-harness')
        self.mcp.add_middleware(MissingArgumentMiddleware())
        self._register_tools()

    def _register_tools(self) -> None:
        calls = self.calls

        @self.mcp.tool
        def submit(
            task_id: str,
            branch: str,
            worktree: Annotated[str, Field(description='Absolute path of the task worktree')],
            note: str = '',
            flag: bool = False,
        ) -> dict:
            calls.append(
                {'task_id': task_id, 'branch': branch, 'worktree': worktree, 'note': note, 'flag': flag}
            )
            return {'ok': True}

        @self.mcp.tool
        def pair(alpha: str, beta: int) -> dict:
            calls.append({'alpha': alpha, 'beta': beta})
            return {'ok': True}

        @self.mcp.tool
        def explode(task_id: str) -> dict:
            _NeedsField.model_validate({})
            return {'ok': True}

    async def call(self, tool: str, arguments: dict[str, Any]):
        async with Client(self.mcp) as client:
            return await client.call_tool(tool, arguments)

    async def refusal(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with pytest.raises(ToolError) as excinfo:
            await self.call(tool, arguments)
        return json.loads(str(excinfo.value))

    async def raw_error(self, tool: str, arguments: dict[str, Any]) -> str:
        with pytest.raises(ToolError) as excinfo:
            await self.call(tool, arguments)
        return str(excinfo.value)


SIGHTING_CALL = {'task_id': '5099', 'branch': '5099', 'note': 'x', 'flag': False}


@pytest.fixture
def harness() -> Harness:
    return Harness()


async def test_sighting_shape_is_refused_with_structured_payload(harness):
    payload = await harness.refusal('submit', SIGHTING_CALL)

    assert payload['code'] == MISSING_ARGUMENT_CODE
    assert payload['tool'] == 'submit'
    assert [m['name'] for m in payload['missing']] == ['worktree']
    assert payload['missing'][0]['schema']['description'] == 'Absolute path of the task worktree'
    assert payload['missing'][0]['schema']['type'] == 'string'
    assert payload['provided'] == ['task_id', 'branch', 'note', 'flag']
    assert isinstance(payload['error'], str) and payload['error']
    assert 'worktree' in payload['error']
    assert isinstance(payload['hint'], str) and payload['hint']
    assert payload['other_errors'] == []
    assert harness.calls == []


async def test_example_call_is_a_declaration_ordered_template(harness):
    example = (await harness.refusal('submit', SIGHTING_CALL))['example_call']

    assert example.startswith('submit(')
    assert 'worktree=<string>' in example
    for name in ('task_id', 'branch', 'note', 'flag'):
        assert f'{name}=<as sent>' in example
    assert example.index('task_id=') < example.index('worktree=') < example.index('note=')
    assert "note='x'" not in example


async def test_several_missing_arguments_are_listed_in_declaration_order(harness):
    payload = await harness.refusal('pair', {})

    assert [m['name'] for m in payload['missing']] == ['alpha', 'beta']
    assert payload['missing'][1]['schema']['type'] == 'integer'
    assert 'description' not in payload['missing'][1]['schema']
    assert payload['example_call'] == 'pair(alpha=<string>, beta=<integer>)'


async def test_a_second_defect_in_the_same_call_is_preserved(harness):
    payload = await harness.refusal('submit', {'task_id': '1', 'branch': 'b', 'bogus': 1})

    assert [m['name'] for m in payload['missing']] == ['worktree']
    assert len(payload['other_errors']) == 1
    other = payload['other_errors'][0]
    assert other['type'] == 'unexpected_keyword_argument'
    assert other['loc'] == ['bogus']
    assert isinstance(other['msg'], str) and other['msg']
    assert 'bogus' not in payload['example_call']


async def test_a_complete_call_passes_through_untouched(harness):
    result = await harness.call('submit', {**SIGHTING_CALL, 'worktree': '/tmp/wt'})

    assert result.data == {'ok': True}
    assert len(harness.calls) == 1


async def test_a_non_missing_validation_error_keeps_fastmcps_text(harness):
    text = await harness.raw_error('submit', {**SIGHTING_CALL, 'worktree': '/tmp/wt', 'bogus': 1})

    assert MISSING_ARGUMENT_CODE not in text
    assert 'Unexpected keyword argument' in text
    assert harness.calls == []


async def test_a_validation_error_raised_by_the_tool_body_is_not_misreported(harness):
    text = await harness.raw_error('explode', {'task_id': '1'})

    assert MISSING_ARGUMENT_CODE not in text
    assert 'required_inner_field' in text
