"""``step_type`` is a closed vocabulary on add_plan_step / replace_plan_step.

``plans/plan-deviation-recording-prd.md`` D14 and contract §6.2: ``STEP_TYPES``
is a public constant of ``orchestrator.mcp.plan_tools``, and a value outside it
is refused before anything is written (task 5520 stored ``'tdd'``). A sibling of
``test_plan_tools_server.py``.
"""
from __future__ import annotations

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import plan_tools
from orchestrator.mcp.plan_tools import (
    STEP_TYPES,
    _add_plan_step,
    _create_plan,
    _replace_plan_step,
)


@pytest.fixture()
def artifacts(tmp_path):
    a = TaskArtifacts(tmp_path)
    a.init('test-1', 'Test task', 'A test')
    _create_plan(a, 'test-1', 'Test task', 'Analysis', ['mod_a/foo.py'])
    return a


def _assert_refused(result: dict, step_type: str) -> None:
    assert result['status'] == 'error'
    for word in (step_type, 'test', 'impl'):
        assert word in result['message']
    assert result['step_type'] == step_type
    assert result['allowed_step_types'] == sorted(STEP_TYPES)


def test_step_types_is_the_public_vocabulary():
    assert frozenset({'test', 'impl'}) == STEP_TYPES


@pytest.mark.parametrize('step_type', sorted(STEP_TYPES))
def test_add_plan_step_stores_each_allowed_type_under_type(artifacts, step_type):
    assert _add_plan_step(artifacts, 'step-1', step_type, 'X')['status'] == 'ok'

    assert artifacts.read_plan()['steps'][0]['type'] == step_type


@pytest.mark.parametrize('step_type', sorted(STEP_TYPES))
def test_replace_plan_step_accepts_each_allowed_type(artifacts, step_type):
    _add_plan_step(artifacts, 'step-1', 'test', 'X')

    assert _replace_plan_step(artifacts, 'step-1', step_type, 'Y')['status'] == 'ok'

    assert artifacts.read_plan()['steps'][0]['type'] == step_type


def test_add_plan_step_refuses_an_unknown_type_and_writes_nothing(artifacts):
    result = _add_plan_step(artifacts, 'step-1', 'tdd', 'X')

    _assert_refused(result, 'tdd')
    assert artifacts.read_plan()['steps'] == []


def test_replace_plan_step_refuses_an_unknown_type_and_keeps_the_step(artifacts):
    _add_plan_step(artifacts, 'step-1', 'test', 'X')

    result = _replace_plan_step(artifacts, 'step-1', 'tdd', 'Y')

    _assert_refused(result, 'tdd')
    step = artifacts.read_plan()['steps'][0]
    assert step['type'] == 'test'
    assert step['description'] == 'X'


@pytest.mark.asyncio
async def test_the_registered_tool_refuses_an_unknown_type(artifacts):
    tool = await plan_tools.create_server(artifacts).get_tool('add_plan_step')
    assert tool is not None

    result = await tool.run({'step_id': 'step-1', 'step_type': 'red', 'description': 'X'})

    assert result.structured_content is not None
    assert result.structured_content['status'] == 'error'
