"""The architect's plan-schema-repair and post-review replan briefings.

Both passes run on a plan that already exists, so each briefing must carry
what every other architect pass carries (task-scoped memory, the agent's
identity, the task text) and must send the architect to the plan's single
home on disk rather than a copy frozen into the prompt. Each may prescribe
only tools the ARCHITECT role actually holds.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _briefing_helpers import (
    _mcp_search_envelope,
    _result,
    _search_arguments,
    briefing,  # noqa: F401 — re-export: pytest fixture used by test methods
    memory_transport,
)
from test_roles_ancestry_check import _tool_is_granted

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.roles import ARCHITECT
from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import plan_tools

_TASK = {
    'id': '5356',
    'title': 'Repair a stepless plan',
    'description': 'DESC-SENTINEL',
    'metadata': {'files': ['orchestrator/src/orchestrator/workflow.py']},
}
_ARCHITECT_ID = 'claude-task-5356-architect'
_PLAN_PATH = '.task/plan.json'


def _memory_reply() -> dict:
    entry = _result('1', 'A recalled fact.', source_store='mem0')
    entry['category'] = 'preferences_and_norms'
    entry['created_at'] = '2026-08-15T22:22:49+00:00'
    return _mcp_search_envelope([entry])


def _assert_carries_task_identity_and_context(prompt: str) -> None:
    assert 'CONTEXT-SENTINEL' in prompt
    assert '**ID:** 5356' in prompt
    assert 'DESC-SENTINEL' in prompt
    assert _ARCHITECT_ID in prompt


def _assert_recalled_under_architect_identity(prompt: str, mcp: AsyncMock) -> None:
    assert 'A recalled fact.' in prompt
    arguments = _search_arguments(mcp)
    assert arguments
    for args in arguments:
        assert args['caller_agent_id'] == _ARCHITECT_ID
        assert args['caller_task_id'] == '5356'


class TestPlanSchemaRepairPrompt:
    @pytest.mark.asyncio
    async def test_carries_task_identity_and_context(self, briefing: BriefingAssembler):
        prompt = await briefing.build_plan_schema_repair_prompt(
            _TASK, context='CONTEXT-SENTINEL',
        )
        _assert_carries_task_identity_and_context(prompt)

    @pytest.mark.asyncio
    async def test_points_at_the_plan_file_and_requires_attestation(
        self, briefing: BriefingAssembler,
    ):
        prompt = await briefing.build_plan_schema_repair_prompt(_TASK, context='')
        assert _PLAN_PATH in prompt
        assert 'add_plan_step' in prompt
        assert 'confirm_plan' in prompt

    @pytest.mark.asyncio
    async def test_recalls_memory_when_no_context_is_passed(
        self, briefing: BriefingAssembler,
    ):
        mcp = AsyncMock(return_value=_memory_reply())
        with memory_transport(mcp):
            prompt = await briefing.build_plan_schema_repair_prompt(_TASK)
        _assert_recalled_under_architect_identity(prompt, mcp)


class TestReplanPrompt:
    @pytest.mark.asyncio
    async def test_carries_task_identity_context_and_feedback(
        self, briefing: BriefingAssembler,
    ):
        prompt = await briefing.build_replan_prompt(
            _TASK, 'FEEDBACK-SENTINEL', context='CONTEXT-SENTINEL',
        )
        _assert_carries_task_identity_and_context(prompt)
        assert 'FEEDBACK-SENTINEL' in prompt

    @pytest.mark.asyncio
    async def test_points_at_the_plan_file_and_prescribes_plan_tools(
        self, briefing: BriefingAssembler,
    ):
        prompt = await briefing.build_replan_prompt(_TASK, 'FEEDBACK', context='')
        assert _PLAN_PATH in prompt
        assert 'add_plan_step' in prompt
        assert 'update_plan_metadata' in prompt

    @pytest.mark.asyncio
    async def test_recalls_memory_when_no_context_is_passed(
        self, briefing: BriefingAssembler,
    ):
        mcp = AsyncMock(return_value=_memory_reply())
        with memory_transport(mcp):
            prompt = await briefing.build_replan_prompt(_TASK, 'FEEDBACK')
        _assert_recalled_under_architect_identity(prompt, mcp)


async def _repair_prompt(briefing: BriefingAssembler) -> str:
    return await briefing.build_plan_schema_repair_prompt(_TASK, context='')


async def _replan_prompt(briefing: BriefingAssembler) -> str:
    return await briefing.build_replan_prompt(_TASK, 'FEEDBACK', context='')


_BUILDERS: list[Callable[[BriefingAssembler], Awaitable[str]]] = [
    _repair_prompt, _replan_prompt,
]


class TestArchitectPassToolContract:
    """Each pass may prescribe only tools the ARCHITECT role holds.

    ``test_roles_ancestry_check.py::test_role_holds_every_mcp_tool_its_prompt_names``
    scans only ``role.system_prompt``, so it cannot see these dispatch-time
    prompts.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize('build', _BUILDERS, ids=lambda b: b.__name__)
    async def test_every_plan_tool_named_is_granted_to_architect(
        self, briefing: BriefingAssembler, tmp_path: Path, build,
    ):
        artifacts = TaskArtifacts(tmp_path / 'wt')
        artifacts.init('5356', 'T', 'd')
        server = plan_tools.create_server(artifacts)
        registry = {tool.name for tool in await server.list_tools()}

        prompt = await build(briefing)
        named = {n for n in registry if re.search(rf'\b{re.escape(n)}\b', prompt)}
        assert named, 'the prompt names no registered plan tool, so this check is vacuous'

        granted = set(ARCHITECT.allowed_tools)
        missing = sorted(
            qualified for qualified in (f'mcp__plan-tools__{n}' for n in named)
            if not _tool_is_granted(qualified, granted)
        )
        assert not missing, f'prescribed plan tools absent from ARCHITECT.allowed_tools: {missing}'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('build', _BUILDERS, ids=lambda b: b.__name__)
    async def test_never_prescribes_a_builtin_tool_the_architect_is_denied(
        self, briefing: BriefingAssembler, build,
    ):
        prompt = await build(briefing)
        denied_builtins = [t for t in ARCHITECT.disallowed_tools if not t.startswith('mcp__')]
        assert denied_builtins, 'ARCHITECT denies no built-in tool, so this check is vacuous'
        for tool in denied_builtins:
            assert f'{tool} tool' not in prompt
            assert f'`{tool}`' not in prompt
