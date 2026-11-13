"""SIMPLE_TASK's workflow has one home (task 6564).

Every parameterised plan-tools call SIMPLE_TASK's workflow spells lives in
``orchestrator/src/orchestrator/agents/roles.py::SIMPLE_TASK_WORKFLOW``, which
``SIMPLE_TASK.system_prompt`` carries. This mirrors task 5532's single-home test
for the architect's plan-tools recipe.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pytest
from _briefing_helpers import briefing as briefing
from _role_splice_contract import assert_nonempty
from shared.followup_items import FOLLOWUP_ITEMS_METADATA_KEY, observed_item

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.roles import ROLES, SIMPLE_TASK, SIMPLE_TASK_WORKFLOW

_TOOL_CALL = re.compile(r'`(?:mcp__plan-tools__)?(\w+)\(')

_FOLLOWUP_SUMMARY = 'The retry loop swallows the timeout'
_DELIVERED_CHECK_NAME = 'spot-6564-check'


def _task(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': '6564',
        'title': 'A simple change',
        'description': 'A mechanically simple change.',
        'metadata': metadata,
    }


_RENDER_VARIANTS: dict[str, dict[str, Any]] = {
    'files_listed': _task({'files': ['a.py', 'b.py']}),
    'no_files': _task({}),
    'every_optional_section': _task({
        'files': ['a.py', 'b.py'],
        'delivered_checks': [
            {'name': _DELIVERED_CHECK_NAME, 'kind': 'grep', 'pattern': 'X', 'expect': 'present'},
        ],
        FOLLOWUP_ITEMS_METADATA_KEY: [
            observed_item(
                summary=_FOLLOWUP_SUMMARY, files=['a.py'], origin_task='8000',
            ).model_dump(mode='json'),
        ],
    }),
}


def _workflow_tool_names() -> list[str]:
    names = sorted(set(_TOOL_CALL.findall(SIMPLE_TASK_WORKFLOW)))
    assert names, (
        'SIMPLE_TASK_WORKFLOW spells no backticked tool call, so every '
        'signature check below would be vacuous.'
    )
    return names


def _signature_spellings(text: str) -> list[str]:
    """Every parameter-list spelling of a workflow tool in *text*.

    Any parameter list counts, under any parameter names or line wrapping;
    ``tool(...)`` and a bare backticked name do not.
    """
    return [
        match.group(0)
        for name in _workflow_tool_names()
        for match in re.finditer(re.escape(name) + r'\(([^)`]+)\)', text)
        if match.group(1) != '...'
    ]


def _tool_name(spelling: str) -> str:
    return spelling.split('(', 1)[0]


class TestSimpleTaskWorkflowSingleHome:
    def test_workflow_reaches_only_simple_task_exactly_once(self):
        assert_nonempty(
            'SIMPLE_TASK_WORKFLOW', SIMPLE_TASK_WORKFLOW,
            remedy='Restore the workflow and rejection exits to roles.py::SIMPLE_TASK_WORKFLOW.',
        )
        carriers = [
            name for name, role in ROLES.items()
            if SIMPLE_TASK_WORKFLOW in role.system_prompt
        ]
        assert carriers == ['simple_task'], (
            f'SIMPLE_TASK_WORKFLOW should reach only the simple_task role; '
            f'it reaches {carriers}.'
        )
        count = SIMPLE_TASK.system_prompt.count(SIMPLE_TASK_WORKFLOW)
        assert count == 1, (
            f'SIMPLE_TASK.system_prompt carries SIMPLE_TASK_WORKFLOW {count} times; '
            'expected exactly once.'
        )

    def test_workflow_spells_each_tool_signature_once(self):
        spellings = _signature_spellings(SIMPLE_TASK_WORKFLOW)
        assert spellings, 'SIMPLE_TASK_WORKFLOW spells no tool signature.'
        repeated = {
            name: count
            for name, count in Counter(_tool_name(s) for s in spellings).items()
            if count > 1
        }
        assert not repeated, (
            f'SIMPLE_TASK_WORKFLOW spells these tools more than once: {repeated}. '
            'Spell each signature once in the home and name the tool bare elsewhere.'
        )


@pytest.mark.asyncio
@pytest.mark.parametrize('variant', sorted(_RENDER_VARIANTS))
async def test_dispatch_spells_each_workflow_signature_once(
    briefing: BriefingAssembler, variant: str,
):
    rendered = await briefing.build_simple_task_prompt(
        _RENDER_VARIANTS[variant], worktree=None, context='',
    )
    if variant == 'every_optional_section':
        assert _FOLLOWUP_SUMMARY in rendered, rendered
        assert _DELIVERED_CHECK_NAME in rendered, rendered

    dispatched = Counter(_signature_spellings(SIMPLE_TASK.system_prompt + '\n' + rendered))
    home = Counter(_signature_spellings(SIMPLE_TASK_WORKFLOW))
    assert dispatched == home, (
        f'Re-spelled outside the home: {sorted((dispatched - home).elements())}; '
        f'missing from the dispatch: {sorted((home - dispatched).elements())}. '
        "SIMPLE_TASK's workflow has one home, roles.py::SIMPLE_TASK_WORKFLOW, which "
        'SIMPLE_TASK.system_prompt carries into every simple-task dispatch. A briefing '
        'or another spliced block that re-spells a signature shows the agent a second '
        'copy that can drift (task 6564). Name the tool without its signature '
        '(`tool(...)`) or point at the role instructions instead.'
    )
