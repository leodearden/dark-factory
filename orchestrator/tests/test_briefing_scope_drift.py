"""The scope-drift section of the implementer and debugger briefings (task 3090).

After a writer boundary applied the agent's scope requests, the next writer's
briefing names each granted path and each contested path with its holder.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _briefing_helpers import memory_transport

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.pin_reservation import Blocker, BlockerKind
from orchestrator.scope_drift import ScopeBoundary, ScopeDriftReport, build_scope_drift_report

_PLAN = {'task_id': '1', 'title': 'T', 'analysis': 'A'}
_FAILURES = '## Test Failures\n\n```\nFAILED tests/test_x.py::test_a\n```'
_GRANTED = 'x/granted.py'
_HELD = 'x/held.py'
_HOLDER = '777'


@pytest.fixture
def briefing(tmp_path: Path) -> Iterator[BriefingAssembler]:
    config = OrchestratorConfig(
        project_root=tmp_path,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )
    with memory_transport(AsyncMock(side_effect=ConnectionRefusedError('no memory in tests'))):
        yield BriefingAssembler(config)


def _report(requested: list[str], contested: list[Blocker]) -> ScopeDriftReport:
    return build_scope_drift_report(
        requested, contested, lock_depth=12, boundary=ScopeBoundary.EXECUTE,
        role='implementer', iteration=1, task_id='1',
    )


_MIXED = _report([_GRANTED, _HELD], [Blocker(_HELD, _HOLDER, BlockerKind.HELD)])
_ALL_GRANTED = _report([_GRANTED], [])


async def _implementer(briefing: BriefingAssembler, **kwargs) -> str:
    return await briefing.build_implementer_prompt(_PLAN, context='', task_id='1', **kwargs)


async def _debugger(briefing: BriefingAssembler, **kwargs) -> str:
    return await briefing.build_debugger_prompt(
        _FAILURES, _PLAN, context='', task_id='1', **kwargs
    )


async def _resume(briefing: BriefingAssembler, **kwargs) -> str:
    return await briefing.build_resume_prompt(
        {'id': '1', 'title': 'T', 'description': 'd'}, _PLAN, 'issue', 'resolution', **kwargs
    )


async def _amender(briefing: BriefingAssembler, **kwargs) -> str:
    return await briefing.build_amender_prompt(
        _PLAN, [], ['x/a.py'], context='', task_id='1', **kwargs
    )


_BUILDERS = [_implementer, _debugger, _resume, _amender]


@pytest.mark.asyncio
@pytest.mark.parametrize('build', _BUILDERS)
async def test_contested_path_is_named_with_its_holder_on_one_line(briefing, build):
    prompt = await build(briefing, scope_drift=_MIXED)

    assert _GRANTED in prompt
    [held_line] = [line for line in prompt.splitlines() if _HELD in line]
    assert _HOLDER in held_line


@pytest.mark.asyncio
@pytest.mark.parametrize('build', _BUILDERS)
async def test_all_granted_report_names_the_path_and_no_holder(briefing, build):
    prompt = await build(briefing, scope_drift=_ALL_GRANTED)

    assert _GRANTED in prompt
    assert _HELD not in prompt
    assert _HOLDER not in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize('build', _BUILDERS)
async def test_no_report_leaves_the_prompt_unchanged(briefing, build):
    assert await build(briefing, scope_drift=None) == await build(briefing)


@pytest.mark.asyncio
async def test_the_follow_up_prompt_names_the_contested_path_and_its_holder(briefing):
    prompt = await briefing.build_scope_drift_followup_prompt(_PLAN, _MIXED, task_id='1')

    [held_line] = [line for line in prompt.splitlines() if _HELD in line]
    assert _HOLDER in held_line
