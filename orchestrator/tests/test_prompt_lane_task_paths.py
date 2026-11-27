"""Agents reach their task's state only through the lane `.task/` links
(task 3160): no prompt computes a `.task-meta` path, and every `.task/<name>`
a prompt names resolves into this task's durable storage. The linked names
are `orchestrator/src/orchestrator/artifacts.py::LANE_SYMLINKED_ARTIFACTS`.
"""

import re
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _briefing_helpers import _mcp_search_envelope, memory_transport
from _role_splice_contract import SpliceContract, assert_nonempty, assert_opens_own_section

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.lane_task_guidance import LANE_TASK_LINKS_GUIDANCE, LANE_TASK_LINKS_SECTION
from orchestrator.agents.roles import ROLES, AgentRole
from orchestrator.artifacts import LANE_SYMLINKED_ARTIFACTS, TaskArtifacts

_LANE_PATH = re.compile(r'\.task/[A-Za-z0-9_.-]*[A-Za-z0-9_]')
_LINKED_PATHS = tuple(f'.task/{name}' for name in LANE_SYMLINKED_ARTIFACTS)
_UNLINKED_TASK_STATE = ('metadata.json', 'reviews/', 'verdicts/')


def build_dual_root_lane(tmp_path: Path) -> tuple[Path, TaskArtifacts]:
    """A dispatched lane for task 3160 holding a plan and one logged iteration."""
    wt = tmp_path / 'worktrees' / '_lane-3'
    wt.mkdir(parents=True)
    arts = TaskArtifacts.for_task(wt, '3160')
    arts.init('3160', 'T', 'D')
    arts.ensure_lane_symlinks()
    arts.write_plan({'steps': []})
    arts.append_iteration_log({'iteration': 1, 'agent': 'implementer', 'summary': 'S'})
    return wt, arts


def assert_lane_paths_resolve(label: str, prompt: str, wt: Path, arts: TaskArtifacts) -> None:
    task_root = arts.task_root.resolve()
    for rel in sorted(set(_LANE_PATH.findall(prompt))):
        lane_path = wt / rel
        assert lane_path.resolve().is_relative_to(task_root), (
            f'{label} names {rel}, which does not resolve into the task root {task_root}'
        )
        assert lane_path.read_text() == arts.artifact_path(Path(rel).name).read_text(), (
            f'{label} names {rel}, which does not read as this task\'s {Path(rel).name}'
        )


@pytest.mark.parametrize('name', sorted(ROLES))
def test_no_role_prompt_computes_a_meta_root_path(name):
    prompt = ROLES[name].system_prompt
    assert '<worktree_base>' not in prompt, f'{name} names <worktree_base>'
    assert '.task-meta/<' not in prompt, f'{name} builds a .task-meta path'


@pytest.mark.parametrize('name', ['implementer', 'debugger'])
def test_history_reading_roles_name_both_lane_paths(name):
    prompt = ROLES[name].system_prompt
    for path in _LINKED_PATHS:
        assert path in prompt, f'{name} does not name {path}'


def test_lane_task_dir_exposes_only_the_linked_artifacts(tmp_path):
    wt, arts = build_dual_root_lane(tmp_path)
    assert sorted(p.name for p in (wt / '.task').iterdir()) == sorted(LANE_SYMLINKED_ARTIFACTS)
    metadata = arts.artifact_path('metadata.json')
    assert metadata.is_file() and not metadata.is_relative_to(wt), (
        f'metadata.json was written to {metadata}, not outside the worktree {wt}'
    )


def _names_a_lane_link(role: AgentRole) -> bool:
    return any(path in role.system_prompt for path in _LINKED_PATHS)


_LANE_LINK_ROLES = tuple(sorted(name for name, role in ROLES.items() if _names_a_lane_link(role)))


def test_some_role_names_a_lane_link():
    assert _LANE_LINK_ROLES, f'no role names any of {_LINKED_PATHS}, so the layout test checks nothing'


@pytest.mark.parametrize('name', _LANE_LINK_ROLES)
def test_lane_link_roles_name_the_linked_and_the_unlinked_task_state(name):
    prompt = ROLES[name].system_prompt
    for path in _LINKED_PATHS:
        assert path in prompt, f'{name} does not name {path}'
    for unlinked in _UNLINKED_TASK_STATE:
        assert unlinked in prompt, (
            f'{name} describes the .task/ links without saying {unlinked} is not exposed there'
        )


def test_every_role_naming_a_lane_link_carries_the_link_section_once():
    remedy = 'Restore the block in orchestrator/src/orchestrator/agents/lane_task_guidance.py.'
    assert_nonempty('LANE_TASK_LINKS_GUIDANCE', LANE_TASK_LINKS_GUIDANCE, remedy=remedy)
    assert_opens_own_section('LANE_TASK_LINKS_SECTION', LANE_TASK_LINKS_SECTION, remedy=remedy)
    naming = frozenset(name for name, role in ROLES.items() if _names_a_lane_link(role))
    assert naming, f'no role names any of {_LINKED_PATHS}, so this invariant checks nothing'
    SpliceContract(
        constant_name='LANE_TASK_LINKS_SECTION',
        constant=LANE_TASK_LINKS_SECTION,
        roles=naming,
        role_set_name='the roles naming a .task/ link',
        capability=_names_a_lane_link,
        capability_description=f'a mention of {_LINKED_PATHS}',
    ).assert_spliced_exactly_once(
        remedy='Splice roles.py\'s LANE_TASK_LINKS_SECTION once into that role, as every carrier does.',
    )


@pytest.mark.parametrize('name', sorted(ROLES))
def test_every_lane_path_a_role_names_resolves_to_this_tasks_state(tmp_path, name):
    wt, arts = build_dual_root_lane(tmp_path)
    assert_lane_paths_resolve(f'role {name}', ROLES[name].system_prompt, wt, arts)


_PLAN = {'title': 'T', 'analysis': 'A', 'steps': [], 'files': []}
_TASK = {'id': '3160', 'title': 'T', 'description': 'D', 'metadata': {}}


async def _implementer(b: BriefingAssembler) -> str:
    return await b.build_implementer_prompt(_PLAN, context='', task_id='3160')


async def _amender(b: BriefingAssembler) -> str:
    return await b.build_amender_prompt(
        _PLAN,
        suggestions=[{'reviewer': 'r', 'category': 'c', 'location': 'l', 'description': 'd'}],
        locked_modules=['orchestrator'],
        context='',
        task_id='3160',
    )


async def _resume(b: BriefingAssembler) -> str:
    with memory_transport(AsyncMock(return_value=_mcp_search_envelope([]))):
        return await b.build_resume_prompt(_TASK, _PLAN, 'ISSUE', 'RESOLUTION')


async def _plan_schema_repair(b: BriefingAssembler) -> str:
    return await b.build_plan_schema_repair_prompt(_TASK, context='')


async def _replan(b: BriefingAssembler) -> str:
    return await b.build_replan_prompt(_TASK, 'FEEDBACK', context='')


_BRIEFINGS = {
    'implementer': _implementer,
    'amender': _amender,
    'resume': _resume,
    'plan_schema_repair': _plan_schema_repair,
    'replan': _replan,
}


@pytest.mark.asyncio
@pytest.mark.parametrize('name', sorted(_BRIEFINGS))
async def test_no_briefing_computes_a_meta_root_path(briefing, name):
    prompt = await _BRIEFINGS[name](briefing)
    assert '<worktree_base>' not in prompt, f'{name} names <worktree_base>'
    assert '.task-meta/<' not in prompt, f'{name} builds a .task-meta path'


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['implementer', 'amender', 'resume'])
async def test_history_reading_briefings_name_both_lane_paths(briefing, name):
    prompt = await _BRIEFINGS[name](briefing)
    for path in _LINKED_PATHS:
        assert path in prompt, f'{name} does not name {path}'


@pytest.mark.asyncio
@pytest.mark.parametrize('name', sorted(_BRIEFINGS))
async def test_every_lane_path_a_briefing_names_resolves_to_this_tasks_state(
    briefing, tmp_path, name,
):
    wt, arts = build_dual_root_lane(tmp_path)
    assert_lane_paths_resolve(f'briefing {name}', await _BRIEFINGS[name](briefing), wt, arts)
