"""The implementer brief's Action paragraph follows config.implementer_brief_variant
(task 5385, plans/live-shadow-eval-prd.md decision 14) and nothing else does.

The whole-plan sentence is pinned by the literal oracle in
_implementer_action_oracle.py, independent of the code under change.
"""

from __future__ import annotations

import itertools

import pytest
from _implementer_action_oracle import ACTION_MARKER, WHOLE_PLAN_ACTION_PIN

from orchestrator.agents import briefing
from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.implementer_brief_variant import (
    ImplementerBriefVariant,
    execute_iteration_cap,
    invocation_bounds,
)

pytestmark = pytest.mark.usefixtures('code_default_config')

REBASE_NOTICE = {
    'old_base': '07abf574a61c',
    'new_base': '25439e8d1b02',
    'changed_files': ['orchestrator/src/orchestrator/workflow.py'],
}
WIP_NOTICE = [{'sha': 'fa2520d7e077', 'subject': 'WIP before inter-iteration rebase'}]


def _plan(status: str = 'pending') -> dict:
    def at(index: int) -> str:
        return ('done', 'pending')[index % 2] if status == 'mixed' else status

    return {
        'title': 'One step at a time',
        'analysis': 'Each implementer invocation completes one plan item.',
        'prerequisites': [{'id': 'pre-1', 'description': 'prepare', 'status': at(0)}],
        'steps': [
            {'id': f'step-{n}', 'description': f'step {n}', 'status': at(n)}
            for n in range(1, 4)
        ],
    }


def _config(tmp_path, **overrides) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=tmp_path,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
        **overrides,
    )


async def _render(config: OrchestratorConfig, plan: dict | None = None) -> str:
    return await BriefingAssembler(config).build_implementer_prompt(
        plan if plan is not None else _plan(),
        context='CTX',
        rebase_notice=REBASE_NOTICE,
        task_id='42',
        wip_notice=WIP_NOTICE,
    )


def _split_at_action(prompt: str) -> tuple[str, str]:
    assert prompt.count(ACTION_MARKER) == 1, prompt
    head, tail = prompt.split(ACTION_MARKER)
    return head + ACTION_MARKER, tail


@pytest.mark.asyncio
async def test_whole_plan_action_is_the_pinned_sentence(tmp_path):
    _, action = _split_at_action(await _render(_config(tmp_path)))
    assert action == WHOLE_PLAN_ACTION_PIN


@pytest.mark.asyncio
async def test_default_and_explicit_whole_plan_render_identically(tmp_path):
    default = await _render(_config(tmp_path))
    explicit = await _render(_config(tmp_path, implementer_brief_variant='whole-plan'))
    assert default == explicit


@pytest.mark.asyncio
async def test_one_step_changes_only_the_action_paragraph(tmp_path):
    whole_head, whole_action = _split_at_action(await _render(_config(tmp_path)))
    one_head, one_action = _split_at_action(
        await _render(_config(tmp_path, implementer_brief_variant='one-step')),
    )
    assert one_head == whole_head
    assert one_action != whole_action


@pytest.mark.asyncio
async def test_one_step_action_is_the_registered_one_step_paragraph(tmp_path):
    _, action = _split_at_action(
        await _render(_config(tmp_path, implementer_brief_variant='one-step')),
    )
    assert action == briefing.IMPLEMENTER_ACTIONS[ImplementerBriefVariant.ONE_STEP] + '\n'
    assert set(briefing.IMPLEMENTER_ACTIONS) == set(ImplementerBriefVariant)


@pytest.mark.asyncio
async def test_an_unvalidated_str_variant_is_whole_plan_for_brief_cap_and_bounds(tmp_path):
    config = _config(tmp_path).model_copy(update={'implementer_brief_variant': 'one-step'})
    variant = config.implementer_brief_variant

    _, action = _split_at_action(await _render(config))

    assert action == WHOLE_PLAN_ACTION_PIN
    assert execute_iteration_cap(variant, 10, _plan) == 10
    assert invocation_bounds(variant, 'implementer', 20.0, 160) == (20.0, 160)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('left', 'right'), list(itertools.combinations(('pending', 'done', 'mixed'), 2)),
)
async def test_one_step_brief_carries_no_plan_progress(tmp_path, left, right):
    config = _config(tmp_path, implementer_brief_variant='one-step')
    assert await _render(config, _plan(left)) == await _render(config, _plan(right))
