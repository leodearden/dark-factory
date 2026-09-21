"""Plan-step STATUS VALUES do not influence the implementer briefing (task 5728).

Plan progress has one home the implementer may rely on: ``.task/plan.json``,
which the briefing's own Session Startup Protocol orders it to read. A second
copy frozen into the prompt string is the copy that can lie, because the retry
ladder replays an assembled prompt verbatim into a brand-new CLI session
(``shared/src/shared/cli_invoke.py::_reset_for_fresh_retry``).

The pin is the property, not the absent wording: every plan below differs from
the others ONLY in its steps' and prerequisites' ``status`` values, so any
byte of difference between the rendered prompts is status-derived. Four status
assignments are compared pairwise — homogeneous done, homogeneous pending, a
MIXED plan, and a status the renderer has no branch for — because a counter
reintroduced behind ``if done and pending`` is invisible to two homogeneous
plans alone. Every optional section (recent iterations, rebase notice, WIP
notice) is populated identically in each call, so a counter hidden inside one
of them is caught too.

SCOPE: this cannot catch a progress claim derived from ``iteration_log``
rather than from ``plan`` — that argument is held identical here by design, so
a count taken from it is not status-derived and is out of this test's reach.
``## Recent Iterations`` is such a render today; see task 5728's record.
"""

from __future__ import annotations

import itertools

import pytest

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.config import GitConfig, OrchestratorConfig

# Held identical across every call so the optional sections all render and the
# only free variable is the plan's status values.
ITERATION_LOG = [
    {'iteration': 1, 'steps_completed': ['step-1'], 'summary': 'landed the envelope'},
    {'iteration': 2, 'steps_completed': ['step-2'], 'summary': 'landed the registry'},
]
REBASE_NOTICE = {
    'old_base': '07abf574a61c',
    'new_base': '25439e8d1b02',
    'changed_files': ['dashboard/src/dashboard/static/redux/data.js'],
}
WIP_NOTICE = [{'sha': 'fa2520d7e077', 'subject': 'WIP before inter-iteration rebase'}]

STATUSES = ('pending', 'done', 'mixed', 'blocked')


def assembler(tmp_path) -> BriefingAssembler:
    config = OrchestratorConfig(
        project_root=tmp_path,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )
    return BriefingAssembler(config)


def plan_all(status: str) -> dict:
    """A plan whose items all carry *status* — or alternate, when 'mixed'.

    Ids, descriptions, ordering, title and analysis are identical for every
    *status*, so two plans built here differ in nothing else.
    """
    def at(index: int) -> str:
        return ('done', 'pending')[index % 2] if status == 'mixed' else status

    return {
        'title': 'Render a Datum everywhere',
        'analysis': 'The shared components become the only place a Datum is rendered.',
        'prerequisites': [
            {'id': 'pre-1', 'description': 'hoist the staleness module', 'status': at(0)},
        ],
        'steps': [
            {'id': f'step-{n}', 'description': f'step {n}', 'status': at(n)}
            for n in range(1, 19)
        ],
    }


async def render(tmp_path, status: str) -> str:
    """The implementer prompt for *status*, with every optional section filled.

    ``context=''`` short-circuits the memory recall inside
    ``build_implementer_prompt``, so no backend is reached.
    """
    return await assembler(tmp_path).build_implementer_prompt(
        plan_all(status),
        ITERATION_LOG,
        context='',
        rebase_notice=REBASE_NOTICE,
        task_id='5728',
        wip_notice=WIP_NOTICE,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('left,right', list(itertools.combinations(STATUSES, 2)))
async def test_step_status_does_not_change_the_implementer_prompt(tmp_path, left, right):
    """Plans differing only in status must render byte-identical prompts."""
    assert await render(tmp_path, left) == await render(tmp_path, right), (
        f'The implementer prompt differs between an all-{left} and an all-{right} '
        'plan, so it carries a progress snapshot that can go stale when the '
        'prompt is replayed. Plan progress belongs in .task/plan.json only.'
    )


@pytest.mark.asyncio
async def test_implementer_prompt_is_not_degenerate(tmp_path):
    """Guard the pin above against passing on an empty or contentless prompt.

    Without this, a ``build_implementer_prompt`` that returned a constant — or
    one whose plan argument never reached the renderer — would satisfy every
    assertion above.
    """
    prompt = await render(tmp_path, 'pending')

    assert 'Render a Datum everywhere' in prompt   # the plan dict reached the renderer
    assert '.task/plan.json' in prompt             # the re-read instruction survives
    assert '## Recent Iterations' in prompt        # every optional section renders
    assert '## Rebase Notice' in prompt
    assert '25439e8d1b02' in prompt

    # And the positive statement of what the deletion achieved: no per-step plan
    # content reaches the prompt at all. step-18 exists only in the plan — the
    # iteration log above names step-1 and step-2 — so its absence is evidence
    # about the plan's steps specifically, not about step ids in general.
    assert 'step-18' not in prompt
