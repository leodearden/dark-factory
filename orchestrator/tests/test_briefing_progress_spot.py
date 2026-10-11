"""Neither plan progress nor iteration history reaches the implementer briefing.

Both have exactly one home the agent may rely on — ``.task/plan.json`` and the
durable ``iterations.jsonl`` — and the briefing's own Session Startup Protocol
orders it to read both. A second copy frozen into the prompt string is the copy
that can lie, because the retry ladder replays an assembled prompt verbatim into
a brand-new CLI session (``shared/src/shared/cli_invoke.py::_reset_for_fresh_retry``),
so a snapshot can reach an agent hours after it was taken. Task 5728 removed the
``## Progress`` counter; task 5744 removed ``## Recent Iterations`` and, with it,
the ``iteration_log`` parameter itself.

WHAT THESE PINS ACTUALLY COVER, stated precisely because an earlier draft of this
file overclaimed and a pre-merge review caught it:

* **Plan status** is pinned by OUTPUT. Four plans differing only in their items'
  ``status`` values (homogeneous pending, homogeneous done, a MIXED plan, and a
  status the renderer has no branch for) must render byte-identical prompts,
  compared pairwise. The mixed plan matters because a counter reintroduced behind
  ``if done and pending`` is invisible to homogeneous plans alone, and every
  optional section is populated identically so a counter hidden inside one of
  them is caught too.
* **Iteration history** is pinned three ways, none of them complete alone: the
  builders may accept no ``iteration_log`` parameter and no ``**kwargs`` through
  which one could be smuggled; neither builder's output may carry the section
  heading; and a history planted on the ``plan`` argument must not surface.

WHAT THEY DO NOT COVER, so nobody mistakes green here for proof: a render fed by
a DIFFERENTLY NAMED parameter, or one the builder reads off disk itself, defeats
every check in this file. Both would be caught by review, not by this test. The
point of the parameter ban is to make either of those a conspicuous edit rather
than an easy one.
"""

from __future__ import annotations

import inspect
import itertools

import pytest

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.config import GitConfig, OrchestratorConfig

# Held identical across every call so the optional sections all render and the
# only free variable is the plan's status values.
REBASE_NOTICE = {
    'old_base': '07abf574a61c',
    'new_base': '25439e8d1b02',
    'changed_files': ['dashboard/src/dashboard/static/redux/data.js'],
}
WIP_SHA = 'fa2520d7e077'
WIP_NOTICE = [{'sha': WIP_SHA, 'subject': 'WIP before inter-iteration rebase'}]

# Planted on the plan dict so a builder that smuggles history through an argument
# it still legitimately receives is caught by the output pins below.
SMUGGLED_HISTORY = [
    {'iteration': 1, 'steps_completed': ['step-1'], 'summary': 'SMUGGLED-HISTORY-MARKER'},
]

STATUSES = ('pending', 'done', 'mixed', 'blocked')

# The builders that must carry neither snapshot. build_completion_judge_prompt is
# deliberately NOT here: its prompt never tells the judge where iterations.jsonl
# is, so its render is the only way that role is pointed at the history, and
# removing it would delete information rather than deduplicate it. (The judge
# does hold Read/Glob/Grep, so it is uninformed rather than incapable — the
# opposite remedy, pointing it at the file, is a separate question.)
DEDUPLICATED_BUILDERS = ('build_implementer_prompt', 'build_amender_prompt')


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
    *status*, so two plans built here differ in nothing else. ``iteration_log``
    rides along as bait: it is not a key the builders read, and the output pins
    below fail if one starts to.
    """
    def at(index: int) -> str:
        return ('done', 'pending')[index % 2] if status == 'mixed' else status

    return {
        'title': 'Render a Datum everywhere',
        'analysis': 'The shared components become the only place a Datum is rendered.',
        'iteration_log': SMUGGLED_HISTORY,
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
        context='',
        rebase_notice=REBASE_NOTICE,
        task_id='5744',
        wip_notice=WIP_NOTICE,
    )


async def render_amender(tmp_path) -> str:
    """The amender prompt, which dispatches under the same IMPLEMENTER role."""
    return await assembler(tmp_path).build_amender_prompt(
        plan=plan_all('pending'),
        suggestions=[{'reviewer': 'r', 'category': 'c', 'location': 'l', 'issue': 'i'}],
        locked_modules=['orchestrator'],
        context='',
        task_id='5744',
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('left,right', list(itertools.combinations(STATUSES, 2)))
async def test_plan_status_does_not_change_the_implementer_prompt(tmp_path, left, right):
    """Plans differing only in status must render byte-identical prompts."""
    assert await render(tmp_path, left) == await render(tmp_path, right), (
        f'The implementer prompt differs between an all-{left} and an all-{right} '
        'plan, so it carries a progress snapshot that can go stale when the '
        'prompt is replayed. Plan progress belongs in .task/plan.json only.'
    )


@pytest.mark.parametrize('builder_name', DEDUPLICATED_BUILDERS)
def test_builder_cannot_receive_an_iteration_log(builder_name):
    """Reintroducing the section via this parameter means reintroducing its name."""
    params = inspect.signature(getattr(BriefingAssembler, builder_name)).parameters

    assert 'iteration_log' not in params, (
        f'{builder_name} accepts an iteration_log again. iterations.jsonl is its '
        'single home and the prompt already orders the agent to read it; a copy '
        'frozen into the prompt string is replayed stale on a fresh retry.'
    )
    var_keyword = [
        name for name, p in params.items() if p.kind is inspect.Parameter.VAR_KEYWORD
    ]
    assert not var_keyword, (
        f'{builder_name} grew {var_keyword} — a **kwargs bag would let an '
        'iteration_log back in without naming it, defeating the ban above.'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('builder_name', DEDUPLICATED_BUILDERS)
async def test_neither_builder_renders_iteration_history(tmp_path, builder_name):
    """Output pin, per builder — the signature ban is not the only guard."""
    prompt = (
        await render(tmp_path, 'pending')
        if builder_name == 'build_implementer_prompt'
        else await render_amender(tmp_path)
    )

    assert '## Recent Iterations' not in prompt, (
        f'{builder_name} renders an iteration-history section again.'
    )
    assert 'SMUGGLED-HISTORY-MARKER' not in prompt, (
        f'{builder_name} rendered history planted on the plan argument, which is '
        'the same stale-snapshot defect wearing a different argument name.'
    )
    assert 'iterations.jsonl' in prompt, (
        f'{builder_name} no longer tells the agent to read the live log — with the '
        'rendered copy gone, that instruction is its only route to the history.'
    )


@pytest.mark.asyncio
async def test_implementer_prompt_is_not_degenerate(tmp_path):
    """Guard the pins above against passing on an empty or contentless prompt.

    Without this, a ``build_implementer_prompt`` that returned a constant — or one
    that silently stopped rendering an optional section, making the status
    comparison vacuous for it — would satisfy every assertion above.
    """
    prompt = await render(tmp_path, 'pending')

    assert 'Render a Datum everywhere' in prompt   # the plan dict reached the renderer
    assert '.task/plan.json' in prompt             # the plan re-read instruction survives
    assert '## Rebase Notice' in prompt            # both optional sections still render
    assert '25439e8d1b02' in prompt
    assert WIP_SHA in prompt

    # step-18 exists only in the plan, so its absence is evidence about the plan's
    # steps specifically, not about step ids in general.
    assert 'step-18' not in prompt
