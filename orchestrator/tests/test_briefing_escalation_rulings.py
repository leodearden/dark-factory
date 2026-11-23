"""Every fresh-dispatch and reviewer briefing renders the task's escalation rulings.

Task 4764.  These tests drive the public ``BriefingAssembler`` builders with a
real :class:`orchestrator.escalation_rulings.Ruling` and assert the block each
one renders and where.  Every call passes ``context`` so no memory recall runs.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest
from _briefing_helpers import briefing as briefing
from escalation.models import Escalation

from orchestrator.agents.briefing import OPEN_INFO_NOTES_HEADING, BriefingAssembler
from orchestrator.agents.review_briefing import ChangedFile, ReviewChange, ReviewSubject
from orchestrator.escalation_rulings import Ruling, RulingAudience, render_rulings_section
from orchestrator.evals.replay_frame import ReplayFramedBriefingAssembler

REVIEWER = 'reviewer_comprehensive'
TASK = {'id': '4764', 'title': 'Brief agents with rulings', 'description': 'D'}
PLAN = {
    'task_id': '4764', 'title': 'Brief agents with rulings', 'analysis': 'A',
    'files': ['m/x.py'], 'steps': [{'id': 'step-1', 'type': 'impl', 'description': 'd', 'status': 'pending'}],
}
DIFF = 'diff --git a/m/x.py b/m/x.py\n--- a/m/x.py\n+++ b/m/x.py\n@@ -0,0 +1,1 @@\n+x = 1\n'
RULINGS = (
    Ruling(
        escalation_id='esc-4764-3', level=2, summary='keep or drop the solver wrap?',
        resolution='KEEP the solver wrap — operator ruling', resolved_at='2026-08-02T00:00:00+00:00',
        resolved_by='interactive', resolution_action='resume',
    ),
)
NOTE = Escalation(
    id='esc-4764-9', task_id='4764', agent_role='implementer', severity='info',
    category='design_concern', summary='helper X is duplicated',
)

Build = Callable[..., Awaitable[str]]


def _subject() -> ReviewSubject:
    return ReviewSubject(
        task=TASK,
        plan=None,
        plan_absent_reason='no plan in this test',
        change=ReviewChange(
            base_sha='0123456789abcdef0123456789abcdef01234567',
            diff=DIFF,
            files=(ChangedFile('m/x.py', 1, 0),),
        ),
        conventions=(),
    )


async def _architect(b: BriefingAssembler, **kw) -> str:
    return await b.build_architect_prompt(TASK, context='CTX', **kw)


async def _completion(b: BriefingAssembler, **kw) -> str:
    return await b.build_plan_completion_prompt(TASK, PLAN, context='CTX', **kw)


async def _revalidation(b: BriefingAssembler, **kw) -> str:
    return await b.build_revalidation_prompt(TASK, PLAN, ['m/x.py'], context='CTX', **kw)


async def _replan(b: BriefingAssembler, **kw) -> str:
    return await b.build_replan_prompt(TASK, 'BLOCKING: the wrap was dropped', context='CTX', **kw)


async def _simple(b: BriefingAssembler, **kw) -> str:
    return await b.build_simple_task_prompt(TASK, context='CTX', **kw)


async def _implementer(b: BriefingAssembler, **kw) -> str:
    return await b.build_implementer_prompt(PLAN, context='CTX', task_id='4764', **kw)


async def _debugger(b: BriefingAssembler, **kw) -> str:
    return await b.build_debugger_prompt('FAILED test_x', PLAN, context='CTX', task_id='4764', **kw)


async def _amender(b: BriefingAssembler, **kw) -> str:
    return await b.build_amender_prompt(
        PLAN, [{'description': 'name the constant'}], ['m'], context='CTX', task_id='4764', **kw,
    )


async def _diff_only_reviewer(b: BriefingAssembler, **kw) -> str:
    return await b.build_reviewer_prompt(REVIEWER, DIFF, context='CTX', **kw)


async def _context_complete_reviewer(b: BriefingAssembler, **kw) -> str:
    return await b.build_context_complete_reviewer_prompt(REVIEWER, _subject(), context='CTX', **kw)


ACTION_SLOTTED = [
    pytest.param(_architect, RulingAudience.PLAN, 1, id='architect'),
    pytest.param(_completion, RulingAudience.PLAN, 2, id='completion'),
    pytest.param(_revalidation, RulingAudience.PLAN, 2, id='revalidation'),
    pytest.param(_replan, RulingAudience.PLAN, 2, id='replan'),
    pytest.param(_simple, RulingAudience.PLAN_AND_EXECUTE, 1, id='simple_task'),
    pytest.param(_implementer, RulingAudience.EXECUTE, 1, id='implementer'),
    pytest.param(_debugger, RulingAudience.EXECUTE, 1, id='debugger'),
    pytest.param(_amender, RulingAudience.EXECUTE, 1, id='amender'),
]
REVIEWERS = [
    pytest.param(_diff_only_reviewer, id='diff_only_reviewer'),
    pytest.param(_context_complete_reviewer, id='context_complete_reviewer'),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(('build', 'audience', 'heading_level'), ACTION_SLOTTED)
async def test_the_rulings_section_sits_directly_before_the_action_heading_at_its_level(
    briefing, build: Build, audience: RulingAudience, heading_level: int,
):
    prompt = await build(briefing, rulings=RULINGS)

    section = render_rulings_section(RULINGS, audience, heading_level=heading_level)
    assert prompt.count(section) == 1
    assert section + '#' * heading_level + ' Action\n' in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize('build', REVIEWERS)
async def test_a_reviewer_is_shown_the_review_section(briefing, build: Build):
    prompt = await build(briefing, rulings=RULINGS)

    assert prompt.count(render_rulings_section(RULINGS, RulingAudience.REVIEW)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('build', REVIEWERS)
async def test_a_reviewer_s_open_info_notes_stay_after_the_rulings(briefing, build: Build):
    prompt = await build(briefing, rulings=RULINGS, open_info_notes=[NOTE])

    section = render_rulings_section(RULINGS, RulingAudience.REVIEW)
    assert prompt.index(section) < prompt.index(OPEN_INFO_NOTES_HEADING)


@pytest.mark.asyncio
@pytest.mark.parametrize(('build', 'audience', 'heading_level'), ACTION_SLOTTED)
async def test_no_rulings_leave_nothing_in_the_action_slot(
    briefing, build: Build, audience: RulingAudience, heading_level: int,
):
    with_rulings = await build(briefing, rulings=RULINGS)

    section = render_rulings_section(RULINGS, audience, heading_level=heading_level)
    assert with_rulings.replace(section, '') == await build(briefing, rulings=())


@pytest.mark.asyncio
@pytest.mark.parametrize('build', REVIEWERS)
async def test_no_rulings_append_nothing_to_a_reviewer_prompt(briefing, build: Build):
    with_rulings = await build(briefing, rulings=RULINGS)

    section = render_rulings_section(RULINGS, RulingAudience.REVIEW)
    assert with_rulings == await build(briefing, rulings=()) + '\n' + section


@pytest.mark.asyncio
async def test_the_replay_framed_architect_forwards_rulings(briefing):
    framed = ReplayFramedBriefingAssembler(briefing.config, base_commit='0123456789abcdef')

    prompt = await framed.build_architect_prompt(TASK, context='CTX', rulings=RULINGS)

    assert render_rulings_section(RULINGS, RulingAudience.PLAN) in prompt
