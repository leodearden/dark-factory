"""The reviewer's notes block: its task's open info escalations, listed for disposition.

plans/info-l0-disposition-router-prd.md D2 (task 5224).  These tests drive the
public seams ``BriefingAssembler.build_reviewer_prompt`` and
``BriefingAssembler.build_context_complete_reviewer_prompt`` with real
``escalation.models.Escalation`` notes and assert what reaches the reviewer.
Every call passes ``context`` so no memory recall is made.
"""

from __future__ import annotations

import pytest
from _briefing_helpers import briefing as briefing
from escalation.disposition import ReviewerDisposition
from escalation.models import Escalation

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.review_briefing import ChangedFile, ReviewChange, ReviewSubject

REVIEWER = 'reviewer_comprehensive'
NOTES_HEADING = '# Implementer notes (open info escalations)'
AMENDMENT_HEADING = '# Amendment Re-Review Scope'
DIFF = (
    'diff --git a/m/x.py b/m/x.py\n--- a/m/x.py\n+++ b/m/x.py\n'
    '@@ -0,0 +1,1 @@\n+x = 1\n'
)
SUGGESTIONS = [{'location': 'm/x.py:1', 'description': 'name the constant'}]


def _note(
    esc_id: str, *, agent_role: str = 'implementer', summary: str = 'helper X is duplicated',
    detail: str = 'Seen while editing foo.py; bar.py carries a copy.',
    suggested_action: str = 'create_followup_task',
    timestamp: str = '2026-10-29T12:00:00+00:00',
) -> Escalation:
    return Escalation(
        id=esc_id, task_id='42', agent_role=agent_role, severity='info',
        category='design_concern', summary=summary, detail=detail,
        suggested_action=suggested_action, timestamp=timestamp,
    )


NOTES = [
    _note('esc-42-1'),
    _note(
        'esc-42-2', agent_role='deep_reviewer', summary='the retry loop swallows timeouts',
        detail='Spotted in retry.py.', suggested_action='no action needed',
        timestamp='2026-10-29T13:30:00+00:00',
    ),
]


def _subject() -> ReviewSubject:
    return ReviewSubject(
        task={'id': '42', 'title': 'T', 'description': 'D'},
        plan=None,
        plan_absent_reason='no plan in this test',
        change=ReviewChange(
            base_sha='0123456789abcdef0123456789abcdef01234567',
            diff=DIFF,
            files=(ChangedFile('m/x.py', 1, 0),),
        ),
        conventions=(),
    )


async def _diff_only(briefing: BriefingAssembler, **kwargs) -> str:
    return await briefing.build_reviewer_prompt(REVIEWER, DIFF, context='', **kwargs)


async def _context_complete(briefing: BriefingAssembler, **kwargs) -> str:
    return await briefing.build_context_complete_reviewer_prompt(
        REVIEWER, _subject(), context='', **kwargs,
    )


BUILDERS = [
    pytest.param(_diff_only, id='diff-only'),
    pytest.param(_context_complete, id='context-complete'),
]


def _assert_lists_every_note(prompt: str) -> None:
    assert prompt.count(NOTES_HEADING) == 1
    block = prompt[prompt.index(NOTES_HEADING):]
    for note in NOTES:
        for value in (
            note.id, note.agent_role, note.timestamp, note.summary, note.suggested_action,
            note.detail,
        ):
            assert value in block, value
    assert 'note_dispositions' in block
    for disposition in ReviewerDisposition:
        assert disposition.value in block


@pytest.mark.asyncio
@pytest.mark.parametrize('build', BUILDERS)
class TestNotesBlock:
    async def test_every_note_is_listed_with_its_fields(self, briefing, build):
        _assert_lists_every_note(await build(briefing, open_info_notes=NOTES))

    @pytest.mark.parametrize('empty', [None, []], ids=['none', 'empty'])
    async def test_no_notes_leave_the_prompt_byte_identical(self, briefing, build, empty):
        without = await build(briefing)

        assert await build(briefing, open_info_notes=empty) == without
        assert NOTES_HEADING not in without

    async def test_the_block_follows_the_amendment_scope(self, briefing, build):
        prompt = await build(briefing, amendment_suggestions=SUGGESTIONS, open_info_notes=NOTES)

        assert AMENDMENT_HEADING in prompt
        _assert_lists_every_note(prompt)
        assert prompt.index(AMENDMENT_HEADING) < prompt.index(NOTES_HEADING)


@pytest.mark.asyncio
class TestNoteDetailCap:
    async def test_a_long_detail_is_cut_at_the_configured_prefix(self, briefing):
        briefing.config.info_l0_note_detail_chars = 40
        detail = ''.join(f'<{n:02d}>' for n in range(25))

        prompt = await _diff_only(briefing, open_info_notes=[_note('esc-42-1', detail=detail)])

        assert detail[:40] in prompt
        assert detail[40:44] not in prompt
        assert '[detail truncated at 40 characters]' in prompt

    async def test_a_detail_within_the_cap_is_shown_whole(self, briefing):
        briefing.config.info_l0_note_detail_chars = 40
        detail = 'x' * 40

        prompt = await _diff_only(briefing, open_info_notes=[_note('esc-42-1', detail=detail)])

        assert detail in prompt
        assert '[detail truncated' not in prompt
