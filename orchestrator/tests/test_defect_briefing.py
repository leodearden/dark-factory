"""Declared defects in the task-preamble briefings (PRD ``plans/defect-tests-prd.md`` decision 17).

The renderer is driven over records built through the real ``shared.defects``
model and ref minters, so no enum value or ref is spelled here. The wiring half
renders, through the real ``BriefingAssembler``, the four briefings that open
with ``_task_preamble`` and the context-complete reviewer's, whose task block
``_format_task`` also renders. Each lists the declared refs and never restates
``DEFECT_TEST_RULE``, whose one site per dispatch is the system prompt.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _briefing_helpers import _mcp_search_envelope, memory_transport
from shared.defects import (
    DEFECT_TEST_RETIREMENTS_METADATA_KEY,
    DEFECTS_METADATA_KEY,
    DeclareClosure,
    DefectDeclaration,
    DefectOrigin,
    DefectStatus,
    Exemption,
    ExemptionReason,
    Retirement,
    RetirementCode,
    RetirementReason,
    RuledBy,
    Ruling,
    RulingSubject,
    Withdrawal,
    WithdrawalCode,
    WithdrawalReason,
    apply_event,
    mint_human_ref,
    mint_redmain_ref,
    mint_review_ref,
)
from shared.governed_exceptions import Debt, TaskRef, TicketRef

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.defect_briefing import (
    CLOSURES_AWAITING_RULING_HEADING,
    DECLARED_DEFECTS_HEADING,
    DEFECT_TEST_RETIREMENTS_HEADING,
    format_closures_awaiting_ruling,
    format_declared_defects,
    format_defect_test_retirements,
)
from orchestrator.agents.review_briefing import (
    ChangedFile,
    ReviewChange,
    ReviewSubject,
    review_action_section,
)
from orchestrator.agents.roles import DEFECT_TEST_RULE
from orchestrator.agents.steward_digest import EarlierAttempts

_DECLARED_AT = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
_WARNING = '⚠'


def _declaration(ref: str, origin: DefectOrigin, summary: str) -> DefectDeclaration:
    return DefectDeclaration(
        ref=ref,
        origin=origin,
        summary=summary,
        declared_at=_DECLARED_AT,
        declared_by='test',
        status=DefectStatus.OPEN,
    )


_HUMAN_SUMMARY = 'Widget count is off by one'
_HUMAN = _declaration(
    mint_human_ref(_DECLARED_AT, _HUMAN_SUMMARY), DefectOrigin.HUMAN, _HUMAN_SUMMARY
)
_REVIEW = _declaration(
    mint_review_ref(7047, 'widget.py', 'retry ignores the timeout'),
    DefectOrigin.REVIEW,
    'Retry ignores the timeout',
)
_DECLARED = (_HUMAN, _REVIEW)


def _wire(*declarations: DefectDeclaration) -> list[dict[str, Any]]:
    return [declaration.model_dump(mode='json') for declaration in declarations]


def _line_naming(rendered: str, ref: str) -> str:
    lines = [line for line in rendered.splitlines() if ref in line]
    assert len(lines) == 1, f'expected one line naming {ref!r}, got {lines!r}'
    return lines[0]


def _warnings(rendered: str) -> list[str]:
    return [line for line in rendered.splitlines() if line.startswith(_WARNING)]


class TestFormatDeclaredDefects:
    def test_each_entry_renders_its_ref_origin_status_and_summary_on_one_line(self):
        rendered = format_declared_defects({DEFECTS_METADATA_KEY: _wire(*_DECLARED)})

        for declaration in _DECLARED:
            line = _line_naming(rendered, declaration.ref)
            assert declaration.summary in line
            # Every ref opens with its origin, so origin and status are sought beside it.
            rest = line.replace(declaration.ref, '').replace(declaration.summary, '')
            assert str(declaration.origin) in rest
            assert str(declaration.status) in rest

    def test_the_block_opens_with_its_heading(self):
        rendered = format_declared_defects({DEFECTS_METADATA_KEY: _wire(_HUMAN)})

        assert DECLARED_DEFECTS_HEADING in rendered

    @pytest.mark.parametrize(
        'metadata',
        [{}, {DEFECTS_METADATA_KEY: None}, {DEFECTS_METADATA_KEY: []}],
        ids=['absent', 'none', 'empty'],
    )
    def test_nothing_declared_renders_nothing(self, metadata):
        assert format_declared_defects(metadata) == ''

    @pytest.mark.parametrize('count', [1, 2])
    def test_malformed_entries_are_counted_not_rendered(self, count):
        malformed = [{'ref': f'not-a-ref-{n}'} for n in range(count)]

        rendered = format_declared_defects({DEFECTS_METADATA_KEY: [*_wire(_HUMAN), *malformed]})

        assert _HUMAN.ref in rendered
        assert 'not-a-ref' not in rendered
        warnings = _warnings(rendered)
        assert len(warnings) == 1
        assert str(count) in warnings[0]

    @pytest.mark.parametrize(
        'value', [{'ref': 'not-a-ref'}, 'not-a-list'], ids=['dict', 'str'],
    )
    def test_a_non_list_value_is_reported_and_never_iterated(self, value):
        rendered = format_declared_defects({DEFECTS_METADATA_KEY: value})

        assert DECLARED_DEFECTS_HEADING in rendered
        warnings = _warnings(rendered)
        assert len(warnings) == 1
        assert type(value).__name__ in warnings[0]
        assert [line for line in rendered.splitlines() if line.startswith('- ')] == []

    def test_a_multi_line_summary_renders_on_one_line(self):
        summary = 'Retry ignores\nthe timeout\n  entirely'
        declaration = _declaration(
            mint_review_ref(7047, 'retry.py', summary), DefectOrigin.REVIEW, summary
        )

        rendered = format_declared_defects({DEFECTS_METADATA_KEY: _wire(declaration)})

        assert ' '.join(summary.split()) in _line_naming(rendered, declaration.ref)


_TASK: dict[str, Any] = {
    'id': '7047',
    'title': 'Fix the widget',
    'description': 'The widget miscounts.',
    'status': 'in-progress',
    'dependencies': [],
    'metadata': {
        'files': ['widget_footprint_guess.py'],
        DEFECTS_METADATA_KEY: _wire(*_DECLARED),
    },
}

_ESCALATION = {
    'id': 'esc-7047-1',
    'category': 'task_failure',
    'summary': 'Execution iterations exhausted',
    'detail': 'Agent ran out of iterations.',
    'severity': 'blocking',
}

_Builder = Callable[[BriefingAssembler, Path], Awaitable[str]]


async def _architect(assembler: BriefingAssembler, worktree: Path) -> str:
    return await assembler.build_architect_prompt(_TASK, worktree=None, context='')


async def _replan(assembler: BriefingAssembler, worktree: Path) -> str:
    return await assembler.build_replan_prompt(_TASK, 'FEEDBACK', context='')


async def _simple_task(assembler: BriefingAssembler, worktree: Path) -> str:
    return await assembler.build_simple_task_prompt(_TASK, worktree=None, context='')


async def _steward_initial(assembler: BriefingAssembler, worktree: Path) -> str:
    with memory_transport(AsyncMock(return_value=_mcp_search_envelope([]))):
        return await assembler.build_steward_initial_prompt(
            task=_TASK,
            escalation=_ESCALATION,
            pending_escalations=[],
            worktree=worktree,
            handled=(),
            earlier_attempts=EarlierAttempts(),
        )


async def _context_complete_reviewer(assembler: BriefingAssembler, worktree: Path) -> str:
    subject = ReviewSubject(
        task=_TASK,
        plan=None,
        plan_absent_reason='no plan in this test',
        change=ReviewChange(
            base_sha='0123456789abcdef0123456789abcdef01234567',
            diff='diff --git a/widget.py b/widget.py\n',
            files=(ChangedFile('widget.py', 1, 0),),
        ),
        conventions=(),
    )
    return await assembler.build_context_complete_reviewer_prompt(
        'reviewer_comprehensive', subject, context='',
    )


_TASK_BLOCK_BUILDERS: dict[str, _Builder] = {
    'architect': _architect,
    'replan': _replan,
    'simple_task': _simple_task,
    'steward_initial': _steward_initial,
    'context_complete_reviewer': _context_complete_reviewer,
}


class TestTaskBlockBriefingsListDeclaredDefects:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('builder', sorted(_TASK_BLOCK_BUILDERS))
    async def test_every_declared_ref_is_listed(
        self, briefing: BriefingAssembler, tmp_path: Path, builder: str,
    ):
        prompt = await _TASK_BLOCK_BUILDERS[builder](briefing, tmp_path)

        missing = [d.ref for d in _DECLARED if d.ref not in prompt]
        assert missing == [], f'the {builder} briefing omits declared defects {missing}'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('builder', sorted(_TASK_BLOCK_BUILDERS))
    async def test_the_rule_is_not_restated(
        self, briefing: BriefingAssembler, tmp_path: Path, builder: str,
    ):
        prompt = await _TASK_BLOCK_BUILDERS[builder](briefing, tmp_path)

        assert DEFECT_TEST_RULE not in prompt, (
            f'the {builder} briefing restates DEFECT_TEST_RULE; its one site per '
            'dispatch is the system prompt (PRD decision 17).'
        )

    @pytest.mark.asyncio
    async def test_the_architects_file_suppression_leaves_the_list(
        self, briefing: BriefingAssembler, tmp_path: Path,
    ):
        prompt = await _architect(briefing, tmp_path)

        # The architect hides the queue-time file guess (C-A1), not the charter.
        assert 'widget_footprint_guess.py' not in prompt
        assert all(d.ref in prompt for d in _DECLARED)


_TICKET_ID = 'tkt_0RW782Z6RY3X8W8Q3NX8J1EKD0'


def _awaiting_ruling(declaration: DefectDeclaration, closure: Exemption | Withdrawal) -> DefectDeclaration:
    return apply_event(declaration, DeclareClosure(ref=declaration.ref, closure=closure))


def _exemption(owner: TaskRef | TicketRef) -> Exemption:
    return Exemption(reason=ExemptionReason.NEEDS_EXTERNAL_SYSTEM, owner=Debt(owner))


_EXEMPT_SUMMARY = 'Flaky against the live payment gateway'
_EXEMPT = _awaiting_ruling(
    _declaration(
        mint_review_ref(7050, 'pay.py', 'gateway retry'), DefectOrigin.REVIEW, _EXEMPT_SUMMARY,
    ),
    _exemption(TicketRef(_TICKET_ID)),
)
_WITHDRAW_SUMMARY = 'Same miscount as the human report'
_WITHDRAW = _awaiting_ruling(
    _declaration(
        mint_review_ref(7050, 'widget.py', 'count off by one'), DefectOrigin.REVIEW, _WITHDRAW_SUMMARY,
    ),
    Withdrawal(reason=WithdrawalReason(reason=WithdrawalCode.DUPLICATE_OF, ref=_HUMAN.ref)),
)


class TestFormatClosuresAwaitingRuling:
    def test_only_entries_awaiting_a_ruling_are_listed(self):
        rendered = format_closures_awaiting_ruling(
            {DEFECTS_METADATA_KEY: _wire(_REVIEW, _EXEMPT, _WITHDRAW)},
        )

        assert CLOSURES_AWAITING_RULING_HEADING in rendered
        assert _REVIEW.ref not in rendered
        assert _EXEMPT.ref in rendered
        assert _WITHDRAW.ref in rendered

    def test_an_exemption_names_its_kind_reason_owner_and_summary(self):
        line = _line_naming(
            format_closures_awaiting_ruling({DEFECTS_METADATA_KEY: _wire(_EXEMPT)}), _EXEMPT.ref,
        )

        assert 'exemption' in line
        assert str(ExemptionReason.NEEDS_EXTERNAL_SYSTEM) in line
        assert _TICKET_ID in line
        assert _EXEMPT_SUMMARY in line

    def test_a_task_owner_is_named_as_that_task(self):
        owed = _awaiting_ruling(_REVIEW, _exemption(TaskRef(4242)))

        line = _line_naming(format_closures_awaiting_ruling({DEFECTS_METADATA_KEY: _wire(owed)}), owed.ref)

        assert 'task 4242' in line

    def test_a_duplicate_withdrawal_names_the_ref_it_duplicates(self):
        line = _line_naming(
            format_closures_awaiting_ruling({DEFECTS_METADATA_KEY: _wire(_WITHDRAW)}), _WITHDRAW.ref,
        )

        assert 'withdrawal' in line
        assert str(WithdrawalCode.DUPLICATE_OF) in line
        assert _HUMAN.ref in line
        assert _WITHDRAW_SUMMARY in line

    def test_the_block_names_the_closure_ruling_subject(self):
        rendered = format_closures_awaiting_ruling({DEFECTS_METADATA_KEY: _wire(_EXEMPT)})

        assert f'`{RulingSubject.CLOSURE}`' in rendered
        assert 'defect_rulings' in rendered

    @pytest.mark.parametrize(
        'metadata',
        [{}, {DEFECTS_METADATA_KEY: None}, {DEFECTS_METADATA_KEY: []}, {DEFECTS_METADATA_KEY: _wire(_HUMAN)}],
        ids=['absent', 'none', 'empty', 'nothing-pending'],
    )
    def test_nothing_awaiting_a_ruling_renders_nothing(self, metadata):
        assert format_closures_awaiting_ruling(metadata) == ''


_RETIRED_REF = mint_redmain_ref('0123456789abcdef')
_SUCCESSOR_REF = mint_redmain_ref('fedcba9876543210')
_PENDING_RETIREMENT = Retirement(
    ref=_RETIRED_REF, reason=RetirementReason(reason=RetirementCode.SUPERSEDED_BY, ref=_SUCCESSOR_REF),
)
_ACCEPTED_RETIREMENT = Retirement(
    ref=_HUMAN.ref,
    reason=RetirementReason(reason=RetirementCode.FEATURE_REMOVED),
    ruling=Ruling(subject=RulingSubject.RETIREMENT, accepted=True, ruled_by=RuledBy.OPERATOR),
)


def _retirements(*records: Retirement) -> dict[str, Any]:
    return {DEFECT_TEST_RETIREMENTS_METADATA_KEY: [record.model_dump(mode='json') for record in records]}


class TestFormatDefectTestRetirements:
    def test_each_retirement_names_its_ref_reason_and_state_on_one_line(self):
        rendered = format_defect_test_retirements(_retirements(_PENDING_RETIREMENT, _ACCEPTED_RETIREMENT))

        assert DEFECT_TEST_RETIREMENTS_HEADING in rendered
        for record in (_PENDING_RETIREMENT, _ACCEPTED_RETIREMENT):
            line = _line_naming(rendered, record.ref)
            assert str(record.reason.reason) in line
            assert str(record.state) in line

    def test_a_superseded_retirement_names_its_successor(self):
        rendered = format_defect_test_retirements(_retirements(_PENDING_RETIREMENT))

        assert _SUCCESSOR_REF in _line_naming(rendered, _RETIRED_REF)

    def test_the_block_names_the_retirement_ruling_subject(self):
        rendered = format_defect_test_retirements(_retirements(_PENDING_RETIREMENT))

        assert f'`{RulingSubject.RETIREMENT}`' in rendered
        assert 'defect_rulings' in rendered

    @pytest.mark.parametrize(
        'metadata',
        [{}, {DEFECT_TEST_RETIREMENTS_METADATA_KEY: None}, {DEFECT_TEST_RETIREMENTS_METADATA_KEY: []}],
        ids=['absent', 'none', 'empty'],
    )
    def test_no_retirement_renders_nothing(self, metadata):
        assert format_defect_test_retirements(metadata) == ''

    def test_malformed_entries_are_counted_not_rendered(self):
        metadata = _retirements(_PENDING_RETIREMENT)
        metadata[DEFECT_TEST_RETIREMENTS_METADATA_KEY].append({'ref': 'not-a-ref'})

        rendered = format_defect_test_retirements(metadata)

        assert _RETIRED_REF in rendered
        assert 'not-a-ref' not in rendered
        (warning,) = _warnings(rendered)
        assert '1' in warning

    def test_a_non_list_value_is_reported_and_never_iterated(self):
        rendered = format_defect_test_retirements({DEFECT_TEST_RETIREMENTS_METADATA_KEY: 'not-a-list'})

        assert DEFECT_TEST_RETIREMENTS_HEADING in rendered
        (warning,) = _warnings(rendered)
        assert 'str' in warning


_RECORDED_TASK: dict[str, Any] = {
    **_TASK,
    'metadata': {
        'files': ['widget_footprint_guess.py'],
        DEFECTS_METADATA_KEY: _wire(_REVIEW, _EXEMPT),
        **_retirements(_PENDING_RETIREMENT),
    },
}
_RECORD_HEADINGS = (
    DECLARED_DEFECTS_HEADING, CLOSURES_AWAITING_RULING_HEADING, DEFECT_TEST_RETIREMENTS_HEADING,
)
_REVIEWER = 'reviewer_comprehensive'
_UNRECORDED_TASK: dict[str, Any] = {**_TASK, 'metadata': {'files': ['widget_footprint_guess.py']}}


def _action_section_close(reviewer_type: str) -> str:
    """What every reviewer briefing's ``# Action`` section ends with, whatever its review step says."""
    step = '<review step>'
    section = review_action_section(reviewer_type, step)
    return section[section.index(step) + len(step):]


def _subject(task: dict[str, Any]) -> ReviewSubject:
    return ReviewSubject(
        task=task,
        plan=None,
        plan_absent_reason='no plan in this test',
        change=ReviewChange(
            base_sha='0123456789abcdef0123456789abcdef01234567',
            diff='diff --git a/widget.py b/widget.py\n',
            files=(ChangedFile('widget.py', 1, 0),),
        ),
        conventions=(),
    )


class TestReviewerBriefingsShowTheRecordsToRuleOn:
    @pytest.mark.asyncio
    async def test_the_diff_only_reviewer_gets_the_declared_defects_and_both_ruling_blocks(
        self, briefing: BriefingAssembler,
    ):
        prompt = await briefing.build_reviewer_prompt(
            'reviewer_comprehensive', 'diff --git a/widget.py b/widget.py\n', context='', task=_RECORDED_TASK,
        )

        for heading in _RECORD_HEADINGS:
            assert heading in prompt
        assert _EXEMPT.ref in prompt
        assert _RETIRED_REF in prompt

    @pytest.mark.asyncio
    async def test_the_context_complete_reviewer_gets_both_ruling_blocks_beside_its_task_block(
        self, briefing: BriefingAssembler,
    ):
        prompt = await briefing.build_context_complete_reviewer_prompt(
            'reviewer_comprehensive', _subject(_RECORDED_TASK), context='',
        )

        assert CLOSURES_AWAITING_RULING_HEADING in prompt
        assert DEFECT_TEST_RETIREMENTS_HEADING in prompt
        assert prompt.count(DECLARED_DEFECTS_HEADING) == 1

    @pytest.mark.asyncio
    async def test_a_task_without_defect_records_ends_the_diff_only_prompt_at_its_action_section(
        self, briefing: BriefingAssembler,
    ):
        prompt = await briefing.build_reviewer_prompt(
            _REVIEWER, 'diff --git a/widget.py b/widget.py\n', context='', task=_UNRECORDED_TASK,
        )

        assert prompt.endswith(_action_section_close(_REVIEWER))
        assert [heading for heading in _RECORD_HEADINGS if heading in prompt] == []

    @pytest.mark.asyncio
    async def test_a_task_without_defect_records_ends_the_context_complete_prompt_at_its_action_section(
        self, briefing: BriefingAssembler,
    ):
        prompt = await briefing.build_context_complete_reviewer_prompt(
            _REVIEWER, _subject(_UNRECORDED_TASK), context='',
        )

        assert prompt.endswith(_action_section_close(_REVIEWER))
        assert [heading for heading in _RECORD_HEADINGS if heading in prompt] == []
