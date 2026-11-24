"""Declared defects in the task-preamble briefings (PRD ``plans/defect-tests-prd.md`` decision 17).

The renderer is driven over records built through the real ``shared.defects``
model and ref minters, so no enum value or ref is spelled here. The wiring half
renders the four briefings that open with ``_task_preamble`` through the real
``BriefingAssembler`` and checks that each lists the declared refs and never
restates ``DEFECT_TEST_RULE``, whose one site per dispatch is the system prompt.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _briefing_helpers import (
    _mcp_search_envelope,
    briefing,  # noqa: F401 — re-export: pytest fixture used by test methods
    memory_transport,
)
from shared.defects import (
    DEFECTS_METADATA_KEY,
    DefectDeclaration,
    DefectOrigin,
    DefectStatus,
    mint_human_ref,
    mint_review_ref,
)

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.defect_briefing import DECLARED_DEFECTS_HEADING, format_declared_defects
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
            assert str(declaration.origin) in line
            assert str(declaration.status) in line
            assert declaration.summary in line

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

    def test_a_malformed_entry_is_counted_not_rendered(self):
        rendered = format_declared_defects(
            {DEFECTS_METADATA_KEY: [*_wire(_HUMAN), {'ref': 'not-a-ref'}]}
        )

        assert _HUMAN.ref in rendered
        assert 'not-a-ref' not in rendered
        warnings = _warnings(rendered)
        assert len(warnings) == 1
        assert '1' in warnings[0]

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


_TASK_PREAMBLE_BUILDERS: dict[str, _Builder] = {
    'architect': _architect,
    'replan': _replan,
    'simple_task': _simple_task,
    'steward_initial': _steward_initial,
}


class TestTaskPreambleBriefingsListDeclaredDefects:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('builder', sorted(_TASK_PREAMBLE_BUILDERS))
    async def test_every_declared_ref_is_listed(
        self, briefing: BriefingAssembler, tmp_path: Path, builder: str,
    ):
        prompt = await _TASK_PREAMBLE_BUILDERS[builder](briefing, tmp_path)

        missing = [d.ref for d in _DECLARED if d.ref not in prompt]
        assert missing == [], f'the {builder} briefing omits declared defects {missing}'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('builder', sorted(_TASK_PREAMBLE_BUILDERS))
    async def test_the_rule_is_not_restated(
        self, briefing: BriefingAssembler, tmp_path: Path, builder: str,
    ):
        prompt = await _TASK_PREAMBLE_BUILDERS[builder](briefing, tmp_path)

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
