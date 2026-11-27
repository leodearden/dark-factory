"""Tests for ``orchestrator.escalation_rulings``: the rulings a dispatched agent is briefed with."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from escalation.models import (
    EVAL_LANE_CONTAINMENT_RESOLVER,
    NOTE_TO_FILE_CLASS,
    PRE_SUBMIT_CHECK_RESOLVER,
    Escalation,
    cascade_resolved_by,
)
from escalation.note_to_file import NOTE_TO_FILE_RESOLUTION, NOTE_TO_FILE_RESOLVER

from orchestrator.escalation_rulings import (
    RULING_INSTRUCTIONS,
    RULINGS_LIMIT,
    RULINGS_TITLE,
    Ruling,
    RulingAudience,
    plan_confirmed_at,
    render_rulings_section,
    ruling_resolution,
    select_rulings,
)
from orchestrator.resume_resolution import select_carried_resolutions


def _esc(
    esc_id: str,
    *,
    status: str = 'resolved',
    resolution: str | None = 'KEEP the solver wrap',
    resolved_at: str | None = '2026-08-01T00:00:00+00:00',
    summary: str = 'synthetic blocker',
    resolved_by: str | None = 'interactive',
    resolution_action: str | None = 'resume',
    level: int = 0,
    severity: str = 'blocking',
    note_to_file: bool = False,
    resolution_class: str | None = None,
) -> Escalation:
    return Escalation(
        id=esc_id,
        task_id='4764',
        agent_role='implementer',
        severity=severity,
        category='design_concern',
        summary=summary,
        status=status,
        resolution=resolution,
        resolved_at=resolved_at,
        resolved_by=resolved_by,
        resolution_action=resolution_action,
        level=level,
        note_to_file=note_to_file,
        resolution_class=resolution_class,
    )


_RECORD_SHAPES = {
    'resolved_with_text': (_esc('esc-4764-1'), True),
    'dismissed_with_text': (_esc('esc-4764-2', status='dismissed', resolution='router close'), False),
    'pending': (_esc('esc-4764-3', status='pending', resolution=None, resolved_at=None), False),
    'resolved_without_text': (_esc('esc-4764-4', resolution=None), False),
    'note_to_file': (
        _esc(
            'esc-4764-5', severity='info', note_to_file=True, resolution_class=NOTE_TO_FILE_CLASS,
            resolved_by=NOTE_TO_FILE_RESOLVER, resolution=NOTE_TO_FILE_RESOLUTION,
            resolution_action='close_only',
        ),
        False,
    ),
    'eval_lane_contained': (
        _esc(
            'esc-4764-6', resolved_by=EVAL_LANE_CONTAINMENT_RESOLVER,
            resolution=(
                'contained: eval-lane artifact (fixture-task-id:df_task_2430_adv_plan); not a '
                'production signal, the eval measurement is scored from the cell result artifacts'
            ),
            resolution_class='benign', resolution_action='close_only',
        ),
        False,
    ),
    'pre_submit_auto_closed': (
        _esc(
            'esc-4764-7', resolved_by=PRE_SUBMIT_CHECK_RESOLVER,
            resolution='auto-resolved: task already terminal (status=done)',
            resolution_class='benign',
        ),
        False,
    ),
    'note_to_file_flag_alone': (_esc('esc-4764-8', note_to_file=True), False),
    'note_to_file_class_alone': (_esc('esc-4764-9', resolution_class=NOTE_TO_FILE_CLASS), False),
}


def _ids(rulings: tuple[Ruling, ...]) -> list[str]:
    return [r.escalation_id for r in rulings]


def _is_ruling(record: Escalation) -> bool:
    return ruling_resolution(record) is not None


class TestRulingResolution:
    @pytest.mark.parametrize(('record', 'expected'), list(_RECORD_SHAPES.values()), ids=list(_RECORD_SHAPES))
    def test_a_ruling_is_a_resolved_record_with_text_not_closed_at_filing(self, record, expected):
        assert _is_ruling(record) is expected

    @pytest.mark.parametrize(('record', 'expected'), list(_RECORD_SHAPES.values()), ids=list(_RECORD_SHAPES))
    def test_resume_carries_a_record_iff_it_is_a_ruling(self, record, expected):
        carried = select_carried_resolutions([record], resolved_before_round=frozenset())
        assert bool(carried) is _is_ruling(record) is expected


class TestSelectRulings:
    def test_every_ruling_carries_its_record_fields(self):
        record = _esc(
            'esc-4764-7', level=2, summary='keep or drop the wrap?',
            resolution='KEEP it', resolved_at='2026-08-02T03:04:05+00:00',
            resolved_by='interactive', resolution_action='resume',
        )

        assert select_rulings([record], since=None) == (
            Ruling(
                escalation_id='esc-4764-7', level=2, summary='keep or drop the wrap?',
                resolution='KEEP it', resolved_at='2026-08-02T03:04:05+00:00',
                resolved_by='interactive', resolution_action='resume',
            ),
        )

    def test_non_rulings_are_never_selected(self):
        records = [record for record, _ in _RECORD_SHAPES.values()]

        assert _ids(select_rulings(records, since=None)) == ['esc-4764-1']

    def test_orders_oldest_first_then_by_id_regardless_of_input_order(self):
        records = [
            _esc('esc-4764-9', resolved_at='2026-08-03T00:00:00+00:00'),
            _esc('esc-4764-5', resolved_at='2026-08-01T00:00:00+00:00'),
            _esc('esc-4764-8', resolved_at='2026-08-02T00:00:00+00:00'),
            _esc('esc-4764-6', resolved_at='2026-08-02T00:00:00+00:00'),
        ]

        assert _ids(select_rulings(records, since=None)) == [
            'esc-4764-5', 'esc-4764-6', 'esc-4764-8', 'esc-4764-9',
        ]

    def test_since_keeps_only_rulings_resolved_strictly_after_it(self):
        since = datetime(2026, 8, 2, tzinfo=UTC)
        records = [
            _esc('esc-4764-1', resolved_at='2026-08-01T00:00:00+00:00'),
            _esc('esc-4764-2', resolved_at='2026-08-02T00:00:00+00:00'),
            _esc('esc-4764-3', resolved_at='2026-08-03T00:00:00+00:00'),
        ]

        assert _ids(select_rulings(records, since=since)) == ['esc-4764-3']

    @pytest.mark.parametrize('resolved_at', [None, 'not-a-timestamp'])
    def test_since_keeps_a_ruling_whose_resolution_time_is_unknown(self, resolved_at):
        since = datetime(2099, 1, 1, tzinfo=UTC)

        selected = select_rulings([_esc('esc-4764-1', resolved_at=resolved_at)], since=since)

        assert _ids(selected) == ['esc-4764-1']

    def test_a_naive_resolution_time_is_read_as_utc(self):
        since = datetime(2026, 8, 2, 12, tzinfo=UTC)
        records = [
            _esc('esc-4764-1', resolved_at='2026-08-02T11:00:00'),
            _esc('esc-4764-2', resolved_at='2026-08-02T13:00:00'),
        ]

        assert _ids(select_rulings(records, since=since)) == ['esc-4764-2']

    def test_a_z_suffixed_resolution_time_parses(self):
        since = datetime(2026, 8, 2, 12, tzinfo=UTC)
        records = [
            _esc('esc-4764-1', resolved_at='2026-08-02T11:00:00Z'),
            _esc('esc-4764-2', resolved_at='2026-08-02T13:00:00Z'),
        ]

        assert _ids(select_rulings(records, since=since)) == ['esc-4764-2']

    def test_a_cascade_member_is_dropped_when_its_parent_is_selected(self):
        records = [
            _esc('esc-4764-1', level=1, resolved_by=cascade_resolved_by('esc-L2-4764-1')),
            _esc('esc-L2-4764-1', level=2),
        ]

        assert _ids(select_rulings(records, since=None)) == ['esc-L2-4764-1']

    def test_a_cascade_member_is_kept_when_its_parent_is_not_selected(self):
        records = [_esc('esc-4764-1', level=1, resolved_by=cascade_resolved_by('esc-L2-9999-1'))]

        assert _ids(select_rulings(records, since=None)) == ['esc-4764-1']


class TestPlanConfirmedAt:
    def test_reads_the_finalized_stamp_as_an_aware_datetime(self):
        plan = {'_finalized_at': '2026-11-10T20:26:26.271694+00:00'}

        assert plan_confirmed_at(plan) == datetime(2026, 11, 10, 20, 26, 26, 271694, tzinfo=UTC)

    @pytest.mark.parametrize(
        'plan',
        [{}, {'_created_at': '2026-11-10T20:26:26+00:00'}, {'_finalized_at': 'never'}],
        ids=['empty', 'no_finalized_stamp', 'unparseable'],
    )
    def test_is_none_without_a_parseable_finalized_stamp(self, plan):
        assert plan_confirmed_at(plan) is None


def _ruling(
    esc_id: str = 'esc-4764-1',
    *,
    level: int = 2,
    summary: str = 'keep or drop the solver wrap?',
    resolution: str = 'KEEP the solver wrap',
    resolved_at: str | None = '2026-08-02T00:00:00+00:00',
    resolved_by: str | None = 'interactive',
    resolution_action: str | None = 'resume',
) -> Ruling:
    return Ruling(
        escalation_id=esc_id, level=level, summary=summary, resolution=resolution,
        resolved_at=resolved_at, resolved_by=resolved_by, resolution_action=resolution_action,
    )


def _entry_headings(section: str) -> list[str]:
    return [line for line in section.splitlines() if line.startswith('## ')]


class TestRenderRulingsSection:
    @pytest.mark.parametrize('audience', list(RulingAudience))
    def test_no_rulings_render_nothing(self, audience):
        assert render_rulings_section((), audience) == ''

    def test_opens_with_the_heading_and_has_one_entry_per_ruling(self):
        rulings = (_ruling('esc-4764-1', level=1, summary='first'), _ruling('esc-4764-2', summary='second'))

        section = render_rulings_section(rulings, RulingAudience.EXECUTE)

        assert section.startswith(f'# {RULINGS_TITLE}\n')
        assert _entry_headings(section) == [
            '## `esc-4764-1` (level 1): first',
            '## `esc-4764-2` (level 2): second',
        ]

    def test_an_entry_shows_its_attribution(self):
        section = render_rulings_section(
            (_ruling(resolved_by='interactive', resolved_at='2026-08-02T00:00:00+00:00', resolution_action='resume'),),
            RulingAudience.EXECUTE,
        )

        assert '- resolved by: interactive\n' in section
        assert '- resolved at: 2026-08-02T00:00:00+00:00\n' in section
        assert '- action: resume\n' in section

    def test_missing_attribution_reads_unknown(self):
        section = render_rulings_section(
            (_ruling(resolved_by=None, resolved_at=None, resolution_action=None),),
            RulingAudience.EXECUTE,
        )

        assert '- resolved by: unknown\n' in section
        assert '- resolved at: unknown\n' in section
        assert '- action: unknown\n' in section

    def test_resolution_text_is_blockquoted_line_by_line(self):
        resolution = 'KEEP the solver wrap.\n# Not a heading\nWiden scope if needed.'

        section = render_rulings_section((_ruling(resolution=resolution),), RulingAudience.EXECUTE)

        assert '> KEEP the solver wrap.\n> # Not a heading\n> Widen scope if needed.\n' in section
        headings = [line for line in section.splitlines() if line.startswith('#')]
        assert headings == [f'# {RULINGS_TITLE}', '## `esc-4764-1` (level 2): keep or drop the solver wrap?']

    def test_an_empty_resolution_is_quoted_as_none(self):
        section = render_rulings_section((_ruling(resolution=''),), RulingAudience.EXECUTE)

        assert '\n> (none)\n' in section

    def test_a_deeper_heading_level_nests_the_section_and_its_entries(self):
        section = render_rulings_section((_ruling(),), RulingAudience.PLAN, heading_level=2)

        headings = [line for line in section.splitlines() if line.startswith('#')]
        assert headings == [f'## {RULINGS_TITLE}', '### `esc-4764-1` (level 2): keep or drop the solver wrap?']

    @pytest.mark.parametrize('heading_level', [0, 6])
    def test_a_heading_level_its_entries_cannot_nest_under_is_refused(self, heading_level):
        with pytest.raises(ValueError, match='heading_level'):
            render_rulings_section((_ruling(),), RulingAudience.PLAN, heading_level=heading_level)

    def test_a_multi_line_summary_is_collapsed_into_its_heading(self):
        section = render_rulings_section(
            (_ruling(summary='keep or drop\n# the solver wrap?'),), RulingAudience.EXECUTE,
        )

        assert _entry_headings(section) == ['## `esc-4764-1` (level 2): keep or drop # the solver wrap?']

    def test_every_audience_has_its_own_instruction(self):
        assert set(RULING_INSTRUCTIONS) == set(RulingAudience)
        assert len(set(RULING_INSTRUCTIONS.values())) == len(RulingAudience)
        assert '`escalate_blocker`' in RULING_INSTRUCTIONS[RulingAudience.EXECUTE]
        assert '`escalate_blocker`' in RULING_INSTRUCTIONS[RulingAudience.PLAN_AND_EXECUTE]

    @pytest.mark.parametrize('audience', list(RulingAudience))
    def test_the_section_carries_its_audience_instruction(self, audience):
        section = render_rulings_section((_ruling(),), audience)

        assert RULING_INSTRUCTIONS[audience] in section
        others = set(RulingAudience) - {audience}
        assert not any(RULING_INSTRUCTIONS[other] in section for other in others)

    def test_only_the_most_recent_rulings_are_listed_and_older_ids_named(self):
        rulings = tuple(_ruling(f'esc-4764-{n:02d}', summary=f'ruling {n}') for n in range(1, RULINGS_LIMIT + 3))

        section = render_rulings_section(rulings, RulingAudience.REVIEW)

        listed = rulings[-RULINGS_LIMIT:]
        omitted = rulings[:-RULINGS_LIMIT]
        assert _entry_headings(section) == [
            f'## `{r.escalation_id}` (level 2): {r.summary}' for r in listed
        ]
        overflow = [line for line in section.splitlines() if all(f'`{r.escalation_id}`' in line for r in omitted)]
        assert len(overflow) == 1
        assert not overflow[0].startswith('#')

    @pytest.mark.parametrize('audience', list(RulingAudience))
    def test_the_section_ends_with_a_blank_line(self, audience):
        section = render_rulings_section((_ruling(),), audience)

        assert section.endswith('\n\n')
        assert not section.endswith('\n\n\n')
