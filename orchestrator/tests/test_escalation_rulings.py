"""Tests for ``orchestrator.escalation_rulings``: the rulings a dispatched agent is briefed with."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from escalation.models import Escalation, cascade_resolved_by

from orchestrator.escalation_rulings import (
    Ruling,
    is_ruling,
    plan_confirmed_at,
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
) -> Escalation:
    return Escalation(
        id=esc_id,
        task_id='4764',
        agent_role='implementer',
        severity='blocking',
        category='design_concern',
        summary=summary,
        status=status,
        resolution=resolution,
        resolved_at=resolved_at,
        resolved_by=resolved_by,
        resolution_action=resolution_action,
        level=level,
    )


_RECORD_SHAPES = {
    'resolved_with_text': (_esc('esc-4764-1'), True),
    'dismissed_with_text': (_esc('esc-4764-2', status='dismissed', resolution='router close'), False),
    'pending': (_esc('esc-4764-3', status='pending', resolution=None, resolved_at=None), False),
    'resolved_without_text': (_esc('esc-4764-4', resolution=None), False),
}


def _ids(rulings: tuple[Ruling, ...]) -> list[str]:
    return [r.escalation_id for r in rulings]


class TestIsRuling:
    @pytest.mark.parametrize(('record', 'expected'), list(_RECORD_SHAPES.values()), ids=list(_RECORD_SHAPES))
    def test_only_a_resolved_record_with_resolution_text_is_a_ruling(self, record, expected):
        assert is_ruling(record) is expected

    @pytest.mark.parametrize(('record', 'expected'), list(_RECORD_SHAPES.values()), ids=list(_RECORD_SHAPES))
    def test_resume_carries_a_record_iff_it_is_a_ruling(self, record, expected):
        carried = select_carried_resolutions([record], resolved_before_round=frozenset())
        assert bool(carried) is is_ruling(record) is expected


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
