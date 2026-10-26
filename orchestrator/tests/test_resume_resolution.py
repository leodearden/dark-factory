"""Tests for ``orchestrator.resume_resolution``: what a resumed agent is told."""

from __future__ import annotations

import dataclasses
import itertools

import pytest
from escalation.models import Escalation

from orchestrator.resume_resolution import (
    NO_CARRIED_RESOLUTIONS,
    CarriedResolution,
    render_carried_resolutions,
    select_carried_resolutions,
)


def _esc(
    esc_id: str,
    *,
    status: str = 'resolved',
    resolution: str | None = 'do the thing',
    resolved_at: str | None = None,
    timestamp: str = '2026-08-01T00:00:00+00:00',
    summary: str = 'synthetic blocker',
    resolved_by: str | None = 'steward',
    resolution_action: str | None = 'resume',
    severity: str = 'blocking',
    level: int = 0,
) -> Escalation:
    return Escalation(
        id=esc_id,
        task_id='3713',
        agent_role='implementer',
        severity=severity,
        category='task_failure',
        summary=summary,
        timestamp=timestamp,
        status=status,
        resolution=resolution,
        resolved_at=resolved_at,
        resolved_by=resolved_by,
        resolution_action=resolution_action,
        level=level,
    )


class TestSelectCarriedResolutions:
    def test_selects_resolved_records_with_prose_oldest_resolution_first(self):
        later = _esc(
            'esc-3713-1', resolution='later answer',
            resolved_at='2026-08-03T00:00:00+00:00', summary='first filed',
        )
        earlier = _esc(
            'esc-3713-2', resolution='earlier answer',
            resolved_at='2026-08-02T00:00:00+00:00', summary='second filed',
            resolved_by='interactive', resolution_action='restart',
        )
        pending = _esc('esc-3713-3', status='pending', resolution=None)
        dismissed = _esc(
            'esc-3713-4', status='dismissed', resolution='Auto-dismissed: ...',
            resolved_at='2026-08-01T00:00:00+00:00', resolved_by='auto-dismissed',
        )
        resolved_without_prose = _esc(
            'esc-3713-5', resolution=None, resolved_at='2026-08-01T12:00:00+00:00',
        )
        records = [later, earlier, pending, dismissed, resolved_without_prose]

        expected = (
            CarriedResolution(
                escalation_id='esc-3713-2',
                summary='second filed',
                resolution='earlier answer',
                resolved_at='2026-08-02T00:00:00+00:00',
                resolved_by='interactive',
                resolution_action='restart',
            ),
            CarriedResolution(
                escalation_id='esc-3713-1',
                summary='first filed',
                resolution='later answer',
                resolved_at='2026-08-03T00:00:00+00:00',
                resolved_by='steward',
                resolution_action='resume',
            ),
        )
        for order in itertools.permutations(records):
            assert select_carried_resolutions(order) == expected

    def test_returns_frozen_items_in_a_tuple(self):
        carried = select_carried_resolutions([_esc('esc-1-1', resolved_at='t')])

        assert isinstance(carried, tuple)
        with pytest.raises(dataclasses.FrozenInstanceError):
            carried[0].resolution = 'rewritten'  # type: ignore[misc]

    def test_falls_back_to_filing_time_then_id_never_input_order(self):
        filed_last = _esc('esc-9-1', timestamp='2026-08-05T00:00:00+00:00')
        tie_high_id = _esc('esc-9-3', timestamp='2026-08-04T00:00:00+00:00')
        tie_low_id = _esc('esc-9-2', timestamp='2026-08-04T00:00:00+00:00')

        for order in itertools.permutations([filed_last, tie_high_id, tie_low_id]):
            ids = [c.escalation_id for c in select_carried_resolutions(order)]
            assert ids == ['esc-9-2', 'esc-9-3', 'esc-9-1']


def _carried(
    esc_id: str,
    resolution: str,
    *,
    summary: str = 'synthetic blocker',
    resolved_at: str | None = '2026-08-02T00:00:00+00:00',
    resolved_by: str | None = 'steward',
    resolution_action: str | None = 'resume',
) -> CarriedResolution:
    return CarriedResolution(
        escalation_id=esc_id,
        summary=summary,
        resolution=resolution,
        resolved_at=resolved_at,
        resolved_by=resolved_by,
        resolution_action=resolution_action,
    )


class TestRenderCarriedResolutions:
    def test_every_resolution_is_headed_by_its_own_escalation(self):
        first = _carried(
            'esc-3713-4', 'RESUME with step 20', summary='step 19 verify red',
            resolved_at='2026-08-02T01:00:00+00:00', resolution_action='resume',
        )
        second = _carried(
            'esc-3713-7', 'do X instead', summary='scope too narrow',
            resolved_at='2026-08-04T09:30:00+00:00', resolution_action='restart',
        )

        text = render_carried_resolutions([first, second])

        first_id = text.index('esc-3713-4')
        first_prose = text.index('RESUME with step 20')
        second_id = text.index('esc-3713-7')
        second_prose = text.index('do X instead')
        assert first_id < first_prose < second_id < second_prose
        first_block = text[first_id:second_id]
        assert 'step 19 verify red' in first_block
        assert '2026-08-02T01:00:00+00:00' in first_block
        assert 'resume' in first_block
        second_block = text[second_id:]
        assert 'scope too narrow' in second_block
        assert '2026-08-04T09:30:00+00:00' in second_block
        assert 'restart' in second_block

    def test_no_resolutions_renders_an_explicit_statement(self):
        assert NO_CARRIED_RESOLUTIONS
        assert render_carried_resolutions([]) == NO_CARRIED_RESOLUTIONS
