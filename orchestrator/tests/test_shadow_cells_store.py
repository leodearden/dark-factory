"""Tests for the ``shadow_cells`` record and store (plans/live-shadow-eval-prd.md C1)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta, timezone

import pytest

from orchestrator.run_store import ShadowCell
from orchestrator.shadow_types import CostBasis, ShadowCellState, ShadowShape

_OPENED_AT = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


def _cell(**overrides) -> ShadowCell:
    fields = {
        'cell_id': '01JCELLAAAAAAAAAAAAAAAAAAA',
        'project_id': 'dark_factory',
        'task_id': '5382',
        'shape': ShadowShape.IMPLEMENTER,
        'candidate': 'glm-5.3-endpoint',
        'incumbent': 'claude-opus-high',
        'stratum': ('df', 'feature', 'full'),
        'base_sha': 'a' * 40,
        'state': ShadowCellState.OPENED,
        'cost_basis': CostBasis.METERED,
        'opened_at': _OPENED_AT,
        'settle_deadline': _OPENED_AT + timedelta(hours=168),
    }
    fields.update(overrides)
    return ShadowCell(**fields)


class TestShadowCellInvariants:
    """C1's per-row invariants, refused at construction with an informative message."""

    def test_a_valid_cell_constructs(self):
        cell = _cell()
        assert cell.state is ShadowCellState.OPENED

    @pytest.mark.parametrize(
        'state',
        [ShadowCellState.FAILED, ShadowCellState.EXPIRED, ShadowCellState.CAP_EXCLUDED],
    )
    @pytest.mark.parametrize('reason', [None, '', '   '])
    def test_non_settled_terminal_state_requires_a_reason(self, state, reason):
        with pytest.raises(ValueError) as excinfo:
            _cell(state=state, reason=reason)
        message = str(excinfo.value)
        assert '01JCELLAAAAAAAAAAAAAAAAAAA' in message
        assert state.value in message
        assert 'reason' in message

    def test_settled_needs_no_reason(self):
        assert _cell(state=ShadowCellState.SETTLED, reason=None).reason is None

    @pytest.mark.parametrize(
        'field',
        ['owner', 'cell_id', 'project_id', 'task_id', 'candidate', 'incumbent', 'base_sha'],
    )
    def test_empty_identity_string_is_refused(self, field):
        with pytest.raises(ValueError, match=field):
            _cell(**{field: ''})

    @pytest.mark.parametrize('field', ['opened_at', 'settle_deadline', 'running_at', 'settled_at'])
    def test_naive_timestamp_is_refused(self, field):
        naive = {
            'opened_at': datetime(2026, 10, 19, 12, 0),
            'settle_deadline': datetime(2026, 10, 26, 12, 0),
            'running_at': datetime(2026, 10, 19, 12, 5),
            'settled_at': datetime(2026, 10, 20, 12, 0),
        }[field]
        with pytest.raises(ValueError, match=field):
            _cell(**{field: naive})

    @pytest.mark.parametrize('offset', [timedelta(0), timedelta(seconds=-1)])
    def test_settle_deadline_must_lie_ahead_of_opened_at(self, offset):
        with pytest.raises(ValueError, match='settle_deadline'):
            _cell(settle_deadline=_OPENED_AT + offset)

    @pytest.mark.parametrize('field', ['cost_usd', 'cost_credits'])
    def test_negative_cost_is_refused(self, field):
        overrides = {field: -0.01, 'cost_basis': CostBasis.SUBSCRIPTION}
        with pytest.raises(ValueError, match=field):
            _cell(**overrides)

    def test_a_metered_arm_never_carries_credits(self):
        with pytest.raises(ValueError, match='cost_credits'):
            _cell(cost_basis=CostBasis.METERED, cost_credits=1.0)

    def test_a_subscription_arm_carries_credits_and_an_imputed_cost(self):
        cell = _cell(cost_basis=CostBasis.SUBSCRIPTION, cost_credits=1234.5, cost_usd=0.42)
        assert (cell.cost_credits, cell.cost_usd) == (1234.5, 0.42)

    def test_a_cell_is_never_its_own_parent(self):
        with pytest.raises(ValueError, match='parent_cell_id'):
            _cell(parent_cell_id='01JCELLAAAAAAAAAAAAAAAAAAA')

    @pytest.mark.parametrize(
        'stratum',
        [
            ('df', '', 'full'),
            ('df', 'feat×ure', 'full'),
            ('df', 'feature'),
            ('df', 'feature', 'full', 'extra'),
        ],
    )
    def test_malformed_stratum_is_refused(self, stratum):
        with pytest.raises(ValueError, match='stratum'):
            _cell(stratum=stratum)

    @pytest.mark.parametrize('field_name', ['state', 'reason', 'settled_at'])
    def test_the_record_is_frozen(self, field_name):
        cell = _cell()
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(cell, field_name, None)

    def test_defaults(self):
        cell = _cell()
        assert cell.owner == 'shadow_coordinator'
        assert cell.cost_usd == 0.0
        assert cell.cost_credits == 0.0

    def test_an_aware_non_utc_timestamp_is_accepted(self):
        plus_two = timezone(timedelta(hours=2))
        cell = _cell(opened_at=_OPENED_AT.astimezone(plus_two))
        assert cell.opened_at == _OPENED_AT
