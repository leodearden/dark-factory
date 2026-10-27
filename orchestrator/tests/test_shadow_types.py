"""Tests for orchestrator.shadow_types — the closed shadow-cell column vocabularies."""

from __future__ import annotations

import json
from enum import StrEnum

import pytest

from orchestrator.evals.configs import CANDIDATE_ENDPOINT_PRICES
from orchestrator.evals.live_fixture import build_live_fixture
from orchestrator.evals.task_sampler import default_verify_commands
from orchestrator.shadow_types import (
    CostBasis,
    PlanSource,
    ReferenceKind,
    ShadowCellState,
    ShadowShape,
    ShadowStormCause,
)


class TestShadowShape:
    """The closed cell-shape vocabulary (PRD C1) and the shape→plan rule."""

    def test_exactly_the_four_c1_values(self):
        assert [s.value for s in ShadowShape] == [
            'implementer',
            'architect',
            'architect-consequence',
            'end-to-end',
        ]

    def test_member_is_a_plain_str_and_json_round_trips(self):
        # C3 types `shape: str`; a StrEnum member satisfies that verbatim and
        # serialises as the bare value (no 'ShadowShape.IMPLEMENTER' leaking
        # into the shadow_cells.shape TEXT column or a persisted result).
        assert isinstance(ShadowShape.IMPLEMENTER, str)
        assert json.dumps(ShadowShape.IMPLEMENTER) == '"implementer"'
        assert json.loads(json.dumps(ShadowShape.IMPLEMENTER)) == 'implementer'

    @pytest.mark.parametrize(
        'raw,member',
        [
            ('implementer', ShadowShape.IMPLEMENTER),
            ('architect', ShadowShape.ARCHITECT),
            ('architect-consequence', ShadowShape.ARCHITECT_CONSEQUENCE),
            ('end-to-end', ShadowShape.END_TO_END),
        ],
    )
    def test_string_coercion_recovers_the_member(self, raw, member):
        # A caller holding a value read back from the shadow_cells TEXT column
        # coerces it without a lookup table of its own.
        assert ShadowShape(raw) is member

    @pytest.mark.parametrize(
        'shape,requires_plan',
        [
            (ShadowShape.IMPLEMENTER, True),
            (ShadowShape.ARCHITECT_CONSEQUENCE, True),
            (ShadowShape.ARCHITECT, False),
            (ShadowShape.END_TO_END, False),
        ],
    )
    def test_requires_plan_is_total_over_the_vocabulary(self, shape, requires_plan):
        assert shape.requires_plan is requires_plan


def test_cost_basis_values():
    assert issubclass(CostBasis, StrEnum)
    assert [b.value for b in CostBasis] == ['metered', 'subscription']


class TestShadowCellState:
    """C1's cell lifecycle and the two predicates its invariants hang on."""

    def test_values_are_the_c1_lifecycle_in_order(self):
        assert [s.value for s in ShadowCellState] == [
            'sampled',
            'opened',
            'running',
            'awaiting_reference',
            'settled',
            'cap_excluded',
            'expired',
            'failed',
        ]

    def test_is_terminal(self):
        assert {s for s in ShadowCellState if s.is_terminal} == {
            ShadowCellState.SETTLED,
            ShadowCellState.CAP_EXCLUDED,
            ShadowCellState.EXPIRED,
            ShadowCellState.FAILED,
        }

    def test_requires_reason_is_every_non_settled_terminal_state(self):
        assert {s for s in ShadowCellState if s.requires_reason} == {
            ShadowCellState.CAP_EXCLUDED,
            ShadowCellState.EXPIRED,
            ShadowCellState.FAILED,
        }


_S = ShadowCellState

_LEGAL_SUCCESSORS: dict[ShadowCellState, set[ShadowCellState]] = {
    _S.SAMPLED: {_S.OPENED, _S.FAILED},
    _S.OPENED: {_S.RUNNING, _S.FAILED},
    _S.RUNNING: {_S.AWAITING_REFERENCE, _S.CAP_EXCLUDED, _S.FAILED},
    _S.AWAITING_REFERENCE: {_S.SETTLED, _S.EXPIRED},
    _S.SETTLED: set(),
    _S.CAP_EXCLUDED: set(),
    _S.EXPIRED: set(),
    _S.FAILED: set(),
}


class TestShadowCellLifecycleRules:
    """The successor table and the two state groups the store and coordinator key on."""

    @pytest.mark.parametrize(
        'source,target',
        [(a, b) for a in ShadowCellState for b in ShadowCellState],
        ids=lambda s: s.value,
    )
    def test_may_become_is_exactly_the_successor_table(self, source, target):
        assert source.may_become(target) is (target in _LEGAL_SUCCESSORS[source])

    def test_table_is_monotone_in_declaration_order(self):
        order = list(ShadowCellState)
        allowed = [(a, b) for a in order for b in order if a.may_become(b)]
        assert allowed
        assert all(order.index(b) > order.index(a) for a, b in allowed)

    def test_no_state_may_become_itself(self):
        assert not any(s.may_become(s) for s in ShadowCellState)

    def test_terminal_states_have_no_successor(self):
        for state in ShadowCellState:
            if state.is_terminal:
                assert not any(state.may_become(t) for t in ShadowCellState)

    def test_held_by_run_is_the_three_pre_run_end_states(self):
        assert {s for s in ShadowCellState if s.held_by_run} == {
            _S.SAMPLED,
            _S.OPENED,
            _S.RUNNING,
        }

    def test_stamps_settled_at_is_settled_and_expired(self):
        assert {s for s in ShadowCellState if s.stamps_settled_at} == {
            _S.SETTLED,
            _S.EXPIRED,
        }

    def test_stamps_run_ended_at_is_every_state_a_held_state_leaves_into(self):
        assert {s for s in ShadowCellState if s.stamps_run_ended_at} == {
            _S.AWAITING_REFERENCE,
            _S.CAP_EXCLUDED,
            _S.FAILED,
        }


def test_storm_cause_values():
    assert issubclass(ShadowStormCause, StrEnum)
    assert {c.value for c in ShadowStormCause} == {
        'daily_budget',
        'failure_streak',
        'cost_ratio',
    }


def test_reference_kind_values():
    assert [k.value for k in ReferenceKind] == ['landed', 'none']


def test_plan_source_values():
    assert [p.value for p in PlanSource] == ['production', 'candidate']


_ALL_MEMBERS: list[StrEnum] = [
    member
    for enum_cls in (
        ShadowShape,
        CostBasis,
        ShadowCellState,
        ReferenceKind,
        PlanSource,
        ShadowStormCause,
    )
    for member in enum_cls
]


@pytest.mark.parametrize('member', _ALL_MEMBERS, ids=lambda m: f'{type(m).__name__}.{m.name}')
def test_every_member_is_a_bare_str_that_round_trips(member: StrEnum):
    assert isinstance(member, str)
    assert json.dumps(member) == json.dumps(member.value)
    assert type(member)(member.value) is member


class TestSingleCopy:
    """The sibling modules consume these exact enums, never a second copy."""

    def test_candidate_prices_use_the_shadow_types_cost_basis(self):
        assert CANDIDATE_ENDPOINT_PRICES['glm-5.3'].cost_basis is CostBasis.SUBSCRIPTION

    def test_build_live_fixture_accepts_a_shadow_types_shape(self, tmp_path):
        fixture = build_live_fixture(
            {'id': '5382', 'title': 't', 'metadata': {}},
            base_sha='a' * 40,
            project_root=tmp_path,
            plan=None,
            verify_commands=default_verify_commands('df'),
            shape=ShadowShape.ARCHITECT,
            cell_id='01JCELL',
        )
        assert fixture['id'] == 'shadow_5382_01JCELL'
