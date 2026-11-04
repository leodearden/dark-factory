"""Tests for evals/plan_predicates.py — does an artifact or a cell carry a plan?

The rationale behind each predicate lives in docs/plan-scoring-and-judge.md
("The plan-production predicate (task 3302)" and "The anti-fabrication floor
applies uniformly (task 3303)").
"""

from __future__ import annotations

import pytest

from orchestrator.evals.judge import score_plan_structure
from orchestrator.evals.plan_predicates import (
    is_scorable_plan,
    plan_confirmed,
    produced_a_plan,
)


def _plan_with_steps() -> dict:
    return {
        'steps': [
            {'id': 'step-1', 'type': 'test', 'description': 'RED test for X'},
            {'id': 'step-2', 'type': 'impl', 'description': 'GREEN implement X'},
        ],
        'files': ['pkg/mod.py'],
        '_finalized_at': '2026-07-19T00:00:00+00:00',
    }


def _lonely_step_plan() -> dict:
    return {'steps': [{'id': 'step-1', 'type': 'impl', 'description': 'all of it'}]}


def _create_plan_stub(steps: list | None) -> dict:
    """The header-only plan ``plan_tools._create_plan`` writes on the first call."""
    return {
        'task_id': 't', 'title': 'x', 'analysis': 'a', 'files': [],
        'prerequisites': [], 'steps': steps, 'design_decisions': [], 'reuse': [],
    }


_UNSCORABLE_PLAN_SHAPES = [
    {},
    None,
    [],
    'x',
    {'steps': []},
    {'steps': None},
    _create_plan_stub([]),
    _create_plan_stub(None),
]


class TestIsScorablePlan:
    @pytest.mark.parametrize('plan', _UNSCORABLE_PLAN_SHAPES)
    def test_shapes_without_steps_are_not_scorable(self, plan):
        assert is_scorable_plan(plan) is False

    def test_a_plan_with_steps_is_scorable(self):
        assert is_scorable_plan(_plan_with_steps()) is True
        # The predicate asks "is there content to measure?", never "is the
        # content good?" — a poor plan must keep earning its low score.
        assert is_scorable_plan(_lonely_step_plan()) is True

    @pytest.mark.parametrize('plan', _UNSCORABLE_PLAN_SHAPES)
    def test_rejected_shapes_are_exactly_the_structural_floor_zero(self, plan):
        """ANTI-DRIFT: the predicate and the floor's short-circuit must agree."""
        assert is_scorable_plan(plan) is False
        assert score_plan_structure(plan) == 0.0


class TestPlanConfirmed:
    def test_finalized_stamp_confirms(self):
        assert plan_confirmed({'_finalized_at': '2026-07-19T12:00:00+00:00'}) is True

    def test_absent_stamp_does_not_confirm(self):
        assert plan_confirmed({}) is False

    def test_empty_stamp_does_not_confirm(self):
        assert plan_confirmed({'_finalized_at': ''}) is False


class TestProducedAPlan:
    """`produced_a_plan(metrics_dict)` — did this architect emit a plan?

    The end-to-end equivalence with the artifact-level twin is pinned through
    the real runner by
    test_eval_architect.py::TestSteplessPlanIsNeverJudged.test_persisted_metrics_agree_with_the_artifact_level_twin.
    """

    def test_nonzero_plan_steps_is_a_plan(self):
        assert produced_a_plan({'plan_steps': 3}) is True

    def test_zero_plan_steps_is_not_a_plan(self):
        assert produced_a_plan({'plan_steps': 0}) is False

    def test_absent_key_is_not_a_plan(self):
        """A legacy / hand-edited metrics dict must not read as a plan."""
        assert produced_a_plan({}) is False

    def test_none_plan_steps_is_not_a_plan(self):
        """The `or 0` empty shape run_architect_eval's len(... or []) guards."""
        assert produced_a_plan({'plan_steps': None}) is False

    def test_does_not_consult_plan_quality(self):
        """A nonzero plan_quality is NOT evidence a plan exists, and a zero one
        is NOT evidence it does not (task 2863 AMENDMENT §1 / e2066ec6)."""
        assert produced_a_plan({'plan_steps': 0, 'plan_quality': 0.95}) is False
        assert produced_a_plan({'plan_steps': 4, 'plan_quality': 0.0}) is True

    @pytest.mark.parametrize('plan', [
        None,
        {},
        {'steps': []},
        {'task_id': 't', 'title': 'x', 'analysis': 'a', 'files': [], 'steps': []},
        {'task_id': 't', 'title': 'x', 'steps': [{'id': 'step-1'}, {'id': 'step-2'}]},
    ])
    def test_agrees_with_the_artifact_level_twin_given_the_same_step_count(self, plan):
        """Neither predicate carries a condition beyond "step count > 0"."""
        persisted = {'plan_steps': len((plan or {}).get('steps') or [])}
        assert produced_a_plan(persisted) is is_scorable_plan(plan)
