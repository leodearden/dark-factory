import logging

import pytest

from orchestrator.evals.cell_admission import (
    cost_per_plan,
    is_plan_only,
    is_unmeasurable,
    judged_blind_count,
    judged_without_reference,
    mean_plan_quality,
    plan_quality_score,
    plan_rate,
)
from orchestrator.evals.metrics import RoleUnderTest

_LOGGER = 'orchestrator.evals.cell_admission'


def _architect(**fields):
    return {'role_under_test': RoleUnderTest.ARCHITECT, **fields}


def _implementer(**fields):
    return {'role_under_test': RoleUnderTest.IMPLEMENTER, **fields}


class TestIsPlanOnly:
    """Only an architect cell is plan-only."""

    def test_architect_cell_is_plan_only(self):
        assert is_plan_only(_architect()) is True

    def test_implementer_cell_is_not_plan_only(self):
        assert is_plan_only(_implementer()) is False

    def test_cell_without_a_role_is_not_plan_only(self):
        assert is_plan_only({}) is False


class TestPlanQualityScore:
    """The score a cell contributes to the plan-quality pool, or None to exclude it."""

    @staticmethod
    def _floor_records(caplog):
        return [r for r in caplog.records if 'Plan-quality floor' in r.getMessage()]

    def test_non_architect_cell_is_excluded(self):
        assert plan_quality_score(_implementer(plan_steps=3, plan_quality=0.8)) is None

    def test_cap_tainted_architect_cell_is_excluded(self):
        cell = _architect(plan_steps=3, plan_quality=0.8, cap_tainted=True)
        assert plan_quality_score(cell) is None

    def test_stepless_architect_cell_floors_to_zero_and_warns_naming_the_cell(
        self, caplog,
    ):
        cell = _architect(plan_steps=0, plan_quality=0.31)
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            score = plan_quality_score(cell, where='fixture-a x arch-cfg')
        assert score == 0.0
        records = self._floor_records(caplog)
        assert len(records) == 1
        assert records[0].name == _LOGGER
        assert records[0].levelno == logging.WARNING
        assert 'fixture-a x arch-cfg' in records[0].getMessage()

    def test_architect_cell_with_a_plan_scores_its_persisted_float(self):
        assert plan_quality_score(_architect(plan_steps=4, plan_quality=0.72)) == 0.72


class TestIsUnmeasurable:
    """A refused architect cell measured nothing; a failed workflow cell did."""

    def test_cap_tainted_architect_cell_is_unmeasurable(self):
        assert is_unmeasurable(_architect(cap_tainted=True)) is True

    def test_cap_tainted_implementer_cell_is_measurable(self):
        assert is_unmeasurable(_implementer(cap_tainted=True)) is False


class TestJudgedWithoutReference:
    """An absent marker is unknown, never False."""

    def test_absent_key_is_unknown(self):
        assert judged_without_reference(_architect()) is None

    @pytest.mark.parametrize('marker', [True, False])
    def test_present_key_reads_as_its_truth_value(self, marker):
        assert judged_without_reference(_architect(judged_without_reference=marker)) is marker

    def test_any_unmeasured_cell_makes_the_count_unknown(self):
        assert judged_blind_count(3, 1) is None

    def test_fully_measured_tally_is_the_count(self):
        assert judged_blind_count(3, 0) == 3


class TestPoolReductions:
    """Empty pools reduce to None, never to a fabricated 0.0 or 1.0."""

    def test_mean_of_an_empty_pool_is_unknown(self):
        assert mean_plan_quality([]) is None

    def test_mean_is_rounded_to_four_dp(self):
        assert mean_plan_quality([0.1, 0.2, 0.25]) == 0.1833

    def test_plan_rate_of_an_empty_pool_is_unknown(self):
        assert plan_rate(0, 0) is None

    def test_plan_rate_is_the_planned_fraction_of_the_admitted_pool(self):
        assert plan_rate(4, 1) == 0.75

    def test_cost_per_plan_with_no_plans_is_unknown(self):
        assert cost_per_plan(10.0, 0) is None

    def test_cost_per_plan_divides_spend_by_planned_cells(self):
        assert cost_per_plan(9.0, 2) == 4.5
