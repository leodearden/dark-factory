"""ScoringConfig: the C2 sub-budget knobs, their load-time invariants, and their
green-tier reload disposition (task 3819; plans/scheduler-dispatch-scoring-and-lock-layer-prd.md
C2, I-1, I-3, D8).
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from orchestrator.config import (
    AGE_BUDGET,
    AGE_HALF_SECS,
    CONTINUITY_CREDIT,
    CONTINUITY_CREDIT_MAX_AGE_SECS,
    CONTINUITY_PROBE_TTL_SECS,
    CPM_BUDGET,
    CPM_D_CAP,
    RELOADABLE_FIELDS,
    TIER_WIDTH,
    OrchestratorConfig,
    ScoringConfig,
    apply_reload,
)


def _continuity_bound(cfg: ScoringConfig) -> float:
    """The CPM term's value at D=4: continuity must stay strictly below it (I-3)."""
    return cfg.cpm_budget * math.log1p(4) / math.log1p(cfg.cpm_d_cap)


class TestDefaults:
    def test_code_defaults_are_the_mandated_literals(self):
        cfg = ScoringConfig()

        assert cfg.cpm_budget == 300.0
        assert cfg.cpm_d_cap == 32
        assert cfg.age_budget == 500.0
        assert cfg.age_half_secs == 259200
        assert cfg.continuity_credit == 120.0
        assert cfg.new_scoring_min_anchor_coverage == 0.90

    def test_module_constants_are_the_field_defaults(self):
        cfg = ScoringConfig()

        assert (cfg.cpm_budget, cfg.cpm_d_cap, cfg.age_budget, cfg.age_half_secs,
                cfg.continuity_credit) == (CPM_BUDGET, CPM_D_CAP, AGE_BUDGET,
                                           AGE_HALF_SECS, CONTINUITY_CREDIT)

    @pytest.mark.usefixtures('code_default_config')
    def test_orchestrator_config_carries_the_default_submodel(self):
        assert OrchestratorConfig().scoring == ScoringConfig()

    def test_default_budgets_fit_inside_one_tier(self):
        cfg = ScoringConfig()

        assert cfg.cpm_budget + cfg.age_budget + cfg.continuity_credit <= TIER_WIDTH - 1

    def test_default_continuity_sits_below_the_d4_cpm_term(self):
        cfg = ScoringConfig()

        assert cfg.continuity_credit < _continuity_bound(cfg)

    def test_continuity_probe_leaves_default_to_the_mandated_literals(self):
        cfg = ScoringConfig()

        assert cfg.continuity_probe_ttl_secs == 300.0 == CONTINUITY_PROBE_TTL_SECS
        assert cfg.continuity_credit_max_age_secs == 604800.0 == CONTINUITY_CREDIT_MAX_AGE_SECS


class TestBudgetSumInvariant:
    """I-1: the three bonus budgets together can never push a task across a tier."""

    def test_sum_above_the_tier_width_is_rejected_at_load(self):
        with pytest.raises(ValidationError) as excinfo:
            ScoringConfig(age_budget=900.0)

        message = str(excinfo.value)
        assert 'age_budget' in message
        assert '1320' in message
        assert str(TIER_WIDTH - 1) in message

    def test_sum_of_exactly_tier_width_minus_one_is_accepted(self):
        cfg = ScoringConfig(age_budget=float(TIER_WIDTH - 1) - CPM_BUDGET - CONTINUITY_CREDIT)

        assert cfg.cpm_budget + cfg.age_budget + cfg.continuity_credit == TIER_WIDTH - 1

    def test_rejected_when_loaded_through_orchestrator_config(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        with pytest.raises(ValidationError):
            OrchestratorConfig.model_validate({'scoring': {'age_budget': 900.0}})


class TestContinuityBelowCpmInvariant:
    """I-3: continuity alone must never outrank four dependents of critical-path work."""

    def test_continuity_above_the_d4_term_is_rejected(self):
        with pytest.raises(ValidationError, match='continuity_credit'):
            ScoringConfig(continuity_credit=200.0)

    def test_continuity_above_the_d4_term_is_rejected_with_room_in_the_tier(self):
        with pytest.raises(ValidationError) as excinfo:
            ScoringConfig(continuity_credit=200.0, age_budget=400.0)

        message = str(excinfo.value)
        assert 'continuity_credit' in message
        assert 'cpm_d_cap' in message

    def test_continuity_equal_to_the_d4_term_is_rejected(self):
        with pytest.raises(ValidationError, match='continuity_credit'):
            ScoringConfig(continuity_credit=_continuity_bound(ScoringConfig()))


@pytest.mark.parametrize('bad_cap', [0, -1])
def test_a_d_cap_below_one_is_rejected(bad_cap):
    with pytest.raises(ValidationError, match='cpm_d_cap'):
        ScoringConfig(cpm_d_cap=bad_cap)


class TestContinuityProbeBounds:
    """C3: the probe TTL floor bounds git occupancy under any tuning; credit must lapse."""

    def test_a_probe_ttl_below_the_floor_is_rejected(self):
        with pytest.raises(ValidationError, match='continuity_probe_ttl_secs'):
            ScoringConfig(continuity_probe_ttl_secs=59.0)

    def test_a_probe_ttl_at_the_floor_is_accepted(self):
        assert ScoringConfig(continuity_probe_ttl_secs=60.0).continuity_probe_ttl_secs == 60.0

    @pytest.mark.parametrize('bad_age', [0, -1.0])
    def test_a_non_positive_credit_max_age_is_rejected(self, bad_age):
        with pytest.raises(ValidationError, match='continuity_credit_max_age_secs'):
            ScoringConfig(continuity_credit_max_age_secs=bad_age)


class TestScoringReloadDisposition:
    """D8: every scoring knob is green-tier, so a retune costs a reload, not a fleet restart."""

    @pytest.fixture(autouse=True)
    def _no_config_file(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')

    @pytest.mark.parametrize('leaf', list(ScoringConfig.model_fields))
    def test_every_leaf_is_reloadable(self, leaf):
        assert f'scoring.{leaf}' in RELOADABLE_FIELDS

    def test_legacy_formula_knobs_stay_restart_tier(self):
        assert 'age_alpha' not in RELOADABLE_FIELDS
        assert 'cpm_beta' not in RELOADABLE_FIELDS

    def test_a_retune_applies_in_place_to_held_references(self):
        live = OrchestratorConfig()
        held = live.scoring
        fresh = OrchestratorConfig(
            scoring=ScoringConfig(age_half_secs=86400.0, new_scoring_min_anchor_coverage=0.75),
        )

        report = apply_reload(live, fresh)

        assert report['reloaded'] is True
        for path in ('scoring.age_half_secs', 'scoring.new_scoring_min_anchor_coverage'):
            assert path in report['applied']
            assert path not in report['restart_required']
        assert live.scoring is held
        assert held.age_half_secs == 86400.0
        assert held.new_scoring_min_anchor_coverage == 0.75

    def test_a_budget_move_applies_though_one_leaf_order_is_transiently_over_the_tier(self):
        live = OrchestratorConfig()
        fresh = OrchestratorConfig(scoring=ScoringConfig(cpm_budget=400.0, age_budget=400.0))
        cpm_first = fresh.scoring.cpm_budget + live.scoring.age_budget + live.scoring.continuity_credit
        assert cpm_first > TIER_WIDTH - 1

        report = apply_reload(live, fresh)

        assert report['reloaded'] is True, report['error']
        assert (live.scoring.cpm_budget, live.scoring.age_budget) == (400.0, 400.0)

    def test_a_reload_crossing_the_tier_is_rejected_and_rolled_back(self):
        live = OrchestratorConfig()
        fresh = OrchestratorConfig()
        object.__setattr__(fresh.scoring, 'age_budget', 900.0)

        report = apply_reload(live, fresh)

        assert report['reloaded'] is False
        assert 'age_budget' in report['error']
        assert live.scoring.age_budget == AGE_BUDGET
