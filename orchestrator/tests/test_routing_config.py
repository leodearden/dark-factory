"""Tests for config.py's routing-policy schema extension (task epsilon,
plans/adaptive-model-routing-prd.md).

RED phase (step-1): ``RuleMatch``, ``RuleSet``, ``RoutingRule`` do not exist
yet, and ``RoutingConfig`` has no ``ladder``/``per_model_daily_ceiling_usd``/
``rules`` fields -- every test below fails at import or construction time.

Fixtures are kept MODULE-LOCAL (not conftest.py) -- mirrors test_routing.py's
documented rationale (a conftest.py edit forces verify.py's has_conftest to
widen the merge-time scoped-test selection to the whole owning package).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchestrator.config import (
    OrchestratorConfig,
    RoutingConfig,
    RoutingRule,
    RuleMatch,
    RuleSet,
)
from orchestrator.routing import DEFAULT_LADDER


class TestRoutingRuleParses:
    """(a) A well-formed RoutingRule with a closed-vocabulary match/set parses."""

    def test_valid_rule_parses(self):
        rule = RoutingRule(
            id='rust-large-plan-implementer',
            match=RuleMatch(
                role=['implementer'],
                plan_min_steps=12,
                plan_min_modules=3,
                module_prefix='crates/',
            ),
            set=RuleSet(model='opus'),
        )
        assert rule.id == 'rust-large-plan-implementer'
        assert rule.match.role == ['implementer']
        assert rule.match.plan_min_steps == 12
        assert rule.match.plan_min_modules == 3
        assert rule.match.module_prefix == 'crates/'
        assert rule.set.model == 'opus'

    def test_rule_defaults_to_empty_match_and_set(self):
        rule = RoutingRule(id='r1')
        assert rule.match == RuleMatch()
        assert rule.set == RuleSet()


class TestClosedVocabularyForbidsUnknownKeys:
    """(b)/(c) An unknown match/set key raises ValidationError naming the key."""

    def test_unknown_match_key_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            RuleMatch(plan_vibes=3)
        assert 'plan_vibes' in str(exc_info.value)

    def test_unknown_set_key_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            RuleSet(vibe='good')
        assert 'vibe' in str(exc_info.value)


class TestRoutingConfigDefaults:
    """(d) RoutingConfig() defaults: ladder, per_model_daily_ceiling_usd, rules."""

    def test_ladder_defaults_to_default_ladder(self):
        cfg = RoutingConfig()
        assert cfg.ladder == list(DEFAULT_LADDER)

    def test_per_model_daily_ceiling_usd_defaults_empty(self):
        cfg = RoutingConfig()
        assert cfg.per_model_daily_ceiling_usd == {}

    def test_rules_defaults_empty(self):
        cfg = RoutingConfig()
        assert cfg.rules == []


class TestOrchestratorConfigRoundTripsWithNewFields:
    """(e) An OrchestratorConfig() with the new routing fields still
    round-trips ``_validate_models_in_allowlist`` (no regression from
    extending RoutingConfig)."""

    def test_default_orchestrator_config_constructs_ok(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig()
        assert cfg.routing.ladder == list(DEFAULT_LADDER)
        assert cfg.routing.per_model_daily_ceiling_usd == {}
        assert cfg.routing.rules == []

    def test_orchestrator_config_with_rule_constructs_ok(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig(
            routing=RoutingConfig(
                rules=[
                    RoutingRule(
                        id='r1',
                        match=RuleMatch(role=['implementer']),
                        set=RuleSet(model='opus'),
                    ),
                ],
            ),
        )
        assert cfg.routing.rules[0].id == 'r1'
        assert cfg.routing.rules[0].set.model == 'opus'
