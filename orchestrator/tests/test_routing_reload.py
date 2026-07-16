"""Tests for config.py's routing hot-reload disposition (task epsilon,
plans/adaptive-model-routing-prd.md, boundary test 11 + green-tier proof).

RED phase (step-15): exercises diff_config/apply_reload directly on two
OrchestratorConfig instances for the THREE RoutingConfig leaves task epsilon
adds (routing.rules, routing.ladder, routing.per_model_daily_ceiling_usd).
routing.allowed_models' own reload disposition is already covered by
TestRoutingAllowlistReloadDisposition in test_routing.py (task beta) -- this
file is the epsilon-specific companion, verifying:

(a) a valid fresh config that adds a routing.rules entry (and edits ladder /
    per_model_daily_ceiling_usd) applies cleanly via apply_reload -- every
    touched routing.* leaf lands in `applied`, and live is updated in place;
(b) an invalid fresh config -- an operator YAML edit whose rule carries an
    unknown match/set key -- raises pydantic.ValidationError naming the key
    at OrchestratorConfig construction time, BEFORE apply_reload is ever
    reached, so live's prior rules are provably untouched (I1 fail-closed;
    mirrors test_routing.py's
    test_typo_in_fresh_config_fails_closed_before_apply_reload_is_reached,
    now for RuleMatch/RuleSet's own extra='forbid' rather than the
    model-allowlist validator);
(c) routing.rules / routing.ladder / routing.per_model_daily_ceiling_usd are
    all members of RELOADABLE_FIELDS -- auto-covered by the existing
    _submodel_leaf_paths('routing', RoutingConfig) whole-submodel group
    (config.py:3305), so no RELOADABLE_FIELDS edit is expected (step-16
    confirms this explicitly).

Fixtures are kept MODULE-LOCAL (not conftest.py) -- mirrors test_routing.py's
documented rationale (a conftest.py edit forces verify.py's has_conftest to
widen the merge-time scoped-test selection to the whole owning package).
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from orchestrator.config import (
    RELOADABLE_FIELDS,
    OrchestratorConfig,
    RoutingConfig,
    RoutingRule,
    RuleMatch,
    RuleSet,
    apply_reload,
    load_config,
)


@pytest.fixture(autouse=True)
def _isolated_config_env(monkeypatch, tmp_path):
    """Every OrchestratorConfig()/load_config() call in this module reads
    ONLY the package's shipped defaults.yaml (plus, where applicable, this
    test's own explicit YAML) -- no stray project config.yaml, no ambient
    ORCH_CONFIG_PATH. Mirrors test_routing_resolver.py's per-module
    convention. load_config(<explicit path>) always overwrites
    ORCH_CONFIG_PATH itself before constructing, so this fixture's setenv is
    only load-bearing for the direct OrchestratorConfig() construction calls
    below.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')


class TestRoutingNewLeavesAreReloadable:
    """(c) The three RoutingConfig leaves task epsilon adds are already
    green-tier hot-reloadable, auto-covered by the existing whole-submodel
    _submodel_leaf_paths('routing', RoutingConfig) group -- no
    RELOADABLE_FIELDS code edit needed."""

    def test_rules_ladder_and_ceiling_are_in_reloadable_fields(self):
        assert 'routing.rules' in RELOADABLE_FIELDS
        assert 'routing.ladder' in RELOADABLE_FIELDS
        assert 'routing.per_model_daily_ceiling_usd' in RELOADABLE_FIELDS


class TestValidRoutingRulesReloadApplies:
    """(a) A fresh config that adds a routing.rules entry and edits ladder /
    per_model_daily_ceiling_usd is a clean, valid reload."""

    def test_new_rule_and_edited_ladder_and_ceiling_apply_in_place(self):
        live = OrchestratorConfig()
        live_rule_count = len(live.routing.rules)
        live_ladder_before = list(live.routing.ladder)
        new_rule = RoutingRule(
            id='extra-caution-triage',
            match=RuleMatch(role=['triage'], min_routing_tier=2),
            set=RuleSet(model='+1'),
        )
        fresh = OrchestratorConfig(
            routing=RoutingConfig(
                ladder=[*live_ladder_before, 'opus-extra'],
                per_model_daily_ceiling_usd={'opus': 25.0},
                rules=[*live.routing.rules, new_rule],
            ),
        )

        report = apply_reload(live, fresh)

        assert report['reloaded'] is True
        assert report['error'] is None
        assert 'routing.rules' in report['applied']
        assert 'routing.ladder' in report['applied']
        assert 'routing.per_model_daily_ceiling_usd' in report['applied']
        assert len(live.routing.rules) == live_rule_count + 1
        assert live.routing.rules[-1] == new_rule
        assert live.routing.ladder == [*live_ladder_before, 'opus-extra']
        assert live.routing.per_model_daily_ceiling_usd == {'opus': 25.0}


class TestUnknownRuleKeyFailsClosedBeforeApply:
    """(b) An operator YAML edit whose policy rule carries an unknown
    match/set key raises ValidationError naming the key at OrchestratorConfig
    construction time -- reload never reaches apply_reload, so live's prior
    routing.rules are provably untouched."""

    def test_unknown_match_key_raises_naming_the_key(self, tmp_path):
        bad_yaml = tmp_path / 'bad-match.yaml'
        bad_yaml.write_text(yaml.dump({
            'routing': {
                'rules': [
                    {'id': 'bad-rule', 'match': {'plan_vibes': 3}, 'set': {'model': 'opus'}},
                ],
            },
        }))

        with pytest.raises(ValidationError, match='plan_vibes'):
            load_config(bad_yaml)

    def test_unknown_set_key_raises_naming_the_key(self, tmp_path):
        bad_yaml = tmp_path / 'bad-set.yaml'
        bad_yaml.write_text(yaml.dump({
            'routing': {
                'rules': [
                    {
                        'id': 'bad-rule',
                        'match': {'role': ['implementer']},
                        'set': {'model_typo': 'opus'},
                    },
                ],
            },
        }))

        with pytest.raises(ValidationError, match='model_typo'):
            load_config(bad_yaml)

    def test_prior_live_rules_untouched_when_fresh_construction_raises(self, tmp_path):
        live = OrchestratorConfig()
        live_rules_before = list(live.routing.rules)

        bad_yaml = tmp_path / 'bad.yaml'
        bad_yaml.write_text(yaml.dump({
            'routing': {
                'rules': [{'id': 'bad', 'match': {'plan_vibes': 1}, 'set': {}}],
            },
        }))

        with pytest.raises(ValidationError):
            load_config(bad_yaml)

        # apply_reload was never called -- live is untouched by construction.
        assert live.routing.rules == live_rules_before
