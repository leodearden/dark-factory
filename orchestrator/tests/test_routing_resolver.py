"""Tests for orchestrator.routing.resolve_route -- the layered routing
resolver (task epsilon, plans/adaptive-model-routing-prd.md).

RED phase (step-3): ``PlanShape``, ``RoleDefaults``, ``RouteInputs``,
``RoutingDecision``, and ``resolve_route`` do not exist yet in
``orchestrator.routing`` -- this whole module fails to import.

Core layering only: precedence metadata_override > policy_rule > config >
role_default (invariant 1: role_default is the always-available Total base).
Fail-safe validation, ladder-relative bumps, ceilings, and the full closed
condition vocabulary are exercised by later steps (step-5/7 RED, step-6/8
GREEN) -- this file only locks in the four-layer precedence order itself.

Fixtures are kept MODULE-LOCAL (not conftest.py) -- mirrors test_routing.py's
documented rationale (a conftest.py edit forces verify.py's has_conftest to
widen the merge-time scoped-test selection to the whole owning package).
"""

from __future__ import annotations

import pytest

from orchestrator.config import (
    BudgetsConfig,
    EffortConfig,
    ModelsConfig,
    OrchestratorConfig,
    RoutingConfig,
    RoutingRule,
    RuleMatch,
    RuleSet,
    TurnsConfig,
)
from orchestrator.routing import (
    PlanShape,
    RoleDefaults,
    RouteInputs,
    RoutingDecision,
    resolve_route,
)


@pytest.fixture(autouse=True)
def _isolated_config_env(monkeypatch, tmp_path):
    """Every OrchestratorConfig() built in this module reads ONLY the
    package's shipped defaults.yaml -- no stray project config.yaml, no
    ambient ORCH_CONFIG_PATH. Mirrors test_routing.py's per-test convention.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')


def _role_defaults(
    *, model='opus', effort='high', budget_usd=10.0, max_turns=80,
) -> RoleDefaults:
    return RoleDefaults(model=model, effort=effort, budget_usd=budget_usd, max_turns=max_turns)


def _cfg_with_rules(*rules: RoutingRule) -> OrchestratorConfig:
    """A minimal OrchestratorConfig carrying only the given policy rules."""
    return OrchestratorConfig(routing=RoutingConfig(rules=list(rules)))


def _inputs(
    *,
    role_name: str = 'implementer',
    task_metadata: dict | None = None,
    plan_shape: PlanShape | None = None,
    routing_tier: int = 0,
    dispatch_count: int = 0,
    role_defaults: RoleDefaults | None = None,
) -> RouteInputs:
    """A RouteInputs with sensible defaults; role_defaults defaults to a
    'sonnet' sentinel distinguishable from any rule's opus/haiku set.model."""
    return RouteInputs(
        role_name=role_name,
        task_id='1',
        task_metadata={} if task_metadata is None else task_metadata,
        plan_shape=plan_shape,
        routing_tier=routing_tier,
        dispatch_count=dispatch_count,
        role_defaults=role_defaults if role_defaults is not None else _role_defaults(model='sonnet'),
    )


class TestPlanShapeIsPureData:
    """Sanity: PlanShape is a frozen dataclass carrying step_count + module_paths."""

    def test_plan_shape_fields(self):
        shape = PlanShape(step_count=12, module_paths=('crates/a', 'crates/b'))
        assert shape.step_count == 12
        assert shape.module_paths == ('crates/a', 'crates/b')


class TestLayerPrecedenceRoleDefault:
    """(a) invariant 1 (Total): no config field, no rules, no override ->
    RoutingDecision built entirely from role_defaults."""

    def test_unknown_role_falls_back_to_role_default(self):
        cfg = OrchestratorConfig()
        inputs = RouteInputs(
            role_name='not_a_real_role',
            task_id='1',
            task_metadata={},
            plan_shape=None,
            routing_tier=0,
            dispatch_count=0,
            role_defaults=_role_defaults(
                model='sonnet', effort='medium', budget_usd=3.0, max_turns=20,
            ),
        )

        decision = resolve_route(inputs, cfg)

        assert isinstance(decision, RoutingDecision)
        assert decision.model == 'sonnet'
        assert decision.effort == 'medium'
        assert decision.budget_usd == 3.0
        assert decision.max_turns == 20
        assert decision.source_layer == 'role_default'
        assert decision.rule_id is None
        assert decision.rejected == ()


class TestLayerPrecedenceConfig:
    """(b) config.models/budgets/max_turns/effort.<role> wins over role_default."""

    def test_config_layer_wins_over_role_default(self):
        cfg = OrchestratorConfig(
            models=ModelsConfig(implementer='haiku'),
            budgets=BudgetsConfig(implementer=7.0),
            max_turns=TurnsConfig(implementer=40),
            effort=EffortConfig(implementer='low'),
        )
        inputs = RouteInputs(
            role_name='implementer',
            task_id='1',
            task_metadata={},
            plan_shape=None,
            routing_tier=0,
            dispatch_count=0,
            role_defaults=_role_defaults(),
        )

        decision = resolve_route(inputs, cfg)

        assert decision.model == 'haiku'
        assert decision.effort == 'low'
        assert decision.budget_usd == 7.0
        assert decision.max_turns == 40
        assert decision.source_layer == 'config'
        assert decision.rule_id is None


class TestLayerPrecedenceMetadataOverride:
    """(c) boundary test 1: task_metadata['model_overrides'][role] wins over
    both config and role_default when the overriding model is allowed."""

    def test_metadata_override_wins(self):
        cfg = OrchestratorConfig(models=ModelsConfig(implementer='opus'))
        inputs = RouteInputs(
            role_name='implementer',
            task_id='1',
            task_metadata={'model_overrides': {'implementer': 'haiku'}},
            plan_shape=None,
            routing_tier=0,
            dispatch_count=0,
            role_defaults=_role_defaults(),
        )

        decision = resolve_route(inputs, cfg)

        assert decision.model == 'haiku'
        assert decision.source_layer == 'metadata_override'


class TestLayerPrecedencePolicyRule:
    """(d) a matching policy rule wins over config (but is beaten by a
    metadata override, per (c) above -- not exercised in this test)."""

    def test_matching_rule_wins_over_config(self):
        rule = RoutingRule(
            id='force-opus',
            match=RuleMatch(role=['implementer']),
            set=RuleSet(model='opus'),
        )
        cfg = OrchestratorConfig(
            models=ModelsConfig(implementer='sonnet'),
            routing=RoutingConfig(rules=[rule]),
        )
        inputs = RouteInputs(
            role_name='implementer',
            task_id='1',
            task_metadata={},
            plan_shape=None,
            routing_tier=0,
            dispatch_count=0,
            role_defaults=_role_defaults(model='sonnet'),
        )

        decision = resolve_route(inputs, cfg)

        assert decision.model == 'opus'
        assert decision.source_layer == 'policy_rule'
        assert decision.rule_id == 'force-opus'

    def test_non_matching_rule_does_not_apply(self):
        rule = RoutingRule(
            id='force-opus',
            match=RuleMatch(role=['debugger']),
            set=RuleSet(model='opus'),
        )
        cfg = OrchestratorConfig(
            models=ModelsConfig(implementer='sonnet'),
            routing=RoutingConfig(rules=[rule]),
        )
        inputs = RouteInputs(
            role_name='implementer',
            task_id='1',
            task_metadata={},
            plan_shape=None,
            routing_tier=0,
            dispatch_count=0,
            role_defaults=_role_defaults(model='sonnet'),
        )

        decision = resolve_route(inputs, cfg)

        assert decision.model == 'sonnet'
        assert decision.source_layer == 'config'
        assert decision.rule_id is None


# ---------------------------------------------------------------------------
# step-5: closed condition-vocabulary matching (RED until step-6's evaluator)
#
# Each condition is exercised match vs non-match via a single policy rule
# whose `set={model: 'opus'}` -- `rule_id is not None` <=> the rule matched.
# role_defaults/config always resolve to 'sonnet' (never 'opus'), so 'opus'
# in the resolved model is unambiguous proof the rule fired.
# ---------------------------------------------------------------------------


class TestConditionTaskComplexity:
    """task_complexity: equality vs task_metadata['complexity']."""

    def test_matches_when_equal(self):
        rule = RoutingRule(id='r', match=RuleMatch(task_complexity='simple'), set=RuleSet(model='opus'))
        decision = resolve_route(
            _inputs(task_metadata={'complexity': 'simple'}), _cfg_with_rules(rule),
        )
        assert decision.model == 'opus'
        assert decision.rule_id == 'r'

    def test_does_not_match_when_different(self):
        rule = RoutingRule(id='r', match=RuleMatch(task_complexity='simple'), set=RuleSet(model='opus'))
        decision = resolve_route(
            _inputs(task_metadata={'complexity': 'full'}), _cfg_with_rules(rule),
        )
        assert decision.model == 'sonnet'
        assert decision.rule_id is None


class TestConditionTaskPriority:
    """task_priority: equality vs task_metadata['priority']."""

    def test_matches_when_equal(self):
        rule = RoutingRule(id='r', match=RuleMatch(task_priority='critical'), set=RuleSet(model='opus'))
        decision = resolve_route(
            _inputs(task_metadata={'priority': 'critical'}), _cfg_with_rules(rule),
        )
        assert decision.model == 'opus'
        assert decision.rule_id == 'r'

    def test_does_not_match_when_different(self):
        rule = RoutingRule(id='r', match=RuleMatch(task_priority='critical'), set=RuleSet(model='opus'))
        decision = resolve_route(
            _inputs(task_metadata={'priority': 'low'}), _cfg_with_rules(rule),
        )
        assert decision.model == 'sonnet'
        assert decision.rule_id is None


class TestConditionPlanMinSteps:
    """plan_min_steps: plan_shape.step_count >= N; None plan_shape -> no match."""

    def test_matches_when_step_count_at_or_above_threshold(self):
        rule = RoutingRule(id='r', match=RuleMatch(plan_min_steps=12), set=RuleSet(model='opus'))
        decision = resolve_route(
            _inputs(plan_shape=PlanShape(step_count=12, module_paths=())),
            _cfg_with_rules(rule),
        )
        assert decision.rule_id == 'r'

    def test_does_not_match_when_step_count_below_threshold(self):
        rule = RoutingRule(id='r', match=RuleMatch(plan_min_steps=12), set=RuleSet(model='opus'))
        decision = resolve_route(
            _inputs(plan_shape=PlanShape(step_count=11, module_paths=())),
            _cfg_with_rules(rule),
        )
        assert decision.rule_id is None

    def test_does_not_match_when_plan_shape_is_none(self):
        rule = RoutingRule(id='r', match=RuleMatch(plan_min_steps=12), set=RuleSet(model='opus'))
        decision = resolve_route(_inputs(plan_shape=None), _cfg_with_rules(rule))
        assert decision.rule_id is None


class TestConditionPlanMinModulesUnscoped:
    """plan_min_modules alone (no module_prefix): total module count >= N."""

    def test_matches_when_module_count_at_threshold(self):
        rule = RoutingRule(id='r', match=RuleMatch(plan_min_modules=3), set=RuleSet(model='opus'))
        decision = resolve_route(
            _inputs(plan_shape=PlanShape(step_count=1, module_paths=('a', 'b', 'c'))),
            _cfg_with_rules(rule),
        )
        assert decision.rule_id == 'r'

    def test_does_not_match_when_module_count_below_threshold(self):
        rule = RoutingRule(id='r', match=RuleMatch(plan_min_modules=3), set=RuleSet(model='opus'))
        decision = resolve_route(
            _inputs(plan_shape=PlanShape(step_count=1, module_paths=('a', 'b'))),
            _cfg_with_rules(rule),
        )
        assert decision.rule_id is None


class TestConditionModulePrefix:
    """module_prefix alone: >=1 module startswith prefix."""

    def test_matches_when_at_least_one_module_has_prefix(self):
        rule = RoutingRule(id='r', match=RuleMatch(module_prefix='crates/'), set=RuleSet(model='opus'))
        decision = resolve_route(
            _inputs(plan_shape=PlanShape(step_count=1, module_paths=('crates/a', 'lib/b'))),
            _cfg_with_rules(rule),
        )
        assert decision.rule_id == 'r'

    def test_does_not_match_when_no_module_has_prefix(self):
        rule = RoutingRule(id='r', match=RuleMatch(module_prefix='crates/'), set=RuleSet(model='opus'))
        decision = resolve_route(
            _inputs(plan_shape=PlanShape(step_count=1, module_paths=('lib/a', 'lib/b'))),
            _cfg_with_rules(rule),
        )
        assert decision.rule_id is None


class TestConditionModulePrefixScopedPlanMinModules:
    """module_prefix + plan_min_modules together: count of PREFIX-MATCHED
    modules >= N (not the total module count) -- reproduces the Rust
    heuristic exactly."""

    def test_three_prefixed_plus_one_other_matches_threshold_three(self):
        rule = RoutingRule(
            id='r',
            match=RuleMatch(plan_min_modules=3, module_prefix='crates/'),
            set=RuleSet(model='opus'),
        )
        decision = resolve_route(
            _inputs(
                plan_shape=PlanShape(
                    step_count=1,
                    module_paths=('crates/a', 'crates/b', 'crates/c', 'lib/other'),
                ),
            ),
            _cfg_with_rules(rule),
        )
        assert decision.rule_id == 'r'

    def test_two_prefixed_does_not_match_threshold_three(self):
        rule = RoutingRule(
            id='r',
            match=RuleMatch(plan_min_modules=3, module_prefix='crates/'),
            set=RuleSet(model='opus'),
        )
        decision = resolve_route(
            _inputs(
                plan_shape=PlanShape(
                    step_count=1,
                    module_paths=('crates/a', 'crates/b', 'lib/other', 'lib/another'),
                ),
            ),
            _cfg_with_rules(rule),
        )
        assert decision.rule_id is None


class TestConditionMinRoutingTier:
    """min_routing_tier: inputs.routing_tier >= N."""

    def test_matches_when_tier_at_or_above_threshold(self):
        rule = RoutingRule(id='r', match=RuleMatch(min_routing_tier=1), set=RuleSet(model='opus'))
        decision = resolve_route(_inputs(routing_tier=1), _cfg_with_rules(rule))
        assert decision.rule_id == 'r'

    def test_does_not_match_when_tier_below_threshold(self):
        rule = RoutingRule(id='r', match=RuleMatch(min_routing_tier=1), set=RuleSet(model='opus'))
        decision = resolve_route(_inputs(routing_tier=0), _cfg_with_rules(rule))
        assert decision.rule_id is None


class TestConditionMinDispatchCount:
    """min_dispatch_count: inputs.dispatch_count >= N."""

    def test_matches_when_dispatch_count_at_or_above_threshold(self):
        rule = RoutingRule(id='r', match=RuleMatch(min_dispatch_count=2), set=RuleSet(model='opus'))
        decision = resolve_route(_inputs(dispatch_count=2), _cfg_with_rules(rule))
        assert decision.rule_id == 'r'

    def test_does_not_match_when_dispatch_count_below_threshold(self):
        rule = RoutingRule(id='r', match=RuleMatch(min_dispatch_count=2), set=RuleSet(model='opus'))
        decision = resolve_route(_inputs(dispatch_count=1), _cfg_with_rules(rule))
        assert decision.rule_id is None


class TestConditionSimpleSaturated:
    """simple_saturated: equality vs task_metadata['routing']['simple_saturated']
    (mirrors shared.task_metadata.RoutingState's on-disk storage shape)."""

    def test_matches_when_true_and_saturated(self):
        rule = RoutingRule(id='r', match=RuleMatch(simple_saturated=True), set=RuleSet(model='opus'))
        decision = resolve_route(
            _inputs(task_metadata={'routing': {'simple_saturated': True}}),
            _cfg_with_rules(rule),
        )
        assert decision.rule_id == 'r'

    def test_does_not_match_when_true_and_not_saturated(self):
        rule = RoutingRule(id='r', match=RuleMatch(simple_saturated=True), set=RuleSet(model='opus'))
        decision = resolve_route(
            _inputs(task_metadata={'routing': {'simple_saturated': False}}),
            _cfg_with_rules(rule),
        )
        assert decision.rule_id is None

    def test_does_not_match_when_true_and_metadata_absent(self):
        rule = RoutingRule(id='r', match=RuleMatch(simple_saturated=True), set=RuleSet(model='opus'))
        decision = resolve_route(_inputs(task_metadata={}), _cfg_with_rules(rule))
        assert decision.rule_id is None


class TestConditionsAreAnded:
    """Two conditions on one rule must BOTH hold -- one satisfied + one not
    is a non-match."""

    def test_both_conditions_hold_matches(self):
        rule = RoutingRule(
            id='r',
            match=RuleMatch(role=['implementer'], min_routing_tier=1),
            set=RuleSet(model='opus'),
        )
        decision = resolve_route(
            _inputs(role_name='implementer', routing_tier=1), _cfg_with_rules(rule),
        )
        assert decision.rule_id == 'r'

    def test_one_condition_failing_is_a_non_match(self):
        rule = RoutingRule(
            id='r',
            match=RuleMatch(role=['implementer'], min_routing_tier=1),
            set=RuleSet(model='opus'),
        )
        decision = resolve_route(
            _inputs(role_name='implementer', routing_tier=0), _cfg_with_rules(rule),
        )
        assert decision.rule_id is None


class TestFirstMatchWins:
    """When two rules both match, the FIRST in list order supplies the
    overrides and its id is recorded -- the second is never consulted."""

    def test_first_of_two_matching_rules_wins(self):
        first = RoutingRule(id='first', match=RuleMatch(role=['implementer']), set=RuleSet(model='opus'))
        second = RoutingRule(id='second', match=RuleMatch(role=['implementer']), set=RuleSet(model='haiku'))
        decision = resolve_route(
            _inputs(role_name='implementer'), _cfg_with_rules(first, second),
        )
        assert decision.model == 'opus'
        assert decision.rule_id == 'first'
