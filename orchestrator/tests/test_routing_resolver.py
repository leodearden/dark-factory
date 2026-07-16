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
