"""resolve_route sends a one-step implementer a third of its resolved route
(task 5385, plans/live-shadow-eval-prd.md decision 14).

Configs are built under the conftest ``code_default_config`` fixture, so only
the package's defaults.yaml applies and the live yaml cannot leak in.
"""

from __future__ import annotations

import pytest

from orchestrator.agents.roles import ARCHITECT, DEBUGGER, IMPLEMENTER, SIMPLE_TASK, AgentRole
from orchestrator.config import OrchestratorConfig, RoutingRule
from orchestrator.implementer_brief_variant import ImplementerBriefVariant
from orchestrator.routing import RoleDefaults, RouteInputs, RoutingDecision, resolve_route

pytestmark = pytest.mark.usefixtures('code_default_config')


def _route(role: AgentRole, config: OrchestratorConfig) -> RoutingDecision:
    inputs = RouteInputs(
        role_name=role.name,
        task_id='1',
        task_metadata={},
        plan_shape=None,
        routing_tier=0,
        dispatch_count=0,
        role_defaults=RoleDefaults(
            model=role.default_model,
            effort='high',
            budget_usd=role.default_budget,
            max_turns=role.default_max_turns,
        ),
    )
    return resolve_route(inputs, config)


def test_whole_plan_implementer_route_is_the_configured_bounds():
    config = OrchestratorConfig()
    decision = _route(IMPLEMENTER, config)
    assert decision.budget_usd == config.budgets.implementer
    assert decision.max_turns == config.max_turns.implementer


def test_one_step_divides_the_implementer_bounds_and_keeps_model_and_effort():
    bounds = {'budgets': {'implementer': 20.0}, 'max_turns': {'implementer': 160}}
    whole_decision = _route(IMPLEMENTER, OrchestratorConfig(**bounds))
    one_step = ImplementerBriefVariant.ONE_STEP
    one_decision = _route(
        IMPLEMENTER, OrchestratorConfig(implementer_brief_variant=one_step, **bounds),
    )

    assert one_decision.budget_usd == 20.0 / 3
    assert one_decision.max_turns == 54
    assert one_decision.model == whole_decision.model
    assert one_decision.effort == whole_decision.effort


def test_one_step_scales_the_final_route_after_a_policy_rule():
    rule = RoutingRule.model_validate({
        'id': 'implementer-bounds',
        'match': {'role': ['implementer']},
        'set': {'max_turns': 90, 'budget_usd': 9.0},
    })
    config = OrchestratorConfig(implementer_brief_variant=ImplementerBriefVariant.ONE_STEP)
    config.routing.rules = [rule]

    decision = _route(IMPLEMENTER, config)

    assert decision.rule_id == 'implementer-bounds'
    assert decision.max_turns == 30
    assert decision.budget_usd == 3.0


@pytest.mark.parametrize('role', [DEBUGGER, ARCHITECT, SIMPLE_TASK], ids=lambda r: r.name)
def test_one_step_leaves_other_roles_routes_alone(role):
    one_step = OrchestratorConfig(implementer_brief_variant=ImplementerBriefVariant.ONE_STEP)
    assert _route(role, one_step) == _route(role, OrchestratorConfig())
