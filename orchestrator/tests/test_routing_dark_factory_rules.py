"""dark-factory stays out of experiment 1 of
plans/routing-experiment-substrate-prd.md (task 8375, amendment 2026-11-17).

defaults.yaml carries the experiment's split rules; dark-factory-orchestrator.yaml
keeps the pre-experiment table, because its configured turn caps
(`max_turns.implementer`, `max_turns.debugger`) differ from the large-plan
rules' 120, so arm 1 would change turns as well as the model. Copying the split
rules into dark-factory's yaml to "keep it in step" turns this red.

The file is loaded explicitly through ORCH_CONFIG_PATH, not through the autouse
binding. Assertions are on an ordered subsequence of today's rule ids and on
every task id getting one and the same policy-rule decision for a 3-module plan,
not on that decision's model or turns, so a retune of the lift, or rows another
experiment inserts ahead of them with a uniform lift, keeps this green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.routing import PlanShape, RoleDefaults, RouteInputs, resolve_route

DARK_FACTORY_CONFIG = Path(__file__).resolve().parents[2] / 'dark-factory-orchestrator.yaml'

PRE_EXPERIMENT_RULE_IDS = (
    'large-plan-steps', 'large-plan-modules', 'simple-saturated-full-path', 'retry-tier-up',
)

_SENTINEL_ROLE_DEFAULTS = RoleDefaults(
    model='role-default-sentinel', effort='role-default-sentinel',
    budget_usd=-1.0, max_turns=-1,
)


@pytest.fixture
def dark_factory_config(monkeypatch: pytest.MonkeyPatch) -> OrchestratorConfig:
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(DARK_FACTORY_CONFIG))
    return OrchestratorConfig()


def _is_ordered_subsequence(wanted: tuple[str, ...], ids: list[str]) -> bool:
    remaining = iter(ids)
    return all(rule_id in remaining for rule_id in wanted)


def test_keeps_the_pre_experiment_rules_in_order(dark_factory_config):
    ids = [r.id for r in dark_factory_config.routing.rules]

    assert _is_ordered_subsequence(PRE_EXPERIMENT_RULE_IDS, ids), ids


def test_carries_no_experiment_one_rule(dark_factory_config):
    rules = dark_factory_config.routing.rules

    assert 'large-plan-modules-4' not in [r.id for r in rules]
    assert [r.id for r in rules if r.match.split and r.match.split.salt == 'routing-exp-1'] == []


@pytest.mark.parametrize('role_name', ['implementer', 'debugger'])
def test_every_task_id_gets_one_policy_rule_decision(dark_factory_config, role_name):
    outcomes = {
        (d.source_layer, d.model, d.max_turns)
        for d in (
            resolve_route(
                RouteInputs(
                    role_name=role_name,
                    task_id=str(i),
                    task_metadata={},
                    plan_shape=PlanShape(5, ('a', 'b', 'c')),
                    routing_tier=0,
                    dispatch_count=0,
                    role_defaults=_SENTINEL_ROLE_DEFAULTS,
                ),
                dark_factory_config,
            )
            for i in range(1, 201)
        )
    }

    assert len(outcomes) == 1, outcomes
    assert {source_layer for source_layer, _, _ in outcomes} == {'policy_rule'}
