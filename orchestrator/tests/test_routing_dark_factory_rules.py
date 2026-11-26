"""dark-factory stays out of experiment 1 of
plans/routing-experiment-substrate-prd.md (task 8375, amendment 2026-11-17).

defaults.yaml carries the experiment's split rules; dark-factory-orchestrator.yaml
keeps the pre-experiment table, because its configured implementer and debugger
turn caps (160/100) would make arm 1 change turns as well as the model. Copying
the split rules into dark-factory's yaml to "keep it in step" turns this red.

The file is loaded explicitly through ORCH_CONFIG_PATH, not through the autouse
binding. Assertions are on rule semantics (opus, 120 turns) and on an ordered
subsequence of today's rule ids, so rows another experiment inserts ahead of
them with the same lift keep this green.
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
def test_every_task_id_keeps_the_three_module_lift(dark_factory_config, role_name):
    decisions = {
        task_id: resolve_route(
            RouteInputs(
                role_name=role_name,
                task_id=task_id,
                task_metadata={},
                plan_shape=PlanShape(5, ('a', 'b', 'c')),
                routing_tier=0,
                dispatch_count=0,
                role_defaults=_SENTINEL_ROLE_DEFAULTS,
            ),
            dark_factory_config,
        )
        for task_id in (str(i) for i in range(1, 201))
    }

    unlifted = {
        task_id: (d.source_layer, d.model, d.max_turns)
        for task_id, d in decisions.items()
        if (d.source_layer, d.model, d.max_turns) != ('policy_rule', 'opus', 120)
    }
    assert unlifted == {}
