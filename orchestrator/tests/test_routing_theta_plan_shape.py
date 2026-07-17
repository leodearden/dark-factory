"""Post-theta plan-shape fleet-rule tests (task theta, Phase 3 of
plans/adaptive-model-routing-prd.md -- generalizing the Rust-only
`rust-large-plan-implementer` heuristic into two language-agnostic rules).

RED (step-1) until defaults.yaml replaces the single crates/-prefixed
`rust-large-plan-implementer` policy rule with `large-plan-steps` and
`large-plan-modules` (step-2): every model-upgrade assertion below fails
against the current Rust-only defaults.yaml, because a non-Rust (no
`crates/` modules) large plan never satisfies the shipped rule's
`module_prefix: "crates/"` condition, so resolve_route's policy_rule layer
never fires for it.

Fixtures are kept MODULE-LOCAL (not conftest.py) -- mirrors
test_routing_byte_equivalence.py's documented rationale (a conftest.py edit
forces verify.py's has_conftest to widen the merge-time scoped-test
selection to the whole owning package).
"""

from __future__ import annotations

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.routing import PlanShape, RoleDefaults, RouteInputs, resolve_route

# Deliberately nonsense -- config/policy_rule always win for these known
# roles (each has a config field), so a sentinel role_defaults value makes a
# silent role_default-layer fallback (a bug where the config/policy_rule
# layer fails to apply) fail loudly instead of accidentally coinciding with
# a real config value. Mirrors test_routing_byte_equivalence.py's sentinel.
_SENTINEL_ROLE_DEFAULTS = RoleDefaults(
    model='role-default-sentinel', effort='role-default-sentinel',
    budget_usd=-1.0, max_turns=-1,
)


@pytest.fixture(autouse=True)
def _isolated_config_env(monkeypatch, tmp_path):
    """Every OrchestratorConfig() built in this module reads ONLY the
    package's shipped defaults.yaml -- no stray project config.yaml, no
    ambient ORCH_CONFIG_PATH. Mirrors test_routing_byte_equivalence.py's
    per-test convention.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')


def _inputs_for(role_name: str, *, plan_shape: PlanShape | None = None) -> RouteInputs:
    return RouteInputs(
        role_name=role_name,
        task_id='1',
        task_metadata={},
        plan_shape=plan_shape,
        routing_tier=0,
        dispatch_count=0,
        role_defaults=_SENTINEL_ROLE_DEFAULTS,
    )


# A non-Rust large plan that trips only the >=12-steps condition -- 2
# modules, neither under `crates/`. The retired Rust-only rule's
# `module_prefix: "crates/"` condition would never match these module paths.
STEPS_TRIGGER_PLAN = PlanShape(
    step_count=12,
    module_paths=('orchestrator/src/orchestrator', 'shared/src/shared'),
)

# A non-Rust large plan that trips only the >=3-modules condition -- 8 steps
# (<12), 3 modules, none under `crates/`: language-agnostic module count.
MODULES_TRIGGER_PLAN = PlanShape(
    step_count=8,
    module_paths=(
        'orchestrator/src/orchestrator',
        'fused-memory/src/fused_memory',
        'shared/src/shared',
    ),
)

# Neither condition trips: <12 steps, <3 modules.
SMALL_PLAN = PlanShape(
    step_count=8,
    module_paths=('orchestrator/src/orchestrator', 'shared/src/shared'),
)

# The retired Rust-only shape (>=3 'crates/' modules AND >=12 steps) --
# regression-guarded: it must still upgrade to opus, now via the general
# `large-plan-steps` rule rather than the retired crates/-prefixed rule.
RUST_PLAN = PlanShape(
    step_count=12, module_paths=('crates/a', 'crates/b', 'crates/c', 'lib/other'),
)


class TestLargePlanStepsRuleFires:
    """(a) A non-Rust plan with >=12 steps upgrades implementer/debugger to
    opus via the `large-plan-steps` rule, regardless of module language."""

    @pytest.mark.parametrize('role_name', ['implementer', 'debugger'])
    def test_steps_trigger_upgrades_to_opus(self, role_name):
        cfg = OrchestratorConfig()

        decision = resolve_route(_inputs_for(role_name, plan_shape=STEPS_TRIGGER_PLAN), cfg)

        assert decision.model == 'opus'
        assert decision.source_layer == 'policy_rule'
        assert decision.rule_id == 'large-plan-steps'


class TestLargePlanModulesRuleFires:
    """(b) A non-Rust plan with >=3 modules (and <12 steps) upgrades
    implementer/debugger to opus via the `large-plan-modules` rule."""

    @pytest.mark.parametrize('role_name', ['implementer', 'debugger'])
    def test_modules_trigger_upgrades_to_opus(self, role_name):
        cfg = OrchestratorConfig()

        decision = resolve_route(_inputs_for(role_name, plan_shape=MODULES_TRIGGER_PLAN), cfg)

        assert decision.model == 'opus'
        assert decision.source_layer == 'policy_rule'
        assert decision.rule_id == 'large-plan-modules'


class TestSmallPlanDoesNotFire:
    """(c) A plan under both thresholds (<12 steps, <3 modules) resolves the
    plain config model -- no policy rule matches."""

    def test_small_plan_resolves_config_model(self):
        cfg = OrchestratorConfig()

        decision = resolve_route(_inputs_for('implementer', plan_shape=SMALL_PLAN), cfg)

        assert decision.model == cfg.models.implementer
        assert decision.model == 'sonnet'
        assert decision.source_layer != 'policy_rule'


class TestArchitectExempt:
    """(d) A large plan (steps trigger) never upgrades `architect` -- the
    rules' `role` match list is [implementer, debugger] only."""

    def test_architect_not_upgraded_on_large_plan(self):
        cfg = OrchestratorConfig()

        decision = resolve_route(_inputs_for('architect', plan_shape=STEPS_TRIGGER_PLAN), cfg)

        assert decision.model == cfg.models.architect
        assert decision.source_layer != 'policy_rule'


class TestRustPlanRegression:
    """(e) The retired Rust-only shape (>=3 'crates/' modules AND >=12
    steps) still upgrades to opus -- now via the general `large-plan-steps`
    rule rather than the retired `rust-large-plan-implementer` rule."""

    @pytest.mark.parametrize('role_name', ['implementer', 'debugger'])
    def test_rust_shaped_plan_still_upgrades_to_opus(self, role_name):
        cfg = OrchestratorConfig()

        decision = resolve_route(_inputs_for(role_name, plan_shape=RUST_PLAN), cfg)

        assert decision.model == 'opus'
        assert decision.source_layer == 'policy_rule'
        assert decision.rule_id == 'large-plan-steps'
