"""Byte-equivalence tests for orchestrator.routing.resolve_route (task
epsilon, plans/adaptive-model-routing-prd.md, invariant 3 / boundary test 3).

RED (step-9) until defaults.yaml carries the shipped `rust-large-plan-
implementer` policy rule (step-10): the Rust-heuristic sub-tests
(TestRustHeuristicRuleParity) fail until then -- with no rules configured,
resolve_route's policy_rule layer (landed step-4/6/8) is a no-op, so a
matching PlanShape never upgrades the model. The flat per-role
byte-equivalence sub-tests (TestByteEquivalencePerRole) already pass today
(resolve_route's config layer alone reproduces _invoke's pre-epsilon
per-role config resolution when no rule/override applies) but are locked in
here as the single source of truth for invariant 3 going forward.

Task theta (Phase 3 of the same PRD) later generalized defaults.yaml's
single crates/-only `rust-large-plan-implementer` rule into two
language-agnostic rules (`large-plan-steps` / `large-plan-modules`) --
TestRustHeuristicRuleParity now guards only the regression case (a
Rust-shaped plan still resolves opus, via the new `large-plan-steps` rule);
see test_routing_theta_plan_shape.py for theta's own language-agnostic
model-upgrade coverage.

Experiment 1 of plans/routing-experiment-substrate-prd.md (task 8375) puts a
split on the shipped table: the >=3-module lift now holds on arm 0 of salt
routing-exp-1 and arm 1 needs >=4 modules. So the task id moves a decision
for an implementer or debugger with a >=3-module, under-12-step plan and
nowhere else: the outcome for exactly 3 modules, and only the attributed
rule id for 4 or more (TestExperimentOneRuleTable,
TestExperimentOneResolution, TestTaskIdMovesOnlyTheSplitRules).

Fixtures are kept MODULE-LOCAL (not conftest.py) -- mirrors test_routing.py's
documented rationale (a conftest.py edit forces verify.py's has_conftest to
widen the merge-time scoped-test selection to the whole owning package).
"""

from __future__ import annotations

import pytest
from _routing_arms import task_id_in_rule_arm

from orchestrator.config import OrchestratorConfig, RoutingRule, SplitMatch
from orchestrator.routing import PlanShape, RoleDefaults, RouteInputs, resolve_route

pytestmark = pytest.mark.usefixtures("code_default_config")

# Every role _invoke can dispatch today (task epsilon boundary test 3) -- the
# 11 full-name config-addressable roles (task alpha) plus one representative
# reviewer* variant, proving the reviewer-prefix collapse holds through the
# resolver exactly as it does in _invoke today.
ALL_DISPATCHABLE_ROLES: tuple[str, ...] = (
    'architect',
    'implementer',
    'debugger',
    'reviewer',
    'reviewer_comprehensive',
    'merger',
    'steward',
    'triage',
    'module_tagger',
    'deep_reviewer',
    'judge',
    'simple_task',
)

# Deliberately nonsense -- config always wins for these 12 known roles (each
# has a config field), so a sentinel role_defaults value makes a silent
# role_default-layer fallback (a bug where the config layer fails to apply)
# fail loudly instead of accidentally coinciding with a real config value.
_SENTINEL_ROLE_DEFAULTS = RoleDefaults(
    model='role-default-sentinel', effort='role-default-sentinel',
    budget_usd=-1.0, max_turns=-1,
)


def _config_key(role_name: str) -> str:
    """Reviewer* collapse, reconstructed independently of
    ``orchestrator.routing._config_key`` -- see this module's docstring:
    invariant 3 is defined as matching "the same config fields (getattr(...,
    key with reviewer collapse))", so the expected side must not import the
    resolver's own private helper (that would mask a bug in the helper
    itself)."""
    return 'reviewer' if role_name.startswith('reviewer') else role_name


def _expected_from_config(cfg: OrchestratorConfig, role_name: str) -> tuple[str, str, float, int]:
    """The pre-epsilon resolution: today's _invoke reads config.models/
    effort/budgets/max_turns by the reviewer-collapsed key, with no further
    logic, whenever no Rust-heuristic upgrade and no override apply."""
    key = _config_key(role_name)
    return (
        getattr(cfg.models, key),
        getattr(cfg.effort, key),
        getattr(cfg.budgets, key),
        getattr(cfg.max_turns, key),
    )


def _inputs_for(
    role_name: str, *, plan_shape: PlanShape | None = None, task_id: str = '1',
) -> RouteInputs:
    return RouteInputs(
        role_name=role_name,
        task_id=task_id,
        task_metadata={},
        plan_shape=plan_shape,
        routing_tier=0,
        dispatch_count=0,
        role_defaults=_SENTINEL_ROLE_DEFAULTS,
    )


class TestByteEquivalencePerRole:
    """Invariant 3: with empty model_overrides, the shipped default rules,
    and stock config, resolve_route returns exactly (model, effort,
    budget_usd, max_turns) == today's config-layer resolution, for every
    dispatchable role -- including the reviewer* collapse."""

    @pytest.mark.parametrize('role_name', ALL_DISPATCHABLE_ROLES)
    def test_matches_config_layer_resolution(self, role_name):
        cfg = OrchestratorConfig()

        decision = resolve_route(_inputs_for(role_name), cfg)

        expected = _expected_from_config(cfg, role_name)
        assert (decision.model, decision.effort, decision.budget_usd, decision.max_turns) == expected
        assert decision.source_layer == 'config'
        assert decision.rule_id is None
        assert decision.rejected == ()


class TestRustHeuristicRuleParity:
    """A plan matching the retired Rust-only shape (role in (implementer,
    debugger) AND >=3 'crates/' modules AND step_count >= 12) still resolves
    opus -- a regression guard for the pre-epsilon `_select_model_for_role`
    heuristic this rule originally transcribed byte-for-byte.

    Task theta (plans/adaptive-model-routing-prd.md Phase 3) replaced the
    single crates/-prefixed `rust-large-plan-implementer` rule in
    defaults.yaml with two language-agnostic rules, `large-plan-steps` and
    `large-plan-modules` (see test_routing_theta_plan_shape.py). The general
    `large-plan-steps` rule strictly subsumes the old Rust-only shape (>=3
    crates/ modules AND >=12 steps implies >=12 steps), so a Rust-shaped
    plan now resolves via `large-plan-steps` rather than the retired rule.
    This class used to also carry two `does_not_fire` sub-tests
    (two-crates-modules, eleven-steps) -- theta's language-agnostic
    conditions make both of those inputs legitimately fire now, so they were
    removed; see test_routing_theta_plan_shape.py's steps/modules trigger
    coverage instead.
    """

    RUST_PLAN = PlanShape(
        step_count=12, module_paths=('crates/a', 'crates/b', 'crates/c', 'lib/other'),
    )

    @pytest.mark.parametrize('role_name', ['implementer', 'debugger'])
    def test_fires_for_implementer_and_debugger(self, role_name):
        cfg = OrchestratorConfig()

        decision = resolve_route(_inputs_for(role_name, plan_shape=self.RUST_PLAN), cfg)

        assert decision.model == 'opus'
        assert decision.source_layer == 'policy_rule'
        assert decision.rule_id == 'large-plan-steps'

    def test_does_not_fire_for_architect(self):
        cfg = OrchestratorConfig()

        decision = resolve_route(_inputs_for('architect', plan_shape=self.RUST_PLAN), cfg)

        assert decision.model == cfg.models.architect
        assert decision.source_layer != 'policy_rule'


def _rule(cfg: OrchestratorConfig, rule_id: str) -> RoutingRule:
    return next(r for r in cfg.routing.rules if r.id == rule_id)


_THREE_MODULE_PLAN = PlanShape(5, ('a', 'b', 'c'))
_FOUR_MODULE_PLAN = PlanShape(5, ('a', 'b', 'c', 'd'))
_TWELVE_STEP_PLAN = PlanShape(12, ('a',))


class TestExperimentOneRuleTable:
    """The shipped table carries experiment 1 of
    plans/routing-experiment-substrate-prd.md: large-plan-modules-4 (>=4
    modules) on arm 1 of split salt routing-exp-1 and today's >=3-module
    rule on arm 0, both after the unsplit large-plan-steps."""

    def test_rule_ids_in_order(self):
        cfg = OrchestratorConfig()

        assert [r.id for r in cfg.routing.rules] == [
            'large-plan-steps', 'large-plan-modules-4', 'large-plan-modules',
            'simple-saturated-full-path', 'retry-tier-up',
        ]

    def test_arm_one_rule_raises_the_module_threshold_to_four(self):
        rule = _rule(OrchestratorConfig(), 'large-plan-modules-4')

        assert rule.match.role == ['implementer', 'debugger']
        assert rule.match.plan_min_modules == 4
        assert rule.match.split == SplitMatch(salt='routing-exp-1', buckets=2, bucket=1)
        assert (rule.set.model, rule.set.max_turns) == ('opus', 120)

    def test_arm_zero_rule_keeps_the_three_module_threshold(self):
        rule = _rule(OrchestratorConfig(), 'large-plan-modules')

        assert rule.match.role == ['implementer', 'debugger']
        assert rule.match.plan_min_modules == 3
        assert rule.match.split == SplitMatch(salt='routing-exp-1', buckets=2, bucket=0)
        assert (rule.set.model, rule.set.max_turns) == ('opus', 120)

    def test_large_plan_steps_is_not_split(self):
        assert _rule(OrchestratorConfig(), 'large-plan-steps').match.split is None


@pytest.mark.parametrize('role_name', ['implementer', 'debugger'])
class TestExperimentOneResolution:
    """How the shipped table resolves each arm's task ids."""

    def _resolve(self, role_name: str, plan_shape: PlanShape, arm_rule_id: str):
        cfg = OrchestratorConfig()
        task_id = task_id_in_rule_arm(cfg, arm_rule_id)
        return cfg, resolve_route(
            _inputs_for(role_name, plan_shape=plan_shape, task_id=task_id), cfg,
        )

    def test_three_modules_on_arm_zero_lifts_to_opus(self, role_name):
        _, decision = self._resolve(role_name, _THREE_MODULE_PLAN, 'large-plan-modules')

        assert decision.rule_id == 'large-plan-modules'
        assert (decision.model, decision.max_turns) == ('opus', 120)
        assert decision.source_layer == 'policy_rule'

    def test_three_modules_on_arm_one_falls_through_to_config(self, role_name):
        cfg, decision = self._resolve(role_name, _THREE_MODULE_PLAN, 'large-plan-modules-4')

        assert decision.rule_id is None
        assert decision.source_layer == 'config'
        assert decision.model == getattr(cfg.models, role_name)
        assert decision.max_turns == getattr(cfg.max_turns, role_name)

    def test_four_modules_on_arm_one_lifts_via_the_four_module_rule(self, role_name):
        _, decision = self._resolve(role_name, _FOUR_MODULE_PLAN, 'large-plan-modules-4')

        assert decision.rule_id == 'large-plan-modules-4'
        assert (decision.model, decision.max_turns) == ('opus', 120)

    def test_four_modules_on_arm_zero_lifts_via_the_three_module_rule(self, role_name):
        _, decision = self._resolve(role_name, _FOUR_MODULE_PLAN, 'large-plan-modules')

        assert decision.rule_id == 'large-plan-modules'
        assert (decision.model, decision.max_turns) == ('opus', 120)

    @pytest.mark.parametrize('arm_rule_id', ['large-plan-modules', 'large-plan-modules-4'])
    def test_twelve_steps_lifts_via_large_plan_steps_in_both_arms(self, role_name, arm_rule_id):
        _, decision = self._resolve(role_name, _TWELVE_STEP_PLAN, arm_rule_id)

        assert decision.rule_id == 'large-plan-steps'


_TASK_IDS = [str(i) for i in range(1, 201)]


def _decisions_over_task_ids(role_name: str, plan_shape: PlanShape | None):
    cfg = OrchestratorConfig()
    return [
        resolve_route(_inputs_for(role_name, plan_shape=plan_shape, task_id=task_id), cfg)
        for task_id in _TASK_IDS
    ]


class TestTaskIdMovesOnlyTheSplitRules:
    """Invariant 3, extended: the task id changes a decision only where an
    experiment-1 split rule can match -- an implementer or debugger with a
    plan of >=3 modules and under 12 steps. With exactly 3 modules it picks
    one of the two arm outcomes; with 4 or more only the attributed rule id
    differs."""

    @pytest.mark.parametrize(
        'plan_shape',
        [None, PlanShape(5, ('a',)), _THREE_MODULE_PLAN, _TWELVE_STEP_PLAN],
    )
    def test_architect_decision_is_the_same_for_every_task_id(self, plan_shape):
        assert len(set(_decisions_over_task_ids('architect', plan_shape))) == 1

    @pytest.mark.parametrize('role_name', ['implementer', 'debugger'])
    @pytest.mark.parametrize('plan_shape', [None, PlanShape(5, ('a',)), _TWELVE_STEP_PLAN])
    def test_executor_decision_is_the_same_for_every_task_id_outside_the_split(
        self, role_name, plan_shape,
    ):
        assert len(set(_decisions_over_task_ids(role_name, plan_shape))) == 1

    @pytest.mark.parametrize('role_name', ['implementer', 'debugger'])
    def test_three_module_executor_takes_exactly_the_two_arm_outcomes(self, role_name):
        rule_ids = {d.rule_id for d in _decisions_over_task_ids(role_name, _THREE_MODULE_PLAN)}

        assert rule_ids == {'large-plan-modules', None}

    @pytest.mark.parametrize('role_name', ['implementer', 'debugger'])
    def test_four_module_executor_differs_by_arm_only_in_rule_id(self, role_name):
        decisions = _decisions_over_task_ids(role_name, _FOUR_MODULE_PLAN)

        assert {d.rule_id for d in decisions} == {'large-plan-modules-4', 'large-plan-modules'}
        assert len({(d.model, d.max_turns) for d in decisions}) == 1
