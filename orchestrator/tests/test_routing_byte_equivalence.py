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

Fixtures are kept MODULE-LOCAL (not conftest.py) -- mirrors test_routing.py's
documented rationale (a conftest.py edit forces verify.py's has_conftest to
widen the merge-time scoped-test selection to the whole owning package).
"""

from __future__ import annotations

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.routing import PlanShape, RoleDefaults, RouteInputs, resolve_route

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


@pytest.fixture(autouse=True)
def _isolated_config_env(monkeypatch, tmp_path):
    """Every OrchestratorConfig() built in this module reads ONLY the
    package's shipped defaults.yaml -- no stray project config.yaml, no
    ambient ORCH_CONFIG_PATH. Mirrors test_routing.py's per-test convention.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')


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
    """The shipped `rust-large-plan-implementer` policy rule (defaults.yaml,
    step-10) must reproduce the pre-epsilon `_select_model_for_role` Rust
    heuristic exactly: role in (implementer, debugger) AND >=3 'crates/'
    modules AND step_count >= 12 -> opus; else the config model, unchanged.

    Per this task's design decisions, ONE shared rule id
    ('rust-large-plan-implementer') covers both implementer and debugger --
    resolve_route's rule_id does not itself differentiate by role; a
    per-role telemetry id (if any) is an _invoke-level detail.
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
        assert decision.rule_id == 'rust-large-plan-implementer'

    def test_does_not_fire_with_only_two_crates_modules(self):
        cfg = OrchestratorConfig()
        plan = PlanShape(step_count=12, module_paths=('crates/a', 'crates/b', 'lib/other'))

        decision = resolve_route(_inputs_for('implementer', plan_shape=plan), cfg)

        assert decision.model == cfg.models.implementer
        assert decision.source_layer != 'policy_rule'

    def test_does_not_fire_with_only_eleven_steps(self):
        cfg = OrchestratorConfig()
        plan = PlanShape(step_count=11, module_paths=('crates/a', 'crates/b', 'crates/c'))

        decision = resolve_route(_inputs_for('debugger', plan_shape=plan), cfg)

        assert decision.model == cfg.models.debugger
        assert decision.source_layer != 'policy_rule'

    def test_does_not_fire_for_architect(self):
        cfg = OrchestratorConfig()

        decision = resolve_route(_inputs_for('architect', plan_shape=self.RUST_PLAN), cfg)

        assert decision.model == cfg.models.architect
        assert decision.source_layer != 'policy_rule'
