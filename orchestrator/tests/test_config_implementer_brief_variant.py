"""The implementer brief variant: its closed vocabulary, the two bound policies
each member implies, and the green-tier config knob that selects one
(task 5385, plans/live-shadow-eval-prd.md decision 14).

Fixtures are kept MODULE-LOCAL (not conftest.py) for the reason stated in
test_config_pre_merge_reverify_reload.py: a conftest.py edit forces the
merge-time verify to run the full owning-package suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import yaml
from pydantic import ValidationError

from orchestrator.config import (
    RELOADABLE_FIELDS,
    OrchestratorConfig,
    apply_reload,
    census_config_keys,
    diff_config,
)
from orchestrator.implementer_brief_variant import (
    ImplementerBriefVariant,
    execute_iteration_cap,
    invocation_bounds,
)


class TestVocabulary:
    def test_members_are_the_stored_values(self):
        assert {m.value for m in ImplementerBriefVariant} == {'whole-plan', 'one-step'}

    def test_an_underscore_spelling_is_not_a_member(self):
        with pytest.raises(ValueError):
            ImplementerBriefVariant('one_step')


def _plan_must_not_be_read():
    raise AssertionError('whole-plan must not read plan.json')


class TestExecuteIterationCap:
    def test_whole_plan_returns_the_configured_bound_without_reading_the_plan(self):
        cap = execute_iteration_cap(
            ImplementerBriefVariant.WHOLE_PLAN, 10, _plan_must_not_be_read,
        )
        assert cap == 10

    def test_one_step_counts_every_dict_item_plus_slack(self):
        plan = {
            'prerequisites': [{'id': 'pre-1'}],
            'steps': [{'id': 'step-1'}, {'id': 'step-2'}, {'id': 'step-3'}, 'junk'],
        }
        cap = execute_iteration_cap(ImplementerBriefVariant.ONE_STEP, 10, lambda: plan)
        assert cap == 7

    def test_one_step_on_an_empty_plan_is_the_slack(self):
        cap = execute_iteration_cap(ImplementerBriefVariant.ONE_STEP, 10, lambda: {})
        assert cap == 3


class TestInvocationBounds:
    @pytest.mark.parametrize('role', ['implementer', 'debugger', 'architect', 'simple_task'])
    def test_whole_plan_is_the_identity_for_every_role(self, role):
        assert invocation_bounds(ImplementerBriefVariant.WHOLE_PLAN, role, 10.0, 80) == (10.0, 80)

    @pytest.mark.parametrize(
        ('budget_usd', 'max_turns', 'expected'),
        [
            (10.0, 80, (10.0 / 3, 27)),
            (20.0, 160, (20.0 / 3, 54)),
            (0.5, 1, (0.5 / 3, 1)),
        ],
    )
    def test_one_step_divides_the_implementer_bounds(self, budget_usd, max_turns, expected):
        bounds = invocation_bounds(
            ImplementerBriefVariant.ONE_STEP, 'implementer', budget_usd, max_turns,
        )
        assert bounds == expected

    @pytest.mark.parametrize('role', ['debugger', 'architect', 'simple_task'])
    def test_one_step_leaves_other_roles_alone(self, role):
        assert invocation_bounds(ImplementerBriefVariant.ONE_STEP, role, 10.0, 80) == (10.0, 80)

    def test_a_non_member_variant_falls_through_to_whole_plan(self):
        assert invocation_bounds(MagicMock(), 'implementer', 10.0, 80) == (10.0, 80)


@pytest.mark.usefixtures("code_default_config")
class TestConfigKnob:
    def test_code_default_is_whole_plan(self):
        variant = OrchestratorConfig().implementer_brief_variant
        assert variant is ImplementerBriefVariant.WHOLE_PLAN

    def test_the_stored_value_parses_to_the_member(self):
        config = OrchestratorConfig.model_validate({'implementer_brief_variant': 'one-step'})
        assert config.implementer_brief_variant is ImplementerBriefVariant.ONE_STEP

    def test_a_non_member_is_rejected_naming_the_field(self):
        with pytest.raises(ValidationError) as excinfo:
            OrchestratorConfig.model_validate({'implementer_brief_variant': 'one_step'})
        assert 'implementer_brief_variant' in str(excinfo.value)


@pytest.mark.usefixtures("code_default_config")
class TestGreenTier:
    def test_field_is_reloadable(self):
        assert 'implementer_brief_variant' in RELOADABLE_FIELDS

    def test_edit_lands_in_applied_candidates_not_restart_required(self):
        diff = diff_config(
            OrchestratorConfig(),
            OrchestratorConfig(implementer_brief_variant=ImplementerBriefVariant.ONE_STEP),
        )
        assert 'implementer_brief_variant' in diff.applied_candidates
        assert 'implementer_brief_variant' not in diff.restart_required

    def test_apply_reload_flips_the_live_object_in_place(self):
        live = OrchestratorConfig()
        report = apply_reload(
            live, OrchestratorConfig(implementer_brief_variant=ImplementerBriefVariant.ONE_STEP),
        )
        assert report['reloaded'] is True
        assert report['applied'] == {
            'implementer_brief_variant': {'old': 'whole-plan', 'new': 'one-step'},
        }
        assert report['restart_required'] == {}
        assert live.implementer_brief_variant is ImplementerBriefVariant.ONE_STEP


def test_check_config_census_knows_the_key(tmp_path):
    path = tmp_path / 'orchestrator.yaml'
    path.write_text(yaml.safe_dump({'implementer_brief_variant': 'one-step'}))
    unknown_paths = [key.path for key in census_config_keys(path).unknown]
    assert 'implementer_brief_variant' not in unknown_paths
