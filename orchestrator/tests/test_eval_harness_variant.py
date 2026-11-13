"""EvalConfig.harness_variant and the ``<name>@<variant>`` resolver
(task 5385, plans/live-shadow-eval-prd.md decision 14).

A variant that would silently not apply is refused rather than resolved
(INV-11): an unknown suffix, and one-step on a non-implementer candidate.
"""

from __future__ import annotations

import pytest
from _eval_config_lookup import named_eval_config

from orchestrator.config import OrchestratorConfig
from orchestrator.evals.configs import (
    ARCHITECT_EVAL_CONFIGS,
    EvalConfig,
    UnknownHarnessVariant,
    claude_endpoint_candidates,
    get_config_by_name,
)
from orchestrator.evals.runner import build_eval_orch_config
from orchestrator.implementer_brief_variant import ImplementerBriefVariant

BASE = 'claude-sonnet-max'
CARRIED_FIELDS = ('model', 'backend', 'effort', 'role', 'max_budget_usd', 'env_overrides')


class TestEvalConfigField:
    def test_default_is_whole_plan(self):
        config = EvalConfig('x', 'claude', 'sonnet', 'max')
        assert config.harness_variant is ImplementerBriefVariant.WHOLE_PLAN


def _first_endpoint_only_candidate() -> str:
    for candidate in claude_endpoint_candidates():
        if 'ANTHROPIC_BASE_URL' in candidate.env_overrides:
            return candidate.name
    raise AssertionError('claude_endpoint_candidates() carries no proxied endpoint')


class TestResolver:
    def test_a_suffixed_name_resolves_to_a_relabelled_copy_of_its_base(self):
        plain = named_eval_config(BASE)
        variant = named_eval_config(f'{BASE}@one-step')

        assert variant.name == f'{BASE}@one-step'
        assert variant.harness_variant is ImplementerBriefVariant.ONE_STEP
        for name in CARRIED_FIELDS:
            assert getattr(variant, name) == getattr(plain, name), name

    def test_the_registered_entry_is_never_mutated(self):
        named_eval_config(f'{BASE}@one-step')
        plain = named_eval_config(BASE)
        assert plain.name == BASE
        assert plain.harness_variant is ImplementerBriefVariant.WHOLE_PLAN

    def test_an_endpoint_candidate_resolves_too(self):
        name = _first_endpoint_only_candidate()
        variant = named_eval_config(f'{name}@one-step')
        assert variant.harness_variant is ImplementerBriefVariant.ONE_STEP
        assert variant.env_overrides == named_eval_config(name).env_overrides


class TestRefusal:
    @pytest.mark.parametrize(
        ('requested', 'offending'),
        [
            (f'{BASE}@bogus', 'bogus'),
            (f'{BASE}@', ''),
            (f'{BASE}@one-step@x', 'one-step@x'),
            ('no-such-config@bogus', 'bogus'),
        ],
    )
    def test_an_unknown_variant_is_refused_before_the_scan(self, requested, offending):
        with pytest.raises(UnknownHarnessVariant) as excinfo:
            get_config_by_name(requested)
        message = str(excinfo.value)
        assert repr(requested) in message
        assert repr(offending) in message
        for member in ImplementerBriefVariant:
            assert member.value in message
        assert isinstance(excinfo.value, ValueError)

    @pytest.mark.parametrize('requested', ['no-such-config@one-step', 'no-such-config'])
    def test_an_unregistered_base_is_still_a_miss(self, requested):
        assert get_config_by_name(requested) is None

    def test_one_step_on_a_non_implementer_candidate_is_refused(self):
        architect = ARCHITECT_EVAL_CONFIGS[0]
        with pytest.raises(ValueError) as excinfo:
            get_config_by_name(f'{architect.name}@one-step')
        assert repr(architect.role.value) in str(excinfo.value)


@pytest.mark.usefixtures("code_default_config")
class TestBuildEvalOrchConfig:
    def test_a_one_step_candidate_sets_the_cells_knob(self, tmp_path):
        base = OrchestratorConfig(project_root=tmp_path)
        orch = build_eval_orch_config(named_eval_config(f'{BASE}@one-step'), {}, base_config=base)
        assert orch.implementer_brief_variant is ImplementerBriefVariant.ONE_STEP

    def test_the_candidate_governs_the_cell_not_the_base(self, tmp_path):
        base = OrchestratorConfig(
            project_root=tmp_path, implementer_brief_variant=ImplementerBriefVariant.ONE_STEP,
        )
        orch = build_eval_orch_config(named_eval_config(BASE), {}, base_config=base)
        assert orch.implementer_brief_variant is ImplementerBriefVariant.WHOLE_PLAN
