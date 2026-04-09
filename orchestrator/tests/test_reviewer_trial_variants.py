"""Tests for reviewer trial variant definitions."""

from __future__ import annotations

from orchestrator.agents.roles import AgentRole
from orchestrator.evals.reviewer_trial.variants import (
    ALL_VARIANTS,
    VARIANT_A,
    VARIANT_B,
    VARIANT_BASELINE,
    VARIANT_C,
    VARIANT_D,
    ReviewerSpec,
    build_trial_reviewer_role,
)


class TestReviewerSpec:
    def test_defaults(self) -> None:
        spec = ReviewerSpec(name='test', model='sonnet', specialization='Testing.')
        assert spec.budget == 2.0
        assert spec.effort == 'high'

    def test_custom_budget(self) -> None:
        spec = ReviewerSpec(name='test', model='opus', specialization='Testing.', budget=5.0)
        assert spec.budget == 5.0


class TestBuildTrialReviewerRole:
    def test_returns_agent_role(self) -> None:
        spec = ReviewerSpec(name='test_reviewer', model='opus', specialization='Test spec.')
        role = build_trial_reviewer_role(spec)
        assert isinstance(role, AgentRole)

    def test_name_prefix(self) -> None:
        spec = ReviewerSpec(name='my_reviewer', model='sonnet', specialization='Spec.')
        role = build_trial_reviewer_role(spec)
        assert role.name == 'trial_my_reviewer'

    def test_model_passthrough(self) -> None:
        for model in ('opus', 'sonnet'):
            spec = ReviewerSpec(name='r', model=model, specialization='S.')
            role = build_trial_reviewer_role(spec)
            assert role.default_model == model

    def test_budget_passthrough(self) -> None:
        spec = ReviewerSpec(name='r', model='opus', specialization='S.', budget=4.0)
        role = build_trial_reviewer_role(spec)
        assert role.default_budget == 4.0

    def test_system_prompt_contains_specialization(self) -> None:
        spec = ReviewerSpec(name='r', model='sonnet', specialization='Test coverage and quality.')
        role = build_trial_reviewer_role(spec)
        assert 'Test coverage and quality.' in role.system_prompt

    def test_system_prompt_has_json_schema(self) -> None:
        spec = ReviewerSpec(name='r', model='sonnet', specialization='S.')
        role = build_trial_reviewer_role(spec)
        assert '"verdict"' in role.system_prompt
        assert '"issues"' in role.system_prompt
        assert 'blocking' in role.system_prompt

    def test_read_only_tools(self) -> None:
        spec = ReviewerSpec(name='r', model='sonnet', specialization='S.')
        role = build_trial_reviewer_role(spec)
        assert 'Read' in role.allowed_tools
        assert 'Glob' in role.allowed_tools
        assert 'Grep' in role.allowed_tools
        assert 'Edit' in role.disallowed_tools
        assert 'Write' in role.disallowed_tools


class TestVariantDefinitions:
    def test_all_variants_count(self) -> None:
        assert len(ALL_VARIANTS) == 5

    def test_baseline_has_5_sonnet_reviewers(self) -> None:
        assert len(VARIANT_BASELINE.reviewers) == 5
        assert all(r.model == 'sonnet' for r in VARIANT_BASELINE.reviewers)

    def test_variant_a_single_opus(self) -> None:
        assert len(VARIANT_A.reviewers) == 1
        assert VARIANT_A.reviewers[0].model == 'opus'

    def test_variant_b_two_opus(self) -> None:
        assert len(VARIANT_B.reviewers) == 2
        assert all(r.model == 'opus' for r in VARIANT_B.reviewers)

    def test_variant_c_mixed(self) -> None:
        assert len(VARIANT_C.reviewers) == 3
        opus = [r for r in VARIANT_C.reviewers if r.model == 'opus']
        sonnet = [r for r in VARIANT_C.reviewers if r.model == 'sonnet']
        assert len(opus) == 1
        assert len(sonnet) == 2

    def test_variant_d_three_sonnet(self) -> None:
        assert len(VARIANT_D.reviewers) == 3
        assert all(r.model == 'sonnet' for r in VARIANT_D.reviewers)

    def test_all_variants_build_valid_roles(self) -> None:
        """Every spec in every variant builds a valid AgentRole."""
        for variant in ALL_VARIANTS:
            for spec in variant.reviewers:
                role = build_trial_reviewer_role(spec)
                assert isinstance(role, AgentRole)
                assert role.name.startswith('trial_')
                assert role.system_prompt
                assert role.allowed_tools

    def test_unique_reviewer_names_within_variants(self) -> None:
        """No duplicate reviewer names within a single variant."""
        for variant in ALL_VARIANTS:
            names = [r.name for r in variant.reviewers]
            assert len(names) == len(set(names)), f'Duplicate names in {variant.name}: {names}'

    def test_all_variants_have_descriptions(self) -> None:
        for variant in ALL_VARIANTS:
            assert variant.description
            assert variant.name
