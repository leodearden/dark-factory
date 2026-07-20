"""Tests for reviewer trial variant definitions."""

from __future__ import annotations

from orchestrator.agents.roles import AgentRole
from orchestrator.config import default_price_table
from orchestrator.evals.reviewer_trial.variants import (
    _SPEC_COMPREHENSIVE,
    ALL_VARIANTS,
    REVIEWER_REFRESH_VARIANTS,
    VARIANT_A,
    VARIANT_B,
    VARIANT_BASELINE,
    VARIANT_C,
    VARIANT_CROSS_FAMILY,
    VARIANT_D,
    VARIANT_SONNET5_SOLO,
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

    def test_cross_family_field_defaults(self) -> None:
        """A default spec is a native Claude reviewer: backend='claude',
        no env overrides, no oauth-token env var (keeps every existing
        spec byte-identical)."""
        spec = ReviewerSpec(name='test', model='sonnet', specialization='Testing.')
        assert spec.backend == 'claude'
        assert spec.env_overrides is None
        assert spec.oauth_token_env is None

    def test_cross_family_fields_carry_values(self) -> None:
        """A spec built with cross-family fields carries them verbatim."""
        spec = ReviewerSpec(
            name='xfam',
            model='gpt-5.4',
            specialization='Testing.',
            backend='codex',
            env_overrides={'ANTHROPIC_BASE_URL': 'https://x'},
            oauth_token_env='TOK',
        )
        assert spec.backend == 'codex'
        assert spec.env_overrides == {'ANTHROPIC_BASE_URL': 'https://x'}
        assert spec.oauth_token_env == 'TOK'


class TestBuildTrialReviewerRole:
    def test_returns_agent_role(self) -> None:
        spec = ReviewerSpec(name='test_reviewer', model='opus', specialization='Test spec.')
        role = build_trial_reviewer_role(spec)
        assert isinstance(role, AgentRole)

    def test_reviewer_identity_name(self) -> None:
        """role.name must equal ``reviewer_{spec.name}`` (was ``trial_``-prefixed
        pre-2493). The live verdict-tools transport's ``reviewer ==
        --verdict-role`` identity check validates the emitted verdict's
        ``reviewer`` field against ``role.name`` — and the frozen CONTRACT
        instructs the agent to emit ``reviewer_{name}`` — so role.name must
        match for the identity check to pass (mirrors production's
        ``_reviewer_role``).
        """
        spec = ReviewerSpec(name='my_reviewer', model='sonnet', specialization='Spec.')
        role = build_trial_reviewer_role(spec)
        assert role.name == 'reviewer_my_reviewer'

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

    def test_system_prompt_has_frozen_contract_tokens(self) -> None:
        """Post-2484/2493 trial parity: built from the same reviewer
        PromptSpec as production (roles.build_reviewer_prompt_spec), so the
        frozen CONTRACT tokens the live verdict-tools server parses must be
        present verbatim — the agent calls submit_review_verdict, it does
        not emit JSON/prose (prompt parity forces transport parity).
        """
        spec = ReviewerSpec(name='r', model='sonnet', specialization='Some specialization text.')
        role = build_trial_reviewer_role(spec)
        assert 'submit_review_verdict' in role.system_prompt
        assert '**verdict**' in role.system_prompt
        assert '**issues**' in role.system_prompt
        assert 'Some specialization text.' in role.system_prompt

    def test_system_prompt_drops_obsolete_json_output_instruction(self) -> None:
        """The drifted pre-2493 output-schema/pure-JSON instructions must be
        gone: they contradict the submit_review_verdict CONTRACT and would
        leave the agent with no way to emit a verdict via the live transport.
        """
        spec = ReviewerSpec(name='r', model='sonnet', specialization='S.')
        role = build_trial_reviewer_role(spec)
        assert 'Output pure JSON' not in role.system_prompt
        assert 'produce a structured JSON review' not in role.system_prompt

    def test_mcp_families_verdict_tools(self) -> None:
        spec = ReviewerSpec(name='r', model='sonnet', specialization='S.')
        role = build_trial_reviewer_role(spec)
        assert role.mcp_families == frozenset({'verdict_tools'})

    def test_allowed_tools_includes_verdict_tools(self) -> None:
        spec = ReviewerSpec(name='r', model='sonnet', specialization='S.')
        role = build_trial_reviewer_role(spec)
        assert 'mcp__verdict-tools__*' in role.allowed_tools

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
                assert role.name.startswith('reviewer_')
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


class TestRefreshVariants:
    """The eval-revival κ refresh set: 1×Opus incumbent vs Sonnet-5 vs cross-family."""

    def test_sonnet5_solo_single_comprehensive_sonnet(self) -> None:
        assert len(VARIANT_SONNET5_SOLO.reviewers) == 1
        reviewer = VARIANT_SONNET5_SOLO.reviewers[0]
        assert reviewer.model == 'sonnet'
        assert reviewer.specialization == _SPEC_COMPREHENSIVE

    def test_cross_family_single_priced_non_claude(self) -> None:
        assert len(VARIANT_CROSS_FAMILY.reviewers) == 1
        reviewer = VARIANT_CROSS_FAMILY.reviewers[0]
        # A non-Claude backend so its cost comes from the price table...
        assert reviewer.backend != 'claude'
        # ...and its model must be priced (non-sentinel cost by construction).
        assert reviewer.model in default_price_table()
        assert reviewer.specialization == _SPEC_COMPREHENSIVE

    def test_refresh_set_order_and_incumbent(self) -> None:
        assert REVIEWER_REFRESH_VARIANTS == [
            VARIANT_A,
            VARIANT_SONNET5_SOLO,
            VARIANT_CROSS_FAMILY,
        ]
        # The incumbent (first) is the 1×Opus generalist.
        assert REVIEWER_REFRESH_VARIANTS[0].reviewers[0].model == 'opus'

    def test_refresh_variants_not_in_all_variants(self) -> None:
        """Keeps `full`/`sweep` (which run ALL_VARIANTS) byte-identical."""
        assert VARIANT_SONNET5_SOLO not in ALL_VARIANTS
        assert VARIANT_CROSS_FAMILY not in ALL_VARIANTS
