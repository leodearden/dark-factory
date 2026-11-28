"""Tests for reviewer trial variant definitions."""

from __future__ import annotations

import dataclasses

import pytest

from orchestrator.agents.review_briefing import ReviewBriefing
from orchestrator.agents.roles import AgentRole, build_reviewer_role
from orchestrator.config import default_price_table
from orchestrator.evals.reviewer_trial.variants import (
    _SPEC_COMPREHENSIVE,
    ALL_VARIANTS,
    CAMPAIGN_VARIANTS,
    FABLE_EVIDENCE_VARIANTS,
    REVIEWER_REFRESH_VARIANTS,
    VARIANT_A,
    VARIANT_A_XHIGH,
    VARIANT_B,
    VARIANT_BASELINE,
    VARIANT_C,
    VARIANT_CODEX_SOL62,
    VARIANT_CROSS_FAMILY,
    VARIANT_D,
    VARIANT_FABLE51_SOLO,
    VARIANT_FABLE52_SOLO,
    VARIANT_SONNET5_SOLO,
    ReviewerSpec,
    build_trial_reviewer_role,
    campaign_arms,
    panel_budgets,
    trial_production_budget,
    trial_reviewer_budget,
    trial_reviewer_max_turns,
    with_briefing,
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


class TestFableEvidenceVariants:
    """The D5 evidence campaign set: refresh arms + a Fable 5.1 generalist +
    the incumbent at xhigh, every arm the same single-generalist shape."""

    def test_fable51_solo_is_a_model_only_swap_of_the_incumbent(self) -> None:
        from orchestrator.evals.reviewer_trial.variants import VARIANT_FABLE51_SOLO

        assert len(VARIANT_FABLE51_SOLO.reviewers) == 1
        reviewer = VARIANT_FABLE51_SOLO.reviewers[0]
        incumbent = VARIANT_A.reviewers[0]
        assert reviewer.model == 'claude-fable-5-1'
        assert reviewer.backend == 'claude'
        assert reviewer.specialization == incumbent.specialization
        assert reviewer.effort == incumbent.effort
        assert reviewer.name == incumbent.name

    def test_xhigh_arm_varies_only_effort(self) -> None:
        from orchestrator.evals.reviewer_trial.variants import VARIANT_A_XHIGH

        reviewer = VARIANT_A_XHIGH.reviewers[0]
        incumbent = VARIANT_A.reviewers[0]
        assert reviewer.effort == 'xhigh'
        assert (reviewer.model, reviewer.specialization, reviewer.budget) == (
            incumbent.model, incumbent.specialization, incumbent.budget,
        )

    def test_cross_family_targets_gpt6_astra(self) -> None:
        assert VARIANT_CROSS_FAMILY.reviewers[0].model == 'gpt-6-astra'

    def test_evidence_set_leads_with_the_incumbent(self) -> None:
        from orchestrator.evals.reviewer_trial.variants import (
            FABLE_EVIDENCE_VARIANTS,
            VARIANT_A_XHIGH,
            VARIANT_FABLE51_SOLO,
        )

        assert FABLE_EVIDENCE_VARIANTS[0] is VARIANT_A
        names = [v.name for v in FABLE_EVIDENCE_VARIANTS]
        for candidate in (VARIANT_SONNET5_SOLO, VARIANT_FABLE51_SOLO, VARIANT_CROSS_FAMILY, VARIANT_A_XHIGH):
            assert candidate.name in names
        assert len({v.name for v in FABLE_EVIDENCE_VARIANTS}) == len(FABLE_EVIDENCE_VARIANTS)


class TestBriefingArms:
    """A briefing generation is one more arm dimension over an unchanged panel."""

    def test_the_diff_only_arm_is_the_variant_itself(self) -> None:
        assert VARIANT_A.briefing is ReviewBriefing.DIFF_ONLY
        assert with_briefing(VARIANT_A, ReviewBriefing.DIFF_ONLY) is VARIANT_A

    def test_the_d2_arm_is_named_for_its_briefing_and_keeps_the_panel(self) -> None:
        arm = with_briefing(VARIANT_A, ReviewBriefing.CONTEXT_COMPLETE)

        assert arm.name == 'variant_a_d2'
        assert arm.briefing is ReviewBriefing.CONTEXT_COMPLETE
        assert arm.reviewers == VARIANT_A.reviewers
        assert VARIANT_A.name == 'variant_a'
        assert VARIANT_A.briefing is ReviewBriefing.DIFF_ONLY

    def test_campaign_arms_cross_each_variant_with_each_briefing_incumbent_first(self) -> None:
        arms = campaign_arms(
            [VARIANT_A, VARIANT_A_XHIGH],
            [ReviewBriefing.DIFF_ONLY, ReviewBriefing.CONTEXT_COMPLETE],
        )

        assert [arm.name for arm in arms] == [
            'variant_a', 'variant_a_d2', 'variant_a_xhigh', 'variant_a_xhigh_d2',
        ]
        assert arms[0] is VARIANT_A
        assert [arm.briefing for arm in arms] == [
            ReviewBriefing.DIFF_ONLY, ReviewBriefing.CONTEXT_COMPLETE,
        ] * 2

    @pytest.mark.usefixtures('code_default_config')
    def test_a_d2_trial_role_is_the_production_d2_role(self) -> None:
        spec = VARIANT_A.reviewers[0]
        role = build_trial_reviewer_role(spec, ReviewBriefing.CONTEXT_COMPLETE)
        production = build_reviewer_role(
            spec.name, spec.specialization, ReviewBriefing.CONTEXT_COMPLETE,
        )

        assert role.prompt_harness_version == 'reviewer-d2-v3'
        assert role.system_prompt == production.system_prompt
        assert role.allowed_tools == production.allowed_tools
        assert (role.default_model, role.default_budget) == (spec.model, 12.0)

    def test_a_diff_only_trial_role_is_the_production_incumbent(self) -> None:
        spec = VARIANT_A.reviewers[0]
        role = build_trial_reviewer_role(spec)
        production = build_reviewer_role(spec.name, spec.specialization)

        assert role.prompt_harness_version == production.prompt_harness_version
        assert role.system_prompt == production.system_prompt
        assert role.allowed_tools == production.allowed_tools


class TestD2Round2Variants:
    """The d2 reviewer campaign's round-2 arms: a Fable 5.2 and a GPT-6.2 Sol
    generalist, each the incumbent's single-generalist shape."""

    def test_fable52_solo_is_one_fable_5_2_generalist(self) -> None:
        assert VARIANT_FABLE52_SOLO.name == 'variant_fable52_solo'
        assert len(VARIANT_FABLE52_SOLO.reviewers) == 1
        reviewer = VARIANT_FABLE52_SOLO.reviewers[0]
        assert (reviewer.name, reviewer.model, reviewer.backend, reviewer.effort, reviewer.budget) == (
            'comprehensive_reviewer', 'claude-fable-5-2', 'claude', 'high', None,
        )
        assert reviewer.specialization == _SPEC_COMPREHENSIVE
        assert reviewer.oauth_token_env is None
        assert reviewer.env_overrides is None

    def test_codex_sol62_is_one_gpt_6_2_sol_generalist_on_the_chatgpt_login(self) -> None:
        assert VARIANT_CODEX_SOL62.name == 'variant_codex_sol62'
        assert len(VARIANT_CODEX_SOL62.reviewers) == 1
        reviewer = VARIANT_CODEX_SOL62.reviewers[0]
        assert (reviewer.name, reviewer.model, reviewer.backend, reviewer.effort, reviewer.budget) == (
            'comprehensive_reviewer', 'gpt-6.2-sol', 'codex', 'high', 5.0,
        )
        assert reviewer.specialization == _SPEC_COMPREHENSIVE
        assert reviewer.oauth_token_env is None
        assert reviewer.env_overrides is None

    def test_no_campaign_arm_runs_gpt_6_2_astra(self) -> None:
        models = {r.model for variant in CAMPAIGN_VARIANTS.values() for r in variant.reviewers}
        assert 'gpt-6.2-astra' not in models

    def test_the_campaign_registry_names_every_evidence_arm_and_both_new_arms(self) -> None:
        for name, variant in CAMPAIGN_VARIANTS.items():
            assert name == variant.name
        assert {
            'variant_a', 'variant_sonnet5_solo', 'variant_fable51_solo', 'variant_cross_family',
            'variant_a_xhigh', 'variant_fable52_solo', 'variant_codex_sol62',
        } <= set(CAMPAIGN_VARIANTS)
        assert [v.name for v in FABLE_EVIDENCE_VARIANTS] == [
            'variant_a', 'variant_sonnet5_solo', 'variant_fable51_solo', 'variant_cross_family',
            'variant_a_xhigh',
        ]

    def test_round_2_d2_arms_are_named_for_d2_and_keep_their_models(self) -> None:
        arms = campaign_arms(
            [CAMPAIGN_VARIANTS[n] for n in ('variant_a_xhigh', 'variant_fable52_solo', 'variant_codex_sol62')],
            [ReviewBriefing.CONTEXT_COMPLETE],
        )

        assert [arm.name for arm in arms] == [
            'variant_a_xhigh_d2', 'variant_fable52_solo_d2', 'variant_codex_sol62_d2',
        ]
        assert [arm.reviewers[0].model for arm in arms] == ['opus', 'claude-fable-5-2', 'gpt-6.2-sol']


class TestTrialReviewerTurnCap:
    """A trial reviewer runs at production's reviewer turn cap unless told otherwise."""

    @pytest.mark.usefixtures('code_default_config')
    def test_a_d2_arm_built_with_no_override_carries_production_s_90_turns(self) -> None:
        for variant in (VARIANT_A_XHIGH, VARIANT_FABLE52_SOLO, VARIANT_CODEX_SOL62):
            arm = with_briefing(variant, ReviewBriefing.CONTEXT_COMPLETE)
            role = build_trial_reviewer_role(arm.reviewers[0], arm.briefing)
            assert role.default_max_turns == 90
        assert trial_reviewer_max_turns() == 90

    def test_the_cap_follows_production_s_reviewer_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ORCH_MAX_TURNS__REVIEWER', '77')

        assert build_trial_reviewer_role(VARIANT_A.reviewers[0]).default_max_turns == 77
        assert trial_reviewer_max_turns() == 77

    def test_an_explicit_cap_wins_so_the_30_turn_campaign_stays_reproducible(self) -> None:
        role = build_trial_reviewer_role(
            VARIANT_A_XHIGH.reviewers[0], ReviewBriefing.CONTEXT_COMPLETE, max_turns=30,
        )

        assert role.default_max_turns == 30
        assert trial_reviewer_max_turns(30) == 30

    def test_a_non_positive_cap_is_refused_naming_it(self) -> None:
        with pytest.raises(ValueError, match=r'got 0\b'):
            trial_reviewer_max_turns(0)
        with pytest.raises(ValueError, match=r'got 0\b'):
            build_trial_reviewer_role(VARIANT_A.reviewers[0], max_turns=0)


class TestTrialReviewerBudget:
    """An arm that pins no budget runs at production's reviewer budget unless told otherwise.

    A pinned arm keeps its pin.
    """

    @pytest.mark.usefixtures('code_default_config')
    def test_the_unpinned_arms_built_with_no_override_carry_production_s_12_dollars(self) -> None:
        assert trial_production_budget() == 12.0
        for variant in (VARIANT_A, VARIANT_A_XHIGH, VARIANT_FABLE52_SOLO):
            for arm in (variant, with_briefing(variant, ReviewBriefing.CONTEXT_COMPLETE)):
                assert arm.reviewers[0].budget is None
                role = build_trial_reviewer_role(arm.reviewers[0], arm.briefing)
                assert role.default_budget == 12.0

    @pytest.mark.usefixtures('code_default_config')
    def test_the_fable_5_2_candidate_runs_at_the_incumbent_s_budget(self) -> None:
        """Candidate and incumbent differ only in model, so the budget never varies with the arm."""
        candidate = VARIANT_FABLE52_SOLO.reviewers[0]
        assert dataclasses.replace(VARIANT_A.reviewers[0], model=candidate.model) == candidate
        assert panel_budgets(VARIANT_FABLE52_SOLO) == panel_budgets(VARIANT_A) == (12.0,)
        assert panel_budgets(VARIANT_FABLE52_SOLO, 7.5) == panel_budgets(VARIANT_A, 7.5) == (7.5,)

    def test_the_budget_follows_production_s_reviewer_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ORCH_BUDGETS__REVIEWER', '7.5')

        assert build_trial_reviewer_role(VARIANT_A.reviewers[0]).default_budget == 7.5
        assert trial_reviewer_budget(VARIANT_A.reviewers[0]) == 7.5

    def test_an_explicit_budget_wins_so_the_5_dollar_campaign_stays_reproducible(self) -> None:
        role = build_trial_reviewer_role(
            VARIANT_A_XHIGH.reviewers[0], ReviewBriefing.CONTEXT_COMPLETE, production_budget=5.0,
        )

        assert role.default_budget == 5.0
        assert panel_budgets(VARIANT_A, 5.0) == (5.0,)

    def test_an_int_override_resolves_to_a_float(self) -> None:
        assert type(trial_production_budget(5)) is float
        assert [type(budget) for budget in panel_budgets(VARIANT_A, 5)] == [float]

    def test_a_pinned_spec_ignores_production_s_budget_and_the_override(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv('ORCH_BUDGETS__REVIEWER', '7.5')

        fable = build_trial_reviewer_role(VARIANT_FABLE51_SOLO.reviewers[0], production_budget=6.0)
        sol = build_trial_reviewer_role(VARIANT_CODEX_SOL62.reviewers[0], production_budget=6.0)

        assert fable.default_budget == 10.0
        assert sol.default_budget == 5.0
        assert panel_budgets(VARIANT_C, 6.0) == (4.0, 2.0, 2.0)

    def test_a_non_positive_budget_is_refused_naming_it(self) -> None:
        with pytest.raises(ValueError, match=r'got 0\b'):
            trial_production_budget(0)
        with pytest.raises(ValueError):
            build_trial_reviewer_role(VARIANT_A.reviewers[0], production_budget=-1.0)
