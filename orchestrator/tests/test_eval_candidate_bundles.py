"""Tests for ν's Claude-format-endpoint candidate bundles (task 2479).

Non-incumbent (MiniMax M2.5, GLM-5.2, DeepSeek V4, Kimi) + native-cloud
incumbent (Opus/Sonnet) bundles, each a (harness, model) ``EvalConfig``
dispatched via Claude Code against a provider's official Anthropic-format
endpoint (PRD C5: per-role ``ANTHROPIC_BASE_URL``/``ANTHROPIC_AUTH_TOKEN``).

Runtime-behaviour tests only: roster/shape, endpoint+auth env contract,
price-table coverage + λ's ``resolve_cost_usd`` -> ``'price_table'``, and
``get_config_by_name`` resolution + ``build_eval_orch_config`` endpoint
propagation. The env-forwarding wiring itself is already covered by task
2460's ``test_workflow_e2e.py``, and the μ OFAT/matrix driver by task 2478's
``test_eval_driver*.py`` — this module does not re-test either.

Per-test local imports (the ``test_eval_driver_configs.py`` convention) so an
absent symbol fails the one test that needs it, not collection of the file.
"""

from __future__ import annotations


class TestClaudeEndpointCandidatesRoster:
    """Shape of ``claude_endpoint_candidates()``: incumbents + non-incumbents."""

    def test_returns_non_empty_list_with_unique_names(self):
        from orchestrator.evals.configs import claude_endpoint_candidates

        candidates = claude_endpoint_candidates()
        assert candidates
        names = [c.name for c in candidates]
        assert len(names) == len(set(names))

    def test_returns_only_recognized_roles(self):
        """Every candidate's ``role`` is a recognized value.

        Renamed from the former ``test_each_candidate_varies_exactly_one_role``,
        whose name/docstring claimed the "exactly one role varies" invariant
        but whose body only checked role membership — that stronger claim is
        a structural property of ``EvalConfig`` (it carries a single ``role``
        field, so it cannot vary more than one by construction), not
        something a membership check exercises. See
        ``test_non_incumbent_bundles_all_target_the_implementer_role`` below
        for the concrete, testable role invariant ν's bundles actually hold.
        """
        from orchestrator.evals.configs import claude_endpoint_candidates

        candidates = claude_endpoint_candidates()
        assert all(c.role in ('implementer', 'architect') for c in candidates)

    def test_non_incumbent_bundles_all_target_the_implementer_role(self):
        """ν's non-incumbent bundles vary the implementer role only (design
        decision: "Non-incumbent bundles default to role='implementer'") —
        an architect-role ν bundle, and the both-proxied architect×implementer
        matrix, are explicitly out of scope for this task."""
        from orchestrator.evals.configs import (
            DEEPSEEK_MODEL,
            GLM_MODEL,
            KIMI_MODEL,
            MINIMAX_MODEL,
            claude_endpoint_candidates,
        )

        non_incumbent_models = {MINIMAX_MODEL, GLM_MODEL, DEEPSEEK_MODEL, KIMI_MODEL}
        non_incumbents = [
            c for c in claude_endpoint_candidates() if c.model in non_incumbent_models
        ]
        assert non_incumbents
        assert all(c.role == 'implementer' for c in non_incumbents)

    def test_includes_the_four_non_incumbent_models(self):
        from orchestrator.evals.configs import (
            DEEPSEEK_MODEL,
            GLM_MODEL,
            KIMI_MODEL,
            MINIMAX_MODEL,
            claude_endpoint_candidates,
        )

        models = {c.model for c in claude_endpoint_candidates()}
        assert MINIMAX_MODEL in models
        assert GLM_MODEL in models
        assert DEEPSEEK_MODEL in models
        assert KIMI_MODEL in models

    def test_includes_at_least_two_native_incumbents_opus_and_sonnet(self):
        from orchestrator.evals.configs import claude_endpoint_candidates

        # A native incumbent carries no proxy env_overrides — that is what
        # distinguishes it from a proxied non-incumbent bundle (mirrors
        # _cloud_implementer_incumbents' own criterion).
        incumbents = [c for c in claude_endpoint_candidates() if not c.env_overrides]
        assert len(incumbents) >= 2
        models = {c.model for c in incumbents}
        assert 'opus' in models
        assert 'sonnet' in models


class TestEndpointAuthEnvContract:
    """Non-incumbent bundles carry a real endpoint+auth env contract;
    incumbent bundles stay proxy-free (native cloud, no proxy)."""

    def _non_incumbents(self):
        from orchestrator.evals.configs import (
            DEEPSEEK_MODEL,
            GLM_MODEL,
            KIMI_MODEL,
            MINIMAX_MODEL,
            claude_endpoint_candidates,
        )

        non_incumbent_models = {MINIMAX_MODEL, GLM_MODEL, DEEPSEEK_MODEL, KIMI_MODEL}
        return [c for c in claude_endpoint_candidates() if c.model in non_incumbent_models]

    def test_non_incumbents_carry_claude_backend_and_env_contract(self):
        non_incumbents = self._non_incumbents()
        assert non_incumbents
        for c in non_incumbents:
            assert c.backend == 'claude'
            assert c.env_overrides.get('ANTHROPIC_BASE_URL', '').startswith('https://')
            assert 'ANTHROPIC_AUTH_TOKEN' in c.env_overrides

    def test_non_incumbent_base_urls_carry_the_provider_domain(self):
        from orchestrator.evals.configs import (
            DEEPSEEK_MODEL,
            GLM_MODEL,
            KIMI_MODEL,
            MINIMAX_MODEL,
        )

        expected_domain = {
            MINIMAX_MODEL: 'minimax',
            GLM_MODEL: 'z.ai',
            DEEPSEEK_MODEL: 'deepseek',
            KIMI_MODEL: 'moonshot',
        }
        for c in self._non_incumbents():
            assert expected_domain[c.model] in c.env_overrides['ANTHROPIC_BASE_URL']

    def test_monkeypatched_auth_token_env_flows_into_env_overrides(self, monkeypatch):
        from orchestrator.evals.configs import (
            DEEPSEEK_AUTH_TOKEN_ENV,
            DEEPSEEK_MODEL,
            claude_endpoint_candidates,
        )

        monkeypatch.setenv(DEEPSEEK_AUTH_TOKEN_ENV, 'patched-token-xyz')
        candidates = claude_endpoint_candidates()
        bundle = next(c for c in candidates if c.model == DEEPSEEK_MODEL)
        assert bundle.env_overrides['ANTHROPIC_AUTH_TOKEN'] == 'patched-token-xyz'

    def test_missing_auth_token_env_yields_empty_token_and_logs_a_warning(
        self, monkeypatch, caplog,
    ):
        """An operator who forgot to export the provider's token env var gets
        a loud build-time WARNING, not a silent well-formed-looking bundle
        that only fails as an opaque 401 mid-run (loud-over-silent-degradation).
        """
        import logging

        from orchestrator.evals.configs import (
            DEEPSEEK_AUTH_TOKEN_ENV,
            DEEPSEEK_MODEL,
            claude_endpoint_candidates,
        )

        monkeypatch.delenv(DEEPSEEK_AUTH_TOKEN_ENV, raising=False)
        with caplog.at_level(logging.WARNING, logger='orchestrator.evals.configs'):
            candidates = claude_endpoint_candidates()
        bundle = next(c for c in candidates if c.model == DEEPSEEK_MODEL)

        assert bundle.env_overrides['ANTHROPIC_AUTH_TOKEN'] == ''
        assert any(
            record.levelno == logging.WARNING
            and DEEPSEEK_AUTH_TOKEN_ENV in record.getMessage()
            for record in caplog.records
        )

    def test_incumbents_have_no_env_overrides(self):
        from orchestrator.evals.configs import claude_endpoint_candidates

        incumbents = [c for c in claude_endpoint_candidates() if c.model in ('opus', 'sonnet')]
        assert incumbents
        assert all(c.backend == 'claude' for c in incumbents)
        assert all(not c.env_overrides for c in incumbents)


class TestClaudeEndpointPriceTable:
    """claude_endpoint_price_table(): default seeds + candidate-model prices,
    proven to resolve via λ's resolve_cost_usd (cost_source='price_table')."""

    def test_contains_every_default_price_table_key(self):
        from orchestrator.config import default_price_table
        from orchestrator.evals.configs import claude_endpoint_price_table

        table = claude_endpoint_price_table()
        for key in default_price_table():
            assert key in table

    def test_has_a_valid_entry_for_every_non_incumbent_candidate_model(self):
        from orchestrator.evals.configs import (
            DEEPSEEK_MODEL,
            GLM_MODEL,
            KIMI_MODEL,
            MINIMAX_MODEL,
            claude_endpoint_candidates,
            claude_endpoint_price_table,
        )

        non_incumbent_models = {MINIMAX_MODEL, GLM_MODEL, DEEPSEEK_MODEL, KIMI_MODEL}
        candidate_models = {
            c.model for c in claude_endpoint_candidates() if c.model in non_incumbent_models
        }
        # Sanity: cross-checks model-key alignment — every non-incumbent
        # candidate model is accounted for (none silently missing).
        assert candidate_models == non_incumbent_models

        table = claude_endpoint_price_table()
        for model in candidate_models:
            entry = table[model]
            assert entry['input_per_1m'] > 0
            assert entry['output_per_1m'] > 0

    def test_resolve_cost_usd_resolves_price_table_for_a_candidate_model(self):
        from orchestrator.evals.configs import DEEPSEEK_MODEL, claude_endpoint_price_table
        from orchestrator.evals.metrics import resolve_cost_usd

        table = claude_endpoint_price_table()
        entry = table[DEEPSEEK_MODEL]
        input_tokens, output_tokens = 10_000, 2_000
        expected_cost = (
            input_tokens * entry['input_per_1m'] + output_tokens * entry['output_per_1m']
        ) / 1_000_000

        cost, cost_source = resolve_cost_usd(
            input_tokens, output_tokens,
            model=DEEPSEEK_MODEL,
            prices=table,
            cli_cost_usd=99999.0,  # deliberately wrong — must NOT be used
            is_local_model=True,
        )

        assert cost_source == 'price_table'
        assert cost == expected_cost


class TestGetConfigByNameAndPropagation:
    """Selection + endpoint propagation — the dispatch-reaches-endpoint signal:
    an OFAT eval run resolves a ν bundle by name and dispatches the
    implementer to its non-incumbent endpoint."""

    def test_get_config_by_name_resolves_a_non_incumbent_bundle(self):
        from orchestrator.evals.configs import GLM_MODEL, get_config_by_name

        cfg = get_config_by_name('glm-5.2-endpoint')
        assert cfg is not None
        assert cfg.model == GLM_MODEL

    def test_build_eval_orch_config_propagates_endpoint_and_model(self, tmp_path):
        # Mirrors test_eval_driver.py::_base_config's load_config() pattern:
        # a minimal YAML setting only project_root, layered over the packaged
        # defaults.yaml through the real production config-load entry point.
        from orchestrator.config import load_config
        from orchestrator.evals.configs import GLM_BASE_URL, get_config_by_name
        from orchestrator.evals.runner import build_eval_orch_config

        cfg_path = tmp_path / 'orchestrator.yaml'
        cfg_path.write_text(f'project_root: {tmp_path}\n')
        base = load_config(cfg_path)

        bundle = get_config_by_name('glm-5.2-endpoint')
        assert bundle is not None

        orch_config = build_eval_orch_config(bundle, {}, base)

        assert orch_config.env_overrides.get('ANTHROPIC_BASE_URL') == GLM_BASE_URL
        assert orch_config.models.implementer == bundle.model
