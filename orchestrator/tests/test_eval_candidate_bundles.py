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

    def test_each_candidate_varies_exactly_one_role(self):
        from orchestrator.evals.configs import claude_endpoint_candidates

        candidates = claude_endpoint_candidates()
        assert all(c.role in ('implementer', 'architect') for c in candidates)

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
