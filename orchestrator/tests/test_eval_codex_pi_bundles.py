"""Tests for ξ's codex + pi candidate bundles (task 2480).

Additive Phase-4 candidate bundles, resolved by name by μ's OFAT/matrix
driver via ``get_config_by_name`` (mirrors ν's ``claude_endpoint_candidates()``
pattern, PRD §ξ):

  - a three-arm codex slate, a price/capability ladder with effort held at
    'xhigh' so only the model varies: GPT-6 "Astra", GPT-5.6 "Sol" (the
    Rust implementer: evaluate-only, no architect variant, no production
    wiring — RUST CAUTION, PRD decision 14) and GPT-5.6 "Terra" (mid-price
    control).
  - pi + Sonnet harness-isolating control: same model+effort as the
    claude-sonnet-max incumbent, backend swapped to 'pi' only — isolates
    harness effect from model effect.

Config-level tests only: bundle PRESENCE (roster/contract) and PROPAGATION
via ``build_eval_orch_config`` (``backends.implementer`` / ``models.implementer``).
Never a live codex/pi subprocess dispatch — an eval run actually dispatching
the Rust implementer via codex is μ's runtime driver behaviour, requiring
live codex credentials, and is out of scope here (RED-premise class 3).

Per-test local imports (the ``test_eval_candidate_bundles.py`` /
``test_eval_driver_configs.py`` convention) so an absent symbol fails the one
test that needs it, not collection of the file.
"""

from __future__ import annotations

import pytest

# The 2026-09 codex slate as LITERALS (task 5384), deliberately not read from
# the ``CODEX_*_MODEL`` constants: a pin that reads what it protects guards
# nothing (INV-10).
_CODEX_SLATE_MODELS_BY_NAME = {
    'codex-gpt6-astra': 'gpt-6-astra',
    'codex-gpt5.6-sol': 'gpt-5.6-sol',
    'codex-gpt5.6-terra': 'gpt-5.6-terra',
}


class TestCodexPiCandidatesRoster:
    """Shape of ``codex_pi_candidates()``: the codex slate + the pi control."""

    def test_returns_non_empty_list_with_unique_names(self):
        from orchestrator.evals.configs import codex_pi_candidates

        candidates = codex_pi_candidates()
        assert candidates
        names = [c.name for c in candidates]
        assert len(names) == len(set(names))

    def test_bundles_are_additive_not_in_eval_configs(self):
        """The ξ bundles are opt-in Phase-4 candidates, resolved by name via
        get_config_by_name — never injected into the default-matrix
        EVAL_CONFIGS list (would dispatch an evaluate-only backend by
        default and perturb the OFAT incumbent floor)."""
        from orchestrator.evals.configs import EVAL_CONFIGS, codex_pi_candidates

        eval_config_names = {c.name for c in EVAL_CONFIGS}
        for c in codex_pi_candidates():
            assert c.name not in eval_config_names


class TestPiSonnetControl:
    """pi + Sonnet harness-isolating control: same model+effort as the
    claude-sonnet-max incumbent, only the backend varies."""

    def test_pi_bundle_contract(self):
        from orchestrator.evals.configs import PI_CONTROL_MODEL, codex_pi_candidates

        pi_bundles = [c for c in codex_pi_candidates() if c.backend == 'pi']
        assert len(pi_bundles) == 1
        bundle = pi_bundles[0]
        assert bundle.model == PI_CONTROL_MODEL
        assert bundle.role == 'implementer'

    def test_pi_control_isolates_harness_from_claude_sonnet_incumbent(self):
        """Cross-references ``_cloud_implementer_incumbents()`` (not a
        hardcoded literal) so the control property is drift-resistant: the
        pi control must share the claude sonnet incumbent's model AND
        effort, differing ONLY in backend."""
        from orchestrator.evals.configs import (
            _cloud_implementer_incumbents,
            codex_pi_candidates,
        )

        sonnet_incumbent = next(
            c for c in _cloud_implementer_incumbents()
            if c.backend == 'claude' and c.model == 'sonnet'
        )
        pi_control = next(c for c in codex_pi_candidates() if c.backend == 'pi')

        assert pi_control.model == sonnet_incumbent.model
        assert pi_control.effort == sonnet_incumbent.effort
        assert pi_control.backend != sonnet_incumbent.backend


class TestGetConfigByNameAndPropagation:
    """Selection + backend/model propagation — the dispatch-routing signal:
    an OFAT eval run resolves a ξ bundle by name and routes the implementer
    to the codex/pi backend at the right model. Never a live codex/pi
    subprocess — that is μ's runtime driver behaviour, out of scope here."""

    def test_get_config_by_name_resolves_pi_bundle(self):
        from orchestrator.evals.configs import PI_CONTROL_MODEL, get_config_by_name

        cfg = get_config_by_name('pi-sonnet-control')
        assert cfg is not None
        assert cfg.backend == 'pi'
        assert cfg.model == PI_CONTROL_MODEL

    def test_build_eval_orch_config_propagates_backend_and_model_for_every_bundle(
        self, tmp_path,
    ):
        # Mirrors test_eval_candidate_bundles.py's
        # test_build_eval_orch_config_propagates_endpoint_and_model: a minimal
        # YAML setting only project_root, layered over packaged defaults.yaml
        # through the real production config-load entry point.
        from orchestrator.config import load_config
        from orchestrator.evals.configs import get_config_by_name
        from orchestrator.evals.runner import build_eval_orch_config

        cfg_path = tmp_path / 'orchestrator.yaml'
        cfg_path.write_text(f'project_root: {tmp_path}\n')
        base = load_config(cfg_path)

        for name in (*_CODEX_SLATE_MODELS_BY_NAME, 'pi-sonnet-control'):
            bundle = get_config_by_name(name)
            assert bundle is not None, f'{name} did not resolve via get_config_by_name'

            orch_config = build_eval_orch_config(bundle, {}, base)

            assert orch_config.backends.implementer == bundle.backend
            assert orch_config.models.implementer == bundle.model


class TestCodexSlatePin:
    """The 2026-09 codex slate, pinned by LITERAL ids and list prices (task
    5384), so a stale id, a drifted price, or an unnoticed addition or
    removal fails loudly."""

    def test_roster_is_exactly_the_codex_slate_plus_the_pi_control(self):
        from orchestrator.evals.configs import codex_pi_candidates

        assert {c.name for c in codex_pi_candidates()} == {
            *_CODEX_SLATE_MODELS_BY_NAME, 'pi-sonnet-control',
        }

    def test_codex_arms_carry_the_literal_model_ids(self):
        from orchestrator.evals.configs import codex_pi_candidates

        by_name = {c.name: c for c in codex_pi_candidates()}
        for name, model in _CODEX_SLATE_MODELS_BY_NAME.items():
            arm = by_name[name]
            assert arm.model == model
            assert arm.backend == 'codex'
            assert arm.role == 'implementer'
            assert arm.effort == 'xhigh'

    def test_get_config_by_name_resolves_each_codex_arm(self):
        from orchestrator.evals.configs import get_config_by_name

        for name, model in _CODEX_SLATE_MODELS_BY_NAME.items():
            cfg = get_config_by_name(name)
            assert cfg is not None
            assert cfg.backend == 'codex'
            assert cfg.model == model

    @pytest.mark.parametrize(('model', 'input_per_1m', 'output_per_1m'), [
        ('gpt-6-astra', 10.00, 50.00),
        ('gpt-5.6-sol', 4.00, 20.00),
        ('gpt-5.6-terra', 2.00, 12.00),
    ])
    def test_list_prices_are_pinned(self, model, input_per_1m, output_per_1m):
        """Codex reports no native cost, so its rates are the default price seeds."""
        from orchestrator.config import default_price_table

        entry = default_price_table()[model]
        assert entry['input_per_1m'] == input_per_1m
        assert entry['output_per_1m'] == output_per_1m

    def test_every_codex_arm_has_a_price_seed(self):
        """Derived from the roster, so a future arm without a seed reddens."""
        from orchestrator.config import default_price_table
        from orchestrator.evals.configs import codex_pi_candidates

        table = default_price_table()
        codex_models = {c.model for c in codex_pi_candidates() if c.backend == 'codex'}
        assert codex_models
        assert codex_models <= table.keys()

    def test_pi_control_is_unchanged(self):
        from orchestrator.evals.configs import (
            PI_CONTROL_MODEL,
            EvalConfig,
            codex_pi_candidates,
        )

        by_name = {c.name: c for c in codex_pi_candidates()}
        assert by_name['pi-sonnet-control'] == EvalConfig(
            'pi-sonnet-control', 'pi', PI_CONTROL_MODEL, 'max', role='implementer',
        )
