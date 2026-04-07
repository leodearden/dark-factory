"""Tests for eval configuration matrix — EVAL_CONFIGS, VLLM_EVAL_CONFIGS, and get_config_by_name."""

from __future__ import annotations

import pytest

from orchestrator.evals.configs import (
    EVAL_CONFIGS,
    VLLM_EVAL_CONFIGS,
    EvalConfig,
    get_config_by_name,
)


class TestVllmConfigStructure:
    """All vLLM configs must follow the claude-proxy convention."""

    @pytest.mark.parametrize('cfg', VLLM_EVAL_CONFIGS, ids=lambda c: c.name)
    def test_backend_is_claude(self, cfg: EvalConfig):
        """vLLM configs route through the Claude CLI backend."""
        assert cfg.backend == 'claude'

    @pytest.mark.parametrize('cfg', VLLM_EVAL_CONFIGS, ids=lambda c: c.name)
    def test_model_is_sonnet(self, cfg: EvalConfig):
        """vLLM configs use 'sonnet' as the model selector for Claude CLI."""
        assert cfg.model == 'sonnet'

    @pytest.mark.parametrize('cfg', VLLM_EVAL_CONFIGS, ids=lambda c: c.name)
    def test_has_anthropic_api_key_override(self, cfg: EvalConfig):
        """Each vLLM config must set ANTHROPIC_API_KEY in env_overrides."""
        assert 'ANTHROPIC_API_KEY' in cfg.env_overrides

    @pytest.mark.parametrize('cfg', VLLM_EVAL_CONFIGS, ids=lambda c: c.name)
    def test_has_default_sonnet_model_override(self, cfg: EvalConfig):
        """Each vLLM config must set ANTHROPIC_DEFAULT_SONNET_MODEL in env_overrides."""
        assert 'ANTHROPIC_DEFAULT_SONNET_MODEL' in cfg.env_overrides
        # The model path must be a non-empty string (HuggingFace model ID)
        assert len(cfg.env_overrides['ANTHROPIC_DEFAULT_SONNET_MODEL']) > 0

    @pytest.mark.parametrize('cfg', VLLM_EVAL_CONFIGS, ids=lambda c: c.name)
    def test_env_overrides_values_are_strings(self, cfg: EvalConfig):
        """All env_overrides values must be strings (subprocess env requirement)."""
        for key, value in cfg.env_overrides.items():
            assert isinstance(key, str), f'{key!r} key is not a string'
            assert isinstance(value, str), f'{key}={value!r} value is not a string'


class TestGetConfigByName:
    """get_config_by_name must find configs from both lists."""

    def test_find_builtin_config(self):
        """Can find a standard EVAL_CONFIGS entry by name."""
        cfg = get_config_by_name('claude-opus-high')
        assert cfg is not None
        assert cfg.name == 'claude-opus-high'
        assert cfg.backend == 'claude'

    def test_find_vllm_config(self):
        """Can find a VLLM_EVAL_CONFIGS entry by name."""
        # Use the first vLLM config as representative
        first_vllm = VLLM_EVAL_CONFIGS[0]
        cfg = get_config_by_name(first_vllm.name)
        assert cfg is not None
        assert cfg.name == first_vllm.name
        assert cfg.backend == 'claude'

    def test_returns_none_for_unknown(self):
        """Unknown config name returns None."""
        assert get_config_by_name('nonexistent-config-xyz') is None


class TestVllmConfigSet:
    """VLLM_EVAL_CONFIGS must contain exactly the 6 active self-hosted models."""

    EXPECTED_VLLM_NAMES = {
        'minimax-m25-fp8',
        'qwen3-coder-next-fp8',
        'reap-139b-nvfp4',
        'reap-172b-nvfp4',
        'qwen3-coder-30b-q4',
        'devstral-small-2505-q6',
    }

    def test_vllm_config_count_is_six(self):
        """VLLM_EVAL_CONFIGS must have exactly 6 active entries (qwen25 was dropped)."""
        assert len(VLLM_EVAL_CONFIGS) == 6, (
            f'Expected 6 vLLM configs, got {len(VLLM_EVAL_CONFIGS)}: '
            f'{[c.name for c in VLLM_EVAL_CONFIGS]}'
        )

    def test_vllm_config_names_are_exact_set(self):
        """VLLM_EVAL_CONFIGS names must match the canonical active set exactly."""
        actual = {cfg.name for cfg in VLLM_EVAL_CONFIGS}
        assert actual == self.EXPECTED_VLLM_NAMES, (
            f'Extra: {actual - self.EXPECTED_VLLM_NAMES}  '
            f'Missing: {self.EXPECTED_VLLM_NAMES - actual}'
        )


class TestDroppedQwen25Regression:
    """Regression guard: Qwen2.5-Coder-32B must not reappear in any config list."""

    def test_qwen25_32b_not_in_vllm_configs(self):
        """qwen25-coder-32b-q4 must not be in VLLM_EVAL_CONFIGS."""
        names = {cfg.name for cfg in VLLM_EVAL_CONFIGS}
        assert 'qwen25-coder-32b-q4' not in names

    def test_qwen25_32b_not_found_by_name_lookup(self):
        """get_config_by_name must return None for the dropped model."""
        assert get_config_by_name('qwen25-coder-32b-q4') is None

    def test_qwen25_hf_model_not_in_any_config(self):
        """No config should reference Qwen/Qwen2.5-Coder-32B-Instruct as its model."""
        dropped_hf_id = 'Qwen/Qwen2.5-Coder-32B-Instruct'
        for cfg in EVAL_CONFIGS + VLLM_EVAL_CONFIGS:
            assert cfg.env_overrides.get('ANTHROPIC_DEFAULT_SONNET_MODEL') != dropped_hf_id, (
                f'{cfg.name} still references the dropped HF model'
            )


class TestNoNameCollisions:
    """Config names within the canonical list must be unique."""

    def test_vllm_names_are_unique_among_themselves(self):
        """No duplicate names within VLLM_EVAL_CONFIGS."""
        names = [cfg.name for cfg in VLLM_EVAL_CONFIGS]
        assert len(names) == len(set(names)), f'Duplicate vLLM config names: {names}'

    def test_eval_configs_names_unique(self):
        """All names in the canonical EVAL_CONFIGS list are unique."""
        all_names = [cfg.name for cfg in EVAL_CONFIGS]
        assert len(all_names) == len(set(all_names)), f'Duplicate names in EVAL_CONFIGS: {all_names}'


class TestEvalConfigsIncludesVllm:
    """EVAL_CONFIGS must be the canonical list that includes vLLM configs."""

    _CLOUD_BASELINE_COUNT = 7
    _VLLM_COUNT = 6

    def test_eval_configs_includes_all_vllm_configs(self):
        """Every vLLM config name must be present in EVAL_CONFIGS."""
        eval_names = {cfg.name for cfg in EVAL_CONFIGS}
        vllm_names = {cfg.name for cfg in VLLM_EVAL_CONFIGS}
        assert vllm_names.issubset(eval_names), (
            f'vLLM configs missing from EVAL_CONFIGS: {vllm_names - eval_names}'
        )

    def test_eval_configs_total_count(self):
        """EVAL_CONFIGS must have 7 cloud baselines + 6 vLLM = 13 total entries."""
        expected = self._CLOUD_BASELINE_COUNT + self._VLLM_COUNT
        assert len(EVAL_CONFIGS) == expected, (
            f'Expected {expected} configs, got {len(EVAL_CONFIGS)}: '
            f'{[c.name for c in EVAL_CONFIGS]}'
        )

    def test_cloud_baselines_equal_eval_minus_vllm(self):
        """Cloud baselines derived by set-difference must equal the known literal set."""
        derived = {cfg.name for cfg in EVAL_CONFIGS} - {cfg.name for cfg in VLLM_EVAL_CONFIGS}
        expected = {
            'claude-opus-high', 'claude-opus-max', 'claude-sonnet-max',
            'codex-gpt54-xhigh', 'codex-gpt54mini-xhigh',
            'gemini-31-pro-high', 'gemini-3-flash-high',
        }
        assert derived == expected, (
            f'Derived cloud baselines do not match expected set.\n'
            f'  Extra:   {derived - expected}\n'
            f'  Missing: {expected - derived}'
        )


class TestRunnerDefaultIncludesVllm:
    """run_eval_matrix must receive vLLM configs when called with its default EVAL_CONFIGS."""

    def test_runner_module_eval_configs_includes_vllm(self):
        """The EVAL_CONFIGS bound in runner.py must include all vLLM config names."""
        from orchestrator.evals import runner as runner_module
        from orchestrator.evals.configs import VLLM_EVAL_CONFIGS

        # The runner imports EVAL_CONFIGS at module level from .configs
        # After step-6 this should include all vLLM entries.
        runner_eval_names = {cfg.name for cfg in runner_module.EVAL_CONFIGS}
        vllm_names = {cfg.name for cfg in VLLM_EVAL_CONFIGS}
        assert vllm_names.issubset(runner_eval_names), (
            f'Runner EVAL_CONFIGS missing vLLM configs: {vllm_names - runner_eval_names}'
        )


class TestVllmUrlInjection:
    """--vllm-url injection must target only vLLM configs, not cloud baselines."""

    _VLLM_URL = 'http://test-endpoint:8000'
    _CLOUD_BASELINE_NAMES = {
        'claude-opus-high', 'claude-opus-max', 'claude-sonnet-max',
        'codex-gpt54-xhigh', 'codex-gpt54mini-xhigh',
        'gemini-31-pro-high', 'gemini-3-flash-high',
    }

    def test_injection_sets_base_url_on_vllm_configs(self):
        """After injecting vllm_url, every vLLM config must have ANTHROPIC_BASE_URL set."""
        # Save and restore env_overrides to avoid test bleed
        saved = {cfg.name: dict(cfg.env_overrides) for cfg in VLLM_EVAL_CONFIGS}
        try:
            for cfg in VLLM_EVAL_CONFIGS:
                cfg.env_overrides['ANTHROPIC_BASE_URL'] = self._VLLM_URL
            for cfg in VLLM_EVAL_CONFIGS:
                assert cfg.env_overrides.get('ANTHROPIC_BASE_URL') == self._VLLM_URL, (
                    f'{cfg.name} missing ANTHROPIC_BASE_URL after injection'
                )
        finally:
            for cfg in VLLM_EVAL_CONFIGS:
                cfg.env_overrides.clear()
                cfg.env_overrides.update(saved[cfg.name])

    def test_injection_propagates_to_eval_configs_via_shared_references(self):
        """Mutating VLLM_EVAL_CONFIGS configs is visible through EVAL_CONFIGS (same objects)."""
        saved = {cfg.name: dict(cfg.env_overrides) for cfg in VLLM_EVAL_CONFIGS}
        try:
            for cfg in VLLM_EVAL_CONFIGS:
                cfg.env_overrides['ANTHROPIC_BASE_URL'] = self._VLLM_URL
            # EVAL_CONFIGS spreads the same references, so the mutation must be visible
            eval_names_with_base_url = {
                cfg.name for cfg in EVAL_CONFIGS
                if cfg.env_overrides.get('ANTHROPIC_BASE_URL') == self._VLLM_URL
            }
            vllm_names = {cfg.name for cfg in VLLM_EVAL_CONFIGS}
            assert vllm_names == eval_names_with_base_url, (
                f'Shared-reference invariant broken.\n'
                f'  Expected: {vllm_names}\n'
                f'  Got ANTHROPIC_BASE_URL via EVAL_CONFIGS: {eval_names_with_base_url}'
            )
        finally:
            for cfg in VLLM_EVAL_CONFIGS:
                cfg.env_overrides.clear()
                cfg.env_overrides.update(saved[cfg.name])

    def test_cloud_baselines_never_get_base_url(self):
        """Cloud baseline configs must not have ANTHROPIC_BASE_URL after vllm injection."""
        saved = {cfg.name: dict(cfg.env_overrides) for cfg in VLLM_EVAL_CONFIGS}
        try:
            for cfg in VLLM_EVAL_CONFIGS:
                cfg.env_overrides['ANTHROPIC_BASE_URL'] = self._VLLM_URL
            cloud_configs_with_base_url = [
                cfg.name for cfg in EVAL_CONFIGS
                if cfg.name in self._CLOUD_BASELINE_NAMES
                and 'ANTHROPIC_BASE_URL' in cfg.env_overrides
            ]
            assert not cloud_configs_with_base_url, (
                f'Cloud baseline configs leaked ANTHROPIC_BASE_URL: {cloud_configs_with_base_url}'
            )
        finally:
            for cfg in VLLM_EVAL_CONFIGS:
                cfg.env_overrides.clear()
                cfg.env_overrides.update(saved[cfg.name])
