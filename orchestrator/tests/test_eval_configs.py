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
    """vLLM and standard config names must not collide."""

    def test_vllm_names_are_unique_among_themselves(self):
        """No duplicate names within VLLM_EVAL_CONFIGS."""
        names = [cfg.name for cfg in VLLM_EVAL_CONFIGS]
        assert len(names) == len(set(names)), f'Duplicate vLLM config names: {names}'

    def test_vllm_names_do_not_collide_with_eval_configs(self):
        """No vLLM config name appears in EVAL_CONFIGS."""
        eval_names = {cfg.name for cfg in EVAL_CONFIGS}
        vllm_names = {cfg.name for cfg in VLLM_EVAL_CONFIGS}
        overlap = eval_names & vllm_names
        assert not overlap, f'Name collision between EVAL_CONFIGS and VLLM_EVAL_CONFIGS: {overlap}'

    def test_all_eval_config_names_unique(self):
        """All names across both config lists are unique."""
        all_names = [cfg.name for cfg in EVAL_CONFIGS + VLLM_EVAL_CONFIGS]
        assert len(all_names) == len(set(all_names)), f'Duplicate names: {all_names}'
