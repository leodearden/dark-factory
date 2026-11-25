"""Each eval config declares the kind of endpoint serving its model, and its cells record it.

The kind is declared by the config builder, never inferred from env: a vLLM
config carries no ANTHROPIC_BASE_URL until ``orchestrator eval --vllm-url``
adds one, and every vendor-cloud arm carries one, so the env says nothing
reliable about where the model runs.
"""

from __future__ import annotations

import pytest

from orchestrator.evals import configs
from orchestrator.evals.configs import (
    ARCHITECT_EVAL_CONFIGS,
    EVAL_CONFIGS,
    FINAL_RUN_CONFIGS,
    JUDGE_EVAL_CONFIGS,
    VLLM_EVAL_CONFIGS,
    EvalConfig,
    claude_endpoint_candidates,
    codex_pi_candidates,
    get_config_by_name,
    mimo_code_candidates,
    zcode_candidates,
)
from orchestrator.evals.metrics import EndpointKind

# The two FINAL_RUN_CONFIGS entries that re-run the native cloud baselines
# beside the vLLM final-run configs.
FINAL_RUN_CLOUD_BASELINES = {'final-claude-sonnet-max', 'final-claude-opus-max'}


@pytest.fixture(autouse=True)
def _vendor_credentials(monkeypatch):
    monkeypatch.setenv(configs.MIMO_AUTH_TOKEN_ENV, 'k')
    monkeypatch.setenv(configs.MIMO_BASE_URL_ENV, 'https://token-plan-ams.xiaomimimo.com/anthropic')
    for token_env in (
        configs.GLM_AUTH_TOKEN_ENV,
        configs.DEEPSEEK_AUTH_TOKEN_ENV,
        configs.MINIMAX_AUTH_TOKEN_ENV,
        configs.KIMI_AUTH_TOKEN_ENV,
    ):
        monkeypatch.setenv(token_env, 'k')


def _self_hosted_vllm_configs() -> list[EvalConfig]:
    return [
        *VLLM_EVAL_CONFIGS,
        *(c for c in FINAL_RUN_CONFIGS if c.name not in FINAL_RUN_CLOUD_BASELINES),
    ]


def _every_registered_config() -> list[EvalConfig]:
    return [
        *EVAL_CONFIGS,
        *FINAL_RUN_CONFIGS,
        *ARCHITECT_EVAL_CONFIGS,
        *JUDGE_EVAL_CONFIGS,
        *claude_endpoint_candidates(),
        *codex_pi_candidates(),
        *zcode_candidates(),
        *mimo_code_candidates(),
    ]


def test_every_vllm_config_declares_a_self_hosted_endpoint_without_a_base_url():
    vllm_configs = _self_hosted_vllm_configs()
    assert vllm_configs
    for cfg in vllm_configs:
        assert 'ANTHROPIC_BASE_URL' not in cfg.env_overrides, cfg.name
        assert cfg.endpoint_kind is EndpointKind.SELF_HOSTED_VLLM, cfg.name


def test_cloud_baselines_architects_judges_and_the_pi_control_are_native_anthropic():
    vllm_names = {c.name for c in VLLM_EVAL_CONFIGS}
    pi_control = get_config_by_name('pi-sonnet-control')
    assert pi_control is not None
    native = [
        *(c for c in EVAL_CONFIGS if c.name not in vllm_names),
        *(c for c in FINAL_RUN_CONFIGS if c.name in FINAL_RUN_CLOUD_BASELINES),
        *ARCHITECT_EVAL_CONFIGS,
        *JUDGE_EVAL_CONFIGS,
        pi_control,
    ]
    for cfg in native:
        assert cfg.endpoint_kind is EndpointKind.NATIVE_ANTHROPIC, cfg.name


def test_vendor_endpoint_bundles_and_vendor_harness_arms_are_vendor_cloud():
    vendor = [
        *(c for c in claude_endpoint_candidates() if c.env_overrides),
        *mimo_code_candidates(),
        *zcode_candidates(),
        *(c for c in codex_pi_candidates() if c.backend == 'codex'),
    ]
    assert len([c for c in vendor if c.backend == 'codex']) == 3
    assert len([c for c in vendor if c.backend == 'mimo']) == 2
    assert len([c for c in vendor if c.backend == 'zcode']) == 1
    for cfg in vendor:
        assert cfg.endpoint_kind is EndpointKind.VENDOR_CLOUD, cfg.name


def test_only_vllm_configs_report_a_self_hosted_endpoint():
    self_hosted = {
        c.name for c in _every_registered_config()
        if c.endpoint_kind is EndpointKind.SELF_HOSTED_VLLM
    }
    assert self_hosted == {c.name for c in _self_hosted_vllm_configs()}


def test_every_harness_other_than_claude_and_pi_runs_against_a_vendor_cloud():
    for cfg in _every_registered_config():
        if cfg.backend not in ('claude', 'pi'):
            assert cfg.endpoint_kind is EndpointKind.VENDOR_CLOUD, cfg.name


def test_a_bare_config_defaults_to_native_anthropic():
    assert EvalConfig('x', 'claude', 'opus', 'high').endpoint_kind is EndpointKind.NATIVE_ANTHROPIC
