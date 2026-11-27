"""Each eval config declares the kind of endpoint serving its model, and its cells record it.

The kind is declared by the config builder, never inferred from env: a vLLM
config carries no ANTHROPIC_BASE_URL until ``orchestrator eval --vllm-url``
adds one, and every vendor-cloud arm carries one, so the env says nothing
reliable about where the model runs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from _campaign_gate_helpers import eval_base_config
from _eval_workflow_cell import executor_task, stub_workflow_cell

from orchestrator.evals import configs, runner
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
    registered_configs,
    zcode_candidates,
)
from orchestrator.evals.metrics import EndpointKind, EvalMetrics
from orchestrator.evals.results import ResultStore
from orchestrator.workflow import WorkflowOutcome

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
    return [*registered_configs(), *JUDGE_EVAL_CONFIGS]


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


_TASK_PATH = Path('/fake/task.json')


def _resolve(name: str) -> EvalConfig:
    cfg = get_config_by_name(name)
    assert cfg is not None, name
    return cfg


def _stub_done_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workflow = MagicMock()

    async def run():
        return SimpleNamespace(outcome=WorkflowOutcome.DONE)

    workflow.run = run
    stub_workflow_cell(monkeypatch, executor_task(tmp_path), workflow)


@pytest.mark.asyncio
@pytest.mark.parametrize(('name', 'expected'), [
    ('mimo-v2.6-pro-mimocode', EndpointKind.VENDOR_CLOUD),
    ('mimo-v2.6-pro-endpoint', EndpointKind.VENDOR_CLOUD),
    ('qwen3-coder-30b-q4', EndpointKind.SELF_HOSTED_VLLM),
    ('claude-sonnet-max', EndpointKind.NATIVE_ANTHROPIC),
])
async def test_an_implementer_cell_persists_the_endpoint_kind_its_config_declared(
    name: str, expected: EndpointKind, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _stub_done_cell(tmp_path, monkeypatch)
    store = ResultStore(tmp_path / 'results')

    result = await runner.run_eval(
        _TASK_PATH, _resolve(name), eval_base_config(tmp_path), usage_gate=None, store=store,
    )

    assert result.metrics['endpoint_kind'] == expected.value
    [persisted] = store.load_all()
    assert EvalMetrics.from_dict(persisted.metrics).endpoint_kind is expected


@pytest.mark.asyncio
async def test_an_end_to_end_cell_persists_the_implementers_endpoint_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _stub_done_cell(tmp_path, monkeypatch)
    store = ResultStore(tmp_path / 'results')
    architect = ARCHITECT_EVAL_CONFIGS[0]
    assert architect.endpoint_kind is EndpointKind.NATIVE_ANTHROPIC

    result = await runner.run_end_to_end(
        _TASK_PATH, architect, _resolve('mimo-v2.6-pro-mimocode'), eval_base_config(tmp_path),
        usage_gate=None, store=store,
    )

    assert result.metrics['endpoint_kind'] == 'vendor_cloud'
    [persisted] = store.load_all()
    assert EvalMetrics.from_dict(persisted.metrics).endpoint_kind is EndpointKind.VENDOR_CLOUD
