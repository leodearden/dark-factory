"""The two MiMo Code eval candidates and their mandatory same-model Claude-harness controls.

MiMo measured only under the Claude harness measures its imitation of a
Claude-Code-shaped agent (plans/live-shadow-eval-prd.md decision 15), and an
arm without its control measures nothing. So each control is compared live
rather than against literals: the arm must equal its '-endpoint' control in
everything except its name and backend.
"""

from __future__ import annotations

import dataclasses

import pytest

from orchestrator import shadow_eval
from orchestrator.config import load_config
from orchestrator.evals import configs
from orchestrator.evals.configs import (
    EVAL_CONFIGS,
    claude_endpoint_candidates,
    get_config_by_name,
    ofat_candidates,
)
from orchestrator.evals.metrics import RoleUnderTest, is_proxied_endpoint, resolve_cost_usd
from orchestrator.evals.runner import build_eval_orch_config
from orchestrator.implementer_brief_variant import ImplementerBriefVariant
from orchestrator.shadow_types import CostBasis

AMS_BASE_URL = 'https://token-plan-ams.xiaomimimo.com/anthropic'
PAIRS = [
    ('mimo-v2.6-pro-mimocode', 'mimo-v2.6-pro-endpoint', configs.MIMO_PRO_MODEL),
    ('mimo-v2.6-flash-mimocode', 'mimo-v2.6-flash-endpoint', configs.MIMO_FLASH_MODEL),
]
ARMS = [arm for arm, _, _ in PAIRS]


@pytest.fixture(autouse=True)
def _mimo_endpoint(monkeypatch):
    monkeypatch.setenv(configs.MIMO_AUTH_TOKEN_ENV, 'k')
    monkeypatch.setenv(configs.MIMO_BASE_URL_ENV, AMS_BASE_URL)


def _resolve(name: str) -> configs.EvalConfig:
    cfg = get_config_by_name(name)
    assert cfg is not None, f'{name} did not resolve via get_config_by_name'
    return cfg


@pytest.mark.parametrize(('arm', 'control', 'model'), PAIRS)
def test_the_arm_runs_mimo_on_the_mimo_backend_as_the_implementer(arm, control, model):
    cfg = _resolve(arm)
    assert cfg.backend == 'mimo'
    assert cfg.model == model
    assert cfg.role is RoleUnderTest.IMPLEMENTER
    assert cfg.harness_variant is ImplementerBriefVariant.WHOLE_PLAN


@pytest.mark.parametrize(('arm', 'control', 'model'), PAIRS)
def test_the_arm_differs_from_its_claude_harness_control_only_in_harness(arm, control, model):
    arm_cfg = _resolve(arm)
    control_cfg = _resolve(control)
    assert control_cfg.backend == 'claude'
    assert dataclasses.replace(
        arm_cfg, name=control_cfg.name, backend=control_cfg.backend,
    ) == control_cfg


def test_the_family_holds_exactly_the_two_arms_and_they_are_opt_in_by_name_only():
    assert [cfg.name for cfg in configs.mimo_code_candidates()] == ARMS
    for arm in ARMS:
        assert arm not in {c.name for c in EVAL_CONFIGS}
        assert arm not in {c.name for c in ofat_candidates()}
        assert arm not in {c.name for c in claude_endpoint_candidates()}


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('blank', [configs.MIMO_AUTH_TOKEN_ENV, configs.MIMO_BASE_URL_ENV])
def test_a_blank_key_or_base_leaves_the_arm_without_credentials(arm, blank, monkeypatch):
    monkeypatch.setenv(blank, '')
    assert shadow_eval.candidate_has_credentials(_resolve(arm)) is False


@pytest.mark.parametrize('arm', ARMS)
def test_a_set_key_and_base_give_the_arm_credentials(arm):
    assert shadow_eval.candidate_has_credentials(_resolve(arm)) is True


@pytest.mark.parametrize(('arm', 'control', 'model'), PAIRS)
def test_the_cell_runs_mimo_and_prices_the_model_as_imputed(arm, control, model, tmp_path):
    cfg_path = tmp_path / 'orchestrator.yaml'
    cfg_path.write_text(f'project_root: {tmp_path}\n')
    base = load_config(cfg_path)

    orch_config = build_eval_orch_config(_resolve(arm), {}, base)

    assert orch_config.backends.implementer == 'mimo'
    assert orch_config.backends.debugger == 'mimo'
    assert model in orch_config.prices
    is_local_model = is_proxied_endpoint(orch_config.env_overrides)
    assert is_local_model is True
    _, cost_source = resolve_cost_usd(
        1000, 50, model=model, prices=orch_config.prices,
        cli_cost_usd=0.0, is_local_model=is_local_model,
    )
    assert cost_source == 'price_table'
    assert configs.CANDIDATE_ENDPOINT_PRICES[model].cost_basis is CostBasis.SUBSCRIPTION
