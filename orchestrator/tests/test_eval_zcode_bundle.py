"""The 'glm-5.3-zcode' eval candidate and its mandatory same-model Claude-harness control.

GLM measured only under the Claude harness measures its imitation of a
Claude-Code-shaped agent (plans/live-shadow-eval-prd.md decision 15), and an
arm without its control measures nothing. So the control is compared live
rather than against literals: the arm must equal 'glm-5.3-endpoint' in
everything except its name and backend.
"""

from __future__ import annotations

import dataclasses

import pytest

from orchestrator.config import load_config
from orchestrator.evals import configs, zai_probe
from orchestrator.evals.configs import (
    EVAL_CONFIGS,
    claude_endpoint_candidates,
    get_config_by_name,
    ofat_candidates,
)
from orchestrator.evals.metrics import RoleUnderTest, is_proxied_endpoint, resolve_cost_usd
from orchestrator.evals.runner import build_eval_orch_config
from orchestrator.shadow_types import CostBasis

ZCODE_ARM = 'glm-5.3-zcode'
CONTROL = 'glm-5.3-endpoint'


@pytest.fixture(autouse=True)
def _zai_key(monkeypatch):
    monkeypatch.setenv(configs.GLM_AUTH_TOKEN_ENV, 'k')


def _resolve(name: str) -> configs.EvalConfig:
    cfg = get_config_by_name(name)
    assert cfg is not None, f'{name} did not resolve via get_config_by_name'
    return cfg


def test_the_arm_runs_glm_on_the_zcode_backend_as_the_implementer():
    arm = _resolve(ZCODE_ARM)
    assert arm.backend == 'zcode'
    assert arm.model == configs.GLM_MODEL
    assert arm.role is RoleUnderTest.IMPLEMENTER


def test_the_arm_differs_from_its_claude_harness_control_only_in_harness():
    arm = _resolve(ZCODE_ARM)
    control = _resolve(CONTROL)
    assert control.backend == 'claude'
    assert dataclasses.replace(arm, name=control.name, backend=control.backend) == control
    assert arm.env_overrides['ANTHROPIC_BASE_URL'] == configs.GLM_BASE_URL


def test_the_campaign_startup_probe_covers_the_arm():
    assert zai_probe.targets_zai_coding_plan(_resolve(ZCODE_ARM)) is True


def test_the_arm_is_opt_in_by_name_only():
    assert ZCODE_ARM not in {c.name for c in EVAL_CONFIGS}
    assert ZCODE_ARM not in {c.name for c in ofat_candidates()}
    assert ZCODE_ARM not in {c.name for c in claude_endpoint_candidates()}


def test_the_cell_runs_zcode_and_prices_glm_as_imputed(tmp_path):
    cfg_path = tmp_path / 'orchestrator.yaml'
    cfg_path.write_text(f'project_root: {tmp_path}\n')
    base = load_config(cfg_path)

    orch_config = build_eval_orch_config(_resolve(ZCODE_ARM), {}, base)

    assert orch_config.backends.implementer == 'zcode'
    assert orch_config.backends.debugger == 'zcode'
    assert configs.GLM_MODEL in orch_config.prices
    is_local_model = is_proxied_endpoint(orch_config.env_overrides)
    assert is_local_model is True
    _, cost_source = resolve_cost_usd(
        1000, 50, model=configs.GLM_MODEL, prices=orch_config.prices,
        cli_cost_usd=0.0, is_local_model=is_local_model,
    )
    assert cost_source == 'price_table'
    assert configs.CANDIDATE_ENDPOINT_PRICES[configs.GLM_MODEL].cost_basis is CostBasis.SUBSCRIPTION
