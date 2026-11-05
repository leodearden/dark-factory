"""The shadow invocation profile a live shadow cell runs under (task 5386).

``build_shadow_orch_config`` is the corpus eval profile plus exactly one
divergence: sandboxed roles run under the operator's configured sandbox
backend. Strict MCP and the null memory sentinel are inherited from
``EVAL_PROFILE``, never re-set.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from orchestrator import config as orch_config_module
from orchestrator.config import OrchestratorConfig, SandboxConfig
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.live_fixture import build_live_fixture
from orchestrator.evals.runner import build_eval_orch_config
from orchestrator.evals.shadow_invocation import build_shadow_orch_config
from orchestrator.fm_retry import FM_NULL_SENTINEL_URL

_IMPL = EvalConfig('probe', 'claude', 'sonnet', 'high')


def _shadow_fixture(project_root: Path) -> dict:
    return build_live_fixture(
        {'id': '4242', 'title': 'T', 'metadata': {}},
        base_sha='a' * 40,
        project_root=project_root,
        plan={'steps': [{'id': 'step-1', 'type': 'test', 'description': 'd', 'status': 'pending'}]},
        verify_commands={'test': 'true', 'lint': 'true', 'typecheck': 'true'},
        shape='implementer',
        cell_id=uuid4().hex,
    )


def _changed_leaf_paths(a: OrchestratorConfig, b: OrchestratorConfig) -> set[str]:
    a_leaves = dict(orch_config_module._iter_leaves(a))
    b_leaves = dict(orch_config_module._iter_leaves(b))
    return {path for path, a_val in a_leaves.items() if a_val != b_leaves[path]}


def test_strict_mcp_and_null_memory_are_inherited_from_the_eval_profile(tmp_path: Path) -> None:
    result = build_shadow_orch_config(
        _IMPL, _shadow_fixture(tmp_path), OrchestratorConfig(project_root=tmp_path),
    )

    assert result.strict_mcp_config is True
    assert result.fused_memory.url == FM_NULL_SENTINEL_URL


def test_sandbox_is_enabled_under_the_base_backend(tmp_path: Path) -> None:
    base = OrchestratorConfig(
        project_root=tmp_path, sandbox=SandboxConfig(enabled=True, backend='landlock'),
    )

    result = build_shadow_orch_config(_IMPL, _shadow_fixture(tmp_path), base)

    assert result.sandbox.enabled is True
    assert result.sandbox.backend == 'landlock'


@pytest.mark.usefixtures('code_default_config')
def test_shadow_diverges_from_the_corpus_eval_profile_only_in_sandbox_enabled(
    tmp_path: Path,
) -> None:
    base = OrchestratorConfig(project_root=tmp_path)
    fixture = _shadow_fixture(tmp_path)

    shadow = build_shadow_orch_config(_IMPL, fixture, base)
    corpus = build_eval_orch_config(_IMPL, fixture, base)

    assert _changed_leaf_paths(shadow, corpus) == {'sandbox.enabled'}


def test_architect_config_passes_through(tmp_path: Path) -> None:
    architect = EvalConfig('arch', 'claude', 'haiku', 'low')

    result = build_shadow_orch_config(
        _IMPL, _shadow_fixture(tmp_path), OrchestratorConfig(project_root=tmp_path),
        architect_config=architect,
    )

    assert result.models.architect == 'haiku'


@pytest.mark.parametrize('fixture_id', ['df_task_12', '4242'], ids=['corpus-id', 'production-id'])
def test_a_non_shadow_fixture_id_is_refused(tmp_path: Path, fixture_id: str) -> None:
    fixture = {**_shadow_fixture(tmp_path), 'id': fixture_id}

    with pytest.raises(ValueError, match=fixture_id):
        build_shadow_orch_config(_IMPL, fixture, OrchestratorConfig(project_root=tmp_path))


def test_a_memory_endpoint_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='http://127.0.0.1:5'):
        build_shadow_orch_config(
            _IMPL, _shadow_fixture(tmp_path), OrchestratorConfig(project_root=tmp_path),
            memory_endpoint='http://127.0.0.1:5',
        )


def test_a_missing_base_config_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='base_config'):
        build_shadow_orch_config(_IMPL, _shadow_fixture(tmp_path), None)
