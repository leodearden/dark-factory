"""S1 μ (task 9849), boundary row 28: corpus and shadow eval cells run under the base's sandbox."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from orchestrator.config import OrchestratorConfig, SandboxConfig
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.live_fixture import build_live_fixture
from orchestrator.evals.runner import build_eval_orch_config
from orchestrator.evals.shadow_invocation import build_shadow_orch_config

_IMPL = EvalConfig('t', 'claude', 'sonnet', 'high')

_BASE_SANDBOXES = pytest.mark.parametrize(
    'sandbox',
    [
        SandboxConfig(enabled=True, backend='landlock'),
        SandboxConfig(enabled=True, backend='bwrap'),
        SandboxConfig(enabled=False, backend='none'),
    ],
    ids=['landlock', 'bwrap', 'disabled'],
)


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


@_BASE_SANDBOXES
def test_corpus_config_inherits_the_base_sandbox(tmp_path: Path, sandbox: SandboxConfig) -> None:
    base = OrchestratorConfig(project_root=tmp_path, sandbox=sandbox)
    task = {'id': 'df_task_1', 'project_root': str(tmp_path)}

    assert build_eval_orch_config(_IMPL, task, base).sandbox == base.sandbox


@_BASE_SANDBOXES
def test_shadow_config_inherits_the_base_sandbox_with_no_override_of_its_own(
    tmp_path: Path, sandbox: SandboxConfig,
) -> None:
    base = OrchestratorConfig(project_root=tmp_path, sandbox=sandbox)

    result = build_shadow_orch_config(_IMPL, _shadow_fixture(tmp_path), base)

    assert result.sandbox == base.sandbox
    assert result.sandbox.enabled is sandbox.enabled
