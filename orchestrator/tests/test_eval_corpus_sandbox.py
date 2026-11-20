"""S1 μ (task 9849), boundary row 28: corpus and shadow eval cells run under the base's sandbox."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from orchestrator.agents import sandbox_dispatch
from orchestrator.config import (
    PROJECT_CONFIG_FILENAME,
    OrchestratorConfig,
    SandboxConfig,
    project_role_env_overrides,
)
from orchestrator.evals import runner
from orchestrator.evals.cell_sandbox import bind_cell_sandbox_backend
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.live_fixture import build_live_fixture
from orchestrator.evals.metrics import EvalMetrics
from orchestrator.evals.results import EvalResult, ResultStore
from orchestrator.evals.runner import build_eval_orch_config
from orchestrator.evals.shadow_invocation import build_shadow_orch_config
from orchestrator.workflow import WorkflowOutcome

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


def _landlock_available(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    monkeypatch.setattr('orchestrator.agents.landlock.is_landlock_available', lambda: available)


def test_bind_sets_the_process_backend_to_the_configured_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox_dispatch.set_backend('auto')
    monkeypatch.setattr('orchestrator.agents.sandbox.is_bwrap_available', lambda: True)
    config = OrchestratorConfig(
        project_root=tmp_path, sandbox=SandboxConfig(enabled=True, backend='bwrap'),
    )

    assert bind_cell_sandbox_backend(config) == 'bwrap'
    assert sandbox_dispatch.get_backend() == 'bwrap'


def test_bind_reports_none_when_the_config_disables_the_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _landlock_available(monkeypatch, True)
    config = OrchestratorConfig(
        project_root=tmp_path, sandbox=SandboxConfig(enabled=False, backend='landlock'),
    )

    assert bind_cell_sandbox_backend(config) == 'none'


def test_bind_reports_none_under_the_none_escape_hatch(tmp_path: Path) -> None:
    config = OrchestratorConfig(
        project_root=tmp_path, sandbox=SandboxConfig(enabled=True, backend='none'),
    )

    assert bind_cell_sandbox_backend(config) == 'none'


def test_bind_reports_refused_when_the_configured_backend_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _landlock_available(monkeypatch, False)
    config = OrchestratorConfig(
        project_root=tmp_path, sandbox=SandboxConfig(enabled=True, backend='landlock'),
    )

    assert bind_cell_sandbox_backend(config) == 'refused'


def _executor_task(tmp_path: Path) -> dict:
    return {
        'id': 'df_task_mu',
        'project_root': str(tmp_path),
        'pre_task_commit': 'basecommit',
        'task_definition': {'title': 'W', 'description': 'd'},
        'modules': ['pkg/mod.py'],
        'plan': {'steps': [{'id': 's1', 'status': 'pending'}]},
    }


async def _fake_create_wt(*_a, **_k):
    return Path('/fake/wt'), 'run-mu'


async def _fake_collect(_wf, _wt, _task, *, role, replay_frame=None, sandbox_backend):
    return EvalMetrics(role_under_test=role, replay_frame=replay_frame, sandbox_backend=sandbox_backend)


async def _raising_collect(_wf, _wt, _task, *, role, replay_frame=None, sandbox_backend):
    raise RuntimeError('metric collection exploded')


async def _run_landlock_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, collect, store: ResultStore,
) -> tuple[EvalResult, list[sandbox_dispatch.Backend]]:
    """One executor cell over a landlock base, in a process whose backend no Harness bound.

    Returns the cell's result and the process backend each run of its workflow saw.
    """
    sandbox_dispatch.set_backend('auto')
    _landlock_available(monkeypatch, True)
    backends_at_run: list[sandbox_dispatch.Backend] = []

    async def run_workflow():
        backends_at_run.append(sandbox_dispatch.get_backend())
        return SimpleNamespace(outcome=WorkflowOutcome.DONE)

    workflow = MagicMock()
    workflow.run = run_workflow
    monkeypatch.setattr(runner, 'create_eval_worktree', _fake_create_wt)
    monkeypatch.setattr(runner, 'build_workflow', lambda **_kw: workflow)
    monkeypatch.setattr(runner, 'collect_metrics', AsyncMock(side_effect=collect))
    monkeypatch.setattr(runner, 'load_task', lambda _p: _executor_task(tmp_path))
    base = OrchestratorConfig(
        project_root=tmp_path, sandbox=SandboxConfig(enabled=True, backend='landlock'),
    )

    result = await runner.run_eval(
        Path('/fake/task.json'), _IMPL, base, usage_gate=None, store=store,
    )
    return result, backends_at_run


@pytest.mark.asyncio
async def test_a_workflow_cell_records_the_backend_it_ran_under(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ResultStore(tmp_path / 'results')

    result, backends_at_run = await _run_landlock_cell(tmp_path, monkeypatch, _fake_collect, store)

    assert backends_at_run == ['landlock']
    assert result.metrics['sandbox_backend'] == 'landlock'
    [persisted] = store.load_all()
    assert EvalMetrics.from_dict(persisted.metrics).sandbox_backend == 'landlock'


@pytest.mark.asyncio
async def test_a_cell_whose_metric_collection_raised_still_records_the_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = await _run_landlock_cell(
        tmp_path, monkeypatch, _raising_collect, ResultStore(tmp_path / 'results'),
    )

    assert result.metrics['sandbox_backend'] == 'landlock'
    assert 'role_under_test' in result.metrics


def test_a_cell_recorded_before_the_field_reads_back_as_none() -> None:
    assert EvalMetrics.from_dict({'composite_score': 0.5}).sandbox_backend is None


_REIFY_AGENT_ENV = 'role_env_overrides:\n  implementer:\n    CARGO_HOME: /tmp/eval-probe-cargo-home\n'
_BASE_AGENT_ENV = {'implementer': {'BASE_ONLY': '1'}}


def _project(root: Path, yaml_text: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / PROJECT_CONFIG_FILENAME).write_text(yaml_text)
    return root


def _base(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=tmp_path / 'df', role_env_overrides=_BASE_AGENT_ENV)


def _corpus_task(project_root: Path) -> dict:
    return {'id': 'reify_task_12', 'project_root': str(project_root)}


def test_a_fixture_from_another_project_takes_that_projects_role_env_overrides(
    tmp_path: Path,
) -> None:
    reify = _project(tmp_path / 'reify', _REIFY_AGENT_ENV)

    result = build_eval_orch_config(_IMPL, _corpus_task(reify), _base(tmp_path))

    assert result.role_env_overrides == {'implementer': {'CARGO_HOME': '/tmp/eval-probe-cargo-home'}}


def test_a_fixture_project_env_that_would_outrank_the_candidates_env_is_refused(
    tmp_path: Path,
) -> None:
    reify = _project(
        tmp_path / 'reify',
        'role_env_overrides:\n  implementer:\n    ANTHROPIC_BASE_URL: http://127.0.0.1:9\n',
    )
    candidate = EvalConfig(
        't', 'claude', 'sonnet', 'high', env_overrides={'ANTHROPIC_BASE_URL': 'http://candidate'},
    )

    with pytest.raises(ValueError, match="'implementer': \\['ANTHROPIC_BASE_URL'\\]"):
        build_eval_orch_config(candidate, _corpus_task(reify), _base(tmp_path))


def test_a_fixture_of_the_base_project_keeps_the_bases_role_env_overrides(tmp_path: Path) -> None:
    base = _base(tmp_path)

    result = build_eval_orch_config(_IMPL, _corpus_task(base.project_root), base)

    assert result.role_env_overrides == _BASE_AGENT_ENV


def test_a_fixture_project_with_no_config_gets_the_shipped_default(tmp_path: Path) -> None:
    bare = tmp_path / 'bare'
    bare.mkdir()

    result = build_eval_orch_config(_IMPL, _corpus_task(bare), _base(tmp_path))

    assert result.role_env_overrides == {}


def test_project_role_env_overrides_reads_another_projects_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reify = _project(tmp_path / 'reify', _REIFY_AGENT_ENV)
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(tmp_path / 'no-such-config.yaml'))

    assert project_role_env_overrides(reify) == {
        'implementer': {'CARGO_HOME': '/tmp/eval-probe-cargo-home'},
    }


def test_project_role_env_overrides_warns_on_an_unknown_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    project = _project(tmp_path / 'typo', 'role_env_overrides:\n  implementor:\n    A: "1"\n')
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(tmp_path / 'no-such-config.yaml'))

    with caplog.at_level('WARNING', logger='orchestrator.config'):
        project_role_env_overrides(project)

    assert "unrecognized role 'implementor'" in caplog.text
