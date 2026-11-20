"""S1 μ (task 9849), boundary row 28: corpus and shadow eval cells run under the base's sandbox."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from orchestrator.agents import sandbox_dispatch
from orchestrator.config import OrchestratorConfig, SandboxConfig
from orchestrator.evals import runner
from orchestrator.evals.cell_sandbox import bind_cell_sandbox_backend
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.live_fixture import build_live_fixture
from orchestrator.evals.metrics import EvalMetrics
from orchestrator.evals.results import ResultStore
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


def test_bind_reports_none_when_the_configured_backend_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _landlock_available(monkeypatch, False)
    config = OrchestratorConfig(
        project_root=tmp_path, sandbox=SandboxConfig(enabled=True, backend='landlock'),
    )

    assert bind_cell_sandbox_backend(config) == 'none'


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
):
    """One executor cell over a landlock base, in a process whose backend no Harness bound."""
    sandbox_dispatch.set_backend('auto')
    _landlock_available(monkeypatch, True)
    workflow = MagicMock()
    workflow.run = AsyncMock(return_value=SimpleNamespace(outcome=WorkflowOutcome.DONE))
    monkeypatch.setattr(runner, 'create_eval_worktree', _fake_create_wt)
    monkeypatch.setattr(runner, 'build_workflow', lambda **_kw: workflow)
    monkeypatch.setattr(runner, 'collect_metrics', AsyncMock(side_effect=collect))
    monkeypatch.setattr(runner, 'load_task', lambda _p: _executor_task(tmp_path))
    base = OrchestratorConfig(
        project_root=tmp_path, sandbox=SandboxConfig(enabled=True, backend='landlock'),
    )

    return await runner.run_eval(
        Path('/fake/task.json'), _IMPL, base, usage_gate=None, store=store,
    )


@pytest.mark.asyncio
async def test_a_workflow_cell_records_the_backend_it_ran_under(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ResultStore(tmp_path / 'results')

    result = await _run_landlock_cell(tmp_path, monkeypatch, _fake_collect, store)

    assert result.metrics['sandbox_backend'] == 'landlock'
    [persisted] = store.load_all()
    assert EvalMetrics.from_dict(persisted.metrics).sandbox_backend == 'landlock'
    assert sandbox_dispatch.get_backend() == 'landlock'


@pytest.mark.asyncio
async def test_a_cell_whose_metric_collection_raised_still_records_the_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await _run_landlock_cell(
        tmp_path, monkeypatch, _raising_collect, ResultStore(tmp_path / 'results'),
    )

    assert result.metrics['sandbox_backend'] == 'landlock'
    assert 'role_under_test' in result.metrics


def test_a_cell_recorded_before_the_field_reads_back_as_none() -> None:
    assert EvalMetrics.from_dict({'composite_score': 0.5}).sandbox_backend is None
