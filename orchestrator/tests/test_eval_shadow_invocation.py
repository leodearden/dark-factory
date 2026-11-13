"""The shadow invocation profile a live shadow cell runs under (task 5386).

``build_shadow_orch_config`` is the corpus eval profile plus exactly one
divergence: sandboxed roles run under the operator's configured sandbox
backend. Strict MCP and the null memory sentinel are inherited from
``EVAL_PROFILE``, never re-set.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from orchestrator.config import OrchestratorConfig, SandboxConfig, diff_config
from orchestrator.evals import runner
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.live_fixture import build_live_fixture
from orchestrator.evals.runner import build_eval_orch_config, run_end_to_end, run_eval
from orchestrator.evals.shadow_invocation import (
    build_shadow_orch_config,
    run_shadow_end_to_end,
    run_shadow_eval,
)
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
    diff = diff_config(a, b)
    return set(diff.applied_candidates) | set(diff.restart_required)


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


class _Sentinel(Exception):
    """Stops a runner at the seam under test, before any workflow runs."""


class _RecordingBuilder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        config: EvalConfig,
        task: dict,
        base_config: OrchestratorConfig | None = None,
        memory_endpoint: str | None = None,
        architect_config: EvalConfig | None = None,
        judge_config: EvalConfig | None = None,
    ) -> OrchestratorConfig:
        self.calls.append({
            'config': config, 'task': task, 'base_config': base_config,
            'memory_endpoint': memory_endpoint, 'architect_config': architect_config,
            'judge_config': judge_config,
        })
        raise _Sentinel


@pytest.mark.asyncio
async def test_run_eval_builds_the_cell_config_with_the_injected_builder(
    tmp_path: Path, monkeypatch,
) -> None:
    fixture = _shadow_fixture(tmp_path)
    monkeypatch.setattr(runner, 'load_task', lambda _path: fixture)
    base = OrchestratorConfig(project_root=tmp_path)
    judge = EvalConfig('judge', 'claude', 'haiku', 'low')
    recorder = _RecordingBuilder()

    with pytest.raises(_Sentinel):
        await run_eval(
            tmp_path / 'task.json', _IMPL, base,
            worktree_path=tmp_path, memory_endpoint='http://127.0.0.1:54321',
            judge_config=judge, orch_config_builder=recorder,
        )

    assert recorder.calls == [{
        'config': _IMPL, 'task': fixture, 'base_config': base,
        'memory_endpoint': 'http://127.0.0.1:54321', 'architect_config': None,
        'judge_config': judge,
    }]


@pytest.mark.asyncio
async def test_run_end_to_end_builds_the_cell_config_with_the_injected_builder(
    tmp_path: Path, monkeypatch,
) -> None:
    fixture = _shadow_fixture(tmp_path)
    monkeypatch.setattr(runner, 'load_task', lambda _path: fixture)
    monkeypatch.setattr(
        runner, 'create_eval_worktree', AsyncMock(return_value=(tmp_path / 'wt', 'run-e2e')),
    )
    base = OrchestratorConfig(project_root=tmp_path)
    architect = EvalConfig('arch', 'claude', 'opus', 'high')
    recorder = _RecordingBuilder()

    with pytest.raises(_Sentinel):
        await run_end_to_end(
            tmp_path / 'task.json', architect, _IMPL, base,
            memory_endpoint='http://127.0.0.1:54321', orch_config_builder=recorder,
        )

    assert recorder.calls == [{
        'config': _IMPL, 'task': fixture, 'base_config': base,
        'memory_endpoint': 'http://127.0.0.1:54321', 'architect_config': architect,
        'judge_config': None,
    }]


async def _workflow_config_built_by(
    tmp_path: Path, monkeypatch, run: Callable[[OrchestratorConfig], Awaitable[object]],
) -> OrchestratorConfig:
    """The OrchestratorConfig *run* hands ``build_workflow`` for a shadow fixture."""
    captured: dict[str, Any] = {}

    def stop_at_build_workflow(**kwargs: Any):
        captured.update(kwargs)
        raise _Sentinel

    monkeypatch.setattr(runner, 'load_task', lambda _path: _shadow_fixture(tmp_path))
    monkeypatch.setattr(
        runner, 'create_eval_worktree', AsyncMock(return_value=(tmp_path / 'wt', 'run-shadow')),
    )
    monkeypatch.setattr(runner, 'build_workflow', stop_at_build_workflow)

    with pytest.raises(_Sentinel):
        await run(OrchestratorConfig(project_root=tmp_path))

    return captured['config']


@pytest.mark.asyncio
async def test_run_eval_under_the_shadow_builder_runs_its_cell_sandboxed(
    tmp_path: Path, monkeypatch,
) -> None:
    config = await _workflow_config_built_by(
        tmp_path, monkeypatch,
        lambda base: run_eval(
            tmp_path / 'task.json', _IMPL, base, worktree_path=tmp_path, usage_gate=None,
            orch_config_builder=build_shadow_orch_config,
        ),
    )

    assert config.sandbox.enabled is True


@pytest.mark.asyncio
async def test_run_eval_by_default_runs_its_cell_unsandboxed(tmp_path: Path, monkeypatch) -> None:
    config = await _workflow_config_built_by(
        tmp_path, monkeypatch,
        lambda base: run_eval(
            tmp_path / 'task.json', _IMPL, base, worktree_path=tmp_path, usage_gate=None,
        ),
    )

    assert config.sandbox.enabled is False


@pytest.mark.asyncio
async def test_run_shadow_eval_runs_its_cell_sandboxed(tmp_path: Path, monkeypatch) -> None:
    config = await _workflow_config_built_by(
        tmp_path, monkeypatch,
        lambda base: run_shadow_eval(
            tmp_path / 'task.json', _IMPL, base, worktree_path=tmp_path, usage_gate=None,
        ),
    )

    assert config.sandbox.enabled is True


@pytest.mark.asyncio
async def test_run_shadow_end_to_end_runs_its_cell_sandboxed(tmp_path: Path, monkeypatch) -> None:
    architect = EvalConfig('arch', 'claude', 'opus', 'high')

    config = await _workflow_config_built_by(
        tmp_path, monkeypatch,
        lambda base: run_shadow_end_to_end(
            tmp_path / 'task.json', architect, _IMPL, base, usage_gate=None,
        ),
    )

    assert config.sandbox.enabled is True
    assert config.models.architect == 'opus'
