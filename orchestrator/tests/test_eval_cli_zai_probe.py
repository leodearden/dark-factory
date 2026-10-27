"""The eval CLI drivers probe the Z.ai Coding Plan once, before the first GLM cell (task 5997).

A wrong Z.ai endpoint or key must fail the command once, loudly, rather than
degrade into a 4xx on every GLM cell that reads as a model failure (INV-11).
Every driver that can open an eval cell passes its candidate set through the
same gate immediately before its first cell.

Hermetic: the runner stages are faked, so no worktree, LLM or network work
runs. The failure path drives the REAL probe through its missing-key branch,
which sends no request; the success path replaces the public
``zai_probe.require_zai_coding_endpoint``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from _eval_config_lookup import named_eval_config
from click.testing import CliRunner, Result
from shared.config_models import UsageCapConfig

import orchestrator.cli as cli_module
import orchestrator.evals.runner as runner_module
from orchestrator.config import OrchestratorConfig
from orchestrator.evals import configs as configs_module
from orchestrator.evals import zai_probe
from orchestrator.evals.results import EvalResult

_GLM_NAMES = ('glm-5.3-endpoint', 'glm-5.3-flash-endpoint')
_PROBE = ('probe',)


@dataclass(frozen=True)
class _Paths:
    config: Path
    task: Path
    tasks_dir: Path


def _first_ofat(role: str) -> str:
    return next(c.name for c in configs_module.ofat_candidates() if c.role == role)


def _no_results(*_args, **_kwargs) -> list[EvalResult]:
    return []


def _one_result(_task_path, cfg, *_args, **_kwargs) -> EvalResult:
    return EvalResult(
        task_id='df_task_10', config_name=cfg.name, outcome='success',
        metrics={}, worktree_path='/tmp/wt', wall_clock_ms=1,
    )


@dataclass(frozen=True)
class _Invocation:
    """One eval CLI command line, the runner stage it opens cells through, and the
    GLM configs it must add to ``ofat_candidates()`` to reach Z.ai at all."""

    argv: Callable[[_Paths], list[str]]
    stage: str
    stage_result: Callable[..., object] = _no_results
    ofat_additions: tuple[str, ...] = ()


_ZAI_INVOCATIONS = {
    'eval': _Invocation(
        lambda p: [
            'eval', '--task', str(p.task), '--config-name', 'glm-5.3-endpoint',
            '--config', str(p.config),
        ],
        'run_eval', stage_result=_one_result,
    ),
    'eval-confirm': _Invocation(
        lambda p: [
            'eval-confirm', '--config', str(p.config), '--arch', _first_ofat('architect'),
            '--impl', 'glm-5.3-flash-endpoint', '--tasks-dir', str(p.tasks_dir),
        ],
        'run_confirm_stage',
    ),
    'eval-ofat': _Invocation(
        lambda p: ['eval-ofat', '--config', str(p.config), '--tasks-dir', str(p.tasks_dir)],
        'run_ofat_stage', ofat_additions=('glm-5.3-endpoint',),
    ),
    'eval-matrix': _Invocation(
        lambda p: ['eval-matrix', '--config', str(p.config), '--tasks-dir', str(p.tasks_dir)],
        'run_matrix_stage', ofat_additions=('glm-5.3-endpoint',),
    ),
}

_NON_ZAI_INVOCATIONS = {
    'eval': replace(
        _ZAI_INVOCATIONS['eval'],
        argv=lambda p: [
            'eval', '--task', str(p.task), '--config-name', 'claude-opus-high',
            '--config', str(p.config),
        ],
    ),
    'eval-confirm': replace(
        _ZAI_INVOCATIONS['eval-confirm'],
        argv=lambda p: [
            'eval-confirm', '--config', str(p.config), '--arch', _first_ofat('architect'),
            '--impl', _first_ofat('implementer'), '--tasks-dir', str(p.tasks_dir),
        ],
    ),
    'eval-ofat': replace(_ZAI_INVOCATIONS['eval-ofat'], ofat_additions=()),
    'eval-matrix': replace(_ZAI_INVOCATIONS['eval-matrix'], ofat_additions=()),
}


@pytest.fixture
def base_config(tmp_path: Path) -> OrchestratorConfig:
    """Usage cap off, so the single-eval campaign gate degrades to ungated publicly."""
    return OrchestratorConfig(project_root=tmp_path, usage_cap=UsageCapConfig(enabled=False))


@pytest.fixture
def dummy_cfg_file(tmp_path: Path) -> Path:
    """A real file so ``click.Path(exists=True)`` on ``--config`` passes."""
    p = tmp_path / 'dummy.yaml'
    p.write_text('')
    return p


@pytest.fixture
def fixtures_dir(tmp_path: Path) -> Path:
    """A tasks dir with one valid fixture so fixture discovery is non-empty."""
    d = tmp_path / 'tasks'
    d.mkdir()
    (d / 'df_task_10.json').write_text(
        '{"id": "df_task_10", "project_root": "/tmp", "description": "x"}\n'
    )
    return d


@pytest.fixture
def task_file(fixtures_dir: Path) -> Path:
    return fixtures_dir / 'df_task_10.json'


@pytest.fixture
def paths(dummy_cfg_file: Path, task_file: Path, fixtures_dir: Path) -> _Paths:
    return _Paths(config=dummy_cfg_file, task=task_file, tasks_dir=fixtures_dir)


@pytest.fixture
def events() -> list[tuple[str, ...]]:
    """Probe and stage entries in call order: the ordering and the count are the claims."""
    return []


@pytest.fixture(autouse=True)
def _cli_loads_base_config(monkeypatch: pytest.MonkeyPatch, base_config: OrchestratorConfig):
    monkeypatch.setattr(cli_module, 'load_config', lambda _: base_config)


def _invoke(
    invocation: _Invocation, paths: _Paths, events: list[tuple[str, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> Result:
    """Fake *invocation*'s stage, widen ``ofat_candidates()`` as it needs, and run it."""
    argv = invocation.argv(paths)
    if invocation.ofat_additions:
        real_ofat = configs_module.ofat_candidates()
        widened = [*real_ofat, *(named_eval_config(name) for name in invocation.ofat_additions)]
        monkeypatch.setattr(configs_module, 'ofat_candidates', lambda: widened)

    async def fake_stage(*args, **kwargs):
        events.append(('stage', invocation.stage))
        return invocation.stage_result(*args, **kwargs)

    monkeypatch.setattr(runner_module, invocation.stage, fake_stage)
    return CliRunner().invoke(cli_module.main, argv)


def _record_probe(monkeypatch: pytest.MonkeyPatch, events: list[tuple[str, ...]]) -> None:
    def fake_probe(**_kwargs) -> zai_probe.ZaiProbeOutcome:
        events.append(_PROBE)
        return zai_probe.ZaiProbeOutcome(configs_module.ZAI_CODING_BASE_URL, 200, 'HTTP 200')

    monkeypatch.setattr(zai_probe, 'require_zai_coding_endpoint', fake_probe)


@pytest.mark.parametrize('invocation', _ZAI_INVOCATIONS.values(), ids=_ZAI_INVOCATIONS.keys())
def test_a_missing_zai_key_fails_the_command_loudly_before_any_cell(
    invocation: _Invocation, paths: _Paths, events: list[tuple[str, ...]],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv('ZAI_API_KEY', raising=False)

    r = _invoke(invocation, paths, events, monkeypatch)

    assert r.exit_code == 1, r.output
    assert 'ZAI_API_KEY' in r.output
    assert configs_module.ZAI_CODING_BASE_URL in r.output
    assert events == []


@pytest.mark.parametrize('invocation', [
    _ZAI_INVOCATIONS['eval'],
    replace(_ZAI_INVOCATIONS['eval-ofat'], ofat_additions=_GLM_NAMES),
], ids=['eval', 'eval-ofat-two-glm-candidates'])
def test_a_passing_probe_runs_once_before_the_first_cell(
    invocation: _Invocation, paths: _Paths, events: list[tuple[str, ...]],
    monkeypatch: pytest.MonkeyPatch,
):
    _record_probe(monkeypatch, events)

    r = _invoke(invocation, paths, events, monkeypatch)

    assert r.exit_code == 0, r.output
    assert events[0] == _PROBE
    assert events.count(_PROBE) == 1
    assert ('stage', invocation.stage) in events


@pytest.mark.parametrize(
    'invocation', _NON_ZAI_INVOCATIONS.values(), ids=_NON_ZAI_INVOCATIONS.keys(),
)
def test_non_zai_candidates_never_probe(
    invocation: _Invocation, paths: _Paths, events: list[tuple[str, ...]],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv('ZAI_API_KEY', raising=False)
    _record_probe(monkeypatch, events)

    r = _invoke(invocation, paths, events, monkeypatch)

    assert r.exit_code == 0, r.output
    assert events == [('stage', invocation.stage)]
