"""μ OFAT→matrix→confirm driver in evals/runner.py (task 2478).

Hermetic driver tests. The both-live end-to-end path and the three fan-out
stages compose the EXISTING run_eval / run_architect_eval / run_end_to_end
executors, so every test monkeypatches those executors (the test_runner_matrix
pattern) or mocks build_workflow+collect_metrics (the test_eval_architect
pattern) — no live worktree, no LLM, no cloud call.

Step map:
  step-03/04  build_eval_orch_config(architect_config=...) both-live override
  step-05/06  run_end_to_end (architect + implementer both LIVE)
  step-07/08  run_ofat_stage (role-dispatching fan-out)
  step-09/10  run_matrix_stage (architect×implementer cross product)
  step-11/12  run_confirm_stage (single winning combo × N trials)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import load_config
from orchestrator.evals.configs import EvalConfig
from orchestrator.workflow import WorkflowOutcome


def _base_config(tmp_path: Path):
    """A deterministic pure-code-default base config via the REAL load_config().

    Mirrors test_eval_boundary_suite._load_default_config: write a minimal YAML
    setting only project_root so load_config layers it over the packaged
    defaults.yaml — every leaf resolves to its code default through the real
    production config-load entry point (never a hand-built OrchestratorConfig).
    """
    cfg_path = tmp_path / 'orchestrator.yaml'
    cfg_path.write_text(f'project_root: {tmp_path}\n')
    return load_config(cfg_path)


def _impl_cfg() -> EvalConfig:
    return EvalConfig('claude-sonnet-max', 'claude', 'sonnet', 'max')


def _arch_cfg() -> EvalConfig:
    # model!='opus' and effort!='high' so both diverge from the hardcoded pin.
    return EvalConfig('architect-sonnet-max', 'claude', 'sonnet', 'max', role='architect')


# ---------------------------------------------------------------------------
# step-03/04 — build_eval_orch_config gains an optional architect_config param.
#
# Default None keeps the current opus/claude/high architect pin byte-identical
# (every existing caller + the P1/B1 parity tripwire stay intact); a supplied
# architect_config derives models/backends/effort.architect from the candidate
# for the both-live end-to-end run, leaving implementer/reviewer unchanged.
# ---------------------------------------------------------------------------

class TestBuildEvalOrchConfigArchitectOverride:
    def test_default_none_keeps_opus_architect_pin(self, tmp_path: Path):
        from orchestrator.evals.runner import build_eval_orch_config

        base = _base_config(tmp_path)
        cfg = build_eval_orch_config(_impl_cfg(), {}, base, architect_config=None)

        # Current pin, unchanged: architect stays opus/claude/high.
        assert cfg.models.architect == 'opus'
        assert cfg.backends.architect == 'claude'
        assert cfg.effort.architect == 'high'
        # Implementer still driven by the eval config under test.
        assert cfg.models.implementer == 'sonnet'
        assert cfg.backends.implementer == 'claude'
        # Reviewer still the 1× opus comprehensive reviewer.
        assert cfg.models.reviewer == 'opus'

    def test_architect_config_overrides_architect_fields(self, tmp_path: Path):
        from orchestrator.evals.runner import build_eval_orch_config

        base = _base_config(tmp_path)
        impl, arch = _impl_cfg(), _arch_cfg()
        cfg = build_eval_orch_config(impl, {}, base, architect_config=arch)

        # Architect now derives from the candidate (both-live end-to-end run).
        assert cfg.models.architect == arch.model        # 'sonnet' (was 'opus')
        assert cfg.backends.architect == arch.backend     # 'claude'
        assert cfg.effort.architect == arch.effort         # 'max' (was 'high')

        # Implementer / reviewer fields are untouched by the architect override.
        assert cfg.models.implementer == impl.model        # 'sonnet'
        assert cfg.backends.implementer == impl.backend
        assert cfg.models.reviewer == 'opus'
        assert cfg.backends.reviewer == 'claude'

    def test_architect_config_none_is_backward_compatible_positionally(self, tmp_path: Path):
        # The new param must be keyword-optional with a None default so every
        # existing positional caller (run_eval / run_architect_eval) is intact.
        from orchestrator.evals.runner import build_eval_orch_config

        base = _base_config(tmp_path)
        cfg = build_eval_orch_config(_impl_cfg(), {}, base)
        assert cfg.models.architect == 'opus'


# ---------------------------------------------------------------------------
# step-05/06 — run_end_to_end: the ONE both-live executor (architect LIVE +
# implementer LIVE). It builds the both-live orch config
# (build_eval_orch_config(architect_config=arch)) and constructs the workflow
# via build_workflow(initial_plan=None) so the architect plans live and feeds
# the live implementer — the only place the plan-style/implementer coupling
# question exists (PRD decision 9). Boundaries mocked (test_eval_architect
# pattern): create_eval_worktree / build_workflow / collect_metrics / load_task
# / save_result. GitOps/scheduler/briefing/mcp construct for real (I/O-free) so
# the config threaded into build_workflow is the genuine article we assert on.
# ---------------------------------------------------------------------------

def _e2e_task(tmp_path: Path) -> dict:
    return {
        'id': 'df_task_e2e',
        'project_root': str(tmp_path),
        'pre_task_commit': 'basecommit123',
        'task_definition': {'title': 'Widget', 'description': 'build the widget'},
        'modules': ['pkg/mod.py'],
    }


async def _run_end_to_end_hermetic(
    arch_cfg: EvalConfig,
    impl_cfg: EvalConfig,
    base,
    task: dict,
    monkeypatch: pytest.MonkeyPatch,
    *,
    outcome: WorkflowOutcome = WorkflowOutcome.DONE,
):
    """Drive run_end_to_end with the worktree/workflow/metrics boundaries mocked.

    Returns ``(result, captured, mocks)`` where ``captured['build_workflow']``
    is the kwargs dict build_workflow received (for asserting config + plan).
    """
    from orchestrator.evals import runner

    captured: dict = {}

    async def fake_create_wt(*_a, **_k):
        return Path('/fake/wt'), 'run-e2e'

    fake_wf = MagicMock()
    fake_wf.run = AsyncMock(return_value=SimpleNamespace(outcome=outcome))

    def fake_build_workflow(**kwargs):
        captured['build_workflow'] = kwargs
        return fake_wf

    metrics_obj = MagicMock()
    metrics_obj.to_dict.return_value = {'composite_score': 0.9, 'tests_pass': True}
    mock_collect = AsyncMock(return_value=metrics_obj)
    mock_save = MagicMock()

    monkeypatch.setattr(runner, 'create_eval_worktree', fake_create_wt)
    monkeypatch.setattr(runner, 'build_workflow', fake_build_workflow)
    monkeypatch.setattr(runner, 'collect_metrics', mock_collect)
    monkeypatch.setattr(runner, 'load_task', lambda _p: task)
    monkeypatch.setattr(runner, 'save_result', mock_save)

    result = await runner.run_end_to_end(
        Path('/fake/task.json'), arch_cfg, impl_cfg, base,
    )
    return result, captured, {'collect': mock_collect, 'save': mock_save, 'wf': fake_wf}


@pytest.mark.asyncio
class TestRunEndToEnd:
    async def test_builds_both_live_config_and_live_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        base = _base_config(tmp_path)
        arch, impl = _arch_cfg(), _impl_cfg()
        _result, captured, _ = await _run_end_to_end_hermetic(
            arch, impl, base, _e2e_task(tmp_path), monkeypatch,
        )

        # (a) both-live orch config: architect AND implementer from the candidates.
        kw = captured['build_workflow']
        cfg = kw['config']
        assert cfg.models.architect == arch.model      # architect LIVE (sonnet)
        assert cfg.models.implementer == impl.model     # implementer LIVE (sonnet)
        assert cfg.backends.architect == arch.backend

        # (b) build_workflow gets initial_plan=None → the architect plans LIVE
        # (NOT a frozen plan handed in like run_eval does).
        assert kw['initial_plan'] is None

    async def test_result_encodes_combo_and_tags_end_to_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        base = _base_config(tmp_path)
        arch, impl = _arch_cfg(), _impl_cfg()
        result, _captured, mocks = await _run_end_to_end_hermetic(
            arch, impl, base, _e2e_task(tmp_path), monkeypatch,
        )

        # (c) config_name encodes BOTH the architect and implementer ids.
        assert arch.name in result.config_name
        assert impl.name in result.config_name
        # role_under_test stamped 'end_to_end' (distinct from implementer/architect).
        assert result.metrics['role_under_test'] == 'end_to_end'
        assert result.task_id == 'df_task_e2e'
        assert result.outcome == 'done'

        # (d) persisted via save_result.
        mocks['save'].assert_called_once()
