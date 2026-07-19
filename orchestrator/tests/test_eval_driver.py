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
from orchestrator.evals.runner import EvalResult
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
# step-01/02 — build_eval_orch_config gains an optional judge_config param.
#
# Default None keeps the current sonnet/medium/claude judge pin byte-identical
# (every existing caller + the P1/B1 parity tripwire stay intact); a supplied
# judge_config derives ONLY models.judge / effort.judge from the candidate for
# the judge OFAT axis, leaving backends.judge / budgets.judge PINNED (always-
# Claude read-only judge) and implementer/architect/reviewer untouched.
# ---------------------------------------------------------------------------

class TestBuildEvalOrchConfigJudgeOverride:
    def test_default_none_keeps_sonnet_judge_pin(self, tmp_path: Path):
        from orchestrator.evals.runner import build_eval_orch_config

        base = _base_config(tmp_path)
        cfg = build_eval_orch_config(_impl_cfg(), {}, base, judge_config=None)

        # Current pin, unchanged (byte-identical parity tripwire): the completion
        # judge stays sonnet/medium/claude at its pinned 0.50 budget.
        assert cfg.models.judge == 'sonnet'
        assert cfg.effort.judge == 'medium'
        assert cfg.budgets.judge == 0.50
        assert cfg.backends.judge == 'claude'
        # Implementer / architect / reviewer are untouched by the judge knob.
        assert cfg.models.implementer == 'sonnet'
        assert cfg.models.architect == 'opus'
        assert cfg.models.reviewer == 'opus'

    def test_judge_config_overrides_judge_model_and_effort(self, tmp_path: Path):
        from orchestrator.evals.runner import build_eval_orch_config

        base = _base_config(tmp_path)
        # model!='sonnet' AND effort!='medium' so BOTH derived fields diverge.
        judge = EvalConfig('judge-haiku-high', 'claude', 'haiku', 'high', role='judge')
        cfg = build_eval_orch_config(_impl_cfg(), {}, base, judge_config=judge)

        # The judge's model/effort now derive from the candidate (judge OFAT axis).
        assert cfg.models.judge == 'haiku'
        assert cfg.effort.judge == 'high'
        # Backend and budget stay PINNED (not derived) — always-Claude read-only judge.
        assert cfg.backends.judge == 'claude'
        assert cfg.budgets.judge == 0.50
        # Implementer / architect / reviewer are untouched by the judge override.
        assert cfg.models.implementer == 'sonnet'
        assert cfg.models.architect == 'opus'
        assert cfg.models.reviewer == 'opus'

    def test_judge_config_none_backward_compatible_positionally(self, tmp_path: Path):
        # The new param must be keyword-optional with a None default so every
        # existing positional caller (run_eval / run_end_to_end) is intact.
        from orchestrator.evals.runner import build_eval_orch_config

        base = _base_config(tmp_path)
        cfg = build_eval_orch_config(_impl_cfg(), {}, base)
        assert cfg.models.judge == 'sonnet'


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


# ---------------------------------------------------------------------------
# step-07/08 — run_ofat_stage: role-dispatching bounded-concurrency fan-out.
#
# OFAT reuses the EXISTING frozen-input executors (decision 9): an implementer
# candidate (role=='implementer') dispatches to run_eval (frozen plan), an
# architect candidate (role=='architect') to run_architect_eval (live architect,
# downstream frozen). It is a role-dispatching fan-out, not new per-role
# machinery. Mirrors test_runner_matrix's monkeypatch + non-cancel-continue
# regression guard.
# ---------------------------------------------------------------------------

def _ofat_task_loader(path: Path) -> dict:
    return {'id': path.stem, 'project_root': '/fake', 'pre_task_commit': 'x'}


@pytest.mark.asyncio
class TestRunOfatStage:
    async def test_dispatches_each_candidate_by_role_over_every_cell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from orchestrator.evals import runner

        t1 = tmp_path / 'df_task_a.json'
        t2 = tmp_path / 'df_task_b.json'
        t1.touch()
        t2.touch()
        impl_cfg = EvalConfig('claude-opus-high', 'claude', 'opus', 'high')
        arch_cfg = EvalConfig('architect-sonnet-high', 'claude', 'sonnet', 'high', role='architect')

        eval_calls: list[tuple[str, str, int]] = []
        arch_calls: list[tuple[str, str, int]] = []

        async def fake_run_eval(task_path, config, *_a, trial=1, **_k):
            eval_calls.append((task_path.stem, config.name, trial))
            return EvalResult(task_path.stem, config.name, 'done', {}, '/tmp/wt', trial=trial)

        async def fake_run_arch(task_path, config, *_a, trial=1, **_k):
            arch_calls.append((task_path.stem, config.name, trial))
            return EvalResult(task_path.stem, config.name, 'done',
                              {'role_under_test': 'architect'}, '/tmp/wt', trial=trial)

        monkeypatch.setattr(runner, 'load_task', _ofat_task_loader)
        monkeypatch.setattr(runner, 'run_eval', fake_run_eval)
        monkeypatch.setattr(runner, 'run_architect_eval', fake_run_arch)

        results = await runner.run_ofat_stage(
            [t1, t2], [impl_cfg, arch_cfg], base_config=None, trials=2,
        )

        # Implementer candidate → run_eval; architect candidate → run_architect_eval.
        assert {c[1] for c in eval_calls} == {'claude-opus-high'}
        assert {c[1] for c in arch_calls} == {'architect-sonnet-high'}
        # Exactly one dispatch per (fixture, trial) per candidate: 2 fixtures × 2 trials.
        assert len(eval_calls) == 4
        assert len(arch_calls) == 4
        # Flattened results cover every (candidate, fixture, trial) cell.
        assert len(results) == 2 * 2 * 2  # candidates × fixtures × trials

    async def test_non_cancel_failure_in_one_cell_does_not_abort_others(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        import logging

        from orchestrator.evals import runner

        t_ok = tmp_path / 'df_task_ok.json'
        t_fail = tmp_path / 'df_task_fail.json'
        t_ok.touch()
        t_fail.touch()
        impl_cfg = EvalConfig('claude-opus-high', 'claude', 'opus', 'high')

        async def fake_run_eval(task_path, config, *_a, trial=1, **_k):
            if 'fail' in task_path.stem:
                raise RuntimeError('boom in one cell')
            return EvalResult(task_path.stem, config.name, 'done', {}, '/tmp/wt', trial=trial)

        monkeypatch.setattr(runner, 'load_task', _ofat_task_loader)
        monkeypatch.setattr(runner, 'run_eval', fake_run_eval)

        with caplog.at_level(logging.ERROR, logger='orchestrator.evals.runner'):
            results = await runner.run_ofat_stage(
                [t_ok, t_fail], [impl_cfg], base_config=None, trials=1,
            )

        # The failing cell is logged and skipped; the ok cell still returns.
        assert len(results) == 1
        assert results[0].task_id == 'df_task_ok'
        assert any('failed' in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# step-09/10 — run_matrix_stage: the architect×implementer cross product.
#
# The matrix stage is one of the two both-live stages (matrix/confirm): it fans
# run_end_to_end out over configs.matrix_pairs(arch_survivors, impl_survivors) ×
# fixtures × trials — the FULL cross product, INCLUDING the same-family diagonal
# (e.g. sonnet-arch × sonnet-impl), the pair that tests whether a plan style
# couples to its own family's implementer (PRD decision 9). run_end_to_end is
# monkeypatched to record (arch, impl) pairs — the ONLY both-live executor.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRunMatrixStage:
    async def test_runs_end_to_end_over_full_cross_product_incl_diagonal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from orchestrator.evals import configs, runner

        t1 = tmp_path / 'df_task_a.json'
        t2 = tmp_path / 'df_task_b.json'
        t1.touch()
        t2.touch()

        # 2 architect × 2 implementer survivors. The same-model/backend pairs
        # (arch-sonnet × impl-sonnet, arch-opus × impl-opus) are the same-family
        # diagonals the matrix must NOT skip.
        arch_survivors = [
            EvalConfig('arch-sonnet', 'claude', 'sonnet', 'high', role='architect'),
            EvalConfig('arch-opus', 'claude', 'opus', 'high', role='architect'),
        ]
        impl_survivors = [
            EvalConfig('impl-sonnet', 'claude', 'sonnet', 'high'),
            EvalConfig('impl-opus', 'claude', 'opus', 'high'),
        ]

        e2e_calls: list[tuple[str, str, str, int]] = []

        async def fake_run_end_to_end(
            task_path, arch_config, impl_config, *_a, trial=1, **_k,
        ):
            e2e_calls.append(
                (task_path.stem, arch_config.name, impl_config.name, trial)
            )
            return EvalResult(
                task_path.stem, f'{arch_config.name}+{impl_config.name}',
                'done', {'role_under_test': 'end_to_end'}, '/tmp/wt', trial=trial,
            )

        monkeypatch.setattr(runner, 'run_end_to_end', fake_run_end_to_end)

        results = await runner.run_matrix_stage(
            [t1, t2], arch_survivors, impl_survivors, base_config=None, trials=1,
        )

        # Every (arch, impl) pair from the FULL cross product is covered.
        expected_pairs = {
            (a.name, i.name)
            for a, i in configs.matrix_pairs(arch_survivors, impl_survivors)
        }
        seen_pairs = {(c[1], c[2]) for c in e2e_calls}
        assert seen_pairs == expected_pairs
        # Full cross product = len(arch) × len(impl) = 4 pairs; both same-family
        # diagonals present (NOT excluded).
        assert ('arch-sonnet', 'impl-sonnet') in seen_pairs
        assert ('arch-opus', 'impl-opus') in seen_pairs
        # Exactly one run_end_to_end per (pair, fixture, trial): 4 × 2 × 1.
        assert len(e2e_calls) == 4 * 2 * 1
        # Flattened results cover every cell.
        assert len(results) == 4 * 2 * 1
        assert all(r.metrics['role_under_test'] == 'end_to_end' for r in results)

    async def test_covers_pairs_across_multiple_trials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from orchestrator.evals import runner

        t1 = tmp_path / 'df_task_a.json'
        t1.touch()

        arch_survivors = [
            EvalConfig('arch-sonnet', 'claude', 'sonnet', 'high', role='architect'),
        ]
        impl_survivors = [
            EvalConfig('impl-sonnet', 'claude', 'sonnet', 'high'),
            EvalConfig('impl-opus', 'claude', 'opus', 'high'),
        ]

        seen_trials: list[int] = []

        async def fake_run_end_to_end(task_path, arch_config, impl_config, *_a, trial=1, **_k):
            seen_trials.append(trial)
            return EvalResult(
                task_path.stem, f'{arch_config.name}+{impl_config.name}',
                'done', {}, '/tmp/wt', trial=trial,
            )

        monkeypatch.setattr(runner, 'run_end_to_end', fake_run_end_to_end)

        results = await runner.run_matrix_stage(
            [t1], arch_survivors, impl_survivors, base_config=None, trials=3,
        )

        # 1 arch × 2 impl = 2 pairs × 1 fixture × 3 trials = 6 cells.
        assert len(results) == 2 * 1 * 3
        assert sorted(seen_trials) == [1, 1, 2, 2, 3, 3]


# ---------------------------------------------------------------------------
# step-11/12 — run_confirm_stage: ONE end-to-end confirmation batch of the winner.
#
# The final both-live stage: the SINGLE winning (arch, impl) combo run across
# every fixture × N trials, N>=3 by default (decision 10's statistics floor —
# enough trials for a CI95 on the winner, NOT the 1-trial screen default of the
# OFAT/matrix stages). run_end_to_end is monkeypatched to record the combo.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRunConfirmStage:
    async def test_runs_single_winning_combo_over_fixtures_and_trials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from orchestrator.evals import runner

        t1 = tmp_path / 'df_task_a.json'
        t2 = tmp_path / 'df_task_b.json'
        t1.touch()
        t2.touch()

        arch_winner = EvalConfig('arch-opus', 'claude', 'opus', 'high', role='architect')
        impl_winner = EvalConfig('impl-sonnet', 'claude', 'sonnet', 'high')

        e2e_calls: list[tuple[str, str, str, int]] = []

        async def fake_run_end_to_end(task_path, arch_config, impl_config, *_a, trial=1, **_k):
            e2e_calls.append((task_path.stem, arch_config.name, impl_config.name, trial))
            return EvalResult(
                task_path.stem, f'{arch_config.name}+{impl_config.name}',
                'done', {'role_under_test': 'end_to_end'}, '/tmp/wt', trial=trial,
            )

        monkeypatch.setattr(runner, 'run_end_to_end', fake_run_end_to_end)

        results = await runner.run_confirm_stage(
            [t1, t2], arch_winner, impl_winner, base_config=None, trials=4,
        )

        # Exactly ONE combo — the winner — across every cell.
        seen_pairs = {(c[1], c[2]) for c in e2e_calls}
        assert seen_pairs == {('arch-opus', 'impl-sonnet')}
        # N trials per fixture: 2 fixtures × 4 trials = 8 cells.
        assert len(e2e_calls) == 2 * 4
        assert len(results) == 2 * 4
        # N distinct trials per fixture (all confirmation trials of the winner).
        trials_for_t1 = sorted(c[3] for c in e2e_calls if c[0] == 'df_task_a')
        assert trials_for_t1 == [1, 2, 3, 4]

    async def test_default_trials_meets_statistics_floor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        # Decision 10's statistics floor: the confirmation batch defaults to
        # N>=3 trials (enough for a CI95 on the winner) — NOT the 1-trial default
        # of the OFAT/matrix screen stages.
        from orchestrator.evals import runner

        t1 = tmp_path / 'df_task_a.json'
        t1.touch()

        arch_winner = EvalConfig('arch-opus', 'claude', 'opus', 'high', role='architect')
        impl_winner = EvalConfig('impl-sonnet', 'claude', 'sonnet', 'high')

        async def fake_run_end_to_end(task_path, arch_config, impl_config, *_a, trial=1, **_k):
            return EvalResult(
                task_path.stem, f'{arch_config.name}+{impl_config.name}',
                'done', {}, '/tmp/wt', trial=trial,
            )

        monkeypatch.setattr(runner, 'run_end_to_end', fake_run_end_to_end)

        # Called WITHOUT an explicit trials= → the N>=3 default kicks in.
        results = await runner.run_confirm_stage(
            [t1], arch_winner, impl_winner, base_config=None,
        )

        # >=3 results for the single fixture, distinct trials 1..N.
        assert len(results) >= 3
        assert {r.trial for r in results} >= {1, 2, 3}
