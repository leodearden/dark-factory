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
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from _campaign_gate_helpers import (
    NO_INJECTED_GATE as _NO_INJECTED_GATE,
)
from _campaign_gate_helpers import (
    CampaignGateProbe as _CampaignGateProbe,
)
from _campaign_gate_helpers import (
    assert_a_degraded_campaign_stays_ungated as _assert_a_degraded_campaign_stays_ungated,
)
from _campaign_gate_helpers import (
    assert_one_gate_serves_every_cell as _assert_one_gate_serves_every_cell,
)
from _campaign_gate_helpers import (
    assert_teardown_survives_a_failing_cell as _assert_teardown_survives_a_failing_cell,
)
from _campaign_gate_helpers import (
    assert_teardown_survives_cancellation as _assert_teardown_survives_cancellation,
)
from _campaign_gate_helpers import (
    eval_base_config as _base_config,
)

from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.runner import EvalResult
from orchestrator.workflow import WorkflowOutcome


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
# task 2847 BUG 2a — build_eval_orch_config pins the eval verify interpreter to
# the target's .python-version via verify_env['UV_PYTHON'], so `uv run
# pytest/ruff/pyright` runs under the worktree's own 3.13 venv. Absent a
# .python-version, no pin is injected (fail-safe).
# ---------------------------------------------------------------------------

class TestBuildEvalOrchConfigVerifyPythonPin:
    def test_pins_uv_python_from_project_python_version(self, tmp_path: Path):
        from orchestrator.evals.runner import build_eval_orch_config

        # The target project_root pins 3.13 (df fixtures do); base carries a
        # pre-existing verify_env key that must survive the merge.
        (tmp_path / '.python-version').write_text('3.13\n')
        base = _base_config(tmp_path).model_copy(
            update={'verify_env': {'PRESET': 'x'}},
        )
        task = {'project_root': str(tmp_path)}

        cfg = build_eval_orch_config(_impl_cfg(), task, base)

        # Base/profiled keys preserved; the pin is added and wins on conflict.
        assert cfg.verify_env == {'PRESET': 'x', 'UV_PYTHON': '3.13'}

    def test_no_pin_when_project_has_no_python_version(self, tmp_path: Path):
        from orchestrator.evals.runner import build_eval_orch_config

        # No .python-version under project_root → inject nothing (fail-safe):
        # verify_env is left exactly as the profiled/base value.
        base = _base_config(tmp_path).model_copy(
            update={'verify_env': {'PRESET': 'x'}},
        )
        task = {'project_root': str(tmp_path)}

        cfg = build_eval_orch_config(_impl_cfg(), task, base)

        assert cfg.verify_env == {'PRESET': 'x'}
        assert 'UV_PYTHON' not in cfg.verify_env


# ---------------------------------------------------------------------------
# _Sentinel: a short-circuit exception reused by the verify-pin capturing test
# below (test_run_eval_verify_pin_follows_project_root_not_worktree) to abort
# run_eval right after build_eval_orch_config is called, without doing any real
# workflow/worktree work. (Task 2851's worktree-threading guards that formerly
# lived here were removed with the `worktree` param in task 2875.)
# ---------------------------------------------------------------------------

class _Sentinel(Exception):
    """Short-circuit sentinel raised by a capturing build stand-in mid-run_eval."""


# ---------------------------------------------------------------------------
# task 2875 — the eval verify UV_PYTHON pin follows the target project_root's
# CURRENT checkout, NOT the eval worktree checked out at an old pre_task_commit.
# run_eval threads the worktree it reuses/creates into build_eval_orch_config;
# post-2851 build sourced the pin from THAT worktree, which for older fixtures
# predates .python-version → no pin → uv default 3.14t → aiosqlite failure.
#
# A call-site capturing test (decision 3): wrap the REAL build to record
# verify_env then short-circuit via _Sentinel. It is RED on current code
# (worktree threaded → pin follows the pinless worktree → no UV_PYTHON) and
# stays invariant across step-4's removal of the `worktree` param — it never
# references `worktree=`, so **kwargs absorbs whatever the call site forwards.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_eval_verify_pin_follows_project_root_not_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from orchestrator.evals import runner

    # project_root's CURRENT checkout pins 3.13 (df fixtures do); the eval
    # worktree — checked out at an old pre_task_commit — carries NO
    # .python-version (the fixture divergence this task guards against).
    project_root = tmp_path / 'proj'
    project_root.mkdir()
    (project_root / '.python-version').write_text('3.13\n')
    wt = tmp_path / 'wt'
    wt.mkdir()

    # Capture the REAL build's verify_env, then short-circuit before any
    # workflow/worktree work. **kwargs swallows whatever the call site forwards
    # (memory_endpoint / judge_config / — today — worktree), so this test is
    # agnostic to step-4's removal of the worktree kwarg.
    real_build = runner.build_eval_orch_config
    captured: dict = {}

    def _capture_build(config, task, base_config=None, **kwargs):
        cfg = real_build(config, task, base_config, **kwargs)
        captured['verify_env'] = dict(cfg.verify_env)
        raise _Sentinel

    monkeypatch.setattr(runner, 'build_eval_orch_config', _capture_build)
    monkeypatch.setattr(
        runner, 'load_task',
        lambda _p: {'id': 't', 'project_root': str(project_root)},
    )

    # worktree_path=wt short-circuits create_eval_worktree and binds the eval
    # worktree; the build call is NOT inside a try, so _Sentinel propagates.
    with pytest.raises(_Sentinel):
        await runner.run_eval(
            tmp_path / 'task.json', _impl_cfg(),
            base_config=_base_config(project_root), worktree_path=wt,
        )

    # The verify pin follows project_root's 3.13, NOT the pinless worktree.
    assert captured['verify_env'].get('UV_PYTHON') == '3.13', (
        "eval verify UV_PYTHON must pin from project_root's current checkout "
        f"(3.13), got {captured['verify_env'].get('UV_PYTHON')!r}"
    )


# ---------------------------------------------------------------------------
# task 2875 amendment — the SAME project_root-sourced verify pin must hold for
# the OTHER both-live executor, run_end_to_end, not run_eval alone (reviewer
# test-coverage finding). run_end_to_end has NO worktree_path param — it always
# create_eval_worktree()s a fresh worktree at the fixture's old pre_task_commit —
# so this capturing test fakes that boundary and asserts the pin still follows
# project_root's CURRENT checkout. (run_architect_eval calls build_eval_orch_
# config identically but INSIDE a try/finally, so a _Sentinel short-circuit is
# swallowed there; its build path stays covered by the build-level
# TestBuildEvalOrchConfigVerifyPythonPin, which asserts the pin sources from
# task['project_root'].)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_end_to_end_verify_pin_follows_project_root_not_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from orchestrator.evals import runner

    # project_root's CURRENT checkout pins 3.13; the eval worktree
    # create_eval_worktree returns (checked out at an old pre_task_commit) carries
    # NO .python-version — the fixture divergence this task guards against.
    project_root = tmp_path / 'proj'
    project_root.mkdir()
    (project_root / '.python-version').write_text('3.13\n')
    wt = tmp_path / 'wt'
    wt.mkdir()

    real_build = runner.build_eval_orch_config
    captured: dict = {}

    def _capture_build(config, task, base_config=None, **kwargs):
        cfg = real_build(config, task, base_config, **kwargs)
        captured['verify_env'] = dict(cfg.verify_env)
        raise _Sentinel

    async def fake_create_wt(*_a, **_k):
        # run_end_to_end has no worktree_path param — it ALWAYS creates a fresh
        # worktree; return the pinless one without any real git work.
        return wt, 'run-e2e'

    monkeypatch.setattr(runner, 'create_eval_worktree', fake_create_wt)
    monkeypatch.setattr(runner, 'build_eval_orch_config', _capture_build)
    monkeypatch.setattr(
        runner, 'load_task',
        lambda _p: {
            'id': 't', 'project_root': str(project_root),
            'pre_task_commit': 'basecommit',
        },
    )

    # build_eval_orch_config is called OUTSIDE run_end_to_end's try block, so
    # _Sentinel propagates cleanly (mirrors the run_eval capturing test above).
    with pytest.raises(_Sentinel):
        await runner.run_end_to_end(
            tmp_path / 'task.json', _arch_cfg(), _impl_cfg(),
            base_config=_base_config(project_root),
        )

    # The verify pin follows project_root's 3.13, NOT the pinless worktree.
    assert captured['verify_env'].get('UV_PYTHON') == '3.13', (
        "eval end-to-end verify UV_PYTHON must pin from project_root's current "
        f"checkout (3.13), got {captured['verify_env'].get('UV_PYTHON')!r}"
    )


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
# step-03/04 — run_eval threads judge_config into build_eval_orch_config, and
# when supplied RELABELS the result to the judge candidate and stamps
# metrics['role_under_test']='judge' (mirrors run_end_to_end's end_to_end stamp).
#
# The implementer stays PINNED to `config` (only the judge varies); the label
# must key on the judge candidate (not the pinned implementer, which would
# collide both judge rows) so select_survivors groups judge runs as their own
# OFAT axis. Boundaries mocked (the _run_end_to_end_hermetic pattern):
# create_eval_worktree / build_workflow / collect_metrics / load_task /
# save_result. judge_config=None → byte-identical to today.
# ---------------------------------------------------------------------------

def _judge_task(tmp_path: Path) -> dict:
    # run_eval raises without a non-empty plan (the frozen plan it scores).
    return {
        'id': 'df_task_judge',
        'project_root': str(tmp_path),
        'pre_task_commit': 'basecommit',
        'task_definition': {'title': 'W', 'description': 'd'},
        'modules': ['pkg/mod.py'],
        'plan': {'steps': [{'id': 's1', 'status': 'pending'}]},
    }


async def _run_eval_hermetic(
    config: EvalConfig,
    base,
    task: dict,
    monkeypatch: pytest.MonkeyPatch,
    *,
    judge_config: EvalConfig | None = None,
    outcome: WorkflowOutcome = WorkflowOutcome.DONE,
    gate=None,
    injected_gate=_NO_INJECTED_GATE,
    run_side_effect: BaseException | None = None,
    collect_side_effect: BaseException | None = None,
):
    """Drive run_eval with the worktree/workflow/metrics boundaries mocked.

    Returns ``(result, captured)`` where ``captured['build_workflow']`` is the
    kwargs dict build_workflow received (for asserting the threaded config).

    ``_build_eval_usage_gate`` is patched UNCONDITIONALLY (task 4427) and
    exposed as ``captured['build_gate']``. It has to be: the packaged default is
    ``usage_cap.enabled: true``, so the real builder would otherwise construct a
    live ``UsageGate`` off ``_base_config``'s resolved cap block — touching the
    filesystem for probe dirs and account state, and leaving whether the gate
    resolves to ``None`` dependent on whatever credentials the test machine
    happens to export. ``gate=`` sets what the patched builder returns (default
    ``None``, i.e. the ungated cell every pre-4427 case here already meant).

    ``injected_gate=`` forwards to ``run_eval``'s own ``usage_gate=`` parameter —
    a gate a campaign owner already built, which this cell must use without
    tearing down. Omitted, no argument is passed at all (the owned path);
    ``injected_gate=None`` passes an explicit ``None``, which is NOT the same
    thing. ``run_side_effect`` / ``collect_side_effect`` make ``workflow.run()``
    / ``collect_metrics`` raise, the two ways a cell can fail after the gate is
    resolved — which is what proves the owned teardown really is in a
    ``finally``.
    """
    from orchestrator.evals import runner

    captured: dict = {}

    async def fake_create_wt(*_a, **_k):
        return Path('/fake/wt'), 'run-eval'

    fake_wf = MagicMock()
    fake_wf.run = AsyncMock(
        return_value=SimpleNamespace(outcome=outcome), side_effect=run_side_effect,
    )

    def fake_build_workflow(**kwargs):
        captured['build_workflow'] = kwargs
        return fake_wf

    metrics_obj = MagicMock()
    metrics_obj.to_dict.return_value = {'composite_score': 0.9, 'tests_pass': True}
    mock_collect = AsyncMock(return_value=metrics_obj, side_effect=collect_side_effect)
    mock_save = MagicMock()
    mock_build_gate = AsyncMock(return_value=gate)

    monkeypatch.setattr(runner, 'create_eval_worktree', fake_create_wt)
    monkeypatch.setattr(runner, 'build_workflow', fake_build_workflow)
    monkeypatch.setattr(runner, 'collect_metrics', mock_collect)
    monkeypatch.setattr(runner, 'load_task', lambda _p: task)
    monkeypatch.setattr(runner, 'save_result', mock_save)
    monkeypatch.setattr(runner, '_build_eval_usage_gate', mock_build_gate)
    captured['build_gate'] = mock_build_gate
    captured['wf'] = fake_wf

    extra: dict[str, Any] = (
        {} if injected_gate is _NO_INJECTED_GATE else {'usage_gate': injected_gate}
    )
    result = await runner.run_eval(
        Path('/fake/task.json'), config, base, judge_config=judge_config, **extra,
    )
    return result, captured


@pytest.mark.asyncio
class TestRunEvalJudgeConfig:
    async def test_judge_config_threads_into_config_and_relabels(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        base = _base_config(tmp_path)
        impl = EvalConfig('claude-sonnet-max', 'claude', 'sonnet', 'max')
        judge = EvalConfig('judge-haiku', 'claude', 'haiku', 'medium', role='judge')
        result, captured = await _run_eval_hermetic(
            impl, base, _judge_task(tmp_path), monkeypatch, judge_config=judge,
        )

        # (a) judge_config threaded into build_eval_orch_config → the judge model
        # derives from the candidate while the implementer stays PINNED to `config`.
        cfg = captured['build_workflow']['config']
        assert cfg.models.judge == 'haiku'          # the judge varies
        assert cfg.models.implementer == 'sonnet'    # implementer PINNED (NOT the judge model)

        # (b) result RELABELED to the judge candidate (so per-judge composite rows
        # don't collide on the pinned implementer name) and tagged role_under_test.
        assert result.config_name == 'judge-haiku'
        assert result.metrics['role_under_test'] == 'judge'

    async def test_no_judge_config_is_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        base = _base_config(tmp_path)
        impl = EvalConfig('claude-sonnet-max', 'claude', 'sonnet', 'max')
        result, captured = await _run_eval_hermetic(
            impl, base, _judge_task(tmp_path), monkeypatch,
        )

        # judge_config defaults None → label is the implementer config name, the
        # judge stays the sonnet incumbent, and no role_under_test='judge' stamp.
        assert result.config_name == 'claude-sonnet-max'
        assert captured['build_workflow']['config'].models.judge == 'sonnet'
        assert result.metrics.get('role_under_test') != 'judge'


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
    gate=None,
    injected_gate=_NO_INJECTED_GATE,
    run_side_effect: BaseException | None = None,
    collect_side_effect: BaseException | None = None,
):
    """Drive run_end_to_end with the worktree/workflow/metrics boundaries mocked.

    Returns ``(result, captured, mocks)`` where ``captured['build_workflow']``
    is the kwargs dict build_workflow received (for asserting config + plan)
    and ``mocks['build_gate']`` is the patched ``_build_eval_usage_gate``.

    The gate knobs (``gate`` / ``injected_gate`` / ``run_side_effect`` /
    ``collect_side_effect``) mean exactly what they mean in
    :func:`_run_eval_hermetic` — see its docstring for why the builder is
    patched unconditionally.
    """
    from orchestrator.evals import runner

    captured: dict = {}

    async def fake_create_wt(*_a, **_k):
        return Path('/fake/wt'), 'run-e2e'

    fake_wf = MagicMock()
    fake_wf.run = AsyncMock(
        return_value=SimpleNamespace(outcome=outcome), side_effect=run_side_effect,
    )

    def fake_build_workflow(**kwargs):
        captured['build_workflow'] = kwargs
        return fake_wf

    metrics_obj = MagicMock()
    metrics_obj.to_dict.return_value = {'composite_score': 0.9, 'tests_pass': True}
    mock_collect = AsyncMock(return_value=metrics_obj, side_effect=collect_side_effect)
    mock_save = MagicMock()
    mock_build_gate = AsyncMock(return_value=gate)

    monkeypatch.setattr(runner, 'create_eval_worktree', fake_create_wt)
    monkeypatch.setattr(runner, 'build_workflow', fake_build_workflow)
    monkeypatch.setattr(runner, 'collect_metrics', mock_collect)
    monkeypatch.setattr(runner, 'load_task', lambda _p: task)
    monkeypatch.setattr(runner, 'save_result', mock_save)
    monkeypatch.setattr(runner, '_build_eval_usage_gate', mock_build_gate)

    extra: dict[str, Any] = (
        {} if injected_gate is _NO_INJECTED_GATE else {'usage_gate': injected_gate}
    )
    result = await runner.run_end_to_end(
        Path('/fake/task.json'), arch_cfg, impl_cfg, base, **extra,
    )
    return result, captured, {
        'collect': mock_collect, 'save': mock_save, 'wf': fake_wf,
        'build_gate': mock_build_gate,
    }


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
# task 4427 — run_eval / run_end_to_end gain the same injected-gate contract
# run_architect_eval got: build-and-tear-down only when the caller supplied
# nothing; use-and-leave-alone when a campaign owner handed one down.
#
# These two paths had ZERO gate coverage before this task — nothing outside
# test_eval_architect.py ever exercised the seam — and they also never tore
# their OWN gate down (469a2b5bd0 closed that leak for run_architect_eval
# alone), so the owned-path teardown pins below are closing a pre-existing bug
# as well as pinning new behaviour.
# ---------------------------------------------------------------------------

# The two executors differ only in their hermetic driver's arity and in where
# it parks the patched builder, so the contract is pinned ONCE and parametrized
# over them rather than cloned class-for-class. Each adapter normalises to
# ``(result, build_workflow_kwargs, build_gate_mock)``.

async def _drive_run_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **gate_kw):
    result, captured = await _run_eval_hermetic(
        _impl_cfg(), _base_config(tmp_path), _judge_task(tmp_path), monkeypatch,
        **gate_kw,
    )
    return result, captured['build_workflow'], captured['build_gate']


async def _drive_run_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **gate_kw):
    result, captured, mocks = await _run_end_to_end_hermetic(
        _arch_cfg(), _impl_cfg(), _base_config(tmp_path), _e2e_task(tmp_path),
        monkeypatch, **gate_kw,
    )
    return result, captured['build_workflow'], mocks['build_gate']


_GATE_EXECUTORS = [
    pytest.param(_drive_run_eval, id='run_eval'),
    pytest.param(_drive_run_end_to_end, id='run_end_to_end'),
]


@pytest.mark.asyncio
@pytest.mark.parametrize('drive', _GATE_EXECUTORS)
class TestInjectedGateContract:
    async def test_injected_gate_reaches_the_workflow(
        self, drive, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from shared.testing import make_gate_mock

        gate = make_gate_mock()
        _result, wf_kwargs, build_gate = await drive(
            tmp_path, monkeypatch, injected_gate=gate,
        )

        assert wf_kwargs['usage_gate'] is gate
        # The hoist is pointless if the cell builds one anyway.
        build_gate.assert_not_awaited()

    async def test_injected_gate_is_never_torn_down(
        self, drive, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Shutting down a borrowed gate would take failover from every sibling."""
        from shared.testing import make_gate_mock

        gate = make_gate_mock()
        await drive(tmp_path, monkeypatch, injected_gate=gate)

        gate.shutdown.assert_not_awaited()

    async def test_injected_gate_survives_a_failing_workflow(
        self, drive, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from shared.testing import make_gate_mock

        gate = make_gate_mock()
        result, _wf_kwargs, _build_gate = await drive(
            tmp_path, monkeypatch, injected_gate=gate,
            run_side_effect=RuntimeError('workflow exploded'),
        )

        assert result.outcome == 'blocked'
        gate.shutdown.assert_not_awaited()

    async def test_explicit_none_means_ungated_not_unset(
        self, drive, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """A campaign that degraded to ungated must not have cells rebuild."""
        _result, wf_kwargs, build_gate = await drive(
            tmp_path, monkeypatch, injected_gate=None,
        )

        build_gate.assert_not_awaited()
        assert wf_kwargs['usage_gate'] is None

    async def test_owned_gate_is_built_and_torn_down(
        self, drive, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Closes a pre-existing leak: neither executor shut its own gate down."""
        from shared.testing import make_gate_mock

        gate = make_gate_mock()
        _result, wf_kwargs, build_gate = await drive(
            tmp_path, monkeypatch, gate=gate,
        )

        build_gate.assert_awaited_once()
        assert wf_kwargs['usage_gate'] is gate
        gate.shutdown.assert_awaited_once()

    @pytest.mark.parametrize(
        'failure', ['run', 'collect'],
        ids=['workflow_run_raises', 'collect_metrics_raises'],
    )
    async def test_owned_gate_teardown_is_in_a_finally(
        self, drive, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
    ):
        """Both post-gate failure shapes must still reach the teardown.

        A cell that failed is the one most likely to have hit a cap, i.e. the
        one holding a live account-resume probe loop — leaking there leaks
        exactly where it costs most.
        """
        from shared.testing import make_gate_mock

        gate = make_gate_mock()
        boom = RuntimeError('boom')
        kwargs: dict[str, Any] = (
            {'run_side_effect': boom} if failure == 'run'
            else {'collect_side_effect': boom}
        )
        await drive(tmp_path, monkeypatch, gate=gate, **kwargs)

        gate.shutdown.assert_awaited_once()

    async def test_a_failing_owned_shutdown_never_damages_the_cell(
        self, drive, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        """Best-effort teardown, mirroring run_architect_eval's finally."""
        import logging

        from shared.testing import make_gate_mock

        caplog.set_level(logging.WARNING, logger='orchestrator.evals.runner')
        gate = make_gate_mock()
        gate.shutdown = AsyncMock(side_effect=RuntimeError('teardown boom'))

        result, _wf_kwargs, _build_gate = await drive(
            tmp_path, monkeypatch, gate=gate,
        )

        assert result.outcome == 'done'
        assert 'shutdown failed' in caplog.text


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


# ---------------------------------------------------------------------------
# task 4427 — ONE gate per campaign, shared by every cell of a stage fan-out.
#
# A stage IS the campaign: it expands fixtures × candidates × trials and fans
# the cells out. Owning the gate there is what lets cell N+1 inherit cell N's
# cap knowledge instead of re-leasing an account already proved capped — the
# wall-clock win φ's failover was added for, which a per-cell gate forfeits.
#
# The probe, the sentinels and the assertion bodies live in
# _campaign_gate_helpers so test_runner_matrix and test_eval_architect pin the
# same contract from the same code.
# ---------------------------------------------------------------------------

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

    # --- task 4427: the OFAT stage owns ONE gate for the whole screen -------

    def _cells(self, tmp_path: Path):
        t1 = tmp_path / 'df_task_a.json'
        t2 = tmp_path / 'df_task_b.json'
        t1.touch()
        t2.touch()
        return [t1, t2], [
            EvalConfig('claude-opus-high', 'claude', 'opus', 'high'),
            EvalConfig('architect-sonnet-high', 'claude', 'sonnet', 'high',
                       role='architect'),
            EvalConfig('judge-haiku', 'claude', 'haiku', 'medium', role='judge'),
        ]

    def _stage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe, **kw):
        from orchestrator.evals import runner

        paths, candidates = self._cells(tmp_path)
        probe.install('run_eval', 'run_architect_eval', **kw)
        base = _base_config(tmp_path)

        async def stage():
            return await runner.run_ofat_stage(
                paths, candidates, base_config=base, trials=2,
            )
        return stage

    async def test_one_gate_serves_every_cell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from shared.testing import make_gate_mock

        probe = _CampaignGateProbe(monkeypatch, make_gate_mock())
        await _assert_one_gate_serves_every_cell(
            probe, self._stage(tmp_path, monkeypatch, probe),
        )
        # All three role branches (implementer / architect / judge-via-run_eval)
        # ride the same gate — the judge branch pins JUDGE_OFAT_IMPLEMENTER_PIN
        # as its config and would be easy to miss when threading.
        assert len(probe.seen) == 2 * 3 * 2  # fixtures × candidates × trials

    async def test_teardown_survives_a_failing_cell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from shared.testing import make_gate_mock

        probe = _CampaignGateProbe(monkeypatch, make_gate_mock())
        await _assert_teardown_survives_a_failing_cell(
            probe, self._stage(tmp_path, monkeypatch, probe, fail_on='df_task_a'),
        )

    async def test_teardown_survives_cancellation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from shared.testing import make_gate_mock

        probe = _CampaignGateProbe(monkeypatch, make_gate_mock())
        await _assert_teardown_survives_cancellation(
            probe, self._stage(tmp_path, monkeypatch, probe, cancel_on='df_task_a'),
        )

    async def test_a_degraded_campaign_stays_ungated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        probe = _CampaignGateProbe(monkeypatch, None)
        await _assert_a_degraded_campaign_stays_ungated(
            probe, self._stage(tmp_path, monkeypatch, probe),
        )


# ---------------------------------------------------------------------------
# step-07/08 (task 2825) — run_ofat_stage gains a role=='judge' branch.
#
# A judge candidate must NOT dispatch as an implementer (the pre-ο else branch
# would run it as config → the implementer co-varies, swamping the judge signal).
# The judge branch dispatches to run_eval with the implementer PINNED to
# JUDGE_OFAT_IMPLEMENTER_PIN and the judge candidate riding judge_config, so only
# the judge varies (true OFAT). Implementer and architect paths are untouched.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRunOfatStageJudge:
    async def test_judge_candidate_pins_implementer_and_rides_judge_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from orchestrator.evals import runner
        from orchestrator.evals.configs import JUDGE_OFAT_IMPLEMENTER_PIN

        t1 = tmp_path / 'df_task_a.json'
        t1.touch()
        # impl_cfg deliberately != the pin so the judge branch's pin is visible.
        impl_cfg = EvalConfig('claude-opus-high', 'claude', 'opus', 'high')
        judge_cfg = EvalConfig('judge-haiku', 'claude', 'haiku', 'medium', role='judge')

        # Records (fixture, config.name, judge_config.name|None, trial) per dispatch.
        eval_calls: list[tuple[str, str, str | None, int]] = []
        arch_calls: list[str] = []

        async def fake_run_eval(task_path, config, *_a, trial=1, judge_config=None, **_k):
            eval_calls.append(
                (task_path.stem, config.name,
                 judge_config.name if judge_config else None, trial),
            )
            label = judge_config.name if judge_config else config.name
            return EvalResult(task_path.stem, label, 'done', {}, '/tmp/wt', trial=trial)

        async def fake_run_arch(task_path, config, *_a, trial=1, **_k):
            arch_calls.append(config.name)
            return EvalResult(task_path.stem, config.name, 'done', {}, '/tmp/wt', trial=trial)

        monkeypatch.setattr(runner, 'load_task', _ofat_task_loader)
        monkeypatch.setattr(runner, 'run_eval', fake_run_eval)
        monkeypatch.setattr(runner, 'run_architect_eval', fake_run_arch)

        results = await runner.run_ofat_stage(
            [t1], [impl_cfg, judge_cfg], base_config=None, trials=1,
        )

        # (a) the JUDGE candidate dispatches to run_eval with the implementer PINNED
        # to JUDGE_OFAT_IMPLEMENTER_PIN and the judge riding judge_config — NOT
        # config=='judge-haiku' (which would co-vary the implementer).
        judge_dispatch = [c for c in eval_calls if c[2] == 'judge-haiku']
        assert len(judge_dispatch) == 1
        assert judge_dispatch[0][1] == JUDGE_OFAT_IMPLEMENTER_PIN.name  # implementer pinned
        assert judge_dispatch[0][1] != 'judge-haiku'

        # (b) the IMPLEMENTER candidate dispatches to run_eval with judge_config None
        # and config==the implementer candidate itself (unchanged else-branch).
        impl_dispatch = [c for c in eval_calls if c[2] is None]
        assert len(impl_dispatch) == 1
        assert impl_dispatch[0][1] == impl_cfg.name

        # (c) no architect candidate → run_architect_eval never called.
        assert arch_calls == []
        # (d) every (candidate, fixture, trial) cell returned a result.
        assert len(results) == 2

    async def test_mixed_roles_route_each_to_its_executor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from orchestrator.evals import runner
        from orchestrator.evals.configs import JUDGE_OFAT_IMPLEMENTER_PIN

        t1 = tmp_path / 'df_task_a.json'
        t1.touch()
        impl_cfg = EvalConfig('claude-opus-high', 'claude', 'opus', 'high')
        arch_cfg = EvalConfig('architect-sonnet-high', 'claude', 'sonnet', 'high', role='architect')
        judge_cfg = EvalConfig('judge-haiku', 'claude', 'haiku', 'medium', role='judge')

        eval_calls: list[tuple[str, str | None]] = []
        arch_calls: list[str] = []

        async def fake_run_eval(task_path, config, *_a, trial=1, judge_config=None, **_k):
            eval_calls.append((config.name, judge_config.name if judge_config else None))
            return EvalResult(task_path.stem, config.name, 'done', {}, '/tmp/wt', trial=trial)

        async def fake_run_arch(task_path, config, *_a, trial=1, **_k):
            arch_calls.append(config.name)
            return EvalResult(task_path.stem, config.name, 'done',
                              {'role_under_test': 'architect'}, '/tmp/wt', trial=trial)

        monkeypatch.setattr(runner, 'load_task', _ofat_task_loader)
        monkeypatch.setattr(runner, 'run_eval', fake_run_eval)
        monkeypatch.setattr(runner, 'run_architect_eval', fake_run_arch)

        results = await runner.run_ofat_stage(
            [t1], [impl_cfg, arch_cfg, judge_cfg], base_config=None, trials=1,
        )

        # Architect still routes to run_architect_eval UNCHANGED (the μ path).
        assert arch_calls == ['architect-sonnet-high']
        # Implementer (judge_config None) and judge (pinned impl + judge_config) both
        # go through run_eval — the judge NEVER reaches run_architect_eval.
        assert (impl_cfg.name, None) in eval_calls
        assert (JUDGE_OFAT_IMPLEMENTER_PIN.name, 'judge-haiku') in eval_calls
        assert len(results) == 3


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

    def _invoke(self, runner, paths, base):
        arch_survivors = [
            EvalConfig('arch-sonnet', 'claude', 'sonnet', 'high', role='architect'),
            EvalConfig('arch-opus', 'claude', 'opus', 'high', role='architect'),
        ]
        impl_survivors = [EvalConfig('impl-sonnet', 'claude', 'sonnet', 'high')]

        async def stage():
            return await runner.run_matrix_stage(
                paths, arch_survivors, impl_survivors, base_config=base, trials=2,
            )
        return stage

    # --- task 4427: the stage owns ONE gate for the whole fan-out ----------

    def _stage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe, **kw):
        from orchestrator.evals import runner

        t1 = tmp_path / 'df_task_a.json'
        t2 = tmp_path / 'df_task_b.json'
        t1.touch()
        t2.touch()
        probe.install('run_end_to_end', **kw)
        base = _base_config(tmp_path)
        return self._invoke(runner, [t1, t2], base)

    async def test_one_gate_serves_every_cell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from shared.testing import make_gate_mock

        probe = _CampaignGateProbe(monkeypatch, make_gate_mock())
        await _assert_one_gate_serves_every_cell(
            probe, self._stage(tmp_path, monkeypatch, probe),
        )

    async def test_teardown_survives_a_failing_cell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from shared.testing import make_gate_mock

        probe = _CampaignGateProbe(monkeypatch, make_gate_mock())
        await _assert_teardown_survives_a_failing_cell(
            probe, self._stage(tmp_path, monkeypatch, probe, fail_on='df_task_a'),
        )

    async def test_teardown_survives_cancellation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from shared.testing import make_gate_mock

        probe = _CampaignGateProbe(monkeypatch, make_gate_mock())
        await _assert_teardown_survives_cancellation(
            probe, self._stage(tmp_path, monkeypatch, probe, cancel_on='df_task_a'),
        )

    async def test_a_degraded_campaign_stays_ungated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        probe = _CampaignGateProbe(monkeypatch, None)
        await _assert_a_degraded_campaign_stays_ungated(
            probe, self._stage(tmp_path, monkeypatch, probe),
        )
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

    def _invoke(self, runner, paths, base):
        arch_winner = EvalConfig('arch-opus', 'claude', 'opus', 'high',
                                 role='architect')
        impl_winner = EvalConfig('impl-sonnet', 'claude', 'sonnet', 'high')

        async def stage():
            return await runner.run_confirm_stage(
                paths, arch_winner, impl_winner, base_config=base, trials=3,
            )
        return stage

    # --- task 4427: the stage owns ONE gate for the whole fan-out ----------

    def _stage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe, **kw):
        from orchestrator.evals import runner

        t1 = tmp_path / 'df_task_a.json'
        t2 = tmp_path / 'df_task_b.json'
        t1.touch()
        t2.touch()
        probe.install('run_end_to_end', **kw)
        base = _base_config(tmp_path)
        return self._invoke(runner, [t1, t2], base)

    async def test_one_gate_serves_every_cell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from shared.testing import make_gate_mock

        probe = _CampaignGateProbe(monkeypatch, make_gate_mock())
        await _assert_one_gate_serves_every_cell(
            probe, self._stage(tmp_path, monkeypatch, probe),
        )

    async def test_teardown_survives_a_failing_cell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from shared.testing import make_gate_mock

        probe = _CampaignGateProbe(monkeypatch, make_gate_mock())
        await _assert_teardown_survives_a_failing_cell(
            probe, self._stage(tmp_path, monkeypatch, probe, fail_on='df_task_a'),
        )

    async def test_teardown_survives_cancellation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from shared.testing import make_gate_mock

        probe = _CampaignGateProbe(monkeypatch, make_gate_mock())
        await _assert_teardown_survives_cancellation(
            probe, self._stage(tmp_path, monkeypatch, probe, cancel_on='df_task_a'),
        )

    async def test_a_degraded_campaign_stays_ungated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        probe = _CampaignGateProbe(monkeypatch, None)
        await _assert_a_degraded_campaign_stays_ungated(
            probe, self._stage(tmp_path, monkeypatch, probe),
        )
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


# ---------------------------------------------------------------------------
# Task 2820 (ν follow-up, escalation esc-2479-1 finding #3) — build_eval_orch_
# config auto-merges claude_endpoint_price_table() into config.prices whenever
# the candidate's env_overrides carries a proxied ANTHROPIC_BASE_URL, so an
# operator no longer has to seed prices manually for resolve_cost_usd to
# resolve cost_source='price_table' instead of 'unpriced_proxy'. Merge, not
# replace: an existing (manually-seeded) config.prices entry still wins.
# ---------------------------------------------------------------------------

class TestBuildEvalOrchConfigAutoSeedsProxiedPrices:
    def test_proxied_candidate_resolves_price_table_without_manual_seed(
        self, tmp_path: Path,
    ):
        # End-to-end via the driver: build_eval_orch_config (evals/runner.py)
        # feeds the REAL resolve_cost_usd (evals/metrics.py) with no manual
        # price seed on `base` — the auto-seed alone must be enough to land on
        # cost_source='price_table'.
        from orchestrator.evals.configs import MINIMAX_BASE_URL, MINIMAX_MODEL
        from orchestrator.evals.metrics import resolve_cost_usd
        from orchestrator.evals.runner import build_eval_orch_config

        base = _base_config(tmp_path)
        proxied = EvalConfig(
            'minimax-m2.5-endpoint', 'claude', MINIMAX_MODEL, 'high',
            env_overrides={
                'ANTHROPIC_BASE_URL': MINIMAX_BASE_URL,
                'ANTHROPIC_AUTH_TOKEN': 'dummy',
            },
        )

        cfg = build_eval_orch_config(proxied, {}, base)

        cost, source = resolve_cost_usd(
            1_000_000, 1_000_000,
            model=cfg.models.implementer,
            prices=cfg.prices,
            cli_cost_usd=999.0,           # deliberately wrong CLI figure —
            is_local_model=True,          # must be IGNORED once price_table hits
        )
        assert source == 'price_table'
        assert cost != 999.0

    def test_existing_config_prices_entry_wins_over_auto_seed(
        self, tmp_path: Path,
    ):
        from orchestrator.config import PriceEntry
        from orchestrator.evals.configs import MINIMAX_BASE_URL, MINIMAX_MODEL
        from orchestrator.evals.runner import build_eval_orch_config

        manual_rate = PriceEntry(input_per_1m=1234.0, output_per_1m=5678.0)
        base = _base_config(tmp_path).model_copy(
            update={'prices': {MINIMAX_MODEL: manual_rate}},
        )
        proxied = EvalConfig(
            'minimax-m2.5-endpoint', 'claude', MINIMAX_MODEL, 'high',
            env_overrides={
                'ANTHROPIC_BASE_URL': MINIMAX_BASE_URL,
                'ANTHROPIC_AUTH_TOKEN': 'dummy',
            },
        )

        cfg = build_eval_orch_config(proxied, {}, base)

        # The manually-seeded entry is preserved, NOT overwritten by the
        # auto-seeded claude_endpoint_price_table() figure.
        assert cfg.prices[MINIMAX_MODEL].input_per_1m == 1234.0
        assert cfg.prices[MINIMAX_MODEL].output_per_1m == 5678.0

    def test_non_proxied_candidate_prices_are_untouched(self, tmp_path: Path):
        from orchestrator.evals.runner import build_eval_orch_config

        base = _base_config(tmp_path)
        cfg = build_eval_orch_config(_impl_cfg(), {}, base)

        # No ANTHROPIC_BASE_URL override → no auto-seed; prices stay exactly
        # the base/profiled default price table.
        assert cfg.prices == base.prices
