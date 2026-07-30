"""CLI wiring for the μ OFAT→matrix→confirm driver (task 2478, step-19/20).

The G2 top-signal surface: the flat commands ``eval-ofat`` / ``eval-matrix`` /
``eval-confirm`` (the eval-sample convention) drive
``run_ofat_stage`` / ``run_matrix_stage`` / ``run_confirm_stage`` and emit the
C4 composite report; ``eval --matrix`` is retained as a backward-compatible
alias routing to the SAME matrix driver.

Every test here is hermetic: the async driver stages are monkeypatched to
return canned :class:`EvalResult`s (the ``test_runner_matrix`` idiom), so NO
worktree / LLM / network work runs. The commands exercised are the pure wiring
around λ's ``build_composite_report`` → ``format_composite_table``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

import orchestrator.cli as cli_module
import orchestrator.evals.runner as runner_module
from orchestrator.cli import main
from orchestrator.config import OrchestratorConfig
from orchestrator.evals import configs as configs_module
from orchestrator.evals.report import build_composite_report
from orchestrator.evals.runner import EvalResult


def _mk_result(
    config_name: str,
    task_id: str = 'df_task_10',
    *,
    role: str = 'implementer',
    composite: float = 0.8,
    cost: float = 1.5,
    dur_ms: int = 120_000,
    tests_pass: bool = True,
    plan_quality: float | None = None,
    cap_tainted: bool = False,
    invocation_error: str | None = None,
) -> EvalResult:
    """A canned PASSING EvalResult carrying the C4 metrics the report reads.

    ``plan_quality`` / ``cap_tainted`` / ``invocation_error`` mirror the
    PLAN-ONLY architect shape ``run_architect_eval`` writes (task 3099): a
    plan-only cell runs no test, so it carries ``tests_pass=None`` and its
    quality in ``plan_quality``.
    """
    return EvalResult(
        task_id=task_id,
        config_name=config_name,
        outcome='success',
        metrics={
            'composite_score': composite,
            'cost_usd': cost,
            'workflow_duration_ms': dur_ms,
            'tests_pass': tests_pass,
            'role_under_test': role,
            'judge_invocations': 2,
            'judge_cost_usd': 0.05,
            'cost_source': 'price_table',
            'plan_quality': plan_quality,
            'cap_tainted': cap_tainted,
            'invocation_error': invocation_error,
        },
        worktree_path='/tmp/eval-wt',
        wall_clock_ms=dur_ms,
    )


@pytest.fixture
def base_config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=tmp_path)


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


def test_eval_ofat_invokes_run_ofat_stage_and_prints_report(
    monkeypatch, base_config, dummy_cfg_file, fixtures_dir,
):
    """``eval-ofat`` drives run_ofat_stage over ofat_candidates() and prints C4."""
    monkeypatch.setattr(cli_module, 'load_config', lambda _: base_config)

    calls: list[tuple] = []

    async def fake_ofat(*args, **kwargs):
        calls.append((args, kwargs))
        return [_mk_result('claude-opus-high'), _mk_result('claude-sonnet-max')]

    monkeypatch.setattr(runner_module, 'run_ofat_stage', fake_ofat)

    r = CliRunner().invoke(main, [
        'eval-ofat', '--config', str(dummy_cfg_file),
        '--tasks-dir', str(fixtures_dir),
    ])

    assert r.exit_code == 0, r.output
    assert calls, 'run_ofat_stage was not invoked'
    # The driver is fed the one-role-each OFAT candidate set (>=2 impl incumbents).
    candidates = calls[0][0][1]
    assert [c.name for c in candidates] == [
        c.name for c in configs_module.ofat_candidates()
    ]
    assert 'composite report:' in r.output


def test_eval_matrix_prints_composite_report_two_config_floor(
    monkeypatch, base_config, dummy_cfg_file, fixtures_dir,
):
    """``eval-matrix`` drives the matrix stage; report carries >=2 config rows."""
    monkeypatch.setattr(cli_module, 'load_config', lambda _: base_config)

    calls: list[tuple] = []

    async def fake_matrix(*args, **kwargs):
        calls.append((args, kwargs))
        return [
            _mk_result('e2e-opus-opus', role='end_to_end', composite=0.85),
            _mk_result('e2e-sonnet-sonnet', role='end_to_end', composite=0.70),
        ]

    monkeypatch.setattr(runner_module, 'run_matrix_stage', fake_matrix)

    r = CliRunner().invoke(main, [
        'eval-matrix', '--config', str(dummy_cfg_file),
        '--tasks-dir', str(fixtures_dir),
    ])

    assert r.exit_code == 0, r.output
    assert calls, 'run_matrix_stage was not invoked'
    # The CLI splits ofat_candidates() by role into the arch × impl matrix input.
    arch_cfgs, impl_cfgs = calls[0][0][1], calls[0][0][2]
    assert arch_cfgs and all(c.role == 'architect' for c in arch_cfgs)
    assert impl_cfgs and all(c.role == 'implementer' for c in impl_cfgs)
    assert len(impl_cfgs) >= 2, 'the >=2-config floor (Opus + Sonnet incumbents)'

    # >=2 config rows + the rendered C4 columns.
    assert 'composite report:' in r.output
    assert 'e2e-opus-opus' in r.output
    assert 'e2e-sonnet-sonnet' in r.output
    for col in ('composite', 'quality', 'cost_usd', 'latency_secs', 'ci95_composite'):
        assert col in r.output, f'composite report missing column {col!r}'

    # The C4 report the CLI emits carries the per-config judge + CI95 signal
    # (build_composite_report's per-row 'judge'/'ci95' keys), tracking whichever
    # verdict/judge contract is live — format_composite_table renders
    # composite/cost/latency/CI95 as columns; judge rides in the report data.
    report = build_composite_report([_mk_result('a'), _mk_result('b')])
    assert len(report['configs']) >= 2
    assert all('judge' in row and 'ci95' in row for row in report['configs'])


def test_eval_confirm_invokes_run_confirm_stage(
    monkeypatch, base_config, dummy_cfg_file, fixtures_dir,
):
    """``eval-confirm`` drives run_confirm_stage for the single winning combo."""
    monkeypatch.setattr(cli_module, 'load_config', lambda _: base_config)

    cands = configs_module.ofat_candidates()
    arch_name = next(c.name for c in cands if c.role == 'architect')
    impl_name = next(c.name for c in cands if c.role == 'implementer')

    calls: list[tuple] = []

    async def fake_confirm(*args, **kwargs):
        calls.append((args, kwargs))
        return [_mk_result('e2e-winner', role='end_to_end') for _ in range(3)]

    monkeypatch.setattr(runner_module, 'run_confirm_stage', fake_confirm)

    r = CliRunner().invoke(main, [
        'eval-confirm', '--config', str(dummy_cfg_file),
        '--arch', arch_name, '--impl', impl_name,
        '--tasks-dir', str(fixtures_dir),
    ])

    assert r.exit_code == 0, r.output
    assert calls, 'run_confirm_stage was not invoked'
    arch_cfg, impl_cfg = calls[0][0][1], calls[0][0][2]
    assert arch_cfg.name == arch_name
    assert impl_cfg.name == impl_name
    assert 'composite report:' in r.output


def test_eval_confirm_rejects_swapped_arch_impl_flags(
    monkeypatch, base_config, dummy_cfg_file, fixtures_dir,
):
    """``eval-confirm`` fails LOUDLY when --arch/--impl resolve to the wrong role.

    Passing an implementer name to --arch and an architect name to --impl
    resolves fine via get_config_by_name but would silently build a nonsensical
    both-live combo. The driver must reject it, NAME the offending flag, and
    never dispatch the confirmation batch (loud-over-silent).
    """
    monkeypatch.setattr(cli_module, 'load_config', lambda _: base_config)

    cands = configs_module.ofat_candidates()
    arch_name = next(c.name for c in cands if c.role == 'architect')
    impl_name = next(c.name for c in cands if c.role == 'implementer')

    called = False

    async def fake_confirm(*args, **kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(runner_module, 'run_confirm_stage', fake_confirm)

    # Flags swapped: implementer name to --arch, architect name to --impl.
    r = CliRunner().invoke(main, [
        'eval-confirm', '--config', str(dummy_cfg_file),
        '--arch', impl_name, '--impl', arch_name,
        '--tasks-dir', str(fixtures_dir),
    ])

    assert r.exit_code != 0, r.output
    assert '--arch' in r.output, 'the error must name the offending flag'
    assert not called, 'run_confirm_stage must not run on a role mismatch'


def test_eval_matrix_flag_alias_routes_to_matrix_driver(
    monkeypatch, base_config, dummy_cfg_file,
):
    """``eval --matrix`` (legacy flag) routes to the SAME μ matrix driver.

    Pre-μ the flag ran ``run_eval_matrix`` and printed a bespoke summary; the
    alias must now route to ``run_matrix_stage`` and emit the composite report.
    ``run_eval_matrix`` is neutralized so the pre-impl path never does real work.
    """
    monkeypatch.setattr(cli_module, 'load_config', lambda _: base_config)

    async def _neutralized_old_path(*args, **kwargs):
        return []

    monkeypatch.setattr(runner_module, 'run_eval_matrix', _neutralized_old_path)

    calls: list[tuple] = []

    async def fake_matrix(*args, **kwargs):
        calls.append((args, kwargs))
        return [
            _mk_result('e2e-a', role='end_to_end'),
            _mk_result('e2e-b', role='end_to_end'),
        ]

    monkeypatch.setattr(runner_module, 'run_matrix_stage', fake_matrix)

    r = CliRunner().invoke(main, [
        'eval', '--matrix', '--config', str(dummy_cfg_file),
    ])

    assert r.exit_code == 0, r.output
    assert calls, '`eval --matrix` did not route to run_matrix_stage'
    assert 'composite report:' in r.output


def test_eval_help_still_lists_judge_trials_vllm_options():
    """``eval --help`` must keep --judge/--trials/--vllm-url (test_runner_main guard)."""
    r = CliRunner().invoke(main, ['eval', '--help'])
    assert r.exit_code == 0, r.output
    for opt in ('--judge', '--trials', '--vllm-url'):
        assert opt in r.output, f'{opt!r} missing from `eval --help`'


# ---------------------------------------------------------------------------
# task 3099 — _emit_composite_report's architect surface.
#
# _emit_composite_report is the ONE shared surface behind eval-ofat /
# eval-matrix / eval-confirm. The composite row can only report an exclusion
# COUNT; only the plan-quality table breaks it out BY CAUSE, which is what tells
# an operator whether a missing architect cell is a transient cap window (rerun
# it) or a permanent model-not-found (that candidate can never run).
# ---------------------------------------------------------------------------

def _arch_ofat_results():
    """The OFAT shape ``ofat_candidates()`` actually produces: architect
    plan-only cells alongside full-workflow implementer cells."""
    return [
        _mk_result('arch-opus-high', role='architect', plan_quality=0.9,
                   tests_pass=None, cost=0.3, dur_ms=60_000),
        _mk_result('arch-fable-high', role='architect', plan_quality=None,
                   tests_pass=None, cost=0.0, dur_ms=0, cap_tainted=True,
                   invocation_error='architect:model_not_found: no such model'),
        _mk_result('impl-opus-high', role='implementer'),
    ]


class TestEmitCompositeReportArchitectSurface:
    def test_architect_result_set_emits_both_tables(self, capsys):
        cli_module._emit_composite_report(_arch_ofat_results(), {})

        out = capsys.readouterr().out
        assert 'composite report:' in out
        # The per-cell θ scores AND the cap-exclusion CAUSES the composite row
        # can only report as a count.
        assert 'plan_quality report:' in out
        assert 'model_not_found' in out

    def test_architect_surface_reuses_the_plan_quality_table_verbatim(self, capsys):
        """No second rendering path: the emitted section IS
        format_plan_quality_table(build_plan_quality_report(results))."""
        from orchestrator.evals.report import (
            build_plan_quality_report,
            format_plan_quality_table,
        )

        results = _arch_ofat_results()
        cli_module._emit_composite_report(results, {})

        out = capsys.readouterr().out
        assert format_plan_quality_table(build_plan_quality_report(results)) in out

    def test_non_architect_result_set_emits_exactly_todays_output(self, capsys):
        """The existing eval-matrix / eval-confirm end_to_end surfaces are
        unchanged — no extra section when no architect ran."""
        from orchestrator.evals.report import format_composite_table

        results = [
            _mk_result('e2e-opus-opus', role='end_to_end', composite=0.85),
            _mk_result('e2e-sonnet-sonnet', role='end_to_end', composite=0.70),
        ]
        cli_module._emit_composite_report(results, {})

        out = capsys.readouterr().out
        assert 'plan_quality report:' not in out
        expected = format_composite_table(
            build_composite_report(results, price_table={})
        )
        assert out.strip() == expected.strip()

    def test_ofat_run_with_architect_candidates_surfaces_both_tables(
        self, monkeypatch, base_config, dummy_cfg_file, fixtures_dir,
    ):
        """End-to-end through the `eval-ofat` command, not just the helper."""
        monkeypatch.setattr(cli_module, 'load_config', lambda _: base_config)

        async def fake_ofat(*args, **kwargs):
            return _arch_ofat_results()

        monkeypatch.setattr(runner_module, 'run_ofat_stage', fake_ofat)

        r = CliRunner().invoke(main, [
            'eval-ofat', '--config', str(dummy_cfg_file),
            '--tasks-dir', str(fixtures_dir),
        ])

        assert r.exit_code == 0, r.output
        assert 'composite report:' in r.output
        assert 'plan_quality report:' in r.output
