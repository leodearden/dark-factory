"""The composite report reads a plan-only cell's cost as its candidate's spend (task 3848).

Every cost figure on a composite row (both floors, the cost ratio, ``cost_usd`` /
``ci95.cost`` and the ``cost_per_plan`` numerator) is
``cell_admission.candidate_cost_usd``: for a plan-only cell, ``cost_usd`` net of
``judge_cost_usd``. That makes cells written before task 3629 (judge spend 0.0 beside
an architect-only ``cost_usd``) compare consistently with later cells, whose
``cost_usd`` folds in the fixed plan-judge pin.
"""

from __future__ import annotations

import pytest

from orchestrator.evals.composite_report import build_composite_report, select_survivors
from orchestrator.evals.measurement import MeasurementStage, Unmeasured, UnmeasuredCause
from orchestrator.evals.metrics import EvalMetrics, RoleUnderTest, blend_composite
from orchestrator.evals.results import EvalResult


def _cell(
    task_id, config, trial, *, role='architect', cost_usd, judge_cost_usd=0.0,
    quality: float | None = 0.8, plan_steps=6, duration_ms=1000, tests_pass=None,
    cap_tainted=False, invocation_errors=None,
):
    """An EvalResult with production-shaped metrics.

    *quality* is the θ ``plan_quality`` of an architect cell and the
    ``composite_score`` of a workflow cell.
    """
    plan_only = role == 'architect'
    metrics = EvalMetrics(
        tests_pass=tests_pass,
        cost_usd=cost_usd,
        workflow_duration_ms=duration_ms,
        composite_score=0.0 if plan_only or quality is None else quality,
        plan_quality=quality if plan_only else None,
        role_under_test=RoleUnderTest(role),
        plan_steps=plan_steps,
        judge_cost_usd=judge_cost_usd,
        cap_tainted=cap_tainted,
        invocation_errors=invocation_errors,
    )
    return EvalResult(
        task_id=task_id,
        config_name=config,
        outcome='completed',
        metrics=metrics.to_dict(),
        worktree_path='/tmp/eval',
        trial=trial,
    )


def _rows(results):
    return {row['config']: row for row in build_composite_report(results)['configs']}


def test_a_pre_3629_cell_and_a_judge_inclusive_cell_of_equal_architect_spend_tie():
    results = [
        *(_cell('f1', 'OLD', t, cost_usd=1.0) for t in range(3)),
        *(_cell('f1', 'NEW', t, cost_usd=1.4, judge_cost_usd=0.4) for t in range(3)),
    ]
    rows = _rows(results)
    assert rows['OLD']['composite'] == pytest.approx(rows['NEW']['composite'])
    assert rows['OLD']['cost_usd'] == pytest.approx(1.0)
    assert rows['NEW']['cost_usd'] == pytest.approx(1.0)


def test_the_judge_pin_does_not_compress_the_cost_spread():
    quality = 0.8
    rows = _rows([
        _cell('f1', 'CHEAP', 0, cost_usd=1.0, judge_cost_usd=0.5, quality=quality),
        _cell('f1', 'PRICY', 0, cost_usd=1.5, judge_cost_usd=0.5, quality=quality),
    ])
    assert rows['PRICY']['composite'] == pytest.approx(
        blend_composite(quality, 0.5, 1.0, tests_pass=None, plan_only=True),
    )
    assert rows['CHEAP']['composite'] == pytest.approx(
        blend_composite(quality, 1.0, 1.0, tests_pass=None, plan_only=True),
    )


def test_the_genuinely_cheaper_candidate_survives():
    report = build_composite_report([
        _cell('f1', 'A', 0, cost_usd=1.0),
        _cell('f1', 'B', 0, cost_usd=1.3, judge_cost_usd=0.4),
    ])
    assert select_survivors(report, top_k=1, roles=['architect']) == {'architect': ['B']}


def test_cost_per_plan_sums_candidate_spend_including_the_no_plan_cell():
    rows = _rows([
        _cell('f1', 'CFG', 0, cost_usd=1.4, judge_cost_usd=0.4),
        _cell('f1', 'CFG', 1, cost_usd=0.5, quality=0.0, plan_steps=0),
    ])
    assert rows['CFG']['cost_per_plan'] == pytest.approx(1.5)


def test_the_judge_spend_stays_visible_on_the_row():
    row = _rows([_cell('f1', 'CFG', 0, cost_usd=1.4, judge_cost_usd=0.4)])['CFG']
    assert row['judge']['cost_usd'] == pytest.approx(0.4)
    assert row['cost_usd'] + row['judge']['cost_usd'] == pytest.approx(1.4)


def test_a_workflow_cell_keeps_its_completion_judge_spend():
    row = _rows([
        _cell(
            'f1', 'IMPL', 0, role='implementer', cost_usd=2.0, judge_cost_usd=0.5,
            tests_pass=True,
        ),
    ])['IMPL']
    assert row['cost_usd'] == pytest.approx(2.0)


def test_a_judge_spend_above_the_total_raises_naming_the_cell():
    with pytest.raises(ValueError, match='f1 x BAD'):
        build_composite_report([_cell('f1', 'BAD', 0, cost_usd=0.3, judge_cost_usd=0.5)])


def test_an_unmeasurable_cells_cost_is_never_read():
    refused = Unmeasured(
        MeasurementStage.ARCHITECT, UnmeasuredCause.CAP_HIT, 'session limit',
        cost_usd=0.0,
    ).to_record()
    report = build_composite_report([
        _cell(
            'f1', 'CFG', 0, cost_usd=0.0, judge_cost_usd=0.05, quality=None,
            cap_tainted=True, invocation_errors=[refused],
        ),
        _cell('f1', 'CFG', 1, cost_usd=1.0),
    ])
    assert report['configs'][0]['trials'] == 2
