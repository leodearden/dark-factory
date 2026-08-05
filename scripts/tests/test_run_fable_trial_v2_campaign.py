"""Tests for scripts/run_fable_trial_v2_campaign.py — the fable-trial-v2 β2 driver.

Placed in ``scripts/tests/`` rather than ``orchestrator/tests/`` DELIBERATELY.
``scripts/orchestrator.yaml``'s ``test_command`` is ``uv run --project shared
pytest tests/scripts/ scripts/tests/``, and ``verify_plan._derive_module_runs``'s
arm-3 source-only floor runs that command VERBATIM for any ``scripts/**``
production diff — so a test module for this driver placed under
``orchestrator/tests/`` would not gate this diff at all (the vacuous-green class
task 3460 closed). The bare ``import run_fable_trial_v2_campaign`` resolves off
``scripts/tests/conftest.py``'s sys.path insertion, and the first-party
``orchestrator.evals`` imports under ``--project shared`` are precedented by
``test_reviewer_redundancy_diagnostic.py``.

Candidate names used here are ONLY ``architect-opus-max`` and
``architect-fable-high`` — both present in ``ARCHITECT_EVAL_CONFIGS`` today.
``architect-fable-max`` is eval-revival ρ's (task 3627) unlanded deliverable and
must not appear in any test or default, so this suite stays green against the
instrument as it exists rather than as it will exist.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_fable_trial_v2_campaign as mod


def _cell(
    task_id, config_name, *, trial=1, plan_quality=None, plan_steps=0,
    role_under_test='architect', cap_tainted=False, invocation_error=None,
    cost_usd=0.0, judge_cost_usd=0.0, extra_metrics=None,
):
    """Build a synthetic ``EvalResult`` with a production-shaped metrics dict.

    Copies ``orchestrator/tests/test_eval_composite_report.py``'s helper shape:
    ``plan_steps`` is threaded EXPLICITLY because it is the plan-production
    predicate the report layer reads — a cell declaring a ``plan_quality``
    without the step count it came from is a self-contradictory fixture.

    ``extra_metrics`` merges keys the current ``EvalMetrics`` has no field for.
    That is exactly how ``judged_without_reference`` will arrive once
    eval-revival σ (task 3628) adds the field, so tests can exercise the
    post-σ world without the driver forward-referencing an unlanded instrument.
    """
    from orchestrator.evals.metrics import EvalMetrics
    from orchestrator.evals.runner import EvalResult

    m = EvalMetrics(
        plan_quality=plan_quality,
        role_under_test=role_under_test,
        plan_steps=plan_steps,
        cost_usd=cost_usd,
        judge_cost_usd=judge_cost_usd,
        cap_tainted=cap_tainted,
        invocation_error=invocation_error,
    )
    metrics = {**m.to_dict(), **(extra_metrics or {})}
    return EvalResult(
        task_id=task_id,
        config_name=config_name,
        outcome='completed',
        metrics=metrics,
        worktree_path='/tmp/eval',
        trial=trial,
    )


def _mixed_results():
    """Candidate A with one cap-tainted cell, candidate B with one no-plan cell."""
    return [
        # A: one transport refusal (never got to ask the model) + two scored cells.
        _cell('f1', 'architect-opus-max', cap_tainted=True,
              invocation_error='architect: cap_hit'),
        _cell('f2', 'architect-opus-max', plan_quality=0.80, plan_steps=6),
        _cell('f3', 'architect-opus-max', plan_quality=0.60, plan_steps=4),
        # B: one genuine no-plan measurement + two scored cells.
        _cell('f1', 'architect-fable-high', plan_quality=0.0, plan_steps=0),
        _cell('f2', 'architect-fable-high', plan_quality=0.70, plan_steps=5),
        _cell('f3', 'architect-fable-high', plan_quality=0.50, plan_steps=3),
    ]


def _make_fixture_dir(tmp_path, stems, *, name='tasks_hard_v2'):
    """Build a fixture dir holding one minimal ``*.json`` per stem.

    Also drops a non-``.json`` sibling, so a glob that stopped filtering by
    suffix would be caught rather than silently widening the pool.
    """
    d = tmp_path / name
    d.mkdir(parents=True)
    for stem in stems:
        (d / f'{stem}.json').write_text(json.dumps({'task_id': stem}) + '\n')
    (d / 'README.md').write_text('not a fixture\n')
    return d


# ===== step 1: cell enumeration, the dry-run matrix, and loud fixture resolution =====


def test_enumerate_cells_is_the_candidate_x_fixture_x_trial_product(tmp_path):
    """The matrix is the full (fixture × candidate × trial) product, ordered."""
    d = _make_fixture_dir(tmp_path, ['alpha', 'beta', 'gamma'])
    paths = sorted(d.glob('*.json'))
    candidates = ['architect-opus-max', 'architect-fable-high']

    cells = mod.enumerate_cells(paths, candidates, 3)

    assert len(cells) == 3 * 2 * 3
    # Fixture-then-candidate-then-trial ordering, deterministic.
    assert cells[0] == {
        'task_id': 'alpha', 'config_name': 'architect-opus-max', 'trial': 1,
    }
    assert cells[3] == {
        'task_id': 'alpha', 'config_name': 'architect-fable-high', 'trial': 1,
    }
    assert cells[6] == {
        'task_id': 'beta', 'config_name': 'architect-opus-max', 'trial': 1,
    }
    assert [c['trial'] for c in cells[:3]] == [1, 2, 3]
    assert {c['task_id'] for c in cells} == {'alpha', 'beta', 'gamma'}
    assert {c['config_name'] for c in cells} == set(candidates)
    # Same inputs -> byte-identical matrix (a committed γ1 artifact must diff cleanly).
    assert mod.enumerate_cells(paths, candidates, 3) == cells


def test_dry_run_prints_the_matrix_and_spends_nothing(tmp_path, capsys, monkeypatch):
    """``--dry-run`` enumerates the matrix and never reaches the live-spend seam."""
    d = _make_fixture_dir(tmp_path, ['alpha', 'beta', 'gamma'])

    def _boom(*args, **kwargs):
        pytest.fail(
            '--dry-run reached _run_campaign, the live real-API-spend seam. '
            'Dry-run exists precisely so the cell matrix and the comparison '
            'regime can be audited at zero cost before a several-hundred-dollar '
            'campaign is launched.',
            pytrace=False,
        )

    monkeypatch.setattr(mod, '_run_campaign', _boom)

    rc = mod.main([
        '--dry-run', '--tasks-dir', str(d),
        '--candidate', 'architect-opus-max',
        '--candidate', 'architect-fable-high',
        '--trials', '3',
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert '18' in out, 'total cell count (3 fixtures x 2 candidates x 3 trials) not reported'
    for name in ('architect-opus-max', 'architect-fable-high'):
        assert name in out
    for stem in ('alpha', 'beta', 'gamma'):
        assert stem in out
    assert 'README' not in out, 'a non-.json sibling leaked into the fixture pool'


def test_absent_tasks_dir_exits_loudly_naming_the_path(tmp_path):
    """An absent ``--tasks-dir`` exits non-zero naming the path — NO corpus fallback.

    ``tasks_hard_v2/`` is β1's (task 3631) unlanded deliverable, so absent is the
    common early case. A convenience fallback to the standing ``evals/tasks``
    corpus would silently run the campaign over the incumbent-success-biased pool
    the v2 screen exists to escape, at real spend, and produce a plausible-looking
    report answering a different question.
    """
    missing = tmp_path / 'does_not_exist'

    with pytest.raises(SystemExit) as exc:
        mod.main([
            '--dry-run', '--tasks-dir', str(missing),
            '--candidate', 'architect-opus-max',
        ])

    assert exc.value.code != 0
    assert str(missing) in str(exc.value)


def test_empty_tasks_dir_exits_loudly(tmp_path):
    """An existing but fixture-free dir likewise exits non-zero naming the dir."""
    empty = tmp_path / 'empty_pool'
    empty.mkdir()
    (empty / 'README.md').write_text('no fixtures here\n')

    with pytest.raises(SystemExit) as exc:
        mod.main([
            '--dry-run', '--tasks-dir', str(empty),
            '--candidate', 'architect-opus-max',
        ])

    assert exc.value.code != 0
    assert str(empty) in str(exc.value)


# ===== step 3: candidate resolution and per-candidate budget overrides =====


def test_resolve_candidates_returns_architect_configs():
    """A named architect candidate resolves to its ``EvalConfig``."""
    cands = mod.resolve_candidates(['architect-opus-max'])

    assert len(cands) == 1
    assert cands[0].name == 'architect-opus-max'
    assert cands[0].role == 'architect'


def test_unknown_candidate_exits_naming_it():
    """An unresolvable candidate name exits non-zero, naming the offender."""
    with pytest.raises(SystemExit) as exc:
        mod.resolve_candidates(['architect-nope'])

    assert exc.value.code != 0
    assert 'architect-nope' in str(exc.value)


def test_non_architect_candidate_is_rejected():
    """A RESOLVABLE implementer config is still rejected — structurally, not by comment.

    ``claude-opus-max`` resolves fine through ``get_config_by_name`` but carries
    ``role == 'implementer'``, and ``run_ofat_stage`` dispatches BY ROLE: it would
    route this candidate through ``run_eval``, a FULL agentic workflow cell at
    roughly 10x the cost of a plan-only architect cell. That is exactly the spend
    the PRD forbids when it says "Do NOT use eval-ofat". Rejecting at resolution
    time turns a one-character typo from hundreds of dollars of the wrong spend
    into an immediate, named error.
    """
    from orchestrator.evals.configs import get_config_by_name

    resolvable = get_config_by_name('claude-opus-max')
    assert resolvable is not None and resolvable.role == 'implementer', (
        'premise of this test: the name must RESOLVE and be non-architect'
    )

    with pytest.raises(SystemExit) as exc:
        mod.resolve_candidates(['claude-opus-max'])

    assert exc.value.code != 0
    assert 'claude-opus-max' in str(exc.value)
    assert 'architect' in str(exc.value)


def test_parse_budget_spec():
    """``NAME=AMOUNT`` parses; a malformed spec exits naming itself."""
    assert mod._parse_budget_spec('architect-opus-max=15') == ('architect-opus-max', 15.0)
    assert mod._parse_budget_spec('architect-fable-high=7.5') == ('architect-fable-high', 7.5)

    for bad in ('architect-opus-max', 'architect-opus-max=lots', '=15'):
        with pytest.raises(SystemExit) as exc:
            mod._parse_budget_spec(bad)
        assert bad in str(exc.value)


def test_apply_budgets_overrides_only_the_named_candidate():
    """Only the named candidate's budget moves; the inputs are never mutated."""
    cands = mod.resolve_candidates(['architect-opus-max', 'architect-fable-high'])
    before = [(c.name, c.max_budget_usd) for c in cands]

    out = mod.apply_budgets(cands, {'architect-opus-max': 15.0})

    by_name = {c.name: c for c in out}
    assert by_name['architect-opus-max'].max_budget_usd == 15.0
    assert by_name['architect-fable-high'].max_budget_usd == 20.0, (
        'an unnamed candidate must keep its declared default'
    )
    # dataclasses.replace, not in-place mutation: the module-level
    # ARCHITECT_EVAL_CONFIGS entries stay byte-unchanged for every other caller.
    assert [(c.name, c.max_budget_usd) for c in cands] == before
    assert all(a is not b for a, b in zip(cands, out) if a.name == 'architect-opus-max')


def test_budget_for_unselected_candidate_exits():
    """A ``--budget`` naming a candidate that is not being run exits non-zero.

    A silently-ignored budget would run a comparison arm at the WRONG price,
    which invalidates the γ2 regime ruling the whole trial turns on.
    """
    cands = mod.resolve_candidates(['architect-opus-max'])

    with pytest.raises(SystemExit) as exc:
        mod.apply_budgets(cands, {'architect-fable-high': 15.0})

    assert exc.value.code != 0
    assert 'architect-fable-high' in str(exc.value)


def test_dry_run_prints_the_comparison_regime(tmp_path, capsys, monkeypatch):
    """Dry-run shows model / effort / budget per candidate — auditable BEFORE spend."""
    d = _make_fixture_dir(tmp_path, ['alpha'])

    def _boom(*args, **kwargs):
        pytest.fail('--dry-run reached _run_campaign', pytrace=False)

    monkeypatch.setattr(mod, '_run_campaign', _boom)

    rc = mod.main([
        '--dry-run', '--tasks-dir', str(d),
        '--candidate', 'architect-opus-max',
        '--candidate', 'architect-fable-high',
        '--budget', 'architect-opus-max=15',
        '--budget', 'architect-fable-high=15',
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert 'claude-fable-5' in out, 'candidate model not shown'
    assert 'opus' in out
    assert '15' in out, 'the overridden budget — the γ2 comparison regime — not shown'


# ===== step 5: the per-candidate summary SURFACES the report layer =====


def test_summary_rows_match_the_report_layer_field_for_field():
    """THE anti-drift assertion: every count comes from the report layer verbatim.

    ``cap_excluded`` / ``no_plan`` / ``plan_rate`` / ``mean_plan_quality`` are
    the report layer's per-config accumulator (tasks 3118/3302/3379), built on
    the SHARED ``_plan_rate`` and ``_mean_plan_quality`` reductions precisely so
    two adjacent surfaces cannot give contradictory answers about the same
    quantity. Re-deriving any of them in this driver would create exactly that
    second surface. This test pins the equality field for field.
    """
    from orchestrator.evals.report import build_plan_quality_report

    results = _mixed_results()
    rows = mod.summarize_candidates(results)
    reference = {c['config_name']: c for c in build_plan_quality_report(results)['configs']}

    assert {r['config_name'] for r in rows} == set(reference)
    for row in rows:
        ref = reference[row['config_name']]
        for field in ('n', 'total', 'cap_excluded', 'no_plan', 'plan_rate',
                      'mean_plan_quality'):
            assert row[field] == ref[field], f'{row["config_name"]}.{field} drifted'


def test_cap_excluded_is_per_candidate_not_a_report_total():
    """Cap exclusion is reported PER CANDIDATE — the differential-exclusion signal.

    ``build_plan_quality_report`` also emits a report-level ``cap_excluded``
    total; summing every arm into one number would hide the very asymmetry the
    reopening condition watches for, since the costlier candidate is the more
    cap-exposed one.
    """
    rows = {r['config_name']: r for r in mod.summarize_candidates(_mixed_results())}

    assert rows['architect-opus-max']['cap_excluded'] == 1
    assert rows['architect-fable-high']['cap_excluded'] == 0
    # And the counterpart signal stays disjoint from it.
    assert rows['architect-opus-max']['no_plan'] == 0
    assert rows['architect-fable-high']['no_plan'] == 1


def test_no_scored_cells_reports_none_not_zero():
    """A candidate whose every cell was cap-tainted reports n=0, mean=None."""
    results = [
        _cell('f1', 'architect-opus-max', cap_tainted=True,
              invocation_error='architect: cap_hit'),
        _cell('f2', 'architect-opus-max', cap_tainted=True,
              invocation_error='architect: cap_hit'),
    ]

    row = mod.summarize_candidates(results)[0]

    assert row['n'] == 0
    assert row['total'] == 2
    assert row['cap_excluded'] == 2
    assert row['mean_plan_quality'] is None, (
        '"we measured nothing" must never read as "it scored nothing"'
    )


def test_rows_are_sorted_by_config_name():
    """Deterministic ordering, so a committed report artifact diffs cleanly."""
    rows = mod.summarize_candidates(_mixed_results())

    assert [r['config_name'] for r in rows] == sorted(r['config_name'] for r in rows)


# ===== step 7: judged_without_reference — UNMEASURED is not zero =====


def test_marker_absent_everywhere_reports_none_not_zero():
    """With today's instrument, the count is ``None`` for every candidate.

    Asserting ``is None`` and NOT ``== 0`` is the whole point. The marker's
    producer is eval-revival σ (task 3628), which is unlanded, so no cell can
    carry the key yet. ``EvalMetrics.to_dict()`` is an ``asdict`` and therefore
    emits every DECLARED field, which makes "key absent on every cell" mean
    unambiguously "this instrument predates σ" rather than "no offending cell".
    Reporting ``0`` would let ``plan_quality`` read as fully validity-bounded
    when nothing bounded it — the silent degradation the PRD exists to end.
    """
    results = _mixed_results()

    counts = mod.count_judged_without_reference(results)

    assert set(counts) == {'architect-opus-max', 'architect-fable-high'}
    assert all(v is None for v in counts.values())
    assert all(
        row['judged_without_reference'] is None
        for row in mod.summarize_candidates(results)
    )


def test_marker_present_counts_per_candidate():
    """Once σ lands, the key simply starts appearing and the Nones become counts."""
    results = [
        _cell('f1', 'A', plan_quality=0.8, plan_steps=5,
              extra_metrics={'judged_without_reference': True}),
        _cell('f2', 'A', plan_quality=0.8, plan_steps=5,
              extra_metrics={'judged_without_reference': True}),
        _cell('f3', 'A', plan_quality=0.8, plan_steps=5,
              extra_metrics={'judged_without_reference': False}),
        _cell('f4', 'A', plan_quality=0.8, plan_steps=5,
              extra_metrics={'judged_without_reference': False}),
        _cell('f1', 'B', plan_quality=0.7, plan_steps=4,
              extra_metrics={'judged_without_reference': False}),
        _cell('f2', 'B', plan_quality=0.7, plan_steps=4,
              extra_metrics={'judged_without_reference': False}),
    ]

    counts = mod.count_judged_without_reference(results)

    # B's 0 is a REAL measurement here, precisely because the key IS present.
    assert counts == {'A': 2, 'B': 0}


def test_only_architect_cells_are_counted():
    """An implementer cell carrying the marker is ignored, even under the same config.

    An implementer run never invokes the plan judge, so it cannot have been
    judged without a reference; counting it would inflate the very number that
    bounds how far ``plan_quality`` may be trusted.
    """
    results = [
        _cell('f1', 'A', plan_quality=0.8, plan_steps=5,
              extra_metrics={'judged_without_reference': False}),
        _cell('f2', 'A', plan_quality=0.8, plan_steps=5,
              extra_metrics={'judged_without_reference': False}),
        _cell('f3', 'A', role_under_test='implementer',
              extra_metrics={'judged_without_reference': True}),
    ]

    assert mod.count_judged_without_reference(results) == {'A': 0}


def test_rendered_table_marks_unmeasured_loudly():
    """The rendered report says UNMEASURED and names σ — it never prints ``0``."""
    report = {'candidates': mod.summarize_candidates(_mixed_results())}

    text = mod.format_campaign_report(report)

    assert 'unmeasured' in text.lower()
    assert 'judged_without_reference' in text
    # The legend must name the producing instrument change, so an operator
    # reading γ1's output cannot mistake an absent bound for a clean one.
    assert '3628' in text or 'σ' in text


# ===== step 9: the PRD-D6 banding partition — ambiguity resolves to RETAIN =====


def _metrics(**kwargs):
    """The metrics dict of a single synthetic cell."""
    return _cell('f', 'A', **kwargs).metrics


def test_ceiling_band_requires_plan_and_valid_reference_and_quality():
    """All three conjuncts satisfied -> the sole DISCARDED band."""
    m = _metrics(plan_steps=5, plan_quality=0.90,
                 extra_metrics={'judged_without_reference': False})

    assert mod.band_for_cell(m, 0.80, True) == 'ceiling'


def test_planned_below_q_ceiling_is_intermittent():
    """Same cell below the threshold is retained."""
    m = _metrics(plan_steps=5, plan_quality=0.60,
                 extra_metrics={'judged_without_reference': False})

    assert mod.band_for_cell(m, 0.80, True) == 'intermittent'


def test_planned_without_valid_reference_is_intermittent():
    """A high score judged WITHOUT a valid reference can never discard a fixture.

    plan_quality is interpretable only where a real reference block exists, so a
    high plausibility score against no reference is not evidence of a ceiling.
    """
    m = _metrics(plan_steps=5, plan_quality=0.95,
                 extra_metrics={'judged_without_reference': True})

    assert mod.band_for_cell(m, 0.80, True) == 'intermittent'


def test_no_plan_band():
    """No plan -> ``no_plan``, decided by produced_a_plan and NOT by plan_quality.

    metrics.py:204 records why: the two plan scorers disagreed exactly on a
    stepless artifact, so a nonzero plan_quality is not evidence a plan exists.
    """
    m = _metrics(plan_steps=0, plan_quality=0.95,
                 extra_metrics={'judged_without_reference': False})

    assert mod.band_for_cell(m, 0.80, True) == 'no_plan'


def test_cap_tainted_cell_bands_unmeasured():
    """A transport refusal is NAMED, not banded on a refusal we never got to ask.

    γ1's recipe re-runs these so every fixture gets one admissible cell; banding
    one would penalise whichever candidate happened to be scheduled inside a
    session-cap window — a property of the schedule, not of the candidate.
    """
    m = _metrics(plan_steps=5, plan_quality=0.95, cap_tainted=True,
                 invocation_error='architect: cap_hit',
                 extra_metrics={'judged_without_reference': False})

    assert mod.band_for_cell(m, 0.80, True) == 'unmeasured'


def test_marker_unavailable_never_bands_ceiling():
    """THE load-bearing D6 safety rule: unknown validity can never discard.

    With σ unlanded the driver cannot know whether a cell was judged against a
    real reference diff, so the ceiling band's "valid reference exists" conjunct
    is unsatisfiable and no fixture can be discarded. D6 is explicit about the
    asymmetry: misbanding-to-retain costs ~$20 of stage-2 spend, while
    misbanding-to-discard loses signal permanently.
    """
    m = _metrics(plan_steps=5, plan_quality=0.95)
    assert mod.MARKER_KEY not in m, 'premise: this cell predates σ and carries no marker'

    assert mod.band_for_cell(m, 0.80, False) == 'intermittent'


def test_partition_is_exact():
    """retained + discarded partitions the pool exactly — nothing lost or doubled."""
    results = [
        # ceiling: planned, validly referenced, at/over the threshold.
        _cell('f1', 'A', plan_steps=6, plan_quality=0.92,
              extra_metrics={'judged_without_reference': False}),
        # intermittent: planned but below it.
        _cell('f2', 'A', plan_steps=5, plan_quality=0.40,
              extra_metrics={'judged_without_reference': False}),
        # no_plan.
        _cell('f3', 'A', plan_steps=0, plan_quality=0.0,
              extra_metrics={'judged_without_reference': False}),
        # unmeasured: cap-tainted.
        _cell('f4', 'A', cap_tainted=True, invocation_error='architect: cap_hit',
              extra_metrics={'judged_without_reference': False}),
    ]

    part = mod.partition_bands(results, 0.80)

    assert part['q_ceiling'] == 0.80
    assert part['marker_available'] is True
    assert part['by_fixture'] == {
        'f1': 'ceiling', 'f2': 'intermittent', 'f3': 'no_plan', 'f4': 'unmeasured',
    }
    assert part['counts'] == {
        'ceiling': 1, 'intermittent': 1, 'no_plan': 1, 'unmeasured': 1,
    }
    assert part['discarded'] == ['f1']
    assert part['retained'] == ['f2', 'f3', 'f4']
    assert set(part['retained']) & set(part['discarded']) == set()
    assert set(part['retained']) | set(part['discarded']) == set(part['by_fixture'])


def test_partition_bands_on_the_most_retaining_cell():
    """Multiple admitted cells for one fixture -> band on the most RETAINING one.

    Stage 1 is one trial per fixture, but a re-run can leave two; the D6
    ambiguity rule must still hold, so a fixture with any non-ceiling cell is
    retained.
    """
    results = [
        _cell('f1', 'A', trial=1, plan_steps=6, plan_quality=0.95,
              extra_metrics={'judged_without_reference': False}),
        _cell('f1', 'A', trial=2, plan_steps=5, plan_quality=0.20,
              extra_metrics={'judged_without_reference': False}),
    ]

    part = mod.partition_bands(results, 0.80)

    assert part['by_fixture'] == {'f1': 'intermittent'}
    assert part['discarded'] == []


def test_banding_without_q_ceiling_exits(tmp_path, monkeypatch):
    """``--stage1`` with no ``--q-ceiling`` exits BEFORE any spend.

    G6: the threshold is derived in γ1 and ratified by Leo at γ2. A default here
    would silently become the de facto threshold and pre-empt that ruling.
    """
    d = _make_fixture_dir(tmp_path, ['alpha'])

    def _boom(*args, **kwargs):
        pytest.fail('reached the live-spend seam before rejecting a missing --q-ceiling',
                    pytrace=False)

    monkeypatch.setattr(mod, '_run_campaign', _boom)

    with pytest.raises(SystemExit) as exc:
        mod.main([
            '--run', '--tasks-dir', str(d),
            '--candidate', 'architect-opus-max', '--stage1',
        ])

    assert exc.value.code != 0
    assert 'q-ceiling' in str(exc.value)


# ===== step 11: the output schema and the --out JSON artifact =====


def _report(q_ceiling=None):
    """A campaign report over the mixed result set, both candidates resolved."""
    candidates = mod.resolve_candidates(['architect-opus-max', 'architect-fable-high'])
    cell_matrix = mod.enumerate_cells(
        [Path(f'{stem}.json') for stem in ('f1', 'f2', 'f3')],
        [c.name for c in candidates], 1,
    )
    return mod.build_campaign_report(
        _mixed_results(), candidates, cell_matrix, q_ceiling=q_ceiling,
    )


def test_output_schema_carries_the_four_required_per_candidate_fields():
    """THE signal this task exists to produce, mechanically and per candidate.

    planRate, plan_quality, cap_excluded and judged_without_reference — emitted
    by the driver rather than reconstructed by an operator from a log, which is
    what makes γ1's calibration report COMPUTED instead of hand-derived.
    """
    report = _report()

    assert report['candidates']
    for entry in report['candidates']:
        assert set(entry) == {
            'config_name', 'model', 'effort', 'max_budget_usd',
            'n', 'total', 'plan_rate', 'mean_plan_quality',
            'cap_excluded', 'no_plan', 'judged_without_reference',
        }
    by_name = {e['config_name']: e for e in report['candidates']}
    assert by_name['architect-fable-high']['model'] == 'claude-fable-5'
    assert by_name['architect-opus-max']['effort'] == 'max'


def test_report_carries_cells_and_cell_matrix():
    """The raw per-cell dump makes the verdict recomputable without a log."""
    report = _report()

    assert len(report['cells']) == len(_mixed_results())
    for cell in report['cells']:
        assert set(cell) == {
            'task_id', 'config_name', 'trial', 'outcome', 'plan_quality',
            'plan_steps', 'cost_usd', 'judge_cost_usd', 'cap_tainted',
        }
    # The ENUMERATED expected matrix rides along, so a cell that never returned
    # is visible as a gap rather than silently absent.
    assert report['cell_matrix'] == mod.enumerate_cells(
        [Path(f'{stem}.json') for stem in ('f1', 'f2', 'f3')],
        ['architect-opus-max', 'architect-fable-high'], 1,
    )


def test_bands_present_only_when_q_ceiling_given():
    """No q_ceiling -> no ``bands`` key at all, never a dict of fabricated bands."""
    assert 'bands' not in _report()

    banded = _report(q_ceiling=0.80)
    assert banded['bands']['q_ceiling'] == 0.80
    assert set(banded['bands']) == {
        'q_ceiling', 'marker_available', 'by_fixture', 'counts', 'retained', 'discarded',
    }


def test_out_writes_json_and_is_reloadable(tmp_path, capsys):
    """``--out`` writes an artifact that round-trips through ``json.loads``."""
    d = _make_fixture_dir(tmp_path, ['alpha', 'beta'])
    out = tmp_path / 'nested' / 'r.json'

    rc = mod.main([
        '--dry-run', '--tasks-dir', str(d),
        '--candidate', 'architect-opus-max',
        '--trials', '2', '--out', str(out),
    ])

    assert rc == 0
    assert out.exists(), '--out did not create its parent dirs'
    loaded = json.loads(out.read_text())
    assert loaded['cell_matrix'] == mod.enumerate_cells(
        sorted(d.glob('*.json')), ['architect-opus-max'], 2,
    )
    assert json.loads(json.dumps(loaded)) == loaded


def test_rendering_is_deterministic():
    """Formatting the same report twice is byte-identical.

    No wall-clock stamp and no dict-iteration-order dependence, so a committed
    γ1 artifact diffs cleanly against its successor.
    """
    report = _report(q_ceiling=0.80)

    first = mod.format_campaign_report(report)
    second = mod.format_campaign_report(report)

    assert first == second
    assert 'ceiling' in first, 'the band summary is not rendered'


# ===== step 13: the --run path drives run_ofat_stage, never eval-ofat =====


@pytest.fixture
def campaign_seam(monkeypatch):
    """Patch every live seam so ``--run`` is exercised at ZERO spend.

    Three patches, each closing a different way a test could reach the real
    world: ``run_ofat_stage`` (the fan-out that would spawn architect runs),
    ``save_result`` (which writes into the packaged results dir), and
    ``load_config`` (which would demand a real config / OAuth).

    A fourth is a TRIPWIRE rather than a stub: ``ofat_candidates`` fails the
    test outright if the driver ever calls it. The PRD forbids eval-ofat
    because it runs all 8 candidates including the ~10x-cost implementer and
    judge cells, so "the driver only ever passes the explicitly-named architect
    candidates" has to be enforced on every --run test, not just one.
    """
    import orchestrator.config as orch_config
    from orchestrator.evals import configs as eval_configs
    from orchestrator.evals import runner as eval_runner

    calls = {'ofat': [], 'saved': []}

    async def _recorder(task_paths, candidates, base_config=None, **kwargs):
        calls['ofat'].append({
            'task_paths': list(task_paths),
            'candidates': list(candidates),
            'base_config': base_config,
            **kwargs,
        })
        return [
            _cell(path.stem, cfg.name, trial=trial, plan_quality=0.75, plan_steps=5)
            for path in task_paths
            for cfg in candidates
            for trial in range(1, (kwargs.get('trials') or 1) + 1)
        ]

    def _tripwire(*args, **kwargs):
        pytest.fail(
            'the driver called ofat_candidates(). eval-ofat runs all 8 candidates '
            'including the ~10x-cost implementer/judge cells — the exact spend the '
            'PRD forbids. Only the explicitly-named architect candidates may be '
            'passed to run_ofat_stage.',
            pytrace=False,
        )

    monkeypatch.setattr(eval_runner, 'run_ofat_stage', _recorder)
    monkeypatch.setattr(eval_runner, 'save_result', lambda r: calls['saved'].append(r))
    monkeypatch.setattr(orch_config, 'load_config', lambda *a, **k: object())
    monkeypatch.setattr(eval_configs, 'ofat_candidates', _tripwire)
    return calls


def test_run_passes_resolved_candidates_fixtures_trials_and_parallelism(
    tmp_path, campaign_seam,
):
    """``--run`` hands the seam exactly the resolved candidates and parameters."""
    d = _make_fixture_dir(tmp_path, ['alpha', 'beta'])

    rc = mod.main([
        '--run', '--tasks-dir', str(d),
        '--candidate', 'architect-opus-max',
        '--trials', '3', '--max-parallel', '2',
        '--budget', 'architect-opus-max=15',
    ])

    assert rc == 0
    assert len(campaign_seam['ofat']) == 1
    call = campaign_seam['ofat'][0]
    assert call['task_paths'] == sorted(d.glob('*.json'))
    assert len(call['candidates']) == 1
    assert call['candidates'][0].name == 'architect-opus-max'
    assert call['candidates'][0].max_budget_usd == 15.0
    assert call['trials'] == 3
    assert call['max_parallel'] == 2


def test_run_never_uses_the_eval_ofat_candidate_set(tmp_path, campaign_seam):
    """Only the named architect candidates reach the seam — never the 8-candidate set.

    The ``ofat_candidates`` tripwire in ``campaign_seam`` fails this test if the
    driver reaches for the OFAT bundle; the assertions below additionally pin
    that nothing non-architect rode along.
    """
    d = _make_fixture_dir(tmp_path, ['alpha'])

    mod.main([
        '--run', '--tasks-dir', str(d),
        '--candidate', 'architect-opus-max',
        '--candidate', 'architect-fable-high',
    ])

    passed = campaign_seam['ofat'][0]['candidates']
    assert [c.name for c in passed] == ['architect-opus-max', 'architect-fable-high']
    assert all(c.role == 'architect' for c in passed)


def test_run_summarizes_the_returned_cells(tmp_path, campaign_seam, capsys):
    """The returned cells flow into the printed report AND the --out artifact.

    i.e. ``--run`` and the analyze path produce the same schema, so the campaign
    is auditable from the file without re-reading the terminal.
    """
    d = _make_fixture_dir(tmp_path, ['alpha', 'beta'])
    out = tmp_path / 'r.json'

    mod.main([
        '--run', '--tasks-dir', str(d),
        '--candidate', 'architect-opus-max',
        '--out', str(out),
    ])

    printed = capsys.readouterr().out
    assert 'architect-opus-max' in printed
    loaded = json.loads(out.read_text())
    assert len(loaded['cells']) == 2
    assert {c['task_id'] for c in loaded['cells']} == {'alpha', 'beta'}
    assert loaded['candidates'][0]['config_name'] == 'architect-opus-max'
    assert loaded['candidates'][0]['n'] == 2


def test_run_persists_each_cell(tmp_path, campaign_seam):
    """Every returned cell is persisted, so a later --results-dir can re-read it."""
    d = _make_fixture_dir(tmp_path, ['alpha', 'beta'])

    mod.main([
        '--run', '--tasks-dir', str(d),
        '--candidate', 'architect-opus-max', '--trials', '2',
    ])

    assert len(campaign_seam['saved']) == 4
    assert {r.task_id for r in campaign_seam['saved']} == {'alpha', 'beta'}
