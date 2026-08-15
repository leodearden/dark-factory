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

_PRE_SIGMA = (mod.MARKER_KEY,)
"""``drop_metrics`` argument for a cell that PREDATES eval-revival σ (task 3628).

σ declared ``judged_without_reference`` on ``EvalMetrics``, and ``to_dict`` is a
bare ``asdict``, so every cell the live instrument emits now carries the key —
that presence-on-every-cell property is exactly what this driver's per-cell
validity rule (``MARKER_KEY not in metrics`` -> not known-good) reads, and it
must NOT be weakened to a conditional emit. But a keyless cell remains a real
input: ``--results-dir`` replays artifacts written by pre-σ runs. Dropping the
key EXPLICITLY at those call sites states that premise on purpose, instead of
inheriting it from the dataclass not yet having the field.
"""


def _cell(
    task_id, config_name, *, trial=1, plan_quality=None, plan_steps=0,
    role_under_test='architect', cap_tainted=False, invocation_error=None,
    cost_usd=0.0, judge_cost_usd=0.0, extra_metrics=None, drop_metrics=(),
):
    """Build a synthetic ``EvalResult`` with a production-shaped metrics dict.

    Copies ``orchestrator/tests/test_eval_composite_report.py``'s helper shape:
    ``plan_steps`` is threaded EXPLICITLY because it is the plan-production
    predicate the report layer reads — a cell declaring a ``plan_quality``
    without the step count it came from is a self-contradictory fixture.

    ``extra_metrics`` merges keys on top of the dataclass's own fields, and
    ``drop_metrics`` pops named keys AFTER that merge, yielding a genuinely
    KEYLESS cell. Since σ (task 3628) landed, ``judged_without_reference`` is a
    declared field and so arrives on every cell this helper builds; a call site
    that needs the pre-σ shape must therefore ask for it with
    ``drop_metrics=_PRE_SIGMA``. Do not pass the same key in both.
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
    for key in drop_metrics:
        metrics.pop(key, None)
    return EvalResult(
        task_id=task_id,
        config_name=config_name,
        outcome='completed',
        metrics=metrics,
        worktree_path='/tmp/eval',
        trial=trial,
    )


def _mixed_results():
    """Candidate A with one cap-tainted cell, candidate B with one no-plan cell.

    Every cell is PRE-σ (keyless), which is what makes this the fixture for the
    "instrument never measured reference validity" direction — the corpus a
    ``--results-dir`` replay of an old run produces.
    """
    return [
        # A: one transport refusal (never got to ask the model) + two scored cells.
        _cell('f1', 'architect-opus-max', cap_tainted=True,
              invocation_error='architect: cap_hit', drop_metrics=_PRE_SIGMA),
        _cell('f2', 'architect-opus-max', plan_quality=0.80, plan_steps=6,
              drop_metrics=_PRE_SIGMA),
        _cell('f3', 'architect-opus-max', plan_quality=0.60, plan_steps=4,
              drop_metrics=_PRE_SIGMA),
        # B: one genuine no-plan measurement + two scored cells.
        _cell('f1', 'architect-fable-high', plan_quality=0.0, plan_steps=0,
              drop_metrics=_PRE_SIGMA),
        _cell('f2', 'architect-fable-high', plan_quality=0.70, plan_steps=5,
              drop_metrics=_PRE_SIGMA),
        _cell('f3', 'architect-fable-high', plan_quality=0.50, plan_steps=3,
              drop_metrics=_PRE_SIGMA),
    ]


def _mixed_marker_results():
    """A MIXED corpus: one keyless (pre-σ) architect cell + one keyed (post-σ) one.

    Both are planned and high-quality, so the ONLY thing distinguishing them is
    whether their reference validity was ever measured. This is the normal
    transition state, not an exotic one: γ1's recipe re-runs cap-tainted
    fixtures, and a re-run after eval-revival σ (task 3628) lands writes new
    cells carrying the key alongside the old ones that never could.
    """
    return [
        _cell('f_pre', 'A', plan_steps=5, plan_quality=0.95,
              drop_metrics=_PRE_SIGMA),
        _cell('f_post', 'A', plan_steps=5, plan_quality=0.95,
              extra_metrics={'judged_without_reference': False}),
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
    # strict=True is not incidental: apply_budgets is a comprehension over
    # `candidates`, so a length change would be a contract break, not a zip quirk.
    assert all(
        a is not b for a, b in zip(cands, out, strict=True) if a.name == 'architect-opus-max'
    )


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
    """Over a wholly pre-σ corpus, the count is ``None`` for every candidate.

    Asserting ``is None`` and NOT ``== 0`` is the whole point. The marker's
    producer is eval-revival σ (task 3628); cells written before it landed
    cannot carry the key, and ``--results-dir`` still replays them.
    ``EvalMetrics.to_dict()`` is an ``asdict`` and therefore emits every
    DECLARED field, which makes "key absent on every cell" mean unambiguously
    "these cells predate σ" rather than "no offending cell".
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
    """With σ landed the key is simply present, and the Nones become real counts."""
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

    assert mod.band_for_cell(m, 0.80) == 'ceiling'


def test_planned_below_q_ceiling_is_intermittent():
    """Same cell below the threshold is retained."""
    m = _metrics(plan_steps=5, plan_quality=0.60,
                 extra_metrics={'judged_without_reference': False})

    assert mod.band_for_cell(m, 0.80) == 'intermittent'


def test_planned_without_valid_reference_is_intermittent():
    """A high score judged WITHOUT a valid reference can never discard a fixture.

    plan_quality is interpretable only where a real reference block exists, so a
    high plausibility score against no reference is not evidence of a ceiling.
    """
    m = _metrics(plan_steps=5, plan_quality=0.95,
                 extra_metrics={'judged_without_reference': True})

    assert mod.band_for_cell(m, 0.80) == 'intermittent'


def test_no_plan_band():
    """No plan -> ``no_plan``, decided by produced_a_plan and NOT by plan_quality.

    metrics.py:204 records why: the two plan scorers disagreed exactly on a
    stepless artifact, so a nonzero plan_quality is not evidence a plan exists.
    """
    m = _metrics(plan_steps=0, plan_quality=0.95,
                 extra_metrics={'judged_without_reference': False})

    assert mod.band_for_cell(m, 0.80) == 'no_plan'


def test_cap_tainted_cell_bands_unmeasured():
    """A transport refusal is NAMED, not banded on a refusal we never got to ask.

    γ1's recipe re-runs these so every fixture gets one admissible cell; banding
    one would penalise whichever candidate happened to be scheduled inside a
    session-cap window — a property of the schedule, not of the candidate.
    """
    m = _metrics(plan_steps=5, plan_quality=0.95, cap_tainted=True,
                 invocation_error='architect: cap_hit',
                 extra_metrics={'judged_without_reference': False})

    assert mod.band_for_cell(m, 0.80) == 'unmeasured'


def test_marker_unavailable_never_bands_ceiling():
    """THE load-bearing D6 safety rule: unknown validity can never discard.

    For a cell carrying no marker the driver cannot know whether it was judged
    against a real reference diff, so the ceiling band's "valid reference
    exists" conjunct is unsatisfiable and that fixture can never be discarded.
    D6 is explicit about the
    asymmetry: misbanding-to-retain costs ~$20 of stage-2 spend, while
    misbanding-to-discard loses signal permanently.
    """
    m = _metrics(plan_steps=5, plan_quality=0.95, drop_metrics=_PRE_SIGMA)
    assert mod.MARKER_KEY not in m, 'premise: this cell predates σ and carries no marker'

    assert mod.band_for_cell(m, 0.80) == 'intermittent'


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
            # The companion count: `judged_without_reference is None` alone
            # cannot distinguish 1-of-50 unmeasured cells from 50-of-50, so a
            # partially-instrumented corpus stays legible rather than merely
            # unknown.
            'judged_without_reference_unmeasured_cells',
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


def test_results_dir_reanalyzes_persisted_cells(tmp_path, capsys):
    """``--results-dir`` re-reads persisted cells and rebuilds the SAME schema.

    This is what makes a committed γ1 artifact recomputable: the verdict can be
    regenerated from the stored cells without re-spending the campaign. Fields
    unknown to ``EvalResult`` are dropped rather than raising, so a cell
    persisted before a field existed still loads.
    """
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    for stem in ('alpha', 'beta'):
        cell = _cell(stem, 'architect-opus-max', plan_quality=0.75, plan_steps=5)
        payload = {**cell.to_dict(), 'a_field_this_driver_has_never_heard_of': 1}
        (results_dir / f'{stem}.json').write_text(json.dumps(payload))

    d = _make_fixture_dir(tmp_path, ['alpha', 'beta'])
    out = tmp_path / 'reanalyzed.json'

    rc = mod.main([
        '--results-dir', str(results_dir), '--tasks-dir', str(d),
        '--candidate', 'architect-opus-max', '--out', str(out),
    ])

    assert rc == 0
    loaded = json.loads(out.read_text())
    assert {c['task_id'] for c in loaded['cells']} == {'alpha', 'beta'}
    assert loaded['candidates'][0]['n'] == 2
    assert loaded['candidates'][0]['mean_plan_quality'] == 0.75


# ===== step 15: reference validity is a PER-CELL fact, not a run-level one =====


def test_mixed_corpus_never_discards_the_keyless_fixture():
    """THE regression: an unrelated sibling cell must not flip a fixture to DISCARDED.

    Reproduced first-hand against this branch before this test existed:
    ``partition_bands([pre], 0.8)['discarded']`` was ``[]`` but
    ``partition_bands([pre, post], 0.8)['discarded']`` was
    ``['f_post', 'f_pre']`` — adding one post-σ sibling that says nothing
    whatsoever about ``f_pre`` flipped ``f_pre`` from RETAINED to DISCARDED.

    ``f_pre``'s reference validity was NEVER measured, so D6's "ambiguity ->
    retain" must fire on it regardless of what any other cell carries. This is
    the one direction PRD D6 calls permanently lossy: misbanding-to-retain costs
    ~$20 of stage-2 spend, misbanding-to-discard loses the signal for good.
    """
    pre, post = _mixed_marker_results()

    mixed = mod.partition_bands([pre, post], 0.80)
    alone = mod.partition_bands([pre], 0.80)

    assert mixed['by_fixture']['f_pre'] == 'intermittent'
    assert 'f_pre' not in mixed['discarded']
    # The keyless fixture's band must be INDEPENDENT of the sibling's presence.
    assert mixed['by_fixture']['f_pre'] == alone['by_fixture']['f_pre']
    # And the genuinely-measured sibling still bands on its own merits.
    assert mixed['by_fixture']['f_post'] == 'ceiling'
    assert mixed['discarded'] == ['f_post']


def test_band_for_cell_reads_validity_per_cell():
    """``band_for_cell(metrics, q_ceiling)`` — validity read off the CELL, no run flag.

    The keyless and key-``True`` cases must be INDISTINGUISHABLE at the band
    level: both mean "this cell's reference validity is not known-good", and
    ``plan_quality`` is interpretable only where a real reference block exists.
    The run-level parameter is gone entirely rather than ignored — an argument
    that looks load-bearing but is not is precisely the trap that produced this
    bug.
    """
    keyless = _metrics(plan_steps=5, plan_quality=0.95, drop_metrics=_PRE_SIGMA)
    judged_without = _metrics(plan_steps=5, plan_quality=0.95,
                              extra_metrics={'judged_without_reference': True})
    validly_judged = _metrics(plan_steps=5, plan_quality=0.95,
                              extra_metrics={'judged_without_reference': False})

    assert mod.MARKER_KEY not in keyless, 'premise: this cell predates σ'

    assert mod.band_for_cell(keyless, 0.80) == 'intermittent'
    assert mod.band_for_cell(judged_without, 0.80) == 'intermittent'
    assert mod.band_for_cell(validly_judged, 0.80) == 'ceiling'


def test_mixed_corpus_count_reads_unmeasured_not_zero():
    """One unmeasured architect cell makes the candidate's count ``None``, never ``0``.

    Partial measurement is not measurement. ``0`` asserts "we looked at every
    cell and none was judged without a reference" — the exact fabrication the
    module forbids, and it would let ``plan_quality`` read as fully
    validity-bounded when one of the two cells was never bounded at all.
    Verified first-hand that the pre-fix code returned ``{'A': 0}`` here.
    """
    counts = mod.count_judged_without_reference(_mixed_marker_results())

    assert counts['A'] is None
    assert counts['A'] != 0


def test_unmeasured_marker_cells_are_counted_per_candidate():
    """``None`` alone cannot distinguish 1-of-50 unmeasured from 50-of-50.

    So the driver also reports HOW MANY architect cells lacked the key, which
    makes a partially-instrumented corpus legible rather than merely unknown.
    """
    assert mod.count_unmeasured_marker_cells(_mixed_marker_results()) == {'A': 1}

    fully_keyed = [
        _cell('f1', 'A', plan_steps=5, plan_quality=0.9,
              extra_metrics={'judged_without_reference': False}),
        _cell('f2', 'A', plan_steps=5, plan_quality=0.9,
              extra_metrics={'judged_without_reference': True}),
    ]
    assert mod.count_unmeasured_marker_cells(fully_keyed) == {'A': 0}

    fully_keyless = [
        _cell('f1', 'A', plan_steps=5, plan_quality=0.9, drop_metrics=_PRE_SIGMA),
        _cell('f2', 'A', plan_steps=5, plan_quality=0.9, drop_metrics=_PRE_SIGMA),
        _cell('f3', 'A', plan_steps=5, plan_quality=0.9, drop_metrics=_PRE_SIGMA),
    ]
    assert mod.count_unmeasured_marker_cells(fully_keyless) == {'A': 3}


def test_implementer_cell_missing_the_key_does_not_poison_the_count():
    """A non-architect cell cannot flip a fully-measured candidate to unmeasured.

    An implementer run never invokes the plan judge, so it is out of scope by
    construction. The role filter must therefore run BEFORE the key check —
    otherwise a single implementer cell riding along under the same config name
    would erase a genuine, complete measurement.
    """
    results = [
        _cell('f1', 'A', plan_steps=5, plan_quality=0.9,
              extra_metrics={'judged_without_reference': True}),
        _cell('f2', 'A', plan_steps=5, plan_quality=0.9,
              extra_metrics={'judged_without_reference': False}),
        # No marker key at all: the role filter must reject it BEFORE the key check.
        _cell('f3', 'A', role_under_test='implementer', drop_metrics=_PRE_SIGMA),
    ]

    assert mod.count_judged_without_reference(results) == {'A': 1}
    assert mod.count_unmeasured_marker_cells(results) == {'A': 0}


def test_uniform_corpora_are_unchanged():
    """THE anti-overcorrection assertion: only the MIXED case may change.

    Both uniform directions must behave exactly as before — an all-keyless
    corpus still reports every count ``None`` and discards nothing, and an
    all-keyed corpus still counts and bands exactly as it did.
    """
    # All keyless (today's instrument): unmeasured everywhere, nothing discarded.
    keyless = _mixed_results()
    assert all(v is None for v in mod.count_judged_without_reference(keyless).values())
    assert mod.partition_bands(keyless, 0.80)['discarded'] == []

    # All keyed (post-σ): real counts, and the ceiling band is reachable again.
    keyed = [
        _cell('f1', 'A', plan_steps=6, plan_quality=0.92,
              extra_metrics={'judged_without_reference': False}),
        _cell('f2', 'A', plan_steps=5, plan_quality=0.40,
              extra_metrics={'judged_without_reference': False}),
        _cell('f3', 'A', plan_steps=5, plan_quality=0.99,
              extra_metrics={'judged_without_reference': True}),
    ]
    assert mod.count_judged_without_reference(keyed) == {'A': 1}
    part = mod.partition_bands(keyed, 0.80)
    assert part['discarded'] == ['f1']
    assert part['by_fixture'] == {
        'f1': 'ceiling', 'f2': 'intermittent', 'f3': 'intermittent',
    }


# ===== review round 1: ONE marker count, SURFACED — never a second derivation =====
#
# Reproduced first-hand on this branch before these tests existed.
# ``summarize_candidates`` builds ``build_plan_quality_report(results)`` and then,
# one line later, RE-DERIVES the same per-candidate marker count itself. The two
# implementations aggregate over DIFFERENT populations: the report layer counts
# only the ADMITTED pool (a cap-tainted cell never reaches the count — pinned by
# ``orchestrator/tests/test_eval_architect.py``'s
# ``test_a_keyless_cap_tainted_cell_does_not_poison_the_count``), while this
# driver's ``_architect_cells`` yields EVERY architect cell, cap-tainted ones
# included. Measured on ``_cap_tainted_keyless_corpus()`` below:
#
#     build_plan_quality_report(results)['configs'][A]
#         -> judged_without_reference=1, judged_without_reference_unmeasured=0
#     summarize_candidates(results)[A]
#         -> judged_without_reference=None, ..._unmeasured_cells=1
#
# So ONE keyless (pre-σ) cap-tainted cell poisoned the driver's answer to
# UNMEASURED while the report object the same function had just built printed a
# real number. That corpus is the NORMAL transition state, not a corner: γ1's
# recipe re-runs cap-tainted fixtures, so a post-σ re-run writes keyed cells
# beside older keyless ones. Two operator-facing surfaces, built inside one
# function from one input, contradicting each other about one quantity is exactly
# what ``report.py``:334-340 and ``metrics.py`` assert three times over must not
# happen — "two consumers of one field must not answer the same question
# differently".


def _report_entries(results):
    """The report layer's per-config accumulator, keyed by ``config_name``."""
    from orchestrator.evals.report import build_plan_quality_report

    return {c['config_name']: c for c in build_plan_quality_report(results)['configs']}


def _summary_rows(results):
    """The driver's per-candidate summary rows, keyed by ``config_name``."""
    return {r['config_name']: r for r in mod.summarize_candidates(results)}


def _cap_tainted_keyless_corpus():
    """Two admitted keyed cells (one judged blind) + one PRE-σ cap-tainted cell.

    The minimal corpus on which the two surfaces diverged: everything in the
    ADMITTED pool is measured, so the report layer answers ``1``; the only
    keyless cell is the one that never entered that pool.
    """
    return [
        _cell('f1', 'A', plan_quality=0.8, plan_steps=5,
              extra_metrics={mod.MARKER_KEY: True}),
        _cell('f2', 'A', plan_quality=0.6, plan_steps=5,
              extra_metrics={mod.MARKER_KEY: False}),
        _cell('f3', 'A', plan_steps=5, cap_tainted=True,
              invocation_error='architect: cap_hit', drop_metrics=_PRE_SIGMA),
    ]


def test_a_keyless_cap_tainted_cell_does_not_poison_the_summary():
    """THE reproduction, kept as the agreement pin between the two surfaces.

    THE POPULATION RULE: a cap-tainted cell has NO ``plan_quality`` to bound, so
    its keylessness cannot make the candidate's bound unknown. It never entered
    the admitted pool the count describes, and it is already counted —
    disjointly — by ``cap_excluded``. Nothing is hidden by declining to poison
    here: :func:`band_for_cell` still bands that very cell ``unmeasured`` at rung
    1, per cell, where the question asked is a different one.
    """
    results = _cap_tainted_keyless_corpus()
    assert mod.MARKER_KEY not in results[2].metrics, (
        'premise: this cell is keyless on purpose (drop_metrics=_PRE_SIGMA)'
    )

    entry = _report_entries(results)['A']
    row = _summary_rows(results)['A']

    assert row[mod.MARKER_KEY] == entry['judged_without_reference'] == 1
    assert (
        row[mod.UNMEASURED_CELLS_KEY]
        == entry['judged_without_reference_unmeasured']
        == 0
    )


def test_a_keyless_admitted_cell_still_reads_unmeasured_on_both_surfaces():
    """ANTI-OVERCORRECTION: the ``None`` direction must survive the fix.

    A keyless cell INSIDE the admitted pool did contribute a ``plan_quality`` to
    the candidate's aggregate, and nothing measured whether a reference bounded
    it — so the bound is genuinely unknown and both surfaces must say so. Without
    this pin the fix could degenerate into "always report a number", which is the
    fabricated ``0`` the whole marker exists to prevent.
    """
    results = _mixed_marker_results()

    entry = _report_entries(results)['A']
    row = _summary_rows(results)['A']

    assert row[mod.MARKER_KEY] is entry['judged_without_reference'] is None
    assert (
        row[mod.UNMEASURED_CELLS_KEY]
        == entry['judged_without_reference_unmeasured']
        == 1
    )


@pytest.mark.parametrize('corpus', [_mixed_results, _mixed_marker_results],
                         ids=['mixed_results', 'mixed_marker_results'])
def test_both_marker_values_agree_with_the_report_layer(corpus):
    """Field for field, over every corpus this module already builds.

    THE pin that stops the two accumulators drifting again — and it is cheap to
    extend when a future corpus is added, which is the point of parametrizing it
    rather than asserting on one fixture.
    """
    results = corpus()

    entries = _report_entries(results)
    rows = _summary_rows(results)

    assert set(rows) == set(entries)
    for name, row in rows.items():
        entry = entries[name]
        assert row[mod.MARKER_KEY] == entry['judged_without_reference'], (
            f'{name}: marker count drifted from the report layer'
        )
        assert (
            row[mod.UNMEASURED_CELLS_KEY]
            == entry['judged_without_reference_unmeasured']
        ), f'{name}: unmeasured-cell count drifted from the report layer'


def test_an_empty_admitted_pool_answers_zero_on_both_surfaces():
    """A vacuous pool reports ``0``, identically, and visibly is not a quality claim.

    The single cell is cap-tainted but KEYED, so keylessness is not the variable:
    what is pinned is that "no admitted cell was judged blind" over an empty pool
    is a deliberate ``0`` rather than an accident of which surface answered.
    ``mean_plan_quality is None`` is asserted alongside so the ``0`` cannot be
    read as a claim about quality — nothing scored at all.
    """
    results = [
        _cell('f1', 'A', plan_steps=5, cap_tainted=True,
              invocation_error='architect: cap_hit',
              extra_metrics={mod.MARKER_KEY: False}),
    ]

    entry = _report_entries(results)['A']
    row = _summary_rows(results)['A']

    assert row[mod.MARKER_KEY] == entry['judged_without_reference'] == 0
    assert (
        row[mod.UNMEASURED_CELLS_KEY]
        == entry['judged_without_reference_unmeasured']
        == 0
    )
    assert row['mean_plan_quality'] is entry['mean_plan_quality'] is None


def test_the_rendered_fraction_quotes_one_population():
    """``unmeasured (N of M cells)`` must not mix an admitted N with an all-cells M.

    ``architect-opus-max`` in :func:`_mixed_results` has 3 architect cells but
    only 2 admitted ones, and every cell is keyless — so the numerator (keyless
    ADMITTED cells) is 2 and the denominator must be ``n``, not ``total``.
    Rendering ``3 of 3`` would quote a numerator from one population against a
    denominator from another and overstate how much of the admitted pool went
    unmeasured. Asserted on the NUMBERS in that candidate's own line, not on the
    surrounding prose, and not on the whole report — ``architect-fable-high``
    legitimately reads ``3 of 3`` (n == total == 3 there).
    """
    report = {'candidates': mod.summarize_candidates(_mixed_results())}

    text = mod.format_campaign_report(report)

    line = next(
        line for line in text.splitlines()
        if line.startswith('architect-opus-max')
    )
    assert '2 of 2' in line, line
    assert '3 of 3' not in line, line


# ===== step 17: --results-dir must not silently absorb out-of-campaign cells =====
#
# Not hypothetical. ``save_result`` (runner.py:1370) writes into the SHARED
# packaged ``RESULTS_DIR = Path(__file__).parent / 'results'`` (runner.py:50),
# and the main checkout's copy already holds 586 persisted cells — 51 of them v1
# ``architect-fable-high`` runs over the incumbent-biased standing corpus (e.g.
# ``df_task_12__architect-fable-high__*.json``). The documented round-trip
# (``--run`` persists, a later ``--results-dir`` re-reads) therefore points
# straight at the contaminated pool.


def _in_campaign_cell(stem='alpha'):
    return _cell(stem, 'architect-opus-max', plan_quality=0.75, plan_steps=5)


def _out_of_campaign_cell(stem='old_v1_fixture', config='architect-sonnet-high'):
    """A v1-corpus cell: wrong candidate, wrong fixture pool, real persisted shape."""
    return _cell(stem, config, plan_quality=0.90, plan_steps=6)


def test_out_of_campaign_cells_are_dropped():
    """Both axes filter INDEPENDENTLY — candidate name and fixture stem.

    The fixture axis matters on its own: a right-candidate/wrong-fixture cell is
    exactly the v1-corpus contamination case, since ``architect-fable-high`` ran
    in v1 too. Filtering only on candidate name would let every v1 cell for a
    reused candidate through, silently answering the v2 question over the
    incumbent-success-biased pool the screen exists to escape.
    """
    keep = _in_campaign_cell('f1')
    wrong_candidate = _cell('f1', 'architect-sonnet-high', plan_quality=0.9, plan_steps=5)
    wrong_fixture = _cell('old_v1_fixture', 'architect-opus-max', plan_quality=0.9, plan_steps=5)

    kept, dropped = mod.filter_campaign_results(
        [keep, wrong_candidate, wrong_fixture], {'architect-opus-max'}, {'f1', 'f2'},
    )

    assert kept == [keep]
    assert len(dropped) == 2
    assert {(d.task_id, d.config_name) for d in dropped} == {
        ('f1', 'architect-sonnet-high'), ('old_v1_fixture', 'architect-opus-max'),
    }


def test_results_dir_report_excludes_out_of_campaign_cells(tmp_path):
    """End to end: a contaminated dir yields a report about THIS campaign only."""
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    for cell in (_in_campaign_cell('alpha'), _out_of_campaign_cell()):
        (results_dir / f'{cell.task_id}__{cell.config_name}.json').write_text(
            json.dumps(cell.to_dict())
        )
    d = _make_fixture_dir(tmp_path, ['alpha'])
    out = tmp_path / 'r.json'

    rc = mod.main([
        '--results-dir', str(results_dir), '--tasks-dir', str(d),
        '--candidate', 'architect-opus-max', '--out', str(out),
        '--stage1', '--q-ceiling', '0.80',
    ])

    assert rc == 0
    loaded = json.loads(out.read_text())
    assert [c['config_name'] for c in loaded['candidates']] == ['architect-opus-max']
    assert {c['task_id'] for c in loaded['cells']} == {'alpha'}
    assert 'old_v1_fixture' not in loaded['bands']['by_fixture']
    assert 'old_v1_fixture' not in loaded['bands']['retained']
    assert 'old_v1_fixture' not in loaded['bands']['discarded']


def test_dropped_cells_are_reported_loudly(tmp_path, capsys):
    """Silent correctness is nearly as bad as no filtering — the drop is RECORDED.

    The committed γ1 artifact must carry its own exclusion record so the
    calibration is auditable, and an operator who pointed at the wrong directory
    must be told rather than handed a plausible report.
    """
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    for cell in (_in_campaign_cell('alpha'), _out_of_campaign_cell()):
        (results_dir / f'{cell.task_id}__{cell.config_name}.json').write_text(
            json.dumps(cell.to_dict())
        )
    d = _make_fixture_dir(tmp_path, ['alpha'])
    out = tmp_path / 'r.json'

    mod.main([
        '--results-dir', str(results_dir), '--tasks-dir', str(d),
        '--candidate', 'architect-opus-max', '--out', str(out),
    ])

    printed = capsys.readouterr().out
    assert '1' in printed and 'drop' in printed.lower()
    assert 'old_v1_fixture' in printed
    loaded = json.loads(out.read_text())
    assert loaded['filtered'] == {
        'dropped': 1,
        'dropped_config_names': ['architect-sonnet-high'],
        'dropped_task_ids': ['old_v1_fixture'],
    }


def test_results_dir_with_no_in_campaign_cells_exits_loudly(tmp_path):
    """Zero survivors EXITS — "we measured nothing" must not share a shape with a result.

    This is the precise symptom of pointing at the wrong directory (or at the
    shared packaged results dir before this campaign has ever run), and a report
    over an empty pool would present it with exactly the shape of a real one.
    """
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    stray = _out_of_campaign_cell()
    (results_dir / 'stray.json').write_text(json.dumps(stray.to_dict()))
    d = _make_fixture_dir(tmp_path, ['alpha'])

    with pytest.raises(SystemExit) as exc:
        mod.main([
            '--results-dir', str(results_dir), '--tasks-dir', str(d),
            '--candidate', 'architect-opus-max',
        ])

    assert exc.value.code != 0
    assert str(results_dir) in str(exc.value)
    assert 'architect-opus-max' in str(exc.value)


def test_build_campaign_report_rejects_an_unknown_config():
    """Defense in depth: a row that cannot trace to a resolved EvalConfig RAISES.

    Verified first-hand pre-fix: ``getattr(None, 'model', None)`` silently
    yielded ``None``, so one stray ``architect-sonnet-high`` cell emitted a full
    summary row reading ``model: null, effort: null, max_budget_usd: null`` —
    which is what turned a wrong-pool load into a plausible-looking table. A
    null-config row is a silent degradation, not a tolerable gap.
    """
    candidates = mod.resolve_candidates(['architect-opus-max'])

    # SystemExit, matching every other loud failure in this module — and note it
    # derives from BaseException, so a bare `except Exception` upstream cannot
    # swallow it back into the silent degradation this guard exists to end.
    with pytest.raises(SystemExit) as exc:
        mod.build_campaign_report([_out_of_campaign_cell()], candidates, [])

    assert exc.value.code != 0
    assert 'architect-sonnet-high' in str(exc.value)
    assert 'architect-opus-max' in str(exc.value)


def test_run_path_results_are_not_filtered(tmp_path, campaign_seam, monkeypatch):
    """Only the UNTRUSTED --results-dir input is filtered; --run is not.

    ``--run`` results are in-campaign by construction, so silently dropping an
    unexpected one there would hide a real runner bug (e.g. a dispatch routing
    the wrong config) behind a clean-looking report. An unexpected cell from the
    seam must therefore surface LOUDLY through the unknown-config guard rather
    than vanish.
    """
    from orchestrator.evals import runner as eval_runner

    async def _returns_a_stray(task_paths, candidates, base_config=None, **kwargs):
        return [_out_of_campaign_cell()]

    monkeypatch.setattr(eval_runner, 'run_ofat_stage', _returns_a_stray)
    d = _make_fixture_dir(tmp_path, ['alpha'])

    with pytest.raises(SystemExit) as exc:
        mod.main([
            '--run', '--tasks-dir', str(d), '--candidate', 'architect-opus-max',
        ])

    assert exc.value.code != 0
    assert 'architect-sonnet-high' in str(exc.value)
    # It surfaced as an unknown-config error, NOT as a quiet drop.
    assert 'candidates' in str(exc.value)


# ===== amendments: degenerate inputs fail at the flag, not in the artifact =====
#
# Every case below was a SILENT no-op before these guards: the driver accepted
# the flags, computed something shaped like a result, and exited 0. That is the
# one failure mode this module exists to make impossible, since γ1's calibration
# is read off the artifact rather than off the terminal.


def test_q_ceiling_without_stage1_exits(tmp_path):
    """A typed-but-ignored threshold is worse than a missing one — it looks applied.

    Verified pre-fix: ``--dry-run ... --q-ceiling 0.8 --out p`` wrote an artifact
    with no ``bands`` key and printed zero diagnostics, so an operator who typed
    the empirically-anchored number had no way to learn it did nothing. The
    reverse direction (``--stage1`` with no ``--q-ceiling``) was already loud;
    that asymmetry is what made this one dangerous.
    """
    d = _make_fixture_dir(tmp_path, ['alpha'])

    with pytest.raises(SystemExit) as exc:
        mod.main([
            '--dry-run', '--tasks-dir', str(d),
            '--candidate', 'architect-opus-max', '--q-ceiling', '0.8',
        ])

    assert exc.value.code != 0
    message = str(exc.value)
    assert '--q-ceiling' in message and '--stage1' in message


def test_stage1_with_dry_run_exits(tmp_path):
    """Banding a dry run would fabricate an all-zero partition over an empty pool.

    Verified pre-fix, ``--dry-run --stage1 --q-ceiling 0.8`` emitted
    ``{'by_fixture': {}, 'counts': {...all 0...}, 'discarded': [],
    'marker_available': False, 'retained': []}`` — precisely the reading
    ``build_campaign_report`` omits ``bands`` to prevent, re-entering through the
    one mode where by construction nothing was measured. ``marker_available:
    False`` is likewise asserted off an empty list.
    """
    d = _make_fixture_dir(tmp_path, ['alpha'])

    with pytest.raises(SystemExit) as exc:
        mod.main([
            '--dry-run', '--tasks-dir', str(d),
            '--candidate', 'architect-opus-max',
            '--stage1', '--q-ceiling', '0.8',
        ])

    assert exc.value.code != 0
    assert '--stage1' in str(exc.value) and '--dry-run' in str(exc.value)


def test_dry_run_artifact_never_carries_bands(tmp_path):
    """Belt and braces behind the flag rejection: no dry-run artifact is ever banded."""
    d = _make_fixture_dir(tmp_path, ['alpha', 'beta'])
    out = tmp_path / 'r.json'

    mod.main([
        '--dry-run', '--tasks-dir', str(d),
        '--candidate', 'architect-opus-max', '--out', str(out),
    ])

    assert 'bands' not in json.loads(out.read_text())


@pytest.mark.parametrize('flag,value', [
    ('--trials', '0'),
    ('--trials', '-1'),
    ('--max-parallel', '0'),
    ('--timeout', '0'),
])
def test_degenerate_int_flags_are_rejected(tmp_path, flag, value, capsys):
    """``--trials 0`` etc. are silent no-ops, not smaller campaigns.

    ``--trials 0`` yields ``range(1, 1)``, so ``enumerate_cells`` returns ``[]``,
    ``run_ofat_stage`` builds zero thunks and ``_bounded_fanout`` short-circuits
    on ``if not thunks: return []`` — a completed "campaign" that dispatched
    nothing and returned 0. ``--max-parallel 0`` constructs
    ``asyncio.Semaphore(0)`` and deadlocks the fan-out instead.
    """
    d = _make_fixture_dir(tmp_path, ['alpha'])

    with pytest.raises(SystemExit) as exc:
        mod.main([
            '--dry-run', '--tasks-dir', str(d),
            '--candidate', 'architect-opus-max', flag, value,
        ])

    assert exc.value.code != 0
    # argparse renders the type error on stderr, naming the flag and the value.
    printed = capsys.readouterr().err
    assert flag in printed
    assert '>= 1' in printed


def test_nonexistent_results_dir_is_diagnosed_as_a_path_problem(tmp_path):
    """A typo'd --results-dir must not be reported as contamination.

    ``Path.glob`` returns EMPTY for a nonexistent dir rather than raising, so
    pre-fix the load fell through to the zero-survivors branch and answered
    "this usually means the dir predates this campaign or belongs to another
    one" — actively misleading for a plain typo. ``resolve_fixture_paths``
    already applies the ``is_dir()`` rule to the other input path.
    """
    d = _make_fixture_dir(tmp_path, ['alpha'])
    missing = tmp_path / 'no' / 'such' / 'dir'

    with pytest.raises(SystemExit) as exc:
        mod.main([
            '--results-dir', str(missing), '--tasks-dir', str(d),
            '--candidate', 'architect-opus-max',
        ])

    assert exc.value.code != 0
    message = str(exc.value)
    assert str(missing) in message
    assert 'not found' in message
    # It is NOT the empty-pool diagnosis: those are different problems with
    # different fixes, and conflating them sends the operator hunting for a
    # contaminated directory that does not exist.
    assert 'out of campaign' not in message


def test_malformed_result_json_names_the_offending_file(tmp_path):
    """One truncated cell among hundreds must name ITSELF, not raise a bare traceback.

    This path is aimed by default at the shared packaged results dir (586 cells
    in the main checkout, written by many campaigns over time), so a single
    partial write there would otherwise take down the whole re-analysis with a
    ``JSONDecodeError`` naming neither the file nor the dir.
    """
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    good = _in_campaign_cell('alpha')
    (results_dir / 'alpha.json').write_text(json.dumps(good.to_dict()))
    truncated = results_dir / 'truncated.json'
    truncated.write_text('{"task_id": "alpha", "config_nam')
    d = _make_fixture_dir(tmp_path, ['alpha'])

    with pytest.raises(SystemExit) as exc:
        mod.main([
            '--results-dir', str(results_dir), '--tasks-dir', str(d),
            '--candidate', 'architect-opus-max',
        ])

    assert exc.value.code != 0
    assert str(truncated) in str(exc.value)


def test_non_dict_result_json_names_the_offending_file(tmp_path):
    """A JSON list/string payload is named too, not surfaced as a bare AttributeError."""
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    stray = results_dir / 'a_list.json'
    stray.write_text('[1, 2, 3]')
    d = _make_fixture_dir(tmp_path, ['alpha'])

    with pytest.raises(SystemExit) as exc:
        mod.main([
            '--results-dir', str(results_dir), '--tasks-dir', str(d),
            '--candidate', 'architect-opus-max',
        ])

    assert exc.value.code != 0
    assert str(stray) in str(exc.value)


# ===== amendments: the enumerated matrix is COMPARED against what returned =====
#
# `cell_matrix` was shipped with the stated justification that "a cell that never
# returned is visible as a gap rather than silently absent" — but nothing
# performed the comparison. `run_ofat_stage` delegates to `_bounded_fanout`,
# which logs a non-cancel cell failure and CONTINUES, so the failed cell is
# simply absent from the returned list; `build_plan_quality_report` then derives
# `configs` only from cells that returned.


def test_missing_cells_are_detected_and_named(tmp_path, campaign_seam, monkeypatch):
    """A short return surfaces as a first-class gap, not as a quietly low ``n``."""
    from orchestrator.evals import runner as eval_runner

    async def _drops_one(task_paths, candidates, base_config=None, **kwargs):
        cells = [
            _cell(path.stem, cfg.name, plan_quality=0.75, plan_steps=5)
            for path in task_paths for cfg in candidates
        ]
        return cells[:-1]  # the last cell "failed" and was logged-and-skipped

    monkeypatch.setattr(eval_runner, 'run_ofat_stage', _drops_one)
    d = _make_fixture_dir(tmp_path, ['alpha', 'beta'])
    out = tmp_path / 'r.json'

    rc = mod.main([
        '--run', '--tasks-dir', str(d),
        '--candidate', 'architect-opus-max', '--out', str(out),
    ])

    # A partial shortfall weakens the comparison but does not destroy it, so it
    # is loud rather than fatal.
    assert rc == 0
    missing = json.loads(out.read_text())['missing_cells']
    assert missing['count'] == 1
    assert missing['expected'] == 2
    assert missing['cells'] == [
        {'task_id': 'beta', 'config_name': 'architect-opus-max', 'trial': 1},
    ]
    assert missing['candidates_absent'] == []


def test_missing_cells_are_rendered_loudly(tmp_path, campaign_seam, monkeypatch, capsys):
    """The shortfall is printed, naming each absent (fixture, candidate, trial)."""
    from orchestrator.evals import runner as eval_runner

    async def _drops_one(task_paths, candidates, base_config=None, **kwargs):
        return [_cell('alpha', 'architect-opus-max', plan_quality=0.75, plan_steps=5)]

    monkeypatch.setattr(eval_runner, 'run_ofat_stage', _drops_one)
    d = _make_fixture_dir(tmp_path, ['alpha', 'beta'])

    mod.main(['--run', '--tasks-dir', str(d), '--candidate', 'architect-opus-max'])

    printed = capsys.readouterr().out
    assert 'MISSING CELLS' in printed
    assert '1 of 2' in printed
    assert 'beta' in printed


def test_an_entirely_absent_candidate_arm_exits_non_zero(
    tmp_path, campaign_seam, monkeypatch, capsys,
):
    """THE gap that most invalidates the screen: a vanished comparison arm.

    Pre-fix, an auth error or model-not-found on one arm produced no row for it
    at all — ``report['candidates']`` silently held one arm,
    ``format_campaign_report`` printed a one-row table, and ``main`` returned 0.
    A one-arm "comparison" is not a weaker result; it is no result.
    """
    from orchestrator.evals import runner as eval_runner

    async def _one_arm_only(task_paths, candidates, base_config=None, **kwargs):
        return [
            _cell(path.stem, 'architect-opus-max', plan_quality=0.75, plan_steps=5)
            for path in task_paths
        ]

    monkeypatch.setattr(eval_runner, 'run_ofat_stage', _one_arm_only)
    d = _make_fixture_dir(tmp_path, ['alpha'])
    out = tmp_path / 'r.json'

    rc = mod.main([
        '--run', '--tasks-dir', str(d),
        '--candidate', 'architect-opus-max',
        '--candidate', 'architect-fable-high',
        '--out', str(out),
    ])

    assert rc != 0
    captured = capsys.readouterr()
    assert 'ENTIRE CANDIDATE ARM ABSENT' in captured.out
    assert 'architect-fable-high' in captured.err
    # The artifact still lands: the diagnosis is the most valuable thing this
    # run produced and must survive the failure signal rather than be replaced
    # by it.
    loaded = json.loads(out.read_text())
    assert loaded['missing_cells']['candidates_absent'] == ['architect-fable-high']
    assert [c['config_name'] for c in loaded['candidates']] == ['architect-opus-max']


def test_missing_cells_absent_from_a_dry_run_artifact(tmp_path):
    """A dry run dispatched nothing, so "every cell is missing" is not a finding.

    Same rule as ``bands``: reporting an all-missing block for a mode that by
    construction measured nothing would be as misleading as suppressing a real
    shortfall.
    """
    d = _make_fixture_dir(tmp_path, ['alpha', 'beta'])
    out = tmp_path / 'r.json'

    mod.main([
        '--dry-run', '--tasks-dir', str(d),
        '--candidate', 'architect-opus-max', '--out', str(out),
    ])

    assert 'missing_cells' not in json.loads(out.read_text())


def test_complete_run_reports_no_missing_cells(tmp_path, campaign_seam):
    """The happy path records an explicit zero — "we compared and found no gaps"."""
    d = _make_fixture_dir(tmp_path, ['alpha', 'beta'])
    out = tmp_path / 'r.json'

    rc = mod.main([
        '--run', '--tasks-dir', str(d), '--candidate', 'architect-opus-max',
        '--trials', '2', '--out', str(out),
    ])

    assert rc == 0
    missing = json.loads(out.read_text())['missing_cells']
    assert missing == {
        'count': 0, 'expected': 4, 'cells': [], 'candidates_absent': [],
    }


def test_results_dir_shortfall_is_detected(tmp_path):
    """The re-analysis path is diffed against the matrix too, not just ``--run``.

    A dir holding only some of the campaign's cells re-analyzes to a report that
    LOOKS complete; the matrix diff is what says otherwise.
    """
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    cell = _in_campaign_cell('alpha')
    (results_dir / 'alpha.json').write_text(json.dumps(cell.to_dict()))
    d = _make_fixture_dir(tmp_path, ['alpha', 'beta'])
    out = tmp_path / 'r.json'

    rc = mod.main([
        '--results-dir', str(results_dir), '--tasks-dir', str(d),
        '--candidate', 'architect-opus-max', '--out', str(out),
    ])

    assert rc == 0
    missing = json.loads(out.read_text())['missing_cells']
    assert missing['count'] == 1
    assert missing['cells'][0]['task_id'] == 'beta'


def test_find_missing_cells_is_pure_and_matches_on_trial(tmp_path):
    """The diff key is (task_id, config_name, trial) — a missing trial is a gap."""
    matrix = mod.enumerate_cells(
        [Path('alpha.json')], ['architect-opus-max'], 2,
    )
    results = [_cell('alpha', 'architect-opus-max', trial=1, plan_steps=5)]
    candidates = mod.resolve_candidates(['architect-opus-max'])

    missing = mod.find_missing_cells(results, matrix, candidates)

    assert missing['count'] == 1
    assert missing['cells'] == [
        {'task_id': 'alpha', 'config_name': 'architect-opus-max', 'trial': 2},
    ]
    # Trial 1 DID return, so the arm is present — the shortfall is partial.
    assert missing['candidates_absent'] == []


# ===== amendments: the last two uncovered seams on the live --run path =====


def test_run_passes_the_timeout_override(tmp_path, campaign_seam):
    """``--timeout`` reaches ``run_ofat_stage`` as ``timeout_override``.

    Asserted alongside the other pass-throughs because a rename or a dropped
    kwarg here would silently un-bound every cell's runtime with no test failing
    — and this is the only flag on the live path whose absence changes nothing
    observable in the report.
    """
    d = _make_fixture_dir(tmp_path, ['alpha'])

    mod.main([
        '--run', '--tasks-dir', str(d),
        '--candidate', 'architect-opus-max', '--timeout', '7',
    ])

    assert campaign_seam['ofat'][0]['timeout_override'] == 7


def test_missing_config_exits_naming_the_flag(tmp_path, monkeypatch):
    """The one error path inside the live-spend seam is covered.

    ``campaign_seam`` stubs ``load_config`` to always succeed, so this branch —
    the only way ``_run_campaign`` fails before spending — needs its own test.
    It must name the remedy (``--config`` / ``ORCH_CONFIG_PATH``) rather than
    surfacing a raw ``ConfigRequiredError``.
    """
    import orchestrator.config as orch_config
    from orchestrator.config import ConfigRequiredError
    from orchestrator.evals import runner as eval_runner

    def _raises(*args, **kwargs):
        raise ConfigRequiredError('no orchestrator config found')

    async def _never(*args, **kwargs):
        pytest.fail('reached run_ofat_stage despite an unresolvable config',
                    pytrace=False)

    monkeypatch.setattr(orch_config, 'load_config', _raises)
    monkeypatch.setattr(eval_runner, 'run_ofat_stage', _never)
    d = _make_fixture_dir(tmp_path, ['alpha'])

    with pytest.raises(SystemExit) as exc:
        mod.main(['--run', '--tasks-dir', str(d), '--candidate', 'architect-opus-max'])

    assert exc.value.code != 0
    assert '--config' in str(exc.value)
    assert 'ORCH_CONFIG_PATH' in str(exc.value)


def test_architect_cells_is_the_single_source_of_the_role_filter():
    """Both the counting and the banding paths consume ONE role filter.

    The filter's docstring argues at length that it must run first everywhere;
    keeping a second inline copy in ``partition_bands`` would mean a change to
    what counts as an architect cell had to be made twice, with only one site
    carrying the argument for why it matters.
    """
    architect = _cell('f1', 'A', plan_steps=5, plan_quality=0.95,
                      extra_metrics={'judged_without_reference': False})
    implementer = _cell('f2', 'A', plan_steps=5, plan_quality=0.95,
                        role_under_test='implementer',
                        extra_metrics={'judged_without_reference': False})

    cells = mod._architect_cells([architect, implementer])

    assert cells == [('f1', 'A', architect.metrics)]
    # And the banding path agrees: the implementer fixture is not banded at all.
    assert mod.partition_bands([architect, implementer], 0.80)['by_fixture'] == {
        'f1': 'ceiling',
    }


def test_count_judged_without_reference_accepts_a_precomputed_map():
    """Passing the map is an optimisation, never a behaviour change."""
    results = _mixed_marker_results()

    unmeasured = mod.count_unmeasured_marker_cells(results)

    assert mod.count_judged_without_reference(results, unmeasured) == \
        mod.count_judged_without_reference(results)
