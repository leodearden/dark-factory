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

import pytest

import run_fable_trial_v2_campaign as mod


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
