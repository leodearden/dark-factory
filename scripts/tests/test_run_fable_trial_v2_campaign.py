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
