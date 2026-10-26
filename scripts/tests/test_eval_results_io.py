"""scripts/_eval_results_io.py reads a ResultStore's cells, or exits naming what it cannot.

The scripts loader's half of the B1 boundary case: cells written by
``orchestrator/src/orchestrator/evals/results.py::ResultStore.save`` read back
through :func:`load_results_from_dir`, and a bad cell becomes a ``SystemExit``
naming it. The ``orchestrator`` import is deferred because importing
``_eval_results_io`` is what puts ``orchestrator/src`` on ``sys.path``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from _eval_results_io import load_results_from_dir


def _store_with_one_cell(directory: Path):
    from orchestrator.evals.results import EvalResult, ResultStore

    store = ResultStore(directory)
    cell = EvalResult(
        task_id='df_task_b1', config_name='claude-sonnet-max', outcome='done',
        metrics={'composite_score': 0.9, 'tests_pass': True},
        worktree_path='/fake/wt', wall_clock_ms=1200, run_id='run-b1',
    )
    store.save(cell)
    return cell


def test_a_store_written_cell_reads_back(tmp_path: Path):
    cell = _store_with_one_cell(tmp_path / 'results')
    assert load_results_from_dir(tmp_path / 'results') == [cell]


def test_a_truncated_cell_beside_a_good_one_exits_naming_it(tmp_path: Path):
    _store_with_one_cell(tmp_path / 'results')
    truncated = tmp_path / 'results' / 'zz_truncated.json'
    truncated.write_text('{"task_id": "x", "conf')

    with pytest.raises(SystemExit) as excinfo:
        load_results_from_dir(tmp_path / 'results')

    assert str(truncated) in str(excinfo.value)


def test_a_missing_dir_exits_naming_it(tmp_path: Path):
    missing = tmp_path / 'no' / 'such' / 'dir'

    with pytest.raises(SystemExit) as excinfo:
        load_results_from_dir(missing)

    message = str(excinfo.value)
    assert str(missing) in message
    assert '--results-dir' in message
