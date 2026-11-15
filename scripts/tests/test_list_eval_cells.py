"""scripts/list_eval_cells.py prints one task's cells as ResultStore names them.

Cells are real files written by
``orchestrator/src/orchestrator/evals/results.py::ResultStore.save`` under
``tmp_path``. ``list_eval_cells`` is imported first because importing it is what
puts ``orchestrator/src`` on ``sys.path``.
"""
from __future__ import annotations

import json
from pathlib import Path

import list_eval_cells
import pytest
from orchestrator.evals.results import EvalResult, ResultStore


def _cell(task_id: str, config_name: str, run_id: str = 'r1') -> EvalResult:
    return EvalResult(
        task_id=task_id, config_name=config_name, outcome='done',
        metrics={'tests_pass': True, 'role_under_test': 'implementer'},
        worktree_path='/fake/wt', run_id=run_id,
    )


def _printed_paths(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return capsys.readouterr().out.splitlines()


def test_prints_each_cell_of_the_task_one_per_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    directory = tmp_path / 'results'
    store = ResultStore(directory)
    saved = [store.save(_cell('df_task_12', 'cfg-b')), store.save(_cell('df_task_12', 'cfg-a'))]
    store.save(_cell('df_task_1', 'cfg-a'))
    payload = json.dumps(_cell('df_task_12', 'cfg').to_dict())
    (directory / 'df_task_12__cfg.json').write_text(payload)
    (directory / 'df_task_12__cfg__r__x.json').write_text(payload)

    assert list_eval_cells.main(['df_task_12', '--results-dir', str(directory)]) == 0

    expected = [str(p) for p in ResultStore(directory).paths_for_task('df_task_12')]
    assert _printed_paths(capsys) == expected
    assert sorted(expected) == sorted(str(p) for p in saved)


@pytest.mark.parametrize('exists', [True, False], ids=['empty-dir', 'missing-dir'])
def test_a_task_with_no_cells_prints_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], exists: bool
):
    directory = tmp_path / 'results'
    if exists:
        directory.mkdir()

    assert list_eval_cells.main(['df_task_12', '--results-dir', str(directory)]) == 0
    assert capsys.readouterr().out == ''


def test_the_default_store_is_the_packaged_one():
    assert list_eval_cells.store_for(None) == ResultStore.packaged()


def test_an_explicit_dir_builds_its_own_store(tmp_path: Path):
    assert list_eval_cells.store_for(tmp_path) == ResultStore(tmp_path)
