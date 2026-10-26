"""orchestrator/scripts/backfill_composite_scores.py re-scores cells through the store."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from orchestrator.evals.metrics import EvalMetrics, compute_composite
from orchestrator.evals.results import EvalResult, ResultStore

_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'backfill_composite_scores.py'
_OLD_MTIME_NS = 1_000_000_000


def _backfill_main():
    spec = importlib.util.spec_from_file_location('backfill_composite_scores', _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _cell(task_id: str, metrics: dict) -> EvalResult:
    return EvalResult(
        task_id=task_id, config_name='cfg', outcome='done', metrics=metrics,
        worktree_path='/fake/wt', run_id='abcd1234',
    )


@pytest.fixture
def packaged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ResultStore:
    store = ResultStore(tmp_path / 'results')
    monkeypatch.setattr(ResultStore, 'packaged', classmethod(lambda cls: store))
    return store


def test_only_a_stale_score_is_rewritten(
    packaged: ResultStore, capsys: pytest.CaptureFixture[str]
):
    scored = {'tests_pass': True, 'plan_steps': 4, 'review_blocking_issues': 1}
    current_score = compute_composite(EvalMetrics(**scored))
    stale = packaged.save(_cell('df_task_1', {**scored, 'composite_score': 0.0}))
    stale_cell = json.loads(stale.read_text())
    stale_cell['retired_field'] = 'kept'
    stale.write_text(json.dumps(stale_cell))
    current = packaged.save(
        _cell('df_task_2', {**scored, 'composite_score': current_score})
    )
    empty = packaged.save(_cell('df_task_3', {}))
    for path in (current, empty):
        os.utime(path, ns=(_OLD_MTIME_NS, _OLD_MTIME_NS))

    assert _backfill_main()() == 0

    rescored = json.loads(stale.read_text())
    assert rescored['metrics']['composite_score'] == current_score
    assert rescored['retired_field'] == 'kept'
    assert [p.stat().st_mtime_ns for p in (current, empty)] == [_OLD_MTIME_NS] * 2
    assert capsys.readouterr().out.splitlines() == [
        'Scanned: 3', 'Updated: 1', 'Unchanged: 2',
    ]
