"""rereview enumerates its candidates and their run ids through a ResultStore."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from orchestrator.evals.rereview import enumerate_candidates
from orchestrator.evals.results import ResultStore

_FAR_FUTURE = datetime(2100, 1, 1, tzinfo=UTC)


def _intact_worktree(root: Path, name: str) -> Path:
    worktree = root / 'worktrees' / name
    (worktree / '.task').mkdir(parents=True)
    (worktree / '.task' / 'metadata.json').write_text(json.dumps({'base_commit': 'abc123'}))
    return worktree


def _write_done_cell(store: ResultStore, filename: str, task_id: str, worktree: Path) -> None:
    store.directory.mkdir(parents=True, exist_ok=True)
    (store.directory / filename).write_text(json.dumps({
        'task_id': task_id,
        'config_name': 'cfg',
        'outcome': 'done',
        'metrics': {'review_blocking_issues': 1, 'review_suggestions': 2},
        'worktree_path': str(worktree),
    }))


@pytest.fixture
def decoy_packaged_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ResultStore:
    decoy = ResultStore(tmp_path / 'packaged')
    monkeypatch.setattr(ResultStore, 'packaged', classmethod(lambda cls: decoy))
    return decoy


def test_run_ids_come_from_the_store_grammar_then_the_worktree(
    tmp_path: Path, decoy_packaged_store: ResultStore
):
    store = ResultStore(tmp_path / 'results')
    _write_done_cell(
        store, 'df_task_1__cfg__abcd1234.json', 'df_task_1',
        _intact_worktree(tmp_path, 'run-ffffffff'),
    )
    _write_done_cell(
        store, 'df_task_2__cfg.json', 'df_task_2',
        _intact_worktree(tmp_path, 'run-deadbeef'),
    )
    _write_done_cell(
        decoy_packaged_store, 'df_task_3__cfg__0badc0de.json', 'df_task_3',
        _intact_worktree(tmp_path, 'run-0badc0de'),
    )

    candidates = enumerate_candidates(cutoff=_FAR_FUTURE, force=False, store=store)

    assert [(c.task_id, c.run_id) for c in candidates] == [
        ('df_task_1', 'abcd1234'),
        ('df_task_2', 'deadbeef'),
    ]
