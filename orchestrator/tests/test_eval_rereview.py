"""rereview enumerates its candidates, and patches their cells, through a ResultStore."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from orchestrator.evals import rereview
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


@pytest.fixture
def fake_reviewer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The opus reviewer finds 0 blocking issues and 3 suggestions."""

    async def diff(*_args) -> str:
        return 'diff --git a/x b/x'

    async def review(*_args, **_kwargs):
        issues = [{'severity': 'suggestion'}] * 3
        return 'opus', {'verdict': 'PASS', 'issues': issues}, 2.5

    monkeypatch.setattr(rereview, 'get_diff', diff)
    monkeypatch.setattr(rereview, '_run_single_reviewer', review)
    monkeypatch.setattr(rereview, 'RE_REVIEWS_DIR', tmp_path / 're_reviews')
    monkeypatch.setattr(rereview, 'PROGRESS_FILE', tmp_path / 're_reviews' / 'p.jsonl')


def _one_candidate(tmp_path: Path, store: ResultStore) -> rereview.Candidate:
    _write_done_cell(
        store, 'df_task_1__cfg__abcd1234.json', 'df_task_1',
        _intact_worktree(tmp_path, 'run-abcd1234'),
    )
    [candidate] = enumerate_candidates(cutoff=_FAR_FUTURE, force=False, store=store)
    return candidate


@pytest.mark.usefixtures('fake_reviewer')
def test_a_rereview_patches_its_cell_and_keeps_the_rest(tmp_path: Path):
    store = ResultStore(tmp_path / 'results')
    candidate = _one_candidate(tmp_path, store)
    cell = json.loads(candidate.result_path.read_text())
    cell['retired_field'] = 'kept'
    candidate.result_path.write_text(json.dumps(cell))

    outcome = asyncio.run(rereview.rereview_one(candidate, store=store))

    assert outcome.status == 'done'
    patched = json.loads(candidate.result_path.read_text())
    assert patched['retired_field'] == 'kept'
    assert patched['metrics']['legacy_sonnet_review_blocking_issues'] == 1
    assert patched['metrics']['legacy_sonnet_review_suggestions'] == 2
    assert patched['metrics']['review_blocking_issues'] == 0
    assert patched['metrics']['review_suggestions'] == 3


@pytest.mark.usefixtures('fake_reviewer')
def test_a_cell_gone_unreadable_before_its_patch_is_an_error_outcome(tmp_path: Path):
    store = ResultStore(tmp_path / 'results')
    candidate = _one_candidate(tmp_path, store)
    candidate.result_path.write_text('{"task_id": "df_task_1", "conf')

    outcome = asyncio.run(rereview.rereview_one(candidate, store=store))

    assert outcome.status == 'error'
    assert str(candidate.result_path) in outcome.message
    assert candidate.result_path.read_text() == '{"task_id": "df_task_1", "conf'
