"""Tests for orchestrator.mcp.foreign_task_meta (task 4825).

The module finds the paths a plan's prose names inside another lane's or
task's ``.task-meta`` directory, and attaches them to a plan-tools response.
Every case runs against a REAL linked worktree, so the write set is derived
from a real ``.git`` gitdir file. The lane is named ``lane`` while the task
is 4825, the production shape where a lane name differs from its task id.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from _git_checkouts import main_checkout_with_lane

from orchestrator.agents.write_set import WriteSet, compute_write_set
from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp.foreign_task_meta import (
    FOREIGN_TASK_META_KEY,
    FOREIGN_TASK_META_NOTE,
    annotate,
    foreign_task_meta_paths,
    step_findings,
)


@pytest.fixture
def wt(tmp_path: Path) -> Path:
    return main_checkout_with_lane(tmp_path)[1]


@pytest.fixture
def tmp_home(tmp_path: Path) -> Path:
    home = tmp_path / 'home'
    home.mkdir()
    return home


@pytest.fixture
def ws(wt: Path, tmp_home: Path) -> WriteSet:
    return compute_write_set(wt, task_id='4825', home=tmp_home)


@pytest.fixture
def meta_base(wt: Path) -> str:
    return str(wt.parent / '.task-meta')


class TestForeignTaskMetaPaths:
    def test_a_backticked_sibling_plan_is_reported_as_written(self, ws: WriteSet, meta_base: str):
        text = f'cp `{meta_base}/4762/plan.json` aside.'
        assert foreign_task_meta_paths(text, ws) == (f'{meta_base}/4762/plan.json',)

    @pytest.mark.parametrize('punctuation', ['.', ':', '!', '?'])
    def test_trailing_sentence_punctuation_is_stripped(
        self, ws: WriteSet, meta_base: str, punctuation: str,
    ):
        text = f'Then read {meta_base}/4762/plan.json{punctuation}'
        assert foreign_task_meta_paths(text, ws) == (f'{meta_base}/4762/plan.json',)

    def test_a_worktree_relative_dotdot_path_is_reported_as_written(self, ws: WriteSet):
        text = 'cp ../.task-meta/4762/plan.json x'
        assert foreign_task_meta_paths(text, ws) == ('../.task-meta/4762/plan.json',)

    def test_a_home_relative_path_is_reported_as_written(
        self, ws: WriteSet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv('HOME', str(tmp_path))
        text = 'cp ~/.task-meta/4762/plan.json x'
        assert foreign_task_meta_paths(text, ws) == ('~/.task-meta/4762/plan.json',)

    def test_distinct_paths_are_reported_once_each_in_first_seen_order(
        self, ws: WriteSet, meta_base: str,
    ):
        plan = f'{meta_base}/4762/plan.json'
        superseded = f'{meta_base}/4762/plan.superseded-by-4822.json'
        text = f'5a: ls -la {plan}\n5d: cp {plan} {superseded}'
        assert foreign_task_meta_paths(text, ws) == (plan, superseded)

    def test_the_plans_own_lane_and_task_dirs_are_not_reported(
        self, ws: WriteSet, meta_base: str,
    ):
        text = f'write {meta_base}/lane/plan.json and {meta_base}/task-4825/plan.json'
        assert foreign_task_meta_paths(text, ws) == ()

    def test_a_dir_named_for_the_task_id_but_not_its_lane_is_reported(
        self, ws: WriteSet, meta_base: str,
    ):
        assert foreign_task_meta_paths(f'write {meta_base}/4825/plan.json', ws) == (
            f'{meta_base}/4825/plan.json',
        )

    def test_an_unresolvable_token_is_skipped_and_its_neighbours_still_checked(
        self, ws: WriteSet, meta_base: str,
    ):
        text = f'{meta_base}/47\x0062/plan.json then {meta_base}/4762/plan.json'
        assert foreign_task_meta_paths(text, ws) == (f'{meta_base}/4762/plan.json',)

    @pytest.mark.parametrize('shape', [
        'see `.worktrees/.task-meta/4762/plan.json`',
        'a fixture at tmp/.task-meta/L',
        'a fixture at <base>/.task-meta/other/x',
        'the fixture /w/.task-meta/task-7',
        'copy {meta_base}/<task>/verdicts/x.json in',
        'edit /home/x/orchestrator/foo.py',
    ])
    def test_measured_false_positive_shapes_are_not_reported(
        self, ws: WriteSet, meta_base: str, shape: str,
    ):
        assert foreign_task_meta_paths(shape.format(meta_base=meta_base), ws) == ()


class TestStepFindings:
    def _entries(self, meta_base: str) -> list[object]:
        return [
            {'id': 'step-1', 'description': f'cp {meta_base}/4762/plan.json aside'},
            {'id': 'step-2', 'description': 'clean'},
            'not-a-dict',
            {'id': 'step-3', 'description': None},
        ]

    def test_only_entries_naming_a_foreign_path_are_reported(self, wt: Path, meta_base: str):
        artifacts = TaskArtifacts.for_task(wt, '4825')
        assert step_findings(artifacts, self._entries(meta_base)) == {
            'step-1': [f'{meta_base}/4762/plan.json'],
        }

    def test_a_worktree_with_no_linked_git_fails_open(self, tmp_path: Path, meta_base: str):
        worktree = tmp_path / 'no-git-wt'
        worktree.mkdir()
        assert step_findings(TaskArtifacts(worktree), self._entries(meta_base)) == {}

    @pytest.mark.skipif(os.geteuid() == 0, reason='root reads a mode-000 file')
    def test_an_unreadable_git_file_fails_open(self, wt: Path, meta_base: str):
        (wt / '.git').chmod(0)
        artifacts = TaskArtifacts.for_task(wt, '4825')
        assert step_findings(artifacts, self._entries(meta_base)) == {}


class TestAnnotate:
    def test_no_findings_leaves_the_response_without_the_key(self):
        assert annotate({'status': 'ok'}, {}) == {'status': 'ok'}

    def test_findings_attach_the_paths_and_the_note(self):
        response = annotate({'status': 'ok'}, {'step-1': ['p']})
        assert response[FOREIGN_TASK_META_KEY] == {
            'paths': {'step-1': ['p']},
            'note': FOREIGN_TASK_META_NOTE,
        }
