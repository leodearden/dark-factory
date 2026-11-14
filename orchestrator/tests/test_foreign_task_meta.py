"""Tests for orchestrator.mcp.foreign_task_meta (task 4825).

The module finds the absolute paths a plan's prose names inside another
lane's or task's ``.task-meta`` directory, and attaches them to a
plan-tools response. Every case runs against a REAL linked worktree, so
the write set is derived from a real ``.git`` gitdir file.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from orchestrator.agents.write_set import WriteSet, compute_write_set
from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp.foreign_task_meta import (
    FOREIGN_TASK_META_KEY,
    FOREIGN_TASK_META_NOTE,
    annotate,
    foreign_task_meta_paths,
    step_findings,
)


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def wt(tmp_path: Path) -> Path:
    main = tmp_path / 'main-repo'
    main.mkdir()
    _git(['init', '-b', 'main'], main)
    _git(['config', 'user.email', 'test@test.com'], main)
    _git(['config', 'user.name', 'Test'], main)
    (main / 'README.md').write_text('# test\n')
    _git(['add', '-A'], main)
    _git(['commit', '-m', 'initial commit'], main)
    base = tmp_path / 'worktrees'
    base.mkdir()
    worktree = base / '4825'
    _git(['worktree', 'add', '-b', 'task/4825', str(worktree), 'main'], main)
    return worktree


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

    @pytest.mark.parametrize('punctuation', ['.', ':'])
    def test_trailing_sentence_punctuation_is_stripped(
        self, ws: WriteSet, meta_base: str, punctuation: str,
    ):
        text = f'Then read {meta_base}/4762/plan.json{punctuation}'
        assert foreign_task_meta_paths(text, ws) == (f'{meta_base}/4762/plan.json',)

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
        text = f'write {meta_base}/4825/plan.json and {meta_base}/task-4825/plan.json'
        assert foreign_task_meta_paths(text, ws) == ()

    @pytest.mark.parametrize('shape', [
        'see `.worktrees/.task-meta/4762/plan.json`',
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


class TestAnnotate:
    def test_no_findings_leaves_the_response_without_the_key(self):
        assert annotate({'status': 'ok'}, {}) == {'status': 'ok'}

    def test_findings_attach_the_paths_and_the_note(self):
        response = annotate({'status': 'ok'}, {'step-1': ['p']})
        assert response[FOREIGN_TASK_META_KEY] == {
            'paths': {'step-1': ['p']},
            'note': FOREIGN_TASK_META_NOTE,
        }
