"""orchestrator/src/orchestrator/dispatch_base.py against a real GitOps over a real repo (task 7736)."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import pytest
from _workflow_helpers import _init_repo

from orchestrator.config import GitConfig
from orchestrator.dispatch_base import BaseConflict, find_base_conflict
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.outcome_facts import ConflictEvidence

GREET_FILE = 'lib.py'


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def _commit_file(cwd: Path, name: str, content: str, message: str) -> str:
    (cwd / name).write_text(content)
    _git(cwd, 'add', name)
    _git(cwd, 'commit', '-m', message)
    return _git(cwd, 'rev-parse', 'HEAD')


def _greeting(word: str) -> str:
    return f'def greet(name: str) -> str:\n    return f"{word}, {{name}}"\n'


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / 'repo'
    path.mkdir()
    _init_repo(path)
    return path


@pytest.fixture
def git_ops(repo: Path) -> GitOps:
    return GitOps(
        GitConfig(main_branch='main', branch_prefix='task/', remote='origin', worktree_dir='.worktrees'),
        repo,
    )


@pytest.fixture
async def worktree(git_ops: GitOps) -> Path:
    return (await git_ops.create_worktree('42')).path


async def _conflicting_scene(repo: Path, worktree: Path) -> None:
    _commit_file(worktree, GREET_FILE, _greeting('Hi'), 'task: greet with Hi')
    _commit_file(repo, GREET_FILE, _greeting('Hey'), 'main: greet with Hey')


@pytest.mark.asyncio
class TestFindBaseConflict:
    async def test_same_line_edited_on_both_sides_is_a_conflict(self, repo, git_ops, worktree):
        await _conflicting_scene(repo, worktree)

        conflict = await find_base_conflict(git_ops, worktree)

        assert conflict is not None
        assert conflict.main_sha == _git(repo, 'rev-parse', 'main')
        assert conflict.head_sha == _git(worktree, 'rev-parse', 'HEAD')
        assert conflict.conflicted_paths == (GREET_FILE,)

    async def test_branch_behind_main_on_another_file_is_clean(self, repo, git_ops, worktree):
        _commit_file(worktree, 'notes.txt', 'task notes\n', 'task: notes')
        _commit_file(repo, GREET_FILE, _greeting('Hey'), 'main: greet with Hey')

        assert await find_base_conflict(git_ops, worktree) is None

    async def test_branch_without_own_commits_is_clean(self, repo, git_ops, worktree):
        _commit_file(repo, GREET_FILE, _greeting('Hey'), 'main: greet with Hey')

        assert await find_base_conflict(git_ops, worktree) is None

    async def test_event_payload_spells_evidence_as_merge_events_do(self, repo, git_ops, worktree):
        await _conflicting_scene(repo, worktree)

        conflict = await find_base_conflict(git_ops, worktree)

        assert conflict is not None
        assert conflict.to_event_payload() == {
            'main_sha': conflict.main_sha,
            'head_sha': conflict.head_sha,
            **ConflictEvidence(conflicted_paths=(GREET_FILE,), counterpart_task_ids=()).to_event_payload(),
        }
        assert conflict.to_event_payload()['counterpart_task_ids'] == []


@pytest.mark.parametrize('field_name', ['main_sha', 'head_sha', 'conflicted_paths'])
def test_base_conflict_is_frozen(field_name):
    conflict = BaseConflict(main_sha='a' * 40, head_sha='b' * 40, conflicted_paths=(GREET_FILE,))

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(conflict, field_name, 'other')
