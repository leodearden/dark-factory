"""What the plan-files touched set counts, measured on real git history.

Every range starts at ``git merge-base HEAD main`` in the worktree: the
own-work base ``branch_base.py::own_work_base`` hands the gate.

Scene A is a back-merge carrying an evil hunk. The merge commit's own change
is the task's work; the main-side file it merely brought in is not. Scene B
is a range git cannot read, which must come back as an error rather than as
an empty touched set. Scene C is a mixed stack: a foreign workstream that
already landed on main sits below the task's own commit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _merged_in_branch import git

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.gates import _check_plan_files_touched_in_branch

pytestmark = pytest.mark.asyncio

_UNRESOLVABLE = 'deadbeef' * 5


@dataclass(frozen=True)
class Range:
    git_ops: GitOps
    base: str
    head: str


def _git_ops(repo: Path) -> GitOps:
    config = GitConfig(
        main_branch='main', branch_prefix='task/', remote='origin',
        worktree_dir='.worktrees', push_after_advance=False,
    )
    return GitOps(config, repo)


def _seed(tmp_path: Path, files: Mapping[str, str]) -> Path:
    return seed_repo(
        tmp_path / 'repo', RepoSeed(files=tuple(files.items()), message='base'),
    )


def _commit(cwd: Path, files: Mapping[str, str], message: str) -> None:
    for rel, content in files.items():
        (cwd / rel).write_text(content)
    git(cwd, 'add', *files)
    git(cwd, 'commit', '-q', '-m', message)


def _own_work_range(repo: Path, worktree: Path) -> Range:
    return Range(
        git_ops=_git_ops(repo),
        base=git(worktree, 'merge-base', 'HEAD', 'main'),
        head=git(worktree, 'rev-parse', 'HEAD'),
    )


def _back_merge_with_evil_hunk(tmp_path: Path) -> Range:
    repo = _seed(tmp_path, {'a.py': 'a = 1\n', 'm.py': 'm = 1\n', 'evil.py': 'e = 1\n'})
    worktree = tmp_path / 'wt'
    git(repo, 'worktree', 'add', '-q', '-b', 'task/50', str(worktree))
    _commit(worktree, {'a.py': 'a = 2\n'}, 'task work')
    _commit(repo, {'m.py': 'm = 2\n'}, 'main advances')
    git(worktree, 'merge', '-q', '--no-commit', '--no-ff', 'main')
    (worktree / 'evil.py').write_text('e = 2\n')
    git(worktree, 'add', 'evil.py')
    git(worktree, 'commit', '-q', '--no-edit')
    return _own_work_range(repo, worktree)


def _mixed_stack(tmp_path: Path) -> Range:
    repo = _seed(tmp_path, {'README.md': '# r\n'})
    foreign = tmp_path / 'wt40'
    git(repo, 'worktree', 'add', '-q', '-b', 'task/40', str(foreign))
    _commit(foreign, {'foreign.py': 'f = 1\n'}, 'foreign workstream')
    git(repo, 'merge', '-q', '--no-ff', '--no-edit', 'task/40')
    worktree = tmp_path / 'wt50'
    git(repo, 'worktree', 'add', '-q', '-b', 'task/50', str(worktree), 'task/40')
    _commit(worktree, {'own.py': 'o = 1\n'}, 'own work')
    return _own_work_range(repo, worktree)


async def _not_touched(scene: Range, plan_files: list[str]) -> list[str]:
    check = await _check_plan_files_touched_in_branch(
        plan_files, scene.base, scene.head, scene.git_ops,
    )
    return check.not_touched


class TestBackMergeWithEvilHunk:
    async def test_the_touched_set_lists_the_merge_commits_own_change_only(
        self, tmp_path: Path,
    ) -> None:
        scene = _back_merge_with_evil_hunk(tmp_path)

        touched = await scene.git_ops.get_files_touched_in_branch(scene.base, scene.head)

        assert touched == (['a.py', 'evil.py'], None)

    async def test_a_file_changed_only_inside_the_merge_commit_is_touched(
        self, tmp_path: Path,
    ) -> None:
        scene = _back_merge_with_evil_hunk(tmp_path)

        assert await _not_touched(scene, ['a.py', 'evil.py']) == []

    async def test_a_main_file_the_back_merge_only_brought_in_is_not_touched(
        self, tmp_path: Path,
    ) -> None:
        scene = _back_merge_with_evil_hunk(tmp_path)

        assert await _not_touched(scene, ['a.py', 'm.py']) == ['m.py']


class TestUnreadableRange:
    async def test_the_touched_set_returns_the_error_instead_of_an_empty_list(
        self, tmp_path: Path,
    ) -> None:
        repo = _seed(tmp_path, {'README.md': '# r\n'})

        files, err = await _git_ops(repo).get_files_touched_in_branch(
            _UNRESOLVABLE, 'HEAD',
        )

        assert files == []
        assert err is not None

    async def test_the_gate_marks_every_declared_entry_unmeasured(
        self, tmp_path: Path,
    ) -> None:
        repo = _seed(tmp_path, {'README.md': '# r\n'})
        head = git(repo, 'rev-parse', 'HEAD')

        check = await _check_plan_files_touched_in_branch(
            ['README.md', '', 'gone.py', 'pkg'], _UNRESOLVABLE, head, _git_ops(repo),
        )

        assert check.touched_set_error is not None
        assert check.not_touched == ['README.md', 'gone.py', 'pkg']
        assert check.missing_from_tree == []


class TestMixedStack:
    async def test_the_tasks_own_commit_is_measured(self, tmp_path: Path) -> None:
        scene = _mixed_stack(tmp_path)

        assert await _not_touched(scene, ['own.py']) == []

    async def test_the_landed_foreign_workstream_is_not_credited(
        self, tmp_path: Path,
    ) -> None:
        scene = _mixed_stack(tmp_path)

        assert await _not_touched(scene, ['own.py', 'foreign.py']) == ['foreign.py']
