"""Real-git worktree states in which a not-touched verdict measured nothing.

Each builder seeds a repo under *tmp_path* with ``main`` checked out and a
task worktree beside it: a rebase stopped on a conflict (tasks 5030/5033),
declared files edited but not committed (task 5027), a task branch re-cut
empty while its work lives on a sibling ref (task 3560), and a task whose
work already landed while a stale pre-rebase copy lingers on a sibling ref.
"""
from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from _git_fixtures import RepoSeed, seed_repo
from _merged_in_branch import git

_IGNORES_TASK_DIR = ('.gitignore', '.task/\n')


@dataclass(frozen=True)
class Scene:
    repo: Path
    worktree: Path
    creation_sha: str
    """main's tip when the task worktree was cut."""


def seeded_repo(tmp_path: Path, files: Mapping[str, str]) -> Path:
    seed = RepoSeed(files=(_IGNORES_TASK_DIR, *files.items()), message='base')
    return seed_repo(tmp_path / 'repo', seed)


def add_worktree(repo: Path, path: Path, branch: str, start: str = 'main') -> Path:
    git(repo, 'worktree', 'add', '-q', '-b', branch, str(path), start)
    return path


def commit(cwd: Path, files: Mapping[str, str], message: str) -> str:
    for rel, content in files.items():
        (cwd / rel).parent.mkdir(parents=True, exist_ok=True)
        (cwd / rel).write_text(content)
    git(cwd, 'add', *files)
    git(cwd, 'commit', '-q', '-m', message)
    return git(cwd, 'rev-parse', 'HEAD')


def stopped_rebase(tmp_path: Path, branch: str) -> Scene:
    """*branch* commits x.py then y.py; main's x.py conflicts, so ``git rebase main`` stops at step 1 of 2."""
    repo = seeded_repo(tmp_path, {'x.py': 'x = 1\n', 'y.py': 'y = 1\n', 'z.py': 'z = 1\n'})
    creation_sha = git(repo, 'rev-parse', 'HEAD')
    worktree = add_worktree(repo, tmp_path / 'wt', branch)
    commit(worktree, {'x.py': 'x = 2\n'}, 'task edits x')
    commit(worktree, {'y.py': 'y = 2\n'}, 'task edits y')
    commit(repo, {'x.py': 'x = 3\n'}, 'main edits x')
    stopped = subprocess.run(
        ['git', 'rebase', 'main'], cwd=worktree, capture_output=True, text=True,
    )
    assert stopped.returncode != 0, stopped.stdout
    return Scene(repo=repo, worktree=worktree, creation_sha=creation_sha)


def uncommitted_edits(tmp_path: Path) -> Scene:
    """a.py modified unstaged, new.py untracked, pkg/inner.py new and staged, b.py clean."""
    repo = seeded_repo(tmp_path, {'a.py': 'a = 1\n', 'b.py': 'b = 1\n'})
    worktree = add_worktree(repo, tmp_path / 'wt', 'task/50')
    (worktree / 'a.py').write_text('a = 2\n')
    (worktree / 'new.py').write_text('n = 1\n')
    (worktree / 'pkg').mkdir()
    (worktree / 'pkg' / 'inner.py').write_text('i = 1\n')
    git(worktree, 'add', 'pkg/inner.py')
    return Scene(repo=repo, worktree=worktree, creation_sha=git(repo, 'rev-parse', 'HEAD'))


@dataclass(frozen=True)
class SiblingScene:
    repo: Path
    worktree: Path
    creation_sha: str
    sibling_ref: str
    sibling_tip: str


def work_on_sibling_ref(tmp_path: Path) -> SiblingScene:
    """task/50 is cut from main with no commits; task/50-skip-attempt edits a.py, task/500 edits b.py."""
    repo = seeded_repo(tmp_path, {'a.py': 'a = 1\n', 'b.py': 'b = 1\n'})
    worktree = add_worktree(repo, tmp_path / 'wt', 'task/50')
    skip = add_worktree(repo, tmp_path / 'wt-skip', 'task/50-skip-attempt')
    sibling_tip = commit(skip, {'a.py': 'a = 2\n'}, 'skip-attempt work')
    other = add_worktree(repo, tmp_path / 'wt-500', 'task/500')
    commit(other, {'b.py': 'b = 2\n'}, 'another task')
    return SiblingScene(
        repo=repo, worktree=worktree, creation_sha=git(repo, 'rev-parse', 'HEAD'),
        sibling_ref='task/50-skip-attempt', sibling_tip=sibling_tip,
    )


def landed_with_pre_rebase_sibling(tmp_path: Path) -> SiblingScene:
    """task/50's a.py edit landed on main as 'Merge task/50 into main' and main has moved on, so its pre-rebase copy on task/50-20261101T000000Z no longer matches main."""
    repo = seeded_repo(tmp_path, {'a.py': 'a = 1\n'})
    sibling_ref = 'task/50-20261101T000000Z'
    pre_rebase = add_worktree(repo, tmp_path / 'wt-pre-rebase', sibling_ref)
    sibling_tip = commit(pre_rebase, {'a.py': 'a = 2\n'}, 'pre-rebase task work')
    landing = add_worktree(repo, tmp_path / 'wt-landing', 'landing')
    commit(landing, {'a.py': 'a = 2\n'}, 'task work')
    git(repo, 'merge', '-q', '--no-ff', '-m', 'Merge task/50 into main', 'landing')
    commit(repo, {'a.py': 'a = 2\nextra = 1\n'}, 'main extends a.py')
    creation_sha = git(repo, 'rev-parse', 'HEAD')
    worktree = add_worktree(repo, tmp_path / 'wt', 'task/50')
    return SiblingScene(
        repo=repo, worktree=worktree, creation_sha=creation_sha,
        sibling_ref=sibling_ref, sibling_tip=sibling_tip,
    )
