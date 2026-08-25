"""Real-git tests for ``GitOps.net_diff_is_empty`` — the no-op landing primitive (task 4647).

PRD "landed-not-done-recovery" Open question 2, decided: the empty-net-diff
check is a STANDALONE tri-state primitive in ``git_ops`` rather than logic
inlined in ``landing_evidence.branch_work_landed``.  Nothing under
``orchestrator/src``, ``escalation/src`` or ``shared/src`` implemented such a
predicate before this task, so there is nothing here to reuse and nothing to
collide with; the nearest neighbours are ``branch_content_in_main``
(byte-identity containment) and ``_vacuous_path_survives`` (a zero-added-lines
survival arm), and neither asks about net emptiness.

Every case runs against a REAL temporary repository.  The whole point of the
primitive is what ``git merge-base`` and ``git diff --quiet`` actually return
for merge commits, root commits and disconnected histories, so a mocked
``_run`` would only re-assert this file's assumptions about them.  Fixture
scaffolding is module-local and modelled on ``test_git_ops.py``'s
``git_repo`` / ``git_config`` / ``git_ops`` triple, with both esc-3072-3
isolation guards applied: ``assert_isolated_git_repo`` runs before any
subprocess, and ``git_env_with_ceiling`` caps git's upward repo discovery at
the fixture root.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from _orch_helpers import (
    NonIsolatedGitRepoError,
    assert_isolated_git_repo,
    git_env_with_ceiling,
)

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps


def _numbered(prefix: str, count: int, *, start: int = 0) -> str:
    """Return *count* unique, non-blank lines (see test_git_ops.py's twin)."""
    return ''.join(f'{prefix}_{i:04d} = {i * 7919}\n' for i in range(start, start + count))


class _Repo:
    """A synthetic git repository with the isolation guards always applied."""

    def __init__(self, root: Path) -> None:
        # FIRST, before any subprocess: a rejected root must write nothing.
        assert_isolated_git_repo(root)
        self.root = root
        self._env = git_env_with_ceiling(root)

    @classmethod
    def init(cls, root: Path) -> _Repo:
        """``git init`` *root* and return the guarded wrapper for it.

        The pre-flight refuses any directory that is not ALREADY a repo root,
        which is exactly what this call creates, so the ceiling environment is
        applied to the bootstrap invocation on its own.  ``git init`` only
        writes into its own ``cwd``, so no upward escape is possible even in
        the one call the pre-flight cannot cover.
        """
        proc = subprocess.run(
            ['git', 'init', '-b', 'main'], cwd=str(root), capture_output=True,
            env=git_env_with_ceiling(root), text=True, check=False,
        )
        assert proc.returncode == 0, f'git init failed: {proc.stderr.strip()}'
        return cls(root)

    def git(self, *args: str, check: bool = True) -> str:
        proc = subprocess.run(
            ['git', *args], cwd=str(self.root), capture_output=True,
            env=self._env, text=True, check=False,
        )
        if check:
            assert proc.returncode == 0, (
                f'git {" ".join(args)} failed (rc={proc.returncode}): '
                f'{proc.stderr.strip()}'
            )
        return proc.stdout.strip()

    def git_rc(self, *args: str) -> tuple[int, str]:
        proc = subprocess.run(
            ['git', *args], cwd=str(self.root), capture_output=True,
            env=self._env, text=True, check=False,
        )
        return proc.returncode, proc.stdout.strip()

    def write(self, rel: str, content: str) -> None:
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def commit(self, message: str, files: dict[str, str] | None = None) -> str:
        for rel, content in (files or {}).items():
            self.write(rel, content)
        self.git('add', '-A')
        self.git('commit', '-m', message)
        return self.sha('HEAD')

    def commit_empty(self, message: str) -> str:
        self.git('commit', '--allow-empty', '-m', message)
        return self.sha('HEAD')

    def sha(self, ref: str) -> str:
        return self.git('rev-parse', ref)

    def parents(self, ref: str) -> list[str]:
        return self.git('rev-list', '--parents', '-n', '1', ref).split()[1:]


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A temporary git repository with an initial commit on ``main``."""
    root = tmp_path / 'repo'
    root.mkdir()
    repo = _Repo.init(root)
    repo.git('config', 'user.email', 'test@test.com')
    repo.git('config', 'user.name', 'Test')
    repo.commit('chore: initial commit', {'README.md': '# Test\n'})
    return root


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def repo(git_repo: Path) -> _Repo:
    """The guarded command wrapper for the same repository ``git_ops`` uses."""
    return _Repo(git_repo)


def make_disconnected_root(repo: _Repo, *, branch: str = 'orphan') -> str:
    """An unrelated ROOT history sharing no ancestor with main.

    ``git merge-base`` exits non-zero with no output for such a pair, which is
    the third way the primitive can fail to determine an answer.
    """
    repo.git('checkout', '--orphan', branch)
    repo.git('rm', '-rf', '--cached', '.')
    for path in repo.root.iterdir():
        if path.name != '.git':
            if path.is_dir():
                subprocess.run(['rm', '-rf', str(path)], check=True)
            else:
                path.unlink()
    sha = repo.commit('chore: unrelated root', {'ORPHAN.md': _numbered('orphan', 4)})
    repo.git('checkout', 'main')
    return sha


class TestNoOpFixtures:
    """The scaffolding builds the git shapes the tri-state cases need.

    Git state only — no symbol from this task — so a later failure in the
    primitive's own cases can never be blamed on a fixture that silently
    stopped building the shape it claims.
    """

    def test_repo_starts_on_main_with_one_commit(self, repo: _Repo) -> None:
        assert repo.git('rev-parse', '--abbrev-ref', 'HEAD') == 'main'
        assert repo.parents('main') == [], 'the initial commit is a root commit'

    def test_empty_commit_has_no_net_diff(self, repo: _Repo) -> None:
        base = repo.sha('main')
        repo.git('checkout', '-b', 'task/empty')
        tip = repo.commit_empty('chore: an empty commit')
        assert repo.git_rc('diff', '--quiet', base, tip)[0] == 0

    def test_added_file_has_a_net_diff(self, repo: _Repo) -> None:
        base = repo.sha('main')
        repo.git('checkout', '-b', 'task/adds')
        tip = repo.commit('feat: add', {'pkg/f.py': _numbered('f', 12)})
        assert repo.git_rc('diff', '--quiet', base, tip)[0] == 1

    def test_disconnected_roots_have_no_merge_base(self, repo: _Repo) -> None:
        orphan = make_disconnected_root(repo)
        rc, out = repo.git_rc('merge-base', 'main', orphan)
        assert rc != 0 and out == '', f'expected no merge-base, got rc={rc} out={out!r}'

    def test_fixtures_never_touch_the_enclosing_checkout(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        assert git_ops.project_root == git_repo
        assert (git_repo / '.git').exists(), 'the fixture root must BE a repo root'
        nested = git_repo / 'pkg'
        nested.mkdir(exist_ok=True)
        with pytest.raises(NonIsolatedGitRepoError):
            _Repo(nested)
        assert 'GIT_CEILING_DIRECTORIES' in _Repo(git_repo)._env
