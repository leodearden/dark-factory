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


class TestNetDiffIsEmpty:
    """``GitOps.net_diff_is_empty(upstream, head)`` — the no-op predicate.

    The question is "does *head* contribute any NET change relative to where
    it forked from *upstream*?", i.e. is ``merge-base(upstream, head)..head``
    empty.  Deliberately NOT the same question as
    ``branch_content_in_main``'s byte-identity containment, and deliberately
    TRI-STATE.
    """

    async def test_branch_that_adds_a_file_is_not_a_no_op(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        repo.git('checkout', '-b', 'task/adds')
        repo.commit('feat: add the feature', {'pkg/f.py': _numbered('f', 20)})
        assert await git_ops.net_diff_is_empty('main', 'task/adds') is False

    async def test_no_op_merge_on_main_is_a_no_op(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """The task-1175 shape: real commits, empty net contribution.

        The branch adds the feature and takes it out again, so the merge
        marker on main is genuine while the deliverable is nothing.
        """
        base = repo.sha('main')
        repo.git('checkout', '-b', 'task/noop')
        repo.commit('feat: add the feature', {'pkg/f.py': _numbered('f', 20)})
        repo.git('rm', '-q', 'pkg/f.py')
        repo.git('commit', '-m', 'fix: back it out again')
        tip = repo.sha('HEAD')
        repo.git('checkout', 'main')
        repo.git('merge', '--no-ff', 'task/noop', '-m', 'Merge task/noop into main')

        assert await git_ops.net_diff_is_empty(base, tip) is True

    async def test_empty_commit_is_a_no_op(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """A branch whose only commit is empty contributes genuinely nothing."""
        base = repo.sha('main')
        repo.git('checkout', '-b', 'task/empty')
        repo.commit_empty('chore: an empty commit')
        assert await git_ops.net_diff_is_empty(base, 'task/empty') is True

    async def test_unresolvable_head_is_none_not_false(
        self, git_ops: GitOps,
    ) -> None:
        """THE tri-state assertion. ``None``, never ``False``.

        A ``False`` here would be laundered by the caller into "the branch has
        real content", and a ``True`` into "the task delivered nothing" — both
        of them a git FAILURE silently re-decided as a fact about the task.
        ``branch_work_landed`` maps ``None`` to ``git_error`` for exactly this
        reason, which a bool return could not express.
        """
        assert await git_ops.net_diff_is_empty('main', 'refs/heads/does-not-exist') is None

    async def test_unresolvable_upstream_is_none(self, git_ops: GitOps) -> None:
        assert await git_ops.net_diff_is_empty('refs/heads/nope', 'main') is None

    async def test_uncomputable_merge_base_is_none(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """Two disconnected root histories have no merge-base at all."""
        orphan = make_disconnected_root(repo)
        assert await git_ops.net_diff_is_empty('main', orphan) is None

    async def test_root_commit_head_does_not_raise_and_is_determinate(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """A parentless head must not fall off the parent-reading path."""
        root = repo.git('rev-list', '--max-parents=0', 'main').strip()
        assert repo.parents(root) == []
        result = await git_ops.net_diff_is_empty(root, root)
        assert result is True, 'a commit compared against itself is empty'
        assert result is not None

    async def test_probe_records_the_head_commits_parents(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        """Structured facts for the caller, not a second git call.

        ``branch_work_landed`` puts these in the verdict probe, so an operator
        reading a no_op_landing L1 can see the tip's shape (merge or not)
        without re-running git.
        """
        base = repo.sha('main')
        repo.git('checkout', '-b', 'task/parents')
        first = repo.commit('feat: one', {'pkg/f.py': _numbered('f', 8)})
        tip = repo.commit('feat: two', {'pkg/g.py': _numbered('g', 8)})

        probe: dict[str, object] = {}
        assert await git_ops.net_diff_is_empty(base, tip, probe=probe) is False
        assert probe['net_diff_head_parents'] == [first]
        assert probe['net_diff_merge_base'] == base

    async def test_probe_records_parents_of_a_merge_head(
        self, git_ops: GitOps, repo: _Repo,
    ) -> None:
        base = repo.sha('main')
        repo.git('checkout', '-b', 'task/m')
        repo.commit('feat: branch work', {'pkg/f.py': _numbered('f', 8)})
        repo.git('checkout', 'main')
        main_tip = repo.commit('chore: main work', {'pkg/h.py': _numbered('h', 8)})
        repo.git('merge', '--no-ff', 'task/m', '-m', 'Merge task/m into main')
        merge_sha = repo.sha('main')

        probe: dict[str, object] = {}
        await git_ops.net_diff_is_empty(base, merge_sha, probe=probe)
        parents = probe['net_diff_head_parents']
        assert isinstance(parents, list) and len(parents) == 2
        assert parents[0] == main_tip

    async def test_probe_is_optional(self, git_ops: GitOps, repo: _Repo) -> None:
        """Every caller that does not want facts must not have to pass a dict."""
        assert await git_ops.net_diff_is_empty('main', 'main') is True
