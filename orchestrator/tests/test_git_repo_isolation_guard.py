"""Guard tests: test-suite git helpers cannot escape into an enclosing repo.

Incident esc-3072-3.  Git repository discovery walks UP the directory tree.
A test helper that shells out to git with a caller-supplied ``cwd`` therefore
does not operate on "the directory the caller named" — it operates on
*whatever repo encloses that directory*.  When pytest's basetemp happens to
live inside a live task worktree (``.worktrees/<task>/.pytest-tmp/``), a
helper handed a bare ``tmp_path`` silently retargets production state: three
blobs were written into the live worktree's object store and ``foo.py`` was
staged at stages 1/2/3, leaving ``UU foo.py`` in a real task's index.

The two-layer defence under test here lives in ``_orch_helpers``:

* :func:`assert_isolated_git_repo` — a pure-filesystem pre-flight that runs
  BEFORE any subprocess, so a rejected call writes nothing anywhere.  This is
  the property a mid-sequence git failure cannot provide: ``git hash-object -w``
  writes its blobs before a later ``git update-index`` can fail.
* :func:`git_env_with_ceiling` — a ``GIT_CEILING_DIRECTORIES`` env ceiling that
  makes the upward walk physically unable to leave ``cwd`` even if the
  pre-flight is later refactored away.

This module holds the guard-API unit tests plus the AST recurrence guard over
``test_git_ops.py``.  The escape-path regression tests live in
``test_git_ops.py`` itself, next to the module-private helpers they cover.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from _orch_helpers import (
    NonIsolatedGitRepoError,
    assert_isolated_git_repo,
    git_env_with_ceiling,
)


def _init_repo(path: Path) -> Path:
    """``git init`` a fresh repo at *path* with one commit.  Creates its own
    target, so it cannot escape into an enclosing repo.
    """
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', '-b', 'main'], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'test@test.com'], cwd=path, check=True, capture_output=True,
    )
    subprocess.run(
        ['git', 'config', 'user.name', 'Test'], cwd=path, check=True, capture_output=True,
    )
    (path / 'README.md').write_text('# sentinel\n')
    subprocess.run(['git', 'add', '-A'], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ['git', 'commit', '-m', 'initial'], cwd=path, check=True, capture_output=True,
    )
    return path


def _add_linked_worktree(repo: Path, name: str) -> Path:
    """``git worktree add`` a linked worktree of *repo*; its ``.git`` is a FILE."""
    wt = repo.parent / name
    subprocess.run(
        ['git', 'worktree', 'add', '-b', name, str(wt)],
        cwd=repo, check=True, capture_output=True,
    )
    return wt


class TestAssertIsolatedGitRepo:
    """assert_isolated_git_repo(cwd) admits repo roots, refuses everything else."""

    def test_accepts_normal_repo_root(self, tmp_path: Path):
        """A normal checkout root (``.git`` is a DIRECTORY) is accepted."""
        repo = _init_repo(tmp_path / 'repo')
        assert (repo / '.git').is_dir(), 'precondition: normal checkout has a .git dir'

        assert_isolated_git_repo(repo)  # must not raise

    def test_accepts_linked_worktree_root(self, tmp_path: Path):
        """A linked-worktree root (``.git`` is a FILE) is accepted.

        Load-bearing: two legitimate ``_inject_uu_state`` call sites
        (test_git_ops.py:787 and :5116) pass ``<tmp repo>/.worktrees/<task>``
        roots produced by ``git_ops.create_worktree(...)``, where ``.git`` is a
        gitdir *file*, not a directory.  A naive ``.git``-is-a-directory check
        would reject correct callers.
        """
        repo = _init_repo(tmp_path / 'repo')
        wt = _add_linked_worktree(repo, 'linked-wt')
        assert (wt / '.git').is_file(), 'precondition: linked worktree has a .git file'

        assert_isolated_git_repo(wt)  # must not raise

    def test_raises_on_bare_uninitialized_dir(self, tmp_path: Path):
        """A never-``git init``ed directory is refused."""
        bare = tmp_path / 'never-a-repo'
        bare.mkdir()

        with pytest.raises(NonIsolatedGitRepoError):
            assert_isolated_git_repo(bare)

    def test_raises_on_non_repo_nested_inside_live_repo(self, tmp_path: Path):
        """The esc-3072-3 shape: a non-repo dir NESTED inside a live repo.

        This is exactly the layout that made the incident possible — git's
        upward walk resolves the enclosing repo, so the helper would have
        happily mutated it.  The rejection message must name the rejected path
        so the failure is self-diagnosing.
        """
        sentinel = _init_repo(tmp_path / 'live-worktree')
        nested = sentinel / '.pytest-tmp' / 'test_x0'
        nested.mkdir(parents=True)

        with pytest.raises(NonIsolatedGitRepoError) as excinfo:
            assert_isolated_git_repo(nested)

        message = str(excinfo.value)
        assert str(nested) in message, (
            f'Error must name the rejected path {nested} so the failure is '
            f'self-diagnosing; got: {message}'
        )

    def test_error_names_the_enclosing_repo_git_would_have_hit(self, tmp_path: Path):
        """The message points at the repo the upward walk WOULD have resolved.

        Naming the victim is what turns "this call was refused" into "this call
        was about to write into <that repo>".
        """
        sentinel = _init_repo(tmp_path / 'live-worktree')
        nested = sentinel / '.pytest-tmp' / 'test_x0'
        nested.mkdir(parents=True)

        with pytest.raises(NonIsolatedGitRepoError) as excinfo:
            assert_isolated_git_repo(nested)

        assert str(sentinel) in str(excinfo.value), (
            f'Error must name the enclosing repo {sentinel}; got: {excinfo.value}'
        )

    def test_performs_no_subprocess_work(self, tmp_path: Path, monkeypatch):
        """The guard is pure filesystem — it spawns NO child process.

        This is the zero-write property in executable form.  A guard that shells
        out to ``git rev-parse`` to decide would already be too late for the
        general case, and would itself be subject to the upward walk it exists
        to contain.  Sabotage every spawn path and assert the guard still
        reaches its verdict.
        """
        sentinel = _init_repo(tmp_path / 'live-worktree')
        nested = sentinel / '.pytest-tmp' / 'test_x0'
        nested.mkdir(parents=True)

        def _no_spawn(*args, **kwargs):
            raise AssertionError(
                'assert_isolated_git_repo spawned a child process; it must be '
                'pure-filesystem pre-flight so a rejected call writes nothing.'
            )

        monkeypatch.setattr(subprocess, 'Popen', _no_spawn)
        monkeypatch.setattr(os, 'posix_spawn', _no_spawn)

        # Refusal path: git resolution WOULD have succeeded here (the enclosing
        # sentinel repo is reachable), and the guard refuses anyway.
        with pytest.raises(NonIsolatedGitRepoError):
            assert_isolated_git_repo(nested)

        # Acceptance path: also decided without spawning anything.
        assert_isolated_git_repo(sentinel)

    def test_accepts_str_path(self, tmp_path: Path):
        """A ``str`` cwd is accepted as readily as a ``Path``.

        ``_inject_uu_state`` stringifies its cwd for ``subprocess.run``; callers
        elsewhere in the suite pass either shape.
        """
        repo = _init_repo(tmp_path / 'repo')

        assert_isolated_git_repo(str(repo))  # type: ignore[arg-type]

    def test_raises_on_missing_dir(self, tmp_path: Path):
        """A path that does not exist at all is refused, not crashed on."""
        with pytest.raises(NonIsolatedGitRepoError):
            assert_isolated_git_repo(tmp_path / 'does-not-exist')

    def test_error_is_an_assertion_error(self, tmp_path: Path):
        """``NonIsolatedGitRepoError`` reads as a test-harness contract violation.

        Subclassing ``AssertionError`` (not ``Exception``) keeps an unguarded
        call legible as "the test suite broke its own isolation contract"
        rather than as a product error.
        """
        assert issubclass(NonIsolatedGitRepoError, AssertionError)

        bare = tmp_path / 'nope'
        bare.mkdir()
        with pytest.raises(AssertionError):
            assert_isolated_git_repo(bare)


def _object_store(repo: Path) -> set[str]:
    """Snapshot of every loose/packed object file under ``<repo>/.git/objects``.

    Used to prove that a refused call left the victim's object store byte-for-
    byte unchanged — not merely that the command exited non-zero.
    """
    objects = repo / '.git' / 'objects'
    if not objects.is_dir():
        return set()
    return {str(p.relative_to(objects)) for p in objects.rglob('*') if p.is_file()}


class TestGitEnvWithCeiling:
    """git_env_with_ceiling(cwd) contains git's upward repo discovery at cwd."""

    def test_copies_os_environ(self, tmp_path: Path, monkeypatch):
        """The returned mapping inherits the ambient environment, not a blank one.

        A git child spawned with a blank env loses PATH, HOME and the git config
        discovery it needs; the ceiling has to be an *addition* to os.environ.
        """
        repo = _init_repo(tmp_path / 'repo')
        monkeypatch.setenv('DF_3182_SENTINEL', 'present')

        env = git_env_with_ceiling(repo)

        assert env.get('DF_3182_SENTINEL') == 'present'
        assert 'PATH' in env

    def test_does_not_mutate_os_environ(self, tmp_path: Path):
        """It returns a COPY — the caller's own process env is left alone."""
        repo = _init_repo(tmp_path / 'repo')
        before = os.environ.get('GIT_CEILING_DIRECTORIES')

        env = git_env_with_ceiling(repo)
        env['DF_3182_MUTATION'] = 'x'

        assert os.environ.get('GIT_CEILING_DIRECTORIES') == before
        assert 'DF_3182_MUTATION' not in os.environ

    def test_ceiling_is_the_parent_of_cwd(self, tmp_path: Path):
        """The ceiling is cwd's PARENT: git may inspect cwd, never above it."""
        repo = _init_repo(tmp_path / 'repo')

        env = git_env_with_ceiling(repo)

        assert env['GIT_CEILING_DIRECTORIES'] == str(repo.resolve().parent)

    def test_ceiling_is_absolute(self, tmp_path: Path, monkeypatch):
        """git ignores non-absolute ceiling entries outright, so relative in →
        absolute out is load-bearing, not cosmetic.
        """
        repo = _init_repo(tmp_path / 'repo')
        monkeypatch.chdir(repo)

        env = git_env_with_ceiling(Path('.'))

        assert Path(env['GIT_CEILING_DIRECTORIES']).is_absolute()
        assert env['GIT_CEILING_DIRECTORIES'] == str(repo.resolve().parent)

    def test_ceiling_resolves_symlinks(self, tmp_path: Path):
        """git compares the ceiling against the symlink-resolved path.

        An unresolved entry silently never matches, so the containment would be
        inert — a fail-open no-one would notice.
        """
        repo = _init_repo(tmp_path / 'real' / 'repo')
        link_dir = tmp_path / 'link'
        link_dir.symlink_to(tmp_path / 'real', target_is_directory=True)

        env = git_env_with_ceiling(link_dir / 'repo')

        assert env['GIT_CEILING_DIRECTORIES'] == str((tmp_path / 'real').resolve())

    def test_normal_repo_root_still_resolves(self, tmp_path: Path):
        """A legitimate caller at a normal repo root is unaffected by the ceiling."""
        repo = _init_repo(tmp_path / 'repo')

        proc = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=repo, capture_output=True, env=git_env_with_ceiling(repo),
        )

        assert proc.returncode == 0, f'ceiling broke a normal repo root: {proc.stderr!r}'
        assert Path(proc.stdout.decode().strip()).resolve() == repo.resolve()

    def test_linked_worktree_root_still_resolves(self, tmp_path: Path):
        """A linked-worktree root is unaffected too.

        This is the case that would break the two ``_inject_uu_state`` call
        sites at test_git_ops.py:787 and :5116 if the ceiling were set one level
        too low, so it is pinned explicitly rather than assumed.
        """
        repo = _init_repo(tmp_path / 'repo')
        wt = _add_linked_worktree(repo, 'linked-wt')

        proc = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=wt, capture_output=True, env=git_env_with_ceiling(wt),
        )

        assert proc.returncode == 0, f'ceiling broke a linked worktree: {proc.stderr!r}'
        assert Path(proc.stdout.decode().strip()).resolve() == wt.resolve()

    def test_nested_non_repo_cannot_reach_enclosing_repo(self, tmp_path: Path):
        """The esc-3072-3 shape: contained, and the victim gains NO objects.

        ``git hash-object -w`` is the exact command that leaked blobs into a
        live task worktree.  Under the ceiling it must fail before writing
        anything — the object store is the assertion that matters, not the exit
        code.
        """
        sentinel = _init_repo(tmp_path / 'live-worktree')
        nested = sentinel / '.pytest-tmp' / 'test_x0'
        nested.mkdir(parents=True)
        before = _object_store(sentinel)

        proc = subprocess.run(
            ['git', 'hash-object', '-w', '--stdin'],
            cwd=nested, capture_output=True, input=b'version base\n',
            env=git_env_with_ceiling(nested),
        )

        assert proc.returncode != 0, (
            f'ceiling failed to contain the upward walk; git succeeded with '
            f'stdout={proc.stdout!r}'
        )
        assert _object_store(sentinel) == before, (
            'objects leaked into the enclosing repo despite the ceiling'
        )

    def test_control_without_ceiling_the_walk_escapes(self, tmp_path: Path):
        """Control: the SAME command without the ceiling reaches the enclosing repo.

        Documents in-suite that the containment is what makes the difference —
        without this case a future reader cannot tell whether the test above
        proves anything or merely reflects the ambient directory layout.
        """
        sentinel = _init_repo(tmp_path / 'live-worktree')
        nested = sentinel / '.pytest-tmp' / 'test_x0'
        nested.mkdir(parents=True)
        before = _object_store(sentinel)

        env = os.environ.copy()
        env.pop('GIT_CEILING_DIRECTORIES', None)
        proc = subprocess.run(
            ['git', 'hash-object', '-w', '--stdin'],
            cwd=nested, capture_output=True, input=b'version base\n', env=env,
        )

        assert proc.returncode == 0, (
            f'control precondition failed — git should escape upward here: '
            f'{proc.stderr!r}'
        )
        assert _object_store(sentinel) != before, (
            'control precondition failed — the escaping write should have '
            "landed in the sentinel's object store"
        )
