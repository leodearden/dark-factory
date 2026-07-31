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

import ast
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
        _init_repo(tmp_path / 'real' / 'repo')
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


# ===========================================================================
# Recurrence guard: no NEW unguarded git-writing helper in test_git_ops.py
# ===========================================================================
#
# The behavioural tests above and in test_git_ops.py cover the two helpers that
# exist today.  They would not notice a THIRD one added next month — and the
# incident report frames esc-3072-3 as an ongoing recurrence risk for every task
# that touches test_git_ops.py, not a one-off.  This guard closes that gap.
#
# It reads CALL NODES via the AST, never comments or string literals, so a
# docstring that merely mentions these names can never trip it.  Structural
# code analysis, not a documentation meta-test.

_TARGET_MODULE = Path(__file__).parent / 'test_git_ops.py'

# Module-level helpers that CREATE their own repo before touching it (a
# `git init` / `git clone` earlier in the same body).  Their cwd is a repo root
# by construction, so there is no enclosing repo for git's upward walk to find.
# Kept explicit rather than inferred, and self-verifying: the companion test
# below asserts each entry really does contain that init/clone, so the
# allowlist cannot quietly rot into a blanket exemption.
_SELF_INITIALISING_HELPERS = frozenset({
    '_setup_repo',
    '_setup_repo_with_remote',
    '_push_n_commits_to_origin',
})

# git subcommands that only READ.  Deliberately an allowlist of readers rather
# than a denylist of writers: an unrecognised — or newly invented —
# subcommand is then treated as mutating, so this guard fails CLOSED.
# ``symbolic-ref`` is pointedly absent: its two-argument form writes.
_READ_ONLY_GIT_SUBCOMMANDS = frozenset({
    'blame', 'cat-file', 'check-ignore', 'cherry', 'count-objects', 'describe',
    'diff', 'diff-index', 'diff-tree', 'for-each-ref', 'grep', 'log',
    'ls-files', 'ls-remote', 'ls-tree', 'merge-base', 'name-rev', 'patch-id',
    'rev-list', 'rev-parse', 'shortlog', 'show', 'show-ref', 'status', 'var',
    'verify-pack', 'whatchanged',
})

_GUARD_FUNC = 'assert_isolated_git_repo'


def _param_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    a = func.args
    return {arg.arg for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs)}


def _mentions_param(expr: ast.expr, params: set[str]) -> bool:
    """True if *expr* reads any of *params* (e.g. ``cwd`` inside ``str(cwd)``)."""
    return any(
        isinstance(n, ast.Name) and n.id in params for n in ast.walk(expr)
    )


def _is_caller_supplied_git_write(call: ast.Call, params: set[str]) -> bool:
    """True if *call* runs a MUTATING git command at a caller-supplied ``cwd``.

    Two shapes are recognised, with deliberately different strictness:

    * ``subprocess.run(..., cwd=<anything derived from a parameter>)`` — the
      raw, unwrapped seam.  It bypasses ``_run`` and every convention attached
      to it, so any parameter-derived cwd counts.  This is the shape
      ``_inject_uu_state`` used when it corrupted a live task worktree.
    * ``_run(['git', <subcommand>, ...], cwd=<bare parameter name>)`` — a raw
      path handed straight in by the caller.  An *attribute* such as
      ``git_ops.project_root`` is NOT matched: that path comes off a ``GitOps``
      instance, which is a repository root by construction rather than an
      arbitrary caller-chosen directory.

    Read-only git commands are excluded: they cannot corrupt the repo they
    wrongly resolve to.  (Their premise can still silently change — see
    ``test_non_git_dir_returns_none`` — but that is a hermeticity bug in the
    calling test, not a mutation hazard in the helper.)
    """
    cwd_kw = next((k for k in call.keywords if k.arg == 'cwd'), None)
    if cwd_kw is None:
        return False

    func = call.func
    if (
        isinstance(func, ast.Attribute)
        and func.attr == 'run'
        and isinstance(func.value, ast.Name)
        and func.value.id == 'subprocess'
    ):
        return _mentions_param(cwd_kw.value, params)

    if not (isinstance(func, ast.Name) and func.id == '_run'):
        return False
    if not (isinstance(cwd_kw.value, ast.Name) and cwd_kw.value.id in params):
        return False
    cmd = call.args[0] if call.args else None
    if not (isinstance(cmd, ast.List) and len(cmd.elts) >= 2):
        return False
    program, subcommand = cmd.elts[0], cmd.elts[1]
    if not (isinstance(program, ast.Constant) and program.value == 'git'):
        return False
    if not isinstance(subcommand, ast.Constant):
        return True  # non-literal subcommand: fail closed
    return subcommand.value not in _READ_ONLY_GIT_SUBCOMMANDS


def _module_level_functions(
    tree: ast.Module,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _unguarded_git_writers(tree: ast.Module) -> list[str]:
    """Module-level helpers that write via git at a caller-supplied cwd without
    calling ``assert_isolated_git_repo`` FIRST.

    "First" is literal: the guard must appear at a lower line number than the
    git call.  A guard that ran afterwards would be decorative —
    ``git hash-object -w`` has already written its blobs by then.
    """
    offenders: list[str] = []
    for func in _module_level_functions(tree):
        if func.name in _SELF_INITIALISING_HELPERS:
            continue
        params = _param_names(func)
        guard_lines = [
            n.lineno for n in ast.walk(func)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == _GUARD_FUNC
        ]
        first_guard = min(guard_lines) if guard_lines else None
        for node in ast.walk(func):
            if not isinstance(node, ast.Call):
                continue
            if not _is_caller_supplied_git_write(node, params):
                continue
            if first_guard is None or first_guard >= node.lineno:
                offenders.append(f'{func.name} (git write at line {node.lineno})')
                break
    return offenders


class TestNoUnguardedGitWritersInTestGitOps:
    """No module-level helper in test_git_ops.py writes via git unguarded."""

    def test_every_git_writing_helper_calls_the_guard_first(self) -> None:
        tree = ast.parse(_TARGET_MODULE.read_text(encoding='utf-8'))

        offenders = _unguarded_git_writers(tree)

        assert not offenders, (
            'Unguarded git-writing helper(s) in test_git_ops.py.\n'
            'Each of these runs a MUTATING git command at a cwd handed in by '
            'its caller. Git repository discovery walks UP the directory tree, '
            'so if that cwd is not itself a repo root the command silently '
            'retargets whatever repo encloses it — under a pytest basetemp '
            'nested inside a live task worktree, that is production state '
            '(esc-3072-3).\n'
            'Fix: call assert_isolated_git_repo(<cwd>) as the FIRST statement, '
            'before any subprocess and before any filesystem write, and pass '
            'env=git_env_with_ceiling(<cwd>) to raw subprocess calls. If the '
            'helper git-inits or git-clones its own target, add it to '
            '_SELF_INITIALISING_HELPERS in this file with a comment.\n'
            f'Offenders: {offenders}'
        )

    def test_the_guard_actually_detects_an_unguarded_helper(self) -> None:
        """Self-test: the detector is not vacuously green.

        A structural guard that silently matches nothing is worse than no
        guard, because it reads as coverage. Feed it a helper with the exact
        offending shape and require a hit.
        """
        tree = ast.parse(
            'async def _bad_helper(repo):\n'
            "    await _run(['git', 'add', '-A'], cwd=repo)\n"
        )

        assert _unguarded_git_writers(tree) != []

    def test_the_guard_accepts_a_correctly_guarded_helper(self) -> None:
        """Self-test: adding the guard call clears the finding."""
        tree = ast.parse(
            'async def _good_helper(repo):\n'
            '    assert_isolated_git_repo(repo)\n'
            "    await _run(['git', 'add', '-A'], cwd=repo)\n"
        )

        assert _unguarded_git_writers(tree) == []

    def test_a_guard_placed_after_the_write_does_not_count(self) -> None:
        """Self-test: ordering is enforced, not merely presence.

        A guard that runs after the git call is decorative — the blobs are
        already written by then.
        """
        tree = ast.parse(
            'async def _late_helper(repo):\n'
            "    await _run(['git', 'add', '-A'], cwd=repo)\n"
            '    assert_isolated_git_repo(repo)\n'
        )

        assert _unguarded_git_writers(tree) != []

    def test_read_only_git_commands_are_not_flagged(self) -> None:
        """Self-test: a read-only command cannot corrupt what it mis-resolves."""
        tree = ast.parse(
            'async def _reader(wt_path):\n'
            "    await _run(['git', 'rev-parse', '--git-dir'], cwd=wt_path)\n"
        )

        assert _unguarded_git_writers(tree) == []

    def test_unknown_subcommand_is_treated_as_mutating(self) -> None:
        """Self-test: the reader allowlist fails CLOSED.

        A subcommand nobody has classified must be assumed to write, so a new
        git verb cannot slip through by being unrecognised.
        """
        tree = ast.parse(
            'async def _mystery(repo):\n'
            "    await _run(['git', 'brand-new-verb'], cwd=repo)\n"
        )

        assert _unguarded_git_writers(tree) != []

    def test_allowlist_entries_really_create_their_own_repo(self) -> None:
        """Every _SELF_INITIALISING_HELPERS entry genuinely git-inits/clones.

        Keeps the exemption honest: if one of these is ever refactored to
        operate on a repo it did not create, its exemption stops being true and
        this fails rather than silently widening the hole.
        """
        tree = ast.parse(_TARGET_MODULE.read_text(encoding='utf-8'))
        by_name = {f.name: f for f in _module_level_functions(tree)}

        for name in sorted(_SELF_INITIALISING_HELPERS):
            func = by_name.get(name)
            assert func is not None, (
                f'_SELF_INITIALISING_HELPERS names {name}, which no longer '
                f'exists in test_git_ops.py — drop the stale entry'
            )
            creators: set[str] = {
                node.args[0].elts[1].value
                for node in ast.walk(func)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == '_run'
                and node.args
                and isinstance(node.args[0], ast.List)
                and len(node.args[0].elts) >= 2
                and isinstance(node.args[0].elts[0], ast.Constant)
                and node.args[0].elts[0].value == 'git'
                and isinstance(node.args[0].elts[1], ast.Constant)
                and isinstance(node.args[0].elts[1].value, str)
            }
            assert creators & {'init', 'clone'}, (
                f'{name} is exempted as self-initialising but no longer runs '
                f'git init/clone; it now operates on a repo it did not create. '
                f'Remove the exemption and call {_GUARD_FUNC} instead. '
                f'git subcommands found: {sorted(creators)}'
            )
