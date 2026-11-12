"""The contract of ``tests/scripts/git_listing.py``, driven through its public functions.

Every throwaway repo here is built with the helper's own ``git``, under the
session's isolation fixtures; each test sets the environment it needs with
``monkeypatch.setenv`` so the variable really reaches the helper's call.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from git_listing import git, listed_files, run_git


def _repo(path: Path) -> Path:
    path.mkdir(parents=True)
    git(path, 'init', '-q')
    return path


def test_run_git_keeps_the_discovery_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = _repo(tmp_path / 'outer')
    sub = outer / 'sub'
    sub.mkdir()
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(outer.resolve()))

    proc = run_git(sub, 'rev-parse', '--show-toplevel')

    assert proc.returncode != 0 and proc.stdout == '', (
        'run_git dropped GIT_CEILING_DIRECTORIES: discovery from a directory '
        f'below the ceiling escaped into the enclosing repo {proc.stdout!r} '
        f'(rc={proc.returncode}). The ceiling is the session\'s own containment '
        '(df_pytest_isolation.py::_df_git_ceiling_at_basetemp) and only bounds '
        'the upward walk, so keeping it can never point a call elsewhere.'
    )


def test_git_ignores_a_leaked_git_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = _repo(tmp_path / 'a')
    b = _repo(tmp_path / 'b')
    monkeypatch.setenv('GIT_DIR', str(b / '.git'))

    git_dir = Path(git(a, 'rev-parse', '--absolute-git-dir').strip())

    assert git_dir.resolve() == (a / '.git').resolve(), (
        f'a leaked GIT_DIR retargeted git run in {a} at {git_dir}'
    )


def test_listed_files_is_not_blinded_by_literal_pathspecs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path / 'repo')
    (root / 'a.py').write_text('x = 1\n')
    git(root, 'add', 'a.py')
    monkeypatch.setenv('GIT_LITERAL_PATHSPECS', '1')

    assert listed_files(root, '*.py') == [root / 'a.py'], (
        'GIT_LITERAL_PATHSPECS reached the sweep and turned the glob into a '
        'literal, so it saw nothing. That var redirects nothing, which is why '
        'df_pytest_isolation.py::git_redirect_env does not list it, and why the '
        'helper scrubs every GIT_* var rather than only that narrow list.'
    )


def test_git_fails_loudly_where_run_git_reports(tmp_path: Path) -> None:
    root = _repo(tmp_path / 'repo')
    args = ('rev-parse', '--verify', 'no-such-ref')

    with pytest.raises(AssertionError, match='rc='):
        git(root, *args)
    assert run_git(root, *args).returncode != 0
