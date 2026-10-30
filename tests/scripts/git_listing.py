"""Git as the authority on which files a repo-wide guard sweeps.

Two things every such guard needs from git:

* ``git`` runs it with ``GIT_*`` scrubbed from the environment, because a hook
  or wrapper exporting ``GIT_DIR`` or ``GIT_INDEX_FILE`` (this repo's own
  pre-commit hook does) would silently point the call at a different index.
  A missing git or a non-zero exit raises ``AssertionError`` and never skips:
  a guard whose population git supplies would otherwise pass green without
  having looked at anything. ``run_git`` is the same call for commands whose
  exit code IS the answer (e.g. ``check-ignore``), so it does not assert on it.
* ``listed_files`` is ``--cached --others --exclude-standard``: tracked files
  plus untracked ones that are not ignored. So a newly created, uncommitted
  file is swept, while gitignored foreign checkouts (``.worktrees/`` and the
  like) are not. ``-z`` keeps a path holding a space or a quote unmangled.

IMPORT ME, DO NOT COPY ME. This is the idiom of
``tests/scripts/test_nonmember_ruff_config.py::_git`` and ``::_git_ls_files``,
which still carry their own copy until they import this instead.

Importable from ``tests/scripts/test_*.py`` only because
``tests/scripts/conftest.py`` puts this directory on ``sys.path``.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

_KEPT_GIT_ENV = frozenset({'GIT_CEILING_DIRECTORIES'})


def run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """``git *args`` run in *cwd*, whatever its exit code."""
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith('GIT_') or key in _KEPT_GIT_ENV
    }
    command = ['git', *args]
    try:
        return subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, env=env, check=False,
        )
    except OSError as exc:
        raise AssertionError(f'could not run {command} in {cwd}: {exc!r}') from exc


def git(cwd: Path, *args: str) -> str:
    """The stdout of ``git *args`` run in *cwd*."""
    proc = run_git(cwd, *args)
    assert proc.returncode == 0, (
        f'{proc.args} failed in {cwd} with rc={proc.returncode}; stderr: {proc.stderr!r}'
    )
    return proc.stdout


def listed_files(root: Path, *pathspecs: str) -> list[Path]:
    """Existing files under *root* matching *pathspecs* that git tracks, or sees untracked and not ignored."""
    listing = git(
        root, 'ls-files', '--cached', '--others', '--exclude-standard', '-z', '--', *pathspecs,
    )
    candidates = [root / entry for entry in listing.split('\0') if entry]
    return [path for path in candidates if path.is_file()]
