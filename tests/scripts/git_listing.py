"""Git as the authority on which files a repo-wide guard sweeps.

Two things every such guard needs from git:

* ``git`` runs it with ``GIT_*`` scrubbed from the environment, because a hook
  or wrapper exporting ``GIT_DIR`` or ``GIT_INDEX_FILE`` (this repo's own
  pre-commit hook does) would silently point the call at a different index.
  A missing git or a non-zero exit raises ``AssertionError`` and never skips:
  a guard whose population git supplies would otherwise pass green without
  having looked at anything.
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


def git(cwd: Path, *args: str) -> str:
    """The stdout of ``git *args`` run in *cwd*."""
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    command = ['git', *args]
    try:
        proc = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, env=env, check=False,
        )
    except OSError as exc:
        raise AssertionError(f'could not run {command} in {cwd}: {exc!r}') from exc
    assert proc.returncode == 0, (
        f'{command} failed in {cwd} with rc={proc.returncode}; stderr: {proc.stderr!r}'
    )
    return proc.stdout


def listed_files(root: Path, *pathspecs: str) -> list[Path]:
    """Existing files under *root* matching *pathspecs* that git tracks, or sees untracked and not ignored."""
    listing = git(
        root, 'ls-files', '--cached', '--others', '--exclude-standard', '-z', '--', *pathspecs,
    )
    candidates = [root / entry for entry in listing.split('\0') if entry]
    return [path for path in candidates if path.is_file()]
