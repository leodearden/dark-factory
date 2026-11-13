"""Fail or fake chosen git calls at the process boundary, so no git_ops internal is patched.

``shared/src/shared/git_async.py::run_git`` copies ``os.environ`` at each spawn,
so a ``git`` placed first on ``PATH`` through ``monkeypatch.setenv`` is the one
every GitOps query runs.
"""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path

import pytest


def _install_shim(
    monkeypatch: pytest.MonkeyPatch, bin_dir: Path, argv_glob: str, on_match: str,
) -> None:
    real_git = shutil.which('git')
    assert real_git is not None
    bin_dir.mkdir(parents=True)
    shim = bin_dir / 'git'
    shim.write_text(
        '#!/bin/sh\n'
        f'match_glob={shlex.quote(argv_glob)}\n'
        f'case "$*" in $match_glob)\n{on_match}\n;; esac\n'
        f'exec {shlex.quote(real_git)} "$@"\n'
    )
    shim.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')


def fail_git(
    monkeypatch: pytest.MonkeyPatch,
    bin_dir: Path,
    argv_glob: str,
    *,
    after: int = 0,
    times: int | None = None,
) -> None:
    """Make git calls whose space-joined arguments match the shell pattern
    *argv_glob* exit 128 with a ``fatal:`` line: the first *after* such calls
    reach the real git, the next *times* (every later one when None) fail, and
    any after those reach the real git again, as does every unmatched call. A
    prefix match is ``'<prefix>*'``; *after* and *times* tell apart calls whose
    arguments are identical."""
    matched = shlex.quote(str(bin_dir / 'matched'))
    last_failing = '' if times is None else f' && [ "$n" -le {after + times} ]'
    _install_shim(
        monkeypatch, bin_dir, argv_glob,
        f'  printf x >> {matched}; n=$(wc -c < {matched})\n'
        f'  if [ "$n" -gt {after} ]{last_failing}; then\n'
        "    echo 'fatal: injected failure (test fault injection)' >&2; exit 128\n"
        '  fi',
    )


def answer_git(
    monkeypatch: pytest.MonkeyPatch, bin_dir: Path, argv_glob: str, stdout: str,
) -> None:
    """Make each git call matching *argv_glob* (as in :func:`fail_git`) drain
    its stdin, print exactly *stdout* and exit 0; every other call reaches the
    real git."""
    _install_shim(
        monkeypatch, bin_dir, argv_glob,
        f'  cat > /dev/null; printf %s {shlex.quote(stdout)}; exit 0',
    )
