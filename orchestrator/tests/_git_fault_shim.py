"""Fail chosen git calls at the process boundary, so no git_ops internal is patched.

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


def fail_git(monkeypatch: pytest.MonkeyPatch, bin_dir: Path, argv_glob: str) -> None:
    """Make every git call whose space-joined arguments match the shell pattern
    *argv_glob* exit 128 with a ``fatal:`` line; every other call reaches the
    real git. A prefix match is ``'<prefix>*'``."""
    real_git = shutil.which('git')
    assert real_git is not None
    bin_dir.mkdir(parents=True)
    shim = bin_dir / 'git'
    shim.write_text(
        '#!/bin/sh\n'
        f'fail_glob={shlex.quote(argv_glob)}\n'
        'case "$*" in $fail_glob)\n'
        "  echo 'fatal: injected failure (test fault injection)' >&2; exit 128 ;;\n"
        'esac\n'
        f'exec {shlex.quote(real_git)} "$@"\n'
    )
    shim.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
