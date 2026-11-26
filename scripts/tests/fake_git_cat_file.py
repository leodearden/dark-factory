"""A ``git`` first on PATH whose ``cat-file`` reads one request and gives a canned answer; every other subcommand is the real git."""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path

import pytest

__all__ = ['install_fake_cat_file']


def install_fake_cat_file(monkeypatch: pytest.MonkeyPatch, bin_dir: Path, answer: str) -> None:
    """Shadow ``git -C <root> cat-file ...`` with a child that prints *answer* to its first request, then exits."""
    real_git = shutil.which('git')
    assert real_git is not None, 'the real git is not on PATH'
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake = bin_dir / 'git'
    fake.write_text(
        '#!/bin/sh\n'
        f'if [ "$3" = cat-file ]; then read -r _request; printf %s {shlex.quote(answer)}; exit 0; fi\n'
        f'exec {shlex.quote(real_git)} "$@"\n',
        encoding='utf-8',
    )
    fake.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
