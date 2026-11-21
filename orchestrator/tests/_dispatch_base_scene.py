"""Real-git scene helpers for the dispatch-start settlement tests (task 7736).

Shared by test_dispatch_base.py and test_workflow_dispatch_base_settlement.py:
both seed the e2e repo, whose main carries ``GREET_FILE``, and drive real git.
"""

from __future__ import annotations

import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Generic, ParamSpec, TypeVar

from _workflow_helpers import _init_repo

GREET_FILE = 'lib.py'

P = ParamSpec('P')
R = TypeVar('R')


def git_stdout(cwd: Path, *args: str) -> str:
    done = subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def commit_file(cwd: Path, name: str, content: str, message: str) -> str:
    (cwd / name).write_text(content)
    git_stdout(cwd, 'add', name)
    git_stdout(cwd, 'commit', '-m', message)
    return git_stdout(cwd, 'rev-parse', 'HEAD')


def greeting(word: str) -> str:
    return f'def greet(name: str) -> str:\n    return f"{word}, {{name}}"\n'


def is_ancestor(cwd: Path, ancestor: str, descendant: str) -> bool:
    probe = subprocess.run(['git', 'merge-base', '--is-ancestor', ancestor, descendant], cwd=cwd)
    return probe.returncode == 0


def has_merge_head(cwd: Path) -> bool:
    probe = subprocess.run(
        ['git', 'rev-parse', '-q', '--verify', 'MERGE_HEAD'], cwd=cwd, capture_output=True,
    )
    return probe.returncode == 0


def make_repo(tmp_path: Path) -> Path:
    path = tmp_path / 'repo'
    path.mkdir()
    _init_repo(path)
    return path


class FailFirst(Generic[P, R]):
    """A stand-in for the async *real* that raises *error* on its first call, then delegates."""

    def __init__(self, real: Callable[P, Awaitable[R]], error: Exception) -> None:
        self._real = real
        self._error = error
        self.calls = 0

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        self.calls += 1
        if self.calls == 1:
            raise self._error
        return await self._real(*args, **kwargs)
