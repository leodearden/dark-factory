"""A linked worktree of a temp main repository, holding a copy of a checkout's scripts and shared.

A script run from the copy derives its default project root from where its
``shared`` sits, so a probe run there shows whether that default follows the
temp MAIN checkout rather than the directory the copy happens to sit in.

Stdlib-only on purpose: conftest puts this directory on ``sys.path`` for the
suites here, and a suite in another venv (fused-memory's, for a script that
imports fused_memory) loads this file by path.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def linked_worktree_copy(main: Path, checkout: Path) -> Path:
    """Create a git repository at *main*; return a linked worktree of it holding *checkout*'s scripts and shared."""
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(main)], check=True, capture_output=True)
    _git(main, 'config', 'user.email', 'linked-worktree-copy@example.com')
    _git(main, 'config', 'user.name', 'Linked Worktree Copy')
    _git(main, 'config', 'commit.gpgsign', 'false')
    _git(main, 'commit', '-q', '--allow-empty', '--no-verify', '-m', 'seed')
    tree = main / '.worktrees' / 'w'
    _git(main, 'worktree', 'add', '-q', str(tree), '-b', 'w')
    return checkout_copy(tree, checkout)


def checkout_copy(dest: Path, checkout: Path) -> Path:
    """Copy *checkout*'s pyproject.toml, top-level scripts and shared into *dest*; return *dest*.

    Called alone, it places the same copy outside any checkout, where no default root can be derived.
    """
    dest.mkdir(exist_ok=True)
    shutil.copy2(checkout / 'pyproject.toml', dest / 'pyproject.toml')
    (dest / 'scripts').mkdir()
    for script in (checkout / 'scripts').glob('*.py'):
        shutil.copy2(script, dest / 'scripts' / script.name)
    shutil.copytree(
        checkout / 'shared' / 'src' / 'shared',
        dest / 'shared' / 'src' / 'shared',
        ignore=shutil.ignore_patterns('__pycache__'),
    )
    return dest


def probe_env() -> dict[str, str]:
    """This process's environment without the variables that would bind a probe to another checkout's code."""
    dropped = {'PYTHONPATH', 'PYTHONSAFEPATH', 'VIRTUAL_ENV'}
    env = {key: value for key, value in os.environ.items() if key not in dropped}
    env['PYTHONNOUSERSITE'] = '1'
    return env


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(['git', '-C', str(cwd), *args], check=True, capture_output=True)
