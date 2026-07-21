"""Tests for eval worktree snapshot management (cleanup + setup env).

Hermetic: every test drives the snapshots helpers over ``tmp_path`` with the
git boundary either absent (cleanup's ``git worktree remove`` fails on a
non-git tmp dir and is caught) or unexercised (the pure ``read_python_pin`` /
``_eval_setup_env`` helpers). No real git repo, no ``uv sync`` run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from orchestrator.artifacts import TaskArtifacts


@pytest.mark.asyncio
async def test_cleanup_removes_sibling_relocated_meta_root(tmp_path):
    # BUG 1 GREEN moved the architect plan to the RELOCATED .task-meta/<name>/
    # root — a SIBLING of the worktree, NOT nested under it. `git worktree
    # remove` therefore no longer deletes it, so cleanup_eval_worktree must
    # rmtree the sibling meta root itself or leak one dir per architect run.
    from orchestrator.evals.snapshots import cleanup_eval_worktree

    worktree = tmp_path / 'base' / 'run-abc'
    worktree.mkdir(parents=True)
    meta_root = TaskArtifacts.meta_root_for(worktree.parent, worktree.name)
    assert meta_root == tmp_path / 'base' / '.task-meta' / 'run-abc'
    meta_root.mkdir(parents=True)
    (meta_root / 'plan.json').write_text('{"steps": []}')

    # tmp_path is not a git repo, so the `git worktree remove` raises
    # RuntimeError which cleanup already catches+logs; the sibling meta-root
    # removal runs after that try/except, so no real git repo is needed here.
    await cleanup_eval_worktree(tmp_path, worktree)

    assert not meta_root.exists(), (
        'cleanup_eval_worktree left the sibling relocated meta root behind'
    )


# ---------------------------------------------------------------------------
# BUG 2b: pin the eval worktree `uv sync` to the worktree's own 3.13 venv and
# scrub the orchestrator venv from its env (the 2026-05-29 ghost-venv incident).
# read_python_pin / _eval_setup_env do not exist yet → ImportError until step-6.
# ---------------------------------------------------------------------------

def test_read_python_pin_reads_stripped_first_line(tmp_path):
    from orchestrator.evals.snapshots import read_python_pin

    (tmp_path / '.python-version').write_text('3.13\n')
    assert read_python_pin(tmp_path) == '3.13'


def test_read_python_pin_absent_returns_none(tmp_path):
    from orchestrator.evals.snapshots import read_python_pin

    assert read_python_pin(tmp_path) is None


def test_read_python_pin_non_utf8_returns_none(tmp_path):
    # A garbled (non-UTF-8) .python-version must fail SAFE to None rather than
    # let UnicodeDecodeError (a ValueError, not an OSError) escape the helper
    # and abort the eval — read_python_pin advertises a fail-safe contract.
    from orchestrator.evals.snapshots import read_python_pin

    (tmp_path / '.python-version').write_bytes(b'\xff\xfe3.13')
    assert read_python_pin(tmp_path) is None


def test_eval_setup_env_scrubs_venv_and_pins_python(tmp_path, monkeypatch):
    from orchestrator.evals.snapshots import _eval_setup_env

    # Ambient orchestrator venv activation vars present (as under a live run).
    monkeypatch.setenv('VIRTUAL_ENV', '/orchestrator/.venv')
    monkeypatch.setenv('UV_PROJECT_ENVIRONMENT', '/orchestrator/.venv')
    (tmp_path / '.python-version').write_text('3.13\n')

    env = _eval_setup_env(tmp_path)

    # Scrubbed of the orchestrator venv activation vars (the 2026-05-29
    # ghost-venv hazard) via verify._target_subprocess_env.
    assert 'VIRTUAL_ENV' not in env
    assert 'UV_PROJECT_ENVIRONMENT' not in env
    # Pinned to the worktree's own interpreter.
    assert env['UV_PYTHON'] == '3.13'


def test_eval_setup_env_no_pin_when_no_python_version(tmp_path, monkeypatch):
    from orchestrator.evals.snapshots import _eval_setup_env

    monkeypatch.setenv('VIRTUAL_ENV', '/orchestrator/.venv')
    monkeypatch.delenv('UV_PYTHON', raising=False)
    # No .python-version in the worktree → no UV_PYTHON injected (fail-safe).
    env = _eval_setup_env(tmp_path)
    assert 'UV_PYTHON' not in env
    assert 'VIRTUAL_ENV' not in env


# ---------------------------------------------------------------------------
# task 2875 — create_eval_worktree sources the setup `uv sync` UV_PYTHON pin
# from the target project_root's CURRENT checkout (a 3.13-bearing tree), NOT
# the eval worktree checked out at a pre_task_commit that predates
# `.python-version`. For 11/22 fixtures that baseline predates c5ac23d7ac, so a
# worktree-sourced pin resolves None → uv default 3.14t → aiosqlite
# ModuleNotFoundError at verify. Sourcing from project_root mirrors production
# dispatch and keeps setup==verify interpreter agreement (task 2847).
# ---------------------------------------------------------------------------

def _init_repo_with_late_python_version(project_root: Path) -> str:
    """Build a real 2-commit git repo; return the pre-`.python-version` SHA.

    Commit 1 is a baseline WITHOUT ``.python-version`` (the fixture's
    ``pre_task_commit`` — its checked-out tree carries no pin); commit 2 adds
    ``.python-version``=3.13 so project_root's CURRENT checkout pins 3.13 while
    the returned baseline SHA does not. Exactly the fixture divergence guarded.
    """
    def _git(*args: str) -> str:
        return subprocess.run(
            ['git', *args],
            cwd=project_root, capture_output=True, text=True, check=True,
        ).stdout.strip()

    _git('init')
    _git('config', 'user.email', 'eval@test')
    _git('config', 'user.name', 'Eval Test')
    _git('config', 'commit.gpgsign', 'false')
    (project_root / 'README.md').write_text('hi\n')
    _git('add', 'README.md')
    _git('commit', '--no-verify', '-m', 'baseline without .python-version')
    pre = _git('rev-parse', 'HEAD')
    (project_root / '.python-version').write_text('3.13\n')
    _git('add', '.python-version')
    _git('commit', '--no-verify', '-m', 'add .python-version 3.13')
    return pre


@pytest.mark.asyncio
async def test_create_eval_worktree_sources_setup_pin_from_project_root(
    tmp_path, monkeypatch,
):
    from orchestrator.evals import snapshots
    from orchestrator.evals.snapshots import create_eval_worktree

    project_root = tmp_path / 'proj'
    project_root.mkdir()
    pre = _init_repo_with_late_python_version(project_root)

    # Spy on _eval_setup_env: record its single positional pin_source arg, then
    # delegate to the REAL helper so setup_commands=['true'] still runs.
    real_setup_env = snapshots._eval_setup_env
    captured: dict = {}

    def _spy(pin_source):
        captured['pin_source'] = pin_source
        return real_setup_env(pin_source)

    monkeypatch.setattr(snapshots, '_eval_setup_env', _spy)

    worktree_path, _run_id = await create_eval_worktree(
        project_root, 'df_task_x', pre, setup_commands=['true'],
    )

    # The setup pin sources from project_root's CURRENT checkout (3.13-bearing),
    # NOT the eval worktree checked out at `pre` (which predates .python-version).
    assert captured['pin_source'] == project_root
    assert captured['pin_source'] != worktree_path
