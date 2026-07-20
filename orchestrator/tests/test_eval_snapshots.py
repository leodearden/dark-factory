"""Tests for eval worktree snapshot management (cleanup + setup env).

Hermetic: every test drives the snapshots helpers over ``tmp_path`` with the
git boundary either absent (cleanup's ``git worktree remove`` fails on a
non-git tmp dir and is caught) or unexercised (the pure ``read_python_pin`` /
``_eval_setup_env`` helpers). No real git repo, no ``uv sync`` run.
"""

from __future__ import annotations

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
