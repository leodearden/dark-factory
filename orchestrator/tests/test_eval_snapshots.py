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


# ---------------------------------------------------------------------------
# task 2876 DEFECT 3 — resolve_worktree_venv_pythons mirrors the smoke gate's
# resolve_venv_pythons (scripts/eval_bootstrap_smoke.sh): direct <wt>/.venv, else
# the one-level subproject glob <wt>/*/.venv. Kept in lockstep so the framework
# provisions aiosqlite into EXACTLY the venv(s) the smoke gate version-checks.
# ---------------------------------------------------------------------------

def _plant_venv_python(base: Path) -> Path:
    """Create a fake ``<base>/.venv/bin/python`` file and return its path."""
    py = base / '.venv' / 'bin' / 'python'
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text('#!/usr/bin/env bash\n')
    return py


def test_resolve_worktree_venv_pythons_direct_layout(tmp_path):
    # (a) top-level layout: <wt>/.venv/bin/python exists → returns exactly it.
    from orchestrator.evals.snapshots import resolve_worktree_venv_pythons

    py = _plant_venv_python(tmp_path)
    assert resolve_worktree_venv_pythons(tmp_path) == [py]


def test_resolve_worktree_venv_pythons_subproject_layout(tmp_path):
    # (b) subproject layout (e.g. df_task_12 → <wt>/orchestrator/.venv): the
    # direct <wt>/.venv is absent, so the one-level glob resolves the subproject.
    from orchestrator.evals.snapshots import resolve_worktree_venv_pythons

    py = _plant_venv_python(tmp_path / 'orchestrator')
    assert not (tmp_path / '.venv').exists()
    assert resolve_worktree_venv_pythons(tmp_path) == [py]


def test_resolve_worktree_venv_pythons_multiple_sorted(tmp_path):
    # (c) more than one subproject venv (e.g. fused-memory + orchestrator) →
    # returns ALL of them, sorted, so the gate/provisioner covers every one.
    from orchestrator.evals.snapshots import resolve_worktree_venv_pythons

    py_orch = _plant_venv_python(tmp_path / 'orchestrator')
    py_fm = _plant_venv_python(tmp_path / 'fused-memory')
    assert not (tmp_path / '.venv').exists()
    assert resolve_worktree_venv_pythons(tmp_path) == sorted([py_fm, py_orch])


def test_resolve_worktree_venv_pythons_none(tmp_path):
    # (d) no venv anywhere (e.g. a reify/Rust worktree) → [] (a clean no-op for
    # the provisioner, which then installs nothing).
    from orchestrator.evals.snapshots import resolve_worktree_venv_pythons

    assert resolve_worktree_venv_pythons(tmp_path) == []


# ---------------------------------------------------------------------------
# task 2876 DEFECT 3 — create_eval_worktree provisions the eval-verify
# cross-member dep(s) (EVAL_VERIFY_EXTRA_DEPS = ('aiosqlite',)) into EACH
# setup-created venv AFTER the setup_commands loop, reusing the already-built
# scrubbed setup_env so the install targets the worktree venv and cannot corrupt
# the live orchestrator .venv (2026-05-29 ghost-venv incident). Root cause: the
# old-baseline orchestrator-only `uv sync` omits aiosqlite (predates orchestrator's
# shared[vllm] dep), yet eval verify's pytest collection imports orchestrator.config
# → shared/__init__ → shared.async_sqlite_base → import aiosqlite.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_eval_worktree_provisions_dep_into_subproject_venv(
    tmp_path, monkeypatch,
):
    from orchestrator.evals import snapshots
    from orchestrator.evals.snapshots import create_eval_worktree

    project_root = tmp_path / 'proj'
    project_root.mkdir()
    pre = _init_repo_with_late_python_version(project_root)

    # Ambient orchestrator venv activation vars present (as under a live run) so
    # we can prove the provisioning env was scrubbed of them.
    monkeypatch.setenv('VIRTUAL_ENV', '/orchestrator/.venv')
    monkeypatch.setenv('UV_PROJECT_ENVIRONMENT', '/orchestrator/.venv')

    # Record every _uv_pip_install invocation instead of running a real `uv pip
    # install`. raising=False: the attribute does not exist until step-4.
    calls: list[dict] = []

    async def _spy(venv_python, deps, cwd, env):
        calls.append(
            {'venv_python': venv_python, 'deps': deps, 'cwd': cwd, 'env': env}
        )

    monkeypatch.setattr(snapshots, '_uv_pip_install', _spy, raising=False)

    # setup_commands plant a fake subproject venv exactly where uv would
    # (`cd orchestrator && uv sync` → <wt>/orchestrator/.venv).
    worktree_path, _run_id = await create_eval_worktree(
        project_root, 'df_task_x', pre,
        setup_commands=['mkdir -p orchestrator/.venv/bin && : > orchestrator/.venv/bin/python'],
    )

    # Provisioned exactly once, into the resolved subproject venv, with aiosqlite
    # among the deps and a scrubbed env (never the live orchestrator venv).
    assert len(calls) == 1, f'expected one provisioning call, got {calls}'
    call = calls[0]
    assert call['venv_python'] == worktree_path / 'orchestrator' / '.venv' / 'bin' / 'python'
    assert 'aiosqlite' in call['deps']
    assert 'VIRTUAL_ENV' not in call['env']
    assert 'UV_PROJECT_ENVIRONMENT' not in call['env']


@pytest.mark.asyncio
async def test_create_eval_worktree_no_venv_no_provisioning(tmp_path, monkeypatch):
    # A setup that builds NO venv (reify/Rust-style) → resolver returns [] → the
    # provisioner is a clean no-op (never calls _uv_pip_install).
    from orchestrator.evals import snapshots
    from orchestrator.evals.snapshots import create_eval_worktree

    project_root = tmp_path / 'proj'
    project_root.mkdir()
    pre = _init_repo_with_late_python_version(project_root)

    calls: list[dict] = []

    async def _spy(venv_python, deps, cwd, env):
        calls.append({'venv_python': venv_python})

    monkeypatch.setattr(snapshots, '_uv_pip_install', _spy, raising=False)

    await create_eval_worktree(
        project_root, 'df_task_x', pre, setup_commands=['true'],
    )

    assert calls == [], f'expected no provisioning when no venv exists, got {calls}'
