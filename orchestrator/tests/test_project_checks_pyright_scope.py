"""Behavioral regression test for hooks/project-checks pyright path-filtering.

hooks/project-checks path-filters its pyright loop to the packages whose own
staged .py files changed (task 2551, commit 144a781e29) so docs/plans-only
commits skip pyright entirely instead of paying for an unconditional 3x
pyright run. ``shared`` and ``escalation`` are dependency packages imported
by the other pyright-configured packages, so a commit that stages only
shared/*.py or escalation/*.py must still type-check the consumers, not just
the package it touched. Every package in the merge gate's type_check_command
(task 5338) is a PYRIGHT_PACKAGES member, including shared and escalation
themselves.

These tests run the real hook end-to-end against a throwaway git repo, with
a PATH-stubbed ``uv`` that records the working directory of every
``uv run pyright`` invocation (and no-ops successfully for everything else,
including ``uv run ruff check``), so they observe exactly which packages the
hook type-checks for a given staged change:

- a docs/plans-only change: pyright is skipped entirely (preserved fast path).
- a change scoped to a single consumer package: pyright runs there only
  (preserved per-package filter); the same holds for a leaf like ``sampler``.
- a change under ``shared`` or ``escalation``: pyright must run in EVERY
  package, the touched one included.

See task 2551 and task 5338.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[2]
_HOOK_SOURCE = _REPO_ROOT / "hooks" / "project-checks"

_PACKAGE_DIRS = (
    "shared",
    "escalation",
    "fused-memory",
    "orchestrator",
    "dashboard",
    "sampler",
    "cockpit",
)
_ALL_PYRIGHT_PACKAGES = set(_PACKAGE_DIRS)

# Test stub for `uv`: records the CWD of every `uv run pyright` invocation to
# $UV_STUB_LOG, and exits 0 unconditionally (including for `uv run ruff check`,
# so the hook's ruff gate never blocks these scenarios).
_STUB_UV = """#!/usr/bin/env bash
if [ "$1" = "run" ] && [ "$2" = "pyright" ]; then
    pwd >> "$UV_STUB_LOG"
fi
exit 0
"""


def _make_repo(tmp_path: Path) -> Path:
    """Build a throwaway git repo with empty package dirs and a copy of the real hook."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)

    for pkg in _PACKAGE_DIRS:
        (repo / pkg).mkdir()

    hook_dest = repo / "hooks" / "project-checks"
    hook_dest.parent.mkdir()
    hook_dest.write_text(_HOOK_SOURCE.read_text())
    hook_dest.chmod(0o755)

    return repo


def _make_stub_uv(tmp_path: Path) -> tuple[Path, Path]:
    """Create a stub `uv` in its own PATH dir; return (stub_bin_dir, pyright_cwd_log)."""
    stub_bin = tmp_path / "stub_bin"
    stub_bin.mkdir()
    log_file = tmp_path / "pyright_cwds.log"
    log_file.write_text("")

    uv_stub = stub_bin / "uv"
    uv_stub.write_text(_STUB_UV)
    uv_stub.chmod(0o755)

    return stub_bin, log_file


def _stage_file(repo: Path, rel_path: str, content: str = "x = 1\n") -> None:
    """Write `rel_path` under `repo` and `git add` it. Never under a `tests/` dir —
    that would wake the staged-file asyncmock/bare-magicmock checks, which this
    test intentionally leaves inert (their helper scripts don't exist here).
    """
    target = repo / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    subprocess.run(["git", "add", "--", rel_path], cwd=repo, check=True)


def _run_hook(repo: Path, stub_bin: Path, log_file: Path) -> subprocess.CompletedProcess[str]:
    """Invoke the copied hook against `repo`, with `uv` resolving to the stub."""
    env = dict(os.environ)
    env["PATH"] = f"{stub_bin}{os.pathsep}{env.get('PATH', '')}"
    env["UV_STUB_LOG"] = str(log_file)
    return subprocess.run(
        [str(repo / "hooks" / "project-checks"), str(repo)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )


def _logged_pyright_packages(log_file: Path) -> set[str]:
    """Return the basenames of every CWD the stub `uv run pyright` logged."""
    lines = [line.strip() for line in log_file.read_text().splitlines() if line.strip()]
    return {Path(line).name for line in lines}


def test_docs_only_commit_skips_pyright_entirely(tmp_path: Path) -> None:
    """A docs/plans-only commit must not invoke pyright in any package."""
    repo = _make_repo(tmp_path)
    stub_bin, log_file = _make_stub_uv(tmp_path)
    _stage_file(repo, "README.md", "# hello\n")

    result = _run_hook(repo, stub_bin, log_file)

    assert result.returncode == 0, result.stdout + result.stderr
    assert _logged_pyright_packages(log_file) == set()
    assert "pyright skipped (no Python changes)" in result.stdout


def test_single_package_commit_runs_pyright_there_only(tmp_path: Path) -> None:
    """Staging a .py file under a single consumer package must run pyright in
    that package only, not the others.
    """
    repo = _make_repo(tmp_path)
    stub_bin, log_file = _make_stub_uv(tmp_path)
    _stage_file(repo, "fused-memory/src/foo.py")

    result = _run_hook(repo, stub_bin, log_file)

    assert result.returncode == 0, result.stdout + result.stderr
    assert _logged_pyright_packages(log_file) == {"fused-memory"}


def test_shared_only_commit_runs_pyright_in_every_package(tmp_path: Path) -> None:
    """`shared` is imported by the other packages, so a commit that stages only
    shared/*.py must run pyright everywhere, shared itself included.
    """
    repo = _make_repo(tmp_path)
    stub_bin, log_file = _make_stub_uv(tmp_path)
    _stage_file(repo, "shared/src/foo.py")

    result = _run_hook(repo, stub_bin, log_file)

    assert result.returncode == 0, result.stdout + result.stderr
    assert _logged_pyright_packages(log_file) == _ALL_PYRIGHT_PACKAGES


def test_escalation_only_commit_runs_pyright_in_every_package(tmp_path: Path) -> None:
    """Same as above for `escalation`, the other shared dependency package."""
    repo = _make_repo(tmp_path)
    stub_bin, log_file = _make_stub_uv(tmp_path)
    _stage_file(repo, "escalation/src/foo.py")

    result = _run_hook(repo, stub_bin, log_file)

    assert result.returncode == 0, result.stdout + result.stderr
    assert _logged_pyright_packages(log_file) == _ALL_PYRIGHT_PACKAGES


@pytest.mark.parametrize("pkg", ["sampler", "cockpit"])
def test_leaf_package_commit_runs_pyright_there_only(tmp_path: Path, pkg: str) -> None:
    """`sampler` and `cockpit` are in the merge gate's type_check_command, so a
    commit staging only their files must type-check them (and nothing else).
    """
    repo = _make_repo(tmp_path)
    stub_bin, log_file = _make_stub_uv(tmp_path)
    _stage_file(repo, f"{pkg}/src/foo.py")

    result = _run_hook(repo, stub_bin, log_file)

    assert result.returncode == 0, result.stdout + result.stderr
    assert _logged_pyright_packages(log_file) == {pkg}
