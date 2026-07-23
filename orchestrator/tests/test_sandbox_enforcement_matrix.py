"""Real-kernel Landlock enforcement-matrix suite.

Pins the 12 rows of ``plans/os-sandbox-worktree-containment-prd.md``'s
§Enforcement matrix (task alpha4; D9; INV-1 machine-pin of the §Write-set
contract) against the REAL production chain: ``compute_write_set()`` ->
``build_landlock_command()`` -> ``landlock_exec.py``, driven on a real
landlock-capable kernel (skip-gated elsewhere).

This is a pure characterization / machine-pin suite: no production code
changes ship with it. Every row already passes against the landed
alpha2 (``compute_write_set``) / alpha3 (workflow.py call-site wiring) /
2970 (``~/.claude`` grant narrowing) stack — the "implementation under
test" is the existing sandbox stack, not new code. The RED->GREEN here is
driven by the shared test harness (the ``landlock_matrix_scaffold``
fixture + ``_run_sandboxed``/``_assert_denied`` helpers) growing per row
group, one PRD row group per test-file step pair.

Modeled on ``test_landlock.py``'s ``TestLandlockEnforcement`` /
``TestLandlockClaudeHomeNarrowing``: class-level skip-if-no-landlock,
``/var/tmp`` scaffolding, the ``_reset_landlock_probe`` autouse fixture,
and driving ``build_landlock_command`` directly (not the backend-agnostic
``wrap_command`` dispatcher).

Everything is built under ``/var/tmp``, never ``/tmp``: ``landlock_exec.py``
blanket-grants ``/tmp`` (``FS_V1_ALL``) for agent scratch, so a scaffold
placed under ``/tmp`` would be writable regardless of the ruleset under
test — silently nullifying every denial row (see ``test_landlock.py``'s
matching rationale at its ``TestLandlockEnforcement`` docstring).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from orchestrator.agents.landlock import (
    _reset_probe as _landlock_reset_probe,
)
from orchestrator.agents.landlock import build_landlock_command, is_landlock_available
from orchestrator.agents.write_set import compute_write_set

# subprocess timeout for each sandboxed inner command — comfortably under
# this suite's 60s per-test pytest-timeout even accounting for scaffold
# setup (real git init/worktree-add), matching test_landlock.py's own
# bounded subprocess.run calls.
_RUN_TIMEOUT = 30

# Denial rows accept EITHER EACCES (landlock) or EROFS (bwrap) as a
# permission-error-class signal (PRD D9: denial errno is backend-specific;
# assertions must say "permission error", not a single errno). This suite
# only ever drives the landlock backend directly (build_landlock_command),
# so only EACCES-shaped messages are actually reachable here, but the
# tolerant token list keeps the shared helper's contract aligned with the
# PRD wording rather than hardcoding one backend's errno.
_PERMISSION_SIGNALS = (
    'Permission denied',
    'Read-only file system',
    'Operation not permitted',
    'EACCES',
    'EROFS',
)


@pytest.fixture(autouse=True)
def _reset_landlock_probe():
    """Reset cached probe before and after each test (mirrors test_landlock.py)."""
    _landlock_reset_probe()
    yield
    _landlock_reset_probe()


def _git(args: list[str], cwd: Path) -> str:
    """Run a git command in ``cwd``, returning stripped stdout.

    Raises ``RuntimeError`` (naming the failed command) on non-zero exit —
    every call site here is scaffold setup or a post-run assertion that is
    expected to succeed outside the sandbox (reads/writes as the real
    test-runner user, never itself sandboxed).
    """
    result = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'git {args} failed in {cwd}: {result.stderr}')
    return result.stdout.strip()


@dataclass
class _Scaffold:
    """A realistic linked-worktree environment for one enforcement-matrix row,
    built entirely under ``/var/tmp`` by the ``landlock_matrix_scaffold``
    fixture (grown incrementally alongside the row groups that need each
    field).
    """

    base: Path
    main: Path
    worktree: Path
    name: str
    home: Path
    task_meta: Path


def _run_sandboxed(
    scaffold: _Scaffold, inner_cmd: list[str], *, cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run ``inner_cmd`` landlock-sandboxed, mirroring workflow.py:9911-9914
    exactly: derive the write-set from the scaffold's worktree/home via the
    real ``compute_write_set()``, feed its ``writable_paths()`` into
    ``build_landlock_command()`` as ``writable_extras`` (never a hand-rolled
    path list — INV-5), and run with ``HOME`` redirected to the scaffold's
    hermetic fake home.
    """
    write_set = compute_write_set(scaffold.worktree, home=scaffold.home)
    cmd = build_landlock_command(
        inner_cmd, scaffold.worktree, [],
        writable_extras=[str(p) for p in write_set.writable_paths()],
    )
    return subprocess.run(
        cmd,
        cwd=str(cwd or scaffold.worktree),
        env={**os.environ, 'HOME': str(scaffold.home)},
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT,
    )


def _assert_denied(result: subprocess.CompletedProcess, *, side_effect_verified: bool) -> None:
    """Assert ``result`` represents a denied write.

    Primary signal (mandatory): non-zero exit. Secondary signal: EITHER a
    permission-error-class token appears in stderr, OR the caller has
    already verified the deterministic side effect itself (target absent /
    ref byte-unchanged) and passes ``side_effect_verified=True`` — per PRD
    D9 the denial errno is backend-specific, so the side-effect check is
    the robust primary pin and the stderr token match is a tolerant bonus.
    """
    assert result.returncode != 0, (
        f'expected denial (non-zero exit); got 0. '
        f'stdout={result.stdout!r} stderr={result.stderr!r}'
    )
    permission_signal = any(tok in result.stderr for tok in _PERMISSION_SIGNALS)
    assert permission_signal or side_effect_verified, (
        f'expected a permission-error-class signal in stderr or a '
        f'caller-verified side effect; stderr={result.stderr!r}'
    )


@pytest.fixture
def landlock_matrix_scaffold():
    """Build a realistic linked-worktree environment under ``/var/tmp``.

    BASE layout (grown per row group across this suite's paired impl
    steps): a real ``git init`` main repo with one seed commit and
    repo-local identity; a real linked worktree created via
    ``git worktree add -b task/<name> <base>/<name> main`` (so its
    ``.git`` gitdir file is exactly what ``compute_write_set`` parses in
    production); ``<base>/.task-meta/<name>/`` (the writable task-meta
    carve-out, via the same path shape ``TaskArtifacts.meta_root_for``
    owns); and a hermetic fake ``HOME`` with ``~/.cache/uv/`` seeded so the
    uv-cache carve-out is grantable at wrap time.

    Built under ``/var/tmp`` (never ``/tmp`` — see module docstring) via
    ``tempfile.mkdtemp``, torn down with ``shutil.rmtree`` regardless of
    test outcome.
    """
    base = Path(tempfile.mkdtemp(prefix='landlock-matrix-', dir='/var/tmp'))
    try:
        main = base / 'main'
        main.mkdir()
        _git(['init', '-b', 'main'], main)
        _git(['config', 'user.email', 'landlock-matrix@test.local'], main)
        _git(['config', 'user.name', 'Landlock Matrix Test'], main)
        (main / 'README.md').write_text('seed\n')
        _git(['add', '-A'], main)
        _git(['commit', '-m', 'seed commit'], main)

        name = 'wt-a'
        worktree = base / name
        _git(['worktree', 'add', '-b', f'task/{name}', str(worktree), 'main'], main)

        task_meta = base / '.task-meta' / name
        task_meta.mkdir(parents=True)

        home = base / 'home'
        home.mkdir()
        (home / '.cache' / 'uv').mkdir(parents=True)

        yield _Scaffold(
            base=base, main=main, worktree=worktree, name=name, home=home,
            task_meta=task_meta,
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.mark.skipif(
    not is_landlock_available(),
    reason='landlock not supported on this kernel',
)
class TestSandboxEnforcementMatrix:
    """The 12 §Enforcement-matrix rows, each driven against the real
    ``compute_write_set()`` -> ``build_landlock_command()`` ->
    ``landlock_exec`` production chain via the ``landlock_matrix_scaffold``
    fixture and ``_run_sandboxed`` helper (grown in the paired impl steps).
    """

    # -- Group 1: base filesystem (rows 1/3/11) --------------------------

    def test_row01_worktree_write_allowed(self, landlock_matrix_scaffold):
        scaffold = landlock_matrix_scaffold
        target = scaffold.worktree / 'src' / 'x.py'
        result = _run_sandboxed(
            scaffold,
            ['/bin/sh', '-c', f'mkdir -p {target.parent} && printf X > {target}'],
        )
        assert result.returncode == 0, result.stderr
        assert target.read_text() == 'X'

    def test_row03_main_canary_denied(self, landlock_matrix_scaffold):
        scaffold = landlock_matrix_scaffold
        target = scaffold.main / 'CANARY'
        result = _run_sandboxed(scaffold, ['/bin/sh', '-c', f'printf X > {target}'])
        assert not target.exists()
        _assert_denied(result, side_effect_verified=True)

    def test_row11_tmp_scratch_allowed(self, landlock_matrix_scaffold):
        scaffold = landlock_matrix_scaffold
        scratch = Path(f'/tmp/landlock-matrix-row11-{uuid.uuid4().hex}')
        try:
            result = _run_sandboxed(scaffold, ['/bin/sh', '-c', f'printf X > {scratch}'])
            assert result.returncode == 0, result.stderr
            assert scratch.read_text() == 'X'
        finally:
            scratch.unlink(missing_ok=True)

    # -- Group 2: neighbor denial (rows 4/5) ------------------------------

    def test_row04_sibling_worktree_denied(self, landlock_matrix_scaffold):
        scaffold = landlock_matrix_scaffold
        target = scaffold.sibling_worktree / 'f'
        result = _run_sandboxed(scaffold, ['/bin/sh', '-c', f'printf X > {target}'])
        assert not target.exists()
        _assert_denied(result, side_effect_verified=True)

    def test_row05_other_task_meta_denied(self, landlock_matrix_scaffold):
        scaffold = landlock_matrix_scaffold
        target = scaffold.other_task_meta / 'x'
        result = _run_sandboxed(scaffold, ['/bin/sh', '-c', f'printf X > {target}'])
        assert not target.exists()
        _assert_denied(result, side_effect_verified=True)
