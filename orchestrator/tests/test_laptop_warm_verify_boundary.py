"""Boundary gate H (task 2309): dispatch<->CLI verify-seam two-way boundary tests.

PRD: plans/laptop-warm-verify-flock-orphan-prd.md SS8 contract + SS9 boundary-test
table.  This is the LEAF integration gate for the internal dispatch (RemoteRunner)
<-> laptop `verify-merge` CLI seam: every test here DRIVES THE REAL SEAM (spawns a
real `orchestrator verify-merge` subprocess, acts on it, observes real
process/tree/escalation-queue/worktree state) rather than asserting on synthetic
inputs.

The "build" driven by each spawned verify-merge is a controllable /bin/bash
sleeper (crafted MergeVerifySpec.test_command) against a throwaway git repo --
never a real reify build.  verify.py's `_run_cmd` runs every module command via
``create_subprocess_shell(cmd, executable='/bin/bash', start_new_session=True)``,
so the sleeper faithfully reproduces the exact cargo/rustc session-escape the
watchdog/cancel tree-kill machinery must handle, with zero reify toolchain
dependency.

Fakes/helpers are reused from the sibling test_cli.py harness (established
cross-test-module import pattern -- see test_concurrent_verify_boundary.py) and
from orchestrator.verify_cancel (the SAME /proc walkers the production
watchdog/cancel kill paths use) -- single source of truth, zero divergent
ad-hoc process-tree parsing.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from test_cli import _setup_verify_repo  # noqa: F401 -- reused cross-module

from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.verify_cancel import (  # noqa: F401 -- reused by row tests
    collect_descendants,
    merge_verify_lock_path,
    pgid_file,
    read_lock_holder_pgid,
    read_ppid_map,
    write_lock_holder_pgid,
)
from orchestrator.verify_runner import (
    FLOCK_CONTENTION_CATEGORY,
    MergeVerifySpec,
    UnscopedTypecheckSpec,
    VerifyCommand,
    result_from_json,
    spec_to_json,
)

# ---------------------------------------------------------------------------
# Config writer
# ---------------------------------------------------------------------------


def write_verify_config(
    path: Path,
    repo: Path,
    *,
    persistent_merge_worktree: bool = True,
    reap_build_artifact_dirs: list[str] | None = None,
) -> Path:
    """Write a minimal orchestrator config YAML for a spawned verify-merge subprocess.

    Only the fields the SS9 rows exercise are set.  ``verify_use_cgroup_scope``
    is deliberately left UNSET (defaults to False -- see config.py) so the
    sleeper build stays on the plain ``start_new_session`` path that
    ``collect_descendants``/``killpg`` reproduction depends on being faithful
    to production (the live reify deployment also runs with this knob False).
    """
    dirs = reap_build_artifact_dirs if reap_build_artifact_dirs is not None else ['target']
    dirs_yaml = ', '.join(dirs)
    path.write_text(
        f'project_root: {repo}\n'
        f'git:\n'
        f'  persistent_merge_worktree: {str(persistent_merge_worktree).lower()}\n'
        f'  reap_build_artifact_dirs: [{dirs_yaml}]\n'
    )
    return path


# ---------------------------------------------------------------------------
# Spec builders -- controllable sleeper / fast-exit builds
# ---------------------------------------------------------------------------


def sleeper_spec(sleep_secs: float = 300.0, *, marker: str = 'target/warm.marker') -> MergeVerifySpec:
    """MergeVerifySpec whose scoped test command touches *marker* then blocks.

    Reproduces the real cargo/rustc start_new_session escape (verify.py
    ``_run_cmd``) with trivial /bin/bash -- see module docstring.
    """
    return MergeVerifySpec(
        verify_commands=(
            VerifyCommand('mod', test_command=f'mkdir -p target && touch {marker} && sleep {sleep_secs}'),
        ),
        unscoped_typecheck=UnscopedTypecheckSpec(
            commands=(VerifyCommand('mod', type_check_command='true'),),
            block_on_timeout=True,
        ),
        task_files=('mod/test_x.py',),
        verify_env={},
        cold_timeout_secs=300.0,
    )


def fast_spec(marker: str = 'target/warm.marker') -> MergeVerifySpec:
    """MergeVerifySpec that touches *marker* and exits cleanly (Row 6 warm path)."""
    return MergeVerifySpec(
        verify_commands=(
            VerifyCommand('mod', test_command=f'mkdir -p target && touch {marker} && true'),
        ),
        unscoped_typecheck=UnscopedTypecheckSpec(
            commands=(VerifyCommand('mod', type_check_command='true'),),
            block_on_timeout=True,
        ),
        task_files=('mod/test_x.py',),
        verify_env={},
        cold_timeout_secs=300.0,
    )


# ---------------------------------------------------------------------------
# Real-CLI spawner
# ---------------------------------------------------------------------------


def verify_merge_argv(
    *, sha: str, spec: MergeVerifySpec, cfg_file: Path, request_id: str | None = None,
) -> list[str]:
    """Build the argv for a real ``orchestrator verify-merge`` invocation."""
    argv = [
        sys.executable, '-c', 'from orchestrator.cli import main; main()',
        'verify-merge',
        '--sha', sha,
        '--spec', spec_to_json(spec),
        '--config', str(cfg_file),
    ]
    if request_id is not None:
        argv += ['--request-id', request_id]
    return argv


def subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Env for a spawned verify-merge/dispatcher subprocess.

    Mirrors test_cli.py's real e2e cancel test: PYTHONPATH is set so the child
    imports this worktree's src (not the editable-install main-checkout), and
    ORCH_PROJECT_ROOT is popped because the ``_isolate_orch_config`` autouse
    fixture sets it for the PARENT (pytest) process, which would otherwise
    override the YAML's project_root (pydantic-settings env_settings beats
    yaml_settings) and point git worktree operations at the wrong repo.
    """
    worktree_src = str(Path(__file__).parent.parent / 'src')
    env = dict(os.environ)
    existing_pp = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f'{worktree_src}:{existing_pp}' if existing_pp else worktree_src
    env.pop('ORCH_PROJECT_ROOT', None)
    if extra:
        env.update(extra)
    return env


def spawn_verify_merge(
    *,
    sha: str,
    spec: MergeVerifySpec,
    cfg_file: Path,
    request_id: str | None = None,
    stdin: int | None = subprocess.PIPE,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen:
    """Spawn a real ``orchestrator verify-merge`` subprocess.

    ``stdin=subprocess.PIPE`` by default so callers can drive the
    connection-death protocol (heartbeat writer / EOF-on-close); tests that
    don't pass --request-id never arm the watchdog so stdin is inert for them.
    """
    return subprocess.Popen(
        verify_merge_argv(sha=sha, spec=spec, cfg_file=cfg_file, request_id=request_id),
        env=subprocess_env(extra_env),
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def spawn_ssh_heartbeat_dispatcher(
    *, argv: list[str], cwd: str | None = None, heartbeat_interval: float = 0.2,
) -> subprocess.Popen:
    """Spawn a SEPARATE process running the REAL verify_runner._default_ssh_heartbeat_run.

    Row 1 (orchestrator killed): this process stands in for "the orchestrator
    that owns the ssh child".  SIGKILLing the returned Popen closes ITS end of
    *argv*'s stdin pipe when the OS reclaims its file descriptors, giving the
    grandchild (verify-merge) a clean EOF on fd 0 -- exactly the connection-
    death signal the stdin watchdog (verify_cancel.run_stdin_watchdog) reacts
    to, without needing any process-group trickery.
    """
    worktree_src = str(Path(__file__).parent.parent / 'src')
    script = (
        'import asyncio, sys\n'
        f'sys.path.insert(0, {worktree_src!r})\n'
        'from orchestrator.verify_runner import _default_ssh_heartbeat_run\n'
        f'argv = {argv!r}\n'
        f'cwd = {cwd!r}\n'
        f'asyncio.run(_default_ssh_heartbeat_run(argv, cwd=cwd, heartbeat_interval={heartbeat_interval!r}))\n'
    )
    return subprocess.Popen(
        [sys.executable, '-c', script],
        env=subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


# ---------------------------------------------------------------------------
# In-process controllable dispatcher (Rows 2/3/4/6)
# ---------------------------------------------------------------------------


class HeartbeatWriter:
    """In-process heartbeat writer driving a spawned verify-merge child's stdin.

    Mirrors the real dispatcher half of the connection-death protocol
    (verify_runner._default_ssh_heartbeat_run) but exposes stop_heartbeats()/
    close_stdin() separately so a single test process can script every
    distinct connection-death mode (EOF-while-alive vs. silent-partition)
    without spawning a second process.
    """

    def __init__(self, proc: subprocess.Popen, *, interval: float = 0.2):
        self._proc = proc
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> 'HeartbeatWriter':
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._stop.wait(self._interval):
                return
            try:
                assert self._proc.stdin is not None
                self._proc.stdin.write(b'\n')
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                return

    def stop_heartbeats(self) -> None:
        """Stop sending heartbeats WITHOUT closing stdin (Row 3: hard partition)."""
        self._stop.set()

    def close_stdin(self) -> None:
        """Stop heartbeats and close the child's stdin write-end (Row 2: EOF, dispatcher alive)."""
        self._stop.set()
        self._thread.join(timeout=2)
        assert self._proc.stdin is not None
        with contextlib.suppress(OSError):
            self._proc.stdin.close()


# ---------------------------------------------------------------------------
# Process-tree / pgid-file observers (reuse verify_cancel's real /proc walkers
# -- the SAME functions the production watchdog/cancel kill paths use)
# ---------------------------------------------------------------------------


def wait_for_pgid_file(path: Path, *, timeout: float = 20.0, interval: float = 0.05) -> int:
    """Poll for a pgid file (written by verify-merge --request-id) and return its int value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            with contextlib.suppress(ValueError):
                return int(path.read_text().strip())
        time.sleep(interval)
    raise AssertionError(f'pgid file {path} did not appear within {timeout}s')


def wait_subtree_live(pgid: int, *, timeout: float = 20.0, interval: float = 0.05) -> set[int]:
    """Poll until *pgid* has at least one live descendant; return the descendant set.

    Raises AssertionError on timeout -- a live sleeper never appearing means
    the harness itself is broken, not a seam defect under test.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        descendants = collect_descendants(pgid, read_ppid_map())
        if descendants:
            return descendants
        time.sleep(interval)
    raise AssertionError(f'pgid {pgid}: no descendant appeared within {timeout}s')


def subtree_and_leader_gone(pgid: int) -> bool:
    """True when *pgid* has no live descendants AND the pgid leader itself is gone."""
    if collect_descendants(pgid, read_ppid_map()):
        return False
    try:
        os.killpg(pgid, 0)
        return False  # group still alive
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


def wait_subtree_gone(pgid: int, *, timeout: float, interval: float = 0.1) -> bool:
    """Poll until subtree_and_leader_gone(pgid); return the final boolean (no raise)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if subtree_and_leader_gone(pgid):
            return True
        time.sleep(interval)
    return subtree_and_leader_gone(pgid)


def worktree_base_for(repo: Path) -> Path:
    """Derive worktree_base exactly as the spawned CLI does: GitOps(config.git, repo).worktree_base."""
    config = OrchestratorConfig(project_root=repo)
    return GitOps(config.git, repo).worktree_base


# ---------------------------------------------------------------------------
# Task 2309 step-1 RED -- env-var test seams for remote-side timing constants
# (PRD SS11 Q1/Q2 tunability).  Production defaults are byte-identical when
# unset; these overrides exist ONLY so the SS9 boundary rows below can run
# fast and deterministically instead of being wall-clock-bound on the 10s
# flock wait / 10s+5s watchdog window.  RED until step-2 wires them into
# cli.py.
# ---------------------------------------------------------------------------


def test_flock_wait_env_override_speeds_up_contention_result(tmp_path):
    """ORCH_MERGE_VERIFY_FLOCK_WAIT_SECS overrides the flock bounded wait.

    Holds the real flock (+ writes the holder pgid) exactly as
    test_cli.py:1419 does, then spawns a real knob-on verify-merge with the
    override set small.  Today (RED) verify-merge ignores the env var and
    waits the full 10.0s production window; asserting completion in well
    under that (< 3s) fails until cli.py reads the override.
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    worktree_base = worktree_base_for(repo)
    worktree_base.mkdir(parents=True, exist_ok=True)

    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=True)

    write_lock_holder_pgid(worktree_base, 999999)
    lock_path = merge_verify_lock_path(worktree_base)
    held_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
    fcntl.flock(held_fd, fcntl.LOCK_EX)
    try:
        started = time.monotonic()
        proc = spawn_verify_merge(
            sha=head_sha,
            spec=fast_spec(),
            cfg_file=cfg_file,
            extra_env={'ORCH_MERGE_VERIFY_FLOCK_WAIT_SECS': '0.5'},
        )
        stdout, stderr = proc.communicate(timeout=15)
        elapsed = time.monotonic() - started
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(held_fd, fcntl.LOCK_UN)
        os.close(held_fd)

    assert proc.returncode == 0, (
        f'expected exit 0 (contention result on stdout), got {proc.returncode}; '
        f'stderr={stderr.decode()[:2000]!r}'
    )
    result = result_from_json(stdout.decode())
    assert result.category == FLOCK_CONTENTION_CATEGORY, (
        f'expected flock-contention result, got category={result.category!r} '
        f'stdout={stdout.decode()[:2000]!r}'
    )
    assert elapsed < 3.0, (
        f'expected contention result well under the 10s production wait '
        f'(env override=0.5s) -- took {elapsed:.2f}s; the env override is '
        f'not wired up yet'
    )


def test_watchdog_timeout_env_override_fires_fast_without_heartbeat(tmp_path):
    """ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS/_KILL_GRACE_SECS override the watchdog window.

    Spawns a real verify-merge --request-id with stdin=PIPE and NEVER writes
    a heartbeat.  Today (RED) the watchdog uses the 10s+5s production window;
    asserting the process self-exits non-zero within a few seconds fails
    until cli.py threads the overrides into start_stdin_watchdog.
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=False)

    proc = spawn_verify_merge(
        sha=head_sha,
        spec=sleeper_spec(300.0),
        cfg_file=cfg_file,
        request_id='env-seam-watchdog-test',
        extra_env={
            'ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS': '0.5',
            'ORCH_WATCHDOG_KILL_GRACE_SECS': '0.2',
        },
    )
    try:
        # No heartbeat is ever written -- stdin stays open but silent, which
        # is sufficient to exercise the heartbeat-starvation timing path
        # (Rows 1-3 later distinguish EOF vs. timeout precisely).
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pytest.fail(
                'verify-merge did not self-exit within 5s of no heartbeats -- '
                'watchdog env overrides are not wired up yet (production '
                'window is 10s timeout + 5s grace = 15s)'
            )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        if proc.stdin is not None:
            with contextlib.suppress(OSError):
                proc.stdin.close()

    assert proc.returncode != 0, (
        f'expected non-zero exit (watchdog self-kill), got {proc.returncode}'
    )
