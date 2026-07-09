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

import asyncio
import contextlib
import fcntl
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner
from escalation.queue import EscalationQueue
from test_cli import _setup_verify_repo  # noqa: F401 -- reused cross-module
from test_merge_queue_multihost_wiring import (  # noqa: F401 -- reused cross-module
    _make_config as _mq_make_config,
)
from test_merge_queue_multihost_wiring import (
    _make_git_ops_mock as _mq_make_git_ops_mock,
)
from test_merge_queue_multihost_wiring import (
    _make_merge_request as _mq_make_merge_request,
)

import orchestrator.cli as cli_module
from orchestrator.cli import main
from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_queue import _run_post_merge_verify, _verify_worktree_contention_sentinel
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
    _default_ssh_heartbeat_run,
    is_flock_contention_failure,
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


def apply_dispatcher_env(monkeypatch) -> None:
    """Patch THIS process's os.environ for a direct (in-process) real-dispatcher call.

    :func:`~orchestrator.verify_runner._default_ssh_heartbeat_run` takes no
    ``env=`` override -- it always spawns its child via
    ``asyncio.create_subprocess_exec`` with the ambient ``os.environ``.  When
    a test calls it directly (rather than through a wrapper subprocess with
    its own explicit env dict, as :func:`spawn_verify_merge` uses), the
    ambient environment must get the same treatment as :func:`subprocess_env`:
    PYTHONPATH pointed at this worktree's ``src`` and ``ORCH_PROJECT_ROOT``
    popped (the ``_isolate_orch_config`` autouse fixture sets it to the
    *pytest* tmp_path, which would otherwise beat the ``--config`` YAML's
    project_root -- see subprocess_env's docstring).  Uses the monkeypatch
    fixture so the mutation reverts at test teardown regardless of outcome.
    """
    worktree_src = str(Path(__file__).parent.parent / 'src')
    existing_pp = os.environ.get('PYTHONPATH', '')
    monkeypatch.setenv(
        'PYTHONPATH', f'{worktree_src}:{existing_pp}' if existing_pp else worktree_src
    )
    monkeypatch.delenv('ORCH_PROJECT_ROOT', raising=False)


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
    *,
    argv: list[str],
    cwd: str | None = None,
    heartbeat_interval: float = 0.2,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen:
    """Spawn a SEPARATE process running the REAL verify_runner._default_ssh_heartbeat_run.

    Row 1 (orchestrator killed): this process stands in for "the orchestrator
    that owns the ssh child".  SIGKILLing the returned Popen closes ITS end of
    *argv*'s stdin pipe when the OS reclaims its file descriptors, giving the
    grandchild (verify-merge) a clean EOF on fd 0 -- exactly the connection-
    death signal the stdin watchdog (verify_cancel.run_stdin_watchdog) reacts
    to, without needing any process-group trickery.

    *extra_env* lands in the DISPATCHER's own environment.  Since
    ``_default_ssh_heartbeat_run`` spawns *argv* via
    ``asyncio.create_subprocess_exec`` with no explicit ``env=`` override, the
    grandchild (verify-merge) ambiently inherits it too -- the same
    one-more-remove mechanism :func:`apply_dispatcher_env` documents for the
    in-process case.  Used to thread the step-2 watchdog env-seam overrides
    through for a fast, deterministic Row 1 assertion window.
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
        env=subprocess_env(extra_env),
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

    def start(self) -> HeartbeatWriter:
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

    NOTE: "has a live descendant" fires on the FIRST child the CLI forks --
    for a not-yet-materialised persistent worktree that is a transient ``git``
    subprocess (``git worktree add`` / ``git reset`` / ``git clean`` inside
    ``acquire_host_verify_worktree``), not necessarily the eventual build
    shell.  Callers that need to observe a build-produced artifact (e.g. a
    marker file the build's shell command touches) must poll for that
    artifact directly -- see :func:`wait_for_marker` -- rather than treating
    "subtree live" as a proxy for "build has started".
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        descendants = collect_descendants(pgid, read_ppid_map())
        if descendants:
            return descendants
        time.sleep(interval)
    raise AssertionError(f'pgid {pgid}: no descendant appeared within {timeout}s')


def wait_for_marker(path: Path, *, timeout: float = 20.0, interval: float = 0.05) -> None:
    """Poll for a build-produced marker file to appear; raise AssertionError on timeout.

    Unlike :func:`wait_subtree_live` (which only proves the CLI has forked
    *some* child -- possibly a transient ``git`` setup subprocess), this
    proves the build's shell command has actually executed its
    ``touch <marker>`` step, which is the real precondition tests need before
    reading the marker's mtime or asserting on its retention.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(interval)
    raise AssertionError(f'holder build did not materialize {path} within {timeout}s')


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
        # task 2376: widened from 15s -- host oversubscription can delay
        # subprocess completion past a short deadline; the discriminating
        # invariant is the `elapsed < 9.0` assertion below, not this ceiling.
        stdout, stderr = proc.communicate(timeout=45)
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
    # task 2376: widened from 6.0s -- must stay STRICTLY below the 10.0s
    # production flock-wait so this still detects an un-wired override
    # (i.e. it fails to distinguish the 0.5s override from the 10s
    # production default if raised to >= 10.0).
    assert elapsed < 9.0, (
        f'expected contention result well under the 10s production wait '
        f'(env override=0.5s, generous ceiling for subprocess-startup '
        f'jitter) -- took {elapsed:.2f}s; the env override is not wired up '
        f'yet'
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
            proc.wait(timeout=9.0)
        except subprocess.TimeoutExpired:
            pytest.fail(
                'verify-merge did not self-exit within 9s of no heartbeats -- '
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


# ---------------------------------------------------------------------------
# Task 2309 step-3/4 -- SS9 Row 6: normal warm path (also the harness smoke
# test).  Real dispatcher (live heartbeat), run twice on the same
# worktree_base.  Exercises SS8.1's happy-path invariant (heartbeat must not
# alter VerifyResult) + PRD outcome #1 (retained target/ across merges).
# ---------------------------------------------------------------------------


def test_normal_warm_path_reuses_fixed_worktree_twice_no_escalation(tmp_path, monkeypatch):
    """SS9 Row 6: two consecutive warm verify-merge runs, real live-heartbeat dispatcher.

    A knob-on config + a --request-id'd verify-merge is driven by the REAL
    dispatcher (verify_runner._default_ssh_heartbeat_run, called directly --
    in-process asyncio, not a wrapper subprocess -- so its concurrent
    heartbeat-writer task sends real heartbeats down the child's real stdin
    pipe every 0.2s; see PRD SS8.1) run TWICE against the SAME worktree_base.

    Asserts:

    * both runs use the FIXED <worktree_base>/_merge-verify path -- no
      ephemeral _merge-<uuid> dir is ever created (PRD SS8 eta);
    * target/warm.marker (written by the first run's build) survives the
      second run's reset_persistent_merge_worktree -- reap_build_artifact_dirs
      retention (PRD SS10 invariant 1: source bit-identical to fresh checkout,
      build-cache dirs retained for warmth);
    * both VerifyResults are passed=True, category=='passed' (rc==0's
      ``_classify_failure`` value -- verify.py:643-644 -- not one of the
      failure sentinels like FLOCK_CONTENTION_CATEGORY), contention is None --
      structurally unchanged by the live heartbeat;
    * is_flock_contention_failure(result) is False for both -- the REAL beta
      discriminant predicate that gates _run_post_merge_verify's escalation
      branch (task 2307 beta) never fires on this happy path, so nothing
      downstream would file a born-at-L2 escalation.  (A CLI-only harness
      has no merge_queue in the loop to observe an EscalationQueue directly
      -- Row 5 (step-13/14) drives that consumer explicitly; this is the
      producer-side proof that the happy path never emits the discriminant
      the consumer keys on.)
    * the watchdog never fires -- both runs exit 0 via normal completion,
      not the watchdog's non-zero self-kill exit.
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    worktree_base = worktree_base_for(repo)
    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=True)

    apply_dispatcher_env(monkeypatch)

    def run_once(request_id: str):
        argv = verify_merge_argv(
            sha=head_sha, spec=fast_spec(), cfg_file=cfg_file, request_id=request_id,
        )
        rc, out, err = asyncio.run(
            _default_ssh_heartbeat_run(argv, heartbeat_interval=0.2)
        )
        assert rc == 0, (
            f'verify-merge (request_id={request_id!r}) exited {rc} (watchdog '
            f'fired?) stderr={err[:2000]!r}'
        )
        return result_from_json(out)

    result_1 = run_once('row6-run-1')
    result_2 = run_once('row6-run-2')

    persistent_wt = worktree_base / '_merge-verify'
    assert persistent_wt.is_dir(), f'fixed warm worktree missing: {persistent_wt}'

    leaked_ephemeral = sorted(
        p.name for p in worktree_base.iterdir()
        if p.name.startswith('_merge-') and p.name != '_merge-verify'
    )
    assert leaked_ephemeral == [], (
        f'ephemeral _merge-<uuid> dir(s) leaked (warm path should never '
        f'create one): {leaked_ephemeral}'
    )

    target_dir = persistent_wt / 'target'
    marker = target_dir / 'warm.marker'
    assert marker.exists(), (
        f'target/warm.marker missing after both runs -- reap_build_artifact_dirs '
        f'retention did not survive reset_persistent_merge_worktree: {marker}'
    )
    assert any(target_dir.iterdir()), f'target/ retained but empty: {target_dir}'

    for n, result in ((1, result_1), (2, result_2)):
        assert result.passed is True, f'run {n}: expected passed=True, got {result!r}'
        assert result.category == 'passed', (
            f'run {n}: expected the happy-path category (rc==0 -> "passed"), got '
            f'{result.category!r}'
        )
        assert result.contention is None, (
            f'run {n}: expected no contention payload, got {result.contention!r}'
        )
        assert is_flock_contention_failure(result) is False, (
            f'run {n}: real beta discriminant fired on a happy-path result -- '
            f'_run_post_merge_verify would wrongly escalate: {result!r}'
        )


# ---------------------------------------------------------------------------
# Task 2309 step-5/6 -- SS9 Row 4: cancel-verify under a LIVE watchdog (B2
# coexistence).  Extends test_cli.py:1656's real e2e cancel pattern with a
# live HeartbeatWriter so the stdin watchdog stays armed throughout but never
# self-fires, proving an explicit cancel-verify tree-kills cleanly anyway.
# ---------------------------------------------------------------------------


def test_cancel_verify_tree_kills_under_live_watchdog(tmp_path, monkeypatch):
    """SS9 Row 4: cancel-verify leaves no orphan while the stdin watchdog is live.

    Spawns a real ``verify-merge --request-id`` with a long-running sleeper
    spec (the start_new_session escape) and a LIVE :class:`HeartbeatWriter`
    driving its stdin, so :func:`verify_cancel.run_stdin_watchdog` stays armed
    for the whole test but never times out or self-fires.  Waits for the
    sleeper subtree to appear, then cancels via ``cancel-verify
    --request-id`` through :class:`CliRunner` (the same in-process pattern as
    test_cli.py's ``test_verify_merge_cancel_end_to_end``).

    Asserts:

    * cancel-verify exits 0 and removes the pgid file;
    * the verify-merge subprocess exits promptly;
    * the FULL descendant subtree (including the start_new_session sleeper
      escape) AND the pgid leader itself are gone (``collect_descendants``
      empty; ``os.killpg(pgid, 0)`` raises ``ProcessLookupError``) -- the live
      watchdog and an explicit cancel coexist harmlessly and no orphan
      survives.
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    config_obj = OrchestratorConfig(project_root=repo)
    worktree_base = worktree_base_for(repo)
    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=False)

    REQUEST_ID = 'row4-cancel-under-watchdog'
    pgf = pgid_file(worktree_base, REQUEST_ID)

    child = spawn_verify_merge(
        sha=head_sha, spec=sleeper_spec(300.0), cfg_file=cfg_file, request_id=REQUEST_ID,
    )
    heartbeat = HeartbeatWriter(child, interval=0.2).start()
    try:
        pgid_val = wait_for_pgid_file(pgf)
        wait_subtree_live(pgid_val)

        monkeypatch.setattr(cli_module, 'load_config', lambda _path: config_obj)
        result = CliRunner().invoke(main, [
            'cancel-verify', '--request-id', REQUEST_ID, '--config', str(cfg_file),
        ])
        assert result.exit_code == 0, (
            f'cancel-verify expected exit 0, got {result.exit_code}; '
            f'output={result.output!r}'
        )
        assert not pgf.exists(), 'pgid file must be removed by cancel-verify on success'

        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pytest.fail(
                'verify-merge subprocess did not exit within 20s after cancel-verify'
            )

        assert wait_subtree_gone(pgid_val, timeout=10.0), (
            f'pgid {pgid_val}: subtree and/or leader still alive after '
            f'cancel-verify (live-watchdog coexistence left an orphan)'
        )
    finally:
        heartbeat.stop_heartbeats()
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


# ---------------------------------------------------------------------------
# Task 2309 step-7/8 -- SS9 Row 1: orchestrator (dispatcher) killed mid-build
# (EOF path via whole-dispatcher death).  A SEPARATE killable process runs the
# REAL verify_runner._default_ssh_heartbeat_run; SIGKILLing it closes its end
# of the child's stdin pipe, delivering a clean EOF -- distinct from Row 3's
# heartbeat-timeout (select-timeout) branch, which keeps stdin open but silent.
# ---------------------------------------------------------------------------


def test_orchestrator_killed_mid_build_tree_killed_via_eof(tmp_path):
    """SS9 Row 1: dispatcher process killed -> child sees stdin EOF -> tree-killed.

    Models "the orchestrator holding the ssh child died": spawns a SEPARATE
    dispatcher process running the REAL ``_default_ssh_heartbeat_run`` against
    a local ``verify-merge --request-id`` argv (small heartbeat_interval so
    real heartbeats flow while the dispatcher is alive).  Waits for the
    sleeper subtree to appear, then SIGKILLs the dispatcher process itself --
    when the OS reclaims its file descriptors, ITS end of the child's stdin
    pipe closes, giving the grandchild a clean EOF on fd 0.

    The step-2 env seam is threaded through the dispatcher's own environment
    (:func:`spawn_ssh_heartbeat_dispatcher`'s *extra_env*, ambiently inherited
    by the grandchild verify-merge) so the assertion window is small and
    deterministic rather than the 10s+5s production window.  Per
    ``run_stdin_watchdog``, EOF fires on the very next ``select`` readiness
    check regardless of *heartbeat_timeout* -- only ``grace_secs`` (the
    SIGTERM->SIGKILL pause in ``fire_watchdog_kill``) materially bounds this
    row's timing; both overrides are set for a documented, generous ceiling.

    Asserts within a bounded T: the full descendant subtree (including the
    start_new_session sleeper escape) AND the pgid leader are gone.
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=False)
    worktree_base = worktree_base_for(repo)

    REQUEST_ID = 'row1-orchestrator-killed'
    pgf = pgid_file(worktree_base, REQUEST_ID)

    argv = verify_merge_argv(
        sha=head_sha, spec=sleeper_spec(300.0), cfg_file=cfg_file, request_id=REQUEST_ID,
    )
    dispatcher = spawn_ssh_heartbeat_dispatcher(
        argv=argv,
        heartbeat_interval=0.2,
        extra_env={
            'ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS': '1.0',
            'ORCH_WATCHDOG_KILL_GRACE_SECS': '0.5',
        },
    )
    try:
        pgid_val = wait_for_pgid_file(pgf)
        wait_subtree_live(pgid_val)

        dispatcher.kill()
        dispatcher.wait(timeout=10)

        assert wait_subtree_gone(pgid_val, timeout=10.0), (
            f'pgid {pgid_val}: subtree and/or leader still alive after the '
            f'dispatcher was killed (EOF-triggered watchdog tree-kill did '
            f'not fire)'
        )
    finally:
        if dispatcher.poll() is None:
            dispatcher.kill()
            dispatcher.wait(timeout=5)


# ---------------------------------------------------------------------------
# Task 2309 step-9/10 -- SS9 Row 2: ssh connection dropped mid-build (EOF
# path, dispatcher ALIVE).  Distinct from Row 1 (whole dispatcher process
# death): here the in-process dispatcher (this test) owns the child's
# stdin=PIPE directly and closes ONLY the write end via
# HeartbeatWriter.close_stdin(), while the dispatcher/test process itself
# keeps running throughout -- modeling "the ssh transport dropped", not "the
# orchestrator died".
# ---------------------------------------------------------------------------


def test_ssh_dropped_mid_build_tree_killed_via_eof_dispatcher_alive(tmp_path):
    """SS9 Row 2: ssh channel closes but the dispatcher stays alive -> EOF tree-kill.

    Spawns a real ``verify-merge --request-id`` with a live
    :class:`HeartbeatWriter` driving its stdin (real heartbeats flow while
    connected), waits for the sleeper subtree to appear, then calls
    ``heartbeat.close_stdin()`` -- stopping heartbeats AND closing the
    child's stdin write-end -- while this test process (the dispatcher
    stand-in) keeps running.  The child sees the same fd-0 EOF signal as Row
    1 and the same ``run_stdin_watchdog`` EOF branch fires; the difference
    from Row 1 is that no dispatcher PROCESS dies here, only the transport.

    Uses the step-2 env seam directly on the spawned verify-merge (already
    an in-process dispatcher, so no extra remove is needed, unlike Row 1's
    separate-process case) for a fast, deterministic assertion window.

    Asserts within a bounded T: the full descendant subtree AND the pgid
    leader are gone, and the verify-merge process itself self-exits non-zero
    (the watchdog's controlled ``os._exit(1)``, not an external kill signal).
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=False)
    worktree_base = worktree_base_for(repo)

    REQUEST_ID = 'row2-ssh-dropped'
    pgf = pgid_file(worktree_base, REQUEST_ID)

    child = spawn_verify_merge(
        sha=head_sha,
        spec=sleeper_spec(300.0),
        cfg_file=cfg_file,
        request_id=REQUEST_ID,
        extra_env={
            'ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS': '1.0',
            'ORCH_WATCHDOG_KILL_GRACE_SECS': '0.5',
        },
    )
    heartbeat = HeartbeatWriter(child, interval=0.2).start()
    try:
        pgid_val = wait_for_pgid_file(pgf)
        wait_subtree_live(pgid_val)

        heartbeat.close_stdin()

        # Reap the leader (a DIRECT child of this test process, unlike Row
        # 1's separate-dispatcher indirection or Row 4's cancel-verify path)
        # BEFORE polling wait_subtree_gone: os.killpg(pgid, 0) succeeds
        # against an unreaped zombie too, so checking liveness first would
        # spuriously see the group as "alive" until something calls
        # child.wait() -- confirmed by a manual repro that hung the full
        # poll window with this ordering reversed.
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pytest.fail('verify-merge did not exit within 10s after stdin EOF')
        assert child.returncode != 0, (
            f'expected non-zero exit (watchdog self-kill), got {child.returncode}'
        )

        assert wait_subtree_gone(pgid_val, timeout=10.0), (
            f'pgid {pgid_val}: subtree and/or leader still alive after the '
            f'ssh channel EOF (dispatcher alive) -- watchdog tree-kill did '
            f'not fire'
        )
    finally:
        heartbeat.stop_heartbeats()
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


# ---------------------------------------------------------------------------
# Task 2309 step-11/12 -- SS9 Row 3: heartbeat starved / simulated hard
# partition (heartbeat-TIMEOUT path).  Distinct from Rows 1/2 (both EOF-
# triggered): stdin stays OPEN throughout, only heartbeats stop, exercising
# run_stdin_watchdog's select-timeout branch (empty ready set) rather than
# its EOF branch (readable, read() == b'').
# ---------------------------------------------------------------------------


def test_heartbeat_starved_hard_partition_tree_killed_via_timeout(tmp_path):
    """SS9 Row 3: heartbeat starved, stdin stays OPEN -> select-timeout tree-kill.

    Spawns a real ``verify-merge --request-id`` with a live
    :class:`HeartbeatWriter`, waits for the sleeper subtree to appear, then
    calls ``heartbeat.stop_heartbeats()`` -- which stops writing WITHOUT
    closing stdin (contrast :func:`HeartbeatWriter.close_stdin`, Row 2) --
    modeling a hard network partition where the channel is never cleanly
    closed.  ``run_stdin_watchdog``'s ``select()`` call times out (empty
    ready set) rather than seeing EOF, taking the branch that calls
    ``on_fire`` directly without ever attempting a read.

    Uses the step-2 env seam directly on the spawned verify-merge for a
    fast, deterministic assertion window bounded by
    ``~(heartbeat_timeout + grace_secs)``.

    Asserts the leader self-exits non-zero and, after reaping it (same
    zombie-avoidance ordering as Row 2 -- ``child`` is again a direct child
    of this test process), that the full subtree and pgid leader are gone.
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=False)
    worktree_base = worktree_base_for(repo)

    REQUEST_ID = 'row3-heartbeat-starved'
    pgf = pgid_file(worktree_base, REQUEST_ID)

    child = spawn_verify_merge(
        sha=head_sha,
        spec=sleeper_spec(300.0),
        cfg_file=cfg_file,
        request_id=REQUEST_ID,
        extra_env={
            'ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS': '1.0',
            'ORCH_WATCHDOG_KILL_GRACE_SECS': '0.5',
        },
    )
    heartbeat = HeartbeatWriter(child, interval=0.2).start()
    try:
        pgid_val = wait_for_pgid_file(pgf)
        wait_subtree_live(pgid_val)

        heartbeat.stop_heartbeats()
        assert child.stdin is not None and not child.stdin.closed, (
            'harness bug: stdin must stay OPEN for Row 3 (hard partition) -- '
            'only heartbeats stop; this is what distinguishes it from Row 2'
        )

        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pytest.fail(
                'verify-merge did not self-exit within 10s of heartbeat '
                'starvation (select-timeout branch did not fire)'
            )
        assert child.returncode != 0, (
            f'expected non-zero exit (watchdog self-kill), got {child.returncode}'
        )

        assert wait_subtree_gone(pgid_val, timeout=10.0), (
            f'pgid {pgid_val}: subtree and/or leader still alive after '
            f'heartbeat starvation -- watchdog tree-kill did not fire'
        )
    finally:
        heartbeat.stop_heartbeats()
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        if child.stdin is not None:
            with contextlib.suppress(OSError):
                child.stdin.close()


# ---------------------------------------------------------------------------
# Task 2309 step-13/14 -- SS9 Row 5: flock contention, FULL two-way seam
# (producer + consumer).  The load-bearing SS8.2 seam assertion: a real
# verify-merge #1 holds .merge_verify.lock mid-build; a real verify-merge #2
# on the SAME worktree_base reports the FLOCK_CONTENTION_CATEGORY
# discriminant on stdout WITHOUT ever touching the tree; #2's parsed stdout
# is then fed through the REAL merge_queue consumer (_run_post_merge_verify +
# a real EscalationQueue), asserting a born-at-L2 escalation is filed and the
# MergeOutcome is 'blocked'.
# ---------------------------------------------------------------------------


def test_flock_contention_full_two_way_seam_blocks_and_escalates(tmp_path):
    """SS9 Row 5: producer discriminant -> real consumer -> born-at-L2 + blocked.

    Producer side: a real ``verify-merge #1`` (sleeper build, --request-id)
    materialises the persistent ``_merge-verify`` worktree and holds the real
    ``.merge_verify.lock`` flock for the full duration of its build (cli.py
    acquires the flock BEFORE awaiting the build and releases it only in the
    outer ``finally``).  A real ``verify-merge #2`` on the SAME worktree_base
    with the step-2 env seam (small flock wait) then reports contention.

    Asserts on #2: exits 0 with the FLOCK_CONTENTION_CATEGORY discriminant on
    stdout, contention={host, holder_pgid, waiter_pgid} with holder_pgid
    matching #1's own pgid; no ``_merge-<uuid>`` ephemeral dir is ever
    created; #1's ``_merge-verify`` tree is never touched (marker file mtime
    unchanged -- #2's code path returns before
    ``acquire_host_verify_worktree`` is ever called on contention).

    Consumer side: parses #2's stdout via ``result_from_json`` and feeds it
    through the REAL ``_run_post_merge_verify`` (a stub runner returning that
    exact parsed result -- the established pattern from
    ``TestRunPostMergeVerifyFlockContention`` in
    ``test_merge_queue_multihost_wiring.py``) with a REAL
    ``EscalationQueue(tmp_path)``.  Asserts a born-at-L2 escalation is queued
    (level=2, severity='critical',
    agent_role='orchestrator-verify-host-monitor',
    category='verify_worktree_contention', detail names both pgids) and the
    returned ``MergeOutcome`` is 'blocked'.
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=True)
    worktree_base = worktree_base_for(repo)

    HOLDER_REQUEST_ID = 'row5-holder'
    pgf_holder = pgid_file(worktree_base, HOLDER_REQUEST_ID)

    holder = spawn_verify_merge(
        sha=head_sha, spec=sleeper_spec(300.0), cfg_file=cfg_file, request_id=HOLDER_REQUEST_ID,
    )
    try:
        holder_pgid_val = wait_for_pgid_file(pgf_holder)
        wait_subtree_live(holder_pgid_val)

        persistent_wt = worktree_base / '_merge-verify'
        marker = persistent_wt / 'target' / 'warm.marker'
        wait_for_marker(marker)
        marker_mtime_before = marker.stat().st_mtime_ns

        waiter = spawn_verify_merge(
            sha=head_sha,
            spec=fast_spec(),
            cfg_file=cfg_file,
            request_id='row5-waiter',
            extra_env={'ORCH_MERGE_VERIFY_FLOCK_WAIT_SECS': '0.5'},
        )
        stdout, stderr = waiter.communicate(timeout=15)

        assert waiter.returncode == 0, (
            f'expected exit 0 (contention result on stdout), got '
            f'{waiter.returncode}; stderr={stderr.decode()[:2000]!r}'
        )
        result = result_from_json(stdout.decode())
        assert is_flock_contention_failure(result), (
            f'expected the flock-contention discriminant, got {result!r}'
        )
        assert result.category == FLOCK_CONTENTION_CATEGORY
        assert result.contention is not None
        assert result.contention['host'] == socket.gethostname()
        assert result.contention['holder_pgid'] == holder_pgid_val
        waiter_pgid_val = result.contention['waiter_pgid']
        assert waiter_pgid_val != holder_pgid_val

        leaked_ephemeral = sorted(
            p.name for p in worktree_base.iterdir()
            if p.name.startswith('_merge-') and p.name != '_merge-verify'
        )
        assert leaked_ephemeral == [], (
            f'waiter must never create an ephemeral _merge-<uuid> dir on '
            f'contention: {leaked_ephemeral}'
        )
        assert marker.stat().st_mtime_ns == marker_mtime_before, (
            "waiter mutated the holder's _merge-verify tree on contention "
            "(marker mtime changed)"
        )
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=5)

    # --- Consumer side: feed #2's real discriminant through the real beta
    # consumer (_run_post_merge_verify) with a real EscalationQueue. ---

    stub_runner = MagicMock()
    stub_runner.is_local = False
    stub_runner.run_merge_verify = AsyncMock(return_value=result)

    eq = EscalationQueue(tmp_path / 'escalations')

    async def _drive_consumer():
        config = _mq_make_config()
        req = _mq_make_merge_request(config, task_files=['src/foo.py'], worktree=tmp_path)
        git_ops = _mq_make_git_ops_mock()
        return await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            merge_sha=head_sha,
            runner=stub_runner,
            escalation_queue=eq,
        )

    outcome = asyncio.run(_drive_consumer())

    assert outcome is not None
    assert outcome.status == 'blocked'
    assert outcome.failure_category == FLOCK_CONTENTION_CATEGORY

    sentinel = _verify_worktree_contention_sentinel(socket.gethostname())
    matches = eq.get_by_task(sentinel, status='pending', level=2)
    assert len(matches) == 1, f'expected exactly one born-at-L2 escalation, got {matches!r}'
    esc = matches[0]
    assert esc.level == 2
    assert esc.severity == 'critical'
    assert esc.agent_role == 'orchestrator-verify-host-monitor'
    assert esc.category == 'verify_worktree_contention'
    assert str(holder_pgid_val) in esc.detail
    assert str(waiter_pgid_val) in esc.detail
