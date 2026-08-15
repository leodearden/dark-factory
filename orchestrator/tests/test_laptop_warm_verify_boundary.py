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
import re
import select
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import NamedTuple
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
    lane_lock_path,
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


#: The stock ``python -c`` bootstrap every spawned verify-merge runs: import the
#: real CLI and hand straight over to it.  Overridable per call ONLY so a caller
#: can wrap a production callable in a stopwatch *inside the child* and report
#: what the child's own clock measured (see :data:`FLOCK_GATE_TIMING_BOOTSTRAP`)
#: -- the CLI itself still runs unmodified, on the real argv, config and env.
STOCK_BOOTSTRAP = 'from orchestrator.cli import main; main()'


def verify_merge_argv(
    *, sha: str, spec: MergeVerifySpec, cfg_file: Path, request_id: str | None = None,
    bootstrap: str = STOCK_BOOTSTRAP,
) -> list[str]:
    """Build the argv for a real ``orchestrator verify-merge`` invocation."""
    argv = [
        sys.executable, '-c', bootstrap,
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


def stderr_tail(raw: bytes, *, limit: int = 2000) -> str:
    """Decode *raw* subprocess stderr and keep the TAIL (last *limit* chars).

    The actionable line (a traceback's final ``Error: ...``, a watchdog
    self-kill message) consistently sits at the END of a subprocess's
    stderr, so a head slice silently discards it -- task 3318 hit exactly
    this bug from a head-truncated assertion message.  ``errors='replace'``
    so a slice landing mid multi-byte UTF-8 sequence can't itself raise and
    mask the real failure being reported.
    """
    return raw.decode(errors='replace')[-limit:]


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
    bootstrap: str = STOCK_BOOTSTRAP,
) -> subprocess.Popen:
    """Spawn a real ``orchestrator verify-merge`` subprocess.

    ``stdin=subprocess.PIPE`` by default so callers can drive the
    connection-death protocol (heartbeat writer / EOF-on-close); tests that
    don't pass --request-id never arm the watchdog so stdin is inert for them.

    ``bootstrap`` defaults to :data:`STOCK_BOOTSTRAP` (byte-identical to the
    pre-seam argv); see that constant for the only reason to override it.
    """
    return subprocess.Popen(
        verify_merge_argv(
            sha=sha, spec=spec, cfg_file=cfg_file, request_id=request_id,
            bootstrap=bootstrap,
        ),
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


def wait_subtree_live(
    pgid: int,
    *,
    proc: subprocess.Popen | None = None,
    proc_label: str = 'leader',
    timeout: float = 20.0,
    interval: float = 0.05,
) -> set[int]:
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

    *proc* is the already-spawned leader (or, for a caller observing a
    SEPARATE dispatcher process, that dispatcher) whose ``stdout``/``stderr``
    were opened with ``subprocess.PIPE`` -- see :func:`spawn_verify_merge`.
    It's optional so no existing call site is forced to change semantics.
    *proc_label* names *proc* in the failure message (default ``"leader"``);
    pass e.g. ``proc_label="dispatcher"`` when *proc* is a stand-in process
    rather than the leader itself (e.g. the SSH dispatcher in the Row 1
    orchestrator-killed test), so a reader doesn't apply the rc taxonomy
    below to the wrong process.  When *proc* is given, a timeout failure
    names its exit status (``<proc_label> rc=<n|None>``).  For the LEADER
    specifically, that rc distinguishes a watchdog self-kill (rc == 1, no
    ``Error:`` line), an exception exit (rc == 1 WITH an ``Error:`` line),
    and a merely slow leader (rc is None) -- three causes that were
    otherwise indistinguishable in every log this timeout has ever
    produced; a non-leader *proc* follows its own exit-code conventions,
    not this taxonomy.  ``stderr`` is read ONLY when ``proc.poll()`` is not
    None, and even then via a ``select()``-bounded raw read (2s ceiling)
    rather than a buffered ``.read()`` (which blocks to EOF): ``poll()``
    only reports *proc*'s own exit, and a short-lived helper it spawned
    with inherited fds could still be holding the pipe's write end open,
    which would hang this helper (and the rest of the suite) rather than
    raise.  The tail (not head) is kept -- see :func:`stderr_tail` -- since
    the actionable line is at the END.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        descendants = collect_descendants(pgid, read_ppid_map())
        if descendants:
            return descendants
        time.sleep(interval)
    message = f'pgid {pgid}: no descendant appeared within {timeout}s'
    if proc is not None:
        rc = proc.poll()
        message += f'; {proc_label} rc={rc}'
        if rc is not None and proc.stderr is not None:
            try:
                chunks = []
                read_deadline = time.monotonic() + 2.0
                fd = proc.stderr.fileno()
                while True:
                    remaining = read_deadline - time.monotonic()
                    if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                        break
                    chunk = os.read(fd, 65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                tail = stderr_tail(b''.join(chunks))
            except Exception as e:  # noqa: BLE001
                tail = f'<unreadable: {e!r}>'
            message += f'; stderr tail:\n{tail}'
    raise AssertionError(message)


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


def wait_for_marker_stable(
    path: Path,
    *,
    timeout: float = 20.0,
    interval: float = 0.02,
    stable_reads: int = 6,
    _read_mtime_ns=None,
) -> int:
    """Poll *path*'s mtime until it stops changing; return the settled value.

    GNU ``touch`` on a not-yet-existing file is a two-syscall op:
    ``open(O_CREAT)`` sets an initial mtime, then ``utimensat()`` re-stamps
    it to the final value.  Under full-xdist real-subprocess CPU contention
    those two steps can straddle a scheduler preemption, so a caller that
    captures the mtime the instant the file appears (e.g. via
    :func:`wait_for_marker` alone) can observe the intermediate create-time
    rather than the final, settled value -- task 2819 (~46ms observed drift
    within the same wall-clock second on a loaded 503s full-verify run).

    Reuses :func:`wait_for_marker` as the existence gate, then polls
    *path*'s mtime (via the injectable *_read_mtime_ns* seam, defaulting to
    a real ``path.stat().st_mtime_ns`` read) until it is unchanged across
    *stable_reads* consecutive reads, and returns that settled value.
    Raises AssertionError if it never stabilizes within *timeout* seconds.
    """
    wait_for_marker(path, timeout=timeout, interval=interval)
    read = _read_mtime_ns or (lambda p: p.stat().st_mtime_ns)
    deadline = time.monotonic() + timeout
    last = read(path)
    count = 1
    while time.monotonic() < deadline:
        time.sleep(interval)
        cur = read(path)
        if cur == last:
            count += 1
            if count >= stable_reads:
                return cur
        else:
            last = cur
            count = 1
    raise AssertionError(
        f'{path}: mtime did not settle within {timeout}s '
        f'(last observed mtime_ns={last})'
    )


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
# Task 4025 (de-flake beta) -- pinned SS9 Row 1/2/3 watchdog window.
#
# ROW_WATCHDOG_ENV carries the SAME numbers as today's production constants
# -- verify_cancel.WATCHDOG_HEARTBEAT_TIMEOUT_SECS (10.0) and
# WATCHDOG_KILL_GRACE_SECS (5.0) -- but as PINNED LITERALS, deliberately NOT
# imported from verify_cancel: task 4195 will retune those constants
# (deriving the timeout from SSH_SERVER_ALIVE_INTERVAL *
# SSH_SERVER_ALIVE_COUNT_MAX, toward ~60s+), and tracking them here would
# silently multiply this module's runtime (Row 3 from ~17s to ~70s).
#
# This REPLACES a prior test-only 1.0s/0.5s tightening that bought speed and
# no coverage, and was the measured cause of the "no descendant appeared
# within 20.0s" flake on wait_subtree_live (6 failures / 490 legs on the
# 1.0s arm vs 0 on both the 10s arm and the separate-process producer arm).
#
# HONEST CAVEAT: this MOVES the cliff from 1.0s to 10s, it does not remove
# it.  The watchdog deadline is a wall-clock select() in
# verify_cancel.run_stdin_watchdog -- LOAD-RIGID -- while the producer
# (HeartbeatWriter._run) is an Event.wait(0.2) -- LOAD-ELASTIC.  Measured
# producer inter-write gap at loadavg 113-178 was p999 0.386s / max 0.845s:
# only ~1.2x real margin against a 1.0s deadline, but ~12x against 10.0s.
# Do not try to buy margin by pre-buffering heartbeats -- run_stdin_watchdog
# drains with a single read(4096) per select, so a burst resets exactly one
# window, not N.
ROW_WATCHDOG_ENV: dict[str, str] = {
    'ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS': '10.0',
    'ORCH_WATCHDOG_KILL_GRACE_SECS': '5.0',
}

#: Shared ceiling for the rows' child.wait()/wait_subtree_gone() polls -- 2x
#: the pinned 15s (10.0+5.0) worst-case fire window.  A WEDGE-DETECTOR, not a
#: speed assertion: the rows assert THAT the tree was killed, never how fast,
#: so on the success path a wider ceiling costs zero wall-clock (the poll
#: returns as soon as the condition holds) and is paid only when the test is
#: already failing.
ROW_TREE_KILL_CEILING_SECS: float = 30.0

#: Per-test opt-out from this module's inherited ``timeout = 60``
#: (orchestrator/pyproject.toml:103, thread-mode -- os._exit()s the xdist
#: worker on expiry).  These rows' worst case is two
#: ROW_TREE_KILL_CEILING_SECS-bounded waits plus one full watchdog window,
#: which would otherwise risk that 60s ceiling under full-suite load.
ROW_PER_TEST_TIMEOUT_SECS: int = 120
# ---------------------------------------------------------------------------


def test_row_watchdog_window_is_pinned_and_ceilings_clear_it():
    """ROW_WATCHDOG_ENV is pinned (not derived) and the row ceilings clear it.

    Three assertions, all pure arithmetic on the module constants that the
    SS9 Row 1/2/3 tests below consume -- no subprocess, no sleep, no wall
    clock, so a future edit that raises the window without widening the
    ceilings (or vice versa) is caught in microseconds rather than only
    after ~30s of real subprocess work per row on a full verify leg:

    1. ROW_WATCHDOG_ENV is an exact-equality pin on the literals '10.0'/
       '5.0' -- these must NEVER be derived from
       ``verify_cancel.WATCHDOG_HEARTBEAT_TIMEOUT_SECS`` /
       ``WATCHDOG_KILL_GRACE_SECS``, because task 4195 is filed to raise the
       production timeout toward ~60s+ and tracking it here would silently
       multiply this module's runtime (Row 3 from ~17s to ~70s).
    2. ROW_TREE_KILL_CEILING_SECS clears the worst-case watchdog fire window
       (Row 3's select-timeout branch: heartbeat_timeout + grace_secs) with
       at least 10s of load headroom.
    3. ROW_PER_TEST_TIMEOUT_SECS clears two ceiling-bounded waits
       (``child.wait`` then ``wait_subtree_gone``) plus one full watchdog
       window -- the module inherits ``timeout = 60`` from
       ``orchestrator/pyproject.toml``, whose thread-mode expiry
       ``os._exit()``s the xdist worker rather than just failing this test.
    """
    assert ROW_WATCHDOG_ENV == {
        'ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS': '10.0',
        'ORCH_WATCHDOG_KILL_GRACE_SECS': '5.0',
    }, (
        'ROW_WATCHDOG_ENV must be pinned to the literals 10.0/5.0, not '
        'derived from verify_cancel.WATCHDOG_HEARTBEAT_TIMEOUT_SECS / '
        'WATCHDOG_KILL_GRACE_SECS -- task 4195 will raise the production '
        "timeout toward ~60s+ and tracking it here would silently multiply "
        "this module's runtime (Row 3 from ~17s to ~70s)"
    )

    window = (
        float(ROW_WATCHDOG_ENV['ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS'])
        + float(ROW_WATCHDOG_ENV['ORCH_WATCHDOG_KILL_GRACE_SECS'])
    )
    assert ROW_TREE_KILL_CEILING_SECS >= window + 10.0, (
        f'ROW_TREE_KILL_CEILING_SECS ({ROW_TREE_KILL_CEILING_SECS}) must '
        f'clear the worst-case watchdog fire window ({window}s, Row 3\'s '
        f'select-timeout branch) with at least 10s of load headroom, or the '
        f"rows' child.wait()/wait_subtree_gone() ceilings fail "
        f'DETERMINISTICALLY -- but only after ~30s of real subprocess work '
        f'per row on a full verify leg'
    )

    assert ROW_PER_TEST_TIMEOUT_SECS >= 2 * ROW_TREE_KILL_CEILING_SECS + window, (
        f'ROW_PER_TEST_TIMEOUT_SECS ({ROW_PER_TEST_TIMEOUT_SECS}) must clear '
        f'two ceiling-bounded waits (child.wait then wait_subtree_gone, '
        f'{ROW_TREE_KILL_CEILING_SECS}s each) plus one full watchdog window '
        f'({window}s) -- the module inherits timeout = 60 from '
        f'orchestrator/pyproject.toml, whose thread-mode expiry os._exit()s '
        f'the xdist worker rather than just failing this one test'
    )


# ---------------------------------------------------------------------------
# Task 2819 -- deterministic unit coverage for wait_for_marker_stable, the
# create+utimensat settle helper that de-flakes the Row 5 marker-retention
# assertion below.  Both tests inject a private mtime-reader seam
# (_read_mtime_ns) so there is ZERO real timing -- a real-timing settle test
# would itself be load-sensitive (fixing a flake with a flake).
# ---------------------------------------------------------------------------


def test_wait_for_marker_stable_returns_settled_not_intermediate(tmp_path):
    """wait_for_marker_stable returns the SETTLED mtime, not the intermediate one.

    Models GNU ``touch``'s two-syscall create->utimensat drift (task 2819):
    the injected reader yields the intermediate create-time mtime a few
    times, then the final settled mtime for the remainder of the scripted
    sequence.  Uses a real tmp_path marker so the existence gate
    (wait_for_marker) passes instantly, and interval=0 so the settle loop
    itself burns no real wall-clock time.
    """
    marker = tmp_path / 'warm.marker'
    marker.touch()

    sequence = [100, 100, 100, 150, 150, 150, 150, 150, 150, 150, 150, 150]
    calls = iter(sequence)

    def fake_read_mtime_ns(_path):
        return next(calls)

    result = wait_for_marker_stable(
        marker, interval=0, stable_reads=6, _read_mtime_ns=fake_read_mtime_ns,
    )

    assert result == 150, (
        f'expected the settled mtime (150), not the create-time intermediate '
        f'(100); got {result}'
    )


def test_wait_for_marker_stable_raises_when_never_settles(tmp_path):
    """wait_for_marker_stable raises AssertionError if the mtime never quiets down.

    An always-changing reader can never satisfy the consecutive-unchanged
    window, so with a tiny timeout this must raise promptly rather than
    hang or silently return an unsettled value.
    """
    marker = tmp_path / 'warm.marker'
    marker.touch()

    call_count = [0]

    def fake_read_mtime_ns(_path):
        call_count[0] += 1
        return call_count[0]

    with pytest.raises(AssertionError):
        wait_for_marker_stable(
            marker, timeout=0.05, interval=0, _read_mtime_ns=fake_read_mtime_ns,
        )


# ---------------------------------------------------------------------------
# Task 3369 -- in-child stopwatch for the flock GATE, replacing task 2921/2941's
# outer wall-clock subtraction in test_flock_wait_env_override_speeds_up_
# contention_result below.
#
# The quantity that test is about is the CLI's *bounded flock wait* -- the one
# thing ORCH_MERGE_VERIFY_FLOCK_WAIT_SECS controls.  Measuring it from OUTSIDE
# means measuring `subprocess wall time`, which is dominated by a term the
# override has nothing to do with: interpreter startup + `import
# orchestrator.cli`.  Task 2921 tried to cancel that term by subtracting a
# bare-import baseline, and task 2941 by subtracting a same-shape uncontended
# verify-merge -- but the term is not merely large, it is HIGHLY VARIABLE
# between two sequential subprocesses, so subtracting one sample of it from
# another does not cancel it.  Measured on this box with the override provably
# honored: two back-to-back children reported `import orchestrator.cli` at 1.9s
# and 4.0s, and a single-test isolated run left a 6.0s residue against a 5.0s
# ceiling -- i.e. green behaviour reported red, the same unsound-wall-clock-proxy
# defect fixed for TestB2PreTurn1Wedge in test_liveness_boundary_gate.py.
#
# So instrument the child instead and assert on the code-under-test's OWN clock.
# FLOCK_GATE_TIMING_BOOTSTRAP wraps orchestrator.cli.acquire_merge_verify_flock
# -- the production callable the gate actually invokes -- in a stopwatch and
# emits one line per acquire on STDERR (stdout is reserved for the VerifyResult
# JSON).  The CLI is otherwise untouched: same argv, same config, same env, same
# real fcntl.flock against a really-held lock.  Across every probe run the wait
# it reports for a 0.5s override was 0.507-0.509s, load notwithstanding, against
# the 10.0s MERGE_VERIFY_FLOCK_WAIT_SECS production default an un-wired override
# would leave -- a discriminant with ~20x margin instead of a coin flip.
# ---------------------------------------------------------------------------

#: Marker token the instrumented bootstrap prints, one line per gate acquire.
FLOCK_GATE_MARKER = '__FLOCK_GATE__'

FLOCK_GATE_TIMING_BOOTSTRAP = f"""
import sys, time
import orchestrator.cli as _cli
_real_acquire = _cli.acquire_merge_verify_flock


def _timed_acquire(path, timeout_secs, **kwargs):
    _t0 = time.monotonic()
    fd = _real_acquire(path, timeout_secs, **kwargs)
    print(
        '{FLOCK_GATE_MARKER} lock=%s timeout=%r waited=%.4f acquired=%r'
        % (path.name, timeout_secs, time.monotonic() - _t0, fd is not None),
        file=sys.stderr,
    )
    return fd


_cli.acquire_merge_verify_flock = _timed_acquire
_cli.main()
"""


class FlockGateWait(NamedTuple):
    """One instrumented ``acquire_merge_verify_flock`` call inside the child."""

    lock: str
    timeout_secs: float
    waited_secs: float
    acquired: bool


_FLOCK_GATE_RE = re.compile(
    re.escape(FLOCK_GATE_MARKER)
    + r' lock=(?P<lock>\S+) timeout=(?P<timeout>\S+) '
    r'waited=(?P<waited>\S+) acquired=(?P<acquired>True|False)'
)


def parse_flock_gate_waits(stderr: str) -> list[FlockGateWait]:
    """Parse the :data:`FLOCK_GATE_TIMING_BOOTSTRAP` lines out of a child's stderr."""
    return [
        FlockGateWait(
            lock=m.group('lock'),
            timeout_secs=float(m.group('timeout')),
            waited_secs=float(m.group('waited')),
            acquired=m.group('acquired') == 'True',
        )
        for m in _FLOCK_GATE_RE.finditer(stderr)
    ]


def test_flock_gate_timing_bootstrap_parses_its_own_marker_lines():
    """parse_flock_gate_waits round-trips the bootstrap's marker format.

    The bootstrap emits its lines from inside a spawned child, so a drift
    between what it prints and what the parser accepts would silently return
    zero observations -- which would make the ceiling assertion in
    test_flock_wait_env_override_speeds_up_contention_result vacuous rather
    than red.  The emit format and this parser are therefore pinned together
    here, with no subprocess involved.
    """
    emitted = (
        f'{FLOCK_GATE_MARKER} lock=_merge-verify.lock timeout=0.5 '
        'waited=0.0001 acquired=True\n'
        'some unrelated stderr chatter\n'
        f'{FLOCK_GATE_MARKER} lock=.merge_verify.lock timeout=0.5 '
        'waited=0.5083 acquired=False\n'
    )

    waits = parse_flock_gate_waits(emitted)

    assert waits == [
        FlockGateWait('_merge-verify.lock', 0.5, 0.0001, True),
        FlockGateWait('.merge_verify.lock', 0.5, 0.5083, False),
    ]


def test_parse_flock_gate_waits_returns_empty_for_uninstrumented_stderr():
    """Stderr with no marker lines yields no observations.

    This is the case the ceiling assertion's non-vacuity guard exists to
    catch: if the CLI ever stops reaching the gate through the module-level
    ``orchestrator.cli.acquire_merge_verify_flock`` name the bootstrap
    patches, the child produces no marker lines at all, and an assertion over
    an empty list would pass by default.
    """
    assert parse_flock_gate_waits('Traceback (most recent call last):\nboom\n') == []


# ---------------------------------------------------------------------------
# Task 2309 step-1 RED -- env-var test seams for remote-side timing constants
# (PRD SS11 Q1/Q2 tunability).  Production defaults are byte-identical when
# unset; these overrides exist ONLY so the SS9 boundary rows below can run
# fast and deterministically instead of being wall-clock-bound on the 10s
# flock wait / 10s+5s watchdog window.  RED until step-2 wires them into
# cli.py.
# ---------------------------------------------------------------------------


#: Ceiling on the CLI's own measured flock wait.  Sits an order of magnitude
#: above the 0.5s override the test sets and half way below the 10.0s
#: MERGE_VERIFY_FLOCK_WAIT_SECS production default an un-wired override would
#: leave -- so it discriminates the two cases with a huge margin either side.
FLOCK_WAIT_CEILING_SECS = 5.0


@pytest.mark.timeout(180)  # task 3369: one subprocess again (the baseline probe is gone)
def test_flock_wait_env_override_speeds_up_contention_result(tmp_path):
    """ORCH_MERGE_VERIFY_FLOCK_WAIT_SECS overrides the flock bounded wait.

    Holds the real flock (+ writes the holder pgid) exactly as
    test_cli.py:1419 does, then spawns a real knob-on verify-merge with the
    override set small.  Today (RED) verify-merge ignores the env var and
    waits the full 10.0s production window; asserting the wait came in well
    under that fails until cli.py reads the override.

    task 3369 (de-flake): the assertion is on the wait the CLI's OWN clock
    measured, reported by :data:`FLOCK_GATE_TIMING_BOOTSTRAP` from inside the
    child, NOT on the child's wall-clock duration.

    Tasks 2376/2921/2941 successively widened, then baseline-subtracted, an
    outer wall-clock measurement, and it kept coming back red on green
    behaviour.  The reason is structural rather than a matter of tuning: the
    override governs one term of the child's runtime (the bounded flock wait),
    while its wall clock is dominated by a term the override has nothing to do
    with -- interpreter startup plus ``import orchestrator.cli`` -- and that
    term is not stable enough between two sequential subprocesses to cancel by
    subtracting one sample of it from another.  Probed on this box with the
    override provably honored (the gate itself reported waited=0.508s), two
    back-to-back children put that import at 1.9s and 4.0s, and a
    single-test, otherwise-idle run left a 6.0s residue against the old 5.0s
    ceiling.  Same defect and same remedy as TestB2PreTurn1Wedge in
    test_liveness_boundary_gate.py: stop using wall clock as a proxy for a
    quantity the code under test already measures itself.

    The child still runs the real CLI end to end -- real argv, real config,
    real env, real ``fcntl.flock`` against a really-held lock.  The bootstrap
    only wraps the production ``acquire_merge_verify_flock`` callable in a
    stopwatch, so an un-wired override still shows up as a ~10s wait.  Its
    one failure mode -- the CLI reaching the gate by some path other than the
    patched module-level name, leaving zero observations and a vacuously-true
    ceiling -- is closed by the non-vacuity assertion below.
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
        proc = spawn_verify_merge(
            sha=head_sha,
            spec=fast_spec(),
            cfg_file=cfg_file,
            extra_env={'ORCH_MERGE_VERIFY_FLOCK_WAIT_SECS': '0.5'},
            bootstrap=FLOCK_GATE_TIMING_BOOTSTRAP,
        )
        # task 2376: widened from 15s -- host oversubscription can delay
        # subprocess completion past a short deadline; the discriminating
        # invariant is the FLOCK_WAIT_CEILING_SECS assertion below (task
        # 3369), never this ceiling, which only bounds a wedged child.
        stdout, stderr = proc.communicate(timeout=60)
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(held_fd, fcntl.LOCK_UN)
        os.close(held_fd)

    assert proc.returncode == 0, (
        f'expected exit 0 (contention result on stdout), got {proc.returncode}; '
        f'stderr={stderr_tail(stderr)!r}'
    )
    result = result_from_json(stdout.decode())
    assert result.category == FLOCK_CONTENTION_CATEGORY, (
        f'expected flock-contention result, got category={result.category!r} '
        f'stdout={stdout.decode()[:2000]!r}'
    )

    waits = parse_flock_gate_waits(stderr.decode())
    # Non-vacuity guard: the ceiling below is a max() over these observations,
    # so an empty list would pass by default.  Zero observations means the CLI
    # no longer calls the gate through the module-level
    # ``orchestrator.cli.acquire_merge_verify_flock`` name the bootstrap
    # patches -- a real signal that this test stopped measuring anything, not
    # a pass.
    assert waits, (
        'no instrumented flock-gate observation on the child stderr -- the '
        'timing bootstrap patches orchestrator.cli.acquire_merge_verify_flock, '
        'so zero observations means the gate is no longer reached through that '
        'name and the ceiling assertion below would be vacuous; '
        f'stderr={stderr_tail(stderr)!r}'
    )
    longest = max(w.waited_secs for w in waits)
    assert longest < FLOCK_WAIT_CEILING_SECS, (
        f'expected the CLI-measured flock wait to be well under the 10s '
        f'production wait (MERGE_VERIFY_FLOCK_WAIT_SECS) given the 0.5s env '
        f'override -- longest={longest:.2f}s over {len(waits)} gate acquire(s): '
        f'{waits}; the env override is not wired up yet'
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
    worktree_base = worktree_base_for(repo)

    REQUEST_ID = 'env-seam-watchdog-test'
    pgf = pgid_file(worktree_base, REQUEST_ID)

    proc = spawn_verify_merge(
        sha=head_sha,
        spec=sleeper_spec(300.0),
        cfg_file=cfg_file,
        request_id=REQUEST_ID,
        extra_env={
            'ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS': '0.5',
            'ORCH_WATCHDOG_KILL_GRACE_SECS': '0.2',
        },
    )
    try:
        # task 2921 (load-robustness): the child's ``from orchestrator.cli
        # import main`` startup dominates wall-clock under a full-suite storm
        # (~9s observed on a loaded host) and is unrelated to the watchdog
        # window under test -- so the old bare ``proc.wait(timeout=9.0)`` was
        # import-bound, not watchdog-bound, and flaked under parallel load.
        # The --request-id run writes its pgid file at startup BEFORE doing any
        # work (cli.py verify_merge) and the watchdog self-kills via
        # ``os._exit(1)`` (which bypasses pgid-file cleanup), so waiting for
        # the pgid file to appear absorbs the load-sensitive import cost; the
        # subsequent ``fire_delay`` then measures only the watchdog fire delay.
        wait_for_pgid_file(pgf, timeout=30.0)
        armed = time.monotonic()
        # No heartbeat is ever written -- stdin stays open but silent, which
        # is sufficient to exercise the heartbeat-starvation timing path
        # (Rows 1-3 later distinguish EOF vs. timeout precisely). Wait well
        # past the 15s production window so an un-wired override is OBSERVED
        # self-exiting (not merely timed out) and caught by the delay
        # assertion below rather than masked as a hang.
        try:
            proc.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            pytest.fail(
                'verify-merge did not self-exit within 30s of startup -- '
                'the watchdog did not fire at all'
            )
        fire_delay = time.monotonic() - armed
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
    # task 2921 discriminant: the override (0.5s heartbeat timeout + 0.2s kill
    # grace, ~0.7s total) must fire the watchdog FAR faster than the
    # 10s+5s=15s production window. Measured from pgid-appearance (post-import),
    # so this ceiling is a genuine watchdog budget, not an import budget. The
    # 10.0s value sits well above the ~0.7s override (with generous load
    # headroom) yet well below the 15s production window, so it still catches a
    # regression that un-wires the override.
    assert fire_delay < 10.0, (
        f'watchdog self-exit took {fire_delay:.2f}s after startup -- expected '
        f'well under the 15s production window (env override=0.5s+0.2s); the '
        f'watchdog env overrides are not wired up'
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
        # An ephemeral leak is a _merge-<uuid> *directory*; the shared
        # <lane_dir>.lock flock sibling (<worktree_base>/_merge-verify.lock,
        # task 2685) is a regular file and is expected -- require is_dir().
        if p.is_dir() and p.name.startswith('_merge-') and p.name != '_merge-verify'
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
        wait_subtree_live(pgid_val, proc=child)

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
        # Row 1 owns the DISPATCHER process, not the leader -- the leader's
        # own stdout/stderr aren't piped to this test, so pass the dispatcher.
        wait_subtree_live(pgid_val, proc=dispatcher, proc_label='dispatcher')

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
        wait_subtree_live(pgid_val, proc=child)

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


@pytest.mark.timeout(ROW_PER_TEST_TIMEOUT_SECS)  # task 4025: production 10s+5s watchdog window + real subprocesses under full-suite load
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

    Uses ROW_WATCHDOG_ENV -- the pinned production-equivalent window
    (10.0s heartbeat timeout + 5.0s kill grace) -- directly on the spawned
    verify-merge, so the assertion window here is ~15s, not fast; see
    ROW_WATCHDOG_ENV's comment above for the pin rationale and the honest
    cliff-moved caveat.

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
        extra_env=ROW_WATCHDOG_ENV,
    )
    heartbeat = HeartbeatWriter(child, interval=0.2).start()
    try:
        pgid_val = wait_for_pgid_file(pgf)
        wait_subtree_live(pgid_val, proc=child)

        heartbeat.stop_heartbeats()
        assert child.stdin is not None and not child.stdin.closed, (
            'harness bug: stdin must stay OPEN for Row 3 (hard partition) -- '
            'only heartbeats stop; this is what distinguishes it from Row 2'
        )

        try:
            child.wait(timeout=ROW_TREE_KILL_CEILING_SECS)
        except subprocess.TimeoutExpired:
            pytest.fail(
                f'verify-merge did not self-exit within '
                f'{ROW_TREE_KILL_CEILING_SECS}s of heartbeat starvation '
                f'(select-timeout branch did not fire)'
            )
        assert child.returncode != 0, (
            f'expected non-zero exit (watchdog self-kill), got {child.returncode}'
        )

        assert wait_subtree_gone(pgid_val, timeout=ROW_TREE_KILL_CEILING_SECS), (
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


@pytest.mark.timeout(120)  # task 2921: two real verify-merge subprocesses (holder + waiter) under full-suite load
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
    # task 3318: --request-id arms the holder's stdin watchdog (see
    # spawn_verify_merge docstring), but unlike Row 3 this holder was never
    # given a live HeartbeatWriter -- it self-killed at
    # ~(WATCHDOG_HEARTBEAT_TIMEOUT_SECS + WATCHDOG_KILL_GRACE_SECS) = ~15-17s
    # instead of surviving the full 300s sleeper_spec, well inside the
    # waiter's own budget under full-suite xdist load (Python import alone
    # ~9s). Pair it with a live heartbeat, mirroring Row 3, so it survives
    # for the whole span the test needs it.
    heartbeat_holder = HeartbeatWriter(holder, interval=0.2).start()
    try:
        holder_pgid_val = wait_for_pgid_file(pgf_holder)
        wait_subtree_live(holder_pgid_val, proc=holder)

        persistent_wt = worktree_base / '_merge-verify'
        marker = persistent_wt / 'target' / 'warm.marker'
        # Capture the baseline only AFTER the holder's touch (create +
        # utimensat) has settled -- under full-xdist load those two syscalls
        # can straddle a preemption, so the instant-of-appearance mtime can
        # still drift ~tens of ms (task 2819), which would spuriously trip
        # the retention equality assertion below.
        marker_mtime_before = wait_for_marker_stable(marker)

        waiter = spawn_verify_merge(
            sha=head_sha,
            spec=fast_spec(),
            cfg_file=cfg_file,
            request_id='row5-waiter',
            extra_env={'ORCH_MERGE_VERIFY_FLOCK_WAIT_SECS': '0.5'},
        )
        # task 2921 (load-robustness): widened from 15s. The waiter is a full
        # real verify-merge subprocess (Python import ~9s under a full-suite
        # storm + config load + git worktree setup + the 0.5s flock wait), so a
        # 15s ceiling is import-bound, not flock-bound, under parallel load and
        # flaked (observed: this subprocess timed out at 15s under load). There
        # is NO timing assertion on the waiter here -- the test asserts on the
        # returned FLOCK_CONTENTION_CATEGORY discriminant, not on duration -- so
        # widening the completion ceiling has zero discrimination cost.
        stdout, stderr = waiter.communicate(timeout=60)

        assert waiter.returncode == 0, (
            f'expected exit 0 (contention result on stdout), got '
            # task 3318: tail-sliced (not head-truncated) -- the actual
            # failure cause (e.g. a watchdog self-kill traceback) is at the
            # END of stderr, and a 2000-char head slice was hiding it.
            f'{waiter.returncode}; stderr={stderr_tail(stderr, limit=4000)!r}'
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
            # An ephemeral leak is a _merge-<uuid> *directory*; the shared
            # <lane_dir>.lock flock sibling (<worktree_base>/_merge-verify.lock,
            # task 2685) is a regular file and is expected -- require is_dir().
            if p.is_dir() and p.name.startswith('_merge-') and p.name != '_merge-verify'
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
        heartbeat_holder.stop_heartbeats()
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


# ---------------------------------------------------------------------------
# Task 2830 step-4 -- the SS9 Signal at the real subprocess seam: a live
# knob-ON verify-merge HOLDS the shared <lane_dir>.lock mid-build, so a
# non-blocking flock probe on it from this process is DENIED.  Stronger
# evidence than the mock-based step-1 pin (test_cli.py
# test_verify_merge_holds_lane_lock_during_warm_run): drives the real CLI
# span end-to-end and pins the lock PATH the split lane-lock lifetime
# (esc-2830-1) re-acquires for the build's duration.
# ---------------------------------------------------------------------------


def test_live_verify_merge_holds_lane_lock_real_subprocess(tmp_path):
    """SS9 Signal: a live knob-ON verify-merge holds the SHARED lane lock mid-build (task 2830).

    The task's Signal, at the REAL subprocess seam: spawn a real knob-ON
    ``orchestrator verify-merge`` with a sleeper build, poll until it is genuinely
    mid-build (the sleeper's marker inside the persistent ``_merge-verify`` worktree
    appears), then from THIS test process take a non-blocking exclusive flock on
    ``lane_lock_path(persistent_merge_worktree_path)`` = ``<worktree_base>/
    _merge-verify.lock`` -- the SAME lock reify's seed/thin/gc and
    ``GitOps.merge_verify_lease`` take (task 2685) -- and assert it is DENIED
    (BlockingIOError). This confirms the live CLI span is the holder of the LANE lock
    during the build, so a laptop lane actor (reseed/thin/gc) is mutually excluded from
    a live laptop verify -- closing the divergence DF 2685 left for the remote host and
    that the two remote-twin incidents (reify 5034/5187) turned on.

    With the split lane-lock lifetime (esc-2830-1) the lane lock is re-acquired for the
    build's duration inside ``_run()`` (AFTER ``acquire_host_verify_worktree``'s own
    reset has released it), and the shared holder-pgid is written before the build
    runs. The sleeper's ``touch`` executes inside that build-scoped hold, so the
    marker's existence implies the lock is already held and the holder-pgid recorded.

    Stronger than the mock-based step-1 pin
    (``test_verify_merge_holds_lane_lock_during_warm_run``): it drives the real CLI span
    end-to-end (no monkeypatched ``acquire_host_verify_worktree`` / ``run_merge_verify_``
    ``on_worktree``), so it pins the lock path the production code actually computes and
    holds.
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    worktree_base = worktree_base_for(repo)
    worktree_base.mkdir(parents=True, exist_ok=True)

    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=True)

    # No --request-id -> the stdin watchdog is never armed, so the sleeper holder
    # stays alive until we kill it in the finally (mirrors the env-override probe
    # test's spawn shape, test_flock_wait_env_override_speeds_up_contention_result).
    holder = spawn_verify_merge(sha=head_sha, spec=sleeper_spec(300.0), cfg_file=cfg_file)
    try:
        persistent_wt = worktree_base / '_merge-verify'
        marker = persistent_wt / 'target' / 'warm.marker'
        # Poll until the sleeper build is genuinely mid-run. The build-scoped lane-lock
        # hold is acquired (and the holder-pgid written) BEFORE the build runs, so the
        # marker's appearance implies the lane lock is currently held -- no race.
        wait_for_marker(marker)

        # THE SIGNAL: a non-blocking exclusive flock on the shared lane lock from this
        # (independent) process must be DENIED while the live CLI span holds it.
        lane_lock = lane_lock_path(persistent_wt)
        probe_fd = os.open(lane_lock, os.O_RDWR | os.O_CREAT)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(probe_fd)

        # Corroboration: the build-scoped hold records the shared holder-pgid, so a
        # live positive int (not a stale/absent file) confirms the live CLI owns the
        # lane lock right now.
        holder_pgid = read_lock_holder_pgid(worktree_base)
        assert isinstance(holder_pgid, int) and holder_pgid > 0, (
            f'the live CLI must record a positive shared holder-pgid while holding the '
            f'lane lock mid-build, got {holder_pgid!r}'
        )
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=5)
        if holder.stdin is not None:
            with contextlib.suppress(OSError):
                holder.stdin.close()
