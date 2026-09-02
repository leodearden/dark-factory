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

That single-source rule holds exactly where it decides anything, and task
4014 added ONE deliberate, narrow exception on the other side of a
discovery/assertion split:

* ASSERTIONS -- ``subtree_and_leader_gone``, ``wait_subtree_gone`` -- and the
  descendant set ``wait_subtree_live`` RETURNS still come only from the
  production walkers ``collect_descendants``/``read_ppid_map``.
* the ARRANGE-phase discovery GATE adds a cheap Linux
  ``/proc/<pid>/task/*/children`` pre-filter (:func:`_read_direct_children`)
  that decides nothing except whether to spend a full rescan on a given poll
  tick.

The measurement that motivates it: ``read_ppid_map()`` costs 49.67 ms at 917
live processes, one ``children`` read costs 0.0183 ms -- ~2700x, against a
50 ms poll interval.  So the old shape already doubled the effective
sampling period AT IDLE, and under a full-suite storm (~8000 procs, ~400ms
per scan) it collapsed ~10x -- while burning ~917 file reads per tick in CPU
competition with the very leader it was waiting to see fork.  The probe is
tri-state and degrades to the pre-4014 full walk when it cannot answer, so a
kernel without CONFIG_PROC_CHILDREN keeps today's exact behaviour.

Division of labour with the sibling de-flake: task 4025 addressed the
WATCHDOG WINDOW (and its banner below is explicit that this "MOVES the cliff
from 1.0s to 10s, it does not remove it"); task 4014 addressed the POLLER
above it -- the per-tick cost, the transient-child race, and the flat 20s
discovery ceiling that did not track load.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import fcntl
import math
import os
import re
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import warnings
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
        """Stop sending heartbeats WITHOUT closing stdin (Row 3: hard partition).

        Joins the writer thread (mirroring close_stdin() below) so a caller
        that immediately does other teardown -- e.g. kill_holder_tree
        closing stdin in a finally right after this call -- can never race
        the writer thread's own write()-after-stop-check window: without
        the join, the thread could still be between
        ``self._stop.wait(self._interval)`` returning False and
        ``self._proc.stdin.write(b'\\n')`` when stdin gets closed out from
        under it, and ``write()`` on an already-closed BufferedWriter
        raises ValueError, which ``_run``'s ``except (BrokenPipeError,
        OSError)`` does not catch -- an unhandled thread exception that
        pytest promotes to an error (see
        ``error::pytest.PytestUnhandledThreadExceptionWarning`` in
        orchestrator/pyproject.toml), possibly failing an unrelated test
        under xdist. Joining still leaves stdin OPEN (this method's whole
        contract), it only guarantees the writer has stopped touching it.
        """
        self._stop.set()
        self._thread.join(timeout=2)

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


def wait_for_pgid_file(path: Path, *, timeout: float | None = None, interval: float = 0.05) -> int:
    """Poll for a pgid file (written by verify-merge --request-id) and return its int value.

    *timeout* defaults to :func:`row_discovery_ceiling_secs`, resolved when
    the wait actually STARTS rather than at import, so a row beginning
    mid-storm gets the load-scaled deadline the storm warrants.
    """
    timeout = row_discovery_ceiling_secs() if timeout is None else timeout
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            with contextlib.suppress(ValueError):
                return int(path.read_text().strip())
        time.sleep(interval)
    raise AssertionError(f'pgid file {path} did not appear within {timeout}s')


def _read_direct_children(pid: int) -> set[int] | None:
    """Cheap probe: the DIRECT children of *pid*, from ``/proc/<pid>/task/*/children``.

    A ~2700x cheaper stand-in for a full ``read_ppid_map()`` rescan, used ONLY
    to decide whether spending that rescan is worth it on a given poll tick
    (task 4014).  Measured on this box: 0.0183 ms per ``children`` read vs
    49.67 ms for one ``read_ppid_map()`` at 917 live processes.

    Every thread of *pid* is iterated (the ``task`` directory), not just the
    main one, because ``children`` is a PER-THREAD file: a fork issued from a
    non-main thread is listed only under that thread's tid and would be
    invisible to a bare ``/proc/<pid>/task/<pid>/children`` read.

    This decides NOTHING about production behaviour -- it is a pre-filter on
    the arrange-phase discovery gate.  The descendant set actually returned
    (and every kill assertion in this module) still comes from the production
    walkers ``collect_descendants``/``read_ppid_map``.

    TRI-STATE, and the third state is load-bearing:

    * ``set()``  -- leader is live and has forked nothing yet.  A cheap
      NEGATIVE: the caller can skip the expensive walk this tick.
    * ``{pid, ...}`` -- direct children exist; worth confirming with the
      production walker.
    * ``None``   -- CANNOT probe.  Either the ``children`` files are absent
      (a kernel built without ``CONFIG_PROC_CHILDREN``) or ``/proc/<pid>``
      itself is gone (the leader exited mid-poll).  The caller must fall back
      to the full walk for that tick; conflating this with the cheap negative
      would make the poll spin to its timeout on such a kernel, and letting
      the OSError escape would turn a leader exiting mid-poll into an
      unhandled error inside the timeout diagnostic.

    Any OSError anywhere in the read collapses to ``None``.  Distinguishing
    "no CONFIG_PROC_CHILDREN" from "this thread just exited" would buy
    nothing: both answers are "don't trust the probe on this tick", and the
    fallback they select is precisely this helper's pre-4014 behaviour.

    The empty-``task``-listing branch below is DEFENSIVE-ONLY, and therefore
    deliberately uncovered: a live ``/proc/<pid>/task`` always holds at least
    one tid, and a dead one makes ``iterdir()`` itself raise OSError, which
    the handler already maps to ``None``.  It is retained rather than deleted
    because falling THROUGH it would return the cheap negative ``set()`` --
    "leader is live and has forked nothing" -- for a listing that in fact told
    us nothing, and that is the single answer which makes the caller skip its
    walk every tick and spin to the timeout.
    """
    children: set[int] = set()
    try:
        tid_dirs = list((Path('/proc') / str(pid) / 'task').iterdir())
        if not tid_dirs:
            # Defensive only (see docstring): not reachable on a live or a dead
            # /proc entry, so never trust an empty listing as a cheap negative.
            return None
        for tid_dir in tid_dirs:
            raw = (tid_dir / 'children').read_text()
            children.update(int(token) for token in raw.split())
    except OSError:
        return None
    return children


def wait_subtree_live(
    pgid: int,
    *,
    proc: subprocess.Popen | None = None,
    proc_label: str = 'leader',
    timeout: float | None = None,
    interval: float = 0.05,
    _probe_children=None,
    _ppid_map=None,
) -> set[int]:
    """Poll until *pgid* has at least one live descendant; return the descendant set.

    Raises AssertionError on timeout -- a live sleeper never appearing means
    the harness itself is broken, not a seam defect under test.

    Each tick runs a cheap ``/proc/<pgid>/task/*/children`` probe and pays
    the ~2700x more expensive ``collect_descendants(pgid, read_ppid_map())``
    rescan ONLY when that probe reports a direct child -- task 4014; see
    :func:`_read_direct_children`.  A positive probe whose confirming walk
    comes back EMPTY does not end the poll: the child the probe saw exited in
    between, and the loop keeps going.  So the returned set is always
    NON-EMPTY and, in practice, converges on a child durable enough to
    survive one rescan.

    NOTE: "has a live descendant" still fires on an EARLY child the CLI forks
    -- for a not-yet-materialised persistent worktree that is a transient
    ``git`` subprocess (``git worktree add`` / ``git reset`` / ``git clean``
    inside ``acquire_host_verify_worktree``), not necessarily the eventual
    build shell.  The skip-the-vanished rule above biases toward the durable
    sleeper but does not GUARANTEE it (a transient slow enough to survive a
    rescan is still returnable).  So callers that need to observe a
    build-produced artifact (e.g. a marker file the build's shell command
    touches) must still poll for that artifact directly -- see
    :func:`wait_for_marker` -- rather than treating "subtree live" as a proxy
    for "build has started".

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

    *timeout* defaults to :func:`row_discovery_ceiling_secs`, resolved when
    the wait actually STARTS rather than at import, so a row beginning
    mid-storm gets the load-scaled deadline the storm warrants.

    *_probe_children* / *_ppid_map* are private injectable seams (defaulting
    to :func:`_read_direct_children` and
    :func:`~orchestrator.verify_cancel.read_ppid_map`) so this poll loop can
    be covered deterministically, with no real timing and no real
    subprocess -- the same pattern :func:`wait_for_marker_stable`'s
    ``_read_mtime_ns`` seam established (task 2819).
    """
    timeout = row_discovery_ceiling_secs() if timeout is None else timeout
    probe = _probe_children or _read_direct_children
    ppid_map = _ppid_map or read_ppid_map
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Cheap probe FIRST: pay the ~50ms full-/proc rescan only on the tick
        # that can actually yield a descendant set (task 4014).  `None` means
        # the probe could not answer (no CONFIG_PROC_CHILDREN, or the leader
        # exited) -- fall through to the full walk, i.e. pre-4014 behaviour.
        direct = probe(pgid)
        if direct is None or direct:
            descendants = collect_descendants(pgid, ppid_map())
            if descendants:
                return descendants
            # Positive probe, empty walk: the child the probe saw exited in
            # between (a transient `git` setup subprocess).  Keep polling
            # rather than hand back an empty set -- see the NOTE above.
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


def wait_for_marker(
    path: Path, *, timeout: float | None = None, interval: float = 0.05
) -> None:
    """Poll for a build-produced marker file to appear; raise AssertionError on timeout.

    Unlike :func:`wait_subtree_live` (which only proves the CLI has forked
    *some* child -- possibly a transient ``git`` setup subprocess), this
    proves the build's shell command has actually executed its
    ``touch <marker>`` step, which is the real precondition tests need before
    reading the marker's mtime or asserting on its retention.

    *timeout* defaults to :data:`ROW_MARKER_CEILING_SECS`, resolved in the
    BODY rather than as a default expression because that constant is defined
    with the other row ceilings further down (same shape as
    :func:`wait_for_pgid_file`), so the value a row budgets for and the value
    this helper actually spends cannot drift apart.
    """
    timeout = ROW_MARKER_CEILING_SECS if timeout is None else timeout
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(interval)
    raise AssertionError(f'holder build did not materialize {path} within {timeout}s')


def wait_for_marker_stable(
    path: Path,
    *,
    timeout: float | None = None,
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

    *timeout* defaults to :data:`ROW_MARKER_CEILING_SECS` and is spent TWICE
    in the worst case -- once on the existence gate, then a FRESH deadline on
    the settle loop -- which is why callers budgeting for this helper must
    count that ceiling twice (see ``_ROW5_WORST_CASE_FIXED_SECS``).
    """
    timeout = ROW_MARKER_CEILING_SECS if timeout is None else timeout
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


def kill_holder_tree(
    proc: subprocess.Popen,
    *,
    timeout: float | None = None,
    _ppid_map_provider=read_ppid_map,
    _kill=os.kill,
    _killpg=os.killpg,
) -> None:
    """SIGKILL *proc* and every descendant it forked, including start_new_session escapes.

    Mirrors :func:`orchestrator.verify_cancel.cancel_request`'s algorithm --
    snapshot the ``/proc`` PPID map before sending any signal, collect
    descendants, SIGKILL them, SIGKILL the leader, fire a GUARDED
    ``killpg`` backstop for same-group stragglers while the leader is
    still an unreaped zombie, and only THEN reap it -- but walks
    descendants from ``proc.pid`` rather than a recorded pgid, and the
    ``killpg`` backstop only fires when the holder is provably its own
    group leader.

    Walking from ``proc.pid`` (rather than
    ``os.killpg(os.getpgid(proc.pid), SIGKILL)``, the obvious one-liner) is
    what makes this helper safe at every ``spawn_verify_merge`` call site in
    this module, including ones that never pass ``--request-id`` (e.g. the
    lane-lock holder): ``cli.py`` only calls
    :func:`~orchestrator.verify_cancel.start_own_process_group`
    (``os.setsid``) inside ``if request_id is not None:``, and
    ``spawn_verify_merge`` itself passes no ``start_new_session=``, so a
    holder started without ``--request-id`` stays in the CALLER's (this
    test process's) own process group -- ``os.getpgid(holder.pid) ==
    os.getpgid(0)``.  An unconditional ``killpg`` there would SIGKILL the
    pytest worker running this very test -- see
    ``test_kill_holder_tree_never_signals_the_callers_own_process_group``.
    The descendant walk has no such hazard: the caller is always an
    ANCESTOR of the holder, never a descendant -- the same argument
    :func:`~orchestrator.verify_cancel.start_own_process_group` makes for
    ``sshd`` never being a descendant of the pgid ``cancel_request`` walks.

    *timeout* defaults to :data:`ROW5_HOLDER_TEARDOWN_CEILING_SECS`,
    resolved when the call actually runs rather than at import time -- the
    same lazy-default convention :func:`wait_for_pgid_file` and
    :func:`wait_subtree_live` use above, needed here because that constant
    is defined later in this module.  This is the ONLY blocking wait the
    helper performs (the descendant sweep is pure signalling, no polling
    loop), so the budget ``_ROW5_WORST_CASE_FIXED_SECS`` derives from stays
    valid unchanged.

    *_ppid_map_provider* / *_kill* / *_killpg* are private injectable seams,
    mirroring :func:`cancel_request`'s own convention, so tests can pin the
    session-escape reap deterministically with zero risk of signalling
    anything unintended.

    ALREADY-REAPED SHORT CIRCUIT: an already-reaped leader means
    ``proc.pid`` is a FREE pid, so the walk below must never run against
    it.  Several call sites reap the leader with ``wait()`` inside their
    ``try`` block BEFORE their ``finally`` runs this helper
    (``test_watchdog_timeout_env_override_fires_fast_without_heartbeat``,
    ``test_cancel_verify_tree_kills_under_live_watchdog``,
    ``test_ssh_dropped_mid_build_tree_killed_via_eof_dispatcher_alive``,
    ``test_heartbeat_starved_hard_partition_tree_killed_via_timeout``) --
    on their GREEN path ``proc.pid`` no longer refers to the holder at all.  ``os.getpgid(proc.pid)`` and
    ``collect_descendants(proc.pid, ...)`` would then describe whatever
    process happens to OWN that recycled pid now, and every descendant of
    that stranger would be SIGKILLed (and killpg'd by the backstop above,
    if it happened to be its own group leader) -- precisely what "zero
    risk of signalling anything unintended" above promises never happens.
    pid recycling is observed on this fleet, not theoretical
    (verify_cancel.py:313-315: pid_max=4194304, and the laptop's own pid
    counter demonstrably wrapped on 2026-08-11).  So liveness is captured
    ONCE, at entry, strictly before the pgid read and the ppid-map
    snapshot below: an already-reaped leader's descendants have already
    been reparented to init (or the nearest subreaper) and are no longer
    reachable from ``proc.pid`` anyway, so the early return forgoes no
    reachable kill.

    IT IS NOT FREE, THOUGH -- KNOWN RESIDUE (esc-4092-3, measured
    2026-08-30).  "No longer reachable" is a statement about this helper's
    ability to FIND the descendants, NOT a claim that something else kills
    them.  Nothing does.  A session-escaped grandchild (verify.py's
    ``_run_cmd`` runs every build via
    ``create_subprocess_shell(..., start_new_session=True)``, so it holds
    its OWN pgid/sid) is reachable ONLY via the /proc ppid chain, and that
    chain is severed the instant the leader dies.  When a caller reaps the
    leader inside its ``try`` -- as the four rows named above do -- the
    grandchild is already reparented to init by the time this runs, the
    descendant walk returns EMPTY, and the killpg backstop cannot reach it
    either (different group).  It then survives to its full ``sleeper_spec``
    duration.

    MEASURED, not inferred: over 5 instrumented full-module xdist runs, 3
    leaked exactly one ``sleep 300.0``, each reparented to pid 1940
    (``systemd --user``) in its own pgid/sid.  The surviving orphan's
    ``/proc/<pid>/cwd`` resolved to
    ``.../test_watchdog_timeout_env_over0/repo/.worktrees/_merge-<uuid>``
    -- i.e. one of the four already-reaped rows, whose logged teardown
    state was ``returncode=1, /proc entry gone, descendants=[]``.  The two
    ``_KNOWN_HOLDER_TEARDOWN_ROWS`` anchors did NOT leak: their leaders were
    logged alive (``returncode=None``, state ``S``) with a NON-EMPTY
    descendant set containing the real sleeper, which was swept correctly.

    ACCEPTED for now, deliberately: the residue is bounded by the spec's own
    sleep duration, has no correctness impact on the assertions, and closing
    it needs a design change this helper cannot make alone (the descendant
    set must be captured EARLY, while the leader is provably alive, and
    carried into teardown -- a spawn-time pgid does not suffice, because the
    grandchild setsid's into a group of its own).  Tracked as a follow-up;
    do NOT "fix" it by deleting the short circuit, which would reintroduce
    the free-pid walk described above.
    """
    timeout = ROW5_HOLDER_TEARDOWN_CEILING_SECS if timeout is None else timeout

    # An already-reaped leader means proc.pid is a FREE pid -- see the
    # ALREADY-REAPED SHORT CIRCUIT paragraph above.  Captured once, via
    # proc.returncode (a pure read of state this Popen already owns, no
    # waitpid side effect) rather than a fresh proc.poll(), and not
    # re-read later in this function.  ``proc.poll()`` is deliberately
    # called NOWHERE in this helper: a poll() after this point would
    # ``waitpid``-reap a leader that exited in the meantime and FREE
    # proc.pid, aiming the killpg backstop below at a recycled group --
    # see the unguarded leader SIGKILL further down for the full argument.
    if proc.returncode is not None:
        if proc.stdin is not None:
            with contextlib.suppress(OSError):
                proc.stdin.close()
        return

    # Pre-kill snapshot phase -- BOTH reads must happen before any signal is
    # sent.  Killing the leader first reparents survivors to init and severs
    # the /proc parent chain, making a session-escaped descendant unfindable
    # (the same invariant cancel_request documents at
    # verify_cancel.py:246-250).  The pgid is read only to gate the killpg
    # backstop below; a leader already gone by now makes getpgid raise
    # ProcessLookupError, and a permission mismatch or any other OSError
    # degrades the same way -- "no group to backstop" -- rather than
    # propagating out of a finally and masking whatever real assertion
    # failure the caller was cleaning up after.
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = None
    # A /proc read losing a race with process exit degrades to an empty map
    # rather than propagating -- this helper runs almost exclusively inside
    # a finally block, where an exception would mask whatever real assertion
    # failure the caller was cleaning up after.
    try:
        ppid_map = _ppid_map_provider()
    except OSError:
        ppid_map = {}
    descendants = collect_descendants(proc.pid, ppid_map)

    # SIGKILL every descendant.  Already-dead and not-ours are both expected
    # outcomes, not errors.
    for pid in descendants:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            _kill(pid, signal.SIGKILL)

    # SIGKILL the leader.  UNGUARDED, deliberately: there is no
    # `if proc.poll() is None:` here, and adding one back would REINTRODUCE
    # the pid-recycling hazard the backstop below exists to avoid.
    # ``Popen.poll()`` is not a passive read -- it calls ``_internal_poll()``
    # -> ``os.waitpid(pid, WNOHANG)``, so if the leader exited at any point
    # after the entry-time ``proc.returncode is not None`` short circuit
    # (the Row 5 holder's ``finally`` calls ``stop_heartbeats()`` FIRST,
    # which arms the CLI's stdin-watchdog self-kill, making exactly that a
    # DESIGNED behaviour, not an exotic race), a poll() here would REAP it
    # and FREE proc.pid -- and the backstop below would then evaluate
    # ``pgid == proc.pid`` against a pgid captured pre-reap and fire
    # ``killpg`` at a now-free pgid, SIGKILLing a stranger's whole process
    # group on a shared dev box or CI host.
    #
    # Signalling here is safe unguarded and costs nothing: the entry-time
    # short circuit already established the leader was un-reaped, and this
    # Popen object is the ONLY thing that can reap it, so with no poll() in
    # this function the pid stays PINNED (a zombie at worst) until
    # ``proc.wait()`` below.  SIGKILL to an exited-but-unreaped zombie is a
    # no-op, and ProcessLookupError/PermissionError are suppressed anyway.
    # This is what makes the backstop's "still unreaped, pid provably
    # pinned" premise actually true rather than merely asserted.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        _kill(proc.pid, signal.SIGKILL)

    # Guarded killpg backstop for same-group stragglers -- fires ONLY when
    # the holder is PROVABLY its own group leader (setsid ran, i.e. the CLI
    # was invoked with --request-id) AND that group is PROVABLY not the
    # caller's own.  Both checks are required, not either alone: cli.py's
    # os.setsid is gated on --request-id and spawn_verify_merge never passes
    # start_new_session=, so a holder that never setsid'd (e.g. the
    # lane-lock site) shares this process's own group -- an unguarded
    # killpg there would SIGKILL the pytest worker itself (see
    # test_kill_holder_tree_never_signals_the_callers_own_process_group).
    #
    # Fired HERE -- immediately after the leader SIGKILL and BEFORE
    # proc.wait() reaps it below, not after.  While still unreaped the
    # leader is a zombie that keeps its pid pinned, so pgid provably still
    # denotes the holder's own group; group members that outlive it are
    # killed just as effectively.  Firing this AFTER the reap would let a
    # pid-recycling race aim killpg at a stranger's group instead -- the
    # exact hazard the ALREADY-REAPED SHORT CIRCUIT above exists to avoid,
    # and pid recycling is observed on this fleet, not theoretical
    # (verify_cancel.py:313-315: pid_max=4194304, and the laptop's own pid
    # counter demonstrably wrapped on 2026-08-11).
    if pgid is not None and pgid == proc.pid and pgid != os.getpgid(0):
        with contextlib.suppress(ProcessLookupError, OSError):
            _killpg(pgid, signal.SIGKILL)

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # A leader that will not die must not mask the test's own failure by
        # raising out of a finally -- surface it loudly instead.
        warnings.warn(
            f'kill_holder_tree: leader pid={proc.pid} did not exit within '
            f'{timeout}s of SIGKILL',
            stacklevel=2,
        )

    if proc.stdin is not None:
        with contextlib.suppress(OSError):
            proc.stdin.close()


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
# silently multiply this module's runtime (Row 3 from ~17s to ~70s).  The
# pin is now STRUCTURAL, not just asserted: the literals live on
# ROW_WATCHDOG_HEARTBEAT_TIMEOUT_SECS / ROW_WATCHDOG_KILL_GRACE_SECS below,
# and ROW_TREE_KILL_CEILING_SECS / ROW_PER_TEST_TIMEOUT_SECS are computed
# from those two floats, so they can no longer drift out of step with the
# window.  Nothing here imports verify_cancel's watchdog timing constants
# (WATCHDOG_HEARTBEAT_TIMEOUT_SECS / WATCHDOG_KILL_GRACE_SECS) -- the /proc
# walkers imported at the top of this module (collect_descendants,
# pgid_file, etc.) are deliberately still shared; a dedicated drift test
# below imports the two timing constants LOCALLY, on its own, purely to
# notice if they ever diverge from this pin.
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
ROW_WATCHDOG_HEARTBEAT_TIMEOUT_SECS: float = 10.0
ROW_WATCHDOG_KILL_GRACE_SECS: float = 5.0

#: Worst-case time to fire and finish killing the tree.  Row 3 takes
#: run_stdin_watchdog's select-TIMEOUT branch and pays the full sum; Rows
#: 1/2 take the EOF branch and pay only the grace.
ROW_WATCHDOG_WINDOW_SECS: float = (
    ROW_WATCHDOG_HEARTBEAT_TIMEOUT_SECS + ROW_WATCHDOG_KILL_GRACE_SECS
)  # 15.0

ROW_WATCHDOG_ENV: dict[str, str] = {
    'ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS': str(ROW_WATCHDOG_HEARTBEAT_TIMEOUT_SECS),
    'ORCH_WATCHDOG_KILL_GRACE_SECS': str(ROW_WATCHDOG_KILL_GRACE_SECS),
}

#: Ceiling for the rows' child.wait()/wait_subtree_gone() polls: the full
#: window plus load headroom.  A WEDGE-DETECTOR, not a speed assertion (the
#: rows assert THAT the tree was killed, never how fast), so on the success
#: path a wider ceiling costs zero wall-clock and is paid only when the test
#: is already failing.
ROW_TREE_KILL_CEILING_SECS: float = ROW_WATCHDOG_WINDOW_SECS + 15.0  # 30.0

#: Ceiling for the marker waits (:func:`wait_for_marker` /
#: :func:`wait_for_marker_stable`, whose bare defaults resolve to it).  Value
#: unchanged from the literal it replaces; the point of naming it is that
#: Row 5's budget below references the SAME symbol those helpers spend, so the
#: two cannot drift.  NOT load-scaled: unlike descendant discovery, this waits
#: on a single already-forked build shell reaching one ``touch``, and no
#: measurement in this repo implicates it in a load flake.
ROW_MARKER_CEILING_SECS: float = 20.0

#: Base ceiling for the two DISCOVERY waits every row runs BEFORE the
#: watchdog is even armed -- wait_for_pgid_file and wait_subtree_live.  Rows
#: 1/2/3 pass the resolved ceiling explicitly at their call sites below
#: (instead of relying on the bare default) so this value and those defaults
#: cannot silently drift apart.
ROW_DISCOVERY_CEILING_BASE_SECS: float = 20.0

#: Task 4014.  Top of the measured per-core dilation envelope, NOT a guess.
#: Every dilation figure this repo has actually measured is expressed in
#: loadavg-per-core terms and lands in the 3-6x band: tests/warm-lane's
#: README records 3-6x at loadavg 124 on 32 cores plus a 6.2x module figure;
#: this module's own banner records producer gaps at loadavg 113-178; and the
#: 3689 steward measured THIS test pair stretching 10s -> 51-59s (~5-6x).
#: Scaling keys on loadavg-per-core rather than PSI because this suite's
#: autouse _hermetic_psi_reader fixture injects a stub PSI reader suite-wide,
#: so /proc/pressure readings are deliberately untrustworthy here.
_DISCOVERY_CEILING_MAX_SCALE: float = 6.0

#: Operator knob: pin the discovery ceiling outright (e.g. to reproduce a
#: discovery timeout quickly instead of waiting out a load-scaled deadline).
#: Namespaced ORCH_TEST_*, NOT bare ORCH_*: subprocess_env() copies
#: os.environ into every spawned verify-merge, and the ORCH_* namespace there
#: is production knobs the CLI actually reads.
_DISCOVERY_CEILING_OVERRIDE_ENV: str = 'ORCH_TEST_DISCOVERY_CEILING_SECS'


def _resolve_ceiling_override(environ) -> float | None:
    """Parse :data:`_DISCOVERY_CEILING_OVERRIDE_ENV` out of *environ*.

    Returns None when unset, and warns-then-returns-None on a value that is
    unparseable, non-finite, or non-positive.  Deliberately does NOT raise:
    this is resolved at import, so raising would fail COLLECTION for the
    entire module -- every row and every helper test taken down by one
    mistyped debugging knob, a strictly worse and far more confusing failure
    than the one being guarded against.  Silently ignoring it would be worse
    still, hence the warning.

    NON-FINITE is rejected for exactly that reason, and is not a hypothetical:
    ``inf`` is the natural spelling an operator reaches for to mean "never
    time out", and ``float('inf')`` parses cleanly while satisfying every
    ordering guard (``inf <= 0`` and ``nan <= 0`` are both False).  It would
    therefore flow into ROW_DISCOVERY_CEILING_MAX_SECS and blow up the very
    next module-level statement -- ``int(... + 2 * inf)`` raises OverflowError,
    ``nan`` raises ValueError -- taking down collection of the whole module,
    i.e. precisely the failure this warn-instead-of-raise contract exists to
    prevent.

    Takes an environ MAPPING rather than reading ``os.environ`` so it is
    deterministically testable with no monkeypatch-versus-import-order race.
    """
    raw = environ.get(_DISCOVERY_CEILING_OVERRIDE_ENV)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = None
    if value is None or not math.isfinite(value) or value <= 0:
        warnings.warn(
            f'{_DISCOVERY_CEILING_OVERRIDE_ENV}={raw!r} is not a finite positive number; '
            f'ignoring it and using the load-scaled discovery ceiling instead',
            stacklevel=2,
        )
        return None
    return value


#: Resolved ONCE at import -- see the module's discovery-ceiling notes and
#: :func:`row_discovery_ceiling_secs`.  A per-call env read could return a
#: value the already-baked @pytest.mark.timeout below cannot accommodate.
_DISCOVERY_CEILING_OVERRIDE_SECS: float | None = _resolve_ceiling_override(os.environ)

#: The bound the import-time @pytest.mark.timeout below is derived from.  No
#: value row_discovery_ceiling_secs() can return may exceed it -- see
#: test_row_per_test_timeout_still_covers_the_max_discovery_ceiling.  The
#: override is folded in (rather than sitting outside the bound) so that
#: invariant holds for every reachable value, pinned or scaled.
ROW_DISCOVERY_CEILING_MAX_SECS: float = (
    _DISCOVERY_CEILING_OVERRIDE_SECS
    if _DISCOVERY_CEILING_OVERRIDE_SECS is not None
    else ROW_DISCOVERY_CEILING_BASE_SECS * _DISCOVERY_CEILING_MAX_SCALE
)  # 120.0 unpinned


def row_discovery_ceiling_secs(
    *, _loadavg=None, _cpu_count=None, _override=_DISCOVERY_CEILING_OVERRIDE_SECS
) -> float:
    """Resolve the discovery ceiling for a wait that is starting NOW.

    A WEDGE DETECTOR, not a speed assertion: the rows assert THAT a tree was
    discovered, never how fast, so on the success path a wider ceiling costs
    ZERO wall clock and is paid only when the test is already failing.  That
    asymmetry is what makes scaling all the way to
    :data:`_DISCOVERY_CEILING_MAX_SCALE` safe for suite runtime -- an idle
    box still pays exactly :data:`ROW_DISCOVERY_CEILING_BASE_SECS`.

    Sampled per CALL rather than at import so a row that starts mid-storm
    sees the storm, and clamped at both ends: never tighter than base (a
    quiet box must not get a stricter deadline than it has always had) and
    never wider than :data:`ROW_DISCOVERY_CEILING_MAX_SECS` (the bound the
    import-time pytest timeout is derived from).

    An operator pin via :data:`_DISCOVERY_CEILING_OVERRIDE_ENV` is returned
    VERBATIM, bypassing load scaling in both directions -- otherwise "pin it
    low to reproduce the timeout quickly" would silently do nothing on a busy
    box.  It is safe against the clamp invariant because
    :data:`ROW_DISCOVERY_CEILING_MAX_SECS` folds the same pin in.

    *_loadavg* / *_cpu_count* / *_override* are private injectable seams for
    deterministic coverage, defaulting to the real readers.
    """
    if _override is not None:
        return _override
    loadavg = os.getloadavg()[0] if _loadavg is None else _loadavg
    cpu_count = (os.cpu_count() or 1) if _cpu_count is None else _cpu_count
    scaled = ROW_DISCOVERY_CEILING_BASE_SECS * (loadavg / max(cpu_count, 1))
    # Clamp against the UNPINNED bound, not ROW_DISCOVERY_CEILING_MAX_SECS:
    # the latter folds an operator pin in, and a pin is already returned above.
    # Reading it here would make a low pin silently re-clamp the scaled branch
    # too, so the `_override=None` seam would not mean "as if unpinned".
    unpinned_max = ROW_DISCOVERY_CEILING_BASE_SECS * _DISCOVERY_CEILING_MAX_SCALE
    return min(unpinned_max, max(ROW_DISCOVERY_CEILING_BASE_SECS, scaled))


#: Worst-case BOUNDED work in a row on the failure path, at the WIDEST
#: discovery ceiling reachable: both discovery waits, then one full watchdog
#: window, then both ceiling-bounded kill-confirmation waits run to their
#: full ceiling.  (Deliberately excludes _setup_verify_repo's real git work
#: and the finally block's ~5s child.kill()/wait() tail -- both small next to
#: the headroom below.)  Keyed on ROW_DISCOVERY_CEILING_MAX_SECS, not on the
#: base: task 4014 made the discovery ceiling load-scaled, and a widened wait
#: the @pytest.mark.timeout below cannot accommodate would be TRUNCATED by
#: pytest-timeout's thread-mode os._exit() -- reporting as a worker kill
#: instead of the row's self-diagnosing AssertionError.
_ROW_WORST_CASE_FIXED_SECS: float = (
    ROW_WATCHDOG_WINDOW_SECS + 2 * ROW_TREE_KILL_CEILING_SECS
)  # 75.0

#: Per-test opt-out from this module's inherited `timeout = 60`
#: (orchestrator/pyproject.toml:103, thread-mode -- os._exit()s the xdist
#: worker on expiry).  2x the FIXED terms, because this module's own comment
#: (~:870-879) records ~15x elasticity on `from orchestrator.cli import main`
#: under a full-suite storm and those terms do not track load themselves.
#: The discovery term is counted ONCE and exempted from that 2x: it now
#: scales with loadavg-per-core on its own, so applying the storm factor to
#: it as well would double-count the same elasticity.
ROW_PER_TEST_TIMEOUT_SECS: int = int(
    2 * _ROW_WORST_CASE_FIXED_SECS + 2 * ROW_DISCOVERY_CEILING_MAX_SECS
)  # 390

#: Row 5's own bounded terms.  It does not share Rows 1-4's shape -- no
#: watchdog window, no wait_subtree_gone kill confirmation -- so it cannot
#: share their constant, only their DERIVATION.  Named here (rather than
#: passed as literals at the call sites) so the budget below and the waits it
#: is supposed to cover move together.
ROW5_WAITER_COMPLETION_CEILING_SECS: float = 60.0
ROW5_HOLDER_TEARDOWN_CEILING_SECS: float = 5.0

#: Worst-case FIXED (non-load-scaled) work in Row 5: the marker wait, the
#: waiter subprocess's completion ceiling, and the finally block's holder
#: teardown.  The marker term is counted TWICE because
#: :func:`wait_for_marker_stable` spends its timeout on the existence gate and
#: then a FRESH deadline on the settle loop -- worst case 40s, not 20s, a term
#: the literal mark this replaced hid entirely.  (Deliberately excludes
#: _setup_verify_repo's real git work and the in-process consumer half -- both
#: small next to the headroom below, exactly as for Rows 1-4.)
_ROW5_WORST_CASE_FIXED_SECS: float = (
    2 * ROW_MARKER_CEILING_SECS
    + ROW5_WAITER_COMPLETION_CEILING_SECS
    + ROW5_HOLDER_TEARDOWN_CEILING_SECS
)  # 105.0

#: Row 5's per-test opt-out from the inherited `timeout = 60`, in the same
#: shape as ROW_PER_TEST_TIMEOUT_SECS above: 2x the FIXED terms (they do not
#: track load, and this module records ~15x elasticity on the child's
#: `from orchestrator.cli import main` under a full-suite storm), plus the two
#: discovery waits counted ONCE at MAX and exempted from that 2x -- they now
#: scale with loadavg-per-core themselves, so applying the storm factor to
#: them as well would double-count the same elasticity.
ROW5_PER_TEST_TIMEOUT_SECS: int = int(
    2 * _ROW5_WORST_CASE_FIXED_SECS + 2 * ROW_DISCOVERY_CEILING_MAX_SECS
)  # 450
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Task 4025 amendment -- drift detector for the pin above.  ROW_WATCHDOG_ENV
# is pinned to today's production numbers but deliberately does not TRACK
# verify_cancel's constants (see the banner above); that decoupling means
# nothing else notices the day the two diverge.  This is the one test whose
# whole job is to notice, so the import it performs (LOCAL to the test body,
# not module-level) is a deliberate, narrow exception to "nothing here
# imports verify_cancel's watchdog timing constants" above -- it exists
# specifically to compare against them, not to consume them.
# ---------------------------------------------------------------------------


def test_row_watchdog_window_still_matches_production_constants():
    """Fails the day ROW_WATCHDOG_* stops matching verify_cancel's WATCHDOG_*.

    task 4025 pinned ROW_WATCHDOG_HEARTBEAT_TIMEOUT_SECS /
    ROW_WATCHDOG_KILL_GRACE_SECS as literals specifically so Rows 1/2/3's
    runtime does NOT track verify_cancel.WATCHDOG_HEARTBEAT_TIMEOUT_SECS /
    WATCHDOG_KILL_GRACE_SECS -- task 4195 is expected to retune those toward
    ~60s+.  Today the pinned literals happen to equal production, so this
    passes; it exists to catch the moment that stops being true.

    If this fails: production changed (task 4195 landing is the expected
    trigger).  Do NOT "fix" it by importing the pinned literals from
    verify_cancel -- that reintroduces the silent-runtime-multiplication
    this task exists to prevent.  Instead update the three Row 1/2/3
    docstrings' "pinned production-equivalent window" wording (it will no
    longer be accurate), decide afresh whether
    ROW_WATCHDOG_HEARTBEAT_TIMEOUT_SECS / ROW_WATCHDOG_KILL_GRACE_SECS
    should move, and re-baseline this assert against the new production
    values.
    """
    from orchestrator import verify_cancel

    pinned = (ROW_WATCHDOG_HEARTBEAT_TIMEOUT_SECS, ROW_WATCHDOG_KILL_GRACE_SECS)
    production = (
        verify_cancel.WATCHDOG_HEARTBEAT_TIMEOUT_SECS,
        verify_cancel.WATCHDOG_KILL_GRACE_SECS,
    )
    assert pinned == production, (
        f'pinned ROW_WATCHDOG_* {pinned} no longer match production '
        f'verify_cancel.WATCHDOG_* {production} -- the pin intentionally '
        f'does not track production; update the docstrings\' '
        f'"production-equivalent" wording and re-baseline this assert.'
    )


# ---------------------------------------------------------------------------
# Task 4014 -- deterministic coverage for the load-scaled discovery ceiling.
# Pure function, injected seams: no clock, no /proc, no subprocess.
# ---------------------------------------------------------------------------


def test_row_discovery_ceiling_scales_with_load_and_clamps():
    """The discovery ceiling widens with per-core load and is bounded at both ends.

    This is a WEDGE DETECTOR, not a speed assertion -- the rows assert THAT
    a tree was discovered, never how fast -- so on the success path a wider
    ceiling costs ZERO wall clock and is paid only when the test is already
    failing.  The four properties that matter:

    (a) an IDLE box must get exactly the base ceiling, so a quiet run is
        never slowed down;
    (b) real per-core load must actually widen it (3x per core is the BOTTOM
        of this repo's measured dilation envelope);
    (c) an absurd load must clamp, so a runaway box cannot stretch a row
        past the pytest timeout that is DERIVED from that clamp;
    (d) the mapping is monotone -- more load never yields a tighter deadline.
    """
    idle = row_discovery_ceiling_secs(_override=None, _loadavg=0.1, _cpu_count=32)
    assert idle == ROW_DISCOVERY_CEILING_BASE_SECS, (
        f'an idle box must pay exactly the base ceiling '
        f'({ROW_DISCOVERY_CEILING_BASE_SECS}); got {idle}'
    )

    loaded = row_discovery_ceiling_secs(_override=None, _loadavg=96.0, _cpu_count=32)
    assert loaded > ROW_DISCOVERY_CEILING_BASE_SECS, (
        f'at 3x per-core load -- the BOTTOM of the measured dilation envelope -- '
        f'the ceiling must widen past base; got {loaded}'
    )

    unpinned_clamp = ROW_DISCOVERY_CEILING_BASE_SECS * _DISCOVERY_CEILING_MAX_SCALE
    absurd = row_discovery_ceiling_secs(_override=None, _loadavg=6400.0, _cpu_count=32)
    assert absurd == unpinned_clamp, (
        f'a runaway load must clamp to {unpinned_clamp} -- that clamp is what '
        f'ROW_DISCOVERY_CEILING_MAX_SECS, and in turn the import-time pytest '
        f'timeout, are derived from; got {absurd}'
    )

    ladder = [
        row_discovery_ceiling_secs(_override=None, _loadavg=load, _cpu_count=32)
        for load in (0.0, 1.0, 32.0, 64.0, 96.0, 200.0, 1000.0, 6400.0)
    ]
    assert ladder == sorted(ladder), (
        f'more load must never yield a TIGHTER deadline; got {ladder}'
    )
    assert all(v >= ROW_DISCOVERY_CEILING_BASE_SECS for v in ladder), (
        f'no reachable value may drop below the base ceiling; got {ladder}'
    )


def test_discovery_ceiling_env_override_parses_and_pins():
    """ORCH_TEST_DISCOVERY_CEILING_SECS parses safely and beats load scaling.

    Two halves, both deterministic -- the parser takes an environ MAPPING
    rather than reading os.environ, so neither half monkeypatches import-time
    state or races import order.

    (a) Parsing.  A good value parses; an absent one is None; a typo'd one is
        None AND warns.  Warning rather than raising is deliberate: this
        constant is resolved at IMPORT, so raising would fail COLLECTION for
        the whole module -- every row and every helper test taken down by one
        mistyped debugging knob, which is a strictly worse and far more
        confusing failure than the one being guarded against.  Silently
        ignoring it would be worse still (the project's no-silent-fail-soft
        norm), hence the warning.

        NON-FINITE inputs are covered alongside the typo because they defeat
        that contract in a way a typo does not: 'banana' fails float(), but
        'inf'/'nan' PARSE, and `inf <= 0` / `nan <= 0` are both False, so
        without an explicit finiteness guard they sail through into
        ROW_DISCOVERY_CEILING_MAX_SECS and detonate the next module-level
        statement (`int(... + 2*inf)` -> OverflowError; nan -> ValueError),
        failing collection for the whole module.  'inf' is not an exotic
        input either: it is what an operator writes to mean "never time out".

    (b) Pinning.  An explicit operator pin wins in BOTH directions -- it must
        override load scaling upward and downward alike, or "pin it to 7.5 to
        reproduce the timeout quickly" silently does nothing on a busy box.

    The knob is namespaced ORCH_TEST_*, not bare ORCH_*: subprocess_env()
    copies os.environ into every spawned verify-merge, and the ORCH_*
    namespace there is production knobs the CLI actually reads.
    """
    assert _resolve_ceiling_override({'ORCH_TEST_DISCOVERY_CEILING_SECS': '7.5'}) == 7.5
    assert _resolve_ceiling_override({}) is None

    for bad in ('banana', 'inf', '-inf', 'nan', '0', '-1'):
        with pytest.warns(UserWarning, match='ORCH_TEST_DISCOVERY_CEILING_SECS'):
            assert _resolve_ceiling_override(
                {'ORCH_TEST_DISCOVERY_CEILING_SECS': bad}
            ) is None, f'{bad!r} must be rejected, not pinned'

    # The consequence the rejection above exists to prevent, asserted against
    # the module's OWN import-time constants -- the values THIS process is
    # actually running under, whatever env it was collected with.  A local
    # re-derivation would have been vacuous twice over: every value the
    # resolver accepts is finite and positive by construction, and a copy of
    # the derivation cannot notice the MODULE's derivation being loosened.
    assert math.isfinite(ROW_DISCOVERY_CEILING_MAX_SECS) and ROW_DISCOVERY_CEILING_MAX_SECS > 0, (
        f'ROW_DISCOVERY_CEILING_MAX_SECS resolved to '
        f'{ROW_DISCOVERY_CEILING_MAX_SECS} -- a non-finite or non-positive '
        f'ceiling reached import, so the finiteness guard above is being bypassed'
    )
    for name, derived in (
        ('ROW_PER_TEST_TIMEOUT_SECS', ROW_PER_TEST_TIMEOUT_SECS),
        ('ROW5_PER_TEST_TIMEOUT_SECS', ROW5_PER_TEST_TIMEOUT_SECS),
    ):
        assert derived > 0, (
            f'{name} derived to {derived} -- the per-test timeouts are int() of a '
            f'sum containing the ceiling, so this is what a non-finite ceiling '
            f'looks like on the far side of the derivation'
        )

    assert row_discovery_ceiling_secs(_override=7.5, _loadavg=6400.0, _cpu_count=32) == 7.5
    assert row_discovery_ceiling_secs(_override=7.5, _loadavg=0.0, _cpu_count=32) == 7.5


def test_row_per_test_timeout_still_covers_the_max_discovery_ceiling():
    """The per-test pytest timeout must cover the WIDEST discovery ceiling reachable.

    Extends task 4025's "the coupling is STRUCTURAL, not merely asserted"
    property (commit 71a4d37f17) to the now load-scaled discovery ceiling,
    against the AMBIENT module constants.

    Without (b), a load-widened discovery wait would be silently truncated by
    the import-time ``@pytest.mark.timeout(...)`` decorator -- and a widened
    deadline that never gets to run is WORSE than no fix at all, because
    pytest-timeout's thread mode ``os._exit()``s the xdist worker: the row
    then reports as a worker kill rather than as the self-diagnosing
    AssertionError (leader rc taxonomy + stderr tail) task 4025-alpha built
    precisely so this failure would explain itself.
    """
    resolved = row_discovery_ceiling_secs()
    assert resolved <= ROW_DISCOVERY_CEILING_MAX_SECS, (
        f'row_discovery_ceiling_secs() returned {resolved}, above the '
        f'ROW_DISCOVERY_CEILING_MAX_SECS ({ROW_DISCOVERY_CEILING_MAX_SECS}) that '
        f'ROW_PER_TEST_TIMEOUT_SECS is derived from -- the widened wait would be '
        f'truncated by @pytest.mark.timeout instead of running'
    )
    if _DISCOVERY_CEILING_OVERRIDE_SECS is None:
        # An operator pin deliberately escapes the base floor (pinning LOW is
        # how you reproduce a discovery timeout quickly), so the floor is
        # asserted only for the unpinned, CI-reachable configuration.
        assert resolved >= ROW_DISCOVERY_CEILING_BASE_SECS, (
            f'row_discovery_ceiling_secs() returned {resolved}, below the base '
            f'ceiling ({ROW_DISCOVERY_CEILING_BASE_SECS}) -- load scaling must '
            f'never hand a quiet box a TIGHTER deadline than it always had'
        )

    required = (
        2 * ROW_DISCOVERY_CEILING_MAX_SECS
        + ROW_WATCHDOG_WINDOW_SECS
        + 2 * ROW_TREE_KILL_CEILING_SECS
    )
    assert required <= ROW_PER_TEST_TIMEOUT_SECS, (
        f'ROW_PER_TEST_TIMEOUT_SECS ({ROW_PER_TEST_TIMEOUT_SECS}) does not cover '
        f'the bounded worst case at the MAX discovery ceiling ({required}) -- '
        f'both discovery waits, one full watchdog window, and both '
        f'kill-confirmation waits run to their ceiling'
    )


# ---------------------------------------------------------------------------
# Task 4014 (review fix) -- widen the coupling guard from "the derived
# constant is big enough" to "every CALL SITE performing a scaled discovery
# wait is actually covered by the mark it runs under".
#
# The defect this pair exists to catch is not a wrong number, it is a guard
# whose SCOPE did not include a call site.  The test above pins
# ROW_PER_TEST_TIMEOUT_SECS against ROW_DISCOVERY_CEILING_MAX_SECS but never
# looks at WHO performs a discovery wait -- so Row 5, which relies on the bare
# (now load-scaled) defaults while carrying task 2921's unrelated LITERAL
# @pytest.mark.timeout(120), was invisible to it.  Two 120s-capable waits
# inside a 120s mark means a wedged Row 5 discovery gets truncated by
# pytest-timeout's thread-mode os._exit() of the whole xdist worker instead of
# raising this module's self-diagnosing AssertionError (leader rc taxonomy +
# bounded stderr tail) -- the exact silent-truncation class the derivation
# chain exists to prevent.
#
# The sweep is ENUMERATION-FREE: it derives its call-site list from this
# module's own AST, so a future Row 6 registers itself with the guard simply
# by calling the helper.  A hand-maintained row list would have the identical
# failure mode the day one lands.  Self-parsing AST guards are an established
# idiom in this suite (test_marker_registration_drift.py,
# test_git_repo_isolation_guard.py, test_lock_release_single_writer_guard.py,
# test_event_loop_antipattern_guard.py) and cost nothing at runtime -- pure
# parse, no subprocess, no timing.
# ---------------------------------------------------------------------------

#: The helpers whose ``timeout`` default now resolves to
#: :func:`row_discovery_ceiling_secs`, i.e. can spend up to
#: ROW_DISCOVERY_CEILING_MAX_SECS each.
_SCALED_DISCOVERY_HELPERS: frozenset[str] = frozenset(
    {'wait_for_pgid_file', 'wait_subtree_live'}
)

#: The seams that TOGETHER mean "this call never touches real /proc" -- see
#: :func:`_scaled_discovery_call_count`.  BOTH are required: either one alone
#: leaves the other half of the poll tick reading real /proc.
_DISCOVERY_SEAM_KWARGS: frozenset[str] = frozenset({'_probe_children', '_ppid_map'})

#: Last-resort stand-in for the pytest-timeout budget an unmarked test
#: inherits, used ONLY when pytest-timeout did not configure this run at all
#: (``-p no:timeout``, or a version that stops exposing its resolved value).
#: The live value is read by :func:`_inherited_timeout_budget_secs` --
#: hardcoding orchestrator/pyproject.toml's ``timeout = 60`` would make this
#: guard reason from exactly the kind of unreachable literal its closing
#: assertion bans.
_INHERITED_TIMEOUT_FALLBACK_SECS: float = 60.0


def _inherited_timeout_budget_secs(config) -> float:
    """The pytest-timeout budget an UNMARKED test in this module inherits.

    Read from ``config._env_timeout`` -- the plugin's OWN resolved value, i.e.
    literally what ``pytest_timeout._get_item_settings`` hands an item carrying
    no ``timeout`` marker -- rather than duplicated as a literal, so a change to
    orchestrator/pyproject.toml cannot leave the sweep below reasoning from, and
    misreporting, a stale number.

    ``getini('timeout')`` alone will NOT do, MEASURED here on pytest-timeout
    2.4.0: it returns the STRING ``'60'`` (so float() coercion is mandatory),
    returns ``''`` on a root-bound run because the repo-root pyproject declares
    no ``timeout`` key at all, and -- the one that matters -- stays STALE at
    ``'60'`` under ``--timeout=300``, which this repo's per-module verify
    commands do pass.  ``get_env_settings()`` resolves ``--timeout`` ->
    ``PYTEST_TIMEOUT`` -> truthy ini and stores the winner as ``_env_timeout``;
    reading the winner cannot drift from the plugin's chain the way a local
    re-implementation of that chain would.

    ``None`` means no timeout was configured anywhere (measured: a root-bound
    run), so nothing can truncate an unmarked test and its budget really is
    unbounded.  Reporting the literal there would make this guard fail tests
    over a truncation that cannot happen.
    """
    if not hasattr(config, '_env_timeout'):
        warnings.warn(
            'pytest-timeout did not configure this run (no config._env_timeout), '
            'so the discovery-wait sweep is falling back to the duplicated '
            f'literal {_INHERITED_TIMEOUT_FALLBACK_SECS}s for unmarked tests',
            stacklevel=2,
        )
        return _INHERITED_TIMEOUT_FALLBACK_SECS
    resolved = config._env_timeout
    if resolved is None:
        return math.inf  # no timeout anywhere: an unmarked test cannot be truncated
    return float(resolved)


#: Anti-vacuity floor for the sweep below: an AST matcher that silently stops
#: matching is a guard reporting PASS while guarding nothing.
_KNOWN_DISCOVERY_ROWS: frozenset[str] = frozenset({
    'test_cancel_verify_tree_kills_under_live_watchdog',                  # Row 4
    'test_orchestrator_killed_mid_build_tree_killed_via_eof',             # Row 1
    'test_ssh_dropped_mid_build_tree_killed_via_eof_dispatcher_alive',    # Row 2
    'test_heartbeat_starved_hard_partition_tree_killed_via_timeout',      # Row 3
    'test_flock_contention_full_two_way_seam_blocks_and_escalates',       # Row 5
})


def _scaled_discovery_call_count(func_node) -> int:
    """How many LOAD-SCALED discovery waits *func_node* performs.

    A call to one of :data:`_SCALED_DISCOVERY_HELPERS` counts when its
    ``timeout=`` kwarg is ABSENT (the bare default, which now resolves to
    ``row_discovery_ceiling_secs()``) or is exactly
    ``row_discovery_ceiling_secs()``.  An explicit unscaled value (e.g.
    ``timeout=30.0``) does not count -- it cannot exceed what it names.

    A call passing the FULL private seam set (``_probe_children`` AND
    ``_ppid_map``) is EXEMPT, and that rule is load-bearing: a fully
    seam-injected call never touches real /proc and returns in microseconds at
    ``interval=0``.  Without it the three deterministic helper tests below
    would each be told to carry a 240s mark -- the guard would push this module
    toward SLOWER tests rather than safer ones.

    Requiring BOTH is equally load-bearing, in the other direction.  A PARTIAL
    injection is not exempt, because it still reads real /proc every tick:
    ``_ppid_map=`` alone still runs the real :func:`_read_direct_children`
    probe, and ``_probe_children=`` alone still runs the real
    ``read_ppid_map()`` full walk on every non-negative tick.  Either can also
    still run the poll loop to the full ``row_discovery_ceiling_secs()``
    deadline if the injected half never yields a descendant -- exactly the
    truncatable wait this sweep exists to catch.  No call site does this today;
    exempting on ANY seam would have left the hole open for the first one that
    does.
    """
    count = 0
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Name)
            and node.func.id in _SCALED_DISCOVERY_HELPERS
        ):
            continue
        # Superset, not intersection: the FULL seam set must be injected.
        if {kw.arg for kw in node.keywords} >= _DISCOVERY_SEAM_KWARGS:
            continue
        timeout_arg = next(
            (kw.value for kw in node.keywords if kw.arg == 'timeout'), None
        )
        if timeout_arg is None:
            count += 1  # bare default -> row_discovery_ceiling_secs()
        elif (
            isinstance(timeout_arg, ast.Call)
            and isinstance(timeout_arg.func, ast.Name)
            and timeout_arg.func.id == 'row_discovery_ceiling_secs'
        ):
            count += 1
    return count


def _timeout_mark_argument(func_node):
    """The single argument node of *func_node*'s ``@pytest.mark.timeout(...)``.

    Returns None when the function carries no such decorator (it then
    inherits the run's budget -- see :func:`_inherited_timeout_budget_secs`).
    """
    for decorator in func_node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == 'timeout'
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == 'mark'
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == 'pytest'
        ):
            return decorator.args[0] if decorator.args else None
    return None


def test_row5_per_test_timeout_covers_its_own_bounded_work():
    """Row 5's pytest timeout must cover ITS bounded worst case, and be DERIVED.

    Row 5 does not share Rows 1-4's shape -- no watchdog window, no
    ``wait_subtree_gone`` kill confirmation -- so it cannot share their
    constant.  Its bounded terms are: two discovery waits (both on the
    load-scaled ceiling), the marker wait, the waiter subprocess's completion
    ceiling, and the ``finally`` block's holder teardown.

    The marker term is counted TWICE on purpose:
    :func:`wait_for_marker_stable` spends its ``timeout`` on the
    :func:`wait_for_marker` existence gate and then a FRESH ``timeout`` on the
    settle loop, so its worst case is 2x, not 1x -- a term the old literal 120
    hid entirely.

    The second half reads the mark ACTUALLY APPLIED to the live function
    object rather than trusting the constant in isolation, so the derivation
    and the decorator cannot diverge.
    """
    required = (
        2 * ROW_DISCOVERY_CEILING_MAX_SECS
        + 2 * ROW_MARKER_CEILING_SECS
        + ROW5_WAITER_COMPLETION_CEILING_SECS
        + ROW5_HOLDER_TEARDOWN_CEILING_SECS
    )
    assert required <= ROW5_PER_TEST_TIMEOUT_SECS, (
        f'ROW5_PER_TEST_TIMEOUT_SECS ({ROW5_PER_TEST_TIMEOUT_SECS}) does not cover '
        f"Row 5's bounded worst case ({required}) -- both discovery waits at the "
        f'MAX ceiling, the marker existence gate AND its settle loop, the waiter '
        f"subprocess's completion ceiling, and the holder teardown"
    )

    applied = [
        mark
        for mark in getattr(
            test_flock_contention_full_two_way_seam_blocks_and_escalates,
            'pytestmark',
            [],
        )
        if mark.name == 'timeout'
    ]
    assert len(applied) == 1, (
        f'expected exactly one @pytest.mark.timeout on Row 5, got {applied!r}'
    )
    assert applied[0].args[0] == ROW5_PER_TEST_TIMEOUT_SECS, (
        f'Row 5 runs under a {applied[0].args[0]}s mark but its derivation says '
        f'{ROW5_PER_TEST_TIMEOUT_SECS}s -- the decorator and the derived constant '
        f'have diverged, so the derivation is guarding nothing'
    )


def test_every_scaled_discovery_wait_is_covered_by_its_test_timeout_mark(request):
    """No test may perform more discovery waiting than its own mark allows.

    Enumeration-free sweep of this module's own AST (see the banner above).
    For every ``test_*`` function it counts the LOAD-SCALED discovery waits
    the source actually contains, resolves the EFFECTIVE pytest-timeout
    budget that test runs under -- from the LIVE config for unmarked tests,
    never from a duplicated ini literal -- and asserts the budget covers those
    waits at the widest ceiling they can reach.

    It additionally BANS a literal timeout mark on any test performing such a
    wait.  That kills the root cause directly rather than the symptom: task
    2921's ``@pytest.mark.timeout(120)`` was a literal that no re-derivation
    could reach into, which is precisely why task 4014's load-scaled
    discovery ceiling could not propagate to Row 5.
    """
    tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
    swept: dict[str, tuple[int, ast.AST]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not node.name.startswith('test_'):
            continue
        n_scaled = _scaled_discovery_call_count(node)
        if n_scaled:
            swept[node.name] = (n_scaled, node)

    missing = sorted(_KNOWN_DISCOVERY_ROWS - set(swept))
    assert not missing, (
        f'the AST sweep matched no scaled discovery wait in {missing} -- a matcher '
        f'that silently stops matching is a guard reporting PASS while guarding '
        f'nothing (matched: {sorted(swept)})'
    )

    inherited = _inherited_timeout_budget_secs(request.config)
    under_budget: list[str] = []
    literal_marked: list[str] = []
    for name, (n_scaled, func_node) in sorted(swept.items()):
        arg = _timeout_mark_argument(func_node)
        if arg is None:
            effective = inherited
        elif isinstance(arg, ast.Name) and arg.id in globals():
            effective = float(globals()[arg.id])
        else:
            literal_marked.append(f'{name}: @pytest.mark.timeout({ast.unparse(arg)})')
            effective = (
                float(arg.value)
                if isinstance(arg, ast.Constant) and isinstance(arg.value, int | float)
                else inherited
            )
        required = n_scaled * ROW_DISCOVERY_CEILING_MAX_SECS
        if effective < required:
            under_budget.append(
                f'{name}: {n_scaled} scaled discovery wait(s) needing {required}s '
                f'but running under a {effective}s budget'
            )

    assert not under_budget, (
        'these tests can spend more time in discovery waits than their pytest '
        'timeout allows, so a wedged discovery is truncated by pytest-timeout\'s '
        'thread-mode os._exit() of the xdist worker instead of raising the '
        'self-diagnosing AssertionError:\n  ' + '\n  '.join(under_budget)
    )
    assert not literal_marked, (
        'these tests perform a load-scaled discovery wait but pin their pytest '
        'timeout to a LITERAL, which no re-derivation of the discovery ceiling '
        'can reach into -- use a module-level derived constant instead:\n  '
        + '\n  '.join(literal_marked)
    )


#: Anti-vacuity floor for the sweep below: an AST matcher that silently stops
#: matching is a guard reporting PASS while guarding nothing.  "At minimum"
#: -- the sweep may (and does, e.g. test_watchdog_timeout_env_override_fires_
#: fast_without_heartbeat) match more holder teardowns than this names; the
#: assertion below only requires this set to be a SUBSET of what was matched.
_KNOWN_HOLDER_TEARDOWN_ROWS: frozenset[str] = frozenset({
    'test_flock_contention_full_two_way_seam_blocks_and_escalates',       # Row 5
    'test_live_verify_merge_holds_lane_lock_real_subprocess',             # lane lock
})


def _spawn_verify_merge_bound_names(func_node) -> set[str]:
    """Local names *func_node* binds via a bare ``X = spawn_verify_merge(...)`` call."""
    names: set[str] = set()
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == 'spawn_verify_merge'
        ):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _bare_kill_offenders(func_node, holder_names: set[str]) -> list[str]:
    """``"func:lineno"`` for every swept holder's raw-kill spelling called directly
    in a ``try`` handler/finalbody that does NOT also call ``kill_holder_tree``.

    Three raw-kill spellings are matched, each a way a future teardown
    could reintroduce the exact orphan task 4092 fixed: ``<holder>.kill()``,
    ``<holder>.terminate()``, and an ``os.kill``/``os.killpg`` call whose
    argument subtree mentions a swept holder name -- most pointedly the
    ``os.killpg(os.getpgid(holder.pid), SIGKILL)`` one-liner
    ``kill_holder_tree``'s own docstring calls out as the dangerous thing a
    future author would reach for instead.

    Each ``finally``/``except`` body is checked independently: a raw kill
    in one body is not excused by a ``kill_holder_tree`` call living in a
    DIFFERENT body of the same ``try``.
    """
    offenders: list[str] = []
    for try_node in ast.walk(func_node):
        if not isinstance(try_node, ast.Try):
            continue
        bodies = [try_node.finalbody] + [handler.body for handler in try_node.handlers]
        for body in bodies:
            if not body:
                continue
            bare_kill_lines: list[int] = []
            has_tree_kill = False
            for stmt in body:
                for sub in ast.walk(stmt):
                    if not isinstance(sub, ast.Call):
                        continue
                    if (
                        isinstance(sub.func, ast.Attribute)
                        and sub.func.attr in ('kill', 'terminate')
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id in holder_names
                    ) or (
                        isinstance(sub.func, ast.Attribute)
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == 'os'
                        and sub.func.attr in ('kill', 'killpg')
                        and any(
                            isinstance(name_node, ast.Name) and name_node.id in holder_names
                            for arg in sub.args
                            for name_node in ast.walk(arg)
                        )
                    ):
                        bare_kill_lines.append(sub.lineno)
                    elif isinstance(sub.func, ast.Name) and sub.func.id == 'kill_holder_tree':
                        has_tree_kill = True
            if bare_kill_lines and not has_tree_kill:
                offenders.extend(
                    f'{func_node.name}:{lineno}' for lineno in bare_kill_lines
                )
    return offenders


def test_every_real_subprocess_holder_teardown_uses_the_tree_killer():
    """Every spawn_verify_merge-bound holder's teardown goes through kill_holder_tree.

    Enumeration-free sweep of this module's own AST (see the banner above
    :func:`test_every_scaled_discovery_wait_is_covered_by_its_test_timeout_mark`).
    For every ``test_*`` function, collects the local names bound from a
    bare ``X = spawn_verify_merge(...)`` call, then asserts that no ``try``
    handler/finalbody in that function calls ``<holder>.kill()``,
    ``<holder>.terminate()``, or ``os.kill``/``os.killpg`` (with a swept
    holder name anywhere in the argument subtree) directly unless the SAME
    body also calls ``kill_holder_tree`` -- any of those raw-kill spellings
    SIGKILLs/terminates only the leader, orphaning any session-escaped
    descendant (e.g. one of verify.py's ``start_new_session`` build
    commands) for the rest of its natural life (task 4092).

    Carries the sibling sweep's anti-vacuity discipline: a matcher that
    silently stops matching is a guard reporting PASS while guarding
    nothing, so :data:`_KNOWN_HOLDER_TEARDOWN_ROWS` anchors the functions
    the sweep must always find a spawn_verify_merge-bound holder in,
    independent of whether their teardown currently passes or fails.
    """
    tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
    swept: dict[str, set[str]] = {}
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not node.name.startswith('test_'):
            continue
        holder_names = _spawn_verify_merge_bound_names(node)
        if not holder_names:
            continue
        swept[node.name] = holder_names
        offenders.extend(_bare_kill_offenders(node, holder_names))

    missing = sorted(_KNOWN_HOLDER_TEARDOWN_ROWS - set(swept))
    assert not missing, (
        f'the AST sweep matched no spawn_verify_merge-bound holder in {missing} '
        f'-- a matcher that silently stops matching is a guard reporting PASS '
        f'while guarding nothing (matched: {sorted(swept)})'
    )

    assert not offenders, (
        'these holder teardown(s) call .kill()/.terminate()/os.kill()/'
        'os.killpg() directly instead of routing through kill_holder_tree, '
        'so a session-escaped descendant (e.g. a verify.py '
        "start_new_session build command) survives the leader's own death "
        'as an orphan:\n  ' + '\n  '.join(offenders)
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
# Task 4014 -- deterministic unit coverage for wait_subtree_live's descendant
# discovery poll (the "no descendant appeared within 20.0s" flake).  Every
# test here injects the two private seams (_probe_children / _ppid_map) and
# runs with interval=0, so the POLL LOOP itself has ZERO real timing -- the
# same task-2819 precedent as the wait_for_marker_stable tests above.
# Fixing a flake with a flake is the failure mode to avoid, and the 3689
# steward failed to reproduce this one in 19 of 19 attempts (including
# synthetic pressure at load 100 / ~8000 processes), so a load-based test
# would be non-deterministic in BOTH directions.  The efficiency property
# that actually prevents sampling-rate collapse is therefore pinned
# STRUCTURALLY, by call count, which is exact at any load.
#
# Deliberate, narrow exceptions to "zero real subprocess" (task 4312,
# joining test_read_direct_children_sees_a_real_fork_including_off_main_thread
# below): the two timeout-diagnostic tests spawn a real `proc` because the
# diagnostic message's CONTENT -- the leader's actual returncode and, where
# applicable, its stderr tail -- is exactly what is under test.  The poll
# loop each one drives still runs zero ticks (timeout=0, interval=0), so
# timing stays zero even there.
# ---------------------------------------------------------------------------


def test_wait_subtree_live_gates_the_proc_walk_on_a_cheap_probe():
    """The expensive full /proc rescan runs only once the cheap probe goes positive.

    This is the load-bearing efficiency property of the discovery poll.
    Measured on this box: ``read_ppid_map()`` costs 49.67 ms at 917 live
    processes, while one ``/proc/<pid>/task/<tid>/children`` read costs
    0.0183 ms -- ~2700x.  Against the 50 ms poll interval the full rescan
    already DOUBLES the effective sampling period at idle, and the poller
    burns ~917 file reads per tick competing for CPU with the very leader it
    is waiting to see fork; under a full-suite storm (~8000 procs) the scan
    is ~400ms+ and the sampling period collapses by ~10x.  So the cost is
    worst exactly when the test needs the samples most.

    The call-count assertion below pins "the sampling rate cannot collapse
    under a large process table" as a STRUCTURAL property rather than a
    timing hope: 26 ticks may cost at most ONE expensive walk.
    """
    probe_calls = [0]
    ppid_map_calls = [0]

    def fake_probe(_pid):
        # 25 cheap negatives (leader live, hasn't forked yet), then a child.
        probe_calls[0] += 1
        return {5678} if probe_calls[0] > 25 else set()

    def fake_ppid_map():
        ppid_map_calls[0] += 1
        return {5678: 1234}

    result = wait_subtree_live(
        1234, interval=0, _probe_children=fake_probe, _ppid_map=fake_ppid_map,
    )

    assert result == {5678}, (
        f'expected the production walker\'s descendant set {{5678}}; got {result}'
    )
    assert probe_calls[0] == 26, (
        f'expected the cheap probe on every one of the 26 ticks; '
        f'got {probe_calls[0]} calls'
    )
    assert ppid_map_calls[0] <= 1, (
        f'the expensive read_ppid_map() walk ran {ppid_map_calls[0]} times across '
        f'26 ticks -- it must run ONLY on the tick the cheap probe goes positive, '
        f'or the sampling rate collapses exactly when the process table is largest'
    )


def test_wait_subtree_live_skips_a_vanished_transient_and_returns_the_durable_set():
    """A transient child that exits between probe and walk is skipped, not returned.

    The first child the CLI forks for a not-yet-materialised worktree is a
    SHORT-LIVED ``git`` subprocess (``git worktree add`` / ``git reset`` /
    ``git clean`` inside ``acquire_host_verify_worktree``), not the eventual
    build shell.  Scripted here deterministically: the probe sees pid 111 on
    tick 1, nothing on ticks 2-3, then the durable 300s sleeper (222) from
    tick 4; the paired walk comes back EMPTY on its first call because 111
    had already exited by the time the rescan ran.

    The helper must keep polling rather than hand back that empty set.  Two
    reasons: it makes the return contract "a NON-EMPTY descendant set"
    total, and it biases what the rows observe toward the durable sleeper --
    a transient that self-exits could otherwise let the ``wait_subtree_gone``
    assertion each row makes next pass without the watchdog having killed
    anything.
    """
    probe_calls = [0]
    ppid_map_calls = [0]

    def fake_probe(_pid):
        probe_calls[0] += 1
        if probe_calls[0] == 1:
            return {111}  # transient `git worktree add`
        if probe_calls[0] <= 3:
            return set()  # transient has exited; sleeper not forked yet
        return {222}  # the durable 300s sleeper

    def fake_ppid_map():
        ppid_map_calls[0] += 1
        # First walk races the transient's exit and sees nothing.
        return {} if ppid_map_calls[0] == 1 else {222: 1234}

    result = wait_subtree_live(
        1234, interval=0, _probe_children=fake_probe, _ppid_map=fake_ppid_map,
    )

    assert result == {222}, (
        f'expected the durable sleeper {{222}} -- never the empty set a walk '
        f'racing the transient git child returns; got {result}'
    )


def test_wait_subtree_live_falls_back_to_the_full_walk_when_children_unreadable():
    """A probe that returns None degrades to today's exact per-tick full walk.

    ``/proc/<pid>/task/<tid>/children`` requires CONFIG_PROC_CHILDREN, so the
    probe is TRI-STATE and its third state is load-bearing:

    * ``set()``  -- leader live, no children yet (cheap negative: skip the walk)
    * ``{...}``  -- worth confirming with the production walker
    * ``None``   -- CANNOT probe (no ``children`` file, or the ``/proc/<pid>``
      entry is gone)

    ``None`` must fall through to the full ``collect_descendants(pgid,
    read_ppid_map())`` walk for that tick.  Conflating it with the cheap
    negative would make the helper spin until timeout on a kernel where the
    probe can never go positive -- trading a flake for a hard, universal
    failure on that platform.

    The second half pins the vanishing-pid case that makes ``None``
    reachable at all on a normal kernel: the leader can exit mid-poll, and
    that must yield ``None``, never turn this helper's timeout diagnostic
    into an unhandled OSError.  A pid above ``pid_max`` cannot exist, so it
    stands in deterministically with no process to spawn or reap.
    """
    ppid_map_calls = [0]

    def fake_probe(_pid):
        return None  # kernel without CONFIG_PROC_CHILDREN

    def fake_ppid_map():
        ppid_map_calls[0] += 1
        return {5678: 1234} if ppid_map_calls[0] > 3 else {}

    result = wait_subtree_live(
        1234, interval=0, _probe_children=fake_probe, _ppid_map=fake_ppid_map,
    )

    assert result == {5678}, (
        f'a None (unprobeable) tick must fall back to the full walk, exactly as '
        f'this helper behaved before task 4014; got {result}'
    )

    pid_max = int(Path('/proc/sys/kernel/pid_max').read_text().strip())
    assert _read_direct_children(pid_max + 1) is None, (
        'a pid that does not exist must probe as None (cannot probe), not raise -- '
        'a leader exiting mid-poll must not turn the timeout path into an OSError'
    )


def test_wait_subtree_live_timeout_reports_leader_returncode_and_stderr_tail():
    """The timeout diagnostic names the LEADER's real rc and stderr tail (291a75a919).

    291a75a919 ("De-flake alpha: make wait_subtree_live's timeout path
    self-diagnosing") added the ``proc``/``proc_label`` reporting in the
    timeout branch above (:func:`wait_subtree_live`, ~:582-603) so a
    watchdog self-kill (rc == 1, no ``Error:`` line), an exception exit
    (rc == 1 WITH an ``Error:`` line), and a merely slow leader (rc is
    None) are distinguishable from the failure message alone -- exactly the
    signal an incident needs. It shipped with no test (task 4312).

    Deterministic per the task-4014 precedent above: ``_probe_children``
    always reports no children and ``timeout=0`` so the poll body never
    executes a single tick -- the timeout branch fires immediately, with
    zero real timing.  What IS real is *proc*: an actually-spawned,
    already-``wait()``-ed subprocess with a KNOWN returncode (3) and a
    KNOWN stderr payload (``SENTINEL-XYZ``), so this pins the diagnostic's
    CONTENT rather than merely the presence of the words "rc"/"stderr" --
    the vacuous-guard trap this repo has hit before: a message reporting
    nothing (e.g. ``rc=None; stderr tail:\n``) would also contain those
    labels.
    """
    proc = subprocess.Popen(
        [sys.executable, '-c', 'import sys; sys.stderr.write("SENTINEL-XYZ\\n"); sys.exit(3)'],
        stderr=subprocess.PIPE,
    )
    try:
        proc.wait()
        assert proc.returncode == 3  # sanity: the known rc actually landed

        with pytest.raises(AssertionError) as exc_info:
            wait_subtree_live(
                1234,
                proc=proc,
                proc_label='leader',
                timeout=0,
                interval=0,
                _probe_children=lambda _pid: set(),
                _ppid_map=lambda: {},
            )

        message = str(exc_info.value)
        assert 'leader rc=3' in message, (
            f'expected the leader\'s actual returncode (3), reported under its '
            f'proc_label, in the timeout message; got: {message!r}'
        )
        assert 'SENTINEL-XYZ' in message, (
            f'expected the leader\'s actual stderr tail content in the timeout '
            f'message; got: {message!r}'
        )
    finally:
        proc.stderr.close()


def test_wait_subtree_live_timeout_reports_rc_none_and_omits_stderr_for_a_live_leader():
    """The rc=None ("merely slow leader") taxonomy branch and its anti-hang guard (291a75a919).

    Sibling of
    :func:`test_wait_subtree_live_timeout_reports_leader_returncode_and_stderr_tail`
    above, covering the two properties that rc=3 case cannot reach (task 4312
    review):

    * the ``rc is None`` taxonomy branch itself -- a leader that is merely
      slow (still running when the poll gives up) is the case an incident is
      most likely to actually hit, and it was entirely unpinned.
    * the load-bearing guard :func:`wait_subtree_live`'s docstring calls out:
      stderr is read ONLY when ``proc.poll()`` is not ``None``, precisely
      because a LIVE leader's pipe write end can still be held open by an
      inherited-fd helper it spawned, and a buffered read on that pipe would
      block until EOF -- hanging this helper, and the rest of the suite,
      rather than raising.  A regression that swapped the guarded
      ``select()``-bounded raw read for an unconditional ``proc.stderr.read()``
      would still pass every other test in this module while reintroducing
      exactly that hang.

    *proc* here is a real, still-running subprocess (``time.sleep(300)``, well
    outside this test's own runtime) so ``proc.poll()`` is genuinely ``None``
    -- checking that costs one non-blocking ``waitpid(WNOHANG)`` syscall, so
    this stays zero real TIMING even though *proc* itself is real, same as
    its rc=3 sibling.  ``stderr=PIPE`` is set (mirroring real call sites) so a
    guard regression would hang on the open write end rather than raise
    ``AttributeError`` on a ``None`` pipe, which would make this test pass for
    the wrong reason.
    """
    proc = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(300)'],
        stderr=subprocess.PIPE,
    )
    try:
        with pytest.raises(AssertionError) as exc_info:
            wait_subtree_live(
                1234,
                proc=proc,
                proc_label='leader',
                timeout=0,
                interval=0,
                _probe_children=lambda _pid: set(),
                _ppid_map=lambda: {},
            )

        message = str(exc_info.value)
        assert 'leader rc=None' in message, (
            f'expected the merely-slow-leader taxonomy branch (rc=None, '
            f'reported under its proc_label) in the timeout message; got: '
            f'{message!r}'
        )
        assert 'stderr tail' not in message, (
            f'stderr must be read ONLY when proc.poll() is not None -- seeing '
            f'"stderr tail" here means the anti-hang poll()-gate regressed to '
            f'an unconditional read, which would hang on a live leader whose '
            f'pipe write end is still open; got: {message!r}'
        )
    finally:
        proc.kill()
        proc.wait()
        proc.stderr.close()


#: A child that stays alive but is never waited on -- killed by the test that
#: spawns it.  ``sys.executable`` rather than ``sleep`` so nothing depends on
#: PATH; the probe only needs the fork, not a finished exec.
_DURABLE_CHILD_ARGV = [sys.executable, '-c', 'import time; time.sleep(30)']


def test_read_direct_children_sees_a_real_fork_including_off_main_thread():
    """The real probe's POSITIVE path, against the real kernel interface.

    The three tests above all inject ``_probe_children``, so they pin
    :func:`wait_subtree_live`'s gate LOGIC and never execute
    :func:`_read_direct_children` itself; the only unmocked assertion on it is
    the negative (``None``) case immediately above.  A regression in the
    positive path -- a wrong ``/proc`` path segment, iterating only the main
    tid, a broken ``int(token)`` parse -- would return ``set()`` for a leader
    that HAS forked.  Every deterministic test in this module would stay green
    while all five real rows failed with "no descendant appeared within Ns":
    the exact flake task 4014 exists to remove, made permanent.

    Two halves, both with ZERO real timing -- no sleeps, no polling.
    ``Popen`` has already forked when it returns and the kernel lists the child
    from that moment, so there is nothing to wait for (the ``Event`` waits in
    (b) are handoff barriers, not timing assertions).

    (a) A child forked from THIS thread must be listed.  Pins path
        construction and token parsing.

    (b) A child forked from a SECONDARY thread, probed while that thread is
        still alive, must also be listed.  This is the half that pins the
        per-thread ``task/*/`` iteration the helper's docstring justifies:
        ``children`` is a PER-THREAD file, so a bare
        ``/proc/<pid>/task/<pid>/children`` read would MISS this child.  The
        forking thread is held alive until after the probe precisely because a
        thread that EXITS has its children re-parented onto the group leader,
        which would let a main-tid-only implementation pass and make this half
        vacuous.
    """
    child = subprocess.Popen(_DURABLE_CHILD_ARGV)
    try:
        probed = _read_direct_children(os.getpid())
        assert probed is not None, (
            'probing this live test process must not report "cannot probe" -- '
            'either the /proc path is built wrong or this kernel lacks '
            'CONFIG_PROC_CHILDREN (in which case every row silently runs on the '
            'full-walk fallback and task 4014 buys nothing)'
        )
        assert child.pid in probed, (
            f'just-forked direct child {child.pid} is missing from the probe '
            f'{sorted(probed)} -- a probe that cannot see a fork returns the cheap '
            f'NEGATIVE forever, so wait_subtree_live never spends the confirming '
            f'walk and every real row fails with "no descendant appeared"'
        )
    finally:
        child.kill()
        child.wait()

    forked: dict[str, subprocess.Popen] = {}
    spawned = threading.Event()
    release = threading.Event()

    def fork_off_main_thread() -> None:
        forked['proc'] = subprocess.Popen(_DURABLE_CHILD_ARGV)
        spawned.set()
        release.wait(timeout=30.0)  # stay alive so the child stays MINE

    thread = threading.Thread(target=fork_off_main_thread, daemon=True)
    thread.start()
    try:
        assert spawned.wait(timeout=30.0), 'the helper thread never forked its child'
        off_main = forked['proc']
        probed = _read_direct_children(os.getpid())
        assert probed is not None and off_main.pid in probed, (
            f'child {off_main.pid}, forked from a still-live secondary thread, is '
            f'missing from the probe {probed if probed is None else sorted(probed)} '
            f'-- children is a PER-THREAD file, so this is what a regression to a '
            f'single /proc/<pid>/task/<pid>/children read looks like'
        )
    finally:
        release.set()
        thread.join(timeout=30.0)
        off_main = forked.get('proc')
        if off_main is not None:
            off_main.kill()
            off_main.wait()


# ---------------------------------------------------------------------------
# Task 4092 -- deterministic-ish unit coverage for kill_holder_tree, the
# shared teardown helper that reaps a spawn_verify_merge holder AND every
# descendant it forked (including start_new_session escapes -- verify.py's
# `_run_cmd` runs every build/test command via
# ``create_subprocess_shell(..., start_new_session=True)``, so a killed
# leader alone leaves its build orphaned, reparented to init, for up to its
# full sleep duration).
#
# Every test here spawns a lightweight Popen stand-in (never the real CLI,
# never a git repo) -- the property under test is purely "does a
# session-escaped descendant get reaped" / "is the caller's own process
# group ever signalled", both of which this reproduces exactly.  See the
# plan's design_decisions for why a real verify-merge holder is not used
# here: it would drag in _setup_verify_repo's git work, config YAML, the
# ~9s CLI import and the load-scaled discovery waits, fixing a flake with a
# flake (the module's own task-2819 banner above warns against exactly
# this).  The two real-CLI call sites remain covered end-to-end by the Row
# 5 / lane-lock tests below plus their post-run pgrep check.
# ---------------------------------------------------------------------------


def _pid_gone(pid: int) -> bool:
    """Best-effort liveness probe: True when *pid* no longer refers to a live process.

    ``os.kill(pid, 0)`` sends no signal, only checks existence/permission.
    ``PermissionError`` means the pid exists but isn't ours -- that is NOT
    "gone", so it returns False rather than masking a real survivor.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def test_kill_holder_tree_reaps_a_session_escaped_grandchild():
    """kill_holder_tree reaps BOTH the leader and a start_new_session grandchild.

    Reproduces the exact escape verify.py's ``_run_cmd`` produces
    (``create_subprocess_shell(cmd, start_new_session=True)``) with a
    lightweight stand-in leader: ``subprocess.Popen`` running a `-c` snippet
    that forks a ``sleep`` grandchild via ``start_new_session=True`` and
    then blocks itself for a long time, mirroring ``sleeper_spec``'s shape
    without needing the real CLI or a throwaway git repo.

    The sleep duration is derived from this test process's own pid so it
    cannot collide with an unrelated ``sleep`` on a shared dev box -- in
    particular with this very module's own ``sleeper_spec`` 300s sleeper.

    Asserts BOTH the leader is reaped and every captured grandchild pid is
    actually gone (not just "signalled") -- and is self-cleaning (a finally
    that SIGKILLs any surviving captured pid) so a failing/RED run of this
    test never itself leaks the orphan it exists to pin.
    """
    sleep_secs = f'271.{os.getpid() % 1000:03d}'
    leader = subprocess.Popen([
        sys.executable, '-c',
        f'import subprocess, time\n'
        f'subprocess.Popen(["sleep", "{sleep_secs}"], start_new_session=True)\n'
        f'time.sleep(300)\n',
    ])
    grandchildren: set[int] = set()
    try:
        discovery_deadline = time.monotonic() + 10.0
        while time.monotonic() < discovery_deadline:
            grandchildren = collect_descendants(leader.pid, read_ppid_map())
            if grandchildren:
                break
            time.sleep(0.05)
        assert grandchildren, (
            f'no descendant of leader pid={leader.pid} appeared within 10s -- '
            f'harness bug (the stand-in leader never forked its sleep '
            f'grandchild), not a seam defect under test'
        )

        kill_holder_tree(leader, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS)

        assert leader.poll() is not None, (
            'kill_holder_tree must reap the leader -- poll() is still None '
            'after the call returned'
        )

        gone_deadline = time.monotonic() + 5.0
        survivors = set(grandchildren)
        while survivors and time.monotonic() < gone_deadline:
            survivors = {pid for pid in survivors if not _pid_gone(pid)}
            if survivors:
                time.sleep(0.05)
        assert not survivors, (
            f'kill_holder_tree left session-escaped descendant(s) alive: '
            f'{sorted(survivors)} -- the exact orphan task 4092 exists to fix'
        )
    finally:
        if leader.poll() is None:
            leader.kill()
            leader.wait(timeout=5)
        for pid in grandchildren:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)


def test_kill_holder_tree_never_signals_the_callers_own_process_group():
    """kill_holder_tree must NEVER killpg the CALLER's own process group.

    This is the guard that makes this task's literally-prescribed
    ``os.killpg(os.getpgid(holder.pid), SIGKILL)`` one-liner unwritable.
    The lane-lock holder
    (``test_live_verify_merge_holds_lane_lock_real_subprocess``) is spawned
    WITHOUT ``--request-id``, so ``cli.py`` never calls ``os.setsid`` on it
    (:func:`~orchestrator.verify_cancel.start_own_process_group` is gated on
    ``if request_id is not None:``) and the holder stays in THIS process's
    (pytest's) own group -- ``os.getpgid(holder.pid) == os.getpgid(0)``. A
    naive killpg backstop there would SIGKILL the pytest worker running
    this very test.

    Reproduces that exact shape: a leader spawned via plain
    ``subprocess.Popen`` with no ``start_new_session``.  The precondition
    assertion proves this test really exercises the shared-group case
    rather than silently testing nothing.

    kill_holder_tree's guarded killpg backstop (see its own docstring)
    fires only when the holder is provably its own process-group leader
    (``pgid == proc.pid``) AND that group is provably not the caller's
    own (``pgid != os.getpgid(0)``). A holder that never setsid'd -- like
    this one -- fails the first condition, so the backstop must stay
    silent. This test pins that guard directly against the SHIPPED
    (killpg-aware) helper: it fails the moment a future edit drops either
    condition, or otherwise makes the backstop unconditional.
    """
    leader = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
    try:
        assert os.getpgid(leader.pid) == os.getpgid(0), (
            'harness bug: a plain subprocess.Popen with no start_new_session '
            'must inherit the caller\'s own process group, or this test is '
            'not exercising the shared-group case it exists to guard'
        )

        killpg_calls: list[tuple[int, int]] = []

        def spy(pgid, sig):
            killpg_calls.append((pgid, sig))

        kill_holder_tree(
            leader, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS, _killpg=spy,
        )

        assert leader.poll() is not None, (
            'kill_holder_tree must still reap the leader even with the '
            'killpg backstop suppressed -- a guard satisfied by a helper '
            'that does nothing at all would be worthless'
        )

        own_pgid = os.getpgid(0)
        assert all(pgid != own_pgid for pgid, _sig in killpg_calls), (
            f"kill_holder_tree called killpg with the CALLER'S OWN process "
            f'group {own_pgid}: {killpg_calls} -- this would SIGKILL the '
            f'pytest worker itself'
        )
        assert killpg_calls == [], (
            f'kill_holder_tree must not call killpg at all when the holder '
            f"shares the caller's process group (never setsid'd): "
            f'{killpg_calls}'
        )
    finally:
        if leader.poll() is None:
            leader.kill()
            leader.wait(timeout=5)


def test_kill_holder_tree_killpg_backstop_fires_for_a_setsid_leader():
    """The guarded killpg backstop's POSITIVE path: it must actually fire.

    test_kill_holder_tree_never_signals_the_callers_own_process_group above
    only pins the NEGATIVE path -- killpg must stay silent when the holder
    never setsid'd. Nothing asserts that the backstop actually fires when
    the holder DID setsid, which is exactly the shape spawn_verify_merge
    produces when the CLI is launched with ``--request-id`` (cli.py:578-581
    gates ``start_own_process_group()`` on ``if request_id is not None:``).
    A regression that made the guard's admission condition unconditionally
    false -- e.g. inverting the ``pgid == proc.pid`` comparison, or
    dropping the branch entirely -- would leave both existing
    kill_holder_tree killpg unit tests green: exactly the "guard reporting
    PASS while guarding nothing" failure mode this module's anti-vacuity
    discipline exists to prevent (see
    test_every_scaled_discovery_wait_is_covered_by_its_test_timeout_mark's
    banner).

    Reproduces the setsid'd shape directly with ``start_new_session=True``,
    which is precisely what makes a Popen leader its own process-group
    leader (``os.getpgid(leader.pid) == leader.pid``). The two precondition
    assertions below prove BOTH halves of the backstop's admission
    condition actually hold before kill_holder_tree runs, so a pass here
    cannot be a vacuous accident of the harness's own process group.
    """
    leader = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(300)'],
        start_new_session=True,
    )
    try:
        leader_pgid = os.getpgid(leader.pid)
        assert leader_pgid == leader.pid, (
            'harness bug: start_new_session=True must make the leader its '
            'own process-group leader, or this test is not exercising the '
            "killpg backstop's admission condition it exists to guard"
        )
        assert leader_pgid != os.getpgid(0), (
            "harness bug: the setsid'd leader's group must differ from "
            "the caller's own, or this test is not exercising the killpg "
            "backstop's admission condition it exists to guard"
        )

        killpg_calls: list[tuple[int, int]] = []

        def spy(pgid, sig):
            killpg_calls.append((pgid, sig))

        kill_holder_tree(
            leader, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS, _killpg=spy,
        )

        assert leader.poll() is not None, (
            "kill_holder_tree must reap a setsid'd leader the same as any "
            'other'
        )
        assert killpg_calls == [(leader_pgid, signal.SIGKILL)], (
            f"kill_holder_tree must killpg a leader that provably setsid'd "
            f'into its own, non-caller process group -- got {killpg_calls}, '
            f'expected exactly one call with pgid={leader_pgid}'
        )
    finally:
        if leader.poll() is None:
            leader.kill()
            leader.wait(timeout=5)


def _wait_until_zombie(pid: int, *, timeout: float = 5.0) -> bool:
    """Poll ``/proc/<pid>/stat`` until *pid* is a zombie (state ``Z``).

    Deliberately does NOT use ``Popen.wait()``/``poll()``: those reap the
    process, which is precisely the state transition the caller here needs
    to NOT happen.  Reads the state field positionally from the tail after
    the last ``)`` so a comm containing spaces or parens cannot skew it.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            stat = Path(f'/proc/{pid}/stat').read_text()
        except OSError:
            return False  # already reaped by someone else, or gone
        tail = stat.rpartition(')')[2].split()
        if tail and tail[0] == 'Z':
            return True
        time.sleep(0.02)
    return False


def test_kill_holder_tree_does_not_reap_a_leader_that_exits_mid_teardown():
    """A leader that exits DURING teardown must not be reaped before the backstop.

    Closes the window between the entry-time ``proc.returncode is not
    None`` short circuit and the killpg backstop.  The two existing killpg
    unit tests only cover the endpoints -- "already reaped at entry"
    (``test_kill_holder_tree_is_safe_when_the_leader_already_exited``) and
    "alive throughout"
    (``test_kill_holder_tree_killpg_backstop_fires_for_a_setsid_leader``)
    -- and both stay GREEN with a ``poll()`` reintroduced at the leader
    kill, so neither can catch this.

    THE HAZARD.  ``Popen.poll()`` is not a passive read: it calls
    ``os.waitpid(pid, WNOHANG)``.  A ``poll()`` at the leader-kill step
    would therefore REAP a leader that exited since entry and FREE
    ``proc.pid`` -- after which the backstop's ``pgid == proc.pid`` test
    compares a pre-reap pgid against a freed pid, and fires
    ``killpg(pgid, SIGKILL)`` at a group the kernel is free to have
    reassigned to a stranger.  On a shared dev box or CI host that is an
    unrelated user's entire process group.

    THIS IS THE ROW 5 HOLDER'S DESIGNED BEHAVIOUR, not an exotic race: its
    ``finally`` calls ``heartbeat_holder.stop_heartbeats()`` immediately
    before ``kill_holder_tree``, and stopping heartbeats ARMS the CLI's
    stdin-watchdog self-kill, so the holder exiting on its own inside the
    teardown window is exactly what the surrounding code sets up.

    THE ASSERTION.  The ``_ppid_map_provider`` seam is used to land the
    leader's exit strictly inside the window: the provider SIGKILLs the
    leader and waits (via ``/proc``, never ``wait()``) until it is a
    ZOMBIE, so the leader is provably exited-but-unreaped by the time the
    helper reaches the leader kill.  The killpg spy then records
    ``proc.returncode`` AT CALL TIME.  Under the fixed helper that is
    ``None`` -- nothing in the helper reaped it, so the pid is still
    PINNED and ``pgid`` provably still denotes the holder's own group.
    Reintroduce the ``poll()`` guard and the spy sees a non-``None``
    returncode: the pid was freed before the backstop fired, and the test
    goes RED.
    """
    leader = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(300)'],
        start_new_session=True,
    )
    try:
        leader_pgid = os.getpgid(leader.pid)
        assert leader_pgid == leader.pid, (
            'harness bug: start_new_session=True must make the leader its '
            'own process-group leader, or this test is not exercising the '
            "killpg backstop's admission condition it exists to guard"
        )
        assert leader_pgid != os.getpgid(0), (
            "harness bug: the setsid'd leader's group must differ from the "
            "caller's own, or this test is not exercising the killpg "
            "backstop's admission condition it exists to guard"
        )
        assert leader.returncode is None, (
            'harness bug: the leader must be un-reaped at entry, or '
            "kill_holder_tree's already-reaped short circuit fires and this "
            'test exercises nothing'
        )

        became_zombie: list[bool] = []

        def provider_that_kills_the_leader():
            """Land the leader's exit strictly inside the teardown window."""
            os.kill(leader.pid, signal.SIGKILL)
            became_zombie.append(_wait_until_zombie(leader.pid))
            return {}  # no descendants; the pid-pinning property is under test

        killpg_calls: list[tuple[int, int, int | None]] = []

        def killpg_spy(pgid, sig):
            # Record the leader's REAPED-ness as the backstop sees it.
            killpg_calls.append((pgid, sig, leader.returncode))

        kill_holder_tree(
            leader,
            timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS,
            _ppid_map_provider=provider_that_kills_the_leader,
            _killpg=killpg_spy,
        )

        assert became_zombie == [True], (
            f'harness bug: the leader never became an unreaped zombie inside '
            f'the teardown window ({became_zombie}) -- this test is not '
            f'exercising the exit-mid-teardown race it exists to pin'
        )
        assert killpg_calls, (
            'kill_holder_tree must still fire its killpg backstop for a '
            "leader that setsid'd and then exited mid-teardown -- same-group "
            'stragglers still need reaping'
        )
        assert [(pgid, sig) for pgid, sig, _rc in killpg_calls] == [
            (leader_pgid, signal.SIGKILL)
        ], (
            f'kill_holder_tree must killpg exactly the holder\'s own group -- '
            f'got {killpg_calls}, expected pgid={leader_pgid}'
        )
        assert all(rc is None for _pgid, _sig, rc in killpg_calls), (
            f'kill_holder_tree REAPED the leader before firing its killpg '
            f'backstop (returncode at killpg time: '
            f'{[rc for _p, _s, rc in killpg_calls]}) -- proc.pid was FREED, '
            f'so pgid={leader_pgid} may already belong to a stranger. This is '
            f'the pid-recycling hazard the backstop is documented to avoid; '
            f'a poll()/wait() must not run before the backstop.'
        )
        assert leader.poll() is not None, (
            'kill_holder_tree must still reap the leader by the time it '
            'returns -- pinning the pid across the backstop is not a licence '
            'to leak a zombie'
        )
    finally:
        if leader.poll() is None:
            leader.kill()
            leader.wait(timeout=5)


def test_kill_holder_tree_is_safe_when_the_leader_already_exited():
    """An already-reaped leader means proc.pid is a FREE pid -- signal NOTHING.

    Four converted call sites reap the leader with ``wait()`` inside their
    ``try`` block BEFORE the ``finally`` runs ``kill_holder_tree``
    (``test_watchdog_timeout_env_override_fires_fast_without_heartbeat``,
    ``test_cancel_verify_tree_kills_under_live_watchdog``,
    ``test_ssh_dropped_mid_build_tree_killed_via_eof_dispatcher_alive``,
    ``test_heartbeat_starved_hard_partition_tree_killed_via_timeout``) --
    so on their GREEN path ``proc.pid`` no longer refers to the holder at
    all by the time this helper runs.
    ``os.getpgid(proc.pid)`` and ``collect_descendants(proc.pid, ...)``
    would then describe whatever process happens to OWN that pid now, and
    every descendant of that stranger would be SIGKILLed (and killpg'd by
    the backstop, if it happened to be its own group leader).  This is not
    theoretical: pid recycling is observed on this fleet
    (verify_cancel.py:313-315 -- pid_max=4194304, and the laptop's own pid
    counter demonstrably wrapped on 2026-08-11).

    Three cases; (a) and (b) share ONE already-reaped leader:

    (a) REAL-PATH no-signal contract: every seam is spied rather than
        faked, and NONE may fire -- in particular the ppid-map provider
        must never be CALLED AT ALL, which is what proves the short
        circuit lands before the snapshot phase rather than merely seeing
        a map with no descendants in it.  Also pins that the early-return
        path still closes stdin, since the lane-lock site
        (``test_live_verify_merge_holds_lane_lock_real_subprocess``)
        delegates its stdin cleanup to this helper.

    (b) PID-REUSE MODEL -- the case that makes this RED non-vacuous.
        Against a reaped leader the REAL ``/proc`` map almost always has
        no descendants, so asserting against it would pass by luck even
        without the short circuit.  A SYNTHETIC ppid map gives the reaped
        (free) pid fabricated children, modeling a stranger process
        reusing that pid -- ``collect_descendants`` accepts any dict and
        is cycle-safe (verify_cancel.py:197), and every pid is fake with
        ``_kill`` spied, so this can never signal a real process even if
        the guard under test is missing.

    (c) the OSError-degradation coverage (a)/(b) can no longer reach once
        the short circuit exists, since it never gets far enough to call
        the ppid-map provider -- moved onto a LIVE leader so the
        ``except OSError: ppid_map = {}`` branch stays under test.
    """
    # (a) + (b): one already-reaped leader, shared.
    leader = subprocess.Popen([sys.executable, '-c', 'pass'], stdin=subprocess.PIPE)
    leader.wait(timeout=10)

    # (a) REAL-PATH: every seam spied; none may fire for an already-reaped
    # leader, and the ppid-map provider must not even be CALLED.
    kill_calls_a: list[tuple[int, int]] = []
    killpg_calls_a: list[tuple[int, int]] = []
    ppid_map_call_count = [0]

    def kill_spy_a(pid, sig):
        kill_calls_a.append((pid, sig))

    def killpg_spy_a(pgid, sig):
        killpg_calls_a.append((pgid, sig))

    def ppid_map_spy_a():
        ppid_map_call_count[0] += 1
        return {}

    kill_holder_tree(
        leader,
        timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS,
        _ppid_map_provider=ppid_map_spy_a,
        _kill=kill_spy_a,
        _killpg=killpg_spy_a,
    )

    assert kill_calls_a == [], (
        f'kill_holder_tree signalled pid(s) {kill_calls_a} for an '
        f'already-reaped leader -- proc.pid is FREE, so this can only '
        f'reach an unrelated stranger process'
    )
    assert killpg_calls_a == [], (
        f'kill_holder_tree called killpg{killpg_calls_a} for an '
        f'already-reaped leader'
    )
    assert ppid_map_call_count[0] == 0, (
        'kill_holder_tree consulted the ppid-map provider for an '
        'already-reaped leader -- the short circuit must land BEFORE the '
        'snapshot phase, not merely happen to see a map with no '
        'descendants in it'
    )
    assert leader.stdin is not None and leader.stdin.closed, (
        'kill_holder_tree must still close stdin on the already-reaped '
        'early-return path -- the lane-lock site '
        '(test_live_verify_merge_holds_lane_lock_real_subprocess) '
        'delegates its stdin cleanup to this helper'
    )

    # (b) PID-REUSE MODEL: a synthetic ppid map gives the reaped (free)
    # leader pid fabricated children, modeling a stranger process reusing
    # that pid.  Every pid below is fake and _kill/_killpg stay spied, so
    # this can never signal a real process.
    synthetic_ppid_map = {
        leader.pid: 1,
        99990001: leader.pid,
        99990002: 99990001,
    }
    kill_calls_b: list[tuple[int, int]] = []
    killpg_calls_b: list[tuple[int, int]] = []

    def kill_spy_b(pid, sig):
        kill_calls_b.append((pid, sig))

    def killpg_spy_b(pgid, sig):
        killpg_calls_b.append((pgid, sig))

    kill_holder_tree(
        leader,
        timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS,
        _ppid_map_provider=lambda: synthetic_ppid_map,
        _kill=kill_spy_b,
        _killpg=killpg_spy_b,
    )

    assert kill_calls_b == [], (
        f'kill_holder_tree signalled fabricated pid(s) {kill_calls_b}, '
        f"reachable only through a stranger's reuse of the reaped "
        f"leader's pid {leader.pid} -- an already-reaped leader must "
        f'signal NOTHING'
    )
    assert killpg_calls_b == [], (
        f'kill_holder_tree called killpg{killpg_calls_b} against a '
        f"stranger's fabricated process group for an already-reaped "
        f'leader'
    )

    # (c) OSError-degradation coverage, moved onto a LIVE leader: once (a)
    # short-circuits before ever calling the ppid-map provider, a reaped
    # leader can no longer exercise the `except OSError: ppid_map = {}`
    # guard.  _kill is deliberately left as the REAL os.kill here so the
    # leader is genuinely SIGKILLed and reaped fast, well inside the 5.0s
    # ceiling -- spying it would leave the process alive for the whole
    # timeout and emit the "did not exit" warning instead.
    def raising_ppid_map_provider():
        raise OSError('simulated /proc read racing process exit')

    live_leader = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(300)'],
    )
    try:
        kill_holder_tree(
            live_leader,
            timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS,
            _ppid_map_provider=raising_ppid_map_provider,
        )
        assert live_leader.poll() is not None, (
            'kill_holder_tree must still reap a LIVE leader when the '
            'ppid-map provider raises OSError (a /proc read racing '
            'process exit)'
        )
    finally:
        if live_leader.poll() is None:
            live_leader.kill()
            live_leader.wait(timeout=5)


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
# Task 4474 -- watchdog twin of the flock-gate timing bootstrap above.  Same
# defect class, same remedy: task 4248's verify attempt-1 observed
# test_watchdog_timeout_env_override_fires_fast_without_heartbeat's OUTER
# wall-clock fire_delay (pgid-file appearance to proc.wait() return) at
# 19.75s under a full-suite xdist storm -- past even the 15.0s production
# un-wired-override window, so no widened ceiling could both accommodate
# that and still discriminate a wired override from an un-wired one.  The
# outer measurement was never purely the watchdog's own window: it also
# carries scheduler latency between the watchdog thread actually firing and
# this PARENT process observing proc.wait() return, which balloons under
# CPU contention exactly like the import cost the old pgid-file wait was
# already compensating for (task 2921's own inline comment claiming
# otherwise is what the 19.75s observation falsifies).
#
# So, exactly as FLOCK_GATE_TIMING_BOOTSTRAP does for
# orchestrator.cli.acquire_merge_verify_flock, WATCHDOG_GATE_TIMING_BOOTSTRAP
# wraps the production callables cli.py invokes on this seam.  It patches
# TWO module-level names, because the quantity under test --
# ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS *and* ORCH_WATCHDOG_KILL_GRACE_SECS
# together -- spans two calls:
#
# * orchestrator.cli.start_stdin_watchdog is wrapped ONLY to capture t0
#   (time.monotonic(), armed the instant before the watchdog thread starts
#   its blocking select loop); its ``fire`` callback is passed through
#   untouched, so a caller that ever stopped passing ``fire=`` explicitly
#   would still get the real production default instead of a silently
#   disabled tree-kill (amendment-pass fix: an earlier draft wrapped
#   ``fire`` itself and dropped the kill in that branch).
# * orchestrator.cli.fire_watchdog_kill is wrapped to intercept TWO of its
#   injectable arguments:
#     - ``sleep``: replaced with a stopwatch wrapper that records how long
#       the SIGTERM -> SIGKILL grace pause ACTUALLY took, so
#       ORCH_WATCHDOG_KILL_GRACE_SECS is a MEASURED term rather than an
#       assumed one.  cli.py:773 calls
#       ``fire_watchdog_kill(pgid, grace_secs=grace_secs)`` with no
#       ``sleep=``, so this injection displaces no production argument (and
#       it falls back to the caller's own ``sleep`` if one is ever passed,
#       exactly as the ``exit_fn`` interception does).
#     - ``exit_fn``: the wrapped exit_fn prints the marker line to stderr
#       (flushed explicitly, since the real exit_fn is os._exit, which
#       skips normal buffer flushing), then delegates to the real exit_fn.
#
#   The reported fire_delay is RECONSTRUCTED from two measured terms rather
#   than read off a single clock:
#
#       fire_delay = (t_fire_entry - t0) + observed_grace_sleep
#
#   where t_fire_entry is monotonic() at ENTRY to the wrapped
#   fire_watchdog_kill -- which genuinely is the fire-callback entry,
#   because cli.py's ``_on_watchdog_fire`` (cli.py:771-773) resolves the
#   ``fire_watchdog_kill`` module global at CALL time and therefore lands in
#   the already-wrapped callable.  No wrapper around ``fire`` itself is
#   needed, so the warning above against wrapping ``fire`` stays honoured.
#
# THIRD CORRECTION (this pass -- task 4559 verify).  Commit aa3095175c
# ("amend: measure the full watchdog arm-to-kill window, not just
# fire-callback entry") moved the stop point from fire-callback ENTRY to
# post-kill inside exit_fn, precisely so ORCH_WATCHDOG_KILL_GRACE_SECS came
# under the ceiling assertion -- right in intent, but over-inclusive in
# extent.  Stopping the clock in exit_fn also swept in fire_watchdog_kill's
# TWO ppid_map_provider() calls (verify_cancel.py:774 before the SIGTERM
# sweep and verify_cancel.py:782 after the grace sleep; each is a full
# /proc iterdir plus one read_text() per pid -- read_ppid_map,
# verify_cancel.py:165).  Neither walk is an env-override quantity under
# test, and both scale with machine load: measured across a ~300x load
# swing the two walks moved 0.56s -> 2.60s while the heartbeat wait held at
# 0.513-0.530s and the grace sleep at 0.200-0.217s, carrying the total to
# 5.18s -- past the 4.0s ceiling -- on a tree where BOTH overrides were
# correctly wired.  That is the same load-sensitive-proxy defect as the
# 19.75s outer wall clock, merely relocated inside the child.
#
# The reconstruction above is the third position, and it keeps what each
# earlier one had to trade away: the heartbeat override stays asserted via
# (t_fire_entry - t0), as in the pre-amendment form; the kill-grace
# override stays asserted via observed_grace_sleep, as the amendment
# intended; and the /proc walks -- which appear in neither override -- are
# excluded from both terms.  The ceiling stays at 4.0s deliberately: it is
# the DISCRIMINANT (the nearest broken shape -- kill-grace regressed to its
# 5.0s production default with the heartbeat still wired -- lands at ~5.5s),
# so the MEASUREMENT was the thing to fix, never the threshold.
#
# Because observed_grace_sleep is measured rather than assumed, a silent
# failure of the ``sleep=`` injection would zero that term and quietly
# retire the kill-grace axis from the ceiling.  The marker line therefore
# carries ``grace=`` alongside ``fire_delay=``, and the ceiling test
# asserts the grace pause was actually OBSERVED -- see
# WATCHDOG_KILL_GRACE_OVERRIDE_SECS.  This class of test has now been
# mis-measured three times (2921's outer wall clock -> 4248 falsified it at
# 19.75s -> 4474 moved the clock inside the child but left the /proc walks
# in); that non-vacuity guard is what stops a fourth.
#
# The reported fire_delay is therefore immune both to outer scheduler noise
# and to the /proc-walk cost that machine load makes unbounded.
# ---------------------------------------------------------------------------

#: Marker token the instrumented bootstrap prints, one line per watchdog fire.
WATCHDOG_GATE_MARKER = '__WATCHDOG_GATE__'

#: printf-style line format shared between the bootstrap's child-side print
#: (embedded below via f-string substitution) and the parser self-test's
#: synthetic ``emitted`` fixture, so a field rename is a rename in both
#: places rather than two hand-written copies that can silently drift apart
#: (amendment-pass fix -- see test_watchdog_gate_timing_bootstrap_parses_
#: its_own_marker_lines).
WATCHDOG_GATE_LINE_FMT = f'{WATCHDOG_GATE_MARKER} fire_delay=%.4f grace=%.4f'

WATCHDOG_GATE_TIMING_BOOTSTRAP = f"""
import os, sys, time
import orchestrator.cli as _cli
_real_start_stdin_watchdog = _cli.start_stdin_watchdog
_real_fire_watchdog_kill = _cli.fire_watchdog_kill
_t0 = None


def _timed_start_stdin_watchdog(pgid, **kwargs):
    global _t0
    _t0 = time.monotonic()
    return _real_start_stdin_watchdog(pgid, **kwargs)


def _timed_fire_watchdog_kill(pgid, *, exit_fn=None, sleep=None, **kwargs):
    # Entry to THIS wrapper is the fire-callback entry: cli.py's
    # _on_watchdog_fire resolves the fire_watchdog_kill module global at call
    # time, so it lands here.  Snapshot before delegating, so the two
    # load-sensitive /proc walks inside the real body stay OUT of the
    # heartbeat term.
    _t_fire_entry = time.monotonic()
    _real_exit = exit_fn if exit_fn is not None else os._exit
    _real_sleep = sleep if sleep is not None else time.sleep
    _observed_grace = [0.0]

    def _timed_sleep(secs):
        _s0 = time.monotonic()
        try:
            return _real_sleep(secs)
        finally:
            _observed_grace[0] += time.monotonic() - _s0

    def _timed_exit(code):
        _elapsed = (_t_fire_entry - _t0) + _observed_grace[0]
        print(
            '{WATCHDOG_GATE_LINE_FMT}' % (_elapsed, _observed_grace[0]),
            file=sys.stderr,
        )
        sys.stderr.flush()
        _real_exit(code)

    return _real_fire_watchdog_kill(
        pgid, exit_fn=_timed_exit, sleep=_timed_sleep, **kwargs
    )


_cli.start_stdin_watchdog = _timed_start_stdin_watchdog
_cli.fire_watchdog_kill = _timed_fire_watchdog_kill
_cli.main()
"""


class WatchdogGateFire(NamedTuple):
    """One instrumented ``fire_watchdog_kill`` call inside the child.

    ``fire_delay_secs`` is the reconstructed
    ``(t_fire_entry - t0) + grace_secs`` window -- the heartbeat wait plus
    the observed SIGTERM->SIGKILL grace pause, deliberately EXCLUDING
    ``fire_watchdog_kill``'s two load-sensitive /proc walks (see the banner
    above).  ``grace_secs`` is broken out separately so the ceiling test can
    assert that term was actually observed rather than silently defaulted to
    zero by a failed ``sleep=`` injection.
    """

    fire_delay_secs: float
    grace_secs: float


_WATCHDOG_GATE_RE = re.compile(
    re.escape(WATCHDOG_GATE_MARKER)
    + r' fire_delay=(?P<fire_delay>\S+) grace=(?P<grace>\S+)'
)


def parse_watchdog_gate_fire_delays(stderr: str) -> list[WatchdogGateFire]:
    """Parse the :data:`WATCHDOG_GATE_TIMING_BOOTSTRAP` lines out of a child's stderr."""
    return [
        WatchdogGateFire(
            fire_delay_secs=float(m.group('fire_delay')),
            grace_secs=float(m.group('grace')),
        )
        for m in _WATCHDOG_GATE_RE.finditer(stderr)
    ]


def test_watchdog_gate_timing_bootstrap_parses_its_own_marker_lines():
    """parse_watchdog_gate_fire_delays round-trips the bootstrap's marker format.

    Mirrors test_flock_gate_timing_bootstrap_parses_its_own_marker_lines: the
    bootstrap emits its lines from inside a spawned child, so a drift between
    what it prints and what the parser accepts would silently return zero
    observations -- which would make the ceiling assertion in
    test_watchdog_timeout_env_override_fires_fast_without_heartbeat vacuous
    rather than red.  Unlike the flock twin (amendment-pass tightening for
    this test), ``emitted`` below is built from WATCHDOG_GATE_LINE_FMT -- the
    SAME format string the bootstrap embeds into the child -- rather than a
    hand-copied literal, so a field rename in the bootstrap breaks this
    round-trip by construction instead of relying on two copies staying in
    sync by discipline.
    """
    emitted = (
        f'{WATCHDOG_GATE_LINE_FMT % (0.6931, 0.2003)}\n'
        'some unrelated stderr chatter\n'
        f'{WATCHDOG_GATE_LINE_FMT % (14.9807, 5.0012)}\n'
    )

    assert parse_watchdog_gate_fire_delays(emitted) == [
        WatchdogGateFire(fire_delay_secs=0.6931, grace_secs=0.2003),
        WatchdogGateFire(fire_delay_secs=14.9807, grace_secs=5.0012),
    ]


def test_parse_watchdog_gate_fire_delays_returns_empty_for_uninstrumented_stderr():
    """Stderr with no marker lines yields no observations.

    Mirrors test_parse_flock_gate_waits_returns_empty_for_uninstrumented_stderr
    -- this is the case the ceiling assertion's non-vacuity guard exists to
    catch: if the CLI ever stops reaching the watchdog through the
    module-level ``orchestrator.cli.start_stdin_watchdog`` name the bootstrap
    patches, the child produces no marker lines at all, and an assertion over
    an empty list would pass by default.
    """
    assert parse_watchdog_gate_fire_delays('Traceback (most recent call last):\nboom\n') == []


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


#: Ceiling on the CLI's own measured watchdog delay -- armed (immediately
#: before the watchdog thread's blocking select loop starts) to the
#: fire callback's entry, PLUS the observed SIGTERM->SIGKILL grace pause,
#: reported by the child's own clock via WATCHDOG_GATE_TIMING_BOOTSTRAP.
#: Those two terms are exactly the two env overrides under test; the two
#: /proc walks inside fire_watchdog_kill are load-sensitive and belong to
#: neither, so they are deliberately excluded (third correction -- see the
#: banner above WATCHDOG_GATE_TIMING_BOOTSTRAP). This is deliberately NOT just "halfway
#: between the fully-wired and fully-un-wired extremes": it also has to
#: catch either override being wired while the OTHER silently regresses to
#: its production default, e.g. only ORCH_WATCHDOG_KILL_GRACE_SECS stops
#: being threaded through (0.5s override + 5.0s production grace = ~5.5s) or
#: only ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS does (10.0s production timeout
#: + 0.2s override = ~10.2s). 4.0s sits comfortably under the nearer of
#: those two partial-regression cases (5.5s) with room to spare above the
#: fully-wired case (~0.7s: 0.5s heartbeat timeout + 0.2s kill grace) even
#: under load, so it discriminates all three broken shapes from the
#: fully-wired one -- the same discriminating-ceiling shape as
#: FLOCK_WAIT_CEILING_SECS, just against two failure axes instead of one.
WATCHDOG_FIRE_DELAY_CEILING_SECS = 4.0

#: The two env-override values the ceiling test threads into the child, named
#: once so the ``extra_env`` it spawns with and the assertions it makes about
#: what came back cannot drift apart.
WATCHDOG_HEARTBEAT_OVERRIDE_SECS = 0.5
WATCHDOG_KILL_GRACE_OVERRIDE_SECS = 0.2


@pytest.mark.timeout(180)  # task 4474: one subprocess, wedge-detector wait widened
def test_watchdog_timeout_env_override_fires_fast_without_heartbeat(tmp_path):
    """ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS/_KILL_GRACE_SECS override the watchdog window.

    Spawns a real verify-merge --request-id with stdin=PIPE and NEVER writes
    a heartbeat, asserting the watchdog fires and self-exits non-zero.

    task 4474 (de-flake): the assertion is on the delay the CLI's OWN clock
    measured -- armed to fire-callback entry, plus the observed kill grace,
    covering BOTH overrides under test and nothing else -- reported by
    :data:`WATCHDOG_GATE_TIMING_BOOTSTRAP` from inside the child, NOT on an
    outer wall-clock measurement.  (Task 4559 verify, third correction: the
    window deliberately EXCLUDES fire_watchdog_kill's two /proc walks, which
    belong to neither override and grow without bound under load -- see the
    banner above WATCHDOG_GATE_TIMING_BOOTSTRAP.)

    Task 2921 measured this as an OUTER wall-clock quantity -- pgid-file
    appearance to proc.wait() return -- reasoning that absorbing the
    load-sensitive ``from orchestrator.cli import main`` startup via the
    pgid-file wait would leave the remaining wait purely watchdog-bound.
    Task 4248's verify attempt-1 falsified that: under a full-suite xdist
    storm the outer fire_delay came in at 19.75s -- past even the 15.0s
    production un-wired-override window -- while an isolated rerun at the
    same HEAD passed in 6.51s. The outer measurement was never purely the
    watchdog's own window: it also carries scheduler latency between the
    watchdog thread actually firing and this PARENT process observing
    proc.wait() return, which balloons under CPU contention exactly like the
    import cost the pgid-file wait was already compensating for. Same defect
    and same remedy as test_flock_wait_env_override_speeds_up_contention_result
    (task 3369): stop using wall clock as a proxy for a quantity the code
    under test already measures itself.

    The child still runs the real CLI end to end -- real argv, real config,
    real env, a real watchdog thread racing a real select() on fd 0, real
    SIGTERM/SIGKILL signaling. The bootstrap only wraps the production
    ``start_stdin_watchdog``/``fire_watchdog_kill`` callables in a stopwatch,
    so an un-wired override still shows up as a ~15s delay (10.0s heartbeat
    timeout + 5.0s kill grace) -- see WATCHDOG_FIRE_DELAY_CEILING_SECS for
    the exact discriminant, including the two partial-regression shapes.
    Its two failure modes are both closed by non-vacuity assertions below:
    the CLI reaching the watchdog/kill by some path other than the patched
    module-level names (leaving zero observations and a vacuously-true
    ceiling), and the ``sleep=`` injection failing to land (leaving the
    reconstructed delay's grace term at 0.0, which would silently drop
    ORCH_WATCHDOG_KILL_GRACE_SECS out of the ceiling rather than fail it).
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=False)

    REQUEST_ID = 'env-seam-watchdog-test'

    proc = spawn_verify_merge(
        sha=head_sha,
        spec=sleeper_spec(300.0),
        cfg_file=cfg_file,
        request_id=REQUEST_ID,
        extra_env={
            'ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS': str(WATCHDOG_HEARTBEAT_OVERRIDE_SECS),
            'ORCH_WATCHDOG_KILL_GRACE_SECS': str(WATCHDOG_KILL_GRACE_OVERRIDE_SECS),
        },
        bootstrap=WATCHDOG_GATE_TIMING_BOOTSTRAP,
    )
    try:
        # No heartbeat is ever written -- stdin stays open but silent, which
        # is sufficient to exercise the heartbeat-starvation timing path
        # (Rows 1-3 later distinguish EOF vs. timeout precisely). This
        # ceiling is a WEDGE DETECTOR, not a speed assertion (task 2376's
        # lesson for the flock twin): the discriminating invariant is the
        # WATCHDOG_FIRE_DELAY_CEILING_SECS assertion below, never this one,
        # which only bounds a genuinely wedged child and so costs zero
        # wall-clock on the success path. It must NOT be replaced by a bare
        # proc.communicate() here (amendment-pass fix): unlike the flock
        # twin, communicate() with no ``input=`` closes stdin as its FIRST
        # action, which would deliver EOF to the child and race the
        # heartbeat-starvation path this test exists to exercise.
        try:
            proc.wait(timeout=60.0)
        except subprocess.TimeoutExpired:
            pytest.fail(
                'verify-merge did not self-exit within 60s of startup -- '
                'the watchdog did not fire at all'
            )
        # Only NOW -- after the child has already self-exited on its own,
        # confirmed above without ever touching stdin -- is it safe to drain
        # both pipes via communicate(): it also drains stdout (never done
        # before; a pre-fire stdout write bigger than one pipe buffer used to
        # risk a wedge) and closes all three pipe fds, bounded by a short
        # timeout rather than an unbounded read (amendment-pass fix,
        # mirroring test_flock_wait_env_override_speeds_up_contention_
        # result's proc.communicate() as closely as this test's
        # stdin-must-stay-open-until-fire requirement allows).
        try:
            _, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pytest.fail(
                'verify-merge self-exited but its stdout/stderr pipes did '
                'not reach EOF within 5s afterwards -- a descendant may have '
                'survived fire_watchdog_kill and is still holding a pipe open'
            )
    finally:
        # Merge of two independent teardown fixes to this one block; BOTH are
        # load-bearing and neither subsumes the other.
        #
        # (task 4092) kill_holder_tree replaces the former leader-only
        # ``proc.kill()``.  The holder's build is ``sleep 300`` run by the
        # real CLI under ``start_new_session=True``, so it does NOT die with
        # the leader: a leader-only SIGKILL orphaned that sleep to init for
        # up to five minutes after every run.  kill_holder_tree sweeps the
        # descendant tree (plus a guarded killpg backstop) and reaps the
        # leader itself, so it also subsumes the old ``proc.wait(timeout=5)``.
        # Its ``poll()``-free contract is why no ``if proc.poll() is None``
        # guard is reintroduced here -- see its docstring's ALREADY-REAPED
        # SHORT CIRCUIT paragraph for the pid-recycling hazard that adds.
        #
        # (commit aa3095175c) closing the pipe fds on EVERY path, including
        # the wedge/failure path where the ``pytest.fail``s above fire before
        # proc.communicate() ever runs.  kill_holder_tree closes stdin on
        # both of its exits but deliberately leaves stdout/stderr alone (it
        # serves call sites that still read them), so the stdout/stderr
        # closes must stay HERE rather than migrate into the helper.
        kill_holder_tree(proc, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS)
        if proc.stdout is not None:
            with contextlib.suppress(OSError):
                proc.stdout.close()
        if proc.stderr is not None:
            with contextlib.suppress(OSError):
                proc.stderr.close()

    assert proc.returncode != 0, (
        f'expected non-zero exit (watchdog self-kill), got {proc.returncode}'
    )

    fires = parse_watchdog_gate_fire_delays(stderr.decode(errors='replace'))
    # Non-vacuity guard #1: mirrors the flock twin's -- the ceiling assertion
    # below is a max() over these observations, so an empty list would pass
    # by default. Zero observations means the CLI no longer calls
    # start_stdin_watchdog/fire_watchdog_kill through the module-level
    # ``orchestrator.cli`` names the bootstrap patches -- a real signal that
    # this test stopped measuring anything, not a pass.
    assert fires, (
        'no instrumented watchdog-gate observation on the child stderr -- '
        'the timing bootstrap patches orchestrator.cli.start_stdin_watchdog '
        'and orchestrator.cli.fire_watchdog_kill, so zero observations means '
        'the watchdog/kill path is no longer reached through those names '
        f'and the ceiling assertion below would be vacuous; '
        f'stderr={stderr_tail(stderr)!r}'
    )
    # Non-vacuity guard #2 (task 4559 verify, the third correction -- see the
    # banner above WATCHDOG_GATE_TIMING_BOOTSTRAP). fire_delay is now
    # RECONSTRUCTED as (t_fire_entry - t0) + observed_grace_sleep so the two
    # load-sensitive /proc walks stay out of it. That makes the kill-grace
    # axis depend on the bootstrap's ``sleep=`` injection actually landing:
    # if fire_watchdog_kill ever stopped accepting ``sleep`` or stopped
    # sleeping, the grace term would silently collapse to 0.0, every
    # fire_delay would shrink, and the ceiling would keep passing while
    # having quietly stopped covering ORCH_WATCHDOG_KILL_GRACE_SECS at all.
    # A real time.sleep(grace) can only overshoot its argument, never
    # undershoot it, so requiring the observed pause to reach the override is
    # safe under any load -- and a grace that regressed to the 5.0s
    # production default is NOT filtered out here on purpose: it flows into
    # fire_delay and is caught by the ceiling below, which is the axis that
    # is supposed to catch it.
    unmeasured = [f for f in fires if f.grace_secs < WATCHDOG_KILL_GRACE_OVERRIDE_SECS]
    assert not unmeasured, (
        f'the SIGTERM->SIGKILL grace pause was not observed at its '
        f'{WATCHDOG_KILL_GRACE_OVERRIDE_SECS}s override on '
        f'{len(unmeasured)} of {len(fires)} fire(s): {unmeasured}. '
        f'fire_delay is reconstructed as (fire-entry - armed) + observed '
        f'grace, so an unobserved grace term silently retires '
        f'ORCH_WATCHDOG_KILL_GRACE_SECS from the ceiling assertion below '
        f'rather than failing it -- the bootstrap\'s sleep= injection into '
        f'fire_watchdog_kill is no longer taking effect; '
        f'stderr={stderr_tail(stderr)!r}'
    )
    longest = max(f.fire_delay_secs for f in fires)
    assert longest < WATCHDOG_FIRE_DELAY_CEILING_SECS, (
        f'expected the CLI-measured watchdog delay (heartbeat wait + kill '
        f'grace, excluding fire_watchdog_kill\'s /proc walks) to be well '
        f'under the 15s production window (10s heartbeat timeout + 5s kill '
        f'grace) given the '
        f'{WATCHDOG_HEARTBEAT_OVERRIDE_SECS}s+{WATCHDOG_KILL_GRACE_OVERRIDE_SECS}s '
        f'env overrides -- longest={longest:.2f}s over {len(fires)} fire(s): '
        f'{fires}; the watchdog env overrides are not fully wired up'
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


@pytest.mark.timeout(ROW_PER_TEST_TIMEOUT_SECS)  # task 4025 amendment: shares the derived per-row budget -- same real-subprocess exposure as Rows 1/2/3, previously left on the module's bare 60s ini timeout
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
        kill_holder_tree(child, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS)


# ---------------------------------------------------------------------------
# Task 2309 step-7/8 -- SS9 Row 1: orchestrator (dispatcher) killed mid-build
# (EOF path via whole-dispatcher death).  A SEPARATE killable process runs the
# REAL verify_runner._default_ssh_heartbeat_run; SIGKILLing it closes its end
# of the child's stdin pipe, delivering a clean EOF -- distinct from Row 3's
# heartbeat-timeout (select-timeout) branch, which keeps stdin open but silent.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(ROW_PER_TEST_TIMEOUT_SECS)  # task 4025: production 10s+5s watchdog window + real subprocesses under full-suite load
def test_orchestrator_killed_mid_build_tree_killed_via_eof(tmp_path):
    """SS9 Row 1: dispatcher process killed -> child sees stdin EOF -> tree-killed.

    Models "the orchestrator holding the ssh child died": spawns a SEPARATE
    dispatcher process running the REAL ``_default_ssh_heartbeat_run`` against
    a local ``verify-merge --request-id`` argv (small heartbeat_interval so
    real heartbeats flow while the dispatcher is alive).  Waits for the
    sleeper subtree to appear, then SIGKILLs the dispatcher process itself --
    when the OS reclaims its file descriptors, ITS end of the child's stdin
    pipe closes, giving the grandchild a clean EOF on fd 0.

    ROW_WATCHDOG_ENV is threaded through the dispatcher's own environment
    (:func:`spawn_ssh_heartbeat_dispatcher`'s *extra_env*, ambiently inherited
    by the grandchild verify-merge) -- the pinned 10s+5s production-equivalent
    window, not a fast one; see ROW_WATCHDOG_ENV's comment above for the pin
    rationale.  Per ``run_stdin_watchdog``, EOF fires on the very next
    ``select`` readiness check regardless of *heartbeat_timeout* -- only
    ``grace_secs`` (the SIGTERM->SIGKILL pause in ``fire_watchdog_kill``)
    materially bounds this row's timing (~5s, measured ~7.3s end-to-end).

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
        extra_env=ROW_WATCHDOG_ENV,
    )
    try:
        pgid_val = wait_for_pgid_file(pgf, timeout=row_discovery_ceiling_secs())
        # Row 1 owns the DISPATCHER process, not the leader -- the leader's
        # own stdout/stderr aren't piped to this test, so pass the dispatcher.
        wait_subtree_live(
            pgid_val, proc=dispatcher, proc_label='dispatcher',
            timeout=row_discovery_ceiling_secs(),
        )

        dispatcher.kill()
        dispatcher.wait(timeout=10)

        assert wait_subtree_gone(pgid_val, timeout=ROW_TREE_KILL_CEILING_SECS), (
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


@pytest.mark.timeout(ROW_PER_TEST_TIMEOUT_SECS)  # task 4025: production 10s+5s watchdog window + real subprocesses under full-suite load
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

    Uses ROW_WATCHDOG_ENV directly on the spawned verify-merge (already an
    in-process dispatcher, so no extra remove is needed, unlike Row 1's
    separate-process case) -- the pinned 10s+5s production-equivalent
    window.  Row 2 is an EOF row like Row 1, not a timeout row:
    ``close_stdin()`` closes the write end, so only ``grace_secs`` bounds it
    (~5s, measured ~6.7s end-to-end), NOT the ~15s a reader might assume
    from the heartbeat timeout.

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
        extra_env=ROW_WATCHDOG_ENV,
    )
    heartbeat = HeartbeatWriter(child, interval=0.2).start()
    try:
        pgid_val = wait_for_pgid_file(pgf, timeout=row_discovery_ceiling_secs())
        wait_subtree_live(pgid_val, proc=child, timeout=row_discovery_ceiling_secs())

        heartbeat.close_stdin()

        # Reap the leader (a DIRECT child of this test process, unlike Row
        # 1's separate-dispatcher indirection or Row 4's cancel-verify path)
        # BEFORE polling wait_subtree_gone: os.killpg(pgid, 0) succeeds
        # against an unreaped zombie too, so checking liveness first would
        # spuriously see the group as "alive" until something calls
        # child.wait() -- confirmed by a manual repro that hung the full
        # poll window with this ordering reversed.
        try:
            child.wait(timeout=ROW_TREE_KILL_CEILING_SECS)
        except subprocess.TimeoutExpired:
            pytest.fail(
                f'verify-merge did not exit within '
                f'{ROW_TREE_KILL_CEILING_SECS}s after stdin EOF'
            )
        assert child.returncode != 0, (
            f'expected non-zero exit (watchdog self-kill), got {child.returncode}'
        )

        assert wait_subtree_gone(pgid_val, timeout=ROW_TREE_KILL_CEILING_SECS), (
            f'pgid {pgid_val}: subtree and/or leader still alive after the '
            f'ssh channel EOF (dispatcher alive) -- watchdog tree-kill did '
            f'not fire'
        )
    finally:
        heartbeat.stop_heartbeats()
        kill_holder_tree(child, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS)


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
        pgid_val = wait_for_pgid_file(pgf, timeout=row_discovery_ceiling_secs())
        wait_subtree_live(pgid_val, proc=child, timeout=row_discovery_ceiling_secs())

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
        kill_holder_tree(child, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS)


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


# task 2921: two real verify-merge subprocesses (holder + waiter) under
# full-suite load -- the module's most storm-exposed row.  Now DERIVED rather
# than the literal 120 task 2921 wrote: task 4014 made this row's two
# discovery waits load-scaled (up to ROW_DISCOVERY_CEILING_MAX_SECS each), and
# a literal cannot track that -- which is exactly what the review caught.
@pytest.mark.timeout(ROW5_PER_TEST_TIMEOUT_SECS)
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
        stdout, stderr = waiter.communicate(timeout=ROW5_WAITER_COMPLETION_CEILING_SECS)

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
        kill_holder_tree(holder, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS)

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
        kill_holder_tree(holder, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS)
