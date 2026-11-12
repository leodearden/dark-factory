"""The real-subprocess harness the verify-merge boundary tests drive.

Spec builders for a controllable sleeper or fast-exit build, the real-CLI and
dispatcher spawners, the in-process heartbeat writer, the process-tree and
pgid-file observers, and :func:`kill_holder_tree`, which tears a holder's
whole tree down.  Its drivers are the SS9 rows in
test_laptop_warm_verify_boundary.py, the CLI env-override seam tests in
test_verify_merge_env_overrides.py, and its own self-tests in
test_verify_boundary_harness.py.  The timing it spends lives in
_verify_boundary_ceilings.py.

Every observer reuses orchestrator.verify_cancel's /proc walkers -- the SAME
functions the production watchdog/cancel kill paths use -- so there is no
divergent ad-hoc process-tree parsing.  That single-source rule holds exactly
where it decides anything, and task 4014 added ONE deliberate, narrow
exception on the other side of a discovery/assertion split:

* ASSERTIONS -- every row's kill verdict goes through ``wait_pids_exited``
  (a zombie-aware ``shared/src/shared/proc_group.py::read_stat_fields``
  read), applied by ``assert_pre_kill_tree_exited`` to the pids
  ``snapshot_pre_kill_tree`` captured while the leader was alive.  Those
  pids, and the descendant set ``wait_subtree_live`` RETURNS, are still
  discovered only by the production walkers
  ``collect_descendants``/``read_ppid_map``.
* the ARRANGE-phase discovery GATE adds a cheap Linux
  ``/proc/<pid>/task/*/children`` pre-filter (:func:`read_direct_children`)
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
WATCHDOG WINDOW (and the ROW_WATCHDOG banner in _verify_boundary_ceilings.py
is explicit that this "MOVES the cliff from 1.0s to 10s, it does not remove
it"); task 4014 addressed the POLLER above it -- the per-tick cost, the
transient-child race, and the flat 20s discovery ceiling that did not track
load.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
import warnings
from collections.abc import Awaitable, Callable, Mapping
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any, NamedTuple

from _gate_fixtures import gate_argv
from _leader_stderr import drain_exited_stderr
from _verify_boundary_ceilings import (
    ROW5_HOLDER_TEARDOWN_CEILING_SECS,
    ROW_MARKER_CEILING_SECS,
    row_discovery_ceiling_secs,
)
from shared.proc_group import StatFields, read_stat_fields

from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.verify_cancel import collect_descendants, kill_process_tree, read_ppid_map
from orchestrator.verify_runner import (
    MergeVerifySpec,
    UnscopedTypecheckSpec,
    VerifyCommand,
    _default_ssh_heartbeat_run,
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
    to production.
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


def sleeper_spec(
    sleep_secs: float = 300.0,
    *,
    marker: str = 'target/warm.marker',
    build_pgid_file: Path | None = None,
) -> MergeVerifySpec:
    """MergeVerifySpec whose scoped test command touches *marker* then blocks.

    Reproduces the real cargo/rustc start_new_session escape (verify.py
    ``_run_cmd``, which starts every module command with
    ``start_new_session=True``) with trivial /bin/bash.

    With *build_pgid_file*, the build shell atomically writes its own pid
    there (which is also its pgid, since ``_run_cmd`` starts a new session)
    and then execs into the sleeper: the published pid IS the process that
    blocks, and nothing forks after it is published.
    """
    blocker = f'sleep {sleep_secs}'
    if build_pgid_file is not None:
        tmp = build_pgid_file.parent / (build_pgid_file.name + '.tmp')
        blocker = (
            f'echo $$ > {shlex.quote(str(tmp))} && '
            f'mv {shlex.quote(str(tmp))} {shlex.quote(str(build_pgid_file))} && '
            f'exec {blocker}'
        )
    return MergeVerifySpec(
        verify_commands=(
            VerifyCommand('mod', test_command=f'mkdir -p target && touch {marker} && {blocker}'),
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
#: what the child's own clock measured (see
#: test_verify_merge_env_overrides.py::FLOCK_GATE_TIMING_BOOTSTRAP)
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


#: Deliberately NOT ``ORCH_``-prefixed: verify.py::_target_subprocess_env
#: scrubs that namespace from every build it spawns.
HOLDER_TAG_ENV: str = 'DF_TEST_HOLDER_TAG'


class TaggedPopen(subprocess.Popen[bytes]):
    """A Popen whose whole process tree carries ``HOLDER_TAG_ENV=<self.tag>``.

    Every descendant inherits the tag across fork, setsid and exec --
    including a start_new_session build whose /proc ppid chain is severed
    the moment the leader dies -- so :func:`tagged_pids` still names it.
    """

    def __init__(self, args, *, env: Mapping[str, str] | None = None, **kwargs: Any) -> None:
        self.tag = uuid.uuid4().hex
        tagged_env = dict(os.environ if env is None else env)
        tagged_env[HOLDER_TAG_ENV] = self.tag
        super().__init__(args, env=tagged_env, **kwargs)


def tagged_pids(tag: str) -> set[int]:
    """Pids whose environment holds exactly ``HOLDER_TAG_ENV=<tag>``, never this process.

    An unreadable entry (vanished, or another user's) is skipped, and a
    zombie reads an empty environ, so only live processes match.
    """
    wanted = f'{HOLDER_TAG_ENV}={tag}'.encode()
    pids: set[int] = set()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / 'environ').read_bytes()
        except OSError:
            continue
        if wanted in raw.split(b'\0'):
            pids.add(int(entry.name))
    pids.discard(os.getpid())
    return pids


def sweep_tagged(tag: str, *, kill: Callable[[int, int], None] = os.kill) -> None:
    """SIGKILL every live process carrying ``HOLDER_TAG_ENV=<tag>``; one already gone is fine."""
    for pid in tagged_pids(tag):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            kill(pid, signal.SIGKILL)


def spawn_verify_merge(
    *,
    sha: str,
    spec: MergeVerifySpec,
    cfg_file: Path,
    request_id: str | None = None,
    stdin: int | None = subprocess.PIPE,
    extra_env: dict[str, str] | None = None,
    bootstrap: str = STOCK_BOOTSTRAP,
) -> TaggedPopen:
    """Spawn a real ``orchestrator verify-merge`` subprocess, tagged per holder.

    ``stdin=subprocess.PIPE`` by default so callers can drive the
    connection-death protocol (heartbeat writer / EOF-on-close); tests that
    don't pass --request-id never arm the watchdog so stdin is inert for them.

    ``bootstrap`` defaults to :data:`STOCK_BOOTSTRAP` (byte-identical to the
    pre-seam argv); see that constant for the only reason to override it.
    """
    return TaggedPopen(
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


def loopback_ssh_run(
    *, heartbeat_interval: float
) -> Callable[..., Awaitable[tuple[int, str, str]]]:
    """A RemoteRunner ``ssh_run`` that runs the remote command locally, as ssh would.

    ssh becomes a transparent pipe: the leader's stdin is the REAL heartbeat
    pipe of :func:`~orchestrator.verify_runner._default_ssh_heartbeat_run`,
    and its stderr comes back exactly as ssh relays it.  The gate binary the
    remote command names is this worktree's own CLI, so the leader runs
    under the stock bootstrap.  The env reaches the leader ambiently -- see
    :func:`apply_dispatcher_env`.
    """

    async def ssh_run(argv: list[str], *, cwd: str | Path | None = None) -> tuple[int, str, str]:
        assert argv[0] == 'ssh', f'RemoteRunner no longer dispatches through ssh: {argv!r}'
        cli_args = gate_argv(argv[-1])
        return await _default_ssh_heartbeat_run(
            [sys.executable, '-c', STOCK_BOOTSTRAP, *cli_args],
            cwd=cwd,
            heartbeat_interval=heartbeat_interval,
        )

    return ssh_run


async def git_push_succeeds(
    argv: list[str], *, cwd: str | Path | None = None
) -> tuple[int, str, str]:
    """A RemoteRunner ``run`` that accepts every ``git push`` and nothing else.

    With no ``main_branch``, and the gate's own calls answered by
    :func:`~_gate_fixtures.gated_run`, the merge-sha push and its
    ``--delete`` cleanup are the only commands that reach this ``run``.
    """
    assert argv[:2] == ['git', 'push'], (
        f'RemoteRunner sent a non-push command through run: {argv!r}'
    )
    return 0, '', ''


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


def describe_proc_exit(proc: subprocess.Popen, label: str) -> str:
    """Name *proc*'s exit status -- and, once it has exited, its stderr tail.

    The diagnostic a discovery wait appends on timeout: ``<label>
    rc=<n|None>``.  *label* names *proc* in the message; pass e.g.
    ``"dispatcher"`` when *proc* is a stand-in process rather than the leader
    itself, so a reader doesn't apply the rc taxonomy below to the wrong
    process.  For the LEADER specifically, that rc distinguishes a watchdog
    self-kill (rc == 1, no ``Error:`` line), an exception exit (rc == 1 WITH
    an ``Error:`` line), and a merely slow leader (rc is None) -- three
    causes that were otherwise indistinguishable in every log this timeout
    has ever produced; a non-leader *proc* follows its own exit-code
    conventions, not this taxonomy.  ``stderr`` is read ONLY when
    ``proc.poll()`` is not None, and even then through
    :func:`drain_exited_stderr`'s bounded read, so a helper still holding the
    write end cannot hang this helper (and the rest of the suite).  The tail
    (not head) is kept -- see :func:`stderr_tail` -- since the actionable line
    is at the END.
    """
    rc = proc.poll()
    description = f'{label} rc={rc}'
    if rc is not None and proc.stderr is not None:
        try:
            tail = stderr_tail(drain_exited_stderr(proc))
        except Exception as e:  # noqa: BLE001
            tail = f'<unreadable: {e!r}>'
        description += f'; stderr tail:\n{tail}'
    return description


def wait_for_pgid_file(
    path: Path,
    *,
    proc: subprocess.Popen | None = None,
    proc_label: str = 'leader',
    timeout: float | None = None,
    interval: float = 0.05,
) -> int:
    """Poll for a pgid file and return its int value.

    Written by ``verify-merge --request-id`` or by a :func:`sleeper_spec`
    build with *build_pgid_file*.

    *proc*, when given, is the process whose exit a timeout failure reports,
    under *proc_label* -- see :func:`describe_proc_exit`.

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
    message = f'pgid file {path} did not appear within {timeout}s'
    if proc is not None:
        message += f'; {describe_proc_exit(proc, proc_label)}'
    raise AssertionError(message)


def read_direct_children(pid: int, *, _proc_root: Path = Path('/proc')) -> set[int] | None:
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
    (and every kill assertion the boundary rows make) still comes from the production
    walkers ``collect_descendants``/``read_ppid_map``.

    TRI-STATE, and the third state is load-bearing:

    * ``set()``  -- leader is live and has forked nothing yet.  A cheap
      NEGATIVE: the caller can skip the expensive walk this tick.
    * ``{pid, ...}`` -- direct children exist; worth confirming with the
      production walker.
    * ``None``   -- CANNOT probe: ``/proc/<pid>`` is gone (the leader has
      exited), a LIVE thread's ``children`` file cannot be read (a kernel
      built without ``CONFIG_PROC_CHILDREN``, or fd exhaustion), or no thread
      was read at all (the leader exited before its first thread was read).
      The caller must fall back to the full walk for that tick; conflating
      this with the cheap negative would make the poll spin to its timeout,
      and letting the OSError escape would turn a leader exiting mid-poll
      into an unhandled error inside the timeout diagnostic.

    A failed ``children`` read is judged PER THREAD, because threads exit
    routinely in the probed processes -- pytest's own per-test faulthandler /
    pytest-timeout watchdog threads included (task 5945).  If the tid
    directory is gone, the thread exited after the listing and is skipped; if
    it still exists, the probe cannot be trusted and answers ``None``.  So one
    tick can under-report: a child the kernel re-parents from a skipped thread
    onto a sibling already read is missed, and a leader that exits after some
    of its threads were read answers the partial set those threads gave
    (possibly ``set()``), not ``None``.  Either only defers the caller's walk
    by one poll interval -- the next tick sees the re-parented child, or finds
    ``/proc/<pid>`` gone and answers ``None`` -- and the kernel documents
    ``children`` as best-effort anyway.

    *_proc_root* is a private injectable seam (defaulting to the real
    ``/proc``) so the tri-state can be covered against a fake task listing in
    ``tmp_path``, for the branches (no ``CONFIG_PROC_CHILDREN``, every thread
    vanished) that this kernel cannot produce on demand.
    """
    try:
        tid_dirs = list((_proc_root / str(pid) / 'task').iterdir())
    except OSError:
        return None
    children: set[int] = set()
    any_thread_read = False
    for tid_dir in tid_dirs:
        try:
            raw = (tid_dir / 'children').read_text()
        except OSError:
            if tid_dir.exists():
                return None
            continue
        any_thread_read = True
        children.update(int(token) for token in raw.split())
    return children if any_thread_read else None


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
    :func:`read_direct_children`.  A positive probe whose confirming walk
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
    When given, a timeout failure reports its exit under *proc_label* -- see
    :func:`describe_proc_exit`.

    *timeout* defaults to :func:`row_discovery_ceiling_secs`, resolved when
    the wait actually STARTS rather than at import, so a row beginning
    mid-storm gets the load-scaled deadline the storm warrants.

    *_probe_children* / *_ppid_map* are private injectable seams (defaulting
    to :func:`read_direct_children` and
    :func:`~orchestrator.verify_cancel.read_ppid_map`) so this poll loop can
    be covered deterministically, with no real timing and no real
    subprocess -- the same pattern :func:`wait_for_marker_stable`'s
    ``_read_mtime_ns`` seam established (task 2819).
    """
    timeout = row_discovery_ceiling_secs() if timeout is None else timeout
    probe = _probe_children or read_direct_children
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
        message += f'; {describe_proc_exit(proc, proc_label)}'
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
    BODY rather than as a default expression -- the same lazy-default
    convention as :func:`wait_for_pgid_file`, whose discovery ceiling is
    load-sampled per call -- so the value a row budgets for and the value
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
    count that ceiling twice (see
    ``_verify_boundary_ceilings.py::_ROW5_WORST_CASE_FIXED_SECS``).
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


def read_proc_state(pid: int) -> StatFields | None:
    """*pid*'s stat fields, or None once it has no ``/proc`` entry.

    A probe of one pid, so a malformed stat raises rather than reading as gone
    (task 5903).
    """
    return read_stat_fields(Path('/proc') / str(pid), skip_malformed=False)


def wait_pids_exited(
    pids: AbstractSet[int],
    *,
    timeout: float,
    interval: float = 0.05,
    _read_state: Callable[[int], StatFields | None] = read_proc_state,
    _clock: Callable[[], float] = time.monotonic,
) -> dict[int, StatFields]:
    """Poll until every pid in *pids* has exited; return the still-RUNNING ones.

    Each survivor maps to its last observed :class:`StatFields`; an empty dict
    means every pid exited.  A zombie counts as exited: reaping a reparented
    orphan is its subreaper's job (``systemd --user`` on this fleet), not the
    code under test's -- the same criterion as
    ``shared/src/shared/proc_group.py::pid_running`` (task 3955).

    The verdict is always the probe of the returning iteration, so a caller
    descheduled across the deadline never asserts on a stale observation;
    ``timeout=0`` is exactly one probe.
    """
    deadline = _clock() + timeout
    while True:
        running = {
            pid: state for pid in pids
            if (state := _read_state(pid)) is not None and not state.terminated
        }
        if not running or _clock() >= deadline:
            return running
        time.sleep(interval)


class PreKillTree(NamedTuple):
    """A verify-merge leader's process tree, snapshotted while the leader was alive."""

    leader: int
    build: int
    pids: frozenset[int]


def snapshot_pre_kill_tree(leader: int, build: int) -> PreKillTree:
    """Snapshot *leader*'s tree, which must contain the session-escaped *build*.

    Call it only while the leader is provably alive: once the leader dies, the
    ppid chain to the start_new_session build is severed and a walk from the
    leader can no longer reach it.
    """
    pids = frozenset({leader} | collect_descendants(leader, read_ppid_map()))
    assert build in pids, (
        f'harness failure, not a seam defect: build pgid {build} is not in '
        f'leader {leader}\'s subtree {sorted(pids)}'
    )
    return PreKillTree(leader=leader, build=build, pids=pids)


def assert_pre_kill_tree_exited(tree: PreKillTree, *, timeout: float, after: str) -> None:
    """Assert every pid in *tree* exits within *timeout* of the trigger named by *after*.

    A zombie awaiting its subreaper counts as exited, as in :func:`wait_pids_exited`.
    Only the snapshotted pids are judged: a process forked after the snapshot,
    even into the leader's own group, is outside this verdict.
    """
    survivors = wait_pids_exited(tree.pids, timeout=timeout)
    rendered = {pid: (s.state, s.ppid) for pid, s in sorted(survivors.items())}
    assert not survivors, (
        f'leader={tree.leader} build={tree.build}: {rendered} (pid: (state, ppid)) '
        f'still running {timeout}s after {after}. S/T means SIGKILL never reached '
        f'it; R/D means signalled but not yet run to exit'
    )


def _kill_pinned_holder(
    proc: TaggedPopen,
    *,
    timeout: float,
    ppid_map_provider: Callable[[], dict[int, int]],
    kill: Callable[[int, int], None],
    killpg: Callable[[int, int], None],
) -> None:
    """Tree-kill and reap an UN-REAPED *proc*: its pid is pinned until the wait below.

    ``proc.poll()`` is called NOWHERE here: it waitpid-reaps, so a leader
    that exited since the caller's gate (the Row 5 holder's finally arms the
    CLI's stdin-watchdog self-kill just before tearing down) would have its
    pid FREED before the backstop fires, aiming it at a recycled stranger's
    group.  ``kill_process_tree`` never reaps either, so the backstop always
    fires while the pid is still pinned (a zombie at worst).
    """
    # Read before any signal: a leader already gone, or any other OSError,
    # degrades to "no group to backstop" rather than raising out of a finally
    # and masking the caller's real assertion failure.
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = None
    self_led_foreign_group = pgid is not None and pgid == proc.pid and pgid != os.getpgid(0)

    def snapshot_or_empty() -> dict[int, int]:
        try:
            return ppid_map_provider()
        except OSError:
            return {}

    kill_process_tree(
        proc.pid,
        backstop_pgid=pgid if self_led_foreign_group else None,
        ppid_map_provider=snapshot_or_empty,
        kill=kill,
        killpg=killpg,
    )

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # A leader that will not die must not mask the test's own failure by
        # raising out of a finally -- surface it loudly instead.
        warnings.warn(
            f'kill_holder_tree: leader pid={proc.pid} did not exit within '
            f'{timeout}s of SIGKILL',
            stacklevel=3,
        )


def kill_holder_tree(
    proc: TaggedPopen,
    *,
    timeout: float | None = None,
    _ppid_map_provider=read_ppid_map,
    _kill=os.kill,
    _killpg=os.killpg,
) -> None:
    """SIGKILL *proc*'s whole process tree, including start_new_session escapes.

    Two phases, then the stdin close:

    1. PINNED-LEADER PHASE, only while ``proc`` is un-reaped
       (:func:`_kill_pinned_holder`): the snapshot -> walk -> SIGKILL ->
       killpg algorithm of
       ``orchestrator/src/orchestrator/verify_cancel.py::kill_process_tree``,
       rooted at ``proc.pid``, then the reap.
    2. TAG SWEEP, always: SIGKILL every live process carrying this holder's
       :class:`TaggedPopen` tag (:func:`tagged_pids`).

    Walking from ``proc.pid`` (rather than
    ``os.killpg(os.getpgid(proc.pid), SIGKILL)``, the obvious one-liner) is
    what makes this helper safe at every ``spawn_verify_merge`` call site,
    including ones that never pass ``--request-id`` (e.g. the lane-lock
    holder): ``cli.py`` only calls
    :func:`~orchestrator.verify_cancel.start_own_process_group`
    (``os.setsid``) inside ``if request_id is not None:``, and
    ``spawn_verify_merge`` itself passes no ``start_new_session=``, so a
    holder started without ``--request-id`` stays in the CALLER's (this
    test process's) own process group -- ``os.getpgid(holder.pid) ==
    os.getpgid(0)``.  An unconditional ``killpg`` there would SIGKILL the
    pytest worker running this very test -- see
    ``test_verify_boundary_harness.py::``
    ``test_kill_holder_tree_never_signals_the_callers_own_process_group``.
    So the backstop is aimed only at a group the holder provably leads and
    the caller provably is not in.  The descendant walk has no such hazard:
    the caller is always an ANCESTOR of the holder, never a descendant --
    the same argument
    :func:`~orchestrator.verify_cancel.start_own_process_group` makes for
    ``sshd`` never being a descendant of the pgid ``cancel_request`` walks.

    PINNED-LEADER GATE: an already-reaped leader means ``proc.pid`` is a
    FREE pid, so nothing may be derived from it.  Several call sites reap
    the leader with ``wait()`` inside their ``try`` block BEFORE their
    ``finally`` runs this helper (test_verify_merge_env_overrides.py's
    ``test_watchdog_timeout_env_override_fires_fast_without_heartbeat``, and
    test_laptop_warm_verify_boundary.py's
    ``test_cancel_verify_tree_kills_under_live_watchdog``,
    ``test_ssh_dropped_mid_build_tree_killed_via_eof_dispatcher_alive`` and
    ``test_heartbeat_starved_hard_partition_tree_killed_via_timeout``) --
    on their GREEN path ``proc.pid`` no longer refers to the holder at all.
    ``os.getpgid(proc.pid)`` and ``collect_descendants(proc.pid, ...)``
    would then describe whatever process happens to OWN that recycled pid
    now, and every descendant of that stranger would be SIGKILLed (and
    killpg'd by the backstop, if it happened to be its own group leader).
    pid recycling is observed on this fleet, not theoretical (the stale-file
    NOTE in ``orchestrator/src/orchestrator/verify_cancel.py::cancel_request``:
    pid_max=4194304, and the laptop's own pid counter demonstrably wrapped
    on 2026-08-11).  So liveness is read ONCE, at entry, from
    ``proc.returncode`` -- state this Popen already owns, never a reaping
    ``poll()`` -- strictly before the pgid read and the ppid-map snapshot.

    WHY BOTH PHASES.  The walk cannot reach the orphan of an already-reaped
    leader: verify.py's ``_run_cmd`` runs every build with
    ``start_new_session=True``, so the build is reachable only through the
    /proc ppid chain, which is severed the instant the leader dies, and it
    leads its own group, so no killpg reaches it either.  The tag sweep can:
    the tag is in the build's environ from fork onward, and a stranger
    cannot match a uuid it never inherited -- short of a pid freed and
    reused between :func:`tagged_pids`'s environ read and the SIGKILL.
    Conversely, the sweep would miss a descendant that execs with a
    scrubbed environment, which the walk still reaches while the leader is
    pinned.

    *timeout* defaults to :data:`ROW5_HOLDER_TEARDOWN_CEILING_SECS`,
    resolved when the call actually runs rather than at import time -- the
    same lazy-default convention :func:`wait_for_pgid_file` and
    :func:`wait_subtree_live` use above, whose discovery ceiling is
    load-sampled per call.  This is the ONLY blocking wait the helper
    performs (both sweeps are pure signalling, no polling loop), so the
    budget ``_verify_boundary_ceilings.py::_ROW5_WORST_CASE_FIXED_SECS``
    derives from stays valid unchanged.

    *_ppid_map_provider* / *_kill* / *_killpg* are private injectable seams
    (*_kill* also serves the tag sweep), so tests can pin the session-escape
    reap deterministically with zero risk of signalling anything unintended.
    """
    timeout = ROW5_HOLDER_TEARDOWN_CEILING_SECS if timeout is None else timeout

    if proc.returncode is None:
        _kill_pinned_holder(
            proc,
            timeout=timeout,
            ppid_map_provider=_ppid_map_provider,
            kill=_kill,
            killpg=_killpg,
        )

    sweep_tagged(proc.tag, kill=_kill)

    if proc.stdin is not None:
        with contextlib.suppress(OSError):
            proc.stdin.close()


def worktree_base_for(repo: Path) -> Path:
    """Derive worktree_base exactly as the spawned CLI does: GitOps(config.git, repo).worktree_base."""
    config = OrchestratorConfig(project_root=repo)
    return GitOps(config.git, repo).worktree_base
