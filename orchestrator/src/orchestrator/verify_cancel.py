"""Host-side cancellation contract for ``orchestrator verify-merge``.

Overview
--------
When ``orchestrator verify-merge --request-id X`` starts, it:

1. Calls :func:`start_own_process_group` to join a new session/process group
   (``os.setsid``), isolating itself from the SSH session that launched it.
   After ``setsid`` the process's pgid equals its pid, so one integer serves
   as both the tree-walk root and the ``killpg`` backstop target.

2. Writes that pgid to a *pgid file* via :func:`write_pgid_file` so a
   concurrent ``orchestrator cancel-verify --request-id X`` can find it.

3. Removes the file on exit (normal or exceptional) via :func:`remove_pgid_file`.

The verify bundle spawns every build/test command with
``start_new_session=True`` (``verify.py`` ``_run_cmd``), making each command
the leader of its own session and process group — it **escapes** the
``verify-merge`` process group.  A plain ``killpg(recorded_group)`` therefore
strands those zombie builds.  :func:`cancel_request` fixes this by:

* Snapshotting the full ``/proc`` PPID map *before* sending any signals
  (killing the root reparents survivors to init, losing tree linkage).
* Walking the descendant tree of the recorded pgid to collect every
  ``start_new_session`` escape.
* ``SIGKILL``-ing every collected descendant **and** firing a
  ``killpg(pgid, SIGKILL)`` backstop for any same-group stragglers.

pgid-file directory
-------------------
Files live at::

    <worktree_base>/.merge_verify_pgids/<sanitized_request_id>

This mirrors the existing host-state precedent
``<worktree_base>/.merge_verify_host_attempts`` (``git_ops.py``:1864):
host-side, per-project, survives stateless CLI invocations, and never
inside a registered worktree so it is never pruned or ``git``-cleaned.
``worktree_base`` is derived via
``GitOps(config.git, config.project_root).worktree_base`` (single source
of truth, same as ``git_ops.py``:480).

Cross-host rollout
------------------
Both the server and the laptop host run the ``df`` checkout, so landing this
module on ``main`` ships the cancellation contract to the laptop via its
normal checkout sync — no separate deploy step required.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
import select
import signal
import time
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Name of the directory (under ``worktree_base``) where pgid files are kept.
#: Mirrors the ``.merge_verify_host_attempts`` file precedent: host-side,
#: per-project, never pruned/git-cleaned.
PGID_DIR_NAME: str = '.merge_verify_pgids'

# Allowlist for request_id characters: alphanumeric, dash, underscore, dot.
# Anything outside this set is stripped so the sanitized result is a safe
# filename that cannot path-traverse out of PGID_DIR_NAME.
_SAFE_RE = re.compile(r'[^A-Za-z0-9\-_.]')


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def pgid_dir(worktree_base: Path) -> Path:
    """Return the pgid-file directory (not necessarily created yet)."""
    return worktree_base / PGID_DIR_NAME


def pgid_file(worktree_base: Path, request_id: str) -> Path:
    """Return the path for the pgid file for *request_id*.

    *request_id* is sanitized: only ``[A-Za-z0-9-_.]`` chars are kept, so a
    hostile id such as ``'../escape'`` or ``'a/b'`` cannot path-traverse
    outside :data:`PGID_DIR_NAME`.  If the result is empty after sanitization
    a fallback of ``'_unknown'`` is used.
    """
    safe = _SAFE_RE.sub('', request_id)
    if not safe:
        safe = '_unknown'
    return pgid_dir(worktree_base) / safe


# ---------------------------------------------------------------------------
# File lifecycle
# ---------------------------------------------------------------------------


def write_pgid_file(path: Path, pgid: int) -> None:
    """Atomically write *pgid* to *path*, creating parent directories first.

    Uses a sibling tmp file + ``os.replace`` for atomicity so a concurrent
    reader never sees a partial write.  The temp name is derived additively
    (``path.name + '.tmp'``) rather than with ``path.with_suffix('.tmp')``:
    ``with_suffix`` replaces from the *last* dot, so request-ids containing
    dots (e.g. ``'1.2.3'``) would produce colliding tmp names (``'1.tmp'``)
    and defeat the atomicity guarantee for concurrent runs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + '.tmp')
    tmp.write_text(str(pgid))
    os.replace(tmp, path)


def remove_pgid_file(path: Path) -> None:
    """Remove *path* if it exists; silently a no-op when absent (idempotent)."""
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


# ---------------------------------------------------------------------------
# Process-group management
# ---------------------------------------------------------------------------


def start_own_process_group() -> int:
    """Call ``os.setsid()`` to start a new session/process group and return the pgid.

    After ``setsid`` the process's ``pgid`` equals its ``pid``, so the single
    returned integer serves as both the ``/proc`` tree-walk root and the
    ``os.killpg`` backstop target in :func:`cancel_request`.

    The purpose of ``setsid`` is to isolate ``verify-merge`` from the SSH
    session that launched it (``sshd`` is an ancestor, not a descendant, so
    the descendant-tree walk already prevents touching it).  The ``killpg``
    backstop adds a second layer of protection for same-group stragglers.

    If ``setsid`` raises ``OSError`` (``EPERM`` — the process is already a
    session leader, which can happen in a bare ``ssh`` invocation without a
    controlling terminal) we fall back to returning ``os.getpgrp()`` so that a
    valid kill target is always recorded.
    """
    try:
        os.setsid()
        return os.getpgrp()
    except OSError:
        # Already a session leader; return the current group as a fallback.
        return os.getpgrp()


# ---------------------------------------------------------------------------
# /proc PPID map & descendant walk
# ---------------------------------------------------------------------------


def read_ppid_map() -> dict[int, int]:
    """Parse ``/proc/*/stat`` and return ``{pid: ppid}`` for every live process.

    The ``comm`` field (field 2) may contain spaces and parentheses, so we
    parse by finding the *last* ``') '`` separator (``rsplit``) rather than
    splitting on whitespace naïvely.  Vanished or unreadable entries are
    silently skipped (the process exited between the ``glob`` and the
    ``read``).
    """
    ppid_map: dict[int, int] = {}
    proc = Path('/proc')
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        stat_path = entry / 'stat'
        try:
            raw = stat_path.read_text()
        except OSError:
            continue
        # Format: "pid (comm) state ppid ..."
        # rsplit on ') ' to skip over the comm field safely.
        try:
            right = raw.rsplit(') ', 1)[1]
            fields = right.split()
            pid = int(entry.name)
            ppid = int(fields[1])  # field index after stripping pid+comm+state
            ppid_map[pid] = ppid
        except (IndexError, ValueError):
            continue
    return ppid_map


def collect_descendants(root: int, ppid_map: dict[int, int]) -> set[int]:
    """Return the set of all transitive descendant pids of *root* (BFS, cycle-safe).

    *root* itself is NOT included in the result.  Processes not reachable from
    *root* (e.g. unrelated pid 1 and its subtree) are excluded.

    The walk is iterative with a ``visited`` set so a cyclic ``ppid_map``
    (e.g. from a test fixture) cannot loop forever.
    """
    # Build a children map for efficient BFS
    children: dict[int, list[int]] = {}
    for pid, ppid in ppid_map.items():
        children.setdefault(ppid, []).append(pid)

    visited: set[int] = set()
    queue: deque[int] = deque([root])
    while queue:
        current = queue.popleft()
        for child in children.get(current, []):
            if child not in visited:
                visited.add(child)
                queue.append(child)
    # Exclude root itself (it wasn't added to visited on entry)
    visited.discard(root)
    return visited


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def cancel_request(
    path: Path,
    *,
    ppid_map_provider=read_ppid_map,
    kill=os.kill,
    killpg=os.killpg,
    failed_pids_out: list | None = None,
) -> int:
    """Cancel the verify-merge identified by the pgid file at *path*.

    Algorithm
    ---------
    1. Read and parse the pgid file.

       * ``FileNotFoundError`` → nothing to cancel; return 0.
       * Non-integer content → corrupt file; remove it and return 0.
       * Parsed value ≤ 0 → nonsensical pgid (0 kills the caller's group;
         negatives are invalid); treat as corrupt, remove, return 0.

    2. **Snapshot first**: call *ppid_map_provider* to get the full ``/proc``
       PPID map *before* sending any signal.  Killing the root reparents
       survivors to init and severs the tree linkage — the snapshot must
       precede all kills.

    3. ``collect_descendants(pgid, ppid_map)`` to find every
       ``start_new_session`` escape (they remain descendants even after
       escaping the process group).

    4. ``SIGKILL`` every target (descendants ∪ {pgid}):

       * ``ProcessLookupError`` → already dead, counts as success.
       * ``PermissionError`` → live process we cannot kill; record failure
         and append the pid to *failed_pids_out* (if provided) so the
         caller can emit an actionable diagnostic.

    5. ``killpg(pgid, SIGKILL)`` backstop — suppresses ``OSError`` /
       ``ProcessLookupError`` (group may already be gone).

    6. If any live process could not be killed → return 1 and **retain** the
       pgid file (lets a retry or β's quarantine probe act on it).
       Otherwise remove the file and return 0.

    Parameters
    ----------
    failed_pids_out:
        Optional list to which pids that raised ``PermissionError`` on
        ``SIGKILL`` are appended.  Useful for callers that want to emit a
        human-readable diagnostic.  Pass ``[]`` (an empty list) and inspect
        it after the call.

    Return value
    ------------
    * ``0`` — all processes killed (or already gone), file removed.
    * ``1`` — at least one live process raised ``PermissionError``; file kept.
    """
    # Step 1: read pgid file
    try:
        raw = path.read_text().strip()
    except FileNotFoundError:
        return 0

    try:
        pgid = int(raw)
    except ValueError:
        # Corrupt / empty file — clean up and treat as nothing to do
        remove_pgid_file(path)
        return 0

    if pgid <= 0:
        # Parseable but nonsensical (0 would kill the caller's group; negatives are
        # invalid).  Treat as corrupt: remove and exit clean.
        remove_pgid_file(path)
        return 0

    # NOTE — stale-file / PID-reuse window:
    # If verify-merge dies without running its finally-block (hard crash, OOM,
    # host reset) the pgid file is left on disk.  After the original process is
    # gone the OS may recycle that pid/pgid for an unrelated process; a later
    # cancel_request call would then SIGKILL the wrong process.
    # The window is bounded: modern kernels recycle pids slowly (default
    # pid_max = 32768), and the stale file is cleaned up by the next successful
    # cancel-verify or on host reboot.  A future β hardening: stamp a boot-id
    # alongside the pgid (e.g. /proc/sys/kernel/random/boot_id) and validate
    # it here before killing.

    # Step 2: snapshot PPID map BEFORE any kills (invariant: snapshot precedes kill).
    # Killing the root reparents its survivors to init, severing the /proc parent
    # chain — the snapshot MUST capture the full tree first.
    ppid_map = ppid_map_provider()

    # Step 3: collect every descendant (start_new_session escapes included).
    # collect_descendants excludes root, so we union in {pgid} to kill it explicitly —
    # without this, the root would only be hit by the killpg backstop (step 5).
    descendants = collect_descendants(pgid, ppid_map)
    targets = descendants | {pgid}  # root always explicit in the kill set

    # Step 4: SIGKILL every target
    failed = False
    for pid in targets:
        try:
            kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # already dead — success
        except PermissionError:
            failed = True  # live but unkillable
            if failed_pids_out is not None:
                failed_pids_out.append(pid)

    # Step 5: killpg backstop (suppresses errors — group may already be gone)
    with contextlib.suppress(ProcessLookupError, OSError):
        killpg(pgid, signal.SIGKILL)

    # Step 6: decide outcome
    if failed:
        # Retain the file so a retry can act on it
        return 1
    remove_pgid_file(path)
    return 0


# ---------------------------------------------------------------------------
# Flock guard — laptop persistent-worktree verify span (task 2306 α)
#
# PRD: plans/laptop-warm-verify-flock-orphan-prd.md, Change A.  Serializes the
# laptop's persistent warm merge-verify worktree (git.persistent_merge_worktree)
# under a laptop-side exclusive fcntl.flock, since the per-host serial
# invariant that `GitOps._bump_host_verify_attempt_count` relies on is only
# supplied by `enforce_persistent_worktree_serial_lane` at WORKSTATION
# startup — not on the laptop.  On bounded-wait contention, the caller (cli.py
# verify-merge) emits a distinguished VerifyResult
# (verify_runner.make_flock_contention_result) instead of ever falling back to
# an ephemeral worktree.
# ---------------------------------------------------------------------------

#: Fixed filename (under worktree_base) for the laptop persistent-worktree
#: verify-span exclusive lock.
MERGE_VERIFY_LOCK_FILENAME: str = '.merge_verify.lock'


def merge_verify_lock_path(worktree_base: Path) -> Path:
    """Return the fixed lock-file path guarding the persistent merge-verify worktree."""
    return worktree_base / MERGE_VERIFY_LOCK_FILENAME


def acquire_merge_verify_flock(
    path: Path,
    timeout_secs: float,
    *,
    poll_interval: float = 0.1,
    monotonic=time.monotonic,
    sleep=time.sleep,
) -> int | None:
    """Acquire an exclusive ``fcntl.flock`` on *path* with a bounded wait.

    Polls ``fcntl.flock(fd, LOCK_EX | LOCK_NB)`` with short sleeps until
    either the lock is acquired or *timeout_secs* elapses.  This bounded-wait
    polling design avoids the fragility of a blocking ``LOCK_EX`` interrupted
    by ``SIGALRM``.

    Returns the open, locked fd on success — the caller must release it via
    :func:`release_merge_verify_flock`.  Returns ``None`` on timeout (the fd
    is closed before returning).

    *monotonic* / *sleep* are injectable (default ``time.monotonic`` /
    ``time.sleep``) so tests can run the bounded wait fast and deterministically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    deadline = monotonic() + timeout_secs
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except (BlockingIOError, OSError):
            # Lock held by another holder (EAGAIN/EACCES) — poll until deadline.
            pass
        if monotonic() >= deadline:
            os.close(fd)
            return None
        sleep(poll_interval)


def release_merge_verify_flock(fd: int) -> None:
    """Best-effort release of a flock fd acquired via :func:`acquire_merge_verify_flock`.

    Suppresses ``OSError`` from either the unlock or the close — release is
    inherently best-effort (mirrors :func:`remove_pgid_file`'s idempotent,
    exception-suppressing style).
    """
    with contextlib.suppress(OSError):
        fcntl.flock(fd, fcntl.LOCK_UN)
    with contextlib.suppress(OSError):
        os.close(fd)


# ---------------------------------------------------------------------------
# Holder-pgid rendezvous — fixed key (task 2306 α)
#
# A waiter (a verify-merge invocation that lost the bounded wait on the flock)
# cannot know the holder's per-dispatch --request-id, so the holder's pgid is
# recorded at a request-id-independent FIXED key rather than under the
# per-request pgid file.  Built directly on pgid_file/write_pgid_file/
# remove_pgid_file — same atomic write, same never-git-cleaned directory.
# ---------------------------------------------------------------------------

#: Fixed rendezvous key (under PGID_DIR_NAME) recording the pgid of whichever
#: verify-merge invocation currently holds the merge-verify flock.
LOCK_HOLDER_PGID_KEY: str = '_merge_verify_lock_holder'


def write_lock_holder_pgid(worktree_base: Path, pgid: int) -> None:
    """Record *pgid* as the current merge-verify flock holder."""
    write_pgid_file(pgid_file(worktree_base, LOCK_HOLDER_PGID_KEY), pgid)


def read_lock_holder_pgid(worktree_base: Path) -> int | None:
    """Return the recorded flock-holder pgid, or ``None`` if absent/corrupt.

    Fail-safe: a missing file (``FileNotFoundError``), unparseable content
    (``ValueError``), or any other read error (``OSError``) all yield
    ``None`` rather than raising — mirrors the ``_bump_host_verify_attempt_count``
    counter-read precedent.
    """
    path = pgid_file(worktree_base, LOCK_HOLDER_PGID_KEY)
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


def remove_lock_holder_pgid(worktree_base: Path) -> None:
    """Remove the recorded flock-holder pgid file; idempotent when absent."""
    remove_pgid_file(pgid_file(worktree_base, LOCK_HOLDER_PGID_KEY))


# ---------------------------------------------------------------------------
# Connection-death stdin heartbeat-watchdog (task 2308 γ)
#
# PRD: plans/laptop-warm-verify-flock-orphan-prd.md, Change B, §4 B1/B2 + §8.1.
# Ties a dispatched ``verify-merge --request-id`` process's lifetime to its
# ssh dispatch connection: when the VerifyResult becomes undeliverable (the
# orchestrator was killed, ssh dropped, or a hard network partition occurred)
# the remote terminates itself AND its whole build subtree instead of
# surviving as a setsid orphan (the exact failure mode ``cancel_request``
# above cleans up *after the fact*; this watchdog prevents it from ever being
# needed on the connection-death path).
#
# Protocol: the dispatcher (``verify_runner.RemoteRunner.run_merge_verify``)
# opens the ssh child with ``stdin=PIPE`` and writes a heartbeat newline down
# the channel every ``HEARTBEAT_INTERVAL_SECS`` for the full verify span. The
# remote (``orchestrator verify-merge --request-id``) spawns a watchdog
# thread owning fd 0 *before* the build starts (see :func:`start_stdin_watchdog`).
# It fires on stdin EOF (the channel was cleanly closed) or when no heartbeat
# arrives within ``WATCHDOG_HEARTBEAT_TIMEOUT_SECS`` (a hard partition, no
# clean close).  ``setsid`` + the pgid file are unchanged by this protocol —
# ``setsid`` does not close fd 0, so the watchdog reads stdin regardless of
# session, and ``cancel_request`` keeps tree-killing by pgid with zero
# contract churn.
# ---------------------------------------------------------------------------

#: Heartbeat cadence (seconds) the dispatcher writes down the ssh child's
#: stdin for the full verify span.  A module constant rather than a config
#: leaf (PRD §11 Q2: "small constant first"; promote later only if
#: field-tuning is needed).  Both hosts run the same ``df`` checkout, so the
#: dispatcher (which imports this one-way from here — see verify_runner.py)
#: and this watchdog always agree on the cadence.
HEARTBEAT_INTERVAL_SECS: float = 5.0

#: The watchdog fires if no heartbeat (or EOF) arrives within this window.
#: 2x the heartbeat cadence tolerates a single missed/delayed beat before
#: declaring the dispatch channel dead.
WATCHDOG_HEARTBEAT_TIMEOUT_SECS: float = 2 * HEARTBEAT_INTERVAL_SECS

#: Grace period between the SIGTERM and SIGKILL passes when the watchdog
#: fires and kills the build subtree (see :func:`fire_watchdog_kill`).
WATCHDOG_KILL_GRACE_SECS: float = 5.0


def run_stdin_watchdog(
    read_fd: int,
    on_fire,
    *,
    heartbeat_timeout: float = WATCHDOG_HEARTBEAT_TIMEOUT_SECS,
    select_fn=select.select,
    read_fn=os.read,
    read_size: int = 4096,
) -> None:
    """Blocking fd-0 trigger loop: calls *on_fire* once on stdin EOF or heartbeat starvation.

    Intended to run in a dedicated daemon thread (see :func:`start_stdin_watchdog`)
    with a *blocking* ``select`` loop — rather than an asyncio task sharing the
    build's event loop — so it fires even if that loop is saturated or wedged.

    Each iteration waits up to *heartbeat_timeout* seconds for *read_fd* to
    become readable:

    * Timeout (empty ready set) -> no heartbeat arrived in time (hard network
      partition) -> call *on_fire* and return.
    * Readable, but the read returns ``b''`` -> the writing end closed the
      channel (orchestrator killed or ssh dropped) -> call *on_fire* and
      return.
    * Readable with non-empty data -> a heartbeat byte arrived; the window
      resets and the loop continues watching.

    *on_fire* is called at most once — the function returns immediately
    afterward.  *select_fn* / *read_fn* are injectable (default
    ``select.select`` / ``os.read``) so tests can script deterministic fd-0
    behavior without a real pipe or wall-clock waits.
    """
    while True:
        ready, _, _ = select_fn([read_fd], [], [], heartbeat_timeout)
        if not ready:
            on_fire()
            return
        data = read_fn(read_fd, read_size)
        if data == b'':
            on_fire()
            return
        # Non-empty data: a heartbeat arrived -- window resets, keep watching.


def fire_watchdog_kill(
    pgid: int,
    *,
    grace_secs: float = WATCHDOG_KILL_GRACE_SECS,
    ppid_map_provider=read_ppid_map,
    kill=os.kill,
    killpg=os.killpg,
    sleep=time.sleep,
    exit_fn=os._exit,
    exit_code: int = 1,
) -> None:
    """Kill the build subtree rooted at *pgid* and unconditionally self-terminate.

    Invoked by the stdin watchdog (:func:`run_stdin_watchdog` via
    :func:`start_stdin_watchdog`) when the dispatch connection is judged dead.
    Mirrors :func:`cancel_request`'s snapshot -> /proc descendant walk ->
    signal approach (the same ``start_new_session`` escape problem applies:
    ``verify.py`` ``_run_cmd`` spawns every build command with
    ``start_new_session=True``, so cargo/rustc leave the verify-merge process
    group and a bare ``killpg(pgid)`` would strand them), with two
    differences required because *this* code runs **inside** the target
    process group rather than as a separate process:

    * It signals only **descendants** of *pgid* (``collect_descendants``
      already excludes the root) -- never *pgid* itself / the calling
      process.  A ``killpg(pgid, ...)`` here would signal the watchdog's own
      process before it could finish the SIGTERM -> grace -> SIGKILL
      escalation or reach a controlled exit, so *killpg* is accepted for
      signature symmetry with :func:`cancel_request` but is intentionally
      never called.
    * It ends by unconditionally calling ``exit_fn(exit_code)`` -- a
      controlled non-zero self-exit -- rather than returning, so the
      abandoned verify-merge leader always terminates (freeing its flock and
      letting sshd reap it) even if some descendant could not be killed.

    Sequence: snapshot the ``/proc`` PPID map, ``SIGTERM`` every descendant
    (``ProcessLookupError``/``PermissionError`` suppressed -- already dead or
    a permission race is fine, this is a best-effort escalation), sleep
    *grace_secs*, re-snapshot + ``SIGKILL`` every surviving descendant
    (same suppression), then ``exit_fn(exit_code)`` as the final action.
    """
    ppid_map = ppid_map_provider()
    descendants = collect_descendants(pgid, ppid_map)
    for pid in descendants:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            kill(pid, signal.SIGTERM)

    sleep(grace_secs)

    ppid_map = ppid_map_provider()
    survivors = collect_descendants(pgid, ppid_map)
    for pid in survivors:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            kill(pid, signal.SIGKILL)

    exit_fn(exit_code)
