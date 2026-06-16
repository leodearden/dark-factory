"""Process-group signal helpers for clean subprocess tree termination.

When the orchestrator spawns a bash shell that in turn spawns cargo, which
spawns rustc worker threads, a plain ``proc.terminate()`` / ``proc.kill()``
only signals the immediate bash PID.  The cargo and rustc grandchildren
remain alive, keeping files open and consuming CPU/memory after the parent
has nominally "cleaned up".

``terminate_process_group`` fixes this by using ``os.killpg`` to deliver
the signal to every process in the group at once, provided the child was
spawned with ``start_new_session=True`` (which makes it the leader of a
fresh process group).

Usage
-----
The pgid MUST be captured by the caller immediately after spawn::

    proc = await asyncio.create_subprocess_exec(
        *cmd, ..., start_new_session=True,
    )
    pgid = proc.pid  # start_new_session guarantees pgid == pid at spawn

    try:
        await proc.communicate(...)
    except TimeoutError:
        await terminate_process_group(proc, pgid, grace_secs=5.0)

Why the caller captures pgid instead of the helper calling
``os.getpgid(proc.pid)``:
- Once ``proc`` has been reaped (``proc.returncode is not None``), the kernel
  is free to reuse that PID for an unrelated process.  ``os.getpgid`` on a
  reused PID returns the *new* owner's group — which has, in practice,
  ended up being the user ``systemd --user`` manager's group and killed the
  user's entire login session (see root cause for task 845).
- Capturing ``pgid`` at spawn eliminates that TOCTOU entirely.  After spawn,
  ``start_new_session=True`` guarantees the child is the leader of its own
  group with ``pgid == pid``.  The captured int is frozen; it is never
  refreshed from a possibly-reaped PID.

Safety checks
-------------
Even with a correctly-captured pgid, ``terminate_process_group`` refuses to
``killpg`` any of the following as defence-in-depth:

- ``pgid <= 1`` (init or invalid)
- ``pgid == os.getpid()`` (ourselves)
- ``pgid == os.getppid()`` (our parent)
- ``pgid == os.getpgrp()`` (our own process group — hitting this would kill
  our own orchestrator/tests)
- ``pgid != proc.pid`` (mismatch — a caller corrupted the capture, or the
  ``proc`` object was somehow swapped)

If any check fires, the helper logs an error and returns without signalling.

Limitations
-----------
If a grandchild calls ``setsid()`` on its own it will escape into a new
session and process group, making it invisible to ``killpg``.  This is
acceptable for the current codebase because cargo/rustc do not call
``setsid``.  Git sub-processes spawned by short-lived helpers are out of
scope (they are bounded and never appear in stuck-process incidents).
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
from pathlib import Path

logger = logging.getLogger(__name__)


def snapshot_process_group(pgid: int) -> str:
    """Return a human-readable snapshot of all processes in process group *pgid*.

    Walks ``/proc`` to find processes whose process-group id matches *pgid*,
    then formats one line per process with pid, ppid, state, wchan (the kernel
    function the process is currently blocked in), comm (executable name, capped
    at 15 chars by the kernel), and cmdline (full argv from ``/proc/<pid>/cmdline``,
    truncated to 200 chars — identifies the culprit process by name and arguments
    rather than just the short comm).

    Designed to be called from ``_run_subprocess``'s ``TimeoutError`` handler
    **before** ``proc.terminate()`` / ``terminate_process_group`` — at that
    point the wedged child processes are still alive and their ``/proc`` entries
    are readable.  After the kill the group may vanish mid-iteration; all I/O
    is wrapped in try/except so this function *never* raises.

    Returns a non-empty diagnostic string (with a header row) when at least one
    process belongs to *pgid*, or a benign "no processes found" note otherwise.

    Linux-specific: depends on ``/proc/<pid>/stat``, ``/proc/<pid>/wchan``,
    and ``/proc/<pid>/comm``.  The whole ``proc_group`` module already relies on
    Linux semantics (``os.killpg`` / ``os.getpgrp``), so this is acceptable.
    """
    try:
        return _snapshot_process_group_unsafe(pgid)
    except Exception:
        # Belt-and-suspenders: if the outer try fails (unusual /proc layout,
        # permissions, or unexpected exception), return a diagnostic string
        # rather than propagating.
        return f'snapshot_process_group({pgid}): unexpected error — see logs'


def _snapshot_process_group_unsafe(pgid: int) -> str:
    """Implement snapshot_process_group; may raise — caller wraps in try/except."""
    if pgid <= 0:
        return f'snapshot_process_group({pgid}): pgid <= 0 — no snapshot taken'

    proc_dir = Path('/proc')
    if not proc_dir.exists():
        return f'snapshot_process_group({pgid}): /proc not available'

    rows: list[str] = []
    try:
        entries = list(proc_dir.iterdir())
    except OSError:
        return f'snapshot_process_group({pgid}): could not list /proc'

    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)

        # Read /proc/<pid>/stat to get pgrp (field 5, 1-indexed), ppid (4), state (3).
        # stat format: "pid (comm) state ppid pgrp ..."
        try:
            stat_text = (entry / 'stat').read_text()
        except OSError:
            continue

        # Parse: find the closing ')' of the comm field to handle spaces/parens in names.
        try:
            rparen = stat_text.rfind(')')
            if rparen < 0:
                continue
            tail = stat_text[rparen + 2:]  # skip ') '
            fields = tail.split()
            # fields[0]=state, [1]=ppid, [2]=pgrp, [3]=session, ...
            state = fields[0]
            ppid = int(fields[1])
            pgrp = int(fields[2])
        except (IndexError, ValueError):
            continue

        if pgrp != pgid:
            continue

        # Read comm (short executable name, capped at 15 chars by the kernel).
        try:
            comm = (entry / 'comm').read_text().strip()
        except OSError:
            comm = '?'

        # Read wchan (kernel function the task is blocked in, or '0' when running).
        try:
            wchan = (entry / 'wchan').read_text().strip()
        except OSError:
            wchan = '?'

        # Read /proc/<pid>/cmdline (NUL-separated argv → spaces) for the full
        # command-line.  Kernel threads have an empty cmdline; fall back to
        # comm so the field is always populated.  Truncate to ~200 chars for
        # log friendliness.  Mirrors the comm/wchan try/except idiom so
        # snapshot_process_group never raises (module invariant).
        try:
            raw = (entry / 'cmdline').read_bytes()
            cmdline = raw.replace(b'\x00', b' ').decode('utf-8', 'replace').strip()
            if not cmdline:
                cmdline = comm  # kernel thread — fall back to short comm
            if len(cmdline) > 200:
                cmdline = cmdline[:200] + '…'
        except OSError:
            cmdline = '?'

        rows.append(
            f'  pid={pid} ppid={ppid} state={state} wchan={wchan} comm={comm}'
            f' cmdline={cmdline}'
        )

    if not rows:
        return f'snapshot_process_group({pgid}): no processes found in group'

    header = f'snapshot_process_group({pgid}): {len(rows)} process(es) in group:'
    return '\n'.join([header] + rows)


def _unsafe_pgid_reason(pgid: int, proc_pid: int | None) -> str | None:
    """Return a reason string if *pgid* is unsafe to killpg, else ``None``.

    Applied as defence-in-depth: even if a caller (or a PID-reuse race) hands
    us a pgid that targets the user session or ourselves, we refuse.
    """
    if pgid <= 1:
        return f'pgid <= 1 ({pgid!r})'
    if pgid == os.getpid():
        return f'pgid == os.getpid() ({pgid})'
    try:
        ppid = os.getppid()
    except OSError:
        ppid = None
    if ppid is not None and pgid == ppid:
        return f'pgid == os.getppid() ({pgid})'
    try:
        own_pgrp = os.getpgrp()
    except OSError:
        own_pgrp = None
    if own_pgrp is not None and pgid == own_pgrp:
        return f'pgid == os.getpgrp() ({pgid})'
    if proc_pid is not None and pgid != proc_pid:
        return f'pgid ({pgid}) != proc.pid ({proc_pid}) — captured value corrupted'
    return None


async def terminate_process_group(
    proc: asyncio.subprocess.Process,
    pgid: int,
    *,
    grace_secs: float = 5.0,
) -> None:
    """Send SIGTERM to *pgid*, then SIGKILL if the group outlives *grace_secs*.

    *pgid* must be the process-group id captured immediately after spawning
    *proc* with ``start_new_session=True`` (at which point ``pgid == proc.pid``
    by POSIX guarantee).  Passing a value fetched via ``os.getpgid(proc.pid)``
    after *proc* may have been reaped is unsafe — see module docstring.

    Behaviour:
    1. If *proc* has already been reaped (``returncode is not None``), return
       immediately.  The group is already gone with the leader.
    2. Sanity-check *pgid* via :func:`_unsafe_pgid_reason`.  If unsafe, log
       and return without signalling.
    3. ``os.killpg(pgid, SIGTERM)``.  Wait up to *grace_secs* for *proc* to
       exit.
    4. If *proc* is still alive, ``os.killpg(pgid, SIGKILL)`` and wait
       another *grace_secs*.

    All ``killpg`` calls are wrapped in ``contextlib.suppress(ProcessLookupError,
    OSError)`` because the group may vanish between our liveness check and
    the signal dispatch.
    """
    if proc.returncode is not None:
        # Already reaped — the entire group has exited along with the leader.
        return

    reason = _unsafe_pgid_reason(pgid, proc.pid)
    if reason is not None:
        logger.error(
            'terminate_process_group: refusing to killpg — %s. '
            'This indicates a bug in the caller; proc will NOT be signalled.',
            reason,
        )
        return

    with contextlib.suppress(ProcessLookupError, OSError):
        os.killpg(pgid, signal.SIGTERM)

    try:
        await asyncio.wait_for(proc.wait(), grace_secs)
    except TimeoutError:
        # Re-check liveness before escalating.
        if proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(pgid, signal.SIGKILL)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), grace_secs)
