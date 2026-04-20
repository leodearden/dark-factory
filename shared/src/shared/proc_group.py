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

logger = logging.getLogger(__name__)


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
