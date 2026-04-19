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
Wherever a subprocess is created with ``start_new_session=True``, replace
bare ``proc.terminate()`` / ``proc.kill()`` cleanup with::

    await terminate_process_group(proc, grace_secs=5.0)

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
import os
import signal


async def terminate_process_group(
    proc: asyncio.subprocess.Process,
    *,
    grace_secs: float = 5.0,
) -> None:
    """Send SIGTERM to proc's entire process group, then SIGKILL if needed.

    Steps:
    1. If proc has already exited, return immediately (idempotent).
    2. Derive the process group ID via ``os.getpgid(proc.pid)``.  If the OS
       has already reaped the PID, return (race-safe).
    3. ``os.killpg(pgid, SIGTERM)`` — delivers to every member of the group.
    4. Wait up to *grace_secs* for ``proc`` to exit.
    5. If still alive after the grace period, escalate with
       ``os.killpg(pgid, SIGKILL)`` and wait another *grace_secs*.

    All ``os.killpg`` calls are wrapped in ``contextlib.suppress(ProcessLookupError,
    OSError)`` to handle the race where the group vanishes between the
    ``getpgid`` call and the ``killpg`` call.
    """
    if proc.returncode is not None:
        return
    if proc.pid is None:
        return

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        # Process already reaped by the OS.
        return

    # SIGTERM — ask the group to clean up.
    with contextlib.suppress(ProcessLookupError, OSError):
        os.killpg(pgid, signal.SIGTERM)

    try:
        await asyncio.wait_for(proc.wait(), grace_secs)
    except TimeoutError:
        # Still alive after grace period — force kill.
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(pgid, signal.SIGKILL)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), grace_secs)
