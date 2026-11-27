"""The already-exited-script survivor fixture shared by the before_done test files.

Imported by bare module name (``from _script_survivor import ...``), like
``_proc_supervision_doubles`` -- ``orchestrator/tests/`` has no
``__init__.py``.

One copy of the survivor script and its pid and identity helpers, used by
``test_before_done_script.py`` (the teardown disposition at the
``run_script`` seam) and by ``test_deterministic_runner.py`` (the deploy
escalation text built from that disposition).  ``needs_group_pidfd_signals``
skips the tests that expect the survivor killed once its script is reaped.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import threading
import time
from pathlib import Path
from typing import TypeVar

import pytest
from shared.pidfd import pidfd_open, pidfd_send_signal
from shared.process_identity import (
    Liveness,
    ProcessIdentity,
    ProcessIdentityUnreadable,
    liveness,
)
from shared.testing_pidfd import group_pidfd_signals_unsupported_reason

# The already-exited-script timeout shape (task 4252): the script exits with 7
# only once its 4 MiB of output is being drained, while the child it
# backgrounded keeps the merged stdout pipe open.  Every consumer must still
# pass that child to `kill_survivor` in a `finally`, so a failed teardown
# cannot leak a `sleep 30` past pytest exit.
SURVIVOR_SCRIPT = (
    '#!/bin/sh\n'
    'sleep 30 &\n'
    'echo $! > "$1"\n'
    'head -c 4194304 /dev/zero\n'
    'exit 7\n'
)

_UNSUPPORTED_REASON = group_pidfd_signals_unsupported_reason()

needs_group_pidfd_signals = pytest.mark.skipif(
    _UNSUPPORTED_REASON is not None, reason=_UNSUPPORTED_REASON or '',
)

_RunResult = TypeVar('_RunResult')


def _wait_for_recorded_pid(
    pidfile: Path, abandon: threading.Event, *, within: float = 30.0,
) -> int | None:
    """Block until *pidfile* holds a complete pid line, or return None once *abandon* is set.

    Runs off-loop because the loop clock may be held.
    """
    deadline = time.monotonic() + within
    while not abandon.is_set():
        try:
            text = pidfile.read_text()
        except FileNotFoundError:
            text = ''
        if text.endswith('\n'):
            return int(text)
        if time.monotonic() >= deadline:
            raise AssertionError(
                f'{pidfile} never held a complete pid line within {within}s: '
                f'the script never recorded its child'
            )
        abandon.wait(0.01)
    return None


async def await_recorded_pid(pidfile: Path, run: asyncio.Future[_RunResult]) -> int:
    """The pid the script *run* spawned records for its child, or *run*'s own outcome if it ends first."""
    abandon = threading.Event()
    pid_wait = asyncio.ensure_future(
        asyncio.to_thread(_wait_for_recorded_pid, pidfile, abandon),
    )
    try:
        await asyncio.wait({run, pid_wait}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        abandon.set()
    pid = await pid_wait
    if pid is None or run.done():
        outcome = await run
        raise AssertionError(
            f'the run ended with {outcome!r} before the script recorded its child'
        )
    return pid


def kill_survivor(survivor: ProcessIdentity | None) -> None:
    """SIGKILL *survivor* only while it is still the process captured, so safe in a ``finally``.

    The pidfd pins whichever process holds the pid when it is opened, and the
    identity check after it proves that process is still *survivor*, so a pid
    recycled since the survivor died is never signalled (task 845).  None, a
    survivor never captured, is a no-op.
    """
    if survivor is None:
        return
    try:
        pidfd = pidfd_open(survivor.pid)
    except OSError:
        return
    try:
        with contextlib.suppress(OSError, ProcessIdentityUnreadable):
            if liveness(survivor) is Liveness.ALIVE:
                pidfd_send_signal(pidfd, signal.SIGKILL)
    finally:
        os.close(pidfd)
