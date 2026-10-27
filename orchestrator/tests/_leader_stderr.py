"""What an exited verify-merge leader left on its stderr, and why it self-killed.

A leader whose stdin watchdog fires has no structured return path left after
``os._exit``, so the one ``watchdog_fire_trigger=<trigger>`` line on its
stderr is the only record of which branch fired.  :func:`drain_exited_stderr`
reads that pipe without hanging on a helper that still holds its write end,
and :func:`assert_watchdog_fire_trigger` checks the line wherever it ends up:
the leader's own stderr, the ssh-relayed reason, or a log record.
"""
from __future__ import annotations

import os
import select
import subprocess
import time

from orchestrator.verify_cancel import WATCHDOG_FIRE_TRIGGER_TOKEN, WatchdogTrigger

#: How long :func:`drain_exited_stderr` keeps reading before it gives up on EOF.
STDERR_DRAIN_CEILING_SECS: float = 2.0


def drain_exited_stderr(
    proc: subprocess.Popen, *, ceiling_secs: float = STDERR_DRAIN_CEILING_SECS
) -> bytes:
    """Return the raw bytes still unread on an EXITED *proc*'s stderr pipe.

    Reads until EOF or *ceiling_secs*, whichever comes first.  A buffered
    ``.read()`` is wrong here because it blocks until EOF: ``poll()`` reports
    only *proc*'s own exit, and a helper it spawned with inherited fds can
    still hold the pipe's write end open.  A live *proc* is refused, because
    any snapshot of a still-running writer is racy.
    """
    if proc.poll() is None:
        raise AssertionError(f'pid {proc.pid} has not exited; its stderr is still being written')
    if proc.stderr is None:
        raise AssertionError(f'pid {proc.pid} was not spawned with stderr=PIPE')
    chunks = []
    read_deadline = time.monotonic() + ceiling_secs
    fd = proc.stderr.fileno()
    while True:
        remaining = read_deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            break
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    return b''.join(chunks)


def watchdog_fire_trigger_report(trigger: WatchdogTrigger) -> str:
    """The report a self-killed leader writes for *trigger*, without its newline.

    Spelled once on this side of the wire.  The emitter,
    ``orchestrator/verify_cancel.py::fire_watchdog_kill``, still formats its
    own copy inline.
    """
    return f'{WATCHDOG_FIRE_TRIGGER_TOKEN}={trigger.value}'


def assert_watchdog_fire_trigger(text: str, expected: WatchdogTrigger, *, source: str) -> None:
    """Assert *text* reports exactly the *expected* watchdog trigger and no other.

    *source* names where *text* came from (a leader's stderr, a
    RunnerUnavailable reason, a log record) in the failure message.
    """
    reported = sorted(t.value for t in WatchdogTrigger if watchdog_fire_trigger_report(t) in text)
    assert reported == [expected.value], (
        f'{source}: expected exactly {watchdog_fire_trigger_report(expected)}, '
        f'reported {reported or "no watchdog trigger (self-kill unattributable)"}; '
        f'tail: {text[-2000:]!r}'
    )
