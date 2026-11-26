"""The already-exited-script survivor fixture shared by the before_done test files.

Imported by bare module name (``from _script_survivor import ...``), like
``_proc_supervision_doubles`` -- ``orchestrator/tests/`` has no
``__init__.py``.

One copy of the survivor script and its pid helpers, used by
``test_before_done_script.py`` (the teardown disposition at the
``run_script`` seam) and by ``test_deterministic_runner.py`` (the deploy
escalation text built from that disposition).  ``needs_group_pidfd_signals``
skips the tests that expect the survivor killed once its script is reaped.
"""

from __future__ import annotations

import contextlib
import os
import signal
from pathlib import Path

import pytest
from shared.testing_pidfd import group_pidfd_signals_unsupported_reason

# The already-exited-script timeout shape (task 4252): the script exits with 7
# only once its 4 MiB of output is being drained, while the child it
# backgrounded keeps the merged stdout pipe open.  Every consumer must still
# reap that child itself in a `finally`, so a failed teardown cannot leak a
# `sleep 30` past pytest exit.
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


def read_survivor_pid(pidfile: Path) -> int:
    """The pid ``SURVIVOR_SCRIPT`` recorded for the child that outlived it.

    Written milliseconds into the run, long before the 1s timeout fires, so it
    is reliably on disk by the time a caller reads it.
    """
    return int(pidfile.read_text().strip())


def is_alive(pid: int) -> bool:
    """Whether *pid* still exists — the signal-0 existence probe."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def kill_survivor(pidfile: Path) -> None:
    """SIGKILL the process ``SURVIVOR_SCRIPT`` left behind, if still there.

    Tolerates every way the pid can already be gone (no file, empty file, the
    process reaped in the meantime) so it is safe in a ``finally``.
    """
    with contextlib.suppress(OSError, ValueError):
        os.kill(read_survivor_pid(pidfile), signal.SIGKILL)

