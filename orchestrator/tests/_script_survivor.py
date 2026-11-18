"""The already-exited-script survivor fixture shared by the before_done test files.

Imported by bare module name (``from _script_survivor import ...``), like
``_proc_supervision_doubles`` -- ``orchestrator/tests/`` has no
``__init__.py``.

One copy of the survivor script and its pid helpers, used by
``test_before_done_script.py`` (the teardown disposition at the
``run_script`` seam) and by ``test_deterministic_runner.py`` (the deploy
escalation text built from that disposition).
"""

from __future__ import annotations

import contextlib
import os
import signal
from pathlib import Path

# The already-exited-script timeout shape (task 4252): the script exits while a
# child it spawned keeps the merged stdout pipe open, so `terminate_process_tree`'s
# pid-recycling guard dispatches NO signal and that child is still alive when
# the test's assertions run — which is exactly what those tests assert.  Every such test must therefore reap it itself, or leak a live
# `sleep 30` past pytest exit (measured by the reviewer on the first of them).
SURVIVOR_SCRIPT = (
    '#!/bin/sh\n'
    'sleep 30 &\n'
    'echo $! > "$1"\n'
    'echo started\n'
    'exit 7\n'
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

