"""Whether a pid read from untrusted input (a lock file, a JSON record, a directory name) names a live process.

A recycled pid reads as alive here; for a reuse-proof identity, use
shared/src/shared/process_identity.py.
"""

from __future__ import annotations

import os


def pid_alive(value: object) -> bool:
    """Return True iff *value* is a pid naming a live process; never raises.

    - Anything that is not a positive non-bool int is rejected before os.kill,
      because pid 0 and negative pids signal process groups.
    - Signal 0 succeeding, or PermissionError (the process is visible but
      cannot be signalled), means alive.
    - ProcessLookupError, any other OSError, or OverflowError (a value no C
      pid_t can hold) means dead.

    scripts/orchestrator-watchdog.py keeps a forced stdlib copy, pinned by
    tests/scripts/test_watchdog_pid_alive_parity.py.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, OverflowError):
        return False
    return True
