"""Whether this kernel can signal a process group through a pidfd, as a test-skip reason.

``PIDFD_SIGNAL_PROCESS_GROUP`` arrived in Linux 6.9; why groups are signalled
through a pidfd at all is in ``shared/src/shared/proc_group.py::open_leader_pidfd``.
Both ``shared/tests`` and ``orchestrator/tests`` skip on it, and the
orchestrator rootdir cannot import ``shared/tests``, so the probe ships here and
each suite builds its own ``skipif`` from the reason. Pytest-free, like every
``shared.testing_*`` module.
"""

from __future__ import annotations

import functools
import os

from shared.pidfd import PIDFD_SIGNAL_PROCESS_GROUP, pidfd_open, pidfd_send_signal


@functools.cache
def group_pidfd_signals_unsupported_reason() -> str | None:
    """None when the kernel can SIGKILL a group through a pidfd, else the skip reason."""
    if _kernel_signals_groups_through_pidfds():
        return None
    return 'the kernel predates PIDFD_SIGNAL_PROCESS_GROUP (Linux 6.9)'


def _kernel_signals_groups_through_pidfds() -> bool:
    """Signal 0 to our own pidfd's group: EINVAL before Linux 6.9, success or ESRCH after."""
    try:
        fd = pidfd_open(os.getpid())
    except OSError:
        return False
    try:
        pidfd_send_signal(fd, 0, flags=PIDFD_SIGNAL_PROCESS_GROUP)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    finally:
        os.close(fd)
    return True
