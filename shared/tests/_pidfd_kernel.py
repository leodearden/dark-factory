"""Shared test kit: skip where the kernel cannot signal a process group through a pidfd.

``PIDFD_SIGNAL_PROCESS_GROUP`` arrived in Linux 6.9. Before it the group send
fails with EINVAL, ``shared.proc_group.sigkill_process_group`` falls back to its
returncode-gated killpg, and a reaped leader's surviving group goes unsignalled:
the residual ``shared.git_async``'s module docstring declares. A test proving
that group still dies can only pass on a kernel with the flag.

Consumers import by bare module name (``from _pidfd_kernel import ...``):
``conftest.py`` prepends ``shared/tests`` to ``sys.path``.
"""
from __future__ import annotations

import os

import pytest

from shared.pidfd import PIDFD_SIGNAL_PROCESS_GROUP, pidfd_open, pidfd_send_signal


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


needs_group_pidfd_signals = pytest.mark.skipif(
    not _kernel_signals_groups_through_pidfds(),
    reason='the kernel predates PIDFD_SIGNAL_PROCESS_GROUP (Linux 6.9)',
)
