"""Tests for shared.pidfd, the thin Linux pidfd syscall layer, through its public functions."""
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from shared.pidfd import P_PIDFD, PIDFD_SIGNAL_PROCESS_GROUP, pidfd_open, pidfd_send_signal

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='pidfds are Linux-only')


def test_pidfd_open_of_a_pid_no_process_holds_raises_process_lookup_error() -> None:
    """Valid pids are strictly below pid_max, so no process can hold pid_max itself."""
    pid_max = int(Path('/proc/sys/kernel/pid_max').read_text())

    with pytest.raises(ProcessLookupError):
        pidfd_open(pid_max)


def test_a_pidfd_signals_its_own_live_child() -> None:
    child = subprocess.Popen(['sleep', '30'], start_new_session=True)
    fd = None
    try:
        fd = pidfd_open(child.pid)
        pidfd_send_signal(fd, signal.SIGKILL)

        assert child.wait(timeout=10) == -signal.SIGKILL
    finally:
        if fd is not None:
            os.close(fd)
        with contextlib.suppress(OSError):
            child.kill()
        child.wait(timeout=10)


def test_constants_carry_the_kernel_uapi_values() -> None:
    """include/uapi/linux/pidfd.h and include/uapi/linux/wait.h; the stdlib exports neither."""
    assert PIDFD_SIGNAL_PROCESS_GROUP == 4
    assert P_PIDFD == 3
