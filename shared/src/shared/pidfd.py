"""Linux pidfd syscalls, reached through libc's ``syscall(2)``.

The interpreters the factory runs (uv-managed CPython 3.13/3.14) export neither
``os.pidfd_open`` nor ``signal.pidfd_send_signal``, so this module calls the
kernel directly. Why a process group is signalled through a pidfd at all:
``shared/src/shared/proc_group.py::open_leader_pidfd``.

Every failure is an errno-typed ``OSError`` (``ProcessLookupError`` for ESRCH),
and ENOSYS wherever the syscall numbers below do not apply.
"""
from __future__ import annotations

import ctypes
import errno
import functools
import os
import platform
import sys
from collections.abc import Callable

#: The ``os.waitid`` idtype naming a process by pidfd (include/uapi/linux/wait.h).
P_PIDFD = 3
#: ``pidfd_send_signal`` flag: signal the pidfd's whole process group (Linux 6.9+).
PIDFD_SIGNAL_PROCESS_GROUP = 4

# Unified across Linux architectures, except alpha and mips, whose tables are offset.
_NR_PIDFD_SEND_SIGNAL = 424
_NR_PIDFD_OPEN = 434


@functools.cache
def _libc_syscall() -> Callable[..., int] | None:
    """libc's ``syscall``, or None where the numbers above do not apply."""
    if sys.platform != 'linux' or platform.machine().startswith(('alpha', 'mips')):
        return None
    try:
        syscall = ctypes.CDLL(None, use_errno=True).syscall
    except (OSError, AttributeError):
        return None
    syscall.restype = ctypes.c_long
    return syscall


def _syscall(number: int, *args: object) -> int:
    syscall = _libc_syscall()
    if syscall is None:
        raise OSError(errno.ENOSYS, os.strerror(errno.ENOSYS))
    result = syscall(ctypes.c_long(number), *args)
    if result == -1:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return result


def pidfd_open(pid: int) -> int:
    """A close-on-exec pidfd naming *pid*; the caller owns closing it."""
    return _syscall(_NR_PIDFD_OPEN, ctypes.c_int(pid), ctypes.c_uint(0))


def pidfd_send_signal(pidfd: int, sig: int, *, flags: int = 0) -> None:
    """Send *sig* to the process *pidfd* names, or to its group under PIDFD_SIGNAL_PROCESS_GROUP."""
    _syscall(
        _NR_PIDFD_SEND_SIGNAL,
        ctypes.c_int(pidfd),
        ctypes.c_int(sig),
        ctypes.c_void_p(None),
        ctypes.c_uint(flags),
    )
