"""Stamp a session record's process identities, and judge its pids alive by them.

``capture_for_record`` is the stamp side and ``judge`` the use side; on both,
an unreadable /proc never fails the caller.

Contract: plans/spawn-t3-lane-mcp-and-host-registry-prd.md §6.4 and D16.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from shared.process_identity import (
    PROC_ROOT,
    Liveness,
    ProcessIdentity,
    ProcessIdentityUnreadable,
    capture,
    liveness,
)

__all__ = ['capture_for_record', 'judge']

logger = logging.getLogger(__name__)


def judge(
    identity: ProcessIdentity | None,
    pid: int,
    *,
    pid_alive: Callable[[int], bool],
    proc_root: Path = PROC_ROOT,
) -> Liveness:
    if identity is None:
        return Liveness.ALIVE if pid_alive(pid) else Liveness.DEAD
    if identity.pid != pid:
        raise ValueError(f'identity names pid {identity.pid} but the record pid is {pid}')
    try:
        return liveness(identity, proc_root=proc_root)
    except ProcessIdentityUnreadable as exc:
        logger.warning('pid %d judged alive: %s', pid, exc)
        return Liveness.ALIVE


def capture_for_record(pid: int, *, proc_root: Path = PROC_ROOT) -> ProcessIdentity | None:
    """The identity a record stamps for *pid*: ``capture``'s answer, an unreadable /proc logged and read as None."""
    try:
        return capture(pid, proc_root=proc_root)
    except ProcessIdentityUnreadable as exc:
        logger.warning('pid %d stamped unqualified: %s', pid, exc)
        return None
