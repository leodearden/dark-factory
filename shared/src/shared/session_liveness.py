"""Judge whether a session record's pid is alive, by its identity when the record carries one.

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
    liveness,
)

__all__ = ['judge']

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
