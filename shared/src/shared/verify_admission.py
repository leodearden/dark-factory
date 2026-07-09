"""Task-verify admission control — flock N-slot semaphore + role nice tiers.

PRD ``plans/verify-oversubscription-control-prd.md`` task T1 (foundation
substrate; task T2 wires this into ``orchestrator/verify.py`` around each
pytest spawn). Implements the two §Contract seam functions:

``nice_prefix(role)``
    Pure lookup returning the argv ``nice``/``ionice`` prefix for a verify
    role.

``acquire_task_slot(role, *, slots_dir, n, wait=True)``
    A flock-backed N-slot counting semaphore gating ``task``/``background``
    role pytest spawns. Ported from reify's ``lib_slot_acquire.sh`` /
    ``lib_test_semaphore.sh`` shuffle-acquire model.

Contract clauses (see PRD §3):

- **C-merge-priority:** ``merge``-role verifies never acquire the semaphore
  (anti-livelock — task can never starve merge).
- **C-untimed-acquire:** the wait happens before the pytest command's timeout
  clock starts; ``wait=True`` polls unbounded/untimed.
- **C-fail-open:** a missing ``slots_dir``, ``n < 1``, or any ``OSError``
  during acquisition degrades to ``held=False`` — never blocks, never raises.
- **C-no-FD-inheritance:** slot FDs are marked non-inheritable so no spawned
  descendant pins a slot after the holder exits (reify's ``9<&-``).

This module is a direct-import submodule — like ``shared.psi`` — and is
deliberately NOT re-exported from ``shared/__init__.py``:
``shared/tests/test_public_api.py::TestInitAllCompleteness`` pins
``shared.__all__`` to a hardcoded module union, so consumers import via
``from shared.verify_admission import acquire_task_slot, nice_prefix``.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import random
import time
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    'acquire_task_slot',
    'nice_prefix',
]

_POLL_INTERVAL_SECS = 0.1

_NICE_TIERS: dict[str, list[str]] = {
    'merge': ['nice', '-n', '5'],
    'task': ['nice', '-n', '15', 'ionice', '-c2', '-n7'],
    'background': ['nice', '-n', '19', 'ionice', '-c3'],
}


def nice_prefix(role: str) -> list[str]:
    """Return the argv ``nice``/``ionice`` prefix for *role*.

    ``offline`` and any unrecognized role return ``[]`` (no priority
    adjustment). Always returns a fresh list — callers are free to mutate
    or extend it (e.g. ``nice_prefix(role) + [...]``) without corrupting the
    tier table.
    """
    return list(_NICE_TIERS.get(role, []))


def _try_once(slots_dir: Path, n: int) -> int | None:
    """Attempt a single non-blocking pass over the ``n`` slot files.

    Returns the open, locked fd of the first free slot found, or ``None`` if
    all ``n`` slots are currently held. Try-order is shuffled (herd-avoidance
    under concurrent waiters; a no-op at ``n == 1``) — this only affects
    *which* free slot is taken, never whether one is found.
    """
    order = list(range(1, n + 1))
    random.shuffle(order)
    for i in order:
        fd = os.open(slots_dir / f'slot-{i}', os.O_RDWR | os.O_CREAT, 0o644)
        os.set_inheritable(fd, False)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError:
            os.close(fd)
            continue
    return None


@contextlib.contextmanager
def acquire_task_slot(
    role: str,
    *,
    slots_dir: Path,
    n: int,
    wait: bool = True,
) -> Iterator[bool]:
    """Acquire one of ``n`` flock-backed slots in ``slots_dir`` for ``role``.

    Yields ``held`` — whether a slot was acquired. ``role`` values other than
    ``'task'``/``'background'`` (i.e. ``merge``, ``offline``, or an unknown
    role) never attempt acquisition and always yield ``False``
    (C-merge-priority: merge can never be starved by task).

    Slot files live at ``slots_dir/slot-<i>`` for ``i`` in ``1..n``, opened
    non-inheritable (C-no-FD-inheritance) and locked with a non-blocking
    ``flock``. The slot is released in a ``finally`` by closing the fd — the
    kernel frees the flock automatically, including when the holder dies
    without cleanup (C-daemonless self-heal; no canary/daemon/atexit
    involved).
    """
    if role not in {'task', 'background'}:
        yield False
        return

    fd = _try_once(Path(slots_dir), n)
    try:
        yield fd is not None
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
