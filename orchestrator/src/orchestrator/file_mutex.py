"""A reentrant exclusive lock honoured across threads and processes.

``fcntl.flock`` locks belong to an open file description, so every OUTERMOST
acquisition opens the lock file afresh: two threads of one process then hold two
descriptions and exclude each other exactly as two processes do. A nested
acquisition on the thread that already holds the lock only counts depth.

Reentrancy is per THREAD. Two coroutines on one event-loop thread would read as
reentrant, so a critical section must never ``await``.

The wait is bounded: a lock not acquired within *timeout* raises
:class:`FileMutexTimeout`, so a deadlock fails loudly instead of hanging.
"""
from __future__ import annotations

import fcntl
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_POLL_INTERVAL_SECS = 0.01


class FileMutexTimeout(TimeoutError):
    def __init__(self, lock_path: Path, timeout: float) -> None:
        self.lock_path = lock_path
        self.timeout = timeout
        super().__init__(
            f'could not lock {lock_path} within {timeout}s: '
            'another thread or process holds it'
        )


class _ThreadDepths(threading.local):
    def __init__(self) -> None:
        self.by_path: dict[str, int] = {}


_depths = _ThreadDepths()


@contextmanager
def exclusive(lock_path: Path, *, timeout: float) -> Iterator[None]:
    """Hold *lock_path* exclusively; the file is created, its directory never is."""
    depths = _depths.by_path
    key = os.fspath(lock_path)
    if key in depths:
        depths[key] += 1
        try:
            yield
        finally:
            depths[key] -= 1
        return
    with open(lock_path, 'ab') as handle:
        _acquire(handle.fileno(), lock_path, timeout)
        depths[key] = 1
        try:
            yield
        finally:
            del depths[key]
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _acquire(fd: int, lock_path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise FileMutexTimeout(lock_path, timeout) from None
        time.sleep(_POLL_INTERVAL_SECS)
