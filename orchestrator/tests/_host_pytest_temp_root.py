"""A stand-in for the host's default pytest temp root, seeded with a dead session's dirs, for contracts that a nested session leaves that root untouched (tasks 7142, 8844)."""
from __future__ import annotations

import getpass
import os
import time
from pathlib import Path

#: Older than pytest's 3-day LOCK_TIMEOUT, so pytest treats the lock as dead.
DEAD_LOCK_AGE_SECS = 5 * 24 * 60 * 60


def listing(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob('*'))


def seed_dead_session_dirs(host: Path) -> None:
    """``pytest-0`` holds a dead session's leftovers and lock; ``pytest-1..3``
    make it a cleanup candidate as soon as a new session adds ``pytest-4``."""
    user_root = host / f'pytest-of-{getpass.getuser()}'
    (user_root / 'pytest-0' / 'leftover').mkdir(parents=True)
    lock = user_root / 'pytest-0' / '.lock'
    lock.write_text('')
    dead = time.time() - DEAD_LOCK_AGE_SECS
    os.utime(lock, (dead, dead))
    for n in (1, 2, 3):
        (user_root / f'pytest-{n}').mkdir()
