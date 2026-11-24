"""index.lock files written as if some seconds ago (task 5229).

A holder scan is bounded by the lock's mtime, and the stand-off's verdict by
the lock's first-observed age, so tests set both by backdating the lock rather
than by waiting.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

# The reify incident's lock: a healthy commit 55 minutes into its pre-commit hook.
REIFY_LOCK_AGE_S = 3290.0


def index_lock_written_ago(work_tree: Path, seconds: float) -> Path:
    """``<work_tree>/.git/index.lock``, created with an mtime *seconds* in the past."""
    lock = work_tree / '.git' / 'index.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text('')
    written_at = time.time() - seconds
    os.utime(lock, (written_at, written_at))
    return lock
