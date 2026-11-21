"""Stand in for an orchestrator's lock file under a test project root.

The path is the one ``orchestrator/src/orchestrator/harness.py::_acquire_project_lock``
holds. flock is per open file description, so a hold taken here on a separate
``open()`` conflicts with ``is_orchestrator_lock_held`` exactly as the
orchestrator's own hold does.
"""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from shared.project_layout import orchestrator_data_dir


def write_orchestrator_lock(project_root: Path, first_line: str) -> Path:
    lock_path = orchestrator_data_dir(project_root) / 'orchestrator.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(f'{first_line}\n' if first_line else '', encoding='utf-8')
    return lock_path


@contextmanager
def held_orchestrator_lock(project_root: Path, first_line: str = '') -> Iterator[Path]:
    """Hold ``LOCK_EX`` on the root's orchestrator lock for the ``with`` block."""
    lock_path = write_orchestrator_lock(project_root, first_line)
    holder = lock_path.open('r+b')
    try:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield lock_path
    finally:
        holder.close()
