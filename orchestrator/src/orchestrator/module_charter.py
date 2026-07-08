"""Sole orchestrator-side composition of Lock-charter Contract 1.

Contract 1: ``metadata.files`` is ALWAYS file-level; coarsening to depth-N
module lock keys happens at READ time only. This module is the single place
that composes the ``shared.locking`` primitives (``directory_locks``,
``strip_directory_locks``, ``files_to_modules``) into the two operations
every call site needs:

- ``derive_modules``: files -> depth-coarsened module lock keys (READ path).
- ``sanitize_files_for_persist``: files -> file-level-only files (WRITE path).

Both scheduler.py and harness.py route through this module instead of
re-implementing the strip -> coarsen pipeline inline (the pre-task-2122 state
had 4+ divergent copies of this logic).
"""

from __future__ import annotations

import logging

from shared.locking import directory_locks, files_to_modules, strip_directory_locks

logger = logging.getLogger(__name__)


def derive_modules(files: list, depth: int, *, task_id: str = '') -> list[str]:
    """Derive depth-coarsened module lock keys from *files*.

    α strip: remove directory-shaped entries before lock derivation.  A
    directory entry (no recognised file extension) would produce a
    subtree-wide prefix lock that blocks every task touching any file under
    that subtree (reify-3468).  Strip them so only real file siblings derive
    locks.  When ALL entries are directories the stripped list is empty ->
    ``files_to_modules`` returns ``[]`` -> callers fall through to their own
    fallback (e.g. the task-<id> synthetic lock in ``Scheduler._get_modules``).

    Emits one INFO diagnostic naming the stripped directory entries when any
    are found, so a run's logs can be grepped for pathologically wide
    charters.
    """
    dirs = directory_locks(files)
    if dirs:
        logger.info(
            'Task %s: α strip — rejected directory charter entries: %s',
            task_id,
            dirs,
        )
    return files_to_modules(strip_directory_locks(files), depth)


def sanitize_files_for_persist(files: list) -> list[str]:
    """Return only the file-level entries from *files*, for persisting to
    ``metadata.files``.

    Every ``metadata.files`` write path must call this before persisting so
    the field always holds genuine file-level paths (never a directory-shaped
    entry) — the incident class fixed by commit 54ec90fefc, where a
    directory entry written to ``metadata.files`` re-derived a subtree-wide
    lock on the next read.
    """
    return strip_directory_locks(files)
