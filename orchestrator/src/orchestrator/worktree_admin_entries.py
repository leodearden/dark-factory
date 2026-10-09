"""Reclaim git worktree admin entries that git itself will never prune.

``git worktree prune`` skips an entry whose ``locked`` marker exists, however
long its tree has been gone. Nothing in this repo locks a worktree (liveness
is the merge-verify flock), so a ``locked`` marker on an entry whose tree
under ``worktree_base`` is gone is git's abandoned ``initializing`` one from
an interrupted ``worktree add``. Such an entry hard-failed every
``merge_request`` enqueue (task 4828).

Reads git's documented admin layout (gitrepository-layout(5)): each
``<git_dir>/worktrees/<id>/`` holds a ``gitdir`` file naming the worktree's
``.git`` file, and a ``locked`` file while the entry is locked. Removing
``locked`` is exactly what ``git worktree unlock`` does, without a
subprocess.
"""
from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def unlock_dangling_locked_entries(
    git_dir: Path, worktree_base: Path, context: str,
) -> None:
    """Unlink ``locked`` from every entry whose ``worktree_base`` tree is gone.

    An entry whose worktree lives anywhere else keeps its lock: locked and
    absent is git's documented portable-device shape. A no-op when
    *git_dir* is not a directory (a ``.git``-file layout) or *worktree_base*
    is absent. Each unlock is logged at WARNING under *context*. Never
    raises ``OSError``.
    """
    if not git_dir.is_dir() or not worktree_base.is_dir():
        return
    try:
        entries = sorted((git_dir / 'worktrees').iterdir())
    except OSError:
        return
    for entry in entries:
        worktree = _dangling_locked_worktree(entry, worktree_base)
        if worktree is not None:
            _unlock(entry, worktree, context)


def _dangling_locked_worktree(admin_entry: Path, worktree_base: Path) -> Path | None:
    """The worktree of a LOCKED *admin_entry* whose tree is gone, else None.

    Qualifies only a direct child of *worktree_base*.
    """
    try:
        if not (admin_entry / 'locked').exists():
            return None
        gitdir = admin_entry / (admin_entry / 'gitdir').read_text().strip()
        if gitdir.name != '.git':
            return None
        worktree = gitdir.parent
        if worktree.parent.resolve() != worktree_base.resolve() or worktree.is_dir():
            return None
        return worktree
    except OSError:
        return None


def _unlock(admin_entry: Path, worktree: Path, context: str) -> None:
    lock = admin_entry / 'locked'
    reason = ''
    with contextlib.suppress(OSError):
        reason = lock.read_text().strip()
    try:
        os.unlink(lock)
    except OSError as exc:
        logger.warning(
            '%s: could not unlock worktree admin entry %s (%s): %s',
            context, admin_entry.name, worktree, exc,
        )
        return
    logger.warning(
        '%s: unlocked worktree admin entry %s for vanished %s '
        '(lock reason %r) so prune can reclaim it',
        context, admin_entry.name, worktree, reason,
    )
