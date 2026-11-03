"""Recursive deletion that never runs on the event loop thread.

A synchronous recursive delete of a build tree on the orchestrator's one event
loop stalled the in-process escalation MCP for minutes (task 7366; INV-8 in
docs/legibility/design-invariants.md). Here an ``rm -rf`` child does the work,
so no loop thread or thread-pool worker is held, and ``run_git`` kills the
child's process group if the await is cancelled.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path

from shared.git_async import run_git

logger = logging.getLogger(__name__)

_STDERR_TAIL_CHARS = 2000


class TreeRemovalError(OSError):
    """``rm -rf`` exited non-zero for *path*."""

    def __init__(self, path: Path, returncode: int, stderr: str) -> None:
        self.path = path
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f'rm -rf {path} failed (rc={returncode}): {stderr[-_STDERR_TAIL_CHARS:]}',
        )


async def remove_tree(path: Path, *, ignore_errors: bool = False) -> None:
    """Ensure *path* is absent, never following symlinks inside it.

    An absent *path* is already removed. A refused removal raises
    :class:`TreeRemovalError`; with *ignore_errors* it logs a WARNING instead,
    as ``shutil.rmtree(ignore_errors=True)`` never raises. When ``rm`` cannot
    be spawned, ``shutil.rmtree`` does the delete on a worker thread.
    """
    if not os.path.lexists(path):
        return
    try:
        # bounded=False: a multi-minute rm must not hold one of the short-probe
        # spawn slots (shared/src/shared/git_async.py, "who should opt out").
        result = await run_git(['rm', '-rf', '--', str(path)], None, bounded=False)
    except OSError as spawn_err:
        await _remove_on_a_worker_thread(path, spawn_err, ignore_errors=ignore_errors)
        return
    if result.ok:
        return
    failure = TreeRemovalError(path, result.returncode, result.stderr)
    if not ignore_errors:
        raise failure
    _warn_left_in_place(path, failure)


async def _remove_on_a_worker_thread(
    path: Path, spawn_err: OSError, *, ignore_errors: bool,
) -> None:
    """Delete *path* in-process because ``rm`` could not be spawned.

    A spawn fails under process, memory or fd exhaustion (EAGAIN, ENOMEM,
    EMFILE), which is when the crash-safe cleanups need to work, and
    ``shutil.rmtree`` needs no new process. The path is rare enough to justify
    holding a thread-pool worker.
    """
    await asyncio.to_thread(shutil.rmtree, path, ignore_errors=ignore_errors)
    if os.path.lexists(path):
        _warn_left_in_place(path, f'rm could not be spawned ({spawn_err}) and shutil.rmtree left it')
    else:
        logger.warning(
            'remove_tree: rm could not be spawned (%s); removed %s with shutil.rmtree',
            spawn_err, path,
        )


def _warn_left_in_place(path: Path, err: OSError | str) -> None:
    logger.warning('remove_tree: leaving %s in place: %s', path, err)
