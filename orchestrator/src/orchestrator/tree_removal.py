"""Recursive deletion that never runs on the event loop thread.

A synchronous recursive delete of a build tree on the orchestrator's one event
loop stalled the in-process escalation MCP for minutes (task 7366; INV-8 in
docs/legibility/design-invariants.md). Here an ``rm -rf`` child does the work,
so no loop thread or thread-pool worker is held, and ``run_git`` kills the
child's process group if the await is cancelled.
"""
from __future__ import annotations

import logging
import os
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
    :class:`TreeRemovalError`, and an ``rm`` that cannot be spawned raises its
    ``OSError``; with *ignore_errors* either one logs a WARNING instead, as
    ``shutil.rmtree(ignore_errors=True)`` never raises.
    """
    if not os.path.lexists(path):
        return
    try:
        # bounded=False: a multi-minute rm must not hold one of the short-probe
        # spawn slots (shared/src/shared/git_async.py, "who should opt out").
        result = await run_git(['rm', '-rf', '--', str(path)], None, bounded=False)
    except OSError as err:
        if not ignore_errors:
            raise
        _warn_left_in_place(path, err)
        return
    if result.ok:
        return
    failure = TreeRemovalError(path, result.returncode, result.stderr)
    if not ignore_errors:
        raise failure
    _warn_left_in_place(path, failure)


def _warn_left_in_place(path: Path, err: OSError) -> None:
    logger.warning('remove_tree: leaving %s in place: %s', path, err)
