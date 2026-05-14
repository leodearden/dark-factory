"""Subprocess helper for ``test_sqlite_task_backend_crash``.

Opens a :class:`SqliteTaskBackend` against ``argv[1]`` (project_root), writes
``argv[2]`` (int) committed task rows, appends each returned id to
``argv[3]`` (journal_path) as it returns from commit, then SIGKILLs itself.

The parent test process reads the journal back and asserts every journaled
id is present in the DB after reopening the backend with a fresh process.

Run as: ``python _sqlite_chaos_child.py <project_root> <n_commits> <journal>``
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig


async def _run(project_root: str, n_commits: int, journal_path: str) -> None:
    cfg = TaskmasterConfig(project_root=project_root)
    backend = SqliteTaskBackend(cfg)
    await backend.start()
    journal = Path(journal_path)
    for i in range(n_commits):
        dto = await backend.add_task(
            project_root=project_root,
            title=f'chaos-{i}',
            description=f'chaos row #{i}',
        )
        # fsync the journal so the parent has an authoritative record of
        # what the child claims it committed before suiciding.
        with journal.open('a') as fh:
            fh.write(dto['id'] + '\n')
            fh.flush()
            os.fsync(fh.fileno())


def main() -> None:
    project_root, n_commits, journal_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    try:
        asyncio.run(_run(project_root, n_commits, journal_path))
    finally:
        # Suicide regardless of outcome so the parent's assertion runs against
        # the post-kill DB state. SIGKILL bypasses Python's atexit / aiosqlite
        # finalization — that's the point.
        os.kill(os.getpid(), signal.SIGKILL)


if __name__ == '__main__':
    main()
