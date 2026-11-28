"""Shared test kit: stand-in ``/proc`` trees for proc_group tests.

Consumers import by bare module name (``from _synthetic_proc import ...``):
``conftest.py`` prepends ``shared/tests`` to ``sys.path``.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def synthetic_stat_line(
    pid: int,
    pgrp: int,
    *,
    state: str = 'S',
    comm: str = 'weird (name) proc',
    cpu: tuple[int, int, int, int] = (0, 0, 0, 0),
    session: int | None = None,
    start_ticks: int = 0,
) -> str:
    """A real-format ``/proc/<pid>/stat`` for a fabricated /proc: ``pid (comm) state 1 pgrp session ...``.

    The default *comm* deliberately contains spaces AND a ``)`` so the parser's
    ``rfind(')')`` idiom stays pinned — a ``split()``-based parser would
    mis-read this line, and the kernel really does allow it (a process can
    set an arbitrary 15-char comm).  *cpu* is utime, stime, cutime and cstime,
    stat fields 14-17.  *session* is stat field 6, and defaults to *pgrp*.
    *start_ticks* is stat field 22.
    """
    counters = (
        ['0'] * 4 + [str(ticks) for ticks in cpu] + ['0'] * 4 + [str(start_ticks)] + ['0'] * 7
    )
    sid = pgrp if session is None else session
    return (
        f'{pid} ({comm}) {state} 1 {pgrp} {sid} 0 -1 4194304 '
        + ' '.join(counters)
        + '\n'
    )


def real_proc_view(view_dir: Path, pids: Iterable[int]) -> Path:
    """A /proc holding only *pids*, each linked to its real ``/proc/<pid>``, at *view_dir*.

    Also holds ``self``, which ``_RootViews`` reads for the scanner's own mount
    namespace. A scan pointed here reads real kernel-written content while
    walking only the named pids, so its cost no longer depends on the host's
    process count. *view_dir* must not exist yet.
    """
    view_dir.mkdir()
    for pid in pids:
        (view_dir / str(pid)).symlink_to(f'/proc/{pid}')
    (view_dir / 'self').symlink_to('/proc/self')
    return view_dir
