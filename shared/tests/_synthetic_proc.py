"""Shared test kit: lines for a fabricated ``/proc``.

Consumers import by bare module name (``from _synthetic_proc import ...``):
``conftest.py`` prepends ``shared/tests`` to ``sys.path``.
"""
from __future__ import annotations


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
