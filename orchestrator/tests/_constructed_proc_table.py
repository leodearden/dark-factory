"""Hand-made ``/proc`` roots for index.lock holder scans (task 5229).

A holder scan's verdict on the real ``/proc`` depends on every other process
on the host, so tests that pin what the scan CONCLUDES build the process table
themselves.  Each entry carries what the scan reads: ``comm``, a ``cwd`` link,
``cmdline``, and a ``stat`` line whose start time (field 22, clock ticks after
boot) is placed against the table's own ``btime``.
"""
from __future__ import annotations

import os
from pathlib import Path

_BOOT_EPOCH = 1_700_000_000
_STAT_FIELD_COUNT = 52
_START_TICKS_FIELD = 22


def empty_proc_table(parent: Path, *, boot_epoch_known: bool = True) -> Path:
    """A proc root under *parent* holding no process; its ``stat`` names a btime unless told not to."""
    proc_root = parent / 'proc'
    proc_root.mkdir()
    btime_line = f'btime {_BOOT_EPOCH}\n' if boot_epoch_known else ''
    (proc_root / 'stat').write_text(f'cpu  1 2 3 4\n{btime_line}processes 1\n')
    return proc_root


def add_process(
    proc_root: Path,
    pid: int,
    *,
    comm: str,
    cwd: Path | None,
    command: str,
    started_at: float | None,
) -> Path:
    """One process in *proc_root*.

    ``cwd=None`` makes the cwd unreadable; ``started_at=None`` makes the start
    time unparseable (a non-decimal stat field 22).
    """
    entry = proc_root / str(pid)
    entry.mkdir()
    (entry / 'comm').write_text(f'{comm}\n')
    if cwd is None:
        (entry / 'cwd').write_text('')
    else:
        os.symlink(cwd.resolve(), entry / 'cwd')
    (entry / 'cmdline').write_bytes(b'\0'.join(arg.encode() for arg in command.split()) + b'\0')
    (entry / 'stat').write_text(_stat_line(pid, comm, started_at))
    return entry


def add_git_process(
    proc_root: Path,
    pid: int,
    *,
    cwd: Path | None,
    command: str,
    started_at: float | None,
) -> Path:
    return add_process(
        proc_root, pid, comm='git', cwd=cwd, command=command, started_at=started_at,
    )


def _stat_line(pid: int, comm: str, started_at: float | None) -> str:
    start_ticks = (
        '?' if started_at is None
        else str(round((started_at - _BOOT_EPOCH) * os.sysconf('SC_CLK_TCK')))
    )
    after_comm = ['S', '1', str(pid), str(pid)]
    after_comm += ['0'] * (_START_TICKS_FIELD - 3 - len(after_comm))
    after_comm.append(start_ticks)
    after_comm += ['0'] * (_STAT_FIELD_COUNT - 2 - len(after_comm))
    return f'{pid} ({comm}) {" ".join(after_comm)}\n'
