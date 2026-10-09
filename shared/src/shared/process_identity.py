"""Identify a process so that pid reuse and reboots read as its death.

Contract: plans/deferral-flip-condition-prd.md §5.2 and decision 6.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, NamedTuple

__all__ = [
    'PROC_ROOT',
    'Liveness',
    'ProcessIdentity',
    'ProcessIdentityUnreadable',
    'capture',
    'command_name',
    'liveness',
]

PROC_ROOT: Final = Path('/proc')
_EXITED_STATES: Final = frozenset({'Z', 'X'})


class Liveness(StrEnum):
    ALIVE = 'ALIVE'
    DEAD = 'DEAD'
    OTHER_HOST = 'OTHER_HOST'


class ProcessIdentityUnreadable(Exception):
    """/proc could not answer; never evidence that the process is dead."""

    def __init__(self, path: Path, cause: str) -> None:
        super().__init__(f'cannot read {path}: {cause}')
        self.path = path
        self.cause = cause


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    boot_id: str
    host: str

    def __post_init__(self) -> None:
        if self.pid <= 0:
            raise ValueError(f'ProcessIdentity.pid must be > 0, got {self.pid!r}')
        if self.start_ticks < 0:
            raise ValueError(f'ProcessIdentity.start_ticks must be >= 0, got {self.start_ticks!r}')
        if not self.boot_id:
            raise ValueError(f'ProcessIdentity.boot_id must be non-empty, got {self.boot_id!r}')
        if not self.host:
            raise ValueError(f'ProcessIdentity.host must be non-empty, got {self.host!r}')


class _Stat(NamedTuple):
    state: str
    start_ticks: int


def _read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='replace')


def _boot_id(proc_root: Path) -> str:
    return _read_text(proc_root / 'sys' / 'kernel' / 'random' / 'boot_id').strip()


def _assert_can_see_other_processes(proc_root: Path) -> None:
    path = proc_root / '1'
    try:
        os.stat(path)
    except OSError as exc:
        raise ProcessIdentityUnreadable(
            path, f'pid 1 is invisible ({exc.strerror}), so an absent pid proves nothing'
        ) from exc


def _read_pid_file(pid: int, name: str, proc_root: Path) -> str | None:
    try:
        return _read_text(proc_root / str(pid) / name)
    except (FileNotFoundError, ProcessLookupError):
        _assert_can_see_other_processes(proc_root)
        return None


def _read_stat(pid: int, proc_root: Path) -> _Stat | None:
    text = _read_pid_file(pid, 'stat', proc_root)
    if text is None:
        return None
    fields = text[text.rfind(')') + 1 :].split()
    return _Stat(state=fields[0], start_ticks=int(fields[19]))


def _running_stat(pid: int, proc_root: Path) -> _Stat | None:
    stat = _read_stat(pid, proc_root)
    if stat is None or stat.state in _EXITED_STATES:
        return None
    return stat


def capture(pid: int, *, proc_root: Path = PROC_ROOT) -> ProcessIdentity | None:
    stat = _running_stat(pid, proc_root)
    if stat is None:
        return None
    return ProcessIdentity(pid, stat.start_ticks, _boot_id(proc_root), socket.gethostname())


def command_name(pid: int, *, proc_root: Path = PROC_ROOT) -> str | None:
    text = _read_pid_file(pid, 'comm', proc_root)
    return None if text is None else text.removesuffix('\n')


def liveness(identity: ProcessIdentity, *, proc_root: Path = PROC_ROOT) -> Liveness:
    if identity.host != socket.gethostname():
        return Liveness.OTHER_HOST
    if _boot_id(proc_root) != identity.boot_id:
        return Liveness.DEAD
    stat = _running_stat(identity.pid, proc_root)
    if stat is None or stat.start_ticks != identity.start_ticks:
        return Liveness.DEAD
    return Liveness.ALIVE
