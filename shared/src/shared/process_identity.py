"""Identify a process so that pid reuse and reboots read as its death.

Contract: plans/deferral-flip-condition-prd.md §5.2 and decision 6.
"""

from __future__ import annotations

import errno
import os
import socket
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from shared.proc_group import MalformedStatLine, parse_start_ticks
from shared.proc_liveness import pid_alive

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


class _Absent(ProcessIdentityUnreadable):
    """ENOENT/ESRCH: unreadable unless the caller holds positive evidence that it means exited."""


def _errno_cause(exc: OSError) -> str:
    return f'{errno.errorcode.get(exc.errno or 0, f"errno {exc.errno}")} ({exc.strerror})'


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8', errors='replace')
    except (FileNotFoundError, ProcessLookupError) as exc:
        raise _Absent(path, _errno_cause(exc)) from exc
    except OSError as exc:
        raise ProcessIdentityUnreadable(path, _errno_cause(exc)) from exc


def _boot_id(proc_root: Path) -> str:
    path = proc_root / 'sys' / 'kernel' / 'random' / 'boot_id'
    boot_id = _read_text(path).strip()
    if not boot_id:
        raise ProcessIdentityUnreadable(path, 'empty boot id')
    return boot_id


def _assert_can_see_other_processes(proc_root: Path) -> None:
    path = proc_root / '1'
    try:
        os.stat(path)
    except OSError as exc:
        raise ProcessIdentityUnreadable(
            path, f'{_errno_cause(exc)}: pid 1 is invisible, so an absent pid proves nothing'
        ) from exc


def _read_pid_file(path: Path, proc_root: Path) -> str | None:
    try:
        return _read_text(path)
    except _Absent:
        _assert_can_see_other_processes(proc_root)
        return None


def _start_ticks(path: Path, text: str) -> int:
    try:
        return parse_start_ticks(path, text)
    except MalformedStatLine as exc:
        raise ProcessIdentityUnreadable(path, f'{exc.reason}: {text!r}') from exc


def _start_ticks_of_existing_pid(pid: int, proc_root: Path) -> int | None:
    if not pid_alive(pid):
        return None
    path = proc_root / str(pid) / 'stat'
    text = _read_pid_file(path, proc_root)
    return None if text is None else _start_ticks(path, text)


def capture(pid: int, *, proc_root: Path = PROC_ROOT) -> ProcessIdentity | None:
    start_ticks = _start_ticks_of_existing_pid(pid, proc_root)
    if start_ticks is None:
        return None
    return ProcessIdentity(pid, start_ticks, _boot_id(proc_root), socket.gethostname())


def command_name(pid: int, *, proc_root: Path = PROC_ROOT) -> str | None:
    text = _read_pid_file(proc_root / str(pid) / 'comm', proc_root)
    return None if text is None else text.removesuffix('\n')


def liveness(identity: ProcessIdentity, *, proc_root: Path = PROC_ROOT) -> Liveness:
    if identity.host != socket.gethostname():
        return Liveness.OTHER_HOST
    if _boot_id(proc_root) != identity.boot_id:
        return Liveness.DEAD
    start_ticks = _start_ticks_of_existing_pid(identity.pid, proc_root)
    return Liveness.ALIVE if start_ticks == identity.start_ticks else Liveness.DEAD
