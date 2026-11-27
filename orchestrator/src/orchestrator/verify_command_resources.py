"""The C4 per-run verify record: what one verify command used, read in its own scope.

PRD ``plans/per-consumer-resource-recording-prd.md`` C4 and leaf 4. A scoped
command runs inside a script (``CommandCapture.wrap``) that copies its scope's
cgroup files into a directory named by the orchestrator once the command
exits. The orchestrator reads that directory, or the live scope before a kill,
with ``shared/src/shared/cgroup_stats.py::read_cgroup``.

What each capture state means:

- ``exit``: the scope's own script read the counters after the command exited.
- ``before_kill``: the orchestrator read the live scope just before a timeout,
  cancel or fault killed it.
- ``relocated``: some of the command's work ran in other scopes, named in
  ``relocated_scopes``. The counters, read at exit or before the kill, cover
  the launch scope only.
- ``unavailable``: nothing was measured, and ``capture_reason`` says why.

Relocation is seen only while the other scopes are live: by the kill path's
discovery, and by ``watch_relocations``, which samples on a doubling schedule
while the command runs. A command that relocates late and briefly, between two
samples, can still read as ``exit``. No kernel counter records a migration out
of a cgroup.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import os
import shlex
import shutil
import socket
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from shared.cgroup_stats import (
    CGROUP_ROOT,
    CPU_STAT,
    MEMORY_EVENTS,
    MEMORY_PEAK,
    SWAP_PEAK,
    CgroupReading,
    PressureResource,
    Unreadable,
    read_cgroup,
)
from shared.proc_group import unified_cgroup_path

from orchestrator.relocated_scopes import SpawnedChild, relocated_scopes

logger = logging.getLogger(__name__)

__all__ = [
    'Capture',
    'CommandCapture',
    'CommandResources',
    'allocate_command_capture',
    'resources_from_reading',
    'watch_relocations',
]

_CGROUP_FILE = 'cgroup'
_COMPLETE_FILE = 'complete'
_CAPTURED_FILES = (
    CPU_STAT,
    MEMORY_PEAK,
    SWAP_PEAK,
    MEMORY_EVENTS,
    *(resource.file_name for resource in PressureResource),
)


class Capture(StrEnum):
    EXIT = 'exit'
    BEFORE_KILL = 'before_kill'
    RELOCATED = 'relocated'
    UNAVAILABLE = 'unavailable'


_MEASUREMENTS = (
    'cpu_usec',
    'user_usec',
    'system_usec',
    'memory_peak_bytes',
    'swap_peak_bytes',
    'memory_high_events',
    'memory_max_events',
    'oom_kills',
    'cpu_stall_some_usec',
    'memory_stall_some_usec',
    'memory_stall_full_usec',
    'io_stall_some_usec',
)
_TEXTS = (
    'scope_unit',
    'host',
    'project',
    'role',
    'task_id',
    'module',
    'leg',
    'segment',
    'plan_id',
    'started_at',
    'ended_at',
)


@dataclass(frozen=True)
class CommandResources:
    """One verify command's C4 record; ``None`` = not measured, never zero."""

    capture: Capture
    capture_reason: str
    relocated_scopes: frozenset[str] = frozenset()
    cpu_usec: int | None = None
    user_usec: int | None = None
    system_usec: int | None = None
    memory_peak_bytes: int | None = None
    swap_peak_bytes: int | None = None
    memory_high_events: int | None = None
    memory_max_events: int | None = None
    oom_kills: int | None = None
    cpu_stall_some_usec: int | None = None
    memory_stall_some_usec: int | None = None
    memory_stall_full_usec: int | None = None
    io_stall_some_usec: int | None = None
    scope_unit: str | None = None
    host: str | None = None
    project: str | None = None
    role: str | None = None
    task_id: str | None = None
    attempt: int | None = None
    module: str | None = None
    leg: str | None = None
    segment: str | None = None
    plan_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None

    def __post_init__(self) -> None:
        _check_invariants(self)

    def to_dict(self) -> dict[str, object]:
        as_dict: dict[str, object] = {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }
        as_dict['capture'] = self.capture.value
        as_dict['relocated_scopes'] = sorted(self.relocated_scopes)
        return as_dict

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> CommandResources:
        names = [field.name for field in dataclasses.fields(cls)]
        missing = [name for name in names if name not in record]
        if missing:
            raise ValueError(f'C4 record lacks {", ".join(missing)}: {dict(record)!r}')
        try:
            capture = Capture(record['capture'])
        except (ValueError, TypeError):
            raise ValueError(f'capture: not a capture state: {record["capture"]!r}') from None
        scopes = record['relocated_scopes']
        if not isinstance(scopes, list):
            raise ValueError(f'relocated_scopes: not a list: {scopes!r}')
        values = {name: record[name] for name in names}
        return cls(**{**values, 'capture': capture, 'relocated_scopes': frozenset(scopes)})


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _check_invariants(record: CommandResources) -> None:
    _check_types(record)
    _check_capture_agrees(record)


def _check_types(record: CommandResources) -> None:
    if not isinstance(record.capture, Capture):
        raise ValueError(f'capture: not a Capture: {record.capture!r}')
    if not isinstance(record.capture_reason, str) or not record.capture_reason:
        raise ValueError(f'capture_reason: must be a non-empty string: {record.capture_reason!r}')
    scopes = record.relocated_scopes
    if not isinstance(scopes, frozenset) or not all(isinstance(s, str) for s in scopes):
        raise ValueError(f'relocated_scopes: must be a frozenset of unit names: {scopes!r}')
    for name in (*_MEASUREMENTS, 'attempt'):
        value = getattr(record, name)
        if value is not None and not _is_count(value):
            raise ValueError(f'{name}: must be a non-negative int or None: {value!r}')
    for name in _TEXTS:
        value = getattr(record, name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f'{name}: must be a string or None: {value!r}')


def _check_capture_agrees(record: CommandResources) -> None:
    capture, scopes = record.capture, record.relocated_scopes
    if capture is Capture.UNAVAILABLE:
        for name in _MEASUREMENTS:
            value = getattr(record, name)
            if value is not None:
                raise ValueError(f'{name}: an unavailable capture measures nothing: {value!r}')
        return
    if record.cpu_usec is None:
        raise ValueError(f'cpu_usec: a {capture} capture must measure CPU: None')
    if capture is Capture.RELOCATED and not scopes:
        raise ValueError(f'relocated_scopes: a relocated capture names its scopes: {scopes!r}')
    if capture is not Capture.RELOCATED and scopes:
        raise ValueError(
            f'relocated_scopes: a {capture} capture covers its launch scope only: '
            f'{sorted(scopes)!r}'
        )


def _stall(reading: CgroupReading, resource: PressureResource, kind: str) -> int | None:
    totals = reading.pressure.get(resource)
    if totals is None:
        return None
    return totals.some_usec if kind == 'some' else totals.full_usec


def resources_from_reading(
    reading: CgroupReading,
    *,
    capture: Capture,
    reason: str,
    relocated: Iterable[str],
) -> CommandResources:
    """Map a shared cgroup reading to the C4 measurement fields."""
    cpu = reading.cpu_stat or {}
    events = reading.memory_events or {}
    return CommandResources(
        capture=capture,
        capture_reason=reason,
        relocated_scopes=frozenset(relocated),
        cpu_usec=cpu.get('usage_usec'),
        user_usec=cpu.get('user_usec'),
        system_usec=cpu.get('system_usec'),
        memory_peak_bytes=reading.memory_peak,
        swap_peak_bytes=reading.swap_peak,
        memory_high_events=events.get('high'),
        memory_max_events=events.get('max'),
        oom_kills=events.get('oom_kill'),
        cpu_stall_some_usec=_stall(reading, PressureResource.CPU, 'some'),
        memory_stall_some_usec=_stall(reading, PressureResource.MEMORY, 'some'),
        memory_stall_full_usec=_stall(reading, PressureResource.MEMORY, 'full'),
        io_stall_some_usec=_stall(reading, PressureResource.IO, 'some'),
    )


@dataclass(frozen=True)
class _Gap:
    """Why a capture produced no reading."""

    reason: str


_READ_AT = {
    Capture.EXIT: 'read by the scope script after the command exited',
    Capture.BEFORE_KILL: 'read from the live scope before the kill',
}


def _unavailable(reason: str, relocated: frozenset[str]) -> CommandResources:
    return CommandResources(
        capture=Capture.UNAVAILABLE, capture_reason=reason, relocated_scopes=relocated
    )


def _measured(
    reading: CgroupReading, read_at: Capture, relocated: frozenset[str]
) -> CommandResources:
    if reading.cpu_stat is None or 'usage_usec' not in reading.cpu_stat:
        return _unavailable(f'{CPU_STAT} usage_usec was not captured ({_READ_AT[read_at]})', relocated)
    if not relocated:
        return resources_from_reading(
            reading, capture=read_at, reason=_READ_AT[read_at], relocated=relocated
        )
    reason = (
        f'{_READ_AT[read_at]}; work also ran in {", ".join(sorted(relocated))}, '
        'which these counters do not cover'
    )
    return resources_from_reading(
        reading, capture=Capture.RELOCATED, reason=reason, relocated=relocated
    )


def _read_directory(directory: Path) -> CgroupReading | _Gap:
    try:
        present = frozenset(os.listdir(directory))
    except OSError as exc:
        return _Gap(f'{directory} could not be listed: {exc}')
    reading = read_cgroup(directory, present=present)
    if isinstance(reading, Unreadable):
        return _Gap(f'{reading.file} unreadable: {reading.reason}')
    return reading


class CommandCapture:
    """One command's capture: its directory, its launch scope, and what was seen of it."""

    def __init__(
        self,
        directory: Path | None,
        *,
        cgroup_root: Path = CGROUP_ROOT,
        allocation_error: str | None = None,
    ) -> None:
        self.directory = directory
        self._cgroup_root = cgroup_root
        self._allocation_error = allocation_error
        self._scope_unit: str | None = None
        self._relocated: set[str] = set()
        self._before_kill: CgroupReading | _Gap | None = None

    def launched(self, scope_unit: str) -> None:
        self._scope_unit = scope_unit

    def note_relocated(self, scopes: Iterable[str]) -> None:
        self._relocated.update(scopes)

    def wrap(self, cmd: str) -> str:
        """*cmd* in a subshell, with its scope's counters copied to the directory after it.

        The subshell keeps ``exit`` and ``exec`` in *cmd* from skipping the
        copy. *cmd* starts on the script's first line, so bash's line numbers
        for it are unchanged. The script starts nothing but ``cat``, and its
        own statements write nothing to the command's output.
        """
        if self.directory is None:
            return cmd
        directory = shlex.quote(str(self.directory))
        source = shlex.quote(str(self._cgroup_root)) + '"${__df_cg#0::}/$__df_f"'
        files = ' '.join(shlex.quote(name) for name in _CAPTURED_FILES)
        record_cgroup = (
            '{ while read -r __df_cg; do case $__df_cg in 0::*) break;; esac; done'
            f' < /proc/self/cgroup; printf \'%s\\n\' "$__df_cg" > {directory}/{_CGROUP_FILE}; }}'
            ' 2>/dev/null'
        )
        copy_counters = (
            f'{{ for __df_f in {files}; do [ -r {source} ] && cat {source}'
            f' > {directory}/"$__df_f"; done; : > {directory}/{_COMPLETE_FILE}; }} 2>/dev/null'
        )
        return (
            f'{record_cgroup}; ( {cmd}\n)\n__df_rc=$?\n{copy_counters}\nexit "$__df_rc"'
        )

    def read_before_kill(self, relocated: Iterable[str]) -> None:
        """Read the live launch scope now; never raises, so it never costs the kill."""
        try:
            self.note_relocated(relocated)
            self._before_kill = self._read_live_scope()
        except Exception as exc:
            self._before_kill = _Gap(f'before-kill read failed: {type(exc).__name__}: {exc}')

    def record(
        self,
        *,
        project: str | None,
        role: str | None,
        task_id: str | None,
        attempt: int | None,
        module: str | None,
        leg: str | None,
        segment: str | None,
        plan_id: str | None,
        started_at: str | None,
        ended_at: str | None,
    ) -> CommandResources:
        """The command's C4 record; a capture fault reads ``unavailable``, never raises."""
        try:
            measured = self._measure()
        except Exception as exc:
            measured = _unavailable(
                f'capture failed: {type(exc).__name__}: {exc}', frozenset(self._relocated)
            )
        try:
            return dataclasses.replace(
                measured,
                scope_unit=self._scope_unit,
                host=socket.gethostname(),
                project=project,
                role=role,
                task_id=task_id,
                attempt=attempt,
                module=module,
                leg=leg,
                segment=segment,
                plan_id=plan_id,
                started_at=started_at,
                ended_at=ended_at,
            )
        except ValueError as exc:
            return _unavailable(f'record identity refused: {exc}', frozenset())

    def _measure(self) -> CommandResources:
        relocated = frozenset(self._relocated)
        if self.directory is None:
            return _unavailable(
                f'no capture directory could be allocated: {self._allocation_error}', relocated
            )
        if self._scope_unit is None:
            return _unavailable('the command ran in no cgroup scope of its own', relocated)
        if isinstance(self._before_kill, CgroupReading):
            return _measured(self._before_kill, Capture.BEFORE_KILL, relocated)
        at_exit = self._read_exit_capture(self.directory)
        if isinstance(at_exit, CgroupReading):
            return _measured(at_exit, Capture.EXIT, relocated)
        return _unavailable((self._before_kill or at_exit).reason, relocated)

    def _recorded_scope_path(self, directory: Path) -> str | _Gap:
        """The unified cgroup path the scope script recorded, if it is the launched scope."""
        try:
            text = (directory / _CGROUP_FILE).read_text(encoding='utf-8')
        except OSError as exc:
            return _Gap(f'the scope script never recorded its cgroup: {exc}')
        path = unified_cgroup_path(text)
        if path is None:
            return _Gap(f'the recorded cgroup names no unified path: {text!r}')
        ran_in = path.rsplit('/', 1)[-1]
        if ran_in != self._scope_unit:
            return _Gap(f'the command ran in {ran_in}, not the launched scope {self._scope_unit}')
        return path

    def _read_exit_capture(self, directory: Path) -> CgroupReading | _Gap:
        path = self._recorded_scope_path(directory)
        if isinstance(path, _Gap):
            return path
        if not (directory / _COMPLETE_FILE).exists():
            return _Gap('the command ended before its capture completed')
        return _read_directory(directory)

    def _read_live_scope(self) -> CgroupReading | _Gap:
        if self.directory is None or self._scope_unit is None:
            return _Gap('no launched scope to read before the kill')
        path = self._recorded_scope_path(self.directory)
        if isinstance(path, _Gap):
            return path
        return _read_directory(self._cgroup_root / path.lstrip('/'))


@contextlib.contextmanager
def allocate_command_capture(*, cgroup_root: Path = CGROUP_ROOT) -> Iterator[CommandCapture]:
    """A fresh capture directory, removed on exit.

    Allocation failure is logged and yields a capture with no directory, whose
    record reads ``unavailable``, so the verify runs on rather than failing.
    """
    directory: Path | None
    error: str | None = None
    try:
        directory = Path(tempfile.mkdtemp(prefix='df-verify-capture-'))
    except OSError as exc:
        logger.warning('could not allocate a verify capture directory; running without one: %s', exc)
        directory, error = None, f'{type(exc).__name__}: {exc}'
    try:
        yield CommandCapture(directory, cgroup_root=cgroup_root, allocation_error=error)
    finally:
        if directory is not None:
            try:
                shutil.rmtree(directory)
            except OSError as exc:
                logger.warning('could not remove verify capture directory %s: %s', directory, exc)


async def watch_relocations(
    child: SpawnedChild, *, launched_in: str, capture: CommandCapture
) -> None:
    """Note the scopes *child*'s session moved into, after waits of 1, 2, 4 ... seconds.

    Runs until cancelled or *child* exits. A failure is logged at DEBUG and ends
    the watch, so it never reaches the command's runner.
    """
    delay = 1.0
    try:
        while child.returncode is None:
            await asyncio.sleep(delay)
            capture.note_relocated(await relocated_scopes(child, launched_in=launched_in))
            delay *= 2
    except Exception:
        logger.debug('relocation watch of pid %d stopped', child.pid, exc_info=True)
