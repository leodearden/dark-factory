#!/usr/bin/env python3
"""Report a systemd --user service unit's environment, layer by layer.

Asks systemd over D-Bus and reads the unit's own running process; never reads
the user manager's own /proc environ, which is not the block it hands units.
Operator guide: OPERATIONS.md §3 "Inspecting a unit's live environment".
"""
from __future__ import annotations

import argparse
import enum
import errno
import json
import shlex
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Generic, TypeVar

# 0: report printed. 3: nothing was read, the meaning
# scripts/tasks_db_schema.py::EXIT_UNREADABLE gives it. 4: report printed, but
# the live layer was unreadable. Never 2: argparse spends it on usage errors.
EXIT_OK = 0
EXIT_SYSTEMD_UNANSWERED = 3
EXIT_LIVE_UNREADABLE = 4

_DESTINATION = 'org.freedesktop.systemd1'
_MANAGER_PATH = '/org/freedesktop/systemd1'
_MANAGER_INTERFACE = 'org.freedesktop.systemd1.Manager'
_UNIT_INTERFACE = 'org.freedesktop.systemd1.Unit'
_SERVICE_INTERFACE = 'org.freedesktop.systemd1.Service'
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_DEFAULT_PROC_ROOT = Path('/proc')
_DEFAULT_TIMEOUT_SECS = 10.0

# The names systemd.exec(5) "Environment Variables in Spawned Processes" says
# the manager computes for each process it spawns, from per-invocation state or
# unit settings that no D-Bus environment property carries. Its PATH, LANG,
# USER, LOGNAME, HOME, SHELL and XDG_RUNTIME_DIR a user service inherits from
# the manager block, and TERM is set only on a terminal, so those are compared.
_SET_AT_SPAWN = frozenset({
    'INVOCATION_ID',
    'RUNTIME_DIRECTORY', 'STATE_DIRECTORY', 'CACHE_DIRECTORY', 'LOGS_DIRECTORY',
    'CONFIGURATION_DIRECTORY', 'CREDENTIALS_DIRECTORY',
    'MAINPID', 'MANAGERPID', 'SYSTEMD_EXEC_PID', 'PIDFILE',
    'LISTEN_FDS', 'LISTEN_PID', 'LISTEN_FDNAMES', 'REMOTE_ADDR', 'REMOTE_PORT',
    'NOTIFY_SOCKET', 'WATCHDOG_PID', 'WATCHDOG_USEC', 'FDSTORE',
    'LOG_NAMESPACE', 'JOURNAL_STREAM',
    'SERVICE_RESULT', 'EXIT_CODE', 'EXIT_STATUS',
    'MONITOR_SERVICE_RESULT', 'MONITOR_EXIT_CODE', 'MONITOR_EXIT_STATUS',
    'MONITOR_INVOCATION_ID', 'MONITOR_UNIT',
    'TRIGGER_UNIT', 'TRIGGER_PATH', 'TRIGGER_TIMER_REALTIME_USEC',
    'TRIGGER_TIMER_MONOTONIC_USEC',
    'MEMORY_PRESSURE_WATCH', 'MEMORY_PRESSURE_WRITE',
})

_T = TypeVar('_T')


class SystemdUnanswered(Exception):
    """busctl gave no usable reply to *argv*; *detail* says why, verbatim."""

    def __init__(self, argv: Sequence[str], detail: str) -> None:
        self.argv = tuple(argv)
        self.detail = detail
        super().__init__(
            f'{shlex.join(self.argv)}: no usable reply from the systemd --user '
            f'manager — {detail}. Every layer of the report comes from systemd, '
            f'so nothing is printed.'
        )


class UnitNotLoaded(Exception):
    """The manager reports *unit* with a LoadState other than 'loaded'."""

    def __init__(self, unit: str, load_state: str) -> None:
        self.unit = unit
        self.load_state = load_state
        super().__init__(
            f"{unit}: LoadState is {load_state!r}, not 'loaded', so the "
            f'systemd --user manager has no environment for it. Check the name '
            f'with `systemctl --user list-units --all`.'
        )


@dataclass(frozen=True)
class _ReplyType(Generic[_T]):
    """A D-Bus signature busctl must report, and the parser for the data it carries."""

    signature: str
    parse: Callable[[object], _T]


def _strings(data: object) -> tuple[str, ...]:
    if isinstance(data, list) and all(isinstance(entry, str) for entry in data):
        return tuple(data)
    raise ValueError(f'expected a list of strings, got {data!r}')


def _integer(data: object) -> int:
    if isinstance(data, int) and not isinstance(data, bool):
        return data
    raise ValueError(f'expected an integer, got {data!r}')


def _string(data: object) -> str:
    if isinstance(data, str):
        return data
    raise ValueError(f'expected a string, got {data!r}')


def _single_object_path(data: object) -> str:
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], str):
        return data[0]
    raise ValueError(f'expected one object path, got {data!r}')


def _path_flag_pairs(data: object) -> tuple[tuple[str, bool], ...]:
    if isinstance(data, list) and all(
        isinstance(entry, list)
        and len(entry) == 2
        and isinstance(entry[0], str)
        and isinstance(entry[1], bool)
        for entry in data
    ):
        return tuple((entry[0], entry[1]) for entry in data)
    raise ValueError(f'expected a list of [path, flag] pairs, got {data!r}')


_STRING = _ReplyType('s', _string)
_STRINGS = _ReplyType('as', _strings)
_UINT32 = _ReplyType('u', _integer)
_UINT64 = _ReplyType('t', _integer)
_PATH_FLAG_PAIRS = _ReplyType('a(sb)', _path_flag_pairs)
_LOAD_UNIT_REPLY = _ReplyType('o', _single_object_path)


def _parse_reply(stdout: str, expect_type: _ReplyType[_T]) -> _T:
    reply = json.loads(stdout)
    if not isinstance(reply, dict):
        raise ValueError('not a JSON object')
    if reply['type'] != expect_type.signature:
        raise ValueError(f'expected D-Bus type {expect_type.signature!r}, got {reply["type"]!r}')
    return expect_type.parse(reply['data'])


def _busctl(args: Sequence[str], *, expect_type: _ReplyType[_T], timeout_secs: float) -> _T:
    argv = ('busctl', '--user', '--json=short', *args)
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_secs, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemdUnanswered(argv, str(exc)) from exc
    if completed.returncode != 0:
        raise SystemdUnanswered(argv, f'exit {completed.returncode}: {completed.stderr.strip()}')
    try:
        return _parse_reply(completed.stdout, expect_type)
    except (ValueError, KeyError) as exc:
        raise SystemdUnanswered(
            argv, f'malformed reply {completed.stdout.strip()!r}: {exc}'
        ) from exc


@dataclass(frozen=True)
class _UserManager:
    """The systemd --user manager, asked one property per busctl call.

    busctl's JSON reply carries no property name, so a multi-property call
    would tie each value to its name only by position.
    """

    timeout_secs: float

    def environment(self) -> tuple[str, ...]:
        return self._ask(
            ['get-property', _DESTINATION, _MANAGER_PATH, _MANAGER_INTERFACE, 'Environment'],
            _STRINGS,
        )

    def load_unit_path(self, unit: str) -> str:
        return self._ask(
            ['call', _DESTINATION, _MANAGER_PATH, _MANAGER_INTERFACE, 'LoadUnit', 's', unit],
            _LOAD_UNIT_REPLY,
        )

    def unit_property(
        self, path: str, interface: str, name: str, expect_type: _ReplyType[_T]
    ) -> _T:
        return self._ask(['get-property', _DESTINATION, path, interface, name], expect_type)

    def _ask(self, args: Sequence[str], expect_type: _ReplyType[_T]) -> _T:
        return _busctl(args, expect_type=expect_type, timeout_secs=self.timeout_secs)


def parse_assignments(entries: Iterable[str]) -> dict[str, str]:
    """``KEY=VALUE`` entries as a mapping: split on the first ``=``, a later duplicate wins."""
    assignments: dict[str, str] = {}
    for entry in entries:
        name, separator, value = entry.partition('=')
        if separator:
            assignments[name] = value
    return assignments


def _environ_path(pid: int, proc_root: Path) -> Path:
    return proc_root / str(pid) / 'environ'


def read_process_environ(pid: int, *, proc_root: Path = _DEFAULT_PROC_ROOT) -> dict[str, str]:
    """The environment process *pid* was exec'd with."""
    raw = _environ_path(pid, proc_root).read_bytes()
    return parse_assignments(
        entry.decode(errors='surrogateescape') for entry in raw.split(b'\0') if entry
    )


class Verdict(enum.Enum):
    MATCHES = enum.auto()
    DRIFTED = enum.auto()
    NOT_RUNNING = enum.auto()
    UNREADABLE = enum.auto()
    INDETERMINATE = enum.auto()
    SET_AT_SPAWN = enum.auto()


_VERDICT_MEANINGS: Mapping[Verdict, str] = {
    Verdict.MATCHES: 'the running process has the value a restart would give',
    Verdict.DRIFTED: (
        'the running process predates the current value; restarting the unit picks it up'
    ),
    Verdict.NOT_RUNNING: 'the unit has no running process to compare',
    Verdict.UNREADABLE: "the running process's environment could not be read",
    Verdict.INDETERMINATE: (
        'inputs systemd does not expose over D-Bus decide the value a restart would give'
    ),
    Verdict.SET_AT_SPAWN: (
        'systemd sets this anew each time it spawns the process, so no layer holds '
        'the value a restart would give'
    ),
}


@dataclass(frozen=True)
class LiveEnviron:
    pid: int
    started_at: datetime | None
    values: Mapping[str, str]


@dataclass(frozen=True)
class NoLiveProcess:
    pass


@dataclass(frozen=True)
class LiveEnvironUnreadable:
    pid: int
    path: Path
    errno: int
    strerror: str


@dataclass(frozen=True)
class LiveEnvironEmpty:
    """The MainPID's environ held no assignments, as when read mid-exec just after a (re)start."""

    pid: int
    path: Path


LiveNotRead = LiveEnvironUnreadable | LiveEnvironEmpty
LiveLayer = LiveEnviron | NoLiveProcess | LiveNotRead


@dataclass(frozen=True)
class RestartValue:
    """The value a restart would give the variable; None when it would leave it unset."""

    value: str | None


class RestartUnknown(enum.Enum):
    """Why the value a restart would give cannot be read from systemd over D-Bus."""

    UNMODELLED_INPUTS = enum.auto()
    SET_AT_SPAWN = enum.auto()


RestartOutcome = RestartValue | RestartUnknown

_UNKNOWN_RESTART_VERDICTS: Mapping[RestartUnknown, Verdict] = {
    RestartUnknown.UNMODELLED_INPUTS: Verdict.INDETERMINATE,
    RestartUnknown.SET_AT_SPAWN: Verdict.SET_AT_SPAWN,
}


@dataclass(frozen=True)
class VariableReport:
    name: str
    manager_value: str | None
    unit_value: str | None
    restart: RestartOutcome
    live_value: str | None
    verdict: Verdict


@dataclass(frozen=True)
class UnitEnvironment:
    """One unit's environment layers: the manager block, the unit's Environment=, the live process.

    *unmodelled* names the unit's EnvironmentFile=, PassEnvironment= and
    UnsetEnvironment= inputs, whose effect on a restart is not visible over D-Bus.
    """

    unit: str
    manager: Mapping[str, str]
    assignments: Mapping[str, str]
    unmodelled: tuple[str, ...]
    live: LiveLayer

    def variable(self, name: str) -> VariableReport:
        restart = self._restart(name)
        live_value = self.live.values.get(name) if isinstance(self.live, LiveEnviron) else None
        return VariableReport(
            name=name,
            manager_value=self.manager.get(name),
            unit_value=self.assignments.get(name),
            restart=restart,
            live_value=live_value,
            verdict=self._verdict(restart, live_value),
        )

    def _restart(self, name: str) -> RestartOutcome:
        """The highest-precedence source decides, in systemd.exec(5)'s order."""
        if self.unmodelled:
            return RestartUnknown.UNMODELLED_INPUTS
        if name in self.assignments:
            return RestartValue(self.assignments[name])
        if name in _SET_AT_SPAWN:
            return RestartUnknown.SET_AT_SPAWN
        return RestartValue(self.manager.get(name))

    def _verdict(self, restart: RestartOutcome, live_value: str | None) -> Verdict:
        if isinstance(self.live, NoLiveProcess):
            return Verdict.NOT_RUNNING
        if isinstance(self.live, LiveNotRead):
            return Verdict.UNREADABLE
        if isinstance(restart, RestartUnknown):
            return _UNKNOWN_RESTART_VERDICTS[restart]
        return Verdict.MATCHES if restart.value == live_value else Verdict.DRIFTED


def _started_at(microseconds: int) -> datetime | None:
    return _EPOCH + timedelta(microseconds=microseconds) if microseconds else None


def _unmodelled_inputs(manager: _UserManager, path: str) -> tuple[str, ...]:
    files = manager.unit_property(path, _SERVICE_INTERFACE, 'EnvironmentFiles', _PATH_FLAG_PAIRS)
    passed = manager.unit_property(path, _SERVICE_INTERFACE, 'PassEnvironment', _STRINGS)
    unset = manager.unit_property(path, _SERVICE_INTERFACE, 'UnsetEnvironment', _STRINGS)
    return (
        *(f'EnvironmentFile={"-" if optional else ""}{file}' for file, optional in files),
        *(f'PassEnvironment={name}' for name in passed),
        *(f'UnsetEnvironment={name}' for name in unset),
    )


def _live_layer(manager: _UserManager, path: str, proc_root: Path) -> LiveLayer:
    pid = manager.unit_property(path, _SERVICE_INTERFACE, 'MainPID', _UINT32)
    if pid == 0:
        return NoLiveProcess()
    started = manager.unit_property(path, _SERVICE_INTERFACE, 'ExecMainStartTimestamp', _UINT64)
    try:
        values = read_process_environ(pid, proc_root=proc_root)
    except OSError as exc:
        return LiveEnvironUnreadable(
            pid=pid,
            path=_environ_path(pid, proc_root),
            errno=exc.errno or 0,
            strerror=exc.strerror or str(exc),
        )
    if not values:
        return LiveEnvironEmpty(pid=pid, path=_environ_path(pid, proc_root))
    return LiveEnviron(pid, _started_at(started), values)


def inspect_unit(
    unit: str,
    *,
    proc_root: Path = _DEFAULT_PROC_ROOT,
    timeout_secs: float = _DEFAULT_TIMEOUT_SECS,
) -> UnitEnvironment:
    """Read every environment layer of the --user service *unit*."""
    manager = _UserManager(timeout_secs)
    path = manager.load_unit_path(unit)
    load_state = manager.unit_property(path, _UNIT_INTERFACE, 'LoadState', _STRING)
    if load_state != 'loaded':
        raise UnitNotLoaded(unit, load_state)
    environment = manager.unit_property(path, _SERVICE_INTERFACE, 'Environment', _STRINGS)
    return UnitEnvironment(
        unit=unit,
        manager=parse_assignments(manager.environment()),
        assignments=parse_assignments(environment),
        unmodelled=_unmodelled_inputs(manager, path),
        live=_live_layer(manager, path, proc_root),
    )


def exit_status(env: UnitEnvironment) -> int:
    """EXIT_LIVE_UNREADABLE when the report lacks the live layer, else EXIT_OK."""
    return EXIT_LIVE_UNREADABLE if isinstance(env.live, LiveNotRead) else EXIT_OK


def _printable(value: str) -> str:
    """*value* with each environ byte that was not UTF-8 shown as a ``\\xNN`` escape."""
    return value.encode(errors='surrogateescape').decode(errors='backslashreplace')


def _shown(value: str | None, absent: str = '(not set)') -> str:
    return absent if value is None else _printable(value)


_REFUSED_HINT = (
    'the kernel refuses to read the environ of a process running as another uid '
    'or holding capabilities the reader lacks, and a sandbox may deny /proc reads '
    'outright; "manager now" is the way to see what the manager hands units'
)
_EXITED_HINT = 'the process exited between the MainPID query and the read; re-run'
_EMPTY_HINT = 'the process was still being set up by exec, as just after a (re)start; re-run'
_UNREADABLE_HINTS: Mapping[int, str] = {
    errno.EACCES: _REFUSED_HINT,
    errno.EPERM: _REFUSED_HINT,
    errno.ENOENT: _EXITED_HINT,
    errno.ESRCH: _EXITED_HINT,
}


def _unreadable_lines(live: LiveEnvironUnreadable) -> list[str]:
    code = errno.errorcode.get(live.errno, str(live.errno))
    lines = [f'live process: pid {live.pid}, {live.path} unreadable: {code} ({live.strerror})']
    hint = _UNREADABLE_HINTS.get(live.errno)
    return [*lines, f'  {hint}'] if hint is not None else lines


def _live_header(live: LiveLayer) -> list[str]:
    if isinstance(live, NoLiveProcess):
        return ['live process: not running']
    if isinstance(live, LiveEnvironUnreadable):
        return _unreadable_lines(live)
    if isinstance(live, LiveEnvironEmpty):
        return [f'live process: pid {live.pid}, {live.path} read empty', f'  {_EMPTY_HINT}']
    when = 'start time unknown' if live.started_at is None else f'started {live.started_at.isoformat()}'
    return [f'live process: pid {live.pid}, {when}']


def _live_shown(live: LiveLayer, value: str | None) -> str:
    if isinstance(live, NoLiveProcess):
        return '(not running)'
    if isinstance(live, LiveNotRead):
        return '(unreadable)'
    return _shown(value)


_UNKNOWN_RESTART_SHOWN: Mapping[RestartUnknown, str] = {
    RestartUnknown.UNMODELLED_INPUTS: 'unknown (see "not modelled" above)',
    RestartUnknown.SET_AT_SPAWN: 'a new value, set by systemd at spawn',
}


def _restart_shown(restart: RestartOutcome) -> str:
    if isinstance(restart, RestartUnknown):
        return _UNKNOWN_RESTART_SHOWN[restart]
    return _shown(restart.value)


def _header(env: UnitEnvironment) -> list[str]:
    lines = [f'unit: {env.unit}', *_live_header(env.live)]
    if env.unmodelled:
        lines.append(
            f'not modelled: {", ".join(env.unmodelled)} (effect not visible over D-Bus)'
        )
    return lines


def _variable_lines(report: VariableReport, live: LiveLayer) -> list[str]:
    return [
        report.name,
        f'  manager now:   {_shown(report.manager_value)}',
        f'  unit sets:     {_shown(report.unit_value, "(not set by unit)")}',
        f'  restart gets:  {_restart_shown(report.restart)}',
        f'  live process:  {_live_shown(live, report.live_value)}',
        f'  verdict:       {report.verdict.name}: {_VERDICT_MEANINGS[report.verdict]}',
    ]


def render(env: UnitEnvironment, names: Iterable[str]) -> str:
    """A header for the unit, then one block per variable in *names*."""
    blocks = [_header(env), *(_variable_lines(env.variable(name), env.live) for name in names)]
    return '\n\n'.join('\n'.join(block) for block in blocks)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a systemd --user service unit's environment across the "
            'manager block, the unit\'s Environment= and its running process.'
        ),
    )
    parser.add_argument('unit', help='a --user service unit, e.g. orchestrator-dark-factory.service')
    parser.add_argument(
        '--var',
        action='append',
        dest='names',
        metavar='NAME',
        help='a variable to report; repeatable (default: PATH)',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        env = inspect_unit(args.unit)
    except (SystemdUnanswered, UnitNotLoaded) as refusal:
        print(refusal, file=sys.stderr)
        return EXIT_SYSTEMD_UNANSWERED
    print(render(env, args.names or ['PATH']))
    return exit_status(env)


if __name__ == '__main__':
    sys.exit(main())
