#!/usr/bin/env python3
"""Report a systemd --user service unit's environment, layer by layer.

Asks systemd over D-Bus and reads the unit's own running process; never reads
the user manager's own /proc environ, which is not the block it hands units.
Operator guide: OPERATIONS.md §3 "Inspecting a unit's live environment".
"""
from __future__ import annotations

import argparse
import enum
import json
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Generic, TypeVar

EXIT_OK = 0

_DESTINATION = 'org.freedesktop.systemd1'
_MANAGER_PATH = '/org/freedesktop/systemd1'
_MANAGER_INTERFACE = 'org.freedesktop.systemd1.Manager'
_SERVICE_INTERFACE = 'org.freedesktop.systemd1.Service'
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_DEFAULT_PROC_ROOT = Path('/proc')
_DEFAULT_TIMEOUT_SECS = 10.0

_T = TypeVar('_T')


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


_STRINGS = _ReplyType('as', _strings)
_UINT32 = _ReplyType('u', _integer)
_UINT64 = _ReplyType('t', _integer)
_PATH_FLAG_PAIRS = _ReplyType('a(sb)', _path_flag_pairs)
_LOAD_UNIT_REPLY = _ReplyType('o', _single_object_path)


def _busctl(args: Sequence[str], *, expect_type: _ReplyType[_T], timeout_secs: float) -> _T:
    completed = subprocess.run(
        ['busctl', '--user', '--json=short', *args],
        capture_output=True,
        text=True,
        timeout=timeout_secs,
        check=False,
    )
    reply = json.loads(completed.stdout)
    if reply['type'] != expect_type.signature:
        raise ValueError(f'expected D-Bus type {expect_type.signature!r}, got {reply!r}')
    return expect_type.parse(reply['data'])


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


def read_process_environ(pid: int, *, proc_root: Path = _DEFAULT_PROC_ROOT) -> dict[str, str]:
    """The environment process *pid* was exec'd with."""
    raw = (proc_root / str(pid) / 'environ').read_bytes()
    return parse_assignments(
        entry.decode(errors='surrogateescape') for entry in raw.split(b'\0') if entry
    )


class Verdict(enum.Enum):
    MATCHES = enum.auto()
    DRIFTED = enum.auto()
    NOT_RUNNING = enum.auto()
    INDETERMINATE = enum.auto()


_VERDICT_MEANINGS: Mapping[Verdict, str] = {
    Verdict.MATCHES: 'the running process has the value a restart would give',
    Verdict.DRIFTED: (
        'the running process predates the current value; restarting the unit picks it up'
    ),
    Verdict.NOT_RUNNING: 'the unit has no running process to compare',
    Verdict.INDETERMINATE: (
        'inputs systemd does not expose over D-Bus decide the value a restart would give'
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


LiveLayer = LiveEnviron | NoLiveProcess


@dataclass(frozen=True)
class VariableReport:
    name: str
    manager_value: str | None
    unit_value: str | None
    restart_value: str | None
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
        restart_value = self._restart_value(name)
        live_value = self.live.values.get(name) if isinstance(self.live, LiveEnviron) else None
        return VariableReport(
            name=name,
            manager_value=self.manager.get(name),
            unit_value=self.assignments.get(name),
            restart_value=restart_value,
            live_value=live_value,
            verdict=self._verdict(restart_value, live_value),
        )

    def _restart_value(self, name: str) -> str | None:
        if self.unmodelled:
            return None
        return self.assignments.get(name, self.manager.get(name))

    def _verdict(self, restart_value: str | None, live_value: str | None) -> Verdict:
        if isinstance(self.live, NoLiveProcess):
            return Verdict.NOT_RUNNING
        if self.unmodelled:
            return Verdict.INDETERMINATE
        return Verdict.MATCHES if restart_value == live_value else Verdict.DRIFTED


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
    return LiveEnviron(pid, _started_at(started), read_process_environ(pid, proc_root=proc_root))


def inspect_unit(
    unit: str,
    *,
    proc_root: Path = _DEFAULT_PROC_ROOT,
    timeout_secs: float = _DEFAULT_TIMEOUT_SECS,
) -> UnitEnvironment:
    """Read every environment layer of the --user service *unit*."""
    manager = _UserManager(timeout_secs)
    path = manager.load_unit_path(unit)
    environment = manager.unit_property(path, _SERVICE_INTERFACE, 'Environment', _STRINGS)
    return UnitEnvironment(
        unit=unit,
        manager=parse_assignments(manager.environment()),
        assignments=parse_assignments(environment),
        unmodelled=_unmodelled_inputs(manager, path),
        live=_live_layer(manager, path, proc_root),
    )


def _shown(value: str | None, absent: str = '(not set)') -> str:
    return absent if value is None else value


def _live_header(live: LiveLayer) -> str:
    if isinstance(live, NoLiveProcess):
        return 'live process: not running'
    when = live.started_at.isoformat() if live.started_at is not None else 'start time unknown'
    return f'live process: pid {live.pid}, started {when}'


def _header(env: UnitEnvironment) -> list[str]:
    lines = [f'unit: {env.unit}', _live_header(env.live)]
    if env.unmodelled:
        lines.append(
            f'not modelled: {", ".join(env.unmodelled)} (effect not visible over D-Bus)'
        )
    return lines


def _variable_lines(env: UnitEnvironment, report: VariableReport) -> list[str]:
    restart = 'unknown (see "not modelled" above)' if env.unmodelled else _shown(report.restart_value)
    live = _shown(report.live_value) if isinstance(env.live, LiveEnviron) else '(not running)'
    return [
        report.name,
        f'  manager now:   {_shown(report.manager_value)}',
        f'  unit sets:     {_shown(report.unit_value, "(not set by unit)")}',
        f'  restart gets:  {restart}',
        f'  live process:  {live}',
        f'  verdict:       {report.verdict.name}: {_VERDICT_MEANINGS[report.verdict]}',
    ]


def render(env: UnitEnvironment, names: Iterable[str]) -> str:
    """A header for the unit, then one block per variable in *names*."""
    blocks = [_header(env), *(_variable_lines(env, env.variable(name)) for name in names)]
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
    env = inspect_unit(args.unit)
    print(render(env, args.names or ['PATH']))
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
