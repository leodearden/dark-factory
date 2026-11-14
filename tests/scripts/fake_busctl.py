"""A stand-in for ``busctl --user --json=short`` talking to the systemd user manager (task 5978).

It answers only the exact D-Bus call shapes scripts/systemd_unit_environment.py
makes, from a JSON world file named by the ``FAKE_BUSCTL_WORLD`` environment
variable, and rejects every other argv with exit 64 so those shapes stay
pinned. Unit object paths come from the world and are deliberately not
systemd-escaped, so a caller that builds a path instead of asking LoadUnit for
one fails. A unit the world does not define gets a path whose LoadState is
"not-found" and whose Service properties answer their empty defaults, as the
real manager does.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

WORLD_ENV = 'FAKE_BUSCTL_WORLD'
EXIT_UNEXPECTED_ARGV = 64

_PREFIX = ['--user', '--json=short']
_DESTINATION = 'org.freedesktop.systemd1'
_MANAGER = [_DESTINATION, '/org/freedesktop/systemd1', 'org.freedesktop.systemd1.Manager']
_UNIT_INTERFACE = 'org.freedesktop.systemd1.Unit'
_SERVICE_INTERFACE = 'org.freedesktop.systemd1.Service'
_NOT_FOUND_PATH_PREFIX = '/fake/not-found/'

# (interface, property) -> (D-Bus signature, value when the world omits it).
_UNIT_PROPERTIES: Mapping[tuple[str, str], tuple[str, object]] = {
    (_UNIT_INTERFACE, 'LoadState'): ('s', 'loaded'),
    (_SERVICE_INTERFACE, 'Environment'): ('as', []),
    (_SERVICE_INTERFACE, 'EnvironmentFiles'): ('a(sb)', []),
    (_SERVICE_INTERFACE, 'PassEnvironment'): ('as', []),
    (_SERVICE_INTERFACE, 'UnsetEnvironment'): ('as', []),
    (_SERVICE_INTERFACE, 'MainPID'): ('u', 0),
    (_SERVICE_INTERFACE, 'ExecMainStartTimestamp'): ('t', 0),
}

_NOT_FOUND_UNIT: Mapping[str, object] = {'LoadState': 'not-found'}


class _UnexpectedArgv(Exception):
    pass


@dataclass(frozen=True)
class _World:
    manager_environment: list[str]
    units: dict[str, dict[str, object]]

    @classmethod
    def load(cls, path: Path) -> _World:
        raw = json.loads(path.read_text())
        return cls(raw.get('manager_environment', []), raw.get('units', {}))

    def unit_path(self, unit_name: str) -> str:
        unit = self.units.get(unit_name)
        return str(unit['path']) if unit is not None else _NOT_FOUND_PATH_PREFIX + unit_name

    def unit_at(self, path: str) -> Mapping[str, object]:
        for unit in self.units.values():
            if unit['path'] == path:
                return unit
        if path.startswith(_NOT_FOUND_PATH_PREFIX):
            return _NOT_FOUND_UNIT
        raise _UnexpectedArgv


def _reply(signature: str, data: object) -> str:
    return json.dumps({'type': signature, 'data': data})


def _unit_property(world: _World, path: str, interface: str, name: str) -> str:
    unit = world.unit_at(path)
    if (interface, name) not in _UNIT_PROPERTIES:
        raise _UnexpectedArgv
    signature, default = _UNIT_PROPERTIES[(interface, name)]
    return _reply(signature, unit.get(name, default))


def _answer(world: _World, args: list[str]) -> str:
    if args == ['get-property', *_MANAGER, 'Environment']:
        return _reply('as', world.manager_environment)
    if len(args) == 7 and args[:-1] == ['call', *_MANAGER, 'LoadUnit', 's']:
        return _reply('o', [world.unit_path(args[-1])])
    if len(args) == 5 and args[:2] == ['get-property', _DESTINATION]:
        return _unit_property(world, args[2], args[3], args[4])
    raise _UnexpectedArgv


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    try:
        if args[: len(_PREFIX)] != _PREFIX:
            raise _UnexpectedArgv
        world = _World.load(Path(os.environ[WORLD_ENV]))
        print(_answer(world, args[len(_PREFIX):]))
    except _UnexpectedArgv:
        print(f'unexpected argv: {args!r}', file=sys.stderr)
        return EXIT_UNEXPECTED_ARGV
    return 0


def install(bin_dir: Path) -> Path:
    """Write an executable ``busctl`` into *bin_dir* that runs this module; return its path."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    busctl = bin_dir / 'busctl'
    command = shlex.join([sys.executable, str(Path(__file__).resolve())])
    busctl.write_text(f'#!/bin/sh\nexec {command} "$@"\n')
    busctl.chmod(0o755)
    return busctl


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
