"""The project roots the factory's orchestrator units run, read from their unit files.

Contract: plans/deferral-flip-condition-prd.md §9 ε.
"""

from __future__ import annotations

import argparse
import fnmatch
import shlex
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import systemd_unit_parity

__all__ = [
    'DEFAULT_UNITS_DIR',
    'ProjectTarget',
    'UnitScan',
    'add_arguments',
    'resolve_targets',
    'scan_units',
    'targets_from_args',
]

DEFAULT_UNITS_DIR: Final = Path.home() / '.config' / 'systemd' / 'user'
UNIT_PATTERN: Final = 'orchestrator-*.service'
_CONFIG_FLAG: Final = '--config'
_NO_SWEEP_LABEL: Final = 'no sweep: surfaced only'


@dataclass(frozen=True)
class UnitScan:
    """Every project root a unit's ``--config`` names, with those units, and what could not be read."""

    roots: Mapping[Path, tuple[str, ...]]
    problems: tuple[str, ...]


@dataclass(frozen=True)
class ProjectTarget:
    root: Path
    units: tuple[str, ...]

    @property
    def sweep_label(self) -> str:
        return f'sweep: {", ".join(self.units)}' if self.units else _NO_SWEEP_LABEL


class _UnreadableUnit(Exception):
    pass


def _service_exec_starts(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise _UnreadableUnit(f'{path}: cannot read the unit: {exc}') from exc
    return systemd_unit_parity.parse_unit_directives(text).get('Service', {}).get('ExecStart', [])


def _effective_exec_start(unit: Path) -> str | None:
    commands: list[str] = []
    for path in (unit, *systemd_unit_parity.find_dropins(unit.parent, unit.name)):
        for value in _service_exec_starts(path):
            if value:
                commands.append(value)
            else:
                commands.clear()
    return commands[-1] if commands else None


def _config_argument(unit: Path, command: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise _UnreadableUnit(f'{unit}: cannot split ExecStart {command!r}: {exc}') from exc
    for index, token in enumerate(tokens):
        if token == _CONFIG_FLAG:
            return tokens[index + 1] if index + 1 < len(tokens) else ''
        if token.startswith(f'{_CONFIG_FLAG}='):
            return token.removeprefix(f'{_CONFIG_FLAG}=')
    return None


def _unit_root(unit: Path) -> Path | None:
    command = _effective_exec_start(unit)
    config = None if command is None else _config_argument(unit, command)
    if config is None:
        return None
    if not Path(config).is_absolute():
        raise _UnreadableUnit(f'{unit}: {_CONFIG_FLAG} {config!r} is not an absolute path')
    return Path(config).resolve().parent


def _unit_files(units_dir: Path) -> list[Path]:
    return sorted(path for path in units_dir.iterdir() if fnmatch.fnmatchcase(path.name, UNIT_PATTERN))


def scan_units(units_dir: Path) -> UnitScan:
    try:
        units = _unit_files(units_dir)
    except OSError as exc:
        return UnitScan(roots=MappingProxyType({}), problems=(f'{units_dir}: cannot list units: {exc}',))
    roots: dict[Path, list[str]] = {}
    problems: list[str] = []
    for unit in units:
        try:
            root = _unit_root(unit)
        except _UnreadableUnit as exc:
            problems.append(str(exc))
            continue
        if root is not None:
            roots.setdefault(root, []).append(unit.name)
    return UnitScan(
        roots=MappingProxyType({root: tuple(names) for root, names in roots.items()}),
        problems=tuple(problems),
    )


def _absolute(root: Path) -> Path:
    return root.expanduser().resolve()


def resolve_targets(scan: UnitScan, explicit_roots: Iterable[Path] | None) -> tuple[ProjectTarget, ...]:
    """Every scanned root when *explicit_roots* is None, else exactly those roots, in order, once each."""
    if explicit_roots is None:
        roots = sorted(scan.roots)
    else:
        roots = list(dict.fromkeys(_absolute(root) for root in explicit_roots))
    return tuple(ProjectTarget(root, scan.roots.get(root, ())) for root in roots)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--project-root',
        type=Path,
        action='append',
        default=None,
        help='a project root to run on; repeatable; replaces the unit enumeration (default: every '
        f'root an {UNIT_PATTERN} unit runs)',
    )
    parser.add_argument(
        '--units-dir',
        type=Path,
        default=DEFAULT_UNITS_DIR,
        help=f'where the {UNIT_PATTERN} units are read from (default: {DEFAULT_UNITS_DIR})',
    )


def targets_from_args(args: argparse.Namespace) -> tuple[tuple[ProjectTarget, ...], UnitScan]:
    scan = scan_units(args.units_dir)
    return resolve_targets(scan, args.project_root), scan
