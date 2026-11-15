#!/usr/bin/env python3
"""Report, and on request arm, the user timers scripts/expected_user_timers.py declares.

Why each timer is expected, and what armed means for it: OPERATIONS.md §12 "The nightly ladder".
"""

from __future__ import annotations

import argparse
import enum
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

import expected_user_timers
from expected_user_timers import ExpectedTimer
from systemd_unit_parity import emit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared' / 'src'))

from shared.project_id import project_id_for_root
from shared.project_layout import CheckoutRootError, default_project_root

TOOL: Final = 'user_timers'
SHOW_FIELDS: Final = ('Id', 'LoadState', 'UnitFileState', 'ActiveState')
SHOW_TIMEOUT_SECS: Final = 30
NOT_INSTALLED_LOAD_STATES: Final = frozenset({'', 'not-found'})
ARM_HINT: Final = (
    'scripts/user_timers.py arm runs the installer of each one not armed, except masked or unreadable ones'
)


class Arming(enum.Enum):
    ARMED = 'armed'
    NOT_INSTALLED = 'not-installed'
    NOT_ENABLED = 'not-enabled'
    NOT_ACTIVE = 'not-active'
    MASKED = 'masked'
    NOT_LOADED = 'not-loaded'
    UNREADABLE = 'unreadable'


class TimersExit(enum.IntEnum):
    ALL_ARMED = 0
    NEEDS_OPERATOR = 1


@dataclass(frozen=True)
class Target:
    """One expected timer bound to this repo's root and its project's instance."""

    timer: ExpectedTimer
    repo_root: Path
    project_id: str

    @property
    def unit(self) -> str:
        return self.timer.unit_name(self.project_id)

    @property
    def installer_argv(self) -> tuple[str, ...]:
        return self.timer.installer_argv(self.repo_root, self.project_id)

    @property
    def install_command(self) -> str:
        """The installer argv as an operator types it from the repo root."""
        return shlex.join((self.timer.installer, *self.installer_argv[1:]))


def bind_targets(repo_root: Path) -> tuple[Target, ...]:
    project_id = project_id_for_root(repo_root)
    return tuple(Target(timer, repo_root, project_id) for timer in expected_user_timers.EXPECTED_TIMERS)


def classify(fields: Mapping[str, str]) -> Arming:
    load_state = fields.get('LoadState', '')
    if load_state in NOT_INSTALLED_LOAD_STATES:
        return Arming.NOT_INSTALLED
    if load_state == 'masked':
        return Arming.MASKED
    if load_state != 'loaded':
        return Arming.NOT_LOADED
    if fields.get('UnitFileState') != 'enabled':
        return Arming.NOT_ENABLED
    if fields.get('ActiveState') != 'active':
        return Arming.NOT_ACTIVE
    return Arming.ARMED


def read_states(units: Sequence[str]) -> tuple[dict[str, Arming], str]:
    """Each unit's Arming from one `systemctl --user show` call, and a diagnostic when any is unreadable."""
    argv = ('systemctl', '--user', 'show', '-p', ','.join(SHOW_FIELDS), *units)
    unreadable = dict.fromkeys(units, Arming.UNREADABLE)
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=SHOW_TIMEOUT_SECS)
    except subprocess.TimeoutExpired:
        return unreadable, f'systemctl --user show timed out after {SHOW_TIMEOUT_SECS}s'
    except OSError as exc:
        return unreadable, f'systemctl --user show could not start: {exc}'
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or '(no stderr)'
        return unreadable, f'systemctl --user show exited {completed.returncode}: {stderr}'
    blocks = _show_blocks(completed.stdout)
    missing = [unit for unit in units if unit not in blocks]
    states = {unit: classify(blocks[unit]) if unit in blocks else Arming.UNREADABLE for unit in units}
    diagnostic = f'systemctl --user show answered no block for {", ".join(missing)}' if missing else ''
    return states, diagnostic


def _show_blocks(stdout: str) -> dict[str, dict[str, str]]:
    """`systemctl show` output for several units: blank-line-separated KEY=VALUE blocks, keyed by Id."""
    blocks = {}
    for chunk in stdout.split('\n\n'):
        fields = dict(line.split('=', 1) for line in chunk.splitlines() if '=' in line)
        if 'Id' in fields:
            blocks[fields['Id']] = fields
    return blocks


def _remedy(target: Target, arming: Arming) -> str:
    unit, command = target.unit, target.install_command
    match arming:
        case Arming.ARMED:
            return ''
        case Arming.NOT_INSTALLED | Arming.NOT_ENABLED | Arming.NOT_ACTIVE:
            return f'run {command}'
        case Arming.NOT_LOADED:
            return f'systemd rejected the installed unit file; read systemctl --user status {unit}, then run {command}'
        case Arming.MASKED:
            return f'a mask is a deliberate off; if this one is not, systemctl --user unmask {unit}, then run {command}'
        case Arming.UNREADABLE:
            return 'its state could not be read; see the systemctl line above'


def report_line(target: Target, arming: Arming) -> str:
    remedy = _remedy(target, arming)
    return f'[{arming.value}] {target.unit}' + (f': {remedy}' if remedy else '')


def render_report(
    targets: Sequence[Target], states: Mapping[str, Arming], diagnostic: str, *, hint: str, out: TextIO,
) -> bool:
    """Emit one line per target and a summary; True when every target is armed."""
    if diagnostic:
        emit(TOOL, f'systemctl: {diagnostic}', out=out)
    for target in targets:
        emit(TOOL, report_line(target, states[target.unit]), out=out)
    armed = sum(1 for target in targets if states[target.unit] is Arming.ARMED)
    all_armed = armed == len(targets)
    summary = f'{armed} of {len(targets)} expected user timers armed'
    emit(TOOL, summary if all_armed else f'{summary}; {hint}', out=out)
    return all_armed


def check(targets: Sequence[Target], *, out: TextIO) -> TimersExit:
    states, diagnostic = read_states([target.unit for target in targets])
    all_armed = render_report(targets, states, diagnostic, hint=ARM_HINT, out=out)
    return TimersExit.ALL_ARMED if all_armed else TimersExit.NEEDS_OPERATOR


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        '--repo-root',
        type=Path,
        metavar='DIR',
        help='checkout whose installers run and whose basename names template instances '
        '(default: the main checkout enclosing this script)',
    )
    parser = argparse.ArgumentParser(
        description=(
            'Exit 0 only if every user timer scripts/expected_user_timers.py declares is loaded, '
            'enabled and active; every report line is tagged [user_timers].'
        )
    )
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check', parents=[common], help='report each expected timer; never changes the host')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        repo_root = (args.repo_root or default_project_root()).resolve()
        if not repo_root.is_dir():
            parser.error(f'--repo-root {repo_root} is not a directory')
        targets = bind_targets(repo_root)
    except (CheckoutRootError, ValueError) as exc:
        parser.error(str(exc))
    return int(check(targets, out=sys.stdout))


if __name__ == '__main__':
    sys.exit(main())
