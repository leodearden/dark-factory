#!/usr/bin/env python3
"""Read a routing split experiment out of every project ledger, and check an A/A split.

Contract: plans/routing-experiment-substrate-prd.md §"The readout (β)". Strictly
read-only: every store is opened with a ``mode=ro`` URI.

Exit codes:
  0  report printed
  1  --aa: a headline metric's |observed arm difference| exceeds its null p99
  2  usage error (argparse)
  3  nothing was read: no root resolved, or no root's runs.db was readable
  4  unproven: an --expect-rule id is absent from the window, --subset has no
     plan-shape producer row, or an --aa headline metric has too few eligible
     tasks in an arm
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path

import _checkout as _checkout
from shared.project_layout import KNOWN_PROJECT_ROOTS_ENV, known_project_roots
from shared.routing_split import split_bucket
from systemd_unit_environment import parse_assignments

EXIT_OK = 0  # report printed
EXIT_AA_SKEWED = 1  # --aa: a headline metric's |diff| exceeds its null p99
EXIT_NOTHING_READ = 3  # no root resolved, or no runs.db readable
EXIT_UNPROVEN = 4  # a producer or eligibility precondition of the claim is unmet

DEFAULT_SEED = 20261103  # seat N's prototype seed
DEFAULT_RESAMPLES = 4000  # the PRD's A/A null size

FUSED_MEMORY_UNIT = 'fused-memory.service'
_UNIT_ENVIRONMENT_ARGV = (
    'systemctl', '--user', 'show', FUSED_MEMORY_UNIT, '-p', 'Environment', '--value',
)
_UNIT_READ_TIMEOUT_SECS = 10


class RootSource(Enum):
    """Which source named the roots a report was computed over."""

    CLI = '--roots'
    ENVIRONMENT = KNOWN_PROJECT_ROOTS_ENV
    UNIT = f'{FUSED_MEMORY_UNIT} Environment'
    NONE = 'no source'


@dataclass(frozen=True)
class RootSelection:
    roots: tuple[Path, ...]
    source: RootSource


class KeyKind(Enum):
    """The stable key a task's arm is computed from."""

    TASK_ID = 'task_id'
    CANDIDATE_KEY = 'candidate_key'


class Subset(Enum):
    ALL = 'all'
    EXACT_3_MODULES = 'exact-3-modules'


@dataclass(frozen=True)
class Window:
    """The half-open interval [start, end) a report reads, both bounds in UTC."""

    start: datetime
    end: datetime

    @classmethod
    def trailing(cls, end: datetime, days: int) -> Window:
        return cls(end - timedelta(days=days), end)

    def sql_bounds(self) -> tuple[str, str]:
        """The bounds as the ISO-8601 UTC text the ledgers store, for TEXT comparison."""
        return self.start.isoformat(), self.end.isoformat()


def _roots_listed(text: str) -> tuple[Path, ...]:
    return known_project_roots({KNOWN_PROJECT_ROOTS_ENV: text})


def _unit_roots(unit_text: str | None) -> tuple[Path, ...]:
    if not unit_text:
        return ()
    try:
        entries = shlex.split(unit_text)
    except ValueError:
        return ()
    return known_project_roots(parse_assignments(entries))


def resolve_roots(
    cli_roots: str | None,
    environ: Mapping[str, str],
    unit_environment: Callable[[], str | None],
) -> RootSelection:
    """The roots to read: --roots, else the environment, else the installed fused-memory unit.

    There is deliberately no default-project-root fallback: reading dark-factory
    alone would certify only the null-contrast project.
    """
    if cli_roots is not None:
        return RootSelection(_roots_listed(cli_roots), RootSource.CLI)
    from_environment = known_project_roots(environ)
    if from_environment:
        return RootSelection(from_environment, RootSource.ENVIRONMENT)
    from_unit = _unit_roots(unit_environment())
    if from_unit:
        return RootSelection(from_unit, RootSource.UNIT)
    return RootSelection((), RootSource.NONE)


def read_fused_memory_unit_environment() -> str | None:
    """The installed fused-memory unit's Environment= text, or None when systemd does not answer."""
    try:
        completed = subprocess.run(
            _UNIT_ENVIRONMENT_ARGV,
            capture_output=True,
            text=True,
            timeout=_UNIT_READ_TIMEOUT_SECS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _bucket_count(text: str) -> int:
    count = int(text)
    try:
        split_bucket('', '', count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return count


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f'must be >= 1, got {value}')
    return value


def _utc_instant(text: str) -> datetime:
    try:
        instant = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'{text!r} is not ISO-8601') from exc
    if instant.utcoffset() is None:
        raise argparse.ArgumentTypeError(
            f'{text!r} carries no UTC offset; write e.g. 2026-11-20T00:00:00+00:00'
        )
    return instant.astimezone(UTC)


def _choices_metavar(choices: type[Enum]) -> str:
    return '{' + ','.join(member.value for member in choices) + '}'


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--roots',
        help=f'comma-separated project roots; default {KNOWN_PROJECT_ROOTS_ENV}, '
        f'then the installed {FUSED_MEMORY_UNIT} Environment',
    )
    parser.add_argument('--salt', required=True, help="the experiment's split salt")
    parser.add_argument('--buckets', type=_bucket_count, default=2)
    parser.add_argument('--days', type=_positive_int, default=30)
    parser.add_argument(
        '--until',
        type=_utc_instant,
        default=None,
        help='window end, ISO-8601 with a UTC offset (default: now)',
    )
    parser.add_argument(
        '--key', type=KeyKind, choices=list(KeyKind), default=KeyKind.TASK_ID,
        metavar=_choices_metavar(KeyKind),
    )
    parser.add_argument(
        '--subset', type=Subset, choices=list(Subset), default=Subset.ALL,
        metavar=_choices_metavar(Subset),
    )
    parser.add_argument('--aa', action='store_true', help='exit 1 on a skewed split')
    parser.add_argument('--resamples', type=_positive_int, default=DEFAULT_RESAMPLES)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument(
        '--expect-rule',
        action='append',
        default=[],
        help='a routing rule id that must appear in the window (repeatable)',
    )
    parser.add_argument('--json', action='store_true')
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] = os.environ,
    unit_environment: Callable[[], str | None] = read_fused_memory_unit_environment,
) -> int:
    args = _parser().parse_args(argv)
    selection = resolve_roots(args.roots, environ, unit_environment)
    if not selection.roots:
        print(
            f'routing_experiment_readout: no project root resolved: pass --roots, set '
            f'{KNOWN_PROJECT_ROOTS_ENV}, or install {FUSED_MEMORY_UNIT} with it in its '
            'Environment=',
            file=sys.stderr,
        )
        return EXIT_NOTHING_READ
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
