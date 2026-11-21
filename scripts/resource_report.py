#!/usr/bin/env python3
"""Report the factory hosts' resource use by consumer, from the load recorder's store.

``holds`` is PRD ``plans/per-consumer-resource-recording-prd.md`` leaf 2: for
each memory hold in the named projects' runs.db files, which consumers grew and
which paid in the window before it (scripts/resource_report_holds.py, rendered by
scripts/resource_report_holds_render.py). Leaf 8 adds ``costs``.

Runs under /usr/bin/python3 with no venv: scripts/_checkout.py binds this
checkout's workspace members. The project roots are named, never defaulted.
Exit 0 with a report, 2 for a usage error or an unreadable samples store.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import _checkout as _checkout
import resource_report_holds as holds
import resource_report_holds_render as render
from shared.project_layout import default_project_root

_RELATIVE_SINCE = re.compile(r'(\d+)([smhd])')
_SINCE_UNITS = {'s': 'seconds', 'm': 'minutes', 'h': 'hours', 'd': 'days'}


def parse_since(text: str, now: datetime) -> datetime:
    """``<N>s|m|h|d`` before *now*, or an ISO date or datetime (a naive one is UTC), in UTC."""
    relative = _RELATIVE_SINCE.fullmatch(text)
    if relative is not None:
        delta = timedelta(**{_SINCE_UNITS[relative[2]]: int(relative[1])})
        return (now - delta).astimezone(UTC)
    try:
        return holds.parse_utc(text)
    except ValueError:
        raise ValueError(
            f'--since {text!r}: expected an ISO date, an ISO datetime, or <N>s, <N>m, <N>h, <N>d'
        ) from None


def _positive_seconds(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f'must be a positive number of seconds: {text}')
    return value


def default_samples_db() -> Path:
    """The host's one samples store, which the main checkout's sampler unit writes."""
    return default_project_root() / 'data' / 'load-samples.db'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='resource_report.py', description="Resource use by consumer, from the load recorder's store.")
    commands = parser.add_subparsers(dest='command', required=True)
    holds_parser = commands.add_parser(
        'holds', help='which consumers grew and which paid before each memory hold'
    )
    holds_parser.add_argument(
        '--project-root', dest='project_roots', action='append', required=True, type=Path,
        metavar='ROOT', help='a project root whose runs.db to read; repeat for each project',
    )
    holds_parser.add_argument(
        '--since', default='7d',
        help='ISO date, ISO datetime (naive is UTC), or <N>s/m/h/d before now (default 7d)',
    )
    holds_parser.add_argument(
        '--window', type=_positive_seconds, default=120,
        help='seconds before each hold to read (default 120)',
    )
    holds_parser.add_argument('--summary', action='store_true', help='add the summary across holds')
    holds_parser.add_argument('--json', action='store_true', help='print one JSON object')
    holds_parser.add_argument(
        '--samples-db', type=Path, default=None,
        help="the load recorder's store (default: the main checkout's data/load-samples.db)",
    )
    return parser


def _distinct(roots: Iterable[Path]) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(root.resolve() for root in roots))


def main(argv: Sequence[str] | None = None, *, now: datetime | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        since = parse_since(args.since, now or datetime.now(UTC))
    except ValueError as exc:
        parser.error(str(exc))
    samples_db = args.samples_db or default_samples_db()
    try:
        report = holds.read_holds_report(
            _distinct(args.project_roots), samples_db, since=since, window_seconds=args.window
        )
    except holds.SamplesUnavailable as exc:
        print(f'resource_report.py holds: {exc}', file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(render.report_json(report, summary=args.summary), indent=2))
    else:
        print(render.render_text(report, summary=args.summary))
    return 0


if __name__ == '__main__':
    sys.exit(main())
