"""The client-feed command line: argv to the owning module, its result to an exit code (PRD §7.8)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from client_feed.dump_sources import dump_sources, render_dump
from client_feed.policy import PolicyError, load_policy
from client_feed.sources.git import GitError
from client_feed.sources.store import StoreReadError
from client_feed.validate import validate_out_dir


def _run_validate_feed(args: argparse.Namespace) -> int:
    report = validate_out_dir(args.out_dir, expected_project_id=args.project_id)
    print(report.model_dump_json(indent=2))
    return 0 if report.ok else 1


def _add_validate_feed(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        'validate-feed', help='check an out/ directory against the feed contract and its fold'
    )
    parser.add_argument(
        '--out-dir', type=Path, required=True, help='the directory holding feed.jsonl and state.json'
    )
    parser.add_argument(
        '--project-id', default=None, help="the project the feed must name (default: state.json's)"
    )
    parser.set_defaults(run=_run_validate_feed)


def _run_dump_sources(args: argparse.Namespace) -> int:
    try:
        policy = load_policy(args.config)
        dump = dump_sources(
            policy, now=datetime.now(UTC), after_id=args.after_id, limit=args.limit
        )
    except (PolicyError, StoreReadError, GitError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(render_dump(dump))
    return 0


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f'must be at least 1, got {value}')
    return value


def _add_dump_sources(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        'dump-sources', help="print, as JSON, every store source the publisher reads under a policy's root"
    )
    parser.add_argument('--config', type=Path, required=True, help='the publisher policy YAML')
    parser.add_argument(
        '--after-id', type=int, default=0, help='the events cursor: read rows after this id (default 0)'
    )
    parser.add_argument(
        '--limit', type=_positive_int, default=1000, help='the most event rows to read (default 1000)'
    )
    parser.set_defaults(run=_run_dump_sources)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='client-feed', description='The per-project, client-safe evidence feed.'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    _add_validate_feed(subparsers)
    _add_dump_sources(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand and return its exit code."""
    args = _parser().parse_args(argv)
    return args.run(args)
