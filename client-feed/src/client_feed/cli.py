"""The client-feed command line: argv to the owning module, its result to an exit code (PRD §7.8)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='client-feed', description='The per-project, client-safe evidence feed.'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    _add_validate_feed(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand and return its exit code."""
    args = _parser().parse_args(argv)
    return args.run(args)
