"""The client-feed command line: argv to the owning module, its result to an exit code (PRD §7.8)."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from client_feed.contract import FEED_FILENAME, FeedError
from client_feed.dump_sources import dump_sources, render_dump
from client_feed.feed_io import FeedIoError, read_feed_lines
from client_feed.fold import fold
from client_feed.policy import OutDirRefused, Policy, PolicyError, load_policy
from client_feed.publisher import publish, snapshot
from client_feed.sources.git import GitError
from client_feed.sources.store import StoreReadError
from client_feed.validate import validate_out_dir

_REFUSALS: Final = (PolicyError, OutDirRefused, FeedIoError, FeedError)
_STOP_SIGNALS: Final = (signal.SIGTERM, signal.SIGINT)
_JOIN_POLL_SECS: Final = 1.0


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


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _configure_logging() -> None:
    """The publisher's progress, logged to stderr."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')


def _publish_until_signalled(policy: Policy, clock: Callable[[], datetime]) -> None:
    """Publish on a worker thread until SIGTERM or SIGINT; the worker's own failure is raised here.

    The handlers run on this thread, which never waits on `stop` itself, so setting it cannot deadlock.
    """
    stop = threading.Event()
    failures: list[BaseException] = []

    def work() -> None:
        try:
            publish(policy, stop=stop, clock=clock)
        except BaseException as exc:
            failures.append(exc)

    previous = {number: signal.signal(number, lambda *_: stop.set()) for number in _STOP_SIGNALS}
    worker = threading.Thread(target=work, name='client-feed-publish')
    try:
        worker.start()
        while worker.is_alive():
            worker.join(_JOIN_POLL_SECS)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    if failures:
        raise failures[0]


def _run_publish(args: argparse.Namespace) -> int:
    _configure_logging()
    try:
        _publish_until_signalled(load_policy(args.config), _utc_now)
    except _REFUSALS as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


def _add_publish(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        'publish', help='publish every interval_seconds until SIGTERM or SIGINT'
    )
    parser.add_argument('--config', type=Path, required=True, help='the publisher policy YAML')
    parser.set_defaults(run=_run_publish)


def _run_snapshot(args: argparse.Namespace) -> int:
    _configure_logging()
    try:
        report = snapshot(load_policy(args.config), now=_utc_now())
    except _REFUSALS as exc:
        print(exc, file=sys.stderr)
        return 1
    print(report.model_dump_json(indent=2))
    return 0


def _add_snapshot(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        'snapshot', help='publish one pass, re-reading every task, and print its report as JSON'
    )
    parser.add_argument('--config', type=Path, required=True, help='the publisher policy YAML')
    parser.set_defaults(run=_run_snapshot)


def _run_dump_state(args: argparse.Namespace) -> int:
    try:
        lines = read_feed_lines(args.out_dir)
        if not lines:
            print(f'{FEED_FILENAME} in {args.out_dir} holds no complete line', file=sys.stderr)
            return 1
        state = fold(lines, project_id=lines[0].project_id)
    except FileNotFoundError:
        print(f'{FEED_FILENAME} not found in {args.out_dir}', file=sys.stderr)
        return 1
    except (FeedIoError, FeedError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(state.model_dump_json(indent=2))
    return 0


def _add_dump_state(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        'dump-state', help="print, as JSON, the fold of the feed's complete lines"
    )
    parser.add_argument(
        '--out-dir', type=Path, required=True, help='the directory holding feed.jsonl'
    )
    parser.set_defaults(run=_run_dump_state)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='client-feed', description='The per-project, client-safe evidence feed.'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    _add_validate_feed(subparsers)
    _add_dump_sources(subparsers)
    _add_publish(subparsers)
    _add_snapshot(subparsers)
    _add_dump_state(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand and return its exit code."""
    args = _parser().parse_args(argv)
    return args.run(args)
