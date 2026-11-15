#!/usr/bin/env python3
"""Answer "where is the transcript of session <id>" without a ``find /``.

The archive layout's one home is
``shared/src/shared/transcript_archive.py::durable_archive_path``; this tool calls
it per archive key dir rather than re-globbing the layout (invariant I-E).
Exit codes: 0 found, 1 searched and missed, 2 usage, 3 no enclosing checkout.
Task 4637.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import _checkout as _checkout
from shared.project_layout import CheckoutRootError, agent_transcripts_dir, main_checkout_of
from shared.transcript_archive import durable_archive_path

EXIT_FOUND = 0
EXIT_NOT_FOUND = 1
EXIT_NO_CHECKOUT = 3

ARCHIVE_STORE = 'agent-transcript archive'
INTERACTIVE_STORE = 'interactive sessions'


@dataclass(frozen=True)
class RootSearch:
    root: Path
    store: str
    matches: tuple[Path, ...]
    problem: str | None


def _session_id(text: str) -> str:
    try:
        return str(uuid.UUID(text))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'{text!r} is not a full session id: a UUID is required, because the '
            f'lookup matches whole ids only'
        ) from None


def resolve_project_root(explicit: Path | None, cwd: Path) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return main_checkout_of(cwd)


def _search_entries(
    root: Path, store: str, locate: Callable[[Path], Path | None]
) -> RootSearch:
    if not root.exists():
        return RootSearch(root, store, (), 'absent')
    try:
        located = [locate(entry) for entry in root.iterdir()]
    except OSError as exc:
        return RootSearch(root, store, (), f'unreadable: {exc}')
    return RootSearch(root, store, tuple(sorted(p for p in located if p is not None)), None)


def search_archive(archive_root: Path, session_id: str) -> RootSearch:
    return _search_entries(
        archive_root,
        ARCHIVE_STORE,
        lambda key_dir: durable_archive_path(archive_root, key_dir.name, session_id),
    )


def search_cli_projects(projects_dir: Path, session_id: str) -> RootSearch:
    # The Claude CLI's own <config>/projects/<enc>/<sid>.jsonl layout; its other
    # home is shared/src/shared/cli_invoke.py::_resolve_transcript_path.
    def locate(encoded_dir: Path) -> Path | None:
        transcript = encoded_dir / f'{session_id}.jsonl'
        return transcript if transcript.is_file() else None

    return _search_entries(projects_dir, INTERACTIVE_STORE, locate)


def _miss_report(session_id: str, searches: Sequence[RootSearch]) -> str:
    lines = [f'no transcript for session {session_id} in:']
    for search in searches:
        suffix = f' ({search.problem})' if search.problem else ''
        lines.append(f'  {search.root}  [{search.store}]{suffix}')
    return '\n'.join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument('session_id', type=_session_id, help='the full session id (a UUID)')
    parser.add_argument(
        '--project-root',
        type=Path,
        default=None,
        help='the main checkout whose archive to search (default: the one enclosing cwd)',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    session_id: str = args.session_id
    cwd = Path.cwd()
    try:
        project_root = resolve_project_root(args.project_root, cwd)
    except CheckoutRootError as exc:
        print(
            f'cannot find a project archive from {cwd}: {exc}. '
            f'Pass --project-root <main checkout>.',
            file=sys.stderr,
        )
        return EXIT_NO_CHECKOUT
    searches = (
        search_archive(agent_transcripts_dir(project_root), session_id),
        search_cli_projects(Path.home() / '.claude' / 'projects', session_id),
    )
    matches = [path for search in searches for path in search.matches]
    for path in matches:
        print(path)
    if matches:
        return EXIT_FOUND
    print(_miss_report(session_id, searches), file=sys.stderr)
    return EXIT_NOT_FOUND


if __name__ == '__main__':
    sys.exit(main())
