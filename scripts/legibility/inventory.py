#!/usr/bin/env python3
"""Session inventory — enumerate yesterday's sessions for a project (PRD §5.2 point 2).

Walks ``~/.claude/projects/<enc>`` (or an injected ``projects_root``, so
tests never touch the real ``~/.claude`` tree — mirrors
``session_registry.fleet_root(root=)``) for a project whose agents span
many encoded cwd directories: a project's config lists **cwd prefixes**
(``docs/legibility/legibility.yaml``'s ``cwd_prefixes``), and membership is
resolved from each session's REAL ``cwd`` (read from a transcript line) via
path-component semantics, never a raw string-prefix match on the encoded
dir name — the ``~/.claude/projects`` encoding is lossy (``/`` and ``.``
both map to ``-``), so a sibling project sharing the same literal prefix
(e.g. ``dark-factory-cockpit``) would otherwise be over-included.

Task β of the confusion-reduction PRD (plans/confusion-reduction-prd.md
§5.2, contract §7.4). Self-contained — does not import task α's
``digest.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any


def encode_cwd(cwd: str) -> str:
    """Mirror ``session_registry.transcript_path_for_cwd``'s encoding.

    Both ``/`` and ``.`` map to ``-`` — this is the same best-effort
    mirror of Claude Code's own ``~/.claude/projects/<enc>`` naming, kept
    in lockstep with the canonical implementation at
    ``orchestrator/src/orchestrator/session_registry.py:451-459``.
    """
    return cwd.replace('/', '-').replace('.', '-')


def iter_project_dirs(projects_root: Path | str, cwd_prefixes: Sequence[str]) -> Iterator[Path]:
    """Yield candidate project dirs under *projects_root* by a cheap encoded-prefix pre-filter.

    Intentionally imprecise: a directory name that merely *starts with* one
    of the encoded prefixes is yielded, even when it actually belongs to an
    unrelated sibling project (e.g. ``-home-leo-src-dark-factory-cockpit``
    starts with the encoded ``-home-leo-src-dark-factory`` prefix). This is
    only a cheap candidate filter over a directory listing that can hold
    hundreds of entries — callers MUST additionally confirm membership via
    :func:`is_member` against each session's real ``cwd``
    (:func:`session_cwd`) before treating a session as belonging to the
    project.
    """
    root = Path(projects_root)
    if not root.is_dir():
        return
    encoded_prefixes = [encode_cwd(prefix) for prefix in cwd_prefixes]
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if any(entry.name.startswith(enc) for enc in encoded_prefixes):
            yield entry


def is_member(cwd: str, cwd_prefixes: Sequence[str]) -> bool:
    """True iff *cwd* is one of *cwd_prefixes* or a descendant of one.

    Uses ``Path.is_relative_to`` path-COMPONENT semantics (never a raw
    string prefix match), so a sibling directory that merely shares a
    literal string prefix (``dark-factory-cockpit`` vs ``dark-factory``) is
    correctly excluded, while a ``.worktrees``/``.claude-worktrees`` child
    of a real prefix is correctly included.
    """
    cwd_path = Path(cwd)
    return any(cwd_path.is_relative_to(Path(prefix)) for prefix in cwd_prefixes)


def _iter_json_lines(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed dict records from a JSONL file, skipping blank/malformed lines.

    Mirrors ``digest.load_transcript``'s graceful-degrade contract: a
    transcript is written fire-and-forget and can have a truncated or
    corrupt trailing line, which must not abort the whole read. Raises
    ``OSError`` if *path* cannot be opened at all (caller's concern).
    """
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def session_cwd(path: Path) -> str | None:
    """Return the first non-empty ``cwd`` string found in *path*, else None.

    Real transcripts are heterogeneous: some early lines (``ai-title``,
    ``agent-name``, ``queue-operation``) carry no ``cwd`` at all, and a few
    metadata-only stub sessions carry no ``cwd`` anywhere in the file. Both
    an unreadable path and a cwd-less file degrade to ``None`` rather than
    raising.
    """
    try:
        for record in _iter_json_lines(path):
            cwd = record.get('cwd')
            if isinstance(cwd, str) and cwd:
                return cwd
    except OSError:
        return None
    return None


@dataclass(frozen=True)
class SessionRecord:
    """One enumerated session transcript.

    ``encoded_dir`` is the containing directory's basename (a
    ``~/.claude/projects/<enc>`` entry); ``cwd`` is the REAL decoded cwd
    read from the transcript (the value :func:`is_member` was checked
    against), not a re-derivation from ``encoded_dir``.
    """

    path: Path
    encoded_dir: str
    cwd: str
    date: date
    size_bytes: int


def _first_timestamp_date(path: Path) -> date | None:
    """Return the UTC date of the first valid ISO-8601 ``timestamp`` line in *path*.

    Returns None when the file is unreadable, has no timestamp line, or
    every timestamp line is unparseable — callers fall back to file mtime.
    """
    try:
        for record in _iter_json_lines(path):
            ts = record.get('timestamp')
            if not isinstance(ts, str) or not ts:
                continue
            try:
                parsed = datetime.fromisoformat(ts)
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC).date()
    except OSError:
        return None
    return None


def enumerate_sessions(
    projects_root: Path | str,
    cwd_prefixes: Sequence[str],
    target_date: date,
) -> list[SessionRecord]:
    """Enumerate every session transcript for *target_date* across all matching encoded dirs.

    Aggregates across every directory :func:`iter_project_dirs` yields
    (never assumes one dir per project — a project's agents span many
    encodings). For each ``*.jsonl`` file: skip non-existent/unreadable/
    empty files, confirm membership via the session's real ``cwd``
    (:func:`session_cwd` + :func:`is_member`), derive the session's UTC
    date from its first timestamp line (falling back to file mtime when no
    timestamp is present), and keep only sessions matching *target_date*.
    """
    root = Path(projects_root)
    records: list[SessionRecord] = []
    for project_dir in iter_project_dirs(root, cwd_prefixes):
        for session_path in sorted(project_dir.glob('*.jsonl')):
            try:
                size_bytes = session_path.stat().st_size
            except OSError:
                continue
            if size_bytes == 0:
                continue

            cwd = session_cwd(session_path)
            if cwd is None or not is_member(cwd, cwd_prefixes):
                continue

            session_date = _first_timestamp_date(session_path)
            if session_date is None:
                try:
                    session_date = datetime.fromtimestamp(
                        session_path.stat().st_mtime, tz=UTC
                    ).date()
                except OSError:
                    continue
            if session_date != target_date:
                continue

            records.append(
                SessionRecord(
                    path=session_path,
                    encoded_dir=project_dir.name,
                    cwd=cwd,
                    date=session_date,
                    size_bytes=size_bytes,
                )
            )
    return records


def main(argv: Sequence[str]) -> int:
    """Thin standalone CLI: list sessions matching cwd_prefixes for a UTC date.

    A debug/inspection entry point for inventory.py alone — the full
    inventory->score->sample pipeline's CLI acceptance surface is
    ``sampling.main`` (task 2573 step-18).
    """
    parser = argparse.ArgumentParser(
        description="List legibility sessions matching cwd_prefixes for a given UTC date."
    )
    parser.add_argument(
        '--projects-root',
        default=str(Path.home() / '.claude' / 'projects'),
        help='Root of the ~/.claude/projects tree (default: %(default)s)',
    )
    parser.add_argument(
        '--cwd-prefix',
        dest='cwd_prefixes',
        action='append',
        required=True,
        help='A cwd prefix to match (repeatable).',
    )
    parser.add_argument(
        '--date',
        default=None,
        help='UTC date to filter sessions to, YYYY-MM-DD (default: yesterday UTC).',
    )
    args = parser.parse_args(argv)

    target_date = (
        date.fromisoformat(args.date)
        if args.date
        else (datetime.now(UTC) - timedelta(days=1)).date()
    )
    records = enumerate_sessions(args.projects_root, args.cwd_prefixes, target_date)
    for record in records:
        print(
            f'{record.path}\t{record.encoded_dir}\t{record.cwd}\t'
            f'{record.date}\t{record.size_bytes}'
        )
    return 0


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main(sys.argv[1:]))
