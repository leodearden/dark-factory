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

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
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
