#!/usr/bin/env python3
"""Detect leaked server-log lines in a task's ``metadata.done_provenance.note``.

READ-ONLY / DETECTION-ONLY: this module and its CLI never mutate task data.
Every database connection it opens is a read-only SQLite URI
(``sqlite3.connect(f"file:{path}?mode=ro", uri=True)``), so the sweep is
structurally incapable of the auto-mutation this tool is explicitly forbidden
from doing. Remediation of any match it finds is a separate, manual,
individually-reviewed follow-up (matching the safe-vs-irreversible handling
already applied to tasks 2080 and 2865) — never bulk-edited by this script.

Background (task 3286): ``DeterministicRunner._run_predicate`` stamped a
predicate check's raw stdout tail straight into ``done_provenance.note``.
``_default_run_script`` merges stderr into stdout and returns
``decode()[-2000:]``, so a chatty script's server-log noise landed in the
note verbatim — and fused-memory's reconciliation ``_format_outcome_echo``
reads that note and appends it to a Mem0 completion-summary write, which is
how the noise reached the knowledge graph.

Task 2902 is the confirmed specimen: a 1999-char note starting mid-token,
carrying FalkorDB identity-scan WARNINGs for an unrelated project plus
``httpx`` request lines. A live read-only probe found it was 1 of 264 notes,
so this scanner's value is not bulk cleanup but catching RECURRENCE — which
only a committed, tested, re-runnable detector provides (the same reasoning
that produced the 2939 precedent this module's structure is ported from).

This is a SIBLING of ``scan_task_toolcall_leaks.py``, not an extension of it:
that module deliberately EXCLUDES the ``metadata`` column (scanning it
false-positives on stored remediation records), and this leak lives ONLY
inside ``metadata.done_provenance.note``. Folding an opposite scanning policy
into it would break its stated contract.

The discriminator (``LOG_LINE_LEAK``) requires full log-LINE SHAPE — an ISO
timestamp, then a logger-name field, then a level token — rather than a bare
substring such as ``fused_memory.backends.graphiti_client``.
That is not fastidiousness: task 3286's own description quotes both that
logger name and ``httpx INFO HTTP Request`` as PROSE while instructing the
reader to grep for them, so a substring scan would flag the very task hunting
the leak. It is the same discipline ``LEAK_TAIL`` encodes in the precedent
scanner, whose docstring records the identical over-reporting lesson from
tasks 2938/2939 and solves it by requiring structural evidence.

Those three requirements — and NOT a line-start anchor — are what carry that
precision, so the pattern is deliberately unanchored: the fixed writer emits
one-line notes prefixed ``predicate check passed (rc=0): ``, and a recurrence
would therefore appear mid-line. See the comment on ``LOG_LINE_LEAK``.

Known blind spot, shared with the writer it guards: a log line under a
``%(name)s %(message)s``-style formatter (no timestamp, no level token) is
matched by neither this discriminator nor
``deterministic_runner._LOG_LINE_RE``, so such a line can reach a note and go
unreported here. Detecting it would require naming logger substrings, which
is precisely the over-reporting failure above. The writer's cap bounds the
exposure to one ≤400-char line; a predicate script wanting a guaranteed-clean
note should emit its verdict as trailing JSON.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from typing import NamedTuple

from _task_db_scan import (
    add_db_discovery_args,
    discover_db_paths,
    format_json,
    group_matches_by_db,
    run_scan_cli,
    truncate,
)

# A real log line: ISO date+time (with optional ,milliseconds), then a
# whitespace-separated logger-name field, then a standalone level token.
# All three are required — prose naming a logger does not match.
#
# Deliberately NOT anchored at a line start.  Every note the fixed writer
# produces is a SINGLE line of the form `predicate check passed (rc=0):
# <payload>`, so any recurrence lands mid-line, after that prefix — a `^`
# anchor would make this guard structurally blind to the exact format it
# exists to guard, firing only on legacy multi-line notes.  The three
# structural requirements above carry the precision on their own; the anchor
# was never what excluded prose (prose carries no ISO timestamp).
# `re.MULTILINE` is retained solely so the trailing `$` still means
# end-of-LINE, keeping a reported match to one line.
LOG_LINE_LEAK = re.compile(
    r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[,.\d]*\s+\S+\s+'
    r'(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\b.*$',
    re.MULTILINE,
)


def detect_log_leak(text: object) -> str | None:
    """Return the first leaked log line in *text*, or None if it is clean.

    *text* is expected to be a ``done_provenance.note`` value (str) or None (an
    absent key) — falsy input or anything that isn't a str returns None
    without raising, so callers can pass a decoded JSON value straight
    through. Returns the single matched LINE, not the whole note: a leaked
    note is typically a multi-KB blob, and the one line is what a reader needs
    to confirm the match.
    """
    if not text or not isinstance(text, str):
        return None
    match = LOG_LINE_LEAK.search(text)
    if match is None:
        return None
    return match.group(0)


class NoteLeakMatch(NamedTuple):
    """One confirmed leak: which DB/task/provenance it lives in, and the line."""

    db_path: str
    tag: str
    task_id: int
    provenance_kind: str
    leak_line: str


def scan_db(db_path: str) -> list[NoteLeakMatch]:
    """Scan *db_path* read-only for leaked log lines in done_provenance notes.

    Opens the database via a read-only SQLite URI (``mode=ro``) so the scan is
    structurally incapable of mutating live task data — even while the
    fused-memory server holds the same file open in WAL mode for concurrent
    writers. This matters concretely here: task 2902's note is a preserved
    forensic specimen, and this tool must be unable to touch it.

    Every row shape is tolerated: metadata that is NULL, undecodable, or a
    JSON scalar/array rather than an object, and a ``done_provenance`` that is
    absent or not a dict, are all skipped without raising. A single bad blob
    among thousands of rows must never abort a sweep.
    """
    matches: list[NoteLeakMatch] = []
    conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        for tag, task_id, metadata in conn.execute(
            'SELECT tag, id, metadata FROM tasks'
        ):
            try:
                meta = json.loads(metadata)
            except (TypeError, ValueError):
                continue
            if not isinstance(meta, dict):
                continue
            provenance = meta.get('done_provenance')
            if not isinstance(provenance, dict):
                continue
            leak_line = detect_log_leak(provenance.get('note'))
            if leak_line is not None:
                matches.append(NoteLeakMatch(
                    db_path, tag, task_id,
                    str(provenance.get('kind', '<unknown>')), leak_line,
                ))
    finally:
        conn.close()
    return matches


_DEFAULT_MAX_LINE_LEN = 100


def format_report(
    matches: list[NoteLeakMatch], max_line_len: int = _DEFAULT_MAX_LINE_LEN,
) -> str:
    """Render *matches* as a grouped, human-readable report.

    Groups lines by ``db_path``; each line shows task_id/tag/provenance kind
    and the leak line truncated to *max_line_len* characters (the full line is
    only ever emitted by :func:`format_json`). An empty *matches* list yields
    an explicit no-leaks message instead of a blank report.
    """
    if not matches:
        return 'no leaked log lines found in done_provenance notes'

    lines: list[str] = []
    for db_path, db_matches in group_matches_by_db(matches).items():
        lines.append(f'{db_path}:')
        for m in db_matches:
            leak_line = truncate(m.leak_line, max_line_len)
            lines.append(
                f'  task_id={m.task_id} tag={m.tag} '
                f'provenance_kind={m.provenance_kind} leak_line={leak_line!r}'
            )

    distinct_tasks = {(m.db_path, m.tag, m.task_id) for m in matches}
    lines.append(f'{len(matches)} leaked notes across {len(distinct_tasks)} tasks')
    return '\n'.join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'READ-ONLY sweep for leaked server-log lines in a task\'s '
            'metadata.done_provenance.note (e.g. a timestamped '
            'fused_memory/httpx logger line stamped there by a predicate '
            "check's raw stdout tail). Detection/reporting only -- never "
            'mutates task data.'
        ),
    )
    add_db_discovery_args(
        parser,
        json_help='Emit a JSON array (full untruncated leak lines) instead of a report.',
    )
    parser.add_argument(
        '--max-line-len', type=int, default=_DEFAULT_MAX_LINE_LEN,
        help='Truncate leak lines in the human-readable report to this length '
        '(default: %(default)s). Ignored with --json.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Exit codes: 0 = clean, 1 = at least one leak found, 2 = no tasks.db could
    be resolved from --db / --project-root / DASHBOARD_KNOWN_PROJECT_ROOTS /
    the dark-factory default.

    A single unreadable database (e.g. a stale/corrupt file, or a transient
    "database is locked"/"file is not a database" condition) does not abort
    the sweep: it is logged to stderr and skipped so every other resolvable
    database is still scanned and reported.
    """
    def _render(matches: list[NoteLeakMatch], args: argparse.Namespace) -> str:
        if args.json:
            return format_json(matches)
        return format_report(matches, max_line_len=args.max_line_len)

    return run_scan_cli(
        argv, parser=_build_parser(), scan_fn=scan_db, render=_render
    )


if __name__ == '__main__':
    sys.exit(main())
