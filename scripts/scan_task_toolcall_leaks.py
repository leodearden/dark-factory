#!/usr/bin/env python3
"""Detect leaked serialized tool-call XML fragments in Taskmaster task text.

READ-ONLY / DETECTION-ONLY: this module and its CLI never mutate task text.
Every database connection it opens is a read-only SQLite URI
(``sqlite3.connect(f"file:{path}?mode=ro", uri=True)``),
so the sweep is structurally incapable of the auto-mutation this tool is
explicitly forbidden from doing. Remediation of any match this tool finds is
a separate, manual, individually-reviewed follow-up (matching the
safe-vs-irreversible handling already applied to tasks 2080 and 2865) —
never bulk-edited by this script.

Background: a recurring recon Stage-2 serialization bug leaks a stray
``</description>``/``</parameter>``/``</details>`` closing tag, followed by
a real newline and one or more serialized ``<parameter name="...">``
tool-call fragments, into a task's description/details column. First seen
as an apparent one-off on task 2080 (2026-07-04); recurred on task 2865 and,
per a live read-only probe run while planning this task, on ~32 further
tasks — hence this durable, re-runnable detector (task 2939) rather than a
one-off manual fix.

The discriminator (``LEAK_TAIL``) requires one or more REAL whitespace
characters (``\\s+``) between the stray closing tag and the
``<parameter name="...">`` fragment — a closing tag immediately followed by
``<parameter`` with zero whitespace in between does not match. This
deliberately excludes prose that merely *mentions* the leak shape (e.g.
tasks 2938/2939, which quote the ESCAPED literal ``\\n`` — two characters,
backslash then ``n``, which is NOT a real whitespace character — and
continue with trailing prose afterward): a naive ``<parameter name=``
substring scan over-reports on exactly this shape. Once those two
conditions hold (the closing-tag literal, then real whitespace), the
capture group extends greedily to end-of-string (``.*$``); that greedy
tail is simply where the fragment capture ends, not an independent
discriminator in its own right.

Single source of truth (task 3083): ``LEAK_TAIL`` and ``detect_leak`` are
no longer defined here. They live in
:mod:`fused_memory.utils.toolcall_xml_leak` and are re-exported by this
module so its public surface is unchanged. That promotion is deliberate —
the SAME detector also backs the Mem0/Qdrant corpus sweep
(``fused-memory/scripts/sweep_toolcall_xml_leak.py``) and the
``scan_memory_content`` read tool, so those consumers cannot drift apart on
what counts as a leak. The shared module was also generalized for the Mem0
specimens: a stray ``content`` closing tag now counts, as does a bare
closing ``invoke`` tag as the continuation. The real-whitespace
discriminator described above is unchanged.

Note the division of labour with :mod:`fused_memory.server.markup_tripwire`,
which owns the LIVE write-boundary rejection at the
``submit_task``/``update_task``/``add_memory``/``add_episode`` MCP tools and
is the authoritative enumeration of the envelope literals. That guard is
deliberately broader (a bare substring scan, accepting over-reporting to
maximise recall at write time). This detector is deliberately PRECISE,
because it runs over an already-stored corpus where a false positive would
provoke an unnecessary content rewrite. The two are not redundant and must
not be collapsed into one another.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import NamedTuple

from _task_db_scan import (
    add_db_discovery_args,
    format_json,
    group_matches_by_db,
    run_scan_cli,
    truncate,
)

# THE detector lives in fused_memory.utils.toolcall_xml_leak and is re-exported
# here (task 3083). It is deliberately NOT redefined locally: a second copy of
# this regex would drift from the Mem0 corpus sweep, silently reopening the
# ambiguity the promotion closed. See that module's docstring for the
# discriminator's rationale and the root-cause finding. Fix any import problem
# here rather than re-inlining the pattern.
#
# This is a standalone CLI: it must import cleanly when run by a bare
# interpreter with no fused-memory venv active (the documented read-only sweep
# is invoked straight from a shell), and it must resolve `fused_memory` to THIS
# checkout rather than to whatever editable install happens to be on the path —
# otherwise a worktree run silently tests the main checkout's copy of the
# detector. Prepending our own src/ satisfies both: the fused-memory editable
# install is an ordinary .pth path entry, so sys.path order decides the winner.
# Mirrors the bootstrap convention in scripts/reviewer_redundancy_diagnostic.py.
_FM_SRC = Path(__file__).resolve().parent.parent / "fused-memory" / "src"
if str(_FM_SRC) not in sys.path:
    sys.path.insert(0, str(_FM_SRC))

from fused_memory.utils.toolcall_xml_leak import (  # noqa: E402
    LEAK_TAIL,
    detect_leak,
)

# Task text columns scanned for leaks. `metadata` is deliberately excluded —
# it legitimately stores remediation records (e.g. task 2865's
# metadata.stage2_description_corruption_fix.stripped_fragment) that contain
# this exact marker; scanning it would false-positive on already-fixed tasks.
SCANNED_COLUMNS = ("title", "description", "details", "test_strategy")


class LeakMatch(NamedTuple):
    """One confirmed leak: which DB/task/column it lives in and the fragment."""

    db_path: str
    tag: str
    task_id: int
    column: str
    fragment: str


def scan_db(db_path: str) -> list[LeakMatch]:
    """Scan *db_path* read-only for leaked tool-call fragments.

    Opens the database via a read-only SQLite URI (``mode=ro``) so the scan
    is structurally incapable of mutating live task text — even while the
    fused-memory server holds the same file open in WAL mode for concurrent
    writers. Applies :func:`detect_leak` to each of ``SCANNED_COLUMNS`` per
    row; ``metadata`` is deliberately never read (see module docstring).
    """
    matches: list[LeakMatch] = []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute(
            "SELECT tag, id, title, description, details, test_strategy FROM tasks"
        )
        for tag, task_id, title, description, details, test_strategy in cursor:
            values = {
                "title": title,
                "description": description,
                "details": details,
                "test_strategy": test_strategy,
            }
            for column in SCANNED_COLUMNS:
                fragment = detect_leak(values[column])
                if fragment is not None:
                    matches.append(LeakMatch(db_path, tag, task_id, column, fragment))
    finally:
        conn.close()
    return matches


_DEFAULT_MAX_FRAGMENT_LEN = 80


def format_report(matches: list[LeakMatch], max_fragment_len: int = _DEFAULT_MAX_FRAGMENT_LEN) -> str:
    """Render *matches* as a grouped, human-readable report.

    Groups lines by ``db_path``; each line shows task_id/tag/column and the
    fragment truncated to *max_fragment_len* characters (the full fragment
    is only ever emitted by :func:`format_json`). Ends with a summary line
    counting fragments found and distinct tasks affected (a task leaking in
    two columns counts once toward "tasks"). An empty *matches* list yields
    an explicit no-leaks message instead of a blank report.
    """
    if not matches:
        return "no leaked tool-call fragments found"

    lines: list[str] = []
    for db_path, db_matches in group_matches_by_db(matches).items():
        lines.append(f"{db_path}:")
        for m in db_matches:
            fragment = truncate(m.fragment, max_fragment_len)
            lines.append(
                f"  task_id={m.task_id} tag={m.tag} column={m.column} fragment={fragment!r}"
            )

    distinct_tasks = {(m.db_path, m.tag, m.task_id) for m in matches}
    lines.append(f"{len(matches)} leaked fragments across {len(distinct_tasks)} tasks")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "READ-ONLY sweep for leaked serialized tool-call XML fragments in "
            "Taskmaster task text (e.g. a stray </description> followed by a "
            'serialized <parameter name="..."> tool-call fragment). '
            "Detection/reporting only -- never mutates task text."
        ),
    )
    add_db_discovery_args(
        parser,
        json_help="Emit a JSON array (full untruncated fragments) instead of a report.",
    )
    parser.add_argument(
        "--max-fragment-len", type=int, default=_DEFAULT_MAX_FRAGMENT_LEN,
        help="Truncate fragments in the human-readable report to this length "
        "(default: %(default)s). Ignored with --json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Exit codes: 0 = clean, 1 = at least one leak found, 2 = no tasks.db
    could be resolved from --db / --project-root / DASHBOARD_KNOWN_PROJECT_ROOTS
    / the dark-factory default.

    A single unreadable database (e.g. a stale/corrupt file, or a transient
    "database is locked"/"file is not a database" condition) does not abort
    the sweep: it is logged to stderr and skipped so every other resolvable
    database is still scanned and reported.
    """
    def _render(matches: list[LeakMatch], args: argparse.Namespace) -> str:
        if args.json:
            return format_json(matches)
        return format_report(matches, max_fragment_len=args.max_fragment_len)

    return run_scan_cli(
        argv, parser=_build_parser(), scan_fn=scan_db, render=_render
    )


if __name__ == "__main__":
    sys.exit(main())
