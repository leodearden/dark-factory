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

The discriminator (``LOG_LINE_LEAK``) requires full log-LINE SHAPE — a
leading ISO timestamp, then a logger-name field, then a level token — rather
than a bare substring such as ``fused_memory.backends.graphiti_client``.
That is not fastidiousness: task 3286's own description quotes both that
logger name and ``httpx INFO HTTP Request`` as PROSE while instructing the
reader to grep for them, so a substring scan would flag the very task hunting
the leak. It is the same discipline ``LEAK_TAIL`` encodes in the precedent
scanner, whose docstring records the identical over-reporting lesson from
tasks 2938/2939 and solves it by requiring structural evidence.
"""
from __future__ import annotations

import re
from typing import NamedTuple

# A real log line: ISO date+time (with optional ,milliseconds), then a
# whitespace-separated logger-name field, then a standalone level token.
# All three are required — prose naming a logger does not match.
LOG_LINE_LEAK = re.compile(
    r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[,.\d]*\s+\S+\s+'
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
