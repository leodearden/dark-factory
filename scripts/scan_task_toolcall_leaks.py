#!/usr/bin/env python3
"""Detect leaked serialized tool-call XML fragments in Taskmaster task text.

READ-ONLY / DETECTION-ONLY: this module (and its CLI, added in a later
step) never mutates task text. Every database connection it opens is a
read-only SQLite URI (``sqlite3.connect(f"file:{path}?mode=ro", uri=True)``),
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

The discriminator (``LEAK_TAIL``) requires REAL whitespace between the
stray closing tag and the ``<parameter name="...">`` fragment, and requires
the fragment to run all the way to end-of-string. This deliberately excludes
prose that merely *mentions* the leak shape (e.g. tasks 2938/2939, which
quote the ESCAPED literal ``\\n`` — two characters, backslash then ``n`` —
and continue with trailing prose afterward): a naive ``<parameter name=``
substring scan over-reports on exactly this shape.
"""
from __future__ import annotations

import re

LEAK_TAIL = re.compile(
    r'</(?:description|parameter|details)>\s*(<parameter\s+name="[^"]*">.*)$',
    re.DOTALL,
)

# Task text columns scanned for leaks. `metadata` is deliberately excluded —
# it legitimately stores remediation records (e.g. task 2865's
# metadata.stage2_description_corruption_fix.stripped_fragment) that contain
# this exact marker; scanning it would false-positive on already-fixed tasks.
SCANNED_COLUMNS = ("title", "description", "details", "test_strategy")


def detect_leak(text: object) -> str | None:
    """Return the leaked fragment in *text*, or None if it is clean.

    *text* is expected to be a task text column's value (str) or None (a
    NULL column) — falsy input or anything that isn't a str returns None
    without raising, so callers can pass a raw sqlite3 row value straight
    through. The match starts at the stray closing tag
    (``</description>``/``</parameter>``/``</details>``) and runs to
    end-of-string (after ``text.rstrip()``, so trailing whitespace tacked
    onto an otherwise-genuine leak does not defeat detection).
    """
    if not text or not isinstance(text, str):
        return None
    match = LEAK_TAIL.search(text.rstrip())
    if match is None:
        return None
    return match.group(0)
