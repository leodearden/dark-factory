"""Write-time containment tripwire for leaked MCP envelope markup (task 3141).

A recurring harness serialization bug leaks raw MCP envelope fragments — the
closing/opening tags of the tool-call wire format — into the *payload* of
fused-memory writes. Two observed vectors: memory ``content`` arriving with a
``</content>``/``</invoke>`` tail (permanent specimens now sitting in the mem0
and Graphiti corpora), and task text arriving with a ``<parameter name=``
fragment that the interceptor's description parser then mis-parses *silently*
(reify task 3210 was filed ``priority=high`` and stored as ``medium``).

This module is the CONTAINMENT half of that story (PRD
``docs/prds/memory-write-path-convergence.md`` §9 leaf ο, contract C3): it
REJECTS such a write at the boundary, loudly and with the matched pattern
named, so no further specimen enters the corpus and no parser gets the chance
to derive a wrong value from the fragment. Root cause, the Qdrant payload
text-match read tool, and the retroactive corpus sweep belong to **DF task
3083** and are deliberately out of scope here.

:data:`MCP_MARKUP_PATTERNS` is the SINGLE write-time pattern list (INV-5) —
every one of the four MCP write boundaries (``add_memory``, ``add_episode``,
``submit_task``, ``update_task``) rejects against exactly this tuple, and
nothing else in the package enumerates these literals.

Calibration vs. the retrospective scanner
-----------------------------------------
``scripts/scan_task_toolcall_leaks.py`` (task 2939) detects the *same* leak
retrospectively, but with a deliberately DIFFERENT and much narrower matcher
(its ``LEAK_TAIL`` regex — see that module's docstring for the discriminator
and the evidence behind it; it is not restated here). The two are calibrated in
opposite directions on purpose, because the cost of each error mode is
inverted:

* There, a false positive sends a human off to inspect a task that never
  leaked, so precision is bought with a discriminated regex.
* Here, a false positive is a *recoverable* rejection that carries its own
  remediation and an explicit override (``metadata={'allow_mcp_markup': True}``)
  — one retry clears it. A false NEGATIVE, by contrast, is another permanent
  corpus specimen that DF 3083 has to hunt down by eye.

So this write-time check matches bare, case-sensitive literal substrings and
accepts the resulting over-reporting. If you are tempted to "fix" one of the
two pattern sets to match the other: don't — they are answering different
questions.

Structure mirrors the sibling :mod:`fused_memory.server.near_duplicate_guard`:
module-level constants plus pure synchronous functions that do no I/O and never
raise on empty/``None`` input, so callers can hand raw handler arguments
straight through.
"""

from __future__ import annotations

# Raw MCP envelope fragments that must never appear inside a write payload.
#
# Matched as bare, CASE-SENSITIVE substrings (see the module docstring for why
# this deliberately over-reports relative to scripts/scan_task_toolcall_leaks.py).
# This is the single write-time source of truth (INV-5); the same-file drift
# guard in tests/server/test_markup_tripwire.py must be updated alongside it.
MCP_MARKUP_PATTERNS: tuple[str, ...] = ('</content>', '<parameter name=', '</invoke>')


def find_markup_pattern(text: object) -> str | None:
    """Return the first :data:`MCP_MARKUP_PATTERNS` literal occurring in *text*.

    "First" is by POSITION IN THE TEXT, not by position in the pattern tuple:
    when several patterns are present the earliest one is reported, so the
    caller is told where the leaked envelope actually starts rather than
    whichever literal happens to be listed first.

    Matching is case-sensitive — the harness emits lowercase tags, and
    case-folding would only widen the guard onto prose that shouts a tag name.

    Pure and synchronous. *text* is expected to be a handler argument's value
    (``str``) but anything else — ``None``, an absent optional field, a dict —
    returns ``None`` without raising, so call sites need no pre-validation.
    """
    if not text or not isinstance(text, str):
        return None
    best_index = -1
    best_pattern: str | None = None
    for pattern in MCP_MARKUP_PATTERNS:
        index = text.find(pattern)
        if index == -1:
            continue
        if best_index == -1 or index < best_index:
            best_index = index
            best_pattern = pattern
    return best_pattern


def find_markup_violation(fields: dict[str, object]) -> tuple[str, str] | None:
    """Return ``(field_name, matched_pattern)`` for the first violating field.

    *fields* maps a caller-facing field name (``'content'``, ``'description'``,
    …) to its raw value. Fields are checked in dict INSERTION ORDER, so a call
    site controls which field is named when several are dirty. Returns ``None``
    when every field is clean, empty, ``None`` or non-``str``.

    Pure and synchronous; never raises (an empty map is simply not a violation).
    """
    for field_name, value in fields.items():
        pattern = find_markup_pattern(value)
        if pattern is not None:
            return field_name, pattern
    return None
