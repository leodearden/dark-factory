"""Shared detector for leaked serialized tool-call XML fragments (task 3083).

THE SINGLE SOURCE OF TRUTH for "what a tool-call XML leak looks like". Three
consumers depend on this module, and they must never drift apart:

* ``scripts/scan_task_toolcall_leaks.py`` — the READ-ONLY Taskmaster task-DB
  sweep (task 2939), which is where this detector was originally written and
  hardened;
* ``fused-memory/scripts/sweep_toolcall_xml_leak.py`` — the Mem0/Qdrant
  corpus sweep;
* ``Mem0Backend.scan_payload_text`` / the ``scan_memory_content`` MCP tool —
  the read capability that makes the corpus sweepable at all.

This detector is deliberately PRECISE: it requires real whitespace between the
stray closing tag and the continuation, so prose that merely quotes the leak
shape with an escaped ``\n`` is not a hit. That matters because every consumer
above runs over ALREADY-STORED content, where a false positive would provoke an
unnecessary rewrite of a user's memory.

It is NOT the live write-boundary guard. That is
:mod:`fused_memory.server.markup_tripwire`, which is deliberately calibrated
the other way — a bare substring scan that accepts over-reporting to maximise
recall at write time, where the cost of a false positive is only a retry. The
two are complementary and must not be collapsed into one another. Since task
3688 they DO share the envelope-literal enumeration itself, which lives once
in :mod:`shared.toolcall_markup` (INV-5); the two calibrations are two named
predicates over that one set, not two independently spelled sets.

## Root cause (task 3083, WORK a)

These markers NEVER originate in this repository. There is no XML parser
anywhere in ``fused-memory/src/`` (zero hits for ``ElementTree``/``xml.etree``/
``lxml``/``BeautifulSoup``/``html.parser``), and no production write path in
fused-memory, orchestrator, shared, or escalation emits a serialized
tool-call fragment. Their presence in stored text — or in an inbound tool
argument — is therefore POSITIVE EVIDENCE that the harness's tool-call XML
parser terminated a string argument early at a literal closing tag inside
that argument's value.

That single defect has two manifestations, and this detector is what makes
them one bug in code rather than merely one bug in prose:

1. SIBLING-ARGUMENT LOSS (silent, and the dangerous one). Early termination
   swallows the arguments that followed. ``submit_task(priority: str | None
   = None)`` then receives ``None``, and ``sqlite_task_backend.py``'s
   ``priority or 'medium'`` silently substitutes a plausible wrong value. The
   intended value survives only as text inside ``description``. Three of the
   four live fixtures in ``scripts/tests/test_scan_task_toolcall_leaks.py``
   are exactly this shape.
2. CONTENT SELF-DUPLICATION (visible). The truncated argument's remainder is
   re-appended into the same field, yielding a visible tail plus a verbatim
   self-duplicate — the recorded shape of Mem0 records 9f2d2ae6 and c759c53b.

So a caller that sees a hit here must not merely strip the fragment: it must
treat the WHOLE call as suspect, because parameters it never sees may have
been dropped.

## The discriminator, and why it is deliberately conservative

A match requires BOTH halves of the early-termination signature:

* a stray closing tag (``description``/``parameter``/``details``/``content``),
  and
* one or more REAL whitespace characters (``\\s+``), and
* a continuation — either a serialized opening ``parameter`` tag or a bare
  closing ``invoke`` tag.

The real-whitespace requirement is load-bearing and must not be relaxed. It
excludes prose that merely quotes the leak shape using the ESCAPED literal
``\\n`` (two characters, backslash then ``n``, which is NOT whitespace) and
then continues with trailing prose — the tasks 2938/2939 false-positive
shape. A naive substring scan for an opening parameter tag over-reports on
exactly that, which is why this detector is trusted and a naive one is not.

Once both conditions hold, the capture group extends greedily to
end-of-string (``.*$`` under ``re.DOTALL``); that greedy tail is simply where
the fragment capture ends, not an independent discriminator.

## Sentinel-literal hazard — DO NOT "helpfully" un-escape these

Every sentinel below is spelled with the ``\\x3c`` escape for ``<``. This is
NOT stylistic. Writing ``<`` verbatim in these literals forces any agent
editing this file to emit that literal inside its own tool-call XML, which
reproduces the very bug documented above: the agent's own Write/Edit argument
terminates early, truncating this file and dropping the sibling arguments of
that same call. ``\\x3c`` is byte-identical at runtime (both in the regex
engine and in Python string literals) and never appears verbatim in the file
text, so it is immune. Leave it escaped.

This module's own code is pure and stdlib-only, and it deliberately imports
nothing else from ``fused_memory`` so that every layer — backends, services,
middleware, the MCP tools layer, and standalone scripts — can depend on it
without a cycle.

## It is NOT dependency-free — know what the import actually costs

``shared.toolcall_markup`` OWNS the envelope-literal enumeration (INV-5) and
is itself stdlib-only and pure, but importing a shared SUB-module executes
``shared/__init__.py``, which imports the whole shared public API eagerly.
Measured on this worktree, ``import shared.toolcall_markup`` therefore drags
in ``pydantic``, ``pydantic_core``, ``aiosqlite``, ``yaml``, ``dotenv``,
``annotated_types`` and ``typing_extensions``, and costs ~0.4 s warm (several
seconds cold) — none of which this module or ``toolcall_markup`` uses. So a
dependency-free system ``python`` can no longer import this module; an
interpreter that has the project's dependencies (the repo venv, ``uv run``)
still can, and every current consumer has them: fused-memory declares
``dark-factory-shared`` as a workspace dependency, and
``fused-memory/scripts/sweep_toolcall_xml_leak.py`` runs inside that venv.

A standalone script must additionally put ``shared/src`` on ``sys.path``
alongside ``fused-memory/src``; ``scripts/scan_task_toolcall_leaks.py``
carries that bootstrap, records the same cost in a comment, and is the
pattern to copy. Making a shared leaf importable without the package init
(a lazy ``__getattr__`` in ``shared/__init__.py``) would restore the
dependency-light property, but it is a shared-wide change that task 3688
held no lock for and deliberately did not take on.
"""
from __future__ import annotations

import re
from typing import NamedTuple

from shared.toolcall_markup import PARAMETER_CLOSER_NAMES, PREFILTER_NEEDLES

__all__ = [
    'LEAK_TAIL',
    'PREFILTER_NEEDLES',
    'LeakHit',
    'detect_leak',
    'find_toolcall_xml_leak',
    'has_toolcall_xml_leak',
]

# PREFILTER_NEEDLES — imported above, NOT spelled here. The enumeration is
# owned by shared.toolcall_markup (INV-5); its VALUE and ORDER are unchanged by
# the move, and tests/test_mem0_client.py zips this tuple against the Qdrant
# filter clauses with strict=True, so the order stays load-bearing.
#
# Literal substrings a store-side PREFILTER can search for cheaply. Every leak
# necessarily contains one of these — they are the closing-tag alternation of
# LEAK_TAIL, now literally so, since both are built from the same name tuple —
# so OR-ing them is a strict SUPERSET of the real match set and can never miss
# a leak the detector would find.
#
# They are a speed optimisation ONLY. The authoritative verdict is always
# find_toolcall_xml_leak() re-run over each returned record, because a bare
# closing tag is common in ordinary prose and matches nothing on its own.

# Promoted verbatim from scripts/scan_task_toolcall_leaks.py (task 2939), with
# exactly two generalizations for the Mem0 specimens (task 3083):
#   1. `content` added to the stray-closing-tag alternation — Mem0 records
#      leak a closing `content` tag, which the task-DB-era pattern never saw.
#   2. the continuation is now an alternation, so a bare closing `invoke` tag
#      counts even with no opening `parameter` tag after it — the c759c53b
#      tail shape, which the task-DB-era pattern required and therefore missed.
# The \s+ real-whitespace requirement between the two halves is UNCHANGED and
# must stay that way; see the module docstring for the 2938/2939 rationale.
LEAK_TAIL = re.compile(
    r'\x3c/(?:' + '|'.join(PARAMETER_CLOSER_NAMES) + r')>'
    r'\s+'
    r'(\x3c(?:parameter\s+name="[^"]*">|/invoke>).*)$',
    re.DOTALL,
)


class LeakHit(NamedTuple):
    """One detected leak: the matched fragment and where it starts.

    ``fragment`` is the whole match — the stray closing tag onward — so a
    caller can strip or quote it exactly. ``index`` is its 0-based offset
    into the input text, which is what a repair path needs in order to split
    the text into "body before the leak" and "leaked remainder".
    """

    fragment: str
    index: int


def find_toolcall_xml_leak(text: object) -> list[LeakHit]:
    """Return every leaked tool-call fragment in *text*, ordered by index.

    *text* may be any object: a non-str (including ``None``, an int, a dict,
    or a raw ``sqlite3`` row value) or an empty/falsy value returns ``[]``
    without raising, so callers can pass unvalidated input straight through.

    Detection runs against ``text.rstrip()`` so trailing whitespace tacked
    onto an otherwise-genuine leak does not defeat the end-anchored match.
    Because the capture runs greedily to end-of-string, at most one hit is
    possible today; the list return is the stable shape for callers and keeps
    the API honest if the pattern is ever narrowed.
    """
    if not text or not isinstance(text, str):
        return []
    return [
        LeakHit(fragment=match.group(0), index=match.start())
        for match in LEAK_TAIL.finditer(text.rstrip())
    ]


def has_toolcall_xml_leak(text: object) -> bool:
    """Return True when *text* carries a leaked tool-call fragment."""
    return bool(find_toolcall_xml_leak(text))


def detect_leak(text: object) -> str | None:
    """Return the leaked fragment in *text*, or None if it is clean.

    Compatibility shim preserving task 2939's exact contract for
    ``scripts/scan_task_toolcall_leaks.py`` and its 29 existing tests, which
    are the regression proof that promoting this detector changed nothing for
    the task-DB path.
    """
    hits = find_toolcall_xml_leak(text)
    return hits[0].fragment if hits else None
