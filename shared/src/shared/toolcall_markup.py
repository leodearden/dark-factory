"""THE owner of the MCP envelope-literal enumeration (task 3688, INV-5).

PRD ``plans/toolcall-markup-containment-prd.md`` section 4, contract C1.

Before this module there were two enumerations of the same literals, and their
divergence is what made the original diagnosis ambiguous (PRD section 2.2):
``fused_memory.server.markup_tripwire.MCP_MARKUP_PATTERNS`` listed one closing
tag while ``fused_memory.utils.toolcall_xml_leak.PREFILTER_NEEDLES`` listed
four, so a mis-closed ``description`` could not report its own tag and the
write-time guard blamed whatever happened to follow it. Both names are now
re-exports of the ones defined here; no third site enumerates the literals.

## One set, two named predicates — the calibration split is PRESERVED

The two consumers are calibrated in opposite directions ON PURPOSE, and this
module keeps both as NAMED PREDICATES OVER ONE LITERAL SET rather than as two
literal sets:

* :data:`MCP_MARKUP_PATTERNS` — the WRITE-time, recall-first predicate. Bare
  case-sensitive substrings that deliberately over-report, because at the write
  boundary the cost of a false positive is only a retry.
* :data:`PREFILTER_NEEDLES` — the READ-time store prefilter. A cheap strict
  SUPERSET of what the precise ``toolcall_xml_leak`` detector will confirm, run
  over ALREADY-STORED content where a false positive would provoke an
  unnecessary rewrite of a user's memory.

PRD section 7 puts re-litigating that split out of scope, so neither tuple's
VALUE or ORDER may change here. Order is load-bearing for the prefilter:
``fused-memory/tests/test_mem0_client.py`` zips it against the Qdrant filter
clauses with ``strict=True``.

:func:`detect` is the third predicate over the same set — a blanket
earliest-position scan of the UNION, which is what the boundary middleware
consumes.

## Sentinel-literal hazard — DO NOT "helpfully" un-escape these

Every literal below is spelled with the ``\\x3c`` escape for ``<``. This is NOT
stylistic, and the rationale is the same one recorded at
``fused_memory/utils/toolcall_xml_leak.py`` lines 77-86: writing ``<`` verbatim
would force any agent editing this file to emit that literal inside its own
tool-call envelope, which reproduces the very defect this module exists to
contain — the agent's own Write/Edit argument terminates early, truncating this
file and silently dropping the sibling arguments of that same call. ``\\x3c`` is
byte-identical at runtime and never appears verbatim in the file text, so it is
immune. Leave it escaped.

This module is pure and stdlib-only. It deliberately imports nothing from
``fused_memory``, ``orchestrator`` or ``escalation`` so that every layer can
depend on it without a cycle — the same constraint ``toolcall_xml_leak``
documents. It is a sub-module and is NOT re-exported from ``shared/__init__``,
following the ``mcp_envelope`` / ``proc_group`` / ``config_dir`` convention:
import it fully qualified.
"""
from __future__ import annotations

__all__ = [
    'CANONICAL_OPENER_PREFIX',
    'ENVELOPE_LITERALS',
    'INVOKE_CLOSER',
    'MCP_MARKUP_PATTERNS',
    'PARAMETER_CLOSER_NAMES',
    'PREFILTER_NEEDLES',
    'closer_for',
    'detect',
]

# ---------------------------------------------------------------------------
# The single enumeration. Everything else in this module is derived from it.
# ---------------------------------------------------------------------------

# The parameter NAMES whose closing tag the harness has been observed to emit —
# either as the canonical ``parameter`` dialect or as the name-echoing dialect
# the model drifts into. Bound to individual constants so each name string is
# written exactly once and the derived tuples can reference one by meaning
# rather than by index.
_NAME_DESCRIPTION = 'description'
_NAME_PARAMETER = 'parameter'
_NAME_DETAILS = 'details'
_NAME_CONTENT = 'content'

#: The closer names, in the ORDER the read-time prefilter has always used them.
PARAMETER_CLOSER_NAMES: tuple[str, ...] = (
    _NAME_DESCRIPTION,
    _NAME_PARAMETER,
    _NAME_DETAILS,
    _NAME_CONTENT,
)

#: The bare closing ``invoke`` tag — the terminator the parser falls back to
#: when it cannot find the closer it expected (PRD section 2.1, total drift).
INVOKE_CLOSER = '\x3c/invoke>'

#: The canonical opening-tag prefix. Deliberately a PREFIX, not a whole tag:
#: the write-time predicate matches it as a bare substring so a partially
#: serialized opener is still caught.
CANONICAL_OPENER_PREFIX = '\x3cparameter name='


def closer_for(name: str) -> str:
    """Return the closing tag for parameter *name*, e.g. ``description``.

    The one place a closing tag is spelled. Every closer in this module — and,
    via the re-exports, in ``markup_tripwire`` and ``toolcall_xml_leak`` — comes
    from here.
    """
    return '\x3c/' + name + '>'


# ---------------------------------------------------------------------------
# The derived predicates.
# ---------------------------------------------------------------------------

#: WRITE-time, recall-first. Promoted verbatim from
#: ``fused_memory.server.markup_tripwire``; value and order unchanged.
MCP_MARKUP_PATTERNS: tuple[str, ...] = (
    closer_for(_NAME_CONTENT),
    CANONICAL_OPENER_PREFIX,
    INVOKE_CLOSER,
)

#: READ-time store prefilter. Promoted verbatim from
#: ``fused_memory.utils.toolcall_xml_leak``; value and ORDER unchanged.
PREFILTER_NEEDLES: tuple[str, ...] = tuple(
    closer_for(name) for name in PARAMETER_CLOSER_NAMES
)

#: The union of both calibrations, de-duplicated, stable-ordered. This is the
#: set :func:`detect` scans; neither named predicate alone covers it.
ENVELOPE_LITERALS: tuple[str, ...] = tuple(
    dict.fromkeys((*PREFILTER_NEEDLES, *MCP_MARKUP_PATTERNS))
)


def detect(value: object) -> str | None:
    """Return the earliest :data:`ENVELOPE_LITERALS` member occurring in *value*.

    "Earliest" is by POSITION IN THE TEXT, not by position in the tuple —
    the same rule as ``markup_tripwire.find_markup_pattern``, generalised to the
    full literal set. When several literals are present the caller is told where
    the leaked envelope actually starts, rather than whichever literal happens to
    be listed first. Ties break on tuple order, which cannot arise today because
    no two literals share a prefix.

    Matching is CASE-SENSITIVE: the harness emits lowercase tags, and
    case-folding would only widen the guard onto prose that shouts a tag name.

    Pure and synchronous. *value* is expected to be a handler argument's value
    (``str``) but anything else — ``None``, an absent optional field, an int, a
    dict, ``bytes`` — returns ``None`` without raising, so call sites need no
    pre-validation.
    """
    if not value or not isinstance(value, str):
        return None
    best_index = -1
    best_literal: str | None = None
    for literal in ENVELOPE_LITERALS:
        index = value.find(literal)
        if index == -1:
            continue
        if best_index == -1 or index < best_index:
            best_index = index
            best_literal = literal
    return best_literal
