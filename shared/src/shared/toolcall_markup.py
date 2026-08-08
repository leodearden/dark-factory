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

import re
from collections.abc import Collection, Iterable
from functools import lru_cache
from typing import NamedTuple

__all__ = [
    'CANONICAL_OPENER_PREFIX',
    'ENVELOPE_LITERALS',
    'INVOKE_CLOSER',
    'MCP_MARKUP_PATTERNS',
    'PARAMETER_CLOSER_NAMES',
    'PREFILTER_NEEDLES',
    'Repair',
    'closer_for',
    'detect',
    'repair',
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


# ---------------------------------------------------------------------------
# Repair — recovering the parameters the harness parser silently dropped.
# ---------------------------------------------------------------------------

# A pseudo-parameter NAME as the harness emits it. Deliberately narrow: a tag
# whose name is not an identifier is not a dropped parameter, it is prose.
_TAG_NAME = r'[A-Za-z_]\w*'

# Candidate mis-close positions, and the closing half of a pseudo-parameter
# pair. The trailing ``"?`` is the DIALECT BLEND tolerance: PRD section 2.1's
# first specimen closes its metadata parameter with a stray double quote before
# the angle bracket (the model interpolating between the two dialects), and
# boundary row B2 requires that specimen to recover all three of its dropped
# parameters. Without the one-character optionality the blended item is
# leftover text, which rejects the candidate and makes B2 unsatisfiable.
_CLOSER_RE = re.compile(r'\x3c/(' + _TAG_NAME + r')"?>')

# The canonical opening tag, ``parameter name="X"``. Tried FIRST, because its
# closer is named ``parameter`` while a name-echoing opener closes on its own
# name — the one place the two dialects are not symmetric.
_CANONICAL_OPENER_RE = re.compile(r'\x3cparameter\s+name="([^"]+)"\s*"?>')

# The name-echoing opening tag the model drifts into, blend-tolerant. Cannot
# collide with a canonical opener (``parameter`` is followed by whitespace, not
# by the closing bracket) nor with any closing tag (``/`` is not a name char).
_ECHO_OPENER_RE = re.compile(r'\x3c(' + _TAG_NAME + r')"?>')

# Structural bounds. repair() must be total for adversarial input WITHOUT a
# blanket try/except (that is signature (b) of shared/tests/silent_fallthrough_scan.py
# and would demand an allowlist entry), so the two unbounded loops are bounded
# by construction instead. Both ceilings are far above any real call: no MCP
# tool has 64 parameters, and a value carrying 64 qualifying closers is prose
# about markup, not a leak.
_MAX_CANDIDATES = 64
_MAX_TAIL_ITEMS = 64


class Repair(NamedTuple):
    """A validated recovery of one value that absorbed its sibling arguments.

    Returned only when the repair VALIDATES (PRD section 4 C1): the tail parses
    with zero leftover, every recovered name is a real parameter of that tool,
    and none collides with an argument the caller actually supplied.
    """

    #: The caller's intended text: ``value`` up to the mis-close, as a SLICE.
    clean_value: str
    #: The dropped parameters, name -> value. Empty is a success, not a
    #: refusal — the last-parameter case (PRD boundary row B4) drops nothing.
    recovered: dict[str, str]
    #: The envelope literal :func:`detect` matched, i.e. the earliest one BY
    #: TEXT POSITION. Falls back to :attr:`misclose` when the drifted tag is
    #: not itself a literal and no literal appears anywhere in the value.
    pattern: str
    #: The wrong closing tag, verbatim as it appeared — including the dialect
    #: blend's stray quote. This is the diagnostic PRD section 2.2 says the
    #: write-time guard could not produce, because it enumerated one closer.
    misclose: str


@lru_cache(maxsize=256)
def _closer_re(name: str) -> re.Pattern[str]:
    """The blend-tolerant closing-tag matcher for one parameter *name*."""
    return re.compile(r'\x3c/' + re.escape(name) + r'"?>')


def _as_name_set(names: object) -> frozenset[str]:
    """Coerce a caller-supplied name collection to a set, totally.

    A non-iterable or a bare ``str`` (which would iterate into CHARACTERS, a
    caller bug that must never be read as a schema) yields the EMPTY set. That
    is the fail-safe direction: an empty schema qualifies no recovered name, so
    :func:`repair` refuses rather than recovering against a phantom schema.
    """
    if isinstance(names, (str, bytes)) or not isinstance(names, Iterable):
        return frozenset()
    return frozenset(name for name in names if isinstance(name, str))


def _parse_tail(tail: str) -> dict[str, str] | None:
    """Parse *tail* as a sequence of pseudo-parameters, else ``None``.

    The grammar is PRD section 4 C1's, verbatim: a name-echoing pair, a
    canonical ``parameter`` pair, or a final UNTERMINATED opener whose value
    runs to end-of-string (the parser consumed that closer as its terminator),
    with one trailing invoke closer stripped and whitespace allowed between
    items. ``None`` means the tail did not parse with ZERO leftover, which
    rejects the candidate and advances the scan — it never yields a partial
    repair.

    Every returned value is a SLICE of *tail*; nothing is rebuilt or decoded.
    """
    body = tail.rstrip()
    if body.endswith(INVOKE_CLOSER):
        body = body[: -len(INVOKE_CLOSER)].rstrip()

    recovered: dict[str, str] = {}
    pos = 0
    for _ in range(_MAX_TAIL_ITEMS):
        while pos < len(body) and body[pos].isspace():
            pos += 1
        if pos >= len(body):
            return recovered

        match = _CANONICAL_OPENER_RE.match(body, pos)
        if match is not None:
            name = match.group(1)
            closer_name = _NAME_PARAMETER
        else:
            match = _ECHO_OPENER_RE.match(body, pos)
            if match is None:
                return None  # leftover text — not a parse, so not a repair
            name = match.group(1)
            closer_name = name

        closer = _closer_re(closer_name).search(body, match.end())
        if closer is None:
            item_value = body[match.end():]  # final UNTERMINATED opener
            pos = len(body)
        else:
            item_value = body[match.end(): closer.start()]
            pos = closer.end()

        if '\x3c/' in item_value:
            # A second mis-close INSIDE the recovered tail: the value is itself
            # doubly corrupted, so its boundary is a guess. Refuse (PRD B5).
            return None
        if name in recovered:
            return None  # the same parameter twice is not a well-formed tail
        recovered[name] = item_value

    return None  # more items than any real call has — refuse rather than guess


def repair(
    value: str,
    param: str,
    schema_params: Collection[str],
    supplied: Collection[str],
) -> Repair | None:
    """Recover the parameters *value* absorbed, or ``None`` if unrepairable.

    *param* is the name of the parameter *value* was received as,
    *schema_params* every parameter name of that tool, and *supplied* the
    argument names the call actually arrived with.

    Algorithm (PRD section 4 C1, normative). Scan candidate mis-close positions
    left-to-right. A candidate closing tag for ``X`` qualifies iff ``X ==
    param``, ``X`` is in *schema_params*, or ``X`` is the canonical
    ``parameter`` closer. For each candidate the remaining tail is parsed by
    :func:`_parse_tail`; the EARLIEST candidate is accepted for which the tail
    parses with zero leftover, every recovered name is in *schema_params*, and
    no recovered name is in *supplied*. Otherwise ``None`` — never a guess.

    The canonical closer is always a candidate even though C1's literal wording
    does not list it. PRD section 2.1's fourth specimen mis-closes ``content``
    with ``/parameter``, and ``parameter`` is not a parameter of update_memory,
    so the literal reading makes one of the PRD's own four specimens
    unrepairable by construction. Admitting it cannot fabricate anything: the
    accept conditions are unchanged, so an extra candidate position can only
    ever recover more, never synthesise.

    INVARIANT D5 — ENFORCED BY CONSTRUCTION, not by assertion. ``clean_value``
    is ``value[:candidate.start()]``, a slice, so it is always a prefix of the
    input; every recovered value is a slice of the tail, so it is always a
    verbatim substring. That is the line between recovery and fabrication, and
    it holds because no branch here ever builds a string.

    Pure, synchronous, deterministic, and total: it never raises for any input.
    Totality comes from a type guard plus purely structural scanning with
    bounded loops — deliberately NOT from a broad ``except`` returning ``None``,
    which is exactly the silent-fallthrough signature the shared ratchet flags.
    """
    if not value or not isinstance(value, str):
        return None

    schema = _as_name_set(schema_params)
    already_supplied = _as_name_set(supplied)
    detected = detect(value)

    attempts = 0
    for candidate in _CLOSER_RE.finditer(value):
        name = candidate.group(1)
        if name != param and name not in schema and name != _NAME_PARAMETER:
            continue

        attempts += 1
        if attempts > _MAX_CANDIDATES:
            return None

        recovered = _parse_tail(value[candidate.end():])
        if recovered is None:
            continue
        # Accept-time conditions, so a rejected candidate simply advances the
        # scan. Boundary rows B8 (recovered name not in the tool's schema) and
        # B9 (recovered name already supplied by the caller) both land here.
        if not recovered.keys() <= schema:
            continue
        if not recovered.keys().isdisjoint(already_supplied):
            continue

        misclose = candidate.group(0)
        return Repair(
            clean_value=value[: candidate.start()],
            recovered=recovered,
            pattern=detected if detected is not None else misclose,
            misclose=misclose,
        )

    return None
