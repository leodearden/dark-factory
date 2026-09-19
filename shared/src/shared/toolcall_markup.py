"""THE owner of the MCP envelope-literal enumeration (task 3688, INV-5).

PRD ``plans/toolcall-markup-containment-prd.md`` section 4, contract C1.

Before this module there were two enumerations of the same literals, and their
divergence is what made the original diagnosis ambiguous (PRD section 2.2):
``fused_memory.server.markup_tripwire.MCP_MARKUP_PATTERNS`` listed one closing
tag while ``fused_memory.utils.toolcall_xml_leak.PREFILTER_NEEDLES`` listed
four, so a mis-closed ``description`` could not report its own tag and the
write-time guard blamed whatever happened to follow it. Both names are now
re-exports of the ones defined here; no third site enumerates the literals.

## One set, FOUR named predicates — the calibration split is PRESERVED

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
earliest-position scan of the UNION, for callers with NO parameter in hand.

:func:`detect_for` is the fourth, and it exists because the other three were
structurally blind to the dominant leak dialect (task **4696**). The fixed set
above echoes no INVOKED TOOL'S OWN parameter names, while :func:`repair` has
always qualified a candidate on ``X == param`` / ``X in schema_params``. A
value mis-closed with its own name-echoing tag — ``plan-tools`` writes
``rationale``, ``how``, ``decision``, ``what`` — therefore matched no literal,
every gate that asked :func:`detect` first returned ``None``, and the corrupt
value was waved straight to disk even though the repairer standing behind that
gate could already fix it. That asymmetry, a schema-aware repairer behind a
schema-blind detector, WAS the silent write path. Measured over
``.worktrees/.task-meta/*/plan.json`` on 2026-08-25: 444 corrupted entries, of
which **212 (48%) are invisible to the fixed set** — and **212 of 212** of
those are caught by the SELF-NAME closer alone.

:func:`detect_for` widens the scan with ``closer_for(param)`` and the callers'
``schema_params``; it enumerates NO new literal (INV-5) because every added
needle is built by :func:`closer_for`. :func:`detect` is deliberately left
byte-for-byte as it was rather than growing an optional keyword: three call
sites legitimately have no parameter in hand (:func:`repair`'s own diagnostic,
the sweep's bare list items, prose scans), and an optional keyword would make
each site's blindness ungreppable. The predicates split by NAME so a reader can
see at a glance which gates are parameter-aware.

## What ``repair`` guarantees about its OWN output — and what it does not

Two outputs, two different contracts. Confusing them is how this module's one
narrowing (task **4502**) reads as a weakening when it is not.

* ``Repair.clean_value`` is the value the repairer REWROTE, and it is
  ENVELOPE-FREE: ``detect_for(clean_value, param, schema_params) is None``.
  That is contract C1's post-condition, it is what the C2 middleware forwards
  as the repaired argument, and it is UNCHANGED — non-negotiable, because a
  residual envelope there would re-trip the write-time tripwire downstream
  while permanently dropping whatever hid in the residue.
* ``Repair.recovered`` values are VERBATIM CALLER TEXT, guaranteed by invariant
  D5 to be substrings of the input. A recovered value MAY legitimately contain
  an envelope literal, because a faithful REPORT of a markup leak necessarily
  quotes the pattern that tripped the tripwire. Refusing to deliver it drops
  the caller's own characters on the floor — the exact information loss this
  module exists to end. Boundary row B5 is therefore an alternative-boundary
  test (:func:`_inner_closer_blocks`), not a bare substring refusal.

The C2 middleware makes that quoting COUNTABLE rather than silent: it publishes
the recovered parameter names whose delivered value still trips
:func:`detect` on the ``markup_detected`` fact and on both policy payloads.

## Staged API — the consumer of :func:`repair` and :func:`detect`

This is task ALPHA of a staged PRD: it shipped its API one task ahead of the
consumer. That consumer has since landed —
:mod:`shared.mcp_markup_middleware` (``MarkupGuardMiddleware``, task **3689**,
contract C2) wires :func:`detect` and :func:`repair` at the FastMCP boundary,
and is a pure boundary layer over them, contributing policy, facts and the
storm escape but no parsing of its own. Task **3690** registers that middleware
on the servers.

Until 3690 lands, the middleware is CONSTRUCTED only by its own tests, so a
dead-code sweep can still misread this chain as unreachable. It is intentional
staging — check 3690 before concluding otherwise.

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

This module is pure and stdlib-only (``re``, ``json``). It deliberately imports
nothing from ``fused_memory``, ``orchestrator`` or ``escalation`` so that every
layer can depend on it without a cycle — the same constraint
``toolcall_xml_leak`` documents. It is a sub-module and is NOT re-exported from ``shared/__init__``,
following the ``mcp_envelope`` / ``proc_group`` / ``config_dir`` convention:
import it fully qualified.
"""
from __future__ import annotations

import json
import re
from collections.abc import Collection, Iterable
from functools import lru_cache
from typing import Any, NamedTuple

__all__ = [
    'CANONICAL_OPENER_PREFIX',
    'ENVELOPE_LITERALS',
    'INVOKE_CLOSER',
    'MARKUP_OVERRIDE_KEY',
    'MCP_MARKUP_PATTERNS',
    'PARAMETER_CLOSER_NAMES',
    'PREFILTER_NEEDLES',
    'Repair',
    'closer_for',
    'detect',
    'detect_for',
    'markup_override_requested',
    'repair',
    'strip_markup_override',
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


#: ONE pass over the whole literal set, as an alternation in tuple order.
#:
#: ``re.search`` returns the LEFTMOST match and, at one position, tries the
#: branches left to right — which is exactly :func:`detect`'s contract
#: (earliest by text position, ties broken on tuple order), so this is a
#: like-for-like replacement for the per-literal ``str.find`` loop it grew out
#: of, not a redefinition.
#:
#: Why it matters that it is one pass: :mod:`shared.mcp_markup_middleware` calls
#: :func:`detect` on every string argument of every tool call on the server, and
#: the corruption rate measured in PRD section 2.3 is 0.26%. A loop over the six
#: literals paid six full passes over every value on the 99.74% clean path, over
#: values the corpus shows reaching tens of KB. Each branch is ``re.escape``\\ d,
#: so ``match.group(0)`` is always exactly one member of
#: :data:`ENVELOPE_LITERALS`.
_ENVELOPE_RE = re.compile('|'.join(re.escape(literal) for literal in ENVELOPE_LITERALS))


# A pseudo-parameter NAME as the harness emits it. Deliberately narrow: a tag
# whose name is not an identifier is not a dropped parameter, it is prose.
#
# ONE grammar, bounding BOTH halves of the gate/repairer pair. :data:`_CLOSER_RE`
# builds the repairer's candidate matcher from it, so a mis-close whose name
# falls outside this shape can never be QUALIFIED for repair; :func:`_extra_names`
# applies it to the gate's widening vocabulary for exactly that reason, so the
# gate cannot spell a needle the repairer standing behind it could never act on.
# Keeping the two aligned is what stops a widened detection from routing
# authored text into the human queue for nothing — the mirror image of the
# schema-aware-repairer-behind-a-schema-blind-detector asymmetry this module's
# docstring identifies as the original silent write path.
_TAG_NAME = r'[A-Za-z_]\w*'

#: :data:`_TAG_NAME` as a whole-string test, for callers holding a NAME rather
#: than scanning text for one.
_TAG_NAME_RE = re.compile(_TAG_NAME + r'\Z')

#: The canonical empty widening set, so the overwhelmingly common
#: "no schema in hand" call shape reuses one object instead of building one.
_NO_NAMES: frozenset[str] = frozenset()


def detect(value: object) -> str | None:
    """Return the earliest :data:`ENVELOPE_LITERALS` member occurring in *value*.

    "Earliest" is by POSITION IN THE TEXT, not by position in the tuple —
    the same rule as ``markup_tripwire.find_markup_pattern``, generalised to the
    full literal set. When several literals are present the caller is told where
    the leaked envelope actually starts, rather than whichever literal happens to
    be listed first. Ties break on tuple order, which cannot arise today because
    no two literals share a prefix.

    ONE scan of *value*, via :data:`_ENVELOPE_RE`, not one per literal — see that
    constant for why the difference is load-bearing at the middleware boundary.

    Matching is CASE-SENSITIVE: the harness emits lowercase tags, and
    case-folding would only widen the guard onto prose that shouts a tag name.

    Pure and synchronous. *value* is expected to be a handler argument's value
    (``str``) but anything else — ``None``, an absent optional field, an int, a
    dict, ``bytes`` — returns ``None`` without raising, so call sites need no
    pre-validation.
    """
    if not value or not isinstance(value, str):
        return None
    match = _ENVELOPE_RE.search(value)
    return match.group(0) if match is not None else None


@lru_cache(maxsize=256)
def _widened_re(extra_names: frozenset[str]) -> re.Pattern[str]:
    """:data:`_ENVELOPE_RE` widened with the closer of every *extra_names* name.

    ONE compiled alternation, exactly like :data:`_ENVELOPE_RE` and for exactly
    its reason — the widened predicate sits on the same per-tool-call boundary,
    so it must stay one pass over the value rather than becoming a loop over
    ``6 + len(extra_names)`` needles.

    :data:`ENVELOPE_LITERALS` comes FIRST so the fixed set keeps its documented
    tuple order; the added closers are sorted so the pattern (and therefore the
    cache) is deterministic. Neither ordering is observable today: no two
    members can match at the same position, since every closer is
    ``\x3c/NAME>`` and two different names differ before the bracket.

    An EMPTY *extra_names* returns the module-level :data:`_ENVELOPE_RE` object
    itself, so a caller with no usable parameter allocates nothing and gets
    byte-identical behaviour to :func:`detect`.
    """
    if not extra_names:
        return _ENVELOPE_RE
    literals = (
        *ENVELOPE_LITERALS,
        *(closer_for(name) for name in sorted(extra_names)),
    )
    return re.compile('|'.join(re.escape(literal) for literal in literals))


@lru_cache(maxsize=512)
def _extra_names(param: str, schema_params: frozenset[str]) -> frozenset[str]:
    """The widening vocabulary for one ``(param, schema_params)`` pair.

    CACHED, because :func:`detect_for` sits on a per-tool-call boundary and
    answers ``None`` for 99.7% of the values it sees — so on the dominant path
    its whole cost is setup that finds nothing. The distinct pairs a running
    server produces are bounded by its tool table, not by its traffic, so the
    normalization is genuinely a once-per-pair computation that was being
    redone per call.

    Both arguments are pre-coerced by the caller and HASHABLE: *param* is a
    ``str`` (``''`` when the caller had none) and *schema_params* a frozenset,
    produced by the same fail-safe :func:`_as_name_set` :func:`repair` uses, so
    the cache key can never be the caller's own mutable object.

    Two names are dropped, for two different reasons:

    * one whose closer is ALREADY in :data:`ENVELOPE_LITERALS` — re-adding it
      would enumerate a literal twice (INV-5) and change nothing;
    * one outside the :data:`_TAG_NAME` shape — see that constant. The gate's
      widening vocabulary is held exactly equal to the repairer's candidate
      grammar, so a needle can never be spelled for a name :func:`repair` would
      refuse to qualify. It also bounds what a caller-controlled *param* can
      turn into a cache key: ``_first_markup_argument`` passes each key of the
      caller's argument mapping straight through.
    """
    names = set(schema_params)
    if param:
        names.add(param)
    return frozenset(
        name
        for name in names
        if _TAG_NAME_RE.match(name) and closer_for(name) not in ENVELOPE_LITERALS
    )


def detect_for(
    value: object,
    param: object,
    schema_params: object = (),
) -> str | None:
    """:func:`detect`, plus the closing tag of *param* and of *schema_params*.

    THE PARAMETER-AWARE GATE (task **4696**). *param* is the name the value was
    received as and *schema_params* every parameter name of that tool — the
    same two arguments :func:`repair` already qualifies its candidates on, so a
    gate that asks this predicate can no longer be blind to a dialect the
    repairer behind it can fix. See the module docstring for the measurement
    that motivated it (212 of 444 corrupted entries invisible to the fixed set;
    212 of those 212 caught by the SELF-NAME closer alone).

    A STRICT SUPERSET of :func:`detect`: every fixed literal is still scanned,
    still reported EARLIEST BY TEXT POSITION rather than by tuple order, and
    still case-sensitively. Widening can only ADD needles, never shadow,
    reorder or suppress the existing set.

    NO NEW LITERAL IS ENUMERATED (INV-5). Every added needle is built by
    :func:`closer_for`, the one place a closing tag is spelled; a name whose
    closer is already in :data:`ENVELOPE_LITERALS` is dropped rather than
    re-added, and so is a name outside the :data:`_TAG_NAME` shape the
    repairer's own candidate grammar accepts.

    Total, on the same terms as :func:`detect` and for the same reason — the
    gates that call it must need no pre-validation. A *value* that is not a
    non-empty ``str`` returns ``None``. A *param* that is not a non-empty
    ``str`` is DROPPED, degrading to exactly :func:`detect`'s behaviour rather
    than to a degenerate empty-name tag that would match prose. A
    *schema_params* that is not a collection of names — including a bare
    ``str``, which would iterate into CHARACTERS and manufacture one-letter
    needles — contributes nothing, via the same fail-safe :func:`repair` uses.

    COST on the clean path: the widening is normalized once per distinct
    ``(param, schema_params)`` pair by :func:`_extra_names` and compiled once
    per distinct result by :func:`_widened_re`, so a repeat call is two cache
    lookups and one scan of *value*. The no-schema shape — every middleware
    call site — reuses :data:`_NO_NAMES` and allocates nothing at all.
    """
    if not value or not isinstance(value, str):
        return None
    names = _extra_names(
        param if isinstance(param, str) else '',
        _as_name_set(schema_params),
    )
    match = _widened_re(names).search(value)
    return match.group(0) if match is not None else None


# ---------------------------------------------------------------------------
# Repair — recovering the parameters the harness parser silently dropped.
# ---------------------------------------------------------------------------

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
#
# A BOUNDED STEP COUNT IS ONLY A BOUND ON COST WHILE EACH STEP STAYS CHEAP, and
# these ceilings MULTIPLY: candidates x tail items x inner closers. Task 4502's
# ambiguity probe briefly made the innermost step O(len(body)) by slicing, which
# these ceilings do not contain — see :func:`_parse_body`'s *start* parameter.
# Anything added inside these loops must be O(1) in the input length.
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
    #: The needle :func:`detect_for` matched for this ``(value, param,
    #: schema_params)`` triple, i.e. the earliest one BY TEXT POSITION over the
    #: fixed literals widened by the names the repair itself qualifies on.
    #: Falls back to :attr:`misclose` when no needle appears anywhere.
    #:
    #: ONE PATTERN PER EVENT. Every gate in front of :func:`repair` asks that
    #: same predicate on those same inputs, so the value published here, the
    #: fact's ``pattern`` and the caller's ``matched_pattern`` agree BY
    #: CONSTRUCTION rather than by convention — and all three name the HEAD of
    #: the leak rather than whatever fixed literal happens to trail it (PRD
    #: section 2.2). Asking the blanket :func:`detect` here instead is what
    #: made one event publishable with two answers; see the accept site.
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

    :func:`detect_for` is its second consumer (task 4696), deliberately so: the
    GATE and the REPAIRER must agree on what counts as a schema, or a value the
    gate widened onto could reach a repairer that then refuses to qualify it.
    It is defined here rather than beside :func:`detect_for` because
    :func:`repair` was its first consumer and this is where its fail-safe
    direction is argued.

    An empty or unusable collection yields the canonical :data:`_NO_NAMES`
    object rather than a fresh empty frozenset. That is the overwhelmingly
    common shape — every middleware call site passes no schema at all — and it
    is on :func:`detect_for`'s 99.7%-clean path, where building a set to hold
    nothing was measurably the largest remaining per-call allocation.
    """
    if isinstance(names, (str, bytes)) or not isinstance(names, Iterable):
        return _NO_NAMES
    if not names:
        return _NO_NAMES
    return frozenset(name for name in names if isinstance(name, str))


def _inner_closer_blocks(
    body: str,
    value_start: int,
    item_value: str,
    name: str,
    closer_name: str,
) -> bool:
    """Is a closing tag inside a recovered item's value a SECOND mis-close?

    Boundary row B5's real question, asked properly (task **4502**). B5 refuses
    a recovered item whose "value is itself doubly corrupted, so its boundary
    is a guess" — but it was implemented as a bare substring test for a closing
    tag ANYWHERE in the value, which is strictly wider than that. The shape it
    over-refused is a faithful REPORT of a markup leak: such a report quotes
    the pattern that tripped the tripwire, so the quote lands inside a
    swallowed argument and the guard fired on the caller's own prose. Those
    characters were then dropped on the floor — the exact information loss this
    module exists to end, inflicted by its own refusal.

    An inner closer naming ``N`` blocks recovery iff EITHER:

    (i) ``N`` names the item ITSELF, EITHER dialect's closer for it — the
        name-echoing ``closer_name`` AND the canonical ``parameter``, the
        latter regardless of which dialect this item's OPENER used — or
        ``invoke``: a cross-dialect or repeated mis-close of this very item,
        or a tail spanning a tool-call boundary. An item's own closing tag
        appearing inside its own value is a mis-close BY DEFINITION, never
        prose about itself, so this may be stated categorically; or

        ``parameter`` IS LISTED SEPARATELY FROM ``closer_name``, and dropping
        it reintroduces a live corruption (task **4502**, esc-4502-3). The two
        coincide only in the CANONICAL dialect; for an ECHO-dialect item
        ``closer_name`` is the item's own name, so a bare ``(name,
        closer_name)`` membership test collapses to a single entry and lets
        ``parameter`` through. That is not a cosmetic gap: :func:`_parse_body`
        treats the canonical ``parameter`` closer as a UNIVERSAL terminator, so
        its appearance inside a value is ambiguous with the item's real
        boundary in BOTH dialects — a fact about the parser, not about the
        opener. The dialects also demonstrably BLEND (``_CLOSER_RE``'s
        stray-quote tolerance exists for that measured shape). Measured before
        the fix: a value opening echo-dialect for ``agent_id`` and closing with
        the canonical ``parameter`` closer recovered ``agent_id`` as
        ``'claude-interactive'`` plus that closer plus a whole trailing
        next-tool-call paragraph, reported as ``outcome=repaired`` and, under
        FORWARD_REPAIR, written straight into the tool's arguments — the exact
        swallow-the-next-call failure this condition exists to prevent, reached
        through the mirror image of negative control (a). The ambiguity probe
        (ii) does not catch it, because that trailing prose does not itself
        parse as pseudo-parameters; or
    (ii) reading that closer as this item's terminator ALSO yields a valid
        parse of the remainder — the genuine AMBIGUITY B5's own wording
        describes, where the item's boundary really is a guess.

    Otherwise the occurrence is QUOTED PROSE and recovery proceeds.

    CONDITION (i)'S OPENER MIRROR — the one clause here that is not about a
    closer at all, and the reason this function's name is narrower than its
    rule (task **5620**). A well-formed parameter OPENER anywhere in the value
    — EITHER dialect's, exactly as condition (i) lists either dialect's closer
    — blocks recovery outright, before any closer is examined. Task
    **4502** fixed the CLOSER side of exactly this class — condition (i)'s
    separately-listed ``parameter`` entry, esc-4502-3 — but reasoned only about
    closers, and this function iterates over closers alone, so the opener side
    was left open: a well-formed sibling opener was invisible to the rule and
    was glued into the value verbatim while the sibling it named was silently
    NOT recovered. Measured across 4502 (``b88919ad25^`` vs ``1b9fedeb97``):
    such a value went from ``None`` to ``recovered={'evidence': ...}``.

    An opener naming a parameter is a shape :func:`_parse_body` would have
    opened a SIBLING ITEM on, so reading it as prose instead is a guess about
    where this item ends — precisely what B5 refuses — and guessing wrong
    writes one argument's text into another and reports it as ``repaired``.
    Stated CATEGORICALLY for the same reason (i) is, and on a measurement
    rather than a reading: probe (ii) cannot decide this. Run from each inner
    CLOSER, as it is, the remainder ahead of such an opener begins mid-prose
    and does not parse. Run from each inner OPENER instead — the alternative
    weighed for 5620 — it still stays silent whenever the sibling's own text
    carries a closing tag, because the depth-1 bound reads that remainder as
    "does not parse". With the probe unable to answer and the failure mode
    being a silent partial repair, refusing is the only safe direction.

    BOTH DIALECTS ARE LISTED for the reason condition (i) already gives for the
    canonical closer. :func:`_parse_body` tries :data:`_CANONICAL_OPENER_RE`
    first and falls back to :data:`_ECHO_OPENER_RE`, so BOTH forms are shapes
    it would have opened a sibling item on, and which dialect the ENCLOSING
    item happens to use says nothing about which one a quoted sibling wears.
    That is a property of the parser rather than of the opener — the same
    reasoning that puts ``parameter`` in condition (i) independent of the
    item's own dialect, and the same asymmetry esc-4502-3 was.

    THE ECHO PATTERN IS DELIBERATELY BROAD: it matches any identifier-named
    tag, so a recovered value quoting an ARBITRARY opening tag now refuses, not
    only one naming a real parameter. Qualifying it on schema membership was
    available and is rejected, because this function is not given the schema
    and taking it would couple the boundary rule to the caller's tool. Measured
    safe for the whole known population at ``1b9fedeb97`` — the 504-record
    corpus replay, both ``esc-3514`` specimens and 4502's own positive control
    stay green, the last because its quoted literal is a CLOSER rather than an
    opener. It is also the conservative direction for a guard whose failure
    mode is silently writing another parameter's text into this one: refusing
    returns ``None`` and the caller keeps its value intact, while accepting
    wrongly is unrecoverable.

    IT MUST NOT TOUCH ``considered``, whose final ``return considered == 0`` is
    the malformed-closer fallback (negative control (e)) and has to keep
    meaning "no WELL-FORMED CLOSER was present".

    THE COST IS ONE BOUNDED SEARCH at the same level as the cheap prefilter —
    once per (candidate, tail item), never inside the inner-closer loop — over
    a string that prefilter has just scanned, and only when it fired. The clean
    path pays nothing, and nothing is added to the innermost of the three
    multiplying ceilings :data:`_MAX_CANDIDATES` warns about.

    CONDITION (i) IS NOT REDUNDANT, and dropping it is the single most likely
    way a reimplementation goes wrong. The ambiguity probe alone — or
    qualifying inner closers only on schema membership — also accepts
    committed-corpus record 25 (``mcp__plan-tools__add_design_decision`` /
    ``decision``), whose value opens canonically for ``rationale``, closes with
    the name-echoing ``rationale`` closer, and is followed by an invoke closer
    plus the head of a whole NEXT invoke block. The probe does not catch it
    because that residue does not itself parse as pseudo-parameters, so the
    naive rule would silently swallow the next tool call's fragment into the
    recovered ``rationale`` — a no-silent-partial-repair failure, and strictly
    worse than the ``None`` returned today.

    *value_start* is *item_value*'s offset within *body*, so the probe can read
    the remainder from the shared string rather than copying it: it is passed
    as :func:`_parse_body`'s *start*, NOT used to slice. That is a performance
    contract, not a stylistic one — see *start*'s own docstring for the
    measurement and for why slicing here is super-linear on the request path.

    Bounded like everything else here: at most :data:`_MAX_CANDIDATES` inner
    closers are considered, and the probe runs at depth 1 with the blanket
    substring behaviour restored, so it cannot recurse. Beyond the budget the
    answer is BLOCK, the conservative direction.
    """
    if (
        _CANONICAL_OPENER_RE.search(item_value) is not None
        or _ECHO_OPENER_RE.search(item_value) is not None
    ):
        return True  # (i)'s OPENER MIRROR: a sibling this parser would have opened

    considered = 0
    for inner in _CLOSER_RE.finditer(item_value):
        considered += 1
        if considered > _MAX_CANDIDATES:
            return True
        inner_name = inner.group(1)
        if (
            inner_name in (name, closer_name, _NAME_PARAMETER)
            or closer_for(inner_name) == INVOKE_CLOSER
        ):
            return True  # (i) a mis-close of THIS item, or a call boundary
        if _parse_body(body, probe=True, start=value_start + inner.end()) is not None:
            return True  # (ii) the alternative boundary parses too — a guess
    # The prefilter fired but no WELL-FORMED closer is present, so there is
    # nothing to reason about: keep B5's original answer rather than widening
    # the carve-out onto a shape this rule was never measured against.
    return considered == 0


def _parse_body(body: str, *, probe: bool, start: int = 0) -> dict[str, str] | None:
    """The item loop of :func:`_parse_tail`, after the invoke closer is stripped.

    Factored out (task **4502**) so :func:`_inner_closer_blocks` can ask whether
    a remainder ALSO parses without standing up a second parser that could
    drift from this one. *probe* is that reentrant call: it restores the blanket
    bare-substring refusal, which bounds the recursion at depth 1 by
    construction — deliberately a flag rather than a depth counter, because
    there is exactly one legal depth and a counter would invite a second.

    THAT DEPTH-1 REFUSAL NO LONGER FIRES (task **5620**), and is kept anyway.
    :func:`_inner_closer_blocks` now blocks any value carrying a well-formed
    parameter opener, and an opener at the remainder's start position is the
    only thing the probe could have parsed an item from — so the branch is
    unreachable BY CONSTRUCTION rather than merely untested, and condition (ii)
    decides only whether the remainder is blank. Instrumented across the five
    markup suites: 1 execution at ``1b9fedeb97``, 0 at ``715bf54b9d``. Stated
    here as the measurement it is, with the collapse owned by ticket
    ``tkt_0RTT9N05CHRSHN2MSCHNXX4A8D``, so a reader meets a known dead branch
    rather than an oversight. Note what the collapse may NOT take with it:
    *start* below carries its own separately-measured performance contract and
    has nothing to do with this rule.

    *start* is where in *body* to begin, and is what keeps the probe CHEAP. It
    exists instead of the obvious ``_parse_body(body[offset:], ...)`` because
    that slice is O(len(body)) and runs once per inner closer per tail item per
    candidate, while the probe itself almost always answers in O(1) — the
    remainder starts mid-prose, neither opener matches at *pos*, and it returns
    immediately. Measured on the sliced version at a CONSTANT 1024 probes,
    growing only the body: 216 KB -> 0.0042 s, 3.2 MB -> 0.0496 s, i.e. linear
    in a length the parse work does not depend on, and :func:`repair` pays that
    up to :data:`_MAX_CANDIDATES` times over. Since :func:`repair` runs
    synchronously on the middleware's request path, a large leaked argument
    stalled the server for the duration. DO NOT "simplify" *start* back into a
    slice: no regex here is anchored (see :data:`_CANONICAL_OPENER_RE`,
    :data:`_ECHO_OPENER_RE`, :func:`_closer_re` — none uses ``^`` or a
    lookbehind), so passing a position is exactly equivalent and merely free.

    ``None`` means the body did not parse with ZERO leftover.
    """
    recovered: dict[str, str] = {}
    pos = start
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

        # THE CHEAP PREFILTER, retained verbatim so the clean path pays exactly
        # what it paid before: one substring scan, and nothing else.
        if '\x3c/' in item_value:
            if probe:
                # Depth 1. The probe only has to answer "does this remainder
                # parse at all"; re-entering the narrowing here would recurse.
                #
                # MEASURED UNREACHABLE as of task 5620, and deliberately kept.
                # Instrumented across the five markup suites: 1 execution at
                # 1b9fedeb97, 0 at 715bf54b9d. The probe can only parse an item
                # when an opener sits at its start position, and that value is
                # now blocked by the opener mirror before condition (ii) is
                # consulted, so (ii) decides only whether the remainder is
                # blank. Collapsing the apparatus belongs to ticket
                # tkt_0RTT9N05CHRSHN2MSCHNXX4A8D, not here: it means deleting a
                # recursion bound task 4502 landed with an explicit
                # flag-not-counter argument. *start* must survive that collapse
                # regardless — its contract is independent of this rule.
                return None
            if _inner_closer_blocks(body, match.end(), item_value, name, closer_name):
                return None  # a SECOND mis-close: the boundary is a guess (B5)
        if name in recovered:
            return None  # the same parameter twice is not a well-formed tail
        recovered[name] = item_value

    return None  # more items than any real call has — refuse rather than guess


def _parse_tail(tail: str) -> dict[str, str] | None:
    """Parse *tail* as a sequence of pseudo-parameters, else ``None``.

    The grammar is PRD section 4 C1's, verbatim: a name-echoing pair, a
    canonical ``parameter`` pair, or a final UNTERMINATED opener whose value
    runs to end-of-string (the parser consumed that closer as its terminator),
    with one trailing invoke closer stripped and whitespace allowed between
    items. ``None`` means the tail did not parse with ZERO leftover, which
    rejects the candidate and advances the scan — this function never yields a
    partial parse of one tail.

    That is a per-tail guarantee only, and on its own it is NOT enough to rule
    out a partial REPAIR: advancing the scan leaves the rejected closer inside
    the next candidate's prefix. The prefix-clean accept-time condition in
    :func:`repair` is what closes that gap.

    BOUNDARY ROW B5 lives in :func:`_inner_closer_blocks`, which this delegates
    to via :func:`_parse_body`. As of task **4502** it is an ALTERNATIVE-BOUNDARY
    test rather than a bare substring refusal: a closing tag inside a recovered
    item's value blocks recovery when it mis-closes THAT item or spans a
    tool-call boundary, or when reading it as the terminator also parses — but
    NOT when it is merely quoted prose. A recovered value is verbatim caller
    text under invariant D5, and a faithful report of a markup leak necessarily
    quotes the leak; ``clean_value``'s envelope-free post-condition is
    untouched, because that is the value the repairer REWROTE.

    Every returned value is a SLICE of *tail*; nothing is rebuilt or decoded.
    """
    body = tail.rstrip()
    if body.endswith(INVOKE_CLOSER):
        body = body[: -len(INVOKE_CLOSER)].rstrip()

    return _parse_body(body, probe=False)


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
    parses with zero leftover, every recovered name is in *schema_params*, no
    recovered name is in *supplied*, and the resulting ``clean_value`` is
    itself envelope-free. Otherwise ``None`` — never a guess, and never a
    PARTIAL repair.

    ONE FURTHER ACCEPT-TIME CONDITION, THE QUOTATION GUARD (task **4696**
    review). A candidate whose tail is EMPTY, whose ``X != param``, and whose
    closer is not one of the fixed :data:`ENVELOPE_LITERALS` is REFUSED: it
    recovers nothing, so accepting it would merely truncate the caller's text
    at a sibling's closing tag — prose QUOTING that tag, not an argument
    absorbed by it. See the guard at the accept site for the census that says
    the cross-field leak population is zero. Never a TRUNCATION dressed up as
    a repair.

    NO SILENT PARTIAL REPAIR. The returned ``clean_value`` is guaranteed to
    satisfy ``detect_for(clean_value, param, schema_params) is None`` — and
    therefore ``detect(clean_value) is None`` too, since the parameter-aware
    predicate is a strict superset. This is a contract, not an accident of the
    scan: contract C2's middleware forwards ``clean_value`` as the repaired
    argument, so a value that still tripped the detector would re-trip the
    write-time tripwire downstream AND would have silently dropped whatever
    caller arguments were hiding in the residue — the exact failure this module
    exists to end, reintroduced by its own repairer. When the only candidates
    that parse would leave a poisoned prefix, the honest answer is ``None``.

    THE POST-CONDITION IS ON ``clean_value`` ONLY, and deliberately so (task
    **4502**). A RECOVERED value may still trip :func:`detect`: it is verbatim
    caller text under invariant D5, and a faithful report of a markup leak
    quotes the leak. Boundary row B5 is an alternative-boundary test rather
    than a bare substring refusal precisely so those characters are recovered
    instead of dropped — see :func:`_inner_closer_blocks` for the rule and for
    why its own-name condition is not redundant. The C2 middleware surfaces
    which recovered names carry a literal rather than letting it pass silently.

    The guard is stated against :func:`detect_for` rather than :func:`detect`
    as of task **4696**, because the gates that consume ``clean_value`` are
    parameter-aware: a prefix carrying a canonical closer for *param* or for a
    *schema_params* member would re-trip ``plan_tools._carries_markup`` and the
    sweep on the very next read. Measured over the committed corpus at that
    task's HEAD — 504 records, 443 accepted, ZERO carrying a qualifying closer
    in the accepted prefix — so the tightening changed no per-specimen
    expectation; it closed the DOUBLE SELF-NAME MISCLOSE hole and nothing else.
    The committed corpus now reads 504 records / **444 accepted** after task
    **4502** moved one record repaired-ward; the 443 above is 4696's datum and
    stays as it was measured, so a reader who checks it against today's fixture
    and finds 444 knows which task moved it rather than suspecting rot. The
    ZERO clause is unchanged and was re-verified at 4502: the newly-accepted
    record's ``clean_value`` carries no qualifying closer either.

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
        # THE QUOTATION GUARD (task 4696 review). An EMPTY tail means the
        # candidate closer sits at end-of-string with NOTHING after it, so there
        # is no absorbed argument to recover and the "repair" degenerates to a
        # pure TRUNCATION of the caller's own text at that tag.
        #
        # That truncation is right for a SELF-NAME closer (``name == param``):
        # the value was mis-closed with its own tag and was the tool's last
        # parameter, which is PRD boundary row B4 and the entire 212-of-212
        # population the 2026-08-25 census measured. It is also right for the
        # FIXED literal set, whose members trip ``detect`` and have therefore
        # always been repaired here — this task changed nothing for them.
        #
        # It is WRONG for a name contributed by the WIDENING, i.e. a sibling
        # parameter's closer that ``detect_for`` only started spelling at this
        # task. Prose that legitimately ENDS by quoting a sibling's tag pair —
        # ubiquitous in a repo whose plans and escalation records discuss this
        # very markup — is not an absorbed argument, and the same census puts
        # the genuine CROSS-FIELD leak population at ZERO. Truncating it would
        # destroy authored text and, worse, report ``repaired`` while doing so,
        # so nothing would ever surface it for adjudication.
        #
        # Refusing (rather than accepting) leaves the value BYTE-IDENTICAL and
        # lets the caller report it — ``plan_tools`` as ``unrepairable``, the
        # sweep as ``refused`` — into the human queue. Detection breadth is
        # deliberately NOT narrowed: a real cross-field leak still carries a
        # non-empty tail, still parses, and is still repaired here.
        if (
            not recovered
            and name != param
            and closer_for(name) not in ENVELOPE_LITERALS
        ):
            continue
        # Accept-time conditions, so a rejected candidate simply advances the
        # scan. Boundary rows B8 (recovered name not in the tool's schema) and
        # B9 (recovered name already supplied by the caller) both land here.
        if not recovered.keys() <= schema:
            continue
        if not recovered.keys().isdisjoint(already_supplied):
            continue
        clean_value = value[: candidate.start()]
        # PREFIX-CLEAN, the third accept-time condition. The scan steps over a
        # candidate whose tail does not parse, which leaves that closer sitting
        # inside every LATER candidate's prefix — and if it is itself an
        # envelope literal, accepting would hand back a clean_value that still
        # trips the detector while permanently dropping whatever arguments hide
        # in the residue. `continue` rather than `return None` keeps all three
        # refusals one shape; the outcome is identical either way because
        # candidate start positions increase monotonically, so a poisoned
        # prefix stays poisoned — but `continue` needs no such argument to be
        # correct.
        #
        # TASK 4696: this guard was ITSELF blind to the dominant dialect until
        # now. It asked the param-free detect(), which spells none of the
        # invoked tool's own parameter names — so a DOUBLE self-name misclose
        # was accepted with the first closer still in the prefix AND the prose
        # stranded between the two closers silently dropped. It now asks
        # detect_for with the same (param, schema_params) the candidate
        # qualification above uses, which is the only way the guard can be as
        # wide as the rule it is guarding: every candidate this scan can step
        # over is, by construction, a closer detect_for spells.
        if detect_for(clean_value, param, schema) is not None:
            continue

        misclose = candidate.group(0)
        # Both union scans run on the ACCEPT PATH ONLY — the prefix scan just
        # above fires only after a tail has already parsed and passed B8/B9,
        # and this one only after the prefix came back clean. repair() refuses
        # far more often than it reaches either: every value with no qualifying
        # candidate at all, plus every ordinary argument that merely happens to
        # contain a closing tag. Hoisting a detect() above the loop would
        # instead pay a full-string scan on that overwhelmingly common no-op
        # path, at a per-tool-call middleware boundary (PRD contract C2) over
        # values the corpus shows reaching tens of KB. That argument is
        # unchanged by task 4696: the accept path now pays the WIDENED scan
        # above, and the refuse path still pays neither.
        #
        # THE SAME PARAMETER-AWARE PREDICATE the guard above and every gate in
        # front of this function ask (task 5283). This site used to ask the
        # blanket `detect`, on the argument that the `misclose` fallback below
        # "ALREADY self-heals for a name outside the literal set and detect_for
        # here would change no observable value".
        #
        # MEASURED FALSE. Where a fixed literal is present the fallback never
        # fires, so the two predicates are free to disagree — and they do.
        # Specimen: prose, the absorbing parameter's own closer, then a
        # canonical opener naming the swallowed sibling, with param='how' and
        # schema=('what','where','how'):
        #
        #     detect(value)            '\x3cparameter name='   offset 39
        #     detect_for(value, ...)   the `how` closer        offset 33
        #
        # Repair.pattern is published as `matched_pattern` by
        # mcp_markup_middleware's _reject and _forward and as the repaired
        # fact's `pattern` by plan_tools, so the disagreement put ONE event on
        # the wire with TWO answers: the fact stream naming the head of the
        # leak and the caller's payload naming a literal that merely trails it
        # — the exact diagnostic defect PRD section 2.2 exists to close.
        #
        # detect_for is a strict superset of detect's needle set and reports
        # earliest-by-text-position, so this can only ever name an earlier-or-
        # equal literal. The `misclose` fallback is unchanged and still covers
        # the case where nothing at all is found.
        detected = detect_for(value, param, schema)
        return Repair(
            clean_value=clean_value,
            recovered=recovered,
            pattern=detected if detected is not None else misclose,
            misclose=misclose,
        )

    return None


# ---------------------------------------------------------------------------
# The deliberate-quoting override.
# ---------------------------------------------------------------------------
#
# Promoted verbatim from ``fused_memory.server.markup_tripwire`` by task 3689,
# alongside the enumeration task 3688 promoted before it. Both guards that can
# bounce a caller for envelope markup — markup_tripwire at write time and
# ``shared.mcp_markup_middleware`` at the FastMCP boundary (contract C2,
# boundary row B6) — must honour ONE override with ONE lifecycle; a second
# implementation of "is this deliberate?" is exactly the lockstep-duplication
# defect (INV-5) this PRD exists to end.
#
# The override is live, not hypothetical: the decompose session that filed
# these very tasks had to set it to quote the literals in its own task text.


#: Write-time-only control flag that bypasses the markup guards for markup a
#: caller is quoting DELIBERATELY (DF 3083's own task description quotes all
#: three literals in prose, so without this the very sibling that leaf exists to
#: feed could not be updated). Only a literal boolean ``True`` enables it, and
#: it is stripped from metadata before dispatch/persistence at every boundary —
#: mirroring the established ``allow_near_duplicate`` lifecycle in
#: ``fused_memory/server/tools.py``. An accidental harness serialization leak
#: never sets an explicit flag; an author can.
MARKUP_OVERRIDE_KEY = 'allow_mcp_markup'


def _as_metadata_dict(metadata: object) -> dict[str, Any] | None:
    """Best-effort read of *metadata* as a dict, else ``None``.

    ``submit_task``/``update_task`` accept metadata as an object OR a JSON
    string, so both shapes are understood. Anything unparseable — malformed
    JSON, a non-dict JSON payload, a wrong type entirely — yields ``None``
    without raising: validating metadata is not this module's job, and a write
    must never fail because the override helper choked on a field it does not
    own.
    """
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except (ValueError, TypeError):
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def markup_override_requested(metadata: object) -> bool:
    """Return True iff *metadata* carries an explicit :data:`MARKUP_OVERRIDE_KEY` opt-in.

    Fail-closed: ONLY a literal boolean ``True`` counts, mirroring add_memory's
    ``metadata.get('allow_near_duplicate') is True`` check (``tools.py``:1199).
    A truthy-but-not-``True`` value (``'yes'``, ``1``) is far more likely to be
    unrelated data than a considered decision to write raw MCP envelope markup
    into the corpus — and the failure mode being contained, an accidental
    serialization leak, never sets an explicit flag at all.

    Never raises, for any input.
    """
    parsed = _as_metadata_dict(metadata)
    if parsed is None:
        return False
    return parsed.get(MARKUP_OVERRIDE_KEY) is True


def strip_markup_override(metadata: Any) -> Any:
    """Return *metadata* without :data:`MARKUP_OVERRIDE_KEY`, in the same shape.

    The override is a write-time-only control flag: it must never be persisted
    into stored memory metadata or the task metadata vocabulary. Returning the
    shape it was given (dict in / dict out, JSON string in / JSON string out)
    lets a call site substitute the result inline before forwarding downstream.

    NON-mutating — the caller's own dict is left intact, since the handler may
    still need the original and quietly mutating caller-owned metadata is
    action-at-a-distance this guard should not introduce. (This is the one
    deliberate divergence from ``allow_near_duplicate``'s in-place
    ``cleaned_meta.pop`` at ``tools.py``:1266, which operates on a dict it has
    already copied.)

    Unparseable input passes straight through unchanged, never raising.
    """
    if isinstance(metadata, dict):
        if MARKUP_OVERRIDE_KEY not in metadata:
            return metadata
        return {k: v for k, v in metadata.items() if k != MARKUP_OVERRIDE_KEY}
    if isinstance(metadata, str):
        parsed = _as_metadata_dict(metadata)
        if parsed is None or MARKUP_OVERRIDE_KEY not in parsed:
            return metadata
        return json.dumps({k: v for k, v in parsed.items() if k != MARKUP_OVERRIDE_KEY})
    return metadata
