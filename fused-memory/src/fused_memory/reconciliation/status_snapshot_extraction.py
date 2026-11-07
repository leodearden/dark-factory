"""Status-snapshot extraction: the pure lexical layer that reads which task ids
a Graphiti edge fact asserts as non-terminal, consumed by
``stale_status_snapshot_edge_sweep.py``.

Every path here is biased toward missing a snapshot over wrongly retiring
one: under-selection self-heals next cycle, while over-selection is
unrecoverable, since an invalidated Graphiti edge is never re-validated.
Extraction runs once per valid edge with no per-edge timeout, so its cost is
a whole-cycle liveness property, bounded by this module's performance tests.

Design rationale, residuals and measurements:
``docs/stale-status-snapshot-sweep.md``.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Sequence
from typing import NamedTuple

from fused_memory.reconciliation.task_filter import STRICT_CLAUSE_BOUNDARY_RE, TASK_REF_RE

# --------------------------------------------------------------------------- #
# Marker vocabulary, gates and the clause-scoped guards
# --------------------------------------------------------------------------- #

# Status markers split by part of speech. 'blocked' and 'stalled' are also
# transitive verbs, so they match only in copula/article form. See
# docs/stale-status-snapshot-sweep.md §"Marker classes and the compound prefix".
_ADJECTIVE_MARKER_ALT = r'(?:active|pending|in[-\s]?progress)'
_TRANSITIVE_MARKER_ALT = r'(?:blocked|stalled)'

# The closed-class connective: optional for INDIVIDUAL_SNAPSHOT_RE's adjective
# arm, mandatory for its transitive-capable arm.
_COPULA_ALT = r'(?:is|are|was|were|remains?)'

# Closed-class adverbs allowed between copula and marker. Never widen this to
# an open-class gap.
_ADVERB_ALT = r'(?:(?:currently|still|now|already)\s+)*'

# Per-gap-word lookahead for SNAPSHOT_STATUS_PHRASE_RE: refuses negation and
# past-exit qualifiers, which INVERT the assertion.
_GAP_EXCLUDED_ALT = (
    r'(?!(?:not|no|longer|never|previously|formerly|once|briefly|'
    r'until|after|before)\b)'
)

# Per-gap-word lookahead for SNAPSHOT_STATUS_PHRASE_RE: refuses the opening
# token of another task reference, which RE-SUBJECTS the assertion. The
# '\w*\b' tail makes it deliberately wider than TASK_REF_RE ('tasks' too).
_GAP_NO_TASK_REF = r'(?!(?:task|df|\d)\w*\b)'

# One optional hyphenated modifier before the marker ('pairwise-stalled'),
# minus the inverting prefixes ('un-', 'previously-', and a privative 'in-'
# that is not the head of 'in-progress'). The lookaheads sit before the
# optional group, so they also run when no prefix is present. See
# docs/stale-status-snapshot-sweep.md §"The compound prefix".
_COMPOUND_PREFIX = (
    r'(?!(?:un|non|not|never|no|previously|formerly|once|briefly|de|dis)-)'
    r'(?!in-(?!progress\b))'
    r'(?:\w+-)?'
)

# Derived from both classes so the gate and the anchored matchers cannot
# drift. Used only by the gate and the status-noun-anchored phrase form.
_STATUS_MARKER_ALT = r'(?:' + _ADJECTIVE_MARKER_ALT + r'|' + _TRANSITIVE_MARKER_ALT + r')'

# Whole-fact gate: a fact naming no non-terminal marker anywhere is not a
# status snapshot. It only admits a fact; it never anchors a marker.
SNAPSHOT_STATUS_RE: re.Pattern[str] = re.compile(
    r'\b' + _STATUS_MARKER_ALT + r'\b',
    re.IGNORECASE,
)


# The blocked family's marker: the literal 'blocked' alone, since 'stalled'
# has no TaskStatus value a census could contradict.
_BLOCKED_MARKER_ALT = r'(?:blocked)'

# Cheap literal pre-gate for the blocked family, which keeps its five patterns
# off the hot path for edges with no 'blocked' token.
_BLOCKED_MARKER_RE: re.Pattern[str] = re.compile(r'\bblocked\b', re.IGNORECASE)

# Strips 'N <count-noun>' spans from an aggregate list segment before
# bare-digit extraction. Keep it in step with _STATUS_MARKER_ALT: it is the
# one marker consumer that is NOT safe by default under a marker widening.
# Applied only to a list-scoped segment, never to the whole fact. See
# docs/stale-status-snapshot-sweep.md §"Counting phrases".
COUNT_QUANTITY_RE: re.Pattern[str] = re.compile(
    r'\b\d+\s+(?:tasks?|active|pending|in[-\s]?progress|done|cancell?ed|'
    r'blocked|stalled|deferred|review|total|merge[-\s]?deferred)\b',
    re.IGNORECASE,
)

# The enumeration tail: two or more ids. The separator's leading comma or
# whitespace is mandatory, which is what enforces 'two or more'.
#
# Every quantifier is POSSESSIVE, to stop exponential backtracking on a
# failing match. That is safe only while every possessive element is followed
# by something that can never match whitespace: each '\s*+'/'\s++' by ',', a
# letter, '#' or '\d++', and '\d++' by whitespace or ','. Needs Python >= 3.11.
# See docs/stale-status-snapshot-sweep.md §"Possessive quantifiers (ReDoS)".
_ENUM_SEP_ALT = r'(?:\s*+,\s*+|\s++)(?:(?:and|&|/)\s++)?(?:task\s*+)?#?\s*+'
_ENUM_IDS_ALT = r'\d++(?:' + _ENUM_SEP_ALT + r'\d++)+'

# Plural agreement, for the plural-enumeration path only; the shared
# _COPULA_ALT is left untouched. See
# docs/stale-status-snapshot-sweep.md §"Plural-enumeration subjecthood guard".
_PLURAL_COPULA_ALT = r'(?:are|were|remain)'

# Prepositions that make an enumeration a preposition's complement rather than
# the copula's subject, searched over the whole clause before it. A missing
# entry OVER-selects, so add one with a plural-head case: the regression test
# parametrizes over this tuple. The recall measurement and the not-tightened
# verdict are in docs/stale-status-snapshot-sweep.md
# §"Plural-enumeration subjecthood guard".
_ENUM_PREP_WORDS: tuple[str, ...] = (
    'of', 'in', 'on', 'to', 're', 'for', 'with', 'among', 'about', 'across',
    'against', 'between', 'regarding', 'from', 'by', 'under', 'within',
    'around', 'during', 'through', 'via', 'concerning',
)

# Characters that can end a sentence, and so end the span a preposition
# governs. An occurrence counts only when it is not token-internal. Keep the
# class minimal: an extra member truncates the scan and over-selects. See
# docs/stale-status-snapshot-sweep.md §"Clause breaks".
CLAUSE_BREAK_CHARS = '.;!?'


# The head of an HTML character entity reference ('&lt', '&#60', '&#x3c'),
# anchored to end exactly where the ';' that terminates it begins. Searched
# over a bounded window ending at that ';', so the check costs O(1) per
# occurrence and needs no slice.
_HTML_ENTITY_HEAD_RE: re.Pattern[str] = re.compile(
    r'&(?:[A-Za-z][A-Za-z0-9]{1,31}|#[0-9]{1,7}|#[xX][0-9A-Fa-f]{1,6})\Z'
)
_HTML_ENTITY_HEAD_MAX_LEN = 40


def is_token_internal_break(text: str, index: int) -> bool:
    """Does the break character at ``text[index]`` end no sentence?

    True when it is flanked by alphanumerics on both sides (a filename
    extension, version string, dotted path or section number, URL query
    '?'), or when it is a ';' terminating an HTML character entity
    ('&lt;', '&#60;'). Unicode-aware, through ``str.isalnum()``.

    A genuine break written with no following space ('done.Then') reads as
    token-internal, which can only under-select. See
    ``docs/stale-status-snapshot-sweep.md`` §"Clause breaks".
    """
    if (
        index > 0
        and text[index - 1].isalnum()
        and index + 1 < len(text)
        and text[index + 1].isalnum()
    ):
        return True
    return text[index] == ';' and _HTML_ENTITY_HEAD_RE.search(
        text, max(0, index - _HTML_ENTITY_HEAD_MAX_LEN), index
    ) is not None


def _last_clause_break(prefix: str) -> int:
    """Index of the last sentence-plausible clause break in *prefix*, or -1.

    Every member of ``CLAUSE_BREAK_CHARS`` counts only where the occurrence
    is not token-internal (see ``is_token_internal_break``). The walk keeps
    one cursor per break character and repeatedly takes the rightmost; when
    that occurrence is token-internal, only its own cursor moves left, to the
    previous occurrence of the same character. Each cursor only moves left
    and each ``rfind`` resumes where that cursor's previous one stopped, so
    the walk is O(len(prefix)) overall, with no slicing.
    """
    cursors = {c: prefix.rfind(c) for c in CLAUSE_BREAK_CHARS}
    while True:
        char, cursor = max(cursors.items(), key=lambda item: item[1])
        if cursor < 0 or not is_token_internal_break(prefix, cursor):
            return cursor
        cursors[char] = prefix.rfind(char, 0, cursor)


_ENUM_PREP_WORD_RE: re.Pattern[str] = re.compile(
    r'\b(?:' + '|'.join(_ENUM_PREP_WORDS) + r')\b',
    re.IGNORECASE,
)


def _enumeration_is_prepositional_complement(prefix: str) -> bool:
    """Does a listed preposition govern the enumeration at ``prefix``'s end?

    ``prefix`` is the fact text preceding a ``PLURAL_ENUM_SNAPSHOT_RE`` match.
    True means the enumeration is a preposition's complement, not the copula's
    subject, and the match must be discarded. See
    ``docs/stale-status-snapshot-sweep.md``
    §"Plural-enumeration subjecthood guard".
    """
    # search(..., pos) instead of slicing the clause: '\b' at pos still sees
    # the preceding character, and every break char is non-word, so this is
    # exactly equivalent to searching the sliced clause.
    return _ENUM_PREP_WORD_RE.search(prefix, _last_clause_break(prefix) + 1) is not None


def _list_is_governed_by_task_ref(prefix: str) -> bool:
    """Does an explicit task reference precede this list introducer's clause?

    ``prefix`` is the fact text preceding a ``list_introducer`` match. True
    means the '<marker> tasks:' span reads as a TRANSITIVE VERB taking that
    reference as its subject ('Task 5 blocked tasks: 142, 148') rather than as
    a status list ('Blocked tasks: 142, 148'), and the segment must be
    discarded.

    Run by the blocked family only, and clause-scoped like the plural guard.
    It only ever removes ids, so it cannot break the blocked-subset-of-union
    property. See ``docs/stale-status-snapshot-sweep.md``
    §"The blocked family's list guard".
    """
    return TASK_REF_RE.search(prefix, _last_clause_break(prefix) + 1) is not None


# --------------------------------------------------------------------------- #
# Anchored pattern families — one builder, two instantiations
# --------------------------------------------------------------------------- #


class _SnapshotPatterns(NamedTuple):
    """The five anchored extraction patterns for one marker family.

    ``individual`` omits its adjective arm entirely when the family carries no
    adjective markers (see ``_build_snapshot_patterns``).

    ``reject_list_after_task_ref`` is not a pattern but a per-family SETTING of
    the shared algorithm, carried here so ``_extract_ids`` reads it off the
    family it was handed rather than branching on which family it is running.
    """

    individual: re.Pattern[str]
    genitive: re.Pattern[str]
    status_phrase: re.Pattern[str]
    plural_enum: re.Pattern[str]
    list_introducer: re.Pattern[str]
    reject_list_after_task_ref: bool


def _build_snapshot_patterns(
    adjective_alt: str | None,
    transitive_alt: str | None,
    *,
    reject_list_after_task_ref: bool = False,
) -> _SnapshotPatterns:
    """Compile the five anchored extraction patterns over one marker family.

    Every family is built here, so every guard applies to all of them by
    construction. A new path or guard belongs in this builder, never in one
    family. See ``docs/stale-status-snapshot-sweep.md``
    §"Pattern families and the subset property".

    Args:
        adjective_alt: Alternation of ADJECTIVE-only markers, which take an
            OPTIONAL copula ('Task 5 pending' matches). None omits the
            individual form's adjective arm.
        transitive_alt: Alternation of markers that are ALSO common transitive
            verbs, which take a MANDATORY copula or article. None omits that
            arm.
        reject_list_after_task_ref: Drop an aggregate list segment whose
            introducer follows a task reference in the same clause. On for
            the blocked family only; see ``_list_is_governed_by_task_ref``.

    At least one alternation must be supplied. The other four patterns run
    against the union of whichever were supplied, since each pins its marker
    with an independent anchor (a mandatory copula, the literal noun
    'status', or adjacency to the list noun).

    Pure: no I/O, no side effects.
    """
    if adjective_alt is None and transitive_alt is None:
        raise ValueError('_build_snapshot_patterns needs at least one marker alternation')

    supplied = [alt for alt in (adjective_alt, transitive_alt) if alt is not None]
    marker_alt = supplied[0] if len(supplied) == 1 else r'(?:' + r'|'.join(supplied) + r')'
    # Individual form: the marker anchored to its own task reference, with
    # only closed-class words (copula, adverb, article; never 'in') between.
    # See docs/stale-status-snapshot-sweep.md §"Individual form".
    #
    # The arms are assembled conditionally: an empty alternative would match
    # the empty string and turn every bare task reference into a snapshot
    # assertion.
    arms: list[str] = []
    if adjective_alt is not None:
        # adjective arm — optional copula
        arms.append(
            r'\s*' + _COPULA_ALT + r'?\s*' + _ADVERB_ALT + r'(?:an?\s+)?'
            + _COMPOUND_PREFIX + adjective_alt
        )
    if transitive_alt is not None:
        # transitive-capable arm — copula or article is MANDATORY
        arms.append(
            r'\s+(?:' + _COPULA_ALT + r'\s+' + _ADVERB_ALT + r'(?:an?\s+)?|an?\s+)'
            + _COMPOUND_PREFIX + transitive_alt
        )
    individual = re.compile(
        TASK_REF_RE.pattern + r'(?:' + r'|'.join(arms) + r')\b',
        re.IGNORECASE,
    )

    # Genitive form: "Task N's status is <marker>", with either apostrophe.
    # 'status' must be immediately followed by the copula, which keeps
    # "status report is pending" out. See
    # docs/stale-status-snapshot-sweep.md §"Genitive form".
    genitive = re.compile(
        TASK_REF_RE.pattern
        + r"(?:'|’)s\s+status\s+"
        + _COPULA_ALT + r'\s+' + _ADVERB_ALT + r'(?:an?\s+)?'
        + _COMPOUND_PREFIX + marker_alt + r'\b',
        re.IGNORECASE,
    )

    # Status-phrase form: a lazy open-class gap of at most 3 words, admitted
    # only when the marker is immediately followed by the noun 'status'. The
    # gap cannot cross '.', ',' or ';'. Its four precision residuals are in
    # docs/stale-status-snapshot-sweep.md §"Status-phrase form".
    status_phrase = re.compile(
        TASK_REF_RE.pattern
        # per gap word: neither an inverting qualifier nor another task reference
        + r'(?:\s+' + _GAP_EXCLUDED_ALT + _GAP_NO_TASK_REF + r'\w+){0,3}?'
        + r'\s+in\s+(?:an?\s+|the\s+)?'
        + marker_alt + r'\s+status\b'
        # the span must END the noun phrase — else 'status' is a modifier of a
        # following head noun ('status report'), not the phrase head
        + r'(?=\s*[.,;:!?)\]]|\s*$|\s+(?:due|under|pending|since|because|'
        + r'awaiting|after|before|while|until|per|as|with|for|on|from|and|or)\b)',
        re.IGNORECASE,
    )

    # Plural-enumeration form: 'Tasks A, B and C are <marker>'. The plural
    # copula is mandatory, the marker is adjacent to it, and digits come from
    # the 'ids' group only. No COUNT_QUANTITY_RE strip here: it would eat the
    # '1020 task 1030' separator. _enumeration_is_prepositional_complement
    # completes the subjecthood test. See
    # docs/stale-status-snapshot-sweep.md §"Plural-enumeration form".
    plural_enum = re.compile(
        r'\btasks\b\s*#?\s*(?P<ids>' + _ENUM_IDS_ALT + r')\s*,?\s*'
        + _PLURAL_COPULA_ALT + r'\s+' + _ADVERB_ALT + r'(?:all\s+)?(?:an?\s+)?'
        + _COMPOUND_PREFIX + marker_alt + r'\b',
        re.IGNORECASE,
    )

    # Aggregate-list introducer: the marker immediately before the list noun.
    # The 'open' group records '[' or ':' so _list_segment can find the
    # segment's end. See docs/stale-status-snapshot-sweep.md
    # §"Aggregate-list form".
    list_introducer = re.compile(
        r'\b' + marker_alt + r'\s+tasks?\b\s*(?:are|is|were)?\s*(?P<open>[:\[])',
        re.IGNORECASE,
    )

    return _SnapshotPatterns(
        individual=individual,
        genitive=genitive,
        status_phrase=status_phrase,
        plural_enum=plural_enum,
        list_introducer=list_introducer,
        reject_list_after_task_ref=reject_list_after_task_ref,
    )


# The UNION family, every marker class. The five exported constants below are
# bound from it.
_UNION_PATTERNS = _build_snapshot_patterns(_ADJECTIVE_MARKER_ALT, _TRANSITIVE_MARKER_ALT)

# The BLOCKED family. 'blocked' is supplied as the transitive alternation, so
# the mandatory copula keeps 'Task 5 blocked the merge queue' out.
_BLOCKED_PATTERNS = _build_snapshot_patterns(
    None, _BLOCKED_MARKER_ALT, reject_list_after_task_ref=True,
)

INDIVIDUAL_SNAPSHOT_RE: re.Pattern[str] = _UNION_PATTERNS.individual
GENITIVE_STATUS_RE: re.Pattern[str] = _UNION_PATTERNS.genitive
SNAPSHOT_STATUS_PHRASE_RE: re.Pattern[str] = _UNION_PATTERNS.status_phrase
PLURAL_ENUM_SNAPSHOT_RE: re.Pattern[str] = _UNION_PATTERNS.plural_enum
LIST_INTRODUCER_RE: re.Pattern[str] = _UNION_PATTERNS.list_introducer


# Bare digit token, used only inside an already-detected, marker-anchored
# span: an aggregate list segment or a plural enumeration's 'ids' group.
_BARE_DIGIT_RE: re.Pattern[str] = re.compile(r'\b\d+\b')


# Closed by the STRICT boundary, never task_filter.WIDE_CLAUSE_BOUNDARY_RE: a
# longer segment harvests more incidental digits as task ids. See
# docs/stale-status-snapshot-sweep.md §"The segment boundary".
def _list_segment(text: str, start: int, open_char: str) -> str:
    """Return the list-segment substring of *text* beginning at *start*.

    A bracket-opened segment (``open_char == '['``) runs to the matching
    ``']'`` when present, else falls back to the next clause boundary / end
    of string (mirrors the colon case) rather than running away to the end
    of a long text. A colon-opened segment has no closing delimiter of its
    own, so it always runs to the next clause boundary ([.;\\n!?]) or end of
    string.
    """
    if open_char == '[':
        close = text.find(']', start)
        if close != -1:
            return text[start:close]

    boundary = STRICT_CLAUSE_BOUNDARY_RE.search(text, start)
    end = boundary.start() if boundary else len(text)
    return text[start:end]


def extract_snapshot_edge_task_ids(fact: str | None) -> set[int]:
    """Return the task ids *fact* asserts as active/pending/blocked/stalled/in-progress.

    Returns the empty set (never a candidate for invalidation) when:
    - *fact* contains no such status marker at all ('Task 5 is done'); or
    - *fact* is a pure count-only snapshot with no specific task-id reference
      ('There are 8 tasks in progress', '1505 done / 148 cancelled'); or
    - a task id is only incidentally near an unrelated status word ('Task 142
      landed on the active branch' — the active BRANCH).

    Algorithm (steps 2-4 are shared with
    ``extract_blocked_assertion_task_ids`` via ``_extract_ids``; only the
    step-1 gate and the pattern family differ between them):
      1. Gate on SNAPSHOT_STATUS_RE; short-circuit to the empty set when
         absent.
      2. Plural enumerations (PLURAL_ENUM_SNAPSHOT_RE), rejecting any
         enumeration that is a preposition's complement.
      3. Individual, genitive and status-phrase forms
         (INDIVIDUAL_SNAPSHOT_RE, GENITIVE_STATUS_RE,
         SNAPSHOT_STATUS_PHRASE_RE), skipping any match inside a rejected
         enumeration.
      4. Aggregate lists (LIST_INTRODUCER_RE): each segment has
         COUNT_QUANTITY_RE spans stripped, then its bare digits collected.

    See ``docs/stale-status-snapshot-sweep.md`` §"The five extraction paths".

    Pure: no I/O, no side effects.
    """
    return extract_snapshot_edge_task_ids_by_marker_class(fact).all_ids


def extract_blocked_assertion_task_ids(fact: str | None) -> set[int]:
    """Return the subset of task ids *fact* asserts as BLOCKED specifically.

    The blocked-scoped counterpart of ``extract_snapshot_edge_task_ids``: the
    same algorithm over the BLOCKED pattern family, gated on the cheap
    literal ``_BLOCKED_MARKER_RE``. Scoped to the literal 'blocked' marker,
    never 'stalled'.

    Always a SUBSET of the union result, by the seam in
    ``_ids_within_union``. One blocked-only guard narrows it further; see
    ``_list_is_governed_by_task_ref``. See
    ``docs/stale-status-snapshot-sweep.md``
    §"Pattern families and the subset property".

    Pure: no I/O, no side effects.
    """
    return extract_snapshot_edge_task_ids_by_marker_class(fact).blocked_ids


class SnapshotEdgeIds(NamedTuple):
    """One edge fact's extracted task ids, split by marker class.

    ``blocked_ids`` is always a SUBSET of ``all_ids``, guaranteed by
    ``_ids_within_union``, not by pattern nesting.
    """

    #: Every id the fact asserts as active/pending/blocked/stalled/in-progress.
    all_ids: set[int]
    #: The subset of those the fact asserts as BLOCKED specifically.
    blocked_ids: set[int]


def extract_snapshot_edge_task_ids_by_marker_class(fact: str | None) -> SnapshotEdgeIds:
    """Extract BOTH id sets from *fact* in a single pass.

    THE single extraction implementation.
    ``extract_snapshot_edge_task_ids`` and
    ``extract_blocked_assertion_task_ids`` are thin accessors over it. The
    whole-fact gate runs once, and the ``_BLOCKED_MARKER_RE`` pre-gate keeps
    the blocked family off every edge with no 'blocked' token. A caller that
    needs both sets per edge calls this once; see
    ``docs/stale-status-snapshot-sweep.md``
    §"Extraction cost as a liveness property".

    *fact* is typed ``str | None`` because this is where a missing fact is
    normalised, exactly once per entry into the pipeline.

    Pure: no I/O, no side effects.
    """
    fact = fact or ''

    if not SNAPSHOT_STATUS_RE.search(fact):
        # No marker of ANY class — neither family can match, and the blocked
        # family is a subset of this gate, so one scan settles both.
        return SnapshotEdgeIds(all_ids=set(), blocked_ids=set())

    union = _extract_ids(fact, _UNION_PATTERNS)
    if not _BLOCKED_MARKER_RE.search(fact):
        return SnapshotEdgeIds(all_ids=union.ids, blocked_ids=set())
    return SnapshotEdgeIds(
        all_ids=union.ids,
        blocked_ids=_ids_within_union(fact, _BLOCKED_PATTERNS, union),
    )


def extract_marker_task_ids(
    fact: str | None, adjective_alt: str | None, transitive_alt: str | None,
) -> set[int]:
    """The ids *fact* asserts under ONE marker alternation.

    The family is built by the shipped builder from alternations shaped like
    ``_ADJECTIVE_MARKER_ALT`` / ``_TRANSITIVE_MARKER_ALT``, so it carries every
    shipped guard. Its ids pass the same union seam as
    ``SnapshotEdgeIds.blocked_ids``, which keeps them a subset of
    ``extract_snapshot_edge_task_ids(fact)``. A reporting seam for probes that
    ask which ids a fact asserts as, say, pending; selection itself reads
    ``extract_snapshot_edge_task_ids_by_marker_class``.

    Pure: no I/O, no side effects.
    """
    fact = fact or ''
    if not SNAPSHOT_STATUS_RE.search(fact):
        return set()
    union = _extract_ids(fact, _UNION_PATTERNS)
    return _ids_within_union(fact, _marker_family(adjective_alt, transitive_alt), union)


@functools.cache
def _marker_family(adjective_alt: str | None, transitive_alt: str | None) -> _SnapshotPatterns:
    return _build_snapshot_patterns(adjective_alt, transitive_alt)


def _ids_within_union(
    fact: str, family: _SnapshotPatterns, union: _ExtractionResult,
) -> set[int]:
    """*family*'s ids on an already-gated *fact*, inside the union family's result.

    THE seam every narrower family passes through — the blocked family and
    ``extract_marker_task_ids`` alike — so a narrower family can never assert
    an id the union family did not. See ``docs/stale-status-snapshot-sweep.md``
    §"What actually makes the subset property hold".
    """
    # Thread the union family's prepositional-complement rejections in:
    # rejected_spans is per-family algorithm data, and a narrower family may
    # never build the enumeration the union family rejected.
    narrowed = _extract_ids(
        fact, family, inherited_rejected_spans=union.rejected_spans,
    )
    # Intersect at the seam, so containment holds by construction: finditer
    # is non-overlapping, so a narrower alternation can unshadow a later
    # match the union family never reported.
    return narrowed.ids & union.ids


class _ExtractionResult(NamedTuple):
    """One family's extracted ids plus the rejected spans it established.

    The spans are returned rather than kept private because they are the one
    piece of ALGORITHM DATA that must cross the family boundary — see
    ``_extract_ids``'s ``inherited_rejected_spans`` parameter.
    """

    #: Ids this family attributed to the copula's subject.
    ids: set[int]
    #: Enumeration spans established to be a preposition's complement,
    #: inherited from the caller plus this family's own rejections.
    rejected_spans: tuple[tuple[int, int], ...]


def _plural_enum_ids(
    fact: str,
    pattern: re.Pattern[str] = PLURAL_ENUM_SNAPSHOT_RE,
    *,
    guard: Callable[[str], bool] = _enumeration_is_prepositional_complement,
) -> tuple[set[int], list[tuple[int, int]]]:
    """The plural-enumeration arm, in ONE place.

    Returns ``(ids, rejected_spans)``: the ids the plural path yields, and the
    spans of every enumeration *guard* rejected as a preposition's
    complement. Ids come from the match's ``'ids'`` group only.

    *guard* is injectable solely so
    ``fused-memory/scripts/measure_plural_enum_guard_recall.py`` can run this
    identical arm under a candidate guard; no production caller passes one.
    *pattern* defaults to the union family's plural arm, and ``_extract_ids``
    passes its own family's explicitly. See
    ``docs/stale-status-snapshot-sweep.md``
    §"The task-3949 measurement and verdict".

    Pure: no I/O, no side effects.
    """
    ids: set[int] = set()
    rejected_spans: list[tuple[int, int]] = []
    for enum in pattern.finditer(fact):
        # Subjecthood guard, in Python because lookbehind is fixed-width.
        if guard(fact[: enum.start()]):
            rejected_spans.append(enum.span())
            continue
        ids.update(int(tok) for tok in _BARE_DIGIT_RE.findall(enum.group('ids')))
    return ids, rejected_spans


def _extract_ids(
    fact: str,
    patterns: _SnapshotPatterns,
    *,
    inherited_rejected_spans: Sequence[tuple[int, int]] = (),
) -> _ExtractionResult:
    """Run one already-gated *fact* through one marker family's five paths.

    Steps 2-4 of ``extract_snapshot_edge_task_ids``'s algorithm; the gate is
    step 1 and stays with each caller, since the families gate differently.

    ``inherited_rejected_spans`` carries the prepositional-complement
    rejections ANOTHER family already established on the same fact, and the
    returned ``rejected_spans`` carries this family's own on top of them.
    That threading is load-bearing; see ``_ids_within_union``.

    Pure: no I/O, no side effects.
    """
    # This family's plural pattern is passed explicitly: the blocked family
    # narrows it and must not fall back to the union default.
    ids, own_rejected_spans = _plural_enum_ids(fact, patterns.plural_enum)
    rejected_spans: list[tuple[int, int]] = [
        *inherited_rejected_spans,
        *own_rejected_spans,
    ]

    # A rejected span is a preposition's complement, so no id inside it is the
    # copula's subject, whichever path found it. Scoped to the span, so a
    # genuine snapshot in a later clause survives.
    for pattern in (patterns.individual, patterns.genitive, patterns.status_phrase):
        ids.update(
            int(m.group(1))
            for m in pattern.finditer(fact)
            if not any(start <= m.start() < end for start, end in rejected_spans)
        )

    for intro in patterns.list_introducer.finditer(fact):
        # Read off the family, never branched on family identity.
        if patterns.reject_list_after_task_ref and _list_is_governed_by_task_ref(
            fact[: intro.start()]
        ):
            continue
        segment = COUNT_QUANTITY_RE.sub(' ', _list_segment(fact, intro.end(), intro.group('open')))
        ids.update(int(tok) for tok in _BARE_DIGIT_RE.findall(segment))

    return _ExtractionResult(ids=ids, rejected_spans=tuple(rejected_spans))
