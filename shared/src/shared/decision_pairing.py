"""THE owner of the plan-decision cross-pairing predicate (task 3967).

DETECTION-ONLY: this module reports, and has no repair counterpart and no
mutating entry point at all. :func:`scan_design_decisions` never assigns into
the document it walks, and the CLI over it
(``scripts/scan_plan_decision_pairing.py``) has no ``--apply`` or ``--repair``
flag and never will. That is a contract, not an omission — see "Why there is no
repair" below.

## The damage class

A ``design_decisions`` entry whose ``decision`` and ``rationale`` are each
perfectly well-formed prose, but whose *association* is wrong: the rationale
recorded under one decision-line actually argues for a different one. Both
texts are intact. Nothing is malformed, no sentinel is present, no parser is
upset.

This is a DIFFERENT damage class from the one :mod:`shared.toolcall_markup`
owns, and the two are disjoint at the detector. Measured over the 23 committed
specimens' 46 strings, 34 carry no envelope literal at all, so
``toolcall_markup.detect`` returns ``None`` and the write-time markup tripwire
admits them. ``shared/tests/test_decision_pairing_containment.py`` pins that as
an observation rather than an argument.

Folding this predicate into :mod:`shared.toolcall_markup` was considered and
rejected: that module owns an enumeration of ENVELOPE LITERALS, and a semantic
predicate over prose is a different kind of thing. Keeping them apart preserves
the one boundary the toolcall-markup PRD's INV-5 consolidation established.

## The predicate is a CONJUNCTION

An entry is reported when BOTH hold:

1. its ``decision`` is start-anchored (leading whitespace tolerated,
   case-insensitive) on one of :data:`HEADER_MARKERS`; **and**
2. one of :data:`PAIRING_MARKERS` appears anywhere in ``decision`` **or**
   ``rationale``.

Dropping either conjunct costs real specimens, which is why both are pinned as
load-bearing by their own tests:

* Without the **start-anchor**, three separate entries in task 3209's plan that
  merely use the word ``supersedes`` mid-sentence for a metadata edge kind are
  swept in, as is task 3298's ``(restatement of decision #1 ...)`` named
  parenthetically mid-prose, as is task 3692's plan — prose ABOUT another
  plan's mis-pairing, which is the false positive the originating task
  description acknowledged in its own phrase-list predicate.
* Without the **pairing conjunct**, task 3382's decision #5 is swept in: it
  opens ``SUPERSEDES decision #3`` and genuinely does supersede it, but it is a
  design REVERSAL, not a mis-pairing.

This module is the SINGLE OWNER of both marker tuples and of the accept/reject
decision, exactly as :mod:`shared.toolcall_markup` owns the envelope literals
(INV-5). No consumer re-spells a marker: the scanner script imports them, and
:func:`scan_design_decisions` delegates every verdict to
:func:`detect_mispairing` rather than re-implementing the predicate, so the
walker and the entry predicate can never disagree about what counts.

## Every count derived from this is a STRICT LOWER BOUND

The predicate can only ever find a mis-pairing that a human or an agent
NOTICED and then documented in a later entry. A mis-pairing nobody noticed
leaves no trace at all — both texts are well-formed, so there is nothing to
key on. Treat any number this module produces as a floor, never as a
prevalence estimate, and never report it as "N mis-pairings exist".

## Why there is no repair

:func:`shared.toolcall_markup.repair` can exist because envelope damage leaves
RESIDUE: the original argument text is still present, with parseable markup
wrapped around it, so a repair is a slice of its own input. Neither
precondition holds here. A mis-paired document carries both texts but no record
of which belonged where; the association itself is what was lost, and it is not
recoverable from the document. A "repair" would have to guess, and a guess
written back into a plan is worse than the visible damage.

Task 3692 reached the same conclusion from the other direction and deliberately
asserted no test that task 3567's plan is repaired, despite that plan being
named as its concrete damage specimen. That restraint is correct and this
module inherits it.

Symmetrically, no LOCAL DETERMINISTIC predicate can contain this at write time.
A correct ``(decision, rationale)`` pair and a swapped one both arrive as clean,
well-formed prose; nothing in the arguments distinguishes them. An LLM
coherence check is the one thing that could, and the toolcall-markup PRD's
decision D2 already rejected LLM mediation on this exact surface — the same
model class that emits the defect would be sitting in the write path. So the
correction entry an author appends after noticing is not merely the easiest
signal to key on; it is the only machine-visible one there is.

Remediation of anything this module finds is an author appending a correction,
or the supersede mechanism task 3865 will add — never a bulk rewrite.
"""
from __future__ import annotations

import re
from typing import NamedTuple

# ---------------------------------------------------------------------------
# The literal marker sets. ONE owner (INV-5); no consumer re-spells these.
# ---------------------------------------------------------------------------

#: Correction headers, matched only at the START of a ``decision`` (leading
#: whitespace tolerated, case-insensitive). Every one is observed in a live
#: victim plan. Order is the match order, so a decision opening
#: ``CORRECTION/RESTATEMENT`` reports ``CORRECTION``; no marker is a prefix of
#: another, so the order is not otherwise load-bearing.
HEADER_MARKERS: tuple[str, ...] = (
    'CORRECTION',
    'CORRECTED',
    'RESTATEMENT',
    'READ THIS INSTEAD',
    'SUPERSEDES',
)

#: Explicit pairing language, matched anywhere in ``decision`` or ``rationale``,
#: case-insensitively. This is the conjunct that separates a mis-pairing from a
#: genuine supersession. Order is the match order and decides which marker a hit
#: reports when a text carries more than one.
PAIRING_MARKERS: tuple[str, ...] = (
    'mis-paired',
    'cross-paired',
    'mis-titled',
    'mis-attributed',
    'recorded against the wrong',
    'swapped',
    'belongs to THIS decision',
)

#: The fields searched for pairing language, in the order searched. ``decision``
#: is first so a hit's reported field is deterministic when both carry a marker.
_PAIRING_FIELDS: tuple[str, ...] = ('decision', 'rationale')

_HEADER_RES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (marker, re.compile(r'\s*' + re.escape(marker), re.IGNORECASE))
    for marker in HEADER_MARKERS
)

_PAIRING_RES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (marker, re.compile(re.escape(marker), re.IGNORECASE))
    for marker in PAIRING_MARKERS
)


class MispairingHit(NamedTuple):
    """One self-documented mis-pairing, naming the evidence that found it.

    ``header`` and ``marker`` are the module's own declared literals, not the
    matched text, so a triager reading a report never has to log-scrape which
    literal fired (INV-2). ``field`` names which of ``decision``/``rationale``
    carried the pairing marker. ``index`` is the 0-based position in the plan's
    ``design_decisions`` list, and is ``None`` for an entry-level check made
    outside a document walk.
    """

    header: str
    marker: str
    field: str
    index: int | None = None


def _match_header(decision: object) -> str | None:
    """The declared header marker *decision* is start-anchored on, if any."""
    if not isinstance(decision, str):
        return None
    for marker, pattern in _HEADER_RES:
        if pattern.match(decision) is not None:
            return marker
    return None


def _match_pairing(decision: object, rationale: object) -> tuple[str, str] | None:
    """The first ``(marker, field)`` carrying pairing language, if any."""
    for field, value in zip(_PAIRING_FIELDS, (decision, rationale), strict=True):
        if not isinstance(value, str):
            continue
        for marker, pattern in _PAIRING_RES:
            if pattern.search(value) is not None:
                return marker, field
    return None


def detect_mispairing(decision: object, rationale: object) -> MispairingHit | None:
    """Report a self-documented mis-pairing in one ``design_decisions`` entry.

    Pure, synchronous and TOTAL: it never raises, for any input, including
    ``None`` and non-``str`` values. That is load-bearing rather than defensive
    — the scanner calls it over plan documents nobody validated, and a detector
    that raises is a detector that is switched off.

    Both conjuncts are required. The header conjunct is anchored on *decision*
    ALONE, so a non-``str`` decision can never produce a hit however the
    rationale reads.

    Returns the :class:`MispairingHit` naming both matched markers, or ``None``.
    """
    header = _match_header(decision)
    if header is None:
        return None
    pairing = _match_pairing(decision, rationale)
    if pairing is None:
        return None
    marker, field = pairing
    return MispairingHit(header=header, marker=marker, field=field)
