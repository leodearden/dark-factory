#!/usr/bin/env python3
"""Measure the plural-enumeration snapshot path's recall against the live corpus.

Task 3949. The question, inherited from task 3079's recall finding: what
does ``_enumeration_is_prepositional_complement`` actually COST on the real
Graphiti edge corpus?

The guard exists to stop the plural path selecting an enumeration that is a
preposition's complement rather than the copula's subject ('Reviews of tasks
1020 and 1030 are pending.' describes the REVIEWS, not the tasks). Its known
cost is that an ordinary sentence-initial adverbial preamble — a date stamp,
a location, a cycle scope — shares the clause with its own preposition and
so suppresses a genuine snapshot behind it ('As of 2026-08-09, tasks 1020
and 1030 are pending.'). Task 3079 declined to tighten the guard on
speculation and asked for a measurement first. This is that measurement.

Read-only. It enumerates valid edges and classifies their fact text; it
never writes to the corpus it measures.

WHY THIS IS A COMMITTED SCRIPT AND NOT A TRANSCRIPT
'Zero matches today' is a point-in-time fact about a corpus that grows every
cycle, so the verdict is only as good as its re-checkability. Following
census_memory_metadata.py's precedent: a report that exists only on an
operator's disk cannot be cited.

COUNTING RULE (why the fields are not required to sum). A fact counts
  - once toward ``regex_matched``      if >= 1 of its matches matched,
  - once toward ``guard_rejected``     if >= 1 of its matches is rejected,
  - once toward ``selected``           if >= 1 of its matches survives.
A fact carrying two enumerations, one governed by a preposition and one not,
is therefore counted in BOTH guard_rejected and selected. The counts are
per-FACT because the reportable unit is 'edges whose retirement this guard
changes', not 'regex spans'.

``lexical_precondition`` counts facts matching this module's own
``_LEXICAL_PRECONDITION_RE`` (``\\btasks\\b\\s*#?\\s*\\d``) — the necessary
prefix of PLURAL_ENUM_SNAPSHOT_RE, and nothing more. Its purpose is to
separate two very different zeroes: 'the corpus contains no plural-task
shapes at all' (nothing to measure) from 'the corpus contains them and the
guard is eating them' (a real recall cost). Without it a headline zero is
uninterpretable.

Regenerate:

    cd fused-memory && uv run python scripts/measure_plural_enum_guard_recall.py
"""
from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

# The probe IMPORTS the shipped regex and guard rather than re-spelling
# them. This is load-bearing, not stylistic: a copied pattern measures a
# stale spelling of the thing it claims to measure, and would keep
# reporting a reassuring zero after the guard it audits had changed
# underneath it. Any drift now surfaces as an ImportError, not as a wrong
# number.
_SRC = Path(__file__).resolve().parent.parent / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fused_memory.reconciliation.stale_status_snapshot_edge_sweep import (  # noqa: E402
    _BARE_DIGIT_RE,
    _ENUM_PREP_WORD_RE,
    PLURAL_ENUM_SNAPSHOT_RE,
    _enumeration_is_prepositional_complement,
    _last_clause_break,
)

# The necessary lexical prefix of PLURAL_ENUM_SNAPSHOT_RE, spelled here on
# purpose — it is the PROBE's near-miss counter, not a copy of the shipped
# pattern. It answers 'were there any plural-task shapes to lose?', which
# the shipped regex cannot answer about the facts it rejects.
_LEXICAL_PRECONDITION_RE: re.Pattern[str] = re.compile(
    r'\btasks\b\s*#?\s*\d', re.IGNORECASE,
)


@dataclass(frozen=True)
class Rejection:
    """One guard-rejected match, carried into the report for triage.

    Recording the fact text AND the match offset means a future nonzero run
    is diagnosable from the committed artifact alone — the reader can
    reconstruct the exact prefix the guard was handed, without re-running
    the probe against a corpus that has since moved.
    """

    fact: str
    match_start: int


@dataclass(frozen=True)
class ScanResult:
    """Per-corpus counts. See the module docstring for the counting rule."""

    facts_scanned: int = 0
    lexical_precondition: int = 0
    regex_matched: int = 0
    guard_rejected: int = 0
    selected: int = 0
    rejections: list[Rejection] = field(default_factory=list)


def scan_corpus(facts: Iterable[str]) -> ScanResult:
    """Classify every fact in *facts* against the shipped regex and guard.

    Pure: no I/O, no live backend, no ordering dependence beyond the input's
    own order (which the rejection list preserves so the report's samples
    are stable).
    """
    facts_scanned = 0
    lexical_precondition = 0
    regex_matched = 0
    guard_rejected = 0
    selected = 0
    rejections: list[Rejection] = []

    for fact in facts:
        facts_scanned += 1
        if _LEXICAL_PRECONDITION_RE.search(fact):
            lexical_precondition += 1

        matches = list(PLURAL_ENUM_SNAPSHOT_RE.finditer(fact))
        if not matches:
            continue
        regex_matched += 1

        fact_rejections = [
            Rejection(fact=fact, match_start=m.start())
            for m in matches
            # The guard is handed the prefix PRECEDING the match, exactly as
            # extract_snapshot_edge_task_ids hands it.
            if _enumeration_is_prepositional_complement(fact[: m.start()])
        ]
        if fact_rejections:
            guard_rejected += 1
            rejections.extend(fact_rejections)
        if len(fact_rejections) < len(matches):
            selected += 1

    return ScanResult(
        facts_scanned=facts_scanned,
        lexical_precondition=lexical_precondition,
        regex_matched=regex_matched,
        guard_rejected=guard_rejected,
        selected=selected,
        rejections=rejections,
    )


# ---------------------------------------------------------------------------
# Rejection triage (REPORT-ONLY)
# ---------------------------------------------------------------------------

# Closed-class tokens that may open an adverbial preamble WITHOUT being the
# head noun that governs a following preposition. Deliberately tiny: 'as'
# is here for the finding's own motivating shape ('As of <date>, ...'), the
# rest are coordinators and sentence adverbs that can precede a preamble.
# Anything NOT listed reads as a content word — i.e. a possible governing
# head — and pushes the label to 'prepositional_complement', which is the
# fail-safe direction for this heuristic (see triage_rejection).
_PREAMBLE_OPENER_WORDS: frozenset[str] = frozenset({
    'as', 'and', 'but', 'or', 'so', 'yet',
    'then', 'now', 'also', 'however', 'meanwhile',
})

_WORD_TOKEN_RE: re.Pattern[str] = re.compile(r"[^\W\d_][\w'-]*")

ADVERBIAL_PREAMBLE = 'adverbial_preamble'
PREPOSITIONAL_COMPLEMENT = 'prepositional_complement'


def triage_rejection(fact: str, match_start: int) -> str:
    """Classify one guard rejection: recall loss, or a correct rejection?

    Returns ``'adverbial_preamble'`` (the enumeration really is the copula's
    SUBJECT and the guard fired on a scene-setting preamble in front of it —
    genuine recall loss) or ``'prepositional_complement'`` (the copula's real
    subject is an outer head noun and the rejection is correct).

    THIS IS A TRIAGE HEURISTIC FOR REPORTING ONLY. It never feeds extraction,
    so its errors cost report accuracy, not corpus correctness. When in doubt
    it MUST answer ``'prepositional_complement'``: over-reporting recall loss
    is what would wrongly justify tightening the guard, and a tightening
    trades back toward the unrecoverable over-selection direction. Every
    inconclusive branch below therefore falls through to that label.

    The discriminator is structural. A genuine adverbial preamble is
    preposition-INITIAL within its clause (modulo a closed-class opener like
    'as') and is closed off by a comma before the enumeration. A
    prepositional complement has a governing HEAD NOUN in front of the
    preposition — which is what separates

        'As of 2026-08-09, tasks 1020 and 1030 are pending.'   -> preamble
        'Blockers for down-stream, still-unmerged tasks ...'   -> complement

    even though both carry a comma between the preposition and the
    enumeration. That second shape is load-bearing: keying on comma presence
    alone mislabels it, and it is exactly the case candidate tightening (b)
    was measured to get wrong.

    Scoped to the same clause the shipped guard sees — ``_last_clause_break``
    is reused rather than re-derived, so the triage cannot disagree with the
    guard about where the clause starts.
    """
    prefix = fact[:match_start]
    clause = prefix[_last_clause_break(prefix) + 1:]

    # No comma => no preamble boundary at all; the preposition governs
    # straight through to the enumeration.
    last_comma = clause.rfind(',')
    if last_comma < 0:
        return PREPOSITIONAL_COMPLEMENT

    # A listed preposition AFTER the comma governs the enumeration directly,
    # whatever came before it.
    tail = clause[last_comma + 1:]
    if _ENUM_PREP_WORD_RE.search(tail):
        return PREPOSITIONAL_COMPLEMENT

    # A plural head noun between the comma and the enumeration could itself
    # be what the copula agrees with ('..., statuses tasks 1020 and 1030
    # are pending'), so the enumeration is not unambiguously the subject.
    for token in _WORD_TOKEN_RE.findall(tail):
        if token.lower().endswith('s') and token.lower() not in _PREAMBLE_OPENER_WORDS:
            return PREPOSITIONAL_COMPLEMENT

    # The clause's FIRST listed preposition decides it: anything open-class
    # in front of that preposition is a candidate governing head, which makes
    # this a complement rather than a preamble.
    first_prep = _ENUM_PREP_WORD_RE.search(clause)
    if first_prep is None:
        # The guard fired, so a listed preposition exists somewhere in the
        # scanned span; not finding one here means the two disagree about
        # scope. Report conservatively rather than claiming recall loss.
        return PREPOSITIONAL_COMPLEMENT

    before_prep = clause[: first_prep.start()]
    for token in _WORD_TOKEN_RE.findall(before_prep):
        if token.lower() not in _PREAMBLE_OPENER_WORDS:
            return PREPOSITIONAL_COMPLEMENT

    return ADVERBIAL_PREAMBLE


# ---------------------------------------------------------------------------
# Candidate tightenings (REPORT-ONLY SIMULATION)
# ---------------------------------------------------------------------------
#
# The two candidates task 3079 named and declined to apply on speculation.
# Simulated here so the decision is measured rather than argued.
#
# THESE ARE NOT SHIPPABLE GUARDS AND MUST NOT BE IMPORTED BY PRODUCTION
# CODE. They exist to answer 'what would this have cost?' against the
# pinned shape corpus, and they are wired into nothing but the report.
#
# A caveat that bounds what the simulation can prove: each candidate below
# is ONE plausible spelling of the idea, and a different spelling could
# measure differently. That is why the corpus zero, not the simulation,
# carries the verdict — the simulation is corroboration. A candidate that
# looked clean here would still have zero measured benefit on a corpus
# where the regex matches nothing.


def _guard_candidate_a(prefix: str) -> bool:
    """(a) Require a capitalized-or-plural head token before the preposition.

    Spelling assumed: fire only when SOME listed preposition in the clause
    is IMMEDIATELY preceded by a token that looks like a noun-phrase head —
    capitalized, or plural-looking ('...s'). The intent is that 'Reviews of
    tasks ...' fires (head 'Reviews') while a bare adverbial preamble does
    not.

    Measured limitation: it does not recover 'As of <date>, tasks ...',
    because 'of' there is immediately preceded by the capitalized
    sentence-initial 'As'. That is the finding's own motivating shape.
    """
    clause = prefix[_last_clause_break(prefix) + 1:]
    for prep in _ENUM_PREP_WORD_RE.finditer(clause):
        tokens = _WORD_TOKEN_RE.findall(clause[: prep.start()])
        if tokens and (tokens[-1][:1].isupper() or tokens[-1].lower().endswith('s')):
            return True
    return False


def _guard_candidate_b(prefix: str) -> bool:
    """(b) Restart the backward scan after a comma that no preposition follows.

    Spelling assumed: if the clause's LAST comma is followed by no listed
    preposition, treat that comma as a preamble boundary and search only
    the text after it; otherwise scan the whole clause as shipped.

    Measured limitation: a comma can be intra-clause rather than a preamble
    boundary — 'Blockers for down-stream, still-unmerged tasks 1020 and
    1030 are pending.' has a coordinate-adjective comma — so this restart
    discards the governing 'for' and re-opens a pinned over-selection.
    """
    clause = prefix[_last_clause_break(prefix) + 1:]
    comma = clause.rfind(',')
    if comma >= 0 and not _ENUM_PREP_WORD_RE.search(clause, comma + 1):
        clause = clause[comma + 1:]
    return _ENUM_PREP_WORD_RE.search(clause) is not None


_CANDIDATE_GUARDS = {
    'shipped': _enumeration_is_prepositional_complement,
    'a': _guard_candidate_a,
    'b': _guard_candidate_b,
}

CANDIDATE_NAMES: tuple[str, ...] = ('shipped', 'a', 'b')


def extract_plural_ids(fact: str, *, candidate: str = 'shipped') -> set[int]:
    """Ids the PLURAL path yields for *fact* under the named guard.

    Ids come from the match's ``'ids'`` capture group ONLY, via the sweep
    module's ``_BARE_DIGIT_RE`` — preserving extract_snapshot_edge_task_ids'
    invariant (d), that a bare '\\d+' contributes an id only from inside an
    already-detected, marker-anchored span.

    This is the plural path in isolation, deliberately: it is the only path
    either candidate can change, so comparing it isolates the candidate's
    effect from the whole-fact status gate and the other extraction paths.
    """
    guard = _CANDIDATE_GUARDS[candidate]
    ids: set[int] = set()
    for match in PLURAL_ENUM_SNAPSHOT_RE.finditer(fact):
        if guard(fact[: match.start()]):
            continue
        ids.update(int(d) for d in _BARE_DIGIT_RE.findall(match.group('ids')))
    return ids


@dataclass(frozen=True)
class CandidateResult:
    """What one candidate tightening would change against a shape corpus.

    ``over_selected`` is disqualifying and ``recovered`` is the benefit; the
    two are separated by ``triage_rejection``, so the same newly-admitted
    match is scored as a regression or a recovery on its own linguistic
    merits rather than on which list the caller passed it in.
    """

    name: str
    over_selected: list[str] = field(default_factory=list)
    recovered: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)


def simulate_candidate(name: str, facts: Iterable[str]) -> CandidateResult:
    """Compare the named candidate guard's outcome against the shipped one.

    For every regex match the SHIPPED guard rejects but the candidate
    admits, the newly-admitted match is triaged: an 'adverbial_preamble'
    counts as ``recovered`` (the tightening's benefit), a
    'prepositional_complement' counts as ``over_selected`` (a re-opened
    precision regression, the unrecoverable direction).

    A candidate can only ever ADMIT matches the shipped guard rejects if it
    is strictly weaker on some prefix; a candidate that instead rejects
    something the shipped guard admits shows up as a changed id set in
    ``extract_plural_ids``, which the subject-position positives pin.
    """
    over_selected: list[str] = []
    recovered: list[str] = []
    unchanged: list[str] = []

    guard = _CANDIDATE_GUARDS[name]
    for fact in facts:
        changed = False
        for match in PLURAL_ENUM_SNAPSHOT_RE.finditer(fact):
            prefix = fact[: match.start()]
            if not _enumeration_is_prepositional_complement(prefix):
                continue  # shipped already selects it; nothing to recover
            if guard(prefix):
                continue  # candidate agrees with the shipped rejection
            changed = True
            if triage_rejection(fact, match.start()) == ADVERBIAL_PREAMBLE:
                recovered.append(fact)
            else:
                over_selected.append(fact)
        if not changed:
            unchanged.append(fact)

    return CandidateResult(
        name=name,
        over_selected=over_selected,
        recovered=recovered,
        unchanged=unchanged,
    )
