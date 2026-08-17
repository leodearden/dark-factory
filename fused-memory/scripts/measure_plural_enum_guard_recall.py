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

import argparse
import asyncio
import json
import logging
import re
import sys
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

logger = logging.getLogger('measure_plural_enum_guard_recall')

# plans/ lives two levels above scripts/ — same derivation
# census_memory_metadata.py uses for its committed artifact paths.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_JSON_OUT = str(_REPO_ROOT / 'plans' / 'plural-enum-guard-recall-report.json')
DEFAULT_MD_OUT = str(_REPO_ROOT / 'plans' / 'plural-enum-guard-recall-report.md')


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


# ---------------------------------------------------------------------------
# Edge enumeration
# ---------------------------------------------------------------------------

# FalkorDB's server-wide RESULTSET_SIZE is 10000 and nothing in this repo
# overrides it, so ANY single query returning more rows than that is
# SILENTLY truncated — no error, no warning, no partial-result flag.
#
# Measured on the live dark_factory graph (task 3949 planning):
#   get_all_valid_edges' exact query      -> 24902 rows, 10000 returned
#   distinct valid edges actually exposed -> 6376 of 12488 (51%)
#
# Default page size is therefore well under the cap, so a page that comes
# back short really is the end of the data rather than the server's ceiling.
DEFAULT_PAGE_SIZE = 5000

# FalkorDB's server-wide result-set ceiling.
#
# THIS IS AN ASSUMPTION ABOUT SERVER CONFIGURATION, not something this repo
# sets — which is exactly why the census cross-check below does not trust it.
# The value only has to be RIGHT for the structural check to be useful; the
# empirical check is what stays correct when it is wrong.
RESULTSET_SIZE = 10000

# Hard safety cap on the number of SKIP/LIMIT pages a single graph's
# enumeration will fetch, bounding worst-case pagination against a
# pathological corpus. Normal use never approaches it: at the default page
# size this is 5,000,000 edges against a live maximum near 16,000. Copied
# from census_foreign_nodes' MAX_CENSUS_PAGES precedent
# (scripts/migrate_cross_graph_leak.py:161), including its rule that hitting
# the cap on a still-full page is reported, never swallowed.
MAX_ENUM_PAGES = 1000

# The same MATCH pattern GraphitiBackend.get_all_valid_edges uses, so this
# measures the corpus the production sweep is aimed at — but issued with
# SKIP/LIMIT instead of unpaginated. This is DELIBERATELY not a call to
# get_all_valid_edges: inheriting that function's truncation would make the
# headline zero meaningless. The truncation is a real latent coverage bug in
# the production sweep, but it lives in graphiti_client.py, outside this
# task's scope — it is filed as a follow-up and noted in the report rather
# than fixed here.
#
# ORDER BY is load-bearing, not cosmetic. Every page is a SEPARATE query, and
# DISTINCT + SKIP/LIMIT with no total order gives the store no obligation to
# return rows in the same order twice — so `SKIP n` on page 2 can skip rows
# page 1 never returned (silently dropped, permanently) or re-return rows it
# did (harmlessly deduped here, which is what makes the drop so easy to miss).
_EDGE_PAGE_CYPHER = (
    'MATCH (n:Entity)-[e:RELATES_TO]-() '
    'WHERE e.invalid_at IS NULL '
    'RETURN DISTINCT e.uuid, e.fact '
    'ORDER BY e.uuid '
    'SKIP {skip} LIMIT {limit}'
)

# The empirical completeness proof. Deliberately the SAME MATCH and WHERE as
# the page query, so the two numbers describe the same population and are
# therefore comparable; DISTINCT on e.uuid so the undirected match's
# double-attribution collapses exactly as the paged dict-dedup collapses it.
#
# It returns exactly ONE row, which is the whole point: a single-row result
# can never be truncated by the row cap it is being used to detect. That is
# what makes `len(facts) == expected` a proof rather than one more heuristic.
_EDGE_COUNT_CYPHER = (
    'MATCH (n:Entity)-[e:RELATES_TO]-() '
    'WHERE e.invalid_at IS NULL '
    'RETURN count(DISTINCT e.uuid)'
)


async def _census_count(
    query_fn: Callable[[str], Awaitable[Sequence[Sequence[Any]]]],
) -> int | None:
    """Distinct valid edges the store reports, or None if it did not say.

    Every 'it did not say' shape collapses to None — no rows, a null result
    set, a NULL count, a row with no columns, a non-integer — because the
    caller treats None as fail-closed and there is nothing to gain by
    distinguishing between flavours of missing evidence.
    """
    rows = list(await query_fn(_EDGE_COUNT_CYPHER) or [])
    if not rows:
        return None
    row = rows[0]
    if row is None or len(row) == 0 or row[0] is None:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None


async def enumerate_valid_edge_facts(
    query_fn: Callable[[str], Awaitable[Sequence[Sequence[Any]]]],
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    resultset_size: int = RESULTSET_SIZE,
    max_pages: int = MAX_ENUM_PAGES,
) -> tuple[dict[str, str], bool]:
    """Enumerate every valid RELATES_TO edge's fact text, keyed on edge uuid.

    Pages with SKIP/LIMIT until a page comes back short or empty. Dedupes on
    edge uuid: the undirected MATCH attributes each directed edge to BOTH of
    its endpoints, so the same edge uuid legitimately arrives more than once
    (documented on get_all_valid_edges). A NULL fact is coerced to ''.

    Returns ``(facts_by_uuid, complete)``. The flag is the fail-closed hook:
    an under-enumerated corpus must be reported as a FAILURE rather than as a
    smaller report, because the headline result is a zero and a truncated
    zero is worthless.

    ``complete`` is FALSE — and every one of these paths logs a WARNING
    naming the numbers, so a shortfall is never silent — when:

    1. ``page_size >= resultset_size``. The short-page break reasons 'this
       page was not full, so the data is exhausted', which is sound ONLY if
       the server cannot be what shortened it. At or above the cap the two
       causes are indistinguishable, so no enumeration is attempted at all
       and an EMPTY dict is returned: a partial dict invites a caller to use
       it anyway.
    2. The ``max_pages`` bound is reached while the last page was still full
       — a suspected shortfall, reported rather than swallowed.
    3. The census probe did not answer with a usable count. An unavailable
       proof is not a passing proof.
    4. The census count and the number enumerated disagree.

    (1) and (2) are STRUCTURAL; (3) and (4) are EMPIRICAL, and the two kinds
    are deliberately independent rather than redundant.

    The structural check is ``>=`` and not ``>``: equality is arithmetically
    safe on a server configured at exactly ``RESULTSET_SIZE``, but that
    constant is an ASSUMPTION about server configuration, so equality leaves
    zero margin — a server configured one row lower silently re-opens the
    truncation. One page of throughput buys that margin.

    The census check exists because that reasoning bottoms out on a guess,
    and the defect class ('a short page was mistaken for end-of-data') does
    not. If the live server is configured BELOW the assumed constant, then
    ``page_size < resultset_size`` passes the structural check and the
    short-page break lies exactly as it would have before — the identical
    silent truncation, undetected. Comparing what was enumerated against a
    single-row count that the cap cannot truncate catches that, and catches
    causes not enumerated here: unstable DISTINCT+SKIP/LIMIT page
    boundaries, a dropped page.

    Both are kept because they fail differently and usefully. The structural
    check fails FAST, before any work, with a specific operator-actionable
    reason ('re-run with a smaller --page-size'); the census check catches
    everything else, at the cost of only being able to report a bare count
    mismatch.
    """
    if page_size >= resultset_size:
        logger.warning(
            'enumerate_valid_edge_facts: page_size=%d is at or above the '
            "server's result-set cap (resultset_size=%d), so a short page "
            'cannot be distinguished from a server-truncated one and '
            'completeness is unprovable. Refusing to enumerate — re-run '
            'with --page-size well below %d.',
            page_size, resultset_size, resultset_size,
        )
        return {}, False

    # Taken BEFORE paging so the target is fixed up front rather than
    # inferred from the same pages whose completeness is in question.
    expected = await _census_count(query_fn)
    if expected is None:
        logger.warning(
            'enumerate_valid_edge_facts: the census probe returned no usable '
            'count, so completeness cannot be proven. Reporting INCOMPLETE — '
            'an unavailable proof is not a passing one.',
        )

    facts: dict[str, str] = {}
    skip = 0
    paged_to_the_end = False
    for _page in range(max_pages):
        page = await query_fn(
            _EDGE_PAGE_CYPHER.format(skip=skip, limit=page_size),
        )
        rows = list(page or [])
        for row in rows:
            edge_uuid = row[0]
            if edge_uuid is None or edge_uuid in facts:
                continue
            facts[edge_uuid] = row[1] or ''
        if len(rows) < page_size:
            # Short (or empty) page, and page_size < resultset_size was
            # checked above — so the server cannot be what shortened it and
            # the data really is exhausted. A FULL page can never prove that,
            # which is exactly why the loop continues. Note this is only the
            # STRUCTURAL half of the argument; the census check below is what
            # holds when the assumption behind it does not.
            paged_to_the_end = True
            break
        skip += len(rows)

    if not paged_to_the_end:
        logger.warning(
            'enumerate_valid_edge_facts: hit the %d-page cap (page_size=%d, '
            'enumerated=%d) while the last page was still full — enumeration '
            'is incomplete. Re-run with a larger --page-size.',
            max_pages, page_size, len(facts),
        )
    elif expected is not None and len(facts) != expected:
        logger.warning(
            'enumerate_valid_edge_facts: enumerated %d distinct edges but the '
            'census reports %d (page_size=%d) — the enumeration is SHORT. '
            'The most likely cause is a server result-set cap below the '
            'assumed %d; unstable page boundaries would do it too. Reporting '
            'INCOMPLETE.',
            len(facts), expected, page_size, resultset_size,
        )

    complete = (
        paged_to_the_end and expected is not None and len(facts) == expected
    )
    return facts, complete


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

DEFAULT_PROJECT_IDS: tuple[str, ...] = (
    'dark_factory', 'reify', 'solar_challenge_platform',
)
DEFAULT_MAX_SAMPLES = 20

# The test that mechanically re-validates both candidates against the full
# pinned precision parametrization. Named in the report so a reader can go
# check the corroboration rather than take the verdict on faith. The script
# deliberately does NOT import it: the shipped probe must not depend on a
# test module.
REVALIDATION_TEST = (
    'tests/test_measure_plural_enum_guard_recall.py'
    '::test_candidate_b_recovers_every_preamble_but_re_opens_an_over_selection'
)

_VERDICT_ZERO_MATCHES = (
    'DO NOT TIGHTEN. No fact in any enumerated project graph matches '
    'PLURAL_ENUM_SNAPSHOT_RE, so _enumeration_is_prepositional_complement '
    'rejects nothing and its recall cost on this corpus is exactly zero '
    'edges. A tightening can only change an outcome on a fact the regex '
    'already matched; with none, both candidates have provably zero '
    'measured benefit against nonzero unrecoverable over-selection risk.'
)
_VERDICT_PROVISIONAL = (
    'PROVISIONAL — the corpus now contains facts that reach the guard, so '
    'the zero-benefit argument no longer applies unexamined. Read the '
    'triage breakdown below: rejections labelled adverbial_preamble are '
    'genuine recall loss and are what a tightening would be for. No verdict '
    'is asserted here; re-decide against these numbers.'
)


@dataclass(frozen=True)
class ProjectReport:
    """One project graph's measurement."""

    project_id: str
    valid_edges: int
    complete: bool
    scan: ScanResult
    triage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Report:
    """The whole measurement, as committed to plans/."""

    schema_version: int
    measured_at: str
    page_size: int
    projects: list[ProjectReport]
    totals: ScanResult
    total_valid_edges: int
    triage_totals: dict[str, int]
    candidates: list[CandidateResult]
    verdict: str
    complete: bool
    max_samples: int
    revalidation_test: str = REVALIDATION_TEST


def exit_code(report: Report) -> int:
    """0 only for a COMPLETE measurement.

    Fail-closed, copied from census_memory_metadata.py's rule: the artifact
    still lands — the evidence of the shortfall is IN it — but the exit code
    refuses to call an under-enumerated measurement a successful one.
    """
    return 0 if report.complete else 1


def _sum_scans(scans: Iterable[ScanResult]) -> ScanResult:
    """Totals derived FROM the per-project rows, never computed separately."""
    totals = ScanResult()
    for scan in scans:
        totals = ScanResult(
            facts_scanned=totals.facts_scanned + scan.facts_scanned,
            lexical_precondition=(
                totals.lexical_precondition + scan.lexical_precondition
            ),
            regex_matched=totals.regex_matched + scan.regex_matched,
            guard_rejected=totals.guard_rejected + scan.guard_rejected,
            selected=totals.selected + scan.selected,
            rejections=[*totals.rejections, *scan.rejections],
        )
    return totals


def _triage_counts(scan: ScanResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rejection in scan.rejections:
        label = triage_rejection(rejection.fact, rejection.match_start)
        counts[label] = counts.get(label, 0) + 1
    return counts


async def run(args: Any, *, edge_source: Any) -> Report:
    """Measure every requested project through the injected *edge_source*.

    ``edge_source(project_id, *, page_size) -> (facts_by_uuid, complete)`` is
    the only corpus access, mirroring cleanup_count_snapshots.run(args, *,
    memory): every test drives a fake through it, so the whole aggregation
    band is checkable with no live backend.
    """
    projects: list[ProjectReport] = []
    for project_id in args.project_id:
        logger.info('enumerating project=%s', project_id)
        facts_by_uuid, complete = await edge_source(
            project_id, page_size=args.page_size,
        )
        scan = scan_corpus(facts_by_uuid.values())
        projects.append(ProjectReport(
            project_id=project_id,
            valid_edges=len(facts_by_uuid),
            complete=complete,
            scan=scan,
            triage=_triage_counts(scan),
        ))
        logger.info(
            'project=%s edges=%d matched=%d rejected=%d selected=%d complete=%s',
            project_id, len(facts_by_uuid), scan.regex_matched,
            scan.guard_rejected, scan.selected, complete,
        )

    totals = _sum_scans(p.scan for p in projects)
    triage_totals: dict[str, int] = {}
    for project in projects:
        for label, count in project.triage.items():
            triage_totals[label] = triage_totals.get(label, 0) + count

    # The candidates are simulated over the LIVE facts that actually reached
    # the guard. With zero such facts this is empty — which is precisely the
    # measurement: a tightening has nothing to act on.
    live_facts = [r.fact for r in totals.rejections]
    candidates = [
        simulate_candidate(name, live_facts) for name in CANDIDATE_NAMES
    ]

    return Report(
        schema_version=SCHEMA_VERSION,
        measured_at=args.measured_at,
        page_size=args.page_size,
        projects=projects,
        totals=totals,
        total_valid_edges=sum(p.valid_edges for p in projects),
        triage_totals=triage_totals,
        candidates=candidates,
        verdict=(
            _VERDICT_ZERO_MATCHES if totals.regex_matched == 0
            else _VERDICT_PROVISIONAL
        ),
        complete=all(p.complete for p in projects),
        max_samples=args.max_samples,
    )


# ---------------------------------------------------------------------------
# Artifact rendering
# ---------------------------------------------------------------------------
#
# EVERY ordering below is an explicit sort and every timestamp comes from
# report.measured_at. The artifacts are committed, so a dict-iteration or
# clock-derived difference between two identical runs would be permanent
# diff churn hiding the one line that actually changed.

REGENERATE_COMMAND = (
    'cd fused-memory && uv run python scripts/measure_plural_enum_guard_recall.py'
)


def _sorted_projects(report: Report) -> list[ProjectReport]:
    return sorted(report.projects, key=lambda p: p.project_id)


def _scan_payload(scan: ScanResult) -> dict[str, Any]:
    return {
        'facts_scanned': scan.facts_scanned,
        'lexical_precondition': scan.lexical_precondition,
        'regex_matched': scan.regex_matched,
        'guard_rejected': scan.guard_rejected,
        'selected': scan.selected,
    }


def _rejection_payload(rejections: list[Rejection]) -> list[dict[str, Any]]:
    """Rejections, sorted by (fact, offset) so the list never reorders."""
    return [
        {
            'fact': r.fact,
            'match_start': r.match_start,
            'triage': triage_rejection(r.fact, r.match_start),
        }
        for r in sorted(rejections, key=lambda r: (r.fact, r.match_start))
    ]


def render_json(report: Report) -> str:
    """The citable machine-readable record. Never truncated."""
    payload = {
        'schema_version': report.schema_version,
        'measured_at': report.measured_at,
        'generated_by': 'fused-memory/scripts/measure_plural_enum_guard_recall.py',
        'regenerate': REGENERATE_COMMAND,
        'page_size': report.page_size,
        'complete': report.complete,
        'verdict': report.verdict,
        'revalidation_test': report.revalidation_test,
        'total_valid_edges': report.total_valid_edges,
        'totals': _scan_payload(report.totals),
        'triage_totals': dict(sorted(report.triage_totals.items())),
        'projects': [
            {
                'project_id': project.project_id,
                'valid_edges': project.valid_edges,
                'complete': project.complete,
                'scan': _scan_payload(project.scan),
                'triage': dict(sorted(project.triage.items())),
                'rejections': _rejection_payload(project.scan.rejections),
            }
            for project in _sorted_projects(report)
        ],
        'candidates': [
            {
                'name': candidate.name,
                'over_selected': sorted(candidate.over_selected),
                'recovered': sorted(candidate.recovered),
                'unchanged_count': len(candidate.unchanged),
            }
            for candidate in report.candidates
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=False) + '\n'


def render_markdown(report: Report) -> str:
    """The human-readable record: provenance first, verdict last."""
    lines: list[str] = [
        '# Plural-enumeration guard: recall against the live edge corpus',
        '',
        '_Generated by `fused-memory/scripts/measure_plural_enum_guard_recall.py` '
        '(task 3949). Do not hand-edit — regenerate:_',
        '',
        f'    {REGENERATE_COMMAND}',
        '',
        f'- Measured at: `{report.measured_at}`',
        f'- Enumeration page size: `{report.page_size}`',
        f'- Enumeration complete: `{report.complete}`',
        f'- Schema version: `{report.schema_version}`',
        '',
        'Answers task 3079\'s open question: what does '
        '`_enumeration_is_prepositional_complement` actually COST on the real '
        'corpus? The counts below are per-FACT — a fact reaching the guard '
        'more than once is counted once per outcome, so the columns are not '
        'required to sum.',
        '',
        '## Per-project measurement',
        '',
        '| project | valid edges | `tasks <n>` near-misses | regex matches | '
        'guard rejections | selections | complete |',
        '| --- | ---: | ---: | ---: | ---: | ---: | :---: |',
    ]
    for project in _sorted_projects(report):
        scan = project.scan
        lines.append(
            f'| `{project.project_id}` | {project.valid_edges:,} | '
            f'{scan.lexical_precondition:,} | {scan.regex_matched:,} | '
            f'{scan.guard_rejected:,} | {scan.selected:,} | '
            f'{"yes" if project.complete else "**NO**"} |'
        )
    totals = report.totals
    lines += [
        f'| **(all)** | **{report.total_valid_edges:,}** | '
        f'**{totals.lexical_precondition:,}** | **{totals.regex_matched:,}** | '
        f'**{totals.guard_rejected:,}** | **{totals.selected:,}** | '
        f'**{"yes" if report.complete else "NO"}** |',
        '',
        'The near-miss column is what makes a zero interpretable: it separates '
        '"the corpus holds no plural-task shapes at all" from "it holds them '
        'and the guard is eating them".',
        '',
        '## Rejection triage',
        '',
    ]
    if report.triage_totals:
        lines += ['| label | count | meaning |', '| --- | ---: | --- |']
        meanings = {
            ADVERBIAL_PREAMBLE:
                'genuine RECALL LOSS — the enumeration really is the subject',
            PREPOSITIONAL_COMPLEMENT:
                'CORRECT rejection — an outer head noun is the subject',
        }
        for label, count in sorted(report.triage_totals.items()):
            lines.append(f'| `{label}` | {count:,} | {meanings.get(label, "")} |')
    else:
        lines.append('_No rejections to triage — the guard rejected nothing._')
    lines += ['', '### Rejection samples', '']
    sampled_any = False
    for project in _sorted_projects(report):
        samples = _rejection_payload(project.scan.rejections)[: report.max_samples]
        if not samples:
            continue
        sampled_any = True
        total = len(project.scan.rejections)
        lines.append(f'**`{project.project_id}`** '
                     f'(showing {len(samples)} of {total:,}):')
        lines.append('')
        lines += [f'- `{s["triage"]}` @{s["match_start"]}: {s["fact"]}'
                  for s in samples]
        lines.append('')
        if total > len(samples):
            # No silent caps: state what the markdown dropped, and where the
            # untruncated list lives.
            lines += [
                f'_{total - len(samples):,} further rejection(s) omitted here '
                f'(`--max-samples {report.max_samples}`); the JSON artifact '
                f'carries all of them._',
                '',
            ]
    if not sampled_any:
        lines += ['_None._', '']

    lines += [
        '## Candidate tightenings',
        '',
        'The two candidates task 3079 named and declined to apply on '
        'speculation, simulated against the facts that actually reached the '
        'guard in this run. `over_selected` is disqualifying (the '
        'unrecoverable direction); `recovered` is the benefit.',
        '',
        '| candidate | over-selections re-opened | preamble shapes recovered |',
        '| --- | ---: | ---: |',
    ]
    for candidate in report.candidates:
        lines.append(
            f'| `{candidate.name}` | {len(candidate.over_selected):,} | '
            f'{len(candidate.recovered):,} |'
        )
    lines += [
        '',
        'Simulating over the live corpus can only ever move a fact the regex '
        'ALREADY matched, so when the regex matches nothing this table is '
        'necessarily all zeroes — that is the finding, not an omission. The '
        'independent corroboration runs against the sweep suite\'s full pinned '
        'precision parametrization, mechanically, in:',
        '',
        f'    {report.revalidation_test}',
        '',
        'That gate reads the pinned shapes off the sweep suite\'s own '
        '`parametrize` markers, so a shape added upstream re-validates both '
        'candidates automatically.',
        '',
        '## Enumeration coverage',
        '',
        'This probe pages with `SKIP`/`LIMIT` instead of calling '
        '`GraphitiBackend.get_all_valid_edges`. That is deliberate: '
        "FalkorDB's server-wide `RESULTSET_SIZE` is 10000 and nothing in this "
        'repo overrides it, so `get_all_valid_edges` — which issues one '
        'unpaginated query — is silently truncated with no error. Measured on '
        'the live `dark_factory` graph: its exact query has 24902 rows and '
        'returns 10000, exposing 6376 of 12488 distinct valid edges (51%). '
        'Measuring recall through a truncated enumerator would produce a zero '
        'that means nothing. The truncation is a real latent coverage bug in '
        'the production sweep, but it lives in `graphiti_client.py`, outside '
        "task 3949's scope — filed as a follow-up rather than fixed here.",
        '',
        '## VERDICT',
        '',
        report.verdict,
        '',
    ]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

class _AppendReplacingDefault(argparse.Action):
    """``append`` that DISCARDS the default list on first use.

    Plain ``action='append'`` with a list default extends it, so a single
    ``--project-id reify`` would silently measure dark_factory too.
    (Same shape as census_memory_metadata.py.)
    """

    def __call__(self, parser, namespace, values, option_string=None):
        current = getattr(namespace, self.dest, None)
        if current is self.default or current is None:
            current = []
            setattr(namespace, self.dest, current)
        current.append(values)


def _page_size_arg(value: str) -> int:
    """``--page-size`` values that could not yield a provable enumeration.

    argparse turns an ArgumentTypeError into ``parser.error()``, so the
    operator is told at the CLI rather than discovering it as a fail-closed
    exit after the run. The identical check inside
    ``enumerate_valid_edge_facts`` stays regardless: it is the one that is
    directly testable and the one that protects direct callers.
    """
    try:
        size = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{value!r} is not an integer') from None
    if size < 1:
        raise argparse.ArgumentTypeError(f'must be >= 1, got {size}')
    if size >= RESULTSET_SIZE:
        raise argparse.ArgumentTypeError(
            f'{size} is at or above the assumed server result-set cap '
            f'(RESULTSET_SIZE={RESULTSET_SIZE}); a short page would then be '
            f'indistinguishable from a server-truncated one and the '
            f'enumeration could not be proven complete. Use a smaller value.'
        )
    return size


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--project-id', dest='project_id', action=_AppendReplacingDefault,
        default=list(DEFAULT_PROJECT_IDS),
        help=(
            'Project graph to measure; repeatable. '
            f'Default: {" ".join(DEFAULT_PROJECT_IDS)}'
        ),
    )
    parser.add_argument(
        '--page-size', dest='page_size', type=_page_size_arg,
        default=DEFAULT_PAGE_SIZE,
        help=(
            f'SKIP/LIMIT page size (default: {DEFAULT_PAGE_SIZE}). Must be '
            f"well under FalkorDB's RESULTSET_SIZE ({RESULTSET_SIZE}); "
            f'values at or above it are rejected — see '
            f'enumerate_valid_edge_facts.'
        ),
    )
    parser.add_argument(
        '--max-samples', dest='max_samples', type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f'Max rejection samples per project in the MARKDOWN report '
             f'(default: {DEFAULT_MAX_SAMPLES}). The JSON is never truncated.',
    )
    parser.add_argument(
        '--json-out', dest='json_out', default=DEFAULT_JSON_OUT,
        help=f'JSON artifact path (default: {DEFAULT_JSON_OUT})',
    )
    parser.add_argument(
        '--md-out', dest='md_out', default=DEFAULT_MD_OUT,
        help=f'Markdown artifact path (default: {DEFAULT_MD_OUT})',
    )
    parser.add_argument(
        '--config', dest='config', default=None,
        help='Optional CONFIG_PATH override for the live backend.',
    )
    parser.add_argument(
        '--measured-at', dest='measured_at', default=None,
        help=(
            'Timestamp recorded in the artifacts. Injected rather than read '
            'inside the renderers so rendering is deterministic and two '
            'identical runs diff cleanly.'
        ),
    )
    return parser


def _build_live_edge_source(config: Any) -> Any:
    """Read-only edge source over the live Graphiti graphs.

    Deliberately NOT ``MemoryService(config)`` + ``initialize()``: that path
    unconditionally runs the W6-epsilon startup identity scan and therefore
    WRITES. A probe must not mutate the corpus it measures. Constructing
    ``GraphitiBackend`` directly with ``skip_maintenance=True`` and going
    straight to ``ro_query`` is the read-only idiom migrate_cross_graph_leak.py
    and invalidate_fabricated_shipping_edges.py already established.
    """
    from fused_memory.backends.graphiti_client import GraphitiBackend  # noqa: PLC0415

    backend = GraphitiBackend(config)
    initialized = False

    async def edge_source(project_id: str, *, page_size: int):
        nonlocal initialized
        if not initialized:
            await backend.initialize(skip_maintenance=True)
            initialized = True
        graph = backend._graph_for(project_id)  # noqa: SLF001

        async def query_fn(cypher: str):
            result = await graph.ro_query(cypher)
            return result.result_set or []

        return await enumerate_valid_edge_facts(query_fn, page_size=page_size)

    edge_source.backend = backend  # type: ignore[attr-defined]
    return edge_source


def _write_artifacts(report: Report, json_out: str, md_out: str) -> None:
    json_path = Path(json_out)
    md_path = Path(md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(render_json(report))
    md_path.write_text(render_markdown(report))


async def _main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )
    args = _build_parser().parse_args(argv)

    import os  # noqa: PLC0415
    from datetime import UTC, datetime  # noqa: PLC0415

    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415

    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)
    if not args.measured_at:
        # Stamped ONCE here, at the CLI edge, and threaded through as data —
        # never read from the clock inside a renderer.
        args.measured_at = datetime.now(UTC).isoformat()

    edge_source = _build_live_edge_source(FusedMemoryConfig())
    try:
        report = await run(args, edge_source=edge_source)
        _write_artifacts(report, args.json_out, args.md_out)
        logger.info(
            'measured edges=%d matched=%d rejected=%d complete=%s json=%s md=%s',
            report.total_valid_edges, report.totals.regex_matched,
            report.totals.guard_rejected, report.complete,
            args.json_out, args.md_out,
        )
        if not report.complete:
            for project in report.projects:
                if not project.complete:
                    logger.error(
                        'COVERAGE SHORTFALL: project=%s enumeration incomplete',
                        project.project_id,
                    )
        return exit_code(report)
    finally:
        close = getattr(edge_source.backend, 'close', None)
        if close is not None:
            await close()


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(argv))


if __name__ == '__main__':
    sys.exit(main())
