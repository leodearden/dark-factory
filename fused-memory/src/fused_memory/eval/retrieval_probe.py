"""Pure retrieval-health metric math for E1: what a probe's results say about a topic.

``docs/prds/memory-eval-program.md`` §5 leaf β. Every function here takes an
already-fetched result list, or the observations built from such lists, and
returns a record; callers fetch the results and this module never touches a
store. That is what lets the merge lane test the whole metric family with no
Qdrant, no embedder and no OPENAI_API_KEY. What a probed topic is lives in
:mod:`fused_memory.eval.topic_registry`.

**What it measures** (seven metrics, all owned by leaf β):

======================================  ==========  ==================
metric_id                               kind        direction
======================================  ==========  ==================
``topic-canonical-present``             tripwire    (rule (a) is already
                                                    directional)
``canonical-in-top-5``                  proportion  lower_is_worse
``canonical-in-top-10``                 proportion  lower_is_worse
``canonical-in-top-5-held-out``         proportion  lower_is_worse
``claim-recall``                        proportion  lower_is_worse
``contamination-share``                 proportion  higher_is_worse
``superseded-above-successor``          count       higher_is_worse
======================================  ==========  ==================

``dangling-pointers`` and ``successor-pointer-present`` belong to leaf γ (E4)
and are deliberately NOT emitted here.

**Never evaluates a limit.** Per D1 the first run is a baseline snapshot, not
a day-one alarm source, and per G6/M2 every threshold, tolerance,
grandfather set and alarm lives in leaf α's limits evaluator. No pass rate,
bound or verdict appears in this module, in the registry fixture, or in any
of its tests. The only numeric parameter is the depth ``k`` (a caller passes
several as ``ks``, :data:`DEFAULT_KS` being 5 and 10), which is a metric
*parameterisation* — "is the canonical in this list of five" is set
membership, not a tuned bound.

**Schema home (D2).** :func:`build_series` assembles the M1 artifact from the
models, path helper and validator of :mod:`shared.memory_eval_metrics`; none
of that shape is re-declared here.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

from shared.memory_eval_metrics import (
    SCHEMA_VERSION,
    Corpus,
    Metric,
    MetricSeries,
    TripwireItem,
    report_artifact_path,
    validate_metric_series,
)

from fused_memory.eval.topic_registry import (
    TRIPWIRE_ITEM_PREFIX,
    Canonical,
    ClaimQuery,
    Phrasing,
    RegistryEntry,
    TopicRegistry,
    normalize_topic,
)

# ---------------------------------------------------------------------------
# Pinned M1 contract vocabulary
#
# These spellings are COPIED from leaf α's committed consumer-side exemplars
# (shared/tests/fixtures/memory_eval/e1-retrieval-health/metrics-*.json), not
# chosen here. The limits evaluator joins a run to its baseline window BY
# metric_id (shared/src/shared/memory_eval_metrics.py:262-275), so a different
# spelling would not fail loudly — it would make the metric invisible to the
# evaluator and to the dashboard, which is strictly worse than a crash.
# ---------------------------------------------------------------------------

EVAL_ID = 'e1-retrieval-health'
"""Also the artifact directory segment (``<root>/<eval_id>/metrics-<STAMP>.json``)."""

METRIC_TOPIC_CANONICAL_PRESENT = 'topic-canonical-present'
"""Tripwire (M2 rule a). One item per registry topic, keyed ``t-<topic-slug>``."""

METRIC_CANONICAL_IN_TOP_K = 'canonical-in-top-{k}'
"""Proportion over (topic, phrasing) pairs, formatted per ``k``."""

METRIC_CANONICAL_IN_TOP_K_HELD_OUT = 'canonical-in-top-{k}-held-out'
"""The Goodhart guard made visible — held-out phrasings, trended separately."""

METRIC_CLAIM_RECALL = 'claim-recall'
"""Proportion over (topic, claim-query) pairs."""

METRIC_CONTAMINATION_SHARE = 'contamination-share'
"""Proportion of scored results carrying a DIFFERENT registry topic."""

METRIC_SUPERSEDED_ABOVE_SUCCESSOR = 'superseded-above-successor'
"""Count of ranking inversions over registry-recorded (superseded, successor) pairs."""

TRIPWIRE_K = 5
"""The ``k`` the per-topic tripwire predicate is evaluated at.

Matches the exemplar's ``canonical-in-top-5``. Not a threshold: the predicate
is "is the canonical in this list", and the list has to have some length.
"""

DEFAULT_KS: tuple[int, ...] = (5, 10)
"""Default ``ks``: the depths scored when a caller names none. A parameterisation, not a limit (see module docstring)."""

_WHITESPACE_RE = re.compile(r'\s+')


# ---------------------------------------------------------------------------
# Content identity
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """*text* with internal whitespace runs collapsed to one space and its ends stripped.

    The one whitespace rule every content comparison here shares: the
    content identity of :func:`content_key`, needle matching, and the
    registry deriver's human-readable ``content_prefix``.
    """
    return _WHITESPACE_RE.sub(' ', text).strip()


def content_key(text: str) -> str:
    """A stable 16-hex-char identity for a memory's *text*.

    ``sha256(...).hexdigest()[:16]`` — the ``shared/task_metadata.py:310``
    convention, so the digest length matches every other content signature in
    this repo.

    Whitespace is normalised (surrounding stripped, internal runs collapsed to
    one space) BEFORE hashing so that re-indentation, a wrapped line or a
    trailing newline picked up in transit does not read as a different entry.
    Nothing else is normalised: case and punctuation are content, and folding
    them would make two genuinely different claims collide.
    """
    normalized = normalize_whitespace(text)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Canonical-in-top-k — pure over an already-fetched result list
#
# Every function below takes a list of results and returns a record. Nothing
# here touches a store, which is what lets the merge lane test the whole metric
# family with no Qdrant, no embedder and no OPENAI_API_KEY.
# ---------------------------------------------------------------------------

MATCHED_BY_CONTENT_HASH = 'content_hash'
MATCHED_BY_LAST_KNOWN_ID = 'last_known_id'


@dataclass(frozen=True)
class MatchOutcome:
    """Where (and how) a topic's canonical was found in one result list."""

    hit: bool
    rank: int | None
    matched_by: str | None

    @property
    def unmatched(self) -> bool:
        """Neither matcher fired — the entry was not in the list at all.

        Distinct from ``not hit``: an entry found at rank k+1 is present but
        ranked too low, which is a ranking problem. An unmatched entry is
        either absent from the corpus or the fixture has decayed past both
        keys. The report separates the two rather than reporting one number.
        """
        return self.matched_by is None

    @property
    def needs_hash_repair(self) -> bool:
        """The id matched but the hash did not — the fixture needs re-hashing.

        Reported rather than repaired: a probe that silently rewrote its own
        expectations to match what it found could never fail.
        """
        return self.matched_by == MATCHED_BY_LAST_KNOWN_ID


@dataclass(frozen=True)
class PhrasingObservation:
    """One (topic, phrasing, k) probe result."""

    topic: str
    phrasing: str
    held_out: bool
    k: int
    hit: bool
    rank: int | None
    matched_by: str | None
    stores_served: tuple[str, ...] = ()
    """The distinct stores that answered, read off the scored top-k slice.

    ``MemoryService.search`` ROUTES: the read router picks a store set per
    query, and the lists come back homogeneous. A phrasing served entirely by
    Graphiti cannot contain a Mem0 entry's raw content — Graphiti returns
    LLM-extracted edge facts — so its canonical is unfindable there however
    healthy retrieval is.

    The probe pins stores ONLY for an entry that declares the scope its real
    caller searches with (:attr:`RegistryEntry.search_scope`; today the
    briefing channels). Every other entry is left to the router: an agent's
    own search is routed too, so "the router sent this query somewhere the
    canonical does not live" is a real fact about what an agent experiences.
    But a rate dominated by
    routing that does not SAY so is a silent fail-soft — the limits evaluator
    would compute bounds over router coin-flips — so the served set rides
    along with every observation, into the report AND into the artifact.
    """
    degraded: bool = False
    """True when the search that produced this list reported a failed store.

    A degraded observation is EXCLUDED from every metric denominator: charging
    a store outage as a canonical-absent failure would manufacture a
    corpus-wide findability collapse out of an infrastructure blip.
    """

    @property
    def unmatched(self) -> bool:
        return self.matched_by is None

    @property
    def needs_hash_repair(self) -> bool:
        return self.matched_by == MATCHED_BY_LAST_KNOWN_ID


def _result_content(result: Any) -> str:
    return getattr(result, 'content', '') or ''


def _result_id(result: Any) -> str:
    return getattr(result, 'id', '') or ''


UNKNOWN_STORE = 'unknown'
"""Recorded for a result carrying no ``source_store``.

Dropping it would understate the served set for exactly the shapes this probe
does not recognise — the ones most worth knowing about.
"""


def stores_served(results: list, k: int) -> tuple[str, ...]:
    """The distinct stores that answered, over the top-*k* slice only.

    Sorted so the value is stable across runs: it is compared between runs and
    an incidental ordering change would read as a routing change.
    """
    seen = {
        str(getattr(result, 'source_store', '') or UNKNOWN_STORE)
        for result in results[:k]
    }
    return tuple(sorted(seen))


def rank_index(results: list) -> dict[str, int]:
    """``{content_hash: first rank}`` over the WHOLE of *results*, hashed once.

    One search's list is read by three separate metric families — canonical
    presence at every ``k``, superseded inversions, and the comparable-pair
    exposure that inversion count is per — and each of them wants the same
    question answered: at what rank did this content hash come back. Computing
    it here means the list is sha256'd once per query rather than once per
    consumer.

    ``setdefault`` keeps the FIRST rank: a store that returned the same content
    twice has still returned it at its best rank, and taking the later one
    would report a ranking problem that is really a duplication one.
    """
    first_rank: dict[str, int] = {}
    for index, result in enumerate(results, start=1):
        first_rank.setdefault(content_key(_result_content(result)), index)
    return first_rank


def ranks_at_depth(ranks: dict[str, int], k: int) -> dict[str, int]:
    """The depth-scoped view of a full-depth :func:`rank_index` mapping.

    Exactly equal to ``rank_index(results[:k])``: :func:`rank_index` keeps
    the FIRST rank a content hash appeared at, so a hash whose first rank is
    ``<= k`` has that same first rank in the truncated list, and a hash
    whose first rank is ``> k`` does not appear in the truncated list at
    all. Filtering the existing mapping therefore reproduces re-hashing the
    truncation exactly, without re-deriving it.

    Deriving rather than re-hashing preserves :func:`rank_index`'s own
    contract: the list is sha256'd once per query rather than once per
    consumer. A second, truncated re-hash would reintroduce the very second
    pass that contract exists to avoid.

    Returns a NEW mapping — the full-depth index the canonical-presence
    family reads (``canonical_hit`` / ``observe_phrasing``, which deliberately
    scan the whole list so a hit beyond ``k`` is reported at its true rank
    rather than as absent) is left untouched.
    """
    return {content_hash: rank for content_hash, rank in ranks.items() if rank <= k}


def canonical_hit(
    results: list,
    canonical: Canonical,
    k: int,
    *,
    ranks: dict[str, int] | None = None,
) -> MatchOutcome:
    """Find *canonical* in *results*, honouring the top-*k* cut.

    Content hash first, ``last_known_id`` second (D5): memory UUIDs rot on
    re-consolidation, so hashing the returned content is the durable key, while
    the id fallback keeps a benignly-reworded canonical from reading as a
    findability regression. Which matcher fired is recorded either way, because
    both divergences are a fixture-repair signal an operator needs to see.

    The whole list is scanned, not just its first *k*, so a canonical that came
    back at rank k+3 is reported at its true rank instead of as "absent" — a
    ranking problem and an absence problem need different fixes.

    *ranks* is :func:`rank_index` of the same list, threaded in by a caller
    that already built it (a caller scoring several depths calls this once per
    ``k``). Optional
    rather than required so every direct caller and test keeps the one-argument
    shape; computed here when absent, and the answer is identical either way.
    """
    if ranks is None:
        ranks = rank_index(results)
    hash_rank = ranks.get(canonical.content_hash)
    id_rank: int | None = None

    # Only when the primary matcher missed: the id is the FALLBACK, and its
    # rank is never read when the hash has one.
    if hash_rank is None and canonical.last_known_id:
        for index, result in enumerate(results, start=1):
            if _result_id(result) == canonical.last_known_id:
                id_rank = index
                break

    if hash_rank is not None:
        return MatchOutcome(
            hit=hash_rank <= k, rank=hash_rank, matched_by=MATCHED_BY_CONTENT_HASH,
        )
    if id_rank is not None:
        return MatchOutcome(
            hit=id_rank <= k, rank=id_rank, matched_by=MATCHED_BY_LAST_KNOWN_ID,
        )
    return MatchOutcome(hit=False, rank=None, matched_by=None)


def observe_phrasing(
    results: list,
    entry: RegistryEntry,
    phrasing: Phrasing,
    k: int,
    *,
    degraded: bool = False,
    ranks: dict[str, int] | None = None,
) -> PhrasingObservation:
    """Build the (topic, phrasing, k) observation the metrics aggregate over.

    *ranks* is passed straight through to :func:`canonical_hit` — see there.
    """
    outcome = canonical_hit(results, entry.canonical, k, ranks=ranks)
    return PhrasingObservation(
        topic=entry.topic,
        phrasing=phrasing.text,
        held_out=phrasing.held_out,
        k=k,
        hit=outcome.hit,
        rank=outcome.rank,
        matched_by=outcome.matched_by,
        stores_served=stores_served(results, k),
        degraded=degraded,
    )


# ---------------------------------------------------------------------------
# superseded-above-successor — a pure ranking comparison
#
# Reads ONLY registry-recorded hash pairs. metadata['supersedes'] is never
# touched: that shape zoo is normalize_supersedes()' problem (task 3196, leaf
# gamma's hard dependency), and a second parser here would be exactly the
# lockstep duplication INV-5 forbids. Because the relation was recorded offline
# at derivation time, this function needs no pointer knowledge at all.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InversionRecord:
    """A superseded entry that outranked the entry which replaced it."""

    topic: str
    phrasing: str
    superseded_hash: str
    successor_hash: str
    superseded_rank: int
    successor_rank: int


def comparable_pairs(
    entry: RegistryEntry, ranks: dict[str, int] | None = None, results: list | None = None,
) -> int:
    """Registry pairs with BOTH members in the returned list — the real exposure.

    This is the count metric's ``n``, and it is deliberately not
    ``len(entry.supersedes_pairs)``. :func:`superseded_inversions` can only
    ever fire on a both-present pair, so a pair with one member missing
    contributes no possibility of an event. Charging it to the denominator
    anyway makes the rate move for the wrong reason: if retrieval improves so
    that 40 pairs come back both-present instead of 4, the event count can rise
    while a registered-pair ``n`` stays pinned — and leaf α's Poisson tail test
    reads a retrieval IMPROVEMENT as a rate regression.

    Pass *ranks* (:func:`rank_index` of the same list) when the caller already
    has it; *results* is the fallback for a direct caller that does not.
    """
    if ranks is None:
        ranks = rank_index(results or [])
    return sum(
        1 for pair in entry.supersedes_pairs
        if pair.superseded_hash in ranks and pair.successor_hash in ranks
    )


def superseded_inversions(
    results: list,
    entry: RegistryEntry,
    *,
    phrasing: str = '',
    ranks: dict[str, int] | None = None,
) -> list[InversionRecord]:
    """Registry-recorded pairs where the superseded entry outranks its successor.

    Both-present-only: a pair with just one member in *results* yields nothing.
    An absent successor is a findability question ``canonical-in-top-k`` already
    measures, and counting it here as well would charge one defect against two
    metrics — inflating any downstream trend by double-weighting a single fix.
    :func:`comparable_pairs` counts the same both-present population, which is
    why it — and not the registered-pair count — is the metric's exposure.

    Rank is the position in the list the store returned, so equal
    ``relevance_score`` values resolve by that order rather than by an unstable
    re-sort. Two runs over the same list therefore produce the same count; a
    tie that flapped would look to the evaluator like a real regression.

    Records name both hashes and both ranks: a bare count tells an operator
    that something inverted but not which pair to go and look at.
    """
    if not entry.supersedes_pairs:
        return []

    first_rank = rank_index(results) if ranks is None else ranks

    inversions: list[InversionRecord] = []
    for pair in entry.supersedes_pairs:
        superseded_rank = first_rank.get(pair.superseded_hash)
        successor_rank = first_rank.get(pair.successor_hash)
        if superseded_rank is None or successor_rank is None:
            continue
        if superseded_rank < successor_rank:
            inversions.append(InversionRecord(
                topic=entry.topic,
                phrasing=phrasing,
                superseded_hash=pair.superseded_hash,
                successor_hash=pair.successor_hash,
                superseded_rank=superseded_rank,
                successor_rank=successor_rank,
            ))
    return inversions


# ---------------------------------------------------------------------------
# claim-recall — did the claim come back at all?
#
# Deliberately WEAKER than canonical identity. ``canonical-in-top-k`` asks
# whether one specific entry is findable; this asks whether the KNOWLEDGE is,
# from any entry. The distinction is the whole point: consolidation (3111/3112)
# rewrites and merges entries by design, so a metric that demanded the claim
# come back from the same entry would score the fix lineage this eval exists to
# measure as a regression, and the obvious way to make that metric go green
# would be to stop consolidating.
# ---------------------------------------------------------------------------

def _normalize_for_needles(text: str) -> str:
    """Fold *text* for needle matching: whitespace collapsed, then lowercased.

    The whitespace half is exactly :func:`content_key`'s normalisation, so a
    re-wrapped or re-indented stored line does not read as knowledge loss.

    Case folding is the one deliberate DIVERGENCE from ``content_key``. There,
    case is content — two claims differing only in case are two claims, and
    folding them would collide distinct entries under one hash. Here the needle
    is a substring probe against prose, where casing is presentation churn:
    the committed registry carries needles like ``WRITE-SET`` that the corpus
    spells ``write-set``, and a case-sensitive miss would report a knowledge
    loss that did not happen.
    """
    return normalize_whitespace(text).lower()


@dataclass(frozen=True)
class ClaimOutcome:
    """Whether one claim came back, and — when it did not — what was missing."""

    recalled: bool
    missing_needles: tuple[str, ...]
    matched_rank: int | None
    scorable: bool = True
    """False when the claim query carries no needles.

    ``all(needle in text for needle in ())`` is vacuously True, so a needle-less
    claim query would silently score as recalled and inflate the rate for a
    malformed registry entry. Unscorable claims are excluded from the
    denominator and DISCLOSED instead — an unmeasurable claim must not be able
    to masquerade as a healthy one.
    """


def claim_recalled(results: list, claim_query: ClaimQuery, k: int) -> ClaimOutcome:
    """Did *claim_query*'s needles come back in the top-*k* of *results*?

    All needles are required, and they must all come from a SINGLE returned
    entry. Pooling needles across the result set would let an entry saying
    "the merge lane is strictly serial" and an unrelated entry saying "never
    rolls back" jointly satisfy a claim that neither one makes — recall of a
    sentence the corpus never stated.

    When no entry carries every needle, ``missing_needles`` is reported against
    the CLOSEST entry (most needles matched, ties broken by better rank), since
    diffing against the near-miss is what tells an operator which half of the
    claim was lost. With nothing matched at all the full needle list is
    returned, which is the honest answer rather than an empty one.
    """
    needles = tuple(claim_query.needles)
    if not needles:
        return ClaimOutcome(
            recalled=False, missing_needles=(), matched_rank=None, scorable=False,
        )

    normalized_needles = [(needle, _normalize_for_needles(needle)) for needle in needles]

    best_missing: tuple[str, ...] = needles
    best_found = -1
    for rank, result in enumerate(results[:k], start=1):
        haystack = _normalize_for_needles(_result_content(result))
        missing = tuple(
            original for original, folded in normalized_needles if folded not in haystack
        )
        if not missing:
            return ClaimOutcome(
                recalled=True, missing_needles=(), matched_rank=rank, scorable=True,
            )
        found = len(needles) - len(missing)
        if found > best_found:  # strict >: ties keep the better (earlier) rank
            best_found = found
            best_missing = missing

    return ClaimOutcome(
        recalled=False, missing_needles=best_missing, matched_rank=None, scorable=True,
    )


# ---------------------------------------------------------------------------
# contamination-share — how much of what came back belongs to another topic?
#
# This metric has to be WELL-POSED TODAY, before 3195/3201 widen the
# `metadata.topic` vocabulary. The committed census measured 491 of 49,628
# entries carrying a topic at all, so the overwhelming majority of results have
# none, and any definition that treated "no topic" as evidence of anything
# would be measuring stamping coverage while claiming to measure contamination.
#
# So: FOREIGN iff the result carries a topic that is IN the registry and is not
# the probed one. Everything else is UNTOPICED and disclosed alongside the
# count, never folded into the numerator. The definition is monotone in the
# registry — widening it (which is exactly what 3201's retro stamping enables)
# strictly widens the numerator's reach with no code change here.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForeignRecord:
    """One returned entry that belongs to a different registered topic."""

    topic: str
    """The topic being probed."""

    foreign_topic: str
    """The registered topic the returned entry actually carries.

    Named, not counted: "3 foreign results" tells an operator that something
    bled in, while "escalation-ladder bled into merge-lane" tells them where to
    look. A bare count of a cross-topic leak is not actionable.
    """

    rank: int
    result_id: str = ''


class ContaminationOutcome(NamedTuple):
    """``(foreign_records, foreign_count, untopiced_count, scored_total)``.

    A NamedTuple so the four fields both unpack positionally (the shape the
    caller's metric assembly wants) and read by name at the use site.

    ``untopiced_count`` and ``scored_total`` are not diagnostics — they are the
    honesty of the share. ``foreign_count / scored_total`` computed without
    them looks authoritative while, on today's corpus, most of the denominator
    is entries that could not have been classified either way.
    """

    foreign_records: tuple[ForeignRecord, ...]
    foreign_count: int
    untopiced_count: int
    scored_total: int


def _result_topic(result: Any) -> str | None:
    """The result's ``metadata['topic']``, or None when it has no usable one.

    Live metadata is not schema-enforced, so a topic that is missing, empty, or
    not a string is treated identically: unusable. Coercing a list-valued topic
    with ``str()`` would invent a topic value nobody wrote.
    """
    metadata = getattr(result, 'metadata', None)
    if not isinstance(metadata, dict):
        return None
    topic = metadata.get('topic')
    if not isinstance(topic, str) or not topic.strip():
        return None
    return topic


def classify_contamination(
    results: list,
    entry: RegistryEntry,
    registry: TopicRegistry,
    k: int,
) -> ContaminationOutcome:
    """Classify the top-*k* of *results* against the topic being probed.

    Foreignness is gated on registry membership in both directions: the
    returned entry's topic must be one the registry knows, AND it must not fold
    onto *entry*'s own slug. Comparison is via :func:`normalize_topic`, because
    the corpus spells the same topic with ``-`` and ``_`` (the guard cluster's
    ``architect-report-...`` against the census's
    ``architect_report_...``) — an exact comparison would report a topic as
    foreign to itself.

    Returns the foreign records, their count, the untopiced count and the
    scored total, so the share's denominator can never be reported without the
    disclosure of how much of it was unclassifiable.
    """
    probed = normalize_topic(entry.topic)
    known = registry.normalized_topics

    foreign: list[ForeignRecord] = []
    untopiced = 0
    scored = 0

    for rank, result in enumerate(results[:k], start=1):
        scored += 1
        raw_topic = _result_topic(result)
        if raw_topic is None:
            untopiced += 1
            continue
        folded = normalize_topic(raw_topic)
        if folded == probed:
            continue
        if folded not in known:
            # An unregistered topic is not evidence of contamination: the
            # census counted 352 distinct live topic values against the ~32
            # this registry adjudicates, so "unknown to us" is the common case
            # and would swamp the numerator with entries nobody has judged.
            untopiced += 1
            continue
        foreign.append(ForeignRecord(
            topic=entry.topic,
            foreign_topic=raw_topic,
            rank=rank,
            result_id=_result_id(result),
        ))

    return ContaminationOutcome(
        foreign_records=tuple(foreign),
        foreign_count=len(foreign),
        untopiced_count=untopiced,
        scored_total=scored,
    )


# ---------------------------------------------------------------------------
# M1 series assembly
#
# Everything above produces per-probe records; this band turns them into the
# ONE artifact leaf alpha's evaluator and the dashboard read. Every model, path
# helper and validator comes from shared.memory_eval_metrics (D2) — none of
# that shape is re-declared here.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClaimObservation:
    """One (topic, claim-query, k) probe result."""

    topic: str
    query: str
    k: int
    recalled: bool
    missing_needles: tuple[str, ...] = ()
    scorable: bool = True
    degraded: bool = False


@dataclass(frozen=True)
class ContaminationObservation:
    """One (topic, phrasing, k) contamination sample, disclosures included."""

    topic: str
    phrasing: str
    k: int
    foreign_count: int
    untopiced_count: int
    scored_total: int
    foreign_records: tuple[ForeignRecord, ...] = ()
    degraded: bool = False


@dataclass(frozen=True)
class InversionObservation:
    """The inversions found in one (topic, phrasing) probe, plus its exposure.

    TWO exposure numbers, deliberately. A Poisson tail test needs the exposure
    the rate is per — 2 inversions out of 4 pairs and 2 out of 400 are not the
    same observation — but it needs the exposure that could actually have
    produced an event.

    ``pairs_registered`` is every pair the registry recorded for this topic.
    ``pairs_comparable`` is the subset with BOTH members in the returned list,
    which is the only population :func:`superseded_inversions` can fire on, and
    therefore the one that becomes the metric's ``n``. Using the registered
    count instead makes a retrieval IMPROVEMENT (more pairs coming back
    both-present, so more events observable) read to the evaluator as a rate
    regression against an ``n`` that never moved.

    The registered count rides along because the ratio between the two is
    itself the signal: comparable ≪ registered means most pairs are not being
    returned at all, which is a findability fact ``canonical-in-top-k``
    measures and this metric must not be read as if it had.

    ``pairs_beyond_scored_depth`` is a THIRD number, diagnostic rather than an
    exposure: registry pairs both-present in the full FETCHED list but not
    both within the scored depth — the population the scored-depth pin
    removed from ``pairs_comparable``. It is never summed into the metric or
    published as a second ``n``; it exists only so a run where the pin
    genuinely trimmed exposure is distinguishable, in the artifact, from one
    with nothing beyond the scored depth to trim.

    ``k`` is the depth this observation was SCORED at, not the depth fetched
    — mirroring :class:`ContaminationObservation` and :class:`ClaimObservation`.
    It is required rather than defaulted so no construction site can silently
    claim the tripwire's depth for an observation actually built at another
    one; every existing construction site passes it explicitly.
    """

    topic: str
    phrasing: str
    k: int
    pairs_registered: int
    pairs_comparable: int
    pairs_beyond_scored_depth: int
    inversions: tuple[InversionRecord, ...] = ()
    degraded: bool = False


@dataclass(frozen=True)
class DegradedQuery:
    """One query whose search reported a failed store.

    Carries the diagnostics verbatim rather than a boolean: "the run was
    degraded" tells an operator to distrust the numbers, while "mem0 raised
    TimeoutError" tells them what to restart.
    """

    topic: str
    query: str
    failed_stores: tuple[str, ...]
    diagnostics: tuple[dict, ...] = ()


@dataclass
class ProbeObservations:
    """Everything one probe run measured, before aggregation.

    Mutable and append-only during a run; :func:`build_series` reads it without
    modifying it. Keeping the raw records around (rather than accumulating
    counters) is what lets the report name WHICH topic, phrasing and hash
    failed instead of only how many did.
    """

    phrasings: list[PhrasingObservation] = field(default_factory=list)
    claims: list[ClaimObservation] = field(default_factory=list)
    contamination: list[ContaminationObservation] = field(default_factory=list)
    inversions: list[InversionObservation] = field(default_factory=list)
    degraded_queries: list[DegradedQuery] = field(default_factory=list)


def _proportion(
    metric_id: str,
    successes: int,
    trials: int,
    direction: Literal['higher_is_worse', 'lower_is_worse'],
    *,
    details_path: str | None = None,
) -> Metric | None:
    """A proportion Metric, or None when nothing was scored.

    ``None`` rather than a 0/0 metric: the shared validator rejects a
    non-positive denominator, and rightly — a proportion over zero trials is
    not a measurement of anything, and emitting one would put a fabricated
    trial into the baseline window the evaluator computes limits from. An
    absent metric is the honest signal, and
    :func:`metric_families_not_measured` puts the absence in the report (a
    metric that vanishes without explanation reads as healthy).
    """
    if trials <= 0:
        return None
    return Metric(
        metric_id=metric_id,
        kind='proportion',
        value=successes / trials,
        n=trials,
        denominator=trials,
        direction=direction,
        details_path=details_path,
    )


def tripwire_items(observations: ProbeObservations, k: int) -> list[tuple[str, bool]]:
    """(item_key, passed) per topic measured at *k*, in stable slug order.

    A topic passes only when EVERY non-degraded phrasing at *k* — including the
    held-out one — found its canonical. That conjunction is the Goodhart guard
    expressed as a predicate: tuning the known phrasings until they pass cannot
    make the topic pass while the freshly authored phrasing still misses.

    A topic whose every phrasing degraded contributes NO item at all rather
    than a failing one. A store outage must not be able to manufacture a
    corpus-wide canonical-findability collapse for leaf epsilon to file against
    the 3111 lineage.
    """
    passed: dict[str, bool] = {}
    for obs in observations.phrasings:
        if obs.degraded or obs.k != k:
            continue
        passed[obs.topic] = passed.get(obs.topic, True) and obs.hit
    return [(f'{TRIPWIRE_ITEM_PREFIX}{topic}', ok) for topic, ok in sorted(passed.items())]


def _disclosure_counts(observations: ProbeObservations) -> dict[str, int]:
    """The narrowings that must ride along INSIDE the machine-readable artifact.

    Reporting them only in prose would be a silent cap for every consumer that
    reads the JSON — which is all of them. ``corpus.counts`` is free-form
    category -> size by design (its docstring: the bucket vocabulary is not
    this schema's to own), so it is where a per-run disclosure belongs.

    **Every per-observation key names its depth.** Each query produces one
    observation per ``k``, so on the default ``(5, 10)`` run an unsuffixed
    tally counted every query twice — against a ``canonical-in-top-5`` whose
    ``n`` counts it once. Worse, ``stores_served`` is computed over
    ``results[:k]``, so a store appearing only at ranks 6-10 was credited in
    the k=10 tally and not the k=5 one: the two numbers under one key came
    from different slice depths. A consumer comparing them to a rate's
    denominator got a 2x mismatch with nothing in the artifact saying why,
    which is the uninterpretable-rate hazard this disclosure exists to
    prevent. Suffixing makes each number comparable to exactly one metric.
    """
    counts = {
        'contamination_scored_results': sum(
            c.scored_total for c in observations.contamination if not c.degraded
        ),
        'contamination_untopiced_results': sum(
            c.untopiced_count for c in observations.contamination if not c.degraded
        ),
        'claim_queries_unscorable': sum(
            1 for c in observations.claims if not c.degraded and not c.scorable
        ),
        # Per QUERY, not per observation: a degraded search is one event
        # whatever depth it is later scored at, so this counts the
        # `ProbeObservations.degraded_queries` records one for one.
        'degraded_queries': len(observations.degraded_queries),
    }
    # Which store answered, and how many observations degraded, at each depth.
    # An observation served by a store the canonical does not live in cannot
    # hit however healthy retrieval is, so a canonical-in-top-k rate is
    # uninterpretable without this — and a consumer reading only the JSON
    # would never see it said in prose.
    for observation in observations.phrasings:
        if observation.degraded:
            key = f'degraded_observations_at_k{observation.k}'
            counts[key] = counts.get(key, 0) + 1
            continue
        for store in observation.stores_served:
            key = f'observations_served_by_{store}_at_k{observation.k}'
            counts[key] = counts.get(key, 0) + 1
    # The inversion family's own depth-named disclosure. `superseded-above-
    # successor` carries no depth suffix in its metric_id and leaf alpha joins
    # by metric_id alone, so without a row naming the depth a change of `ks` reads
    # to the evaluator as a rate move rather than a re-parameterisation.
    # `pairs_registered` is published because it is the un-narrowed population
    # `pairs_comparable` is a subset of; the two together separate "fewer
    # pairs are coming back" from "the registry recorded fewer pairs". There
    # is deliberately NO `inversion_pairs_comparable` row: leaf beta settled
    # this exact question for the identically-shaped `surfacing` family
    # (fused-memory/scripts/memory_eval_staleness_sweep.py::_disclosure_counts,
    # refusing a `surfacing_pairs_observed` row) — it would be a second name
    # for the number already published as the metric's `n`, and when exposure
    # is zero the metric is absent and `metric_families_not_measured` names
    # the gap, which a bare 0 here would have disguised as a measurement.
    # `pairs_beyond_scored_depth` is NOT that same refusal reversed: it is
    # the population the pin REMOVED (both-present in the full fetch, not
    # both within the scored depth), a number `n` never carries at all.
    # Without it a corpus where superseded entries reliably outrank their
    # successors just past the scored depth is indistinguishable, in this
    # artifact, from one with no such pairs anywhere in reach.
    for observation in observations.inversions:
        if observation.degraded:
            key = f'degraded_inversion_observations_at_k{observation.k}'
            counts[key] = counts.get(key, 0) + 1
            continue
        key = f'inversion_observations_at_k{observation.k}'
        counts[key] = counts.get(key, 0) + 1
        key = f'inversion_pairs_registered_at_k{observation.k}'
        counts[key] = counts.get(key, 0) + observation.pairs_registered
        key = f'inversion_pairs_beyond_scored_depth_at_k{observation.k}'
        counts[key] = counts.get(key, 0) + observation.pairs_beyond_scored_depth
    return counts


def build_series(
    observations: ProbeObservations,
    corpus_counts: dict[str, int],
    project_id: str,
    stamp: str,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> MetricSeries:
    """Assemble the M1 metric series for one probe run.

    Emits at most the seven metrics this leaf owns, in the pinned vocabulary.
    ``dangling-pointers`` / ``successor-pointer-present`` are leaf γ's (E4) and
    never appear here.

    The result is validated before it is returned, so an aggregation bug
    surfaces in the producer rather than in leaf α's evaluator — the M1
    "rejected at emit time, not read time" guarantee applied to the producer's
    own arithmetic.
    """
    # The report filename, not its absolute path: the artifact directory gets
    # copied and served (the dashboard reads it as plain files), and an
    # absolute path from this machine would be a dangling pointer there.
    details_path = report_artifact_path('.', EVAL_ID, stamp).name

    metrics: list[Metric] = []

    items = tripwire_items(observations, TRIPWIRE_K)
    if items:
        metrics.append(Metric(
            metric_id=METRIC_TOPIC_CANONICAL_PRESENT,
            kind='tripwire',
            value=float(sum(1 for _, ok in items if not ok)),
            n=len(items),
            items=[TripwireItem(item_key=key, passed=ok) for key, ok in items],
            details_path=details_path,
        ))

    scored_phrasings = [o for o in observations.phrasings if not o.degraded]
    for k in ks:
        at_k = [o for o in scored_phrasings if o.k == k]
        metric = _proportion(
            METRIC_CANONICAL_IN_TOP_K.format(k=k),
            sum(1 for o in at_k if o.hit),
            len(at_k),
            'lower_is_worse',
            details_path=details_path,
        )
        if metric is not None:
            metrics.append(metric)

    # Held-out only at the tripwire's k, deliberately: the two answer the same
    # question (is the canonical in this list of five) over different phrasing
    # populations, so trending them at one k keeps them comparable. Emitting a
    # held-out variant per k would also grow the metric vocabulary every time
    # a caller added another depth to `ks`.
    held_out = [o for o in scored_phrasings if o.k == TRIPWIRE_K and o.held_out]
    metric = _proportion(
        METRIC_CANONICAL_IN_TOP_K_HELD_OUT.format(k=TRIPWIRE_K),
        sum(1 for o in held_out if o.hit),
        len(held_out),
        'lower_is_worse',
        details_path=details_path,
    )
    if metric is not None:
        metrics.append(metric)

    claims = [c for c in observations.claims if not c.degraded and c.scorable]
    metric = _proportion(
        METRIC_CLAIM_RECALL,
        sum(1 for c in claims if c.recalled),
        len(claims),
        'lower_is_worse',
        details_path=details_path,
    )
    if metric is not None:
        metrics.append(metric)

    contamination = [c for c in observations.contamination if not c.degraded]
    metric = _proportion(
        METRIC_CONTAMINATION_SHARE,
        sum(c.foreign_count for c in contamination),
        sum(c.scored_total for c in contamination),
        'higher_is_worse',
        details_path=details_path,
    )
    if metric is not None:
        metrics.append(metric)

    # Exposure, not observation count. `if inversions:` emitted a value=0 / n=0
    # count metric for any selection whose entries record no supersedes pair —
    # a fabricated "no inversions here" datapoint entering leaf α's baseline
    # window. That is the same hazard `_proportion` refuses a 0/0 proportion
    # for, and a count kind has no more claim to a zero-trial measurement than
    # a proportion does. Absent is the honest signal, and the report names the
    # family so the absence cannot read as health.
    inversions = [i for i in observations.inversions if not i.degraded]
    exposure = sum(i.pairs_comparable for i in inversions)
    if exposure > 0:
        metrics.append(Metric(
            metric_id=METRIC_SUPERSEDED_ABOVE_SUCCESSOR,
            kind='count',
            value=float(sum(len(i.inversions) for i in inversions)),
            n=exposure,
            direction='higher_is_worse',
            details_path=details_path,
        ))

    counts = dict(corpus_counts)
    for key, value in _disclosure_counts(observations).items():
        if key in counts:
            raise ValueError(
                f'corpus_counts key {key!r} collides with a run disclosure this '
                'module computes. Rename the caller-supplied key: silently '
                'overwriting either one would hide a narrowing.'
            )
        counts[key] = value

    series = MetricSeries(
        schema_version=SCHEMA_VERSION,
        eval_id=EVAL_ID,
        run_stamp=stamp,
        corpus=Corpus(project_id=project_id, counts=counts),
        metrics=metrics,
    )
    validate_metric_series(series)
    return series


def normalise_ks(ks: tuple[int, ...]) -> tuple[int, ...]:
    """*ks* with :data:`TRIPWIRE_K` guaranteed present, caller's order kept.

    *ks* is a parameterisation, so ``ks=(7,)`` is a legitimate request — but
    two metrics are DEFINED at k=5 and cannot be computed without it: the
    ``topic-canonical-present`` tripwire and the held-out proportion. Without
    this, ``ks=(7,)`` emitted neither, and did so silently: leaf α's evaluator
    joins a run to its baseline window BY metric_id, so an absent metric is not
    an error there, it simply stops being trended. A metric that quietly stops
    existing is worse than one that crashes.

    Normalising rather than rejecting keeps ``ks=(7,)`` legal — the extra depth
    is genuinely useful — while making the pinned metric set a guaranteed
    superset of every run's, which is what keeps α's join keys stable. The
    added depth is disclosed in the report; it is not left to be inferred
    from the metric list.

    The dedup is order-preserving: a repeated ``k`` keeps its first position,
    and :data:`TRIPWIRE_K` is appended only when the caller did not ask for it.
    """
    return tuple(dict.fromkeys((*ks, TRIPWIRE_K)))


def pinned_metric_ids(ks: tuple[int, ...]) -> tuple[str, ...]:
    """Every metric_id a run at *ks* is expected to emit.

    THE list this leaf owns, in one place, so "which family went unmeasured"
    is answerable by comparing against it rather than by remembering what
    :func:`build_series` can emit. ``ks`` goes through :func:`normalise_ks`
    for the same reason the run does: the tripwire's depth is always measured.
    """
    return (
        METRIC_TOPIC_CANONICAL_PRESENT,
        *(METRIC_CANONICAL_IN_TOP_K.format(k=k) for k in normalise_ks(tuple(ks))),
        METRIC_CANONICAL_IN_TOP_K_HELD_OUT.format(k=TRIPWIRE_K),
        METRIC_CLAIM_RECALL,
        METRIC_CONTAMINATION_SHARE,
        METRIC_SUPERSEDED_ABOVE_SUCCESSOR,
    )


def metric_families_not_measured(series: MetricSeries, ks: tuple[int, ...]) -> list[str]:
    """Pinned metric ids *series* does not carry, in the pinned order.

    Every family in this module declines to emit rather than emit a
    zero-trial datapoint: ``_proportion`` refuses a 0/0 proportion, and the
    count metric refuses a zero-exposure count, because a fabricated trial in
    leaf α's baseline window is worse than a gap in it. But an absent metric
    is not an error to the evaluator either — it joins by metric_id and simply
    stops trending what is missing. So the absence is named HERE, in the run's
    own report, where the alternative is a metric that silently stops existing
    and reads to a human as one that had nothing to report.
    """
    present = {metric.metric_id for metric in series.metrics}
    return [metric_id for metric_id in pinned_metric_ids(ks) if metric_id not in present]


def not_measured_topics(
    observations: ProbeObservations, k: int = TRIPWIRE_K,
) -> list[str]:
    """Topics that were probed but produced no usable observation at *k*.

    Distinct from a failing topic and from an absent one. The report has to say
    "we could not measure this" out loud: a topic that silently vanishes from
    the tripwire's items is indistinguishable, in the artifact alone, from a
    topic that passed.
    """
    attempted: set[str] = set()
    measured: set[str] = set()
    for obs in observations.phrasings:
        if obs.k != k:
            continue
        attempted.add(obs.topic)
        if not obs.degraded:
            measured.add(obs.topic)
    return sorted(attempted - measured)
