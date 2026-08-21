"""Topic-anchored canonical recall: pure selectors for the promoting topic pin (task 3111).

THE RETRIEVAL INVERSION THIS FIXES
----------------------------------
Consolidating a cluster of near-duplicate memories into one canonical record
makes that record the LEAST retrievable member of its own cluster.  A
consolidated canonical is long, general and multi-claim, so its embedding sits
far from any single narrow query, while each surviving narrow sibling stays a
tight cosine match.  At ``limit=5`` the window fills with siblings and the
canonical — the record that actually answers the question — never appears.
Consolidation, done well, therefore makes recall WORSE unless the read path
knows about it.  This module is that knowledge.

THE ARM WAS SELECTED BY MEASUREMENT, NOT BY ARGUMENT
----------------------------------------------------
Gate 3200 ratified Option C's WRITE shape and deliberately ratified NO read
transform, deferring the choice to measurement.  Task 4004 ran that
measurement (``plans/read-transform-selection-report.md`` /
``.json``, ``recommendation.arm = promoting_pin``) under a selection rule
fixed BEFORE the numbers were read: landability first, then claim recall, then
tokens/query, then displacement.  The promoting pin won on every tier it was
scored at — claim recall 1.00, 1070.27 tokens/query against a 1181.29 baseline
(cheaper, not merely affordable), and it drops no ranked records.

The two rejected arms are rejected for reasons that have not changed:

* the TOPIC-KEYED GROUPED READ is excluded outright on LANDABILITY — it needs
  a ``contested`` metadata key, and ``RESERVED_VOCABULARY_KEYS`` in
  ``memory_metadata.py`` is ``{topic, canonical, kind, parent_id, supersedes}``.
  There is no ``contested``, so that arm cannot be built at all today.
* the TOPIC-DIVERSITY CAP scores 0.63 claim recall and drops ranked records.

Do not "simplify" that choice back into an argument.  Re-opening the arm
selection means re-running 4004's measurement, not re-reasoning from the
prose above.

THIS MODULE IS PURE AND SYNCHRONOUS
-----------------------------------
Mirroring ``server/near_duplicate_guard.py``: every function here is a pure
selector over already-fetched data.  The CALLER owns the backend round-trip
and the fail-open.  This module must NEVER import ``memory_service`` (import
cycle) — ``MemoryResult`` is imported under ``TYPE_CHECKING`` only.

NEVER STAMP A SCORE ONTO A PINNED RESULT
----------------------------------------
Since task 3658 ``relevance_score`` is an ordinal RRF fusion value
(rank-1 ~ 0.0164), NOT a cosine; the honest per-store cosine lives in
``metadata['store_score']``.  The write-time near-duplicate guard reads the
cosine from ``metadata['store_score']`` and qualifies on ``>= threshold``
(``near_duplicate_guard.py`` ``_cosine_of`` / ``find_near_duplicate_memory``).
An injected anchor therefore MUST carry no ``store_score`` at all: a missing
cosine means "not comparable" and can never qualify at any threshold, whereas
a synthetic high score would hard-block EVERY ``procedural_knowledge`` write
on a consolidated topic — turning a retrieval fix into a write outage on
precisely the topics it exists to help.  The pin is by ORDER ONLY.

HONESTY CAVEAT — THIS IS A NO-OP ON ALMOST EVERY LIVE SEARCH TODAY
------------------------------------------------------------------
Stamping COVERAGE, not ranking, is the binding constraint, and coverage is
task 4006's scope (still PENDING), not this module's.  4006's census measured
``metadata.topic`` on 491 of 49,628 records and ``metadata.canonical: true``
on 6.  So this transform fires for almost no search on the live corpus as it
stands, and no user-observable live-corpus recall improvement is claimed here.
Task 3659 (briefing assembler) is a FUTURE consumer, explicitly not a live one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fused_memory.models.enums import SourceStore

if TYPE_CHECKING:
    # MemoryResult under TYPE_CHECKING only, matching near_duplicate_guard.py.
    # models.enums is a leaf module and safe to import for real; models.memory
    # is not imported at runtime so this module stays importable from anywhere
    # in the service layer without risking a cycle.
    from fused_memory.models.memory import MemoryResult

# Blast-radius caps, NOT tuning knobs — deliberately not config leaves.
# MemoryService.search is on the hot path for every agent read AND for every
# procedural_knowledge write (the near-dup guard's pre-check), so the cost this
# transform can add to a single search is bounded here by construction rather
# than left to an operator to get right.
#
# Most distinct topics anchored per search => at most this many extra backend
# round-trips, whatever the result window contains.
_MAX_ANCHOR_TOPICS = 3
# Rows fetched per anchored topic.  A well-formed topic has exactly ONE
# canonical (3198's per-(project, topic) uniqueness), so this is headroom for
# the warn-mode duplicate case, not an expected working-set size.
_ANCHOR_SCROLL_LIMIT = 25


def extract_anchor_topics(
    results: list[MemoryResult],
    *,
    max_topics: int,
) -> list[str]:
    """Distinct ``metadata['topic']`` values worth anchoring, in rank order.

    Scans *results* — which the caller has already merged, sorted and
    category-filtered — and returns the distinct topics carried by
    Mem0-sourced results, in FIRST-SEEN order, de-duplicated, truncated to
    *max_topics*.

    Only Mem0 results are considered: ``metadata.topic`` is a Mem0 vocabulary
    key (``memory_metadata.py``), and a Graphiti row carries no such payload.

    Values are read defensively.  These come from raw Qdrant payloads whose
    key presence and value types are not schema-enforced at READ time, so a
    missing, empty, or non-``str`` topic is skipped rather than raised on.
    ``bool`` is excluded despite being an ``int`` subclass, consistent with
    the coercion discipline in ``near_duplicate_guard._cosine_of``.

    Returning ``[]`` is the ZERO-COST path and the caller must treat it as
    "make no backend call at all" — on the live corpus today that is the
    overwhelmingly common case (see the module docstring's honesty caveat).

    Pure and synchronous: does no I/O and raises nothing on empty input.
    """
    topics: list[str] = []
    seen: set[str] = set()
    for result in results:
        if result.source_store != SourceStore.mem0:
            continue
        value = (result.metadata or {}).get('topic')
        if not isinstance(value, str) or isinstance(value, bool) or not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        topics.append(value)
        if len(topics) >= max_topics:
            break
    return topics
