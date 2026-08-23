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

* the TOPIC-KEYED GROUPED READ is excluded outright on LANDABILITY, and the
  reason is NARROWER than "the ``contested`` key does not exist".  A spelling
  does exist in this tree: ``grouped_read.CONTESTED_METADATA_KEY``
  (``x_contested``), placed in the Tier-C ``x_`` experimental namespace
  precisely so it needs no amendment to task 3195's closed
  ``RESERVED_VOCABULARY_KEYS`` set, and ``grouped_read.is_contested_child``
  already reads it.  What is missing is a WRITER: nothing under ``src/``
  stamps that key, and task 4004's stated basis for the exclusion is exactly
  that ``contested`` "has no writer, and no adjudication surface anywhere in
  the running system".  So PRD V2's esc-5712 protection — a contested child is
  NEVER suppressed — cannot be satisfied by any suppressing arm today, and
  every suppressing arm ships without it.  The precondition for re-deciding is
  a writer plus an adjudication surface, NOT a vocabulary amendment.
  4004 says expressly that this is "an exclusion on landability, not a
  measurement verdict", so do not restate it as one: that arm's measured row
  is reported in full so a later reader with a LIVE ``contested`` key can
  re-decide from the numbers.
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

The same caveat is carried in the ``search`` MCP tool docstring and in
``FUSED_MEMORY_INSTRUCTIONS``, so an agent reading either learns both that the
pin exists and that it is currently inert on most queries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

#: The two ``KIND_REGISTRY`` members that attach to a parent (task 3195).
#: DELIBERATELY restated rather than imported from ``server/grouped_read.py``:
#: that module imports ``_MEM0_CONTENT_KEYS`` from ``services/memory_service.py``,
#: which imports this module, so a ``services -> server`` import here would close
#: an import cycle.  ``test_child_kinds_matches_the_grouped_read_registry`` pins
#: the two copies equal so the duplication cannot drift silently (INV-5).
_CHILD_KINDS = frozenset({'amendment', 'sighting'})

#: Sorts ahead of every real ISO-8601 timestamp, so a payload with a missing or
#: non-``str`` ``created_at`` loses the recency tie-break instead of winning it
#: by accident.  Never compared for meaning — only for ordering.
_UNDATED = ''

#: Default when the config is absent, partial, or wrongly typed.  The transform
#: ships ON: it is a strict no-op wherever stamping coverage is absent (see the
#: honesty caveat above), so an unconfigured deployment pays nothing for it.
_DEFAULT_TOPIC_ANCHOR_ENABLED = True


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


def _canonical_meta(payload: object) -> dict | None:
    """The raw Qdrant payload dict of *payload*, or ``None`` if unusable.

    Every access below is defensive by necessity: these dicts come off
    ``Mem0Backend.scroll_by_metadata`` as FULL raw Qdrant payloads, whose key
    presence and value types are not schema-enforced at read time.  A malformed
    entry must degrade to "not canonical" rather than raise — a scroll result
    that trips over one junk row would take down the whole search it was only
    trying to enrich.
    """
    if not isinstance(payload, dict):
        return None
    meta = payload.get('metadata')
    if not isinstance(meta, dict):
        return None
    return meta


def _is_child_shaped(meta: dict) -> bool:
    """True when this payload would be folded into its PARENT downstream.

    ``grouped_read._suppress_child`` runs at the MCP boundary, AFTER
    ``MemoryService.search``, and replaces a child hit with its parent.  A
    child-shaped record pinned at index 0 would therefore be silently swapped
    for a DIFFERENT record than the one selected here.  Requires BOTH halves of
    the link — a child ``kind`` and a non-empty ``str`` ``parent_id`` — matching
    the condition grouped_read itself applies.

    A well-formed canonical is never a child, so this excludes only malformed
    stamped records; it is a defence against the silent swap, not a normal path.
    """
    kind = meta.get('kind')
    if not isinstance(kind, str) or kind not in _CHILD_KINDS:
        return False
    parent_id = meta.get('parent_id')
    return isinstance(parent_id, str) and bool(parent_id)


def select_canonical_payload(
    payloads: list[dict],
    *,
    allowed_categories: set[str] | None,
    include_planned: bool,
) -> dict | None:
    """Pick the ONE scrolled payload to pin for a topic, or ``None``.

    *payloads* are raw scroll dicts of shape ``{'id', 'created_at', 'metadata'}``
    as returned by ``MemoryService.get_memories_by_metadata``.

    Order of operations, which matters:

    1. drop payloads outside *allowed_categories* (when it is not ``None``),
       planned payloads (unless *include_planned*), and child-shaped payloads;
    2. keep only STRICTLY canonical survivors — ``meta.get('canonical') is True``,
       an IDENTITY check, never truthiness.  ``1 == True`` in Python, so
       truthiness would admit an int as canonical; the identity check mirrors
       the write-side uniqueness predicate at ``memory_service.py:940`` so the
       read path and the write path agree on what "canonical" means;
    3. apply the deterministic tie-break.

    Filtering BEFORE selecting is what makes *allowed_categories* a narrowing:
    anchoring may change WHICH member of the permitted set is returned, and may
    return nothing, but it must never widen the set the caller asked for.

    TIE-BREAK — most recent ``created_at``, then lowest ``id``.  More than one
    canonical per topic is REACHABLE, not theoretical: 3198's per-(project,
    topic) uniqueness enforcement ships WARN-MODE FIRST
    (``memory_metadata.enforce`` defaults false) and is inherently
    TOCTOU-windowed besides, so duplicates land through ordinary writes.  The
    ``id`` leg makes the order TOTAL, so the pin is stable whatever order the
    scroll returned rows in.  Recency is compared lexicographically on the raw
    timestamp string — chronological for the offset-normalized ISO-8601 Mem0
    stamps, and harmless otherwise because ``id`` still totalizes the order.

    Pure and synchronous: does no I/O and raises nothing on malformed input —
    and the guard covers the ARGUMENTS too, not only the payload metadata: a
    non-list *payloads*, and a non-``str`` ``category`` under a non-``None``
    *allowed_categories*, both degrade rather than raise.  That makes the
    promise checkable rather than aspirational.
    """
    if not isinstance(payloads, list):
        return None

    survivors: list[dict] = []
    for payload in payloads:
        meta = _canonical_meta(payload)
        if meta is None:
            continue
        # LOAD-BEARING, not defensive boilerplate.  These are FULL raw Qdrant
        # payloads with no read-time schema enforcement, and ``x in some_set``
        # evaluates ``hash(x)`` — so a bare membership test raises TypeError on
        # an unhashable list/dict value.  That is NOT a fail-open here: the
        # caller's two-tier band deliberately RE-RAISES TypeError
        # (``memory_service.search``; identically ``server/tools.py:3102`` for
        # the near-duplicate pre-check), so an unguarded read converts ONE
        # malformed canonical into a hard failure of every category-scoped
        # search in the system and of every procedural_knowledge write on that
        # topic.  Read the value BEFORE testing membership; ``isinstance(...,
        # str)`` also excludes bool/int/None without a separate clause, the same
        # way ``_is_child_shaped`` guards ``kind``.  EXCLUDE on an unreadable
        # category, never admit: anchoring may change WHICH member of the
        # permitted set is returned, never widen it.
        category = meta.get('category')
        if allowed_categories is not None and (
            not isinstance(category, str) or category not in allowed_categories
        ):
            continue
        if not include_planned and meta.get('planned') is True:
            continue
        if _is_child_shaped(meta):
            continue
        if meta.get('canonical') is not True:
            continue
        survivors.append(payload)

    if not survivors:
        return None

    def _recency(payload: dict) -> str:
        created_at = payload.get('created_at')
        return created_at if isinstance(created_at, str) else _UNDATED

    def _id(payload: dict) -> str:
        id_ = payload.get('id')
        return id_ if isinstance(id_, str) else ''

    # Two stable passes rather than one composite key: the two legs sort in
    # OPPOSITE directions (recency descending, id ascending), and a single
    # reverse=True would invert both, turning the documented "lowest id" tie
    # break into the highest.  Python's sort is stable, so the id ordering
    # laid down first survives inside each equal-timestamp group.
    survivors.sort(key=_id)
    survivors.sort(key=_recency, reverse=True)
    return survivors[0]


def _reconciliation_attr(memory_service: Any, attr: str) -> Any:
    """Hop config -> reconciliation -> *attr*, tolerating a break at any hop.

    Same shape as ``near_duplicate_guard._reconciliation_attr``: every hop uses
    ``getattr(..., None)`` so a missing attribute, a ``None`` config, or a
    ``None`` reconciliation section yields ``None`` rather than raising.
    """
    config = getattr(memory_service, 'config', None)
    reconciliation = getattr(config, 'reconciliation', None)
    return getattr(reconciliation, attr, None)


def resolve_topic_anchor_enabled(memory_service: Any) -> bool:
    """Whether the topic pin is enabled, read LIVE off the shared config object.

    Returns ``reconciliation.topic_anchored_recall_enabled`` iff the leaf is a
    real ``bool`` — otherwise :data:`_DEFAULT_TOPIC_ANCHOR_ENABLED`.  The
    ``isinstance(value, bool)`` guard excludes a missing/``None`` config hop,
    a wrongly-typed leaf (``1`` and ``'true'`` are NOT bools), and any attribute
    an unspecced test double might auto-generate — so a non-configuration can
    never read as a configured flag.

    Called PER SEARCH and never captured at construction.  That live read is
    the whole justification for the knob's GREEN-TIER (hot-reloadable)
    classification in ``config/reload.py``: a value captured at construction
    would not observe an in-place reload and would have to stay restart-only.
    """
    value = _reconciliation_attr(memory_service, 'topic_anchored_recall_enabled')
    if isinstance(value, bool):
        return value
    return _DEFAULT_TOPIC_ANCHOR_ENABLED
