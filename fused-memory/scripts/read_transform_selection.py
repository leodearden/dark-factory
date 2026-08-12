#!/usr/bin/env python3
"""Read-transform selection over the ratified C write shape (task 4004).

WHAT THIS IS
------------
``bake_off_storage_shape.py`` arbitrated the WRITE shape (E2, leaf ζ) and
its gate ratified **C-peers**.  That settles where the bytes go; it does
not settle what a reader is handed back.  This script measures the READ
side over that already-ratified write shape, so task **3111** (the topic
pin) can be landed against a measurement instead of an intuition.

Three candidate transforms, each a pure function over an already-fetched
hit list, each scored against the flat read as the baseline:

  1. **promoting topic pin** — the landed ``apply_topic_anchor``'s firing
     rule, but the canonical is promoted INTO the window instead of
     appended past its edge.
  2. **topic-keyed grouped read** — grouping keyed on ``topic`` rather
     than on ``parent_id``, so C's *peers* group without a parent-link
     writer, plus an explicit sighting-crediting knob.
  3. **topic-diversity cap** — at most *n* records per topic in the
     window (MMR-shaped), the cheapest family member.

WHY A SIBLING SCRIPT, NOT MORE ARMS IN THE BAKE-OFF
---------------------------------------------------
``bake_off_storage_shape.py``'s ``ARM_VARIANTS`` / ``_REQUIRED_ARM_METRICS``
/ ``_check_arms`` and its committed-artifact tests assert the E2 arm set and
protocol block exactly, because a partial table must never render as a
complete one.  Injecting read-side arms there would break those guards and
would rewrite an artifact that is the record of a DIFFERENT experiment.
This script therefore emits a SIBLING artifact pair,
``plans/read-transform-selection-report.{json,md}``, and takes the bake-off
module as a library.

WHAT IS REUSED, AND WHY NOTHING IS REIMPLEMENTED
-------------------------------------------------
Everything measurable already exists next door and is imported unchanged:
``ArmRecord``, ``ScoredHit``, ``build_canonical_by_topic``,
``claim_recall_at_k``, ``tokens_returned``, ``resolve_token_estimator``,
``_mean``, ``_rate``, the ``_atomic_write_text`` / ``_cell`` /
``_NO_MEASUREMENT`` artifact discipline, and the fetch cache.  The bake-off
carries an explicit INV-5 note (:1066-1083) about why there are not two
``recall@k`` implementations in this repo; the same argument forbids a
third here.  It also keeps this table's token column directly comparable
with the committed E2 table, since both resolve the identical estimator.

MEASUREMENT DISCIPLINE
----------------------
Inherited verbatim from the bake-off: rank/set-based metrics only, a
``None`` is *no measurement* and never a measured zero, and a missing
metric raises rather than rendering a partial row.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_script(name: str) -> Any:
    """Load a sibling script in ``scripts/`` by path.

    The same mechanism as ``bake_off_storage_shape._load_sibling_script``
    (:1513), and registered in ``sys.modules`` for the same reason:
    ``@dataclass`` and other reflection-based decorators look the defining
    module up there, so a module loaded without registering breaks on the
    way in.

    It is spelled out here rather than imported from the bake-off only
    because of the chicken-and-egg — this is the helper that loads the
    bake-off in the first place.  Once :func:`bake_off` has returned, any
    FURTHER sibling (``memory_eval_transcript_corpus``, the calibration
    fixture loader) is loaded through the bake-off's own helper, so the
    duplication stops at this one bootstrap.
    """
    import importlib.util  # noqa: PLC0415
    import sys  # noqa: PLC0415

    if name in sys.modules:
        return sys.modules[name]
    path = _SCRIPTS_DIR / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load sibling script {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def bake_off() -> Any:
    """``scripts/bake_off_storage_shape.py`` — the measurement library."""
    return _load_script('bake_off_storage_shape')


# ---------------------------------------------------------------------------
# Arm (1): the PROMOTING topic pin
# ---------------------------------------------------------------------------


def apply_promoting_topic_anchor(
    hits: list[Any],
    canonical_by_topic: dict[str, Any],
) -> list[Any]:
    """Pin each present topic's canonical to the FRONT of the ranking.

    The firing rule is the landed ``apply_topic_anchor``'s, unchanged: the
    pin fires when a hit carries ``metadata['topic'] == T``, and the record
    it pins is ``canonical_by_topic[T]``.  The two therefore select the
    identical SET of records and differ in exactly one dimension — WHERE
    the canonical lands — which is what makes this a controlled variant of
    the shipped transform rather than a second, differently-triggered rule.

    WHY A PROMOTING VARIANT EXISTS AT ALL
    -------------------------------------
    ``apply_topic_anchor`` APPENDS, deliberately, so that a measured
    improvement is attributable to the canonical becoming *reachable at
    all* rather than to the transform having hand-placed it high.  But
    ``read_path`` truncates AFTER the transforms (``records[:k]``, :3243),
    so at an already-FULL window the appended canonical is truncated
    straight back off and the pin cannot change the reader's window at all.
    The E2 report's ``pin changed window = 0.00`` under c_peers is that
    arithmetic — it is forced, not a verdict on anchoring.  Only a
    transform that places the canonical INSIDE the budget can be measured
    at a full window, which is what this one does.

    THE COST, STATED
    ----------------
    At ``len(hits) == k`` promotion evicts the k-th ranked record from the
    reader's window.  This function itself drops nothing — it reorders, and
    returns every input record — but downstream truncation at the reader's
    budget makes the pair *lossy at the window edge*.  That is a real cost
    of arm (1) and the report states it rather than leaving 3111 to
    discover it: the pin buys canonical reachability by spending the last
    slot.  It is also why an additive pin remains the right default where
    the window has headroom, where the two transforms agree exactly.

    ORDERING
    --------
    Promoted canonicals lead, ordered by their topic's FIRST APPEARANCE in
    the ranking, and everything else follows in its original relative
    order.  First-appearance is the store's own ordering signal and the
    only one available here; a set-derived order would make the answer
    depend on hash iteration, and a topic-name sort would impose an
    alphabetical prior on the reader's window.  A canonical that was
    ITSELF ranked is MOVED, never copied, so no record is ever duplicated
    and the window size is unchanged.

    ``contested`` is not read — not as an argument, not as a metadata key.
    It is a hand-labelled bake-off FIXTURE field, absent from the live
    ``RESERVED_VOCABULARY_KEYS`` (``fused_memory/memory_metadata.py``:601)
    with no writer and no adjudication surface, so an arm that needed it
    would be unimplementable today.  Arm (1) needs it not at all.

    ``canonical is True`` is NOT re-checked here, for the same reason
    ``apply_topic_anchor`` does not re-check it: β's
    ``invalid_canonical_type`` rule is bool identity and is enforced once,
    at index-construction time, by ``build_canonical_by_topic``.

    Returns the input list unchanged — same object, same elements — when
    nothing is pinnable.  Pure: never mutates *hits*, *canonical_by_topic*,
    or any record in either.
    """
    # dict-not-set: insertion order makes the promoted prefix deterministic.
    present_topics: dict[str, None] = {}
    for hit in hits:
        topic = hit.metadata.get('topic')
        if topic is not None:
            present_topics.setdefault(topic, None)

    promoted: list[Any] = []
    promoted_ids: set[str] = set()
    for topic in present_topics:
        canonical = canonical_by_topic.get(topic)
        if canonical is None:
            continue  # no canonical for this topic — never synthesize one
        if canonical.record_id in promoted_ids:
            continue  # two topics, one canonical: promote it once
        promoted.append(canonical)
        promoted_ids.add(canonical.record_id)

    if not promoted:
        return hits
    # An already-ranked canonical is MOVED to the front, not duplicated;
    # every other hit keeps its relative order.
    return [*promoted, *(hit for hit in hits if hit.record_id not in promoted_ids)]


# ---------------------------------------------------------------------------
# Per-arm declarations the decision table reads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    """What an arm costs a reader, declared at the source.

    "Suppressing read" is TWO facts here, deliberately not collapsed into
    one boolean, because collapsing them would mislead task 3111:

      * ``drops_ranked_records`` — the transform itself removes records the
        store ranked from the window (arm 2 folds members into a document;
        arm 3 caps them out).  This is landable-or-not on its own merits.
      * ``requires_contested_key_for_v2`` — satisfying PRD **V2**'s
        esc-5712 protection ("a contested child is NEVER suppressed") under
        this arm would need a ``contested`` key.  That key does not exist:
        it is absent from ``RESERVED_VOCABULARY_KEYS``
        (``fused_memory/memory_metadata.py``:601), has no writer and no
        adjudication surface, so an arm that needs it cannot ship the
        protection today no matter how good its columns look.
      * ``displaces_at_window_edge`` — a third shape again: the transform
        drops nothing, but ``read_path``'s truncation at the reader's
        budget (:3243) evicts the k-th record because the transform put
        something ahead of it.  Arm (1) is only this.

    An arm can suppress and still be landable (arm 3: it drops records but
    reads only ``metadata['topic']``), so the flags are reported separately
    rather than summed into a verdict.
    """

    key: str
    label: str
    drops_ranked_records: bool
    requires_contested_key_for_v2: bool
    displaces_at_window_edge: bool


ARM_SPECS: dict[str, ArmSpec] = {
    'promoting_pin': ArmSpec(
        key='promoting_pin',
        label='promoting topic pin',
        drops_ranked_records=False,
        requires_contested_key_for_v2=False,
        displaces_at_window_edge=True,
    ),
    'topic_keyed_grouped': ArmSpec(
        key='topic_keyed_grouped',
        label='topic-keyed grouped read',
        drops_ranked_records=True,
        requires_contested_key_for_v2=True,
        displaces_at_window_edge=False,
    ),
}


# ---------------------------------------------------------------------------
# Arm (2): the TOPIC-KEYED grouped read
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentProvenance:
    """Where a synthesized document's ``record_id`` came from.

    A grouped document is emitted under its CANONICAL's ``record_id``, so
    any metric that credits on ``hit.record_id == canonical_record_id``
    scores a fold as "the canonical was found" — whether or not the
    canonical's own stored record ever ranked.  That aliasing is invisible
    in the record alone, so it is recorded here instead of being left for a
    reader of the table to infer.

    It does NOT ride in ``ArmRecord.metadata``: that dict carries only what
    would really be written to Mem0, precisely so
    ``validate_memory_metadata`` can be run over it as a conformance
    oracle, and a provenance key there would manufacture ``unknown_key``
    census noise.  ``ArmRecord`` has no field for it either, and a second
    function that recomputed the grouping to report it could drift from the
    transform it claims to describe — so it is returned alongside.
    """

    record_id: str
    #: Member record_ids folded into this document, in render order.
    aliased_from: tuple[str, ...]
    #: Did the canonical's OWN stored record rank, or was it pulled in?
    canonical_was_itself_ranked: bool


@dataclass(frozen=True)
class TransformResult:
    """What a reader receives, plus how the synthesized parts were made."""

    records: list[Any]
    #: Keyed by the synthesized document's ``record_id``.  Empty for a
    #: transform that synthesizes nothing.
    provenance: dict[str, DocumentProvenance]


def _canonical_for(topic: str, records_by_topic: dict[str, list[Any]]) -> Any:
    """The topic's canonical, via the bake-off's own indexer.

    Delegated rather than re-derived: ``build_canonical_by_topic`` enforces
    bool-identity ``canonical is True`` (β's ``invalid_canonical_type``
    rule, where a truthy ``1`` is FATAL) and RAISES on two canonicals in one
    topic rather than letting the answer depend on scroll order.  Restating
    either rule here would be the second implementation INV-5 forbids.
    """
    members = records_by_topic.get(topic)
    if not members:
        return None
    return bake_off().build_canonical_by_topic(list(members)).get(topic)


def apply_topic_keyed_grouped_read(
    hits: list[Any],
    records_by_topic: dict[str, list[Any]],
    *,
    render_sightings: bool,
) -> TransformResult:
    """Group a topic's ranked peers into one document keyed on ``topic``.

    WHY TOPIC-KEYED AND NOT ``parent_id``-KEYED
    -------------------------------------------
    ``apply_grouped_read`` (:837) keys on ``parent_id``, and the ratified C
    write shape stores PEERS: a topic's records carry ``topic`` and
    ``canonical`` and NO parent link (``_materialize_c_peers``, :653-694).
    Parent-keyed grouping is therefore structurally inert over C — every
    peer is "parentless", passes through in place, and the transform
    measures nothing.  Keying on ``topic`` is the corner eval-design OQ5
    actually asked for (peers PLUS children) and that PRD D9 mislabelled as
    "already C".  It needs no ``parent_id`` writer, no backfill and no new
    vocabulary key: ``topic`` is already reserved and already written.

    THE SIGHTING KNOB
    -----------------
    ``apply_grouped_read`` hard-codes one policy (:927): credit
    ``[canonical, *amendments, *others]`` and collapse sightings to a bare
    count.  Its own comment names the alternative — "if a sighting's claim
    is meant to be recallable, the fix is to render its body and pay the
    tokens, not to credit it unrendered".  ``render_sightings`` is that
    dial, so the report can PRICE the trade instead of inheriting it:

      * ``False`` — sightings collapse to ``[sightings: n]``; their claims
        are NOT credited.  Cheapest, and the ceiling on claim recall is
        ``(claims - sightings + contested) / claims``.
      * ``True`` — sighting bodies are rendered as ``[sighting] …``; their
        claims ARE credited.  The ceiling goes to 1.0 and the token column
        pays for it.

    Crediting and rendering are held in agreement in BOTH settings: a claim
    is credited only if its text actually reached the reader.  Otherwise the
    arm banks recall and a token discount for the same content — a double
    advantage in exactly the two columns the decision table is read on.

    WHAT THE INDEX MAY CONTRIBUTE
    -----------------------------
    ``records_by_topic`` supplies exactly ONE record the store did not
    return: the topic's canonical, so a member hit can resolve UPWARD to
    its anchor (PRD **D6** — a member whose canonical is unreachable is the
    whole objection to δ/Option B).  Every other member of the document was
    itself a hit.  Folding in unranked members would manufacture retrieval:
    the arm would be credited for content the store never surfaced, which
    is not a property of the read transform at all.

    A topic with no canonical is left FLAT rather than anointing its
    best-ranked peer — per-(project, topic) canonical uniqueness is leaf ε's
    rule, and electing a stand-in would make the document depend on ANN
    order.  A group whose only member is the canonical itself is returned by
    identity: a group of one IS its canonical, and synthesizing a
    byte-identical copy would allocate on the hot path for nothing.

    Ranking: a group lands at the BEST rank among its members, the same rule
    ``apply_grouped_read`` applies to rank and ``rescore`` (:3246) applies to
    score.  Demoting it to the worst would make grouping look worse at every
    k as an artifact of this transform rather than of the read shape.

    SUPPRESSION, STATED
    -------------------
    This arm DROPS ranked records: a folded member stops being an
    independent hit.  It therefore carries
    ``ARM_SPECS['topic_keyed_grouped'].drops_ranked_records = True`` and
    ``requires_contested_key_for_v2 = True`` — PRD V2's esc-5712 protection
    ("a contested child is NEVER suppressed") is exactly what a suppressing
    read needs, and ``contested`` is a hand-labelled bake-off fixture field
    with no live vocabulary key, no writer and no adjudication surface.  The
    transform takes no ``contested`` argument because there is nothing that
    could populate one; the report says so rather than implying otherwise.

    Pure — never mutates *hits*, *records_by_topic*, or any record.
    """
    module = bake_off()

    group_members: dict[str, list[Any]] = {}
    group_rank: dict[str, int] = {}
    output: list[tuple[int, Any]] = []

    for rank, hit in enumerate(hits):
        topic = hit.metadata.get('topic')
        if topic is None or _canonical_for(topic, records_by_topic) is None:
            # No topic, or no canonical to anchor a document: flat, in place.
            output.append((rank, hit))
            continue
        if topic not in group_members:
            group_members[topic] = []
            group_rank[topic] = rank
        group_members[topic].append(hit)

    provenance: dict[str, DocumentProvenance] = {}

    for topic, members in group_members.items():
        canonical = _canonical_for(topic, records_by_topic)
        canonical_was_ranked = any(
            m.record_id == canonical.record_id for m in members
        )
        children = [m for m in members if m.record_id != canonical.record_id]

        if not children:
            # A group of one: it IS the canonical. Return the stored record
            # itself rather than a byte-identical synthesized copy.
            output.append((group_rank[topic], canonical))
            continue

        amendments = [c for c in children if c.metadata.get('kind') == 'amendment']
        sightings = [c for c in children if c.metadata.get('kind') == 'sighting']
        others = [
            c for c in children
            if c.metadata.get('kind') not in ('amendment', 'sighting')
        ]

        # Rendering and crediting are decided ONCE, together, so they cannot
        # disagree: `rendered` is what reaches the reader and `credited` is
        # what the metric may count, and the second is derived from the first.
        rendered = [*amendments, *others, *(sightings if render_sightings else [])]
        counted = 0 if render_sightings else len(sightings)
        credited = [canonical, *rendered]

        claim_ids: list[str] = []
        for member in credited:
            for claim_id in member.claim_ids:
                if claim_id not in claim_ids:
                    claim_ids.append(claim_id)

        output.append((group_rank[topic], module.ArmRecord(
            record_id=canonical.record_id,
            content=module._render_grouped_document(canonical, rendered, counted),
            # A fresh dict: the stored canonical's metadata is never mutated.
            metadata=dict(canonical.metadata),
            cluster_id=canonical.cluster_id,
            claim_ids=claim_ids,
            role=module.GROUPED_ROLE,
        )))

        # Recorded HERE, in the one branch that actually synthesizes a
        # document, so `provenance` describes exactly the aliased records
        # and nothing else.  Derived from the same `canonical` / `children`
        # the document was built from, so the disclosure cannot drift from
        # the fold it claims to describe.
        provenance[canonical.record_id] = DocumentProvenance(
            record_id=canonical.record_id,
            aliased_from=tuple(child.record_id for child in children),
            canonical_was_itself_ranked=canonical_was_ranked,
        )

    # Stable sort on the original rank: each rank is consumed once, so the
    # order is total and the transform deterministic.
    output.sort(key=lambda pair: pair[0])
    return TransformResult(
        records=[record for _, record in output], provenance=provenance,
    )


def claim_recall_ceiling(
    records: list[Any],
    *,
    render_sightings: bool,
    contested_ids: frozenset[str] | set[str] = frozenset(),
) -> float | None:
    """The best claim recall a grouped read of *records* could reach.

    A claim carried ONLY by a record whose body never reaches the reader is
    unrecallable no matter how the store ranks — so the sighting-crediting
    policy sets a hard ceiling, and this states it rather than leaving it to
    be inferred from a column.  With one distinct claim per record the
    identity reduces to the arithmetic the report prints:

        ceiling = (claims - sightings + contested) / claims

    The ``contested`` term is PRD V2's protection: a contested member is
    never suppressed, survives as its own hit, and so its claim stays
    reachable.  It is arithmetic that production cannot currently make
    non-zero — ``contested`` is absent from ``RESERVED_VOCABULARY_KEYS``
    with no writer — and *contested_ids* therefore defaults to empty.  The
    term is kept because the report has to state what V2 would buy if the
    key existed, and a hard-coded zero would erase the question.

    Returns ``None`` — no measurement, never a measured zero — when there
    are no claims to recall at all.  A 0.0 there would read as "this arm
    recalls nothing", which is a different and false statement.
    """
    contested = set(contested_ids)
    all_claims = {claim for record in records for claim in record.claim_ids}
    if not all_claims:
        return None
    if render_sightings:
        return 1.0

    reachable = {
        claim
        for record in records
        for claim in record.claim_ids
        if record.metadata.get('kind') != 'sighting' or record.record_id in contested
    }
    return len(reachable) / len(all_claims)
