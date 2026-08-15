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
    'topic_diversity_cap': ArmSpec(
        key='topic_diversity_cap',
        label='topic-diversity cap',
        # It DROPS ranked records — a suppressing read in the PRD V2 sense.
        drops_ranked_records=True,
        # ...and yet it is LANDABLE TODAY.  The cap is computed from
        # `metadata['topic']` alone, so it never needs to ask whether a
        # record is contested.  That is why these two flags are separate:
        # collapsing them into one "suppressing" boolean would tell 3111
        # this arm is blocked on a key that does not exist, which is the
        # opposite of the truth.
        requires_contested_key_for_v2=False,
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


# ---------------------------------------------------------------------------
# Arm (3): the TOPIC-DIVERSITY CAP (MMR-style)
# ---------------------------------------------------------------------------


def apply_topic_diversity_cap(
    hits: list[Any],
    canonical_by_topic: dict[str, Any],
    *,
    per_topic: int = 1,
) -> list[Any]:
    """Keep at most *per_topic* records of any one topic in the window.

    The cheapest member of the family: it renders nothing, credits nothing
    and synthesizes nothing.  Every record it returns is a record it was
    given — the transform ONLY DROPS.  So there is no document to alias, no
    crediting policy to set, and no provenance channel to carry: it returns
    a bare list, where arm (2) must return a :class:`TransformResult`
    because it mints documents under a canonical's ``record_id``.

    WHICH MEMBER SURVIVES
    ---------------------
    The topic's CANONICAL, when the canonical is itself among the ranked
    members; otherwise the best-ranked member.  Diversity should hand the
    reader the topic's ANCHOR rather than whichever member ANN happened to
    like most, which is the whole reason the write shape marks one.

    The cap NEVER pulls an unranked record in — not even the canonical.
    Injecting an absent canonical here would be arm (1)'s promotion, and
    would credit this arm for retrieval the store never performed.  A topic
    whose canonical never ranked therefore keeps its best peer, and the
    reader simply does not learn the anchor exists.

    WHERE THE SURVIVOR SITS
    -----------------------
    At the topic's BEST held rank, the same min-rank rule
    :func:`apply_topic_keyed_grouped_read` applies and that ``rescore``
    (:3246) applies to score.  Demoting the survivor to the canonical's own
    (worse) rank would make this arm look worse at every *k* as an artifact
    of the transform rather than of the read shape.  Survivors are emitted
    in their ORIGINAL relative order, so when nothing is dropped the window
    is returned exactly as it came in — the cap is order-stable, and
    idempotent on an already-capped window.

    UNTOPICED RECORDS ARE INVISIBLE TO IT
    -------------------------------------
    The 300-record distractor slab (``_distractor_records``, :545) carries
    ``category`` and NOT ONE reserved vocabulary key, deliberately, so that
    a distractor cannot become a right answer.  A topic cap that collapsed
    untopiced records would delete the contamination variable the whole
    bake-off is controlled on, so they pass through untouched and un-capped.

    THE FREED SLOT BACKFILLS
    ------------------------
    The returned list is SHORTER; it is never padded.  At a fixed reader
    budget *k* the freed slots are therefore filled from deeper in the
    ranking — records that ranked below *k* move inside it.  That is the
    token win worth measuring, and it is what the report's token column
    prices.

    SUPPRESSION, STATED
    -------------------
    This arm DROPS ranked records, so it is a suppressing read in the PRD
    **V2** sense — and yet it computes from ``metadata['topic']`` alone and
    reads nothing from a ``contested`` key, so unlike arm (2) it is
    LANDABLE TODAY.  Those are the two separate flags on
    ``ARM_SPECS['topic_diversity_cap']``; see :class:`ArmSpec` for why they
    are not one boolean.

    Pure — never mutates *hits*, *canonical_by_topic*, or any record.
    """
    if per_topic < 1:
        # Loud, not fail-soft: a cap of 0 would empty every topic out of the
        # window and report the result as a measurement.
        raise ValueError(f'per_topic must be >= 1, got {per_topic}')

    #: topic -> [(rank, record)], in rank order.
    members: dict[str, list[tuple[int, Any]]] = {}
    output: list[tuple[int, Any]] = []

    for rank, hit in enumerate(hits):
        topic = hit.metadata.get('topic')
        if topic is None:
            # Untopiced: straight through, by identity, in place.
            output.append((rank, hit))
            continue
        members.setdefault(topic, []).append((rank, hit))

    for topic, ranked in members.items():
        if len(ranked) <= per_topic:
            # Nothing to drop — every member keeps its own rank.
            output.extend(ranked)
            continue

        canonical = canonical_by_topic.get(topic)
        canonical_id = None if canonical is None else canonical.record_id
        # Preference order for SELECTION: the canonical (only if it is
        # itself among the ranked members), then the rest by rank.  This
        # decides WHICH records survive, not where they sit.
        preferred = sorted(
            ranked, key=lambda pair: (pair[1].record_id != canonical_id, pair[0]),
        )
        # Placement: the topic's best-held rank slots, with the survivors in
        # their ORIGINAL relative order.  Carried as (rank, record) pairs
        # throughout — resolving a record back to its position by value
        # would let two records with identical fields resolve to the same
        # slot, which is a real corpus (peers share content shape), and
        # would silently drop one of them.
        chosen = sorted(preferred[:per_topic], key=lambda pair: pair[0])
        slots = [rank for rank, _ in ranked[:len(chosen)]]
        output.extend(
            (slot, record)
            for slot, (_, record) in zip(slots, chosen, strict=True)
        )

    # Stable sort on the original rank: each rank is consumed at most once,
    # so the order is total and the transform deterministic.
    output.sort(key=lambda pair: pair[0])
    return [record for _, record in output]


# ---------------------------------------------------------------------------
# The de-aliased canonical metric
# ---------------------------------------------------------------------------


def canonical_discoverability(
    hits: list[Any],
    topic: str,
    canonical_record_id: str,
    k: int,
    *,
    provenance: dict[str, DocumentProvenance] | None = None,
) -> dict[str, Any]:
    """Was the topic's canonical found — aliased, and for real?

    ``topic_discoverability`` (:1116) credits the canonical on
    ``hit.record_id == canonical_record_id``.  A grouped read emits its
    document under the canonical's OWN ``record_id``
    (``apply_grouped_read``:934-935, and
    :func:`apply_topic_keyed_grouped_read` for the same reason — the
    document IS the canonical, amended), so any member folding upward
    scores as "canonical found" whether or not the canonical's own stored
    record ever ranked.  That is a property of the TRANSFORM, not of
    retrieval, and it is the mechanism behind b_grouped's 0.97
    canonical-in-top-5 in the committed E2 table.

    WHY BOTH RATES, RATHER THAN A FIXED COLUMN
    ------------------------------------------
    Simply switching to the honest semantics would make this table silently
    incomparable with the committed one, and would erase a real finding: a
    grouped read genuinely DOES put the canonical's body in the reader's
    window, which is a reader benefit and why the behaviour is arguably
    correct rather than a bug.  So both are reported and the reader is told
    which is which:

      * ``aliased_*`` — the legacy ``record_id``-match semantics, preserved
        exactly so the column stays comparable across tables.
      * ``unaliased_*`` — the canonical's OWN stored record actually
        ranked.

    The GAP between them IS the aliasing, quantified per arm.

    EVIDENCE, NOT INFERENCE
    -----------------------
    A synthesized document is indistinguishable from a stored record at the
    ``record_id`` level — that is the entire problem — so the honest rate
    cannot be computed from *hits* alone.  It is read from the transform's
    own *provenance*, emitted by the transform that did the folding.  With
    no provenance the unaliased verdict is ``None``: **no measurement, not
    a measured zero**, and never a quiet fallback to crediting.  A flat arm
    synthesizes nothing, so passing ``{}`` (rather than ``None``) correctly
    reports the two as equal.

    Ranks are **1-based**, and ``None`` when the canonical is absent
    entirely — inherited from ``topic_discoverability``, where ``0`` would
    collide with a real rank under a 0-based reading and would average as
    "very good" in a mean-rank summary.  A canonical ranked OUTSIDE the
    window still reports its rank: "absent from top-5" and "absent
    entirely" are different findings.

    Rank/set-based only: no score is read.  Pure.
    """
    # The aliased half is DELEGATED, not restated, so it cannot drift from
    # the semantics the committed E2 table was measured under (INV-5).
    landed = bake_off().topic_discoverability(
        hits, topic, canonical_record_id, k,
    )
    aliased_rank = landed['canonical_rank']

    matched = next(
        (hit for hit in hits if hit.record_id == canonical_record_id), None,
    )
    #: Only a SYNTHESIZED document can alias.  A stored record carrying the
    #: canonical's id simply IS the canonical, so a flat arm needs no
    #: disclosure to be scored honestly — which is why a missing
    #: *provenance* is not automatically an unknown.
    synthesized = matched is not None and matched.role == bake_off().GROUPED_ROLE

    unaliased_rank: int | None = aliased_rank
    unaliased_in_top_k: bool | None
    disclosure = None if provenance is None else provenance.get(canonical_record_id)

    if disclosure is not None:
        if not disclosure.canonical_was_itself_ranked:
            # A document minted under the canonical's id, and the canonical
            # itself never ranked.  The reader DID receive its body — which
            # is why the aliased column is not "wrong" — but retrieval did
            # not find it.
            unaliased_rank = None
    elif synthesized:
        # A synthesized document with no disclosure.  Aliasing is invisible
        # in the record alone — that is the entire problem — so this is
        # unknowable, and an unknown is not a zero.
        return {
            'aliased_in_top_k': landed['canonical_in_top_k'],
            'aliased_rank': aliased_rank,
            'unaliased_in_top_k': None,
            'unaliased_rank': None,
            'topic_member_count': landed['topic_member_count'],
        }

    unaliased_in_top_k = unaliased_rank is not None and unaliased_rank <= k

    return {
        'aliased_in_top_k': landed['canonical_in_top_k'],
        'aliased_rank': aliased_rank,
        'unaliased_in_top_k': unaliased_in_top_k,
        'unaliased_rank': unaliased_rank,
        'topic_member_count': landed['topic_member_count'],
    }


# ---------------------------------------------------------------------------
# The UNLABELED production query set
# ---------------------------------------------------------------------------
#
# External validity: the E2 query set is blind-authored, so an arm that wins
# only there has won on questions nobody asked.  `harvest_production_queries`
# samples what actually reaches `search`, and those queries are scored here.
#
# They are UNLABELED by construction — the write journal records what was
# asked, never what should have come back — so the labeled columns are `None`
# and render as `—`.  That is a disclosure, not a gap: the alternative is
# fabricating the ground truth the measurement exists to establish.

#: Production fires the briefing family at ``limit=5``
#: (``orchestrator/src/orchestrator/agents/briefing.py``:1376), NOT the E2
#: default of 10.  Scoring the production set at 10 would measure a window
#: no production reader is ever handed.
PRODUCTION_K = 5

#: Every field a production row must carry.  There is deliberately no
#: ``expects_claim_ids`` here, and :class:`ProductionQuery` has no such
#: attribute — see the class docstring.
_PRODUCTION_FIELDS = (
    'query_id', 'text', 'source', 'observed_count', 'traffic_share',
)

#: Fields whose PRESENCE means an E2 row has strayed into the unlabeled set.
_LABELED_FIELDS = ('expects_claim_ids', 'kind', 'cluster_id')

_UNLABELED_REASON = (
    'Production queries carry no ground truth: the reconciliation write '
    'journal records which query was issued, never which memory should have '
    'been returned. Claim recall and canonical discoverability are therefore '
    'not computable over this set and render as no-measurement, not as a '
    'measured zero.'
)


class ProductionQuerySetError(RuntimeError):
    """The production query fixture is malformed, or is not unlabeled."""


@dataclass(frozen=True)
class ProductionQuery:
    """One sampled production query.

    Deliberately NOT a :class:`bake_off.Query`: that class requires
    ``expects_claim_ids`` (:199) and ``cross_validate_fixtures`` (:477)
    requires it non-empty.  Subclassing or defaulting it to ``[]`` here
    would let a scorer compute ``0/0`` and report a recall for a query with
    no ground truth at all — which is precisely the fabrication this whole
    path exists to avoid.  The absence is structural: there is no attribute
    to read, so no code path can accidentally read one.
    """

    query_id: str
    text: str
    source: str
    observed_count: int
    traffic_share: float | None
    observed_limit: int = PRODUCTION_K
    template: str | None = None
    match: str | None = None


def load_production_queries(path: str | Path) -> list[ProductionQuery]:
    """Read the harvested production sample (unlabeled by construction).

    Refuses a row carrying any labeled field.  The E2 loader already refuses
    an unlabeled row (``load_query_set``:368 raises on the missing ``kind``);
    this is the symmetric half, so the two corpora cannot silently merge in
    either direction.
    """
    import json  # noqa: PLC0415

    target = Path(path)
    if not target.exists():
        raise ProductionQuerySetError(f'production query set not found: {target}')
    queries: list[ProductionQuery] = []
    seen: set[str] = set()
    for lineno, line in enumerate(target.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ProductionQuerySetError(f'{target}:{lineno}: not JSON: {exc}') from exc
        for field_name in _LABELED_FIELDS:
            if field_name in row:
                raise ProductionQuerySetError(
                    f'{target}:{lineno}: row carries the labeled field '
                    f'{field_name!r}. The production set is unlabeled by '
                    'construction; a labeled row belongs in the E2 query set, '
                    'where its ground truth is actually used.'
                )
        for field_name in _PRODUCTION_FIELDS:
            if field_name not in row:
                raise ProductionQuerySetError(
                    f'{target}:{lineno}: production query missing field {field_name!r}'
                )
        query_id = row['query_id']
        if query_id in seen:
            raise ProductionQuerySetError(
                f'{target}:{lineno}: duplicate query_id {query_id!r}'
            )
        seen.add(query_id)
        queries.append(ProductionQuery(
            query_id=query_id,
            text=row['text'],
            source=row['source'],
            observed_count=int(row['observed_count']),
            traffic_share=row['traffic_share'],
            observed_limit=int(row.get('observed_limit', PRODUCTION_K)),
            template=row.get('template'),
            match=row.get('match'),
        ))
    if not queries:
        raise ProductionQuerySetError(f'{target}: no production queries')
    return queries


# ---------------------------------------------------------------------------
# Scoring without labels
# ---------------------------------------------------------------------------


def _topic_of(record: Any) -> str | None:
    metadata = getattr(record, 'metadata', None) or {}
    topic = metadata.get('topic')
    return topic if isinstance(topic, str) else None


def _reachable_ids(
    records: list[Any], provenance: dict[str, DocumentProvenance] | None,
) -> set[str]:
    """Record ids a reader can still SEE, counting folded members.

    A grouped document renders its members' bodies, so the member's
    knowledge is in the window even though the member is no longer its own
    hit.  The fold is only visible in the transform's own provenance — from
    the records alone a folded member is indistinguishable from a dropped
    one, which is exactly why the disclosure is load-bearing.
    """
    reachable = {record.record_id for record in records}
    if provenance:
        for record in records:
            disclosure = provenance.get(record.record_id)
            if disclosure is not None:
                reachable.update(disclosure.aliased_from)
    return reachable


def score_unlabeled_query(
    records: list[Any],
    *,
    baseline: list[Any],
    k: int = PRODUCTION_K,
    provenance: dict[str, DocumentProvenance] | None = None,
    estimator: tuple[str, Any] | None = None,
) -> dict[str, Any]:
    """Every metric computable for a query with no ground truth.

    Labeled metrics are ``None`` — no measurement, never a measured zero —
    and carry their reason so a ``—`` cell in the artifact is explainable
    without reading this file.

    RETENTION vs DISPLACEMENT
    -------------------------
    Two costs, deliberately not collapsed:

      * ``baseline_retention_at_k`` — the fraction of the flat read's top-k
        that is still REACHABLE.  A member folded into a grouped document
        counts as retained: the reader still gets its body.
      * ``window_displacement`` — how many of those records are no longer
        INDEPENDENTLY present.  A folded member is displaced even though it
        is retained; so is a record the cap dropped, and so is one the
        promoting pin pushed past the window edge.

    Collapsing them into a single "changed" number would hide whichever
    half a reader cared about — and they are exactly the two halves task
    3111 has to trade off.

    A pure REORDER inside the window displaces nothing (``window_reordered``
    reports it separately): motion is not eviction, and charging for it
    would penalise the transform for doing its job.

    Rank/set-based only.  Pure over *records* and *baseline*.
    """
    window = records[:k]
    baseline_window = baseline[:k]
    bake = bake_off()

    reachable = _reachable_ids(window, provenance)
    present = {record.record_id for record in window}
    baseline_ids = [record.record_id for record in baseline_window]

    retained = sum(1 for rid in baseline_ids if rid in reachable)
    displaced = sum(1 for rid in baseline_ids if rid not in present)

    topics = {t for t in (_topic_of(r) for r in window) if t is not None}
    untopiced = sum(1 for r in window if _topic_of(r) is None)

    token_block = bake.tokens_returned(window, k, estimator)

    return {
        # --- not computable without ground truth ---------------------
        'claim_recall': None,
        'canonical_aliased_in_top_k': None,
        'canonical_unaliased_in_top_k': None,
        'unlabeled': True,
        'unlabeled_reason': _UNLABELED_REASON,
        # --- computable ----------------------------------------------
        'k': k,
        'window_size': len(window),
        'tokens_per_query': token_block['tokens'],
        'token_estimator': token_block['estimator'],
        'topic_diversity': len(topics),
        'untopiced_in_window': untopiced,
        'baseline_retention_at_k': bake._rate(retained, len(baseline_ids)),
        'window_displacement': displaced,
        'window_reordered': [r.record_id for r in window] != baseline_ids,
    }


def __getattr__(name: str) -> Any:
    """Re-export the bake-off's helpers BY IDENTITY, lazily (INV-5).

    ``_mean`` (:3286), ``_rate`` (:1622) and ``_cell`` (:1955) carry the
    no-measurement
    discipline this module depends on — a ``None`` is a non-observation and
    must never average in as a zero.  A local wrapper would be a second
    implementation that could drift; a module-level import would force the
    bake-off to load at import time.  PEP 562 gives both: the attribute
    resolves to the bake-off's own function object, so
    ``read_transform_selection._mean is bake_off._mean`` holds, and nothing
    loads until it is first asked for.
    """
    if name in ('_mean', '_rate', '_cell'):
        return getattr(bake_off(), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def aggregate_unlabeled(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean each metric across *rows*, holding the no-measurement discipline.

    A column that is ``None`` in every row aggregates to ``None``, not to
    ``0.0``; a column with one ``None`` among measurements averages over the
    measurements only.  Delegated to the bake-off's ``_mean`` (:3286).
    """
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys and isinstance(row.get(key), (int, float, type(None))):
                keys.append(key)
    if not keys:
        keys = [
            'claim_recall', 'tokens_per_query', 'topic_diversity',
            'baseline_retention_at_k', 'window_displacement',
        ]
    mean = bake_off()._mean
    return {
        key: mean([row.get(key) for row in rows])
        for key in keys
    }


# ---------------------------------------------------------------------------
# The driver: four arms, two query sets, one replayed cache
# ---------------------------------------------------------------------------
#
# Every arm is a pure function over an ALREADY-FETCHED hit list, which is the
# whole reason the fetch cache exists: one live seeding run pays for the
# embeddings and the ANN search, and every read-side variant after it is free
# and offline forever.  `fetch_arm` used to return its hit lists to a local
# that `run_arm` dropped on the floor; `dump_fetches` catches them at that
# seam.
#
# The transforms are measured over `c_peers` because that is the write shape
# gate ζ RATIFIED.  Measuring them over a shape the repo is not going to
# write would answer a question nobody asked.

#: The ratified write shape.  The read side is being chosen ON TOP of it.
SELECTION_SHAPE = 'c_peers'

#: The baseline first: every column in the artifact is read as a delta
#: against the flat read, so the flat read has to be measured under the
#: identical harness rather than assumed.
ARM_KEYS: tuple[str, ...] = (
    'flat_read',
    'promoting_pin',
    'topic_keyed_grouped',
    'topic_diversity_cap',
)

ARM_LABELS: dict[str, str] = {
    'flat_read': 'flat read (baseline)',
    **{key: spec.label for key, spec in ARM_SPECS.items()},
}


def _records_by_topic(records: list[Any]) -> dict[str, list[Any]]:
    """``topic -> members``, over every record the arm seeded.

    Built from the SEEDED corpus rather than from the window, because a
    topic-keyed group has to be able to name a canonical that did not itself
    rank — that gap is the aliasing the artifact reports.
    """
    grouped: dict[str, list[Any]] = {}
    for record in records:
        topic = _topic_of(record)
        if topic is not None:
            grouped.setdefault(topic, []).append(record)
    return grouped


def apply_arm(
    arm_key: str,
    hits: list[Any],
    seeded: Any,
    k: int,
    *,
    render_sightings: bool = False,
    per_topic: int = 1,
) -> TransformResult:
    """Run one arm's read transform over *hits*, in production's own order.

    Truncated BEFORE the transform and again after, exactly as ``read_path``
    (:3214) does and for its two stated reasons: the store returns k and the
    read side acts on what came back (transforming first would let a record
    the store never returned into the window), and k is the READER's budget
    so the arms are scored over equal-size windows.  Re-deriving that order
    here rather than calling ``read_path`` is deliberate — ``read_path``
    hard-codes the E2 arm variants (``b_grouped`` plus the additive pin) and
    takes no transform argument.
    """
    if arm_key not in ARM_KEYS:
        raise ValueError(f'unknown arm {arm_key!r}; expected one of {ARM_KEYS}')
    records = [hit.record for hit in hits[:k]]

    if arm_key == 'flat_read':
        result = TransformResult(records=list(records), provenance={})
    elif arm_key == 'promoting_pin':
        result = TransformResult(
            records=apply_promoting_topic_anchor(records, seeded.canonical_by_topic),
            provenance={},
        )
    elif arm_key == 'topic_keyed_grouped':
        result = apply_topic_keyed_grouped_read(
            records,
            _records_by_topic(seeded.records),
            render_sightings=render_sightings,
        )
    else: # topic_diversity_cap
        # Wrapped, not returned bare: the cap SYNTHESIZES NOTHING (it is the
        # cheapest family member — no document, no crediting semantics), so
        # it returns a plain list and its provenance is legitimately empty.
        # An empty dict rather than None is the load-bearing distinction:
        # `canonical_discoverability` reads None as "no measurement" and {} as
        # "this arm folded nothing", which is the true fact here.
        result = TransformResult(
            records=apply_topic_diversity_cap(
                records, seeded.canonical_by_topic, per_topic=per_topic,
            ),
            provenance={},
        )

    return TransformResult(records=result.records[:k], provenance=result.provenance)


def score_labeled_query(
    transform: TransformResult,
    query: Any,
    seeded: Any,
    k: int,
    *,
    baseline: list[Any],
    estimator: tuple[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one E2 (labeled) query, with the unlabeled columns filled in too.

    The unlabeled metrics are computed for BOTH sets on purpose: they are the
    only columns that appear in both halves of the artifact, so a reader can
    see whether an arm's token and displacement behaviour on authored queries
    survives contact with real traffic.
    """
    bake = bake_off()
    window = transform.records[:k]
    scored = score_unlabeled_query(
        window, baseline=baseline, k=k, provenance=transform.provenance,
        estimator=estimator,
    )
    scored['unlabeled'] = False
    scored['unlabeled_reason'] = None
    scored['claim_recall'] = bake.claim_recall_at_k(
        window, list(query.expects_claim_ids), k,
    )
    canonical = seeded.canonical_by_topic.get(query.topic)
    if canonical is None:
        # No canonical for this topic in this shape: no measurement, not a
        # zero.  A topic with no canonical is a real fixture state, not a
        # failure of the transform.
        scored['canonical_aliased_in_top_k'] = None
        scored['canonical_unaliased_in_top_k'] = None
        scored['canonical_aliased_rank'] = None
        scored['canonical_unaliased_rank'] = None
        return scored
    verdict = canonical_discoverability(
        # The WINDOW ITSELF, not a `ScoredHit` re-wrap.  `topic_discoverability`
        # — which `canonical_discoverability` delegates the aliased half to —
        # reads `.record_id` off each element, i.e. it consumes `ArmRecord`s.
        # Wrapping them would also have smuggled a score into a rank-based
        # metric, which `ScoredHit` exists to make structurally impossible
        # (bake_off:1346-1355).
        window,
        query.topic,
        canonical.record_id,
        k,
        provenance=transform.provenance,
    )
    scored['canonical_aliased_in_top_k'] = verdict['aliased_in_top_k']
    scored['canonical_unaliased_in_top_k'] = verdict['unaliased_in_top_k']
    scored['canonical_aliased_rank'] = verdict['aliased_rank']
    scored['canonical_unaliased_rank'] = verdict['unaliased_rank']
    return scored


def score_arm(
    arm_key: str,
    queries: list[Any],
    hits_by_query: dict[str, list[Any]],
    seeded: Any,
    *,
    k: int,
    labeled: bool,
    render_sightings: bool = False,
    per_topic: int = 1,
    estimator: tuple[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one arm across one query set, returning per-query rows + means.

    A query the cache does not carry RAISES rather than being skipped: a
    quietly-shorter query set would move every mean in this arm's row and
    still report as a complete measurement.
    """
    rows: list[dict[str, Any]] = []
    for query in queries:
        hits = hits_by_query.get(query.query_id)
        if hits is None:
            raise KeyError(
                f'arm {arm_key!r}: no cached fetch for query {query.query_id!r}. '
                'Refusing to score a silently-shorter query set.'
            )
        baseline = apply_arm('flat_read', hits, seeded, k).records
        transform = apply_arm(
            arm_key, hits, seeded, k,
            render_sightings=render_sightings, per_topic=per_topic,
        )
        if labeled:
            row = score_labeled_query(
                transform, query, seeded, k, baseline=baseline, estimator=estimator,
            )
        else:
            row = score_unlabeled_query(
                transform.records, baseline=baseline, k=k,
                provenance=transform.provenance, estimator=estimator,
            )
        row['query_id'] = query.query_id
        rows.append(row)
    return {
        'arm': arm_key,
        'label': ARM_LABELS[arm_key],
        'k': k,
        'labeled': labeled,
        'query_count': len(rows),
        'rows': rows,
        'means': aggregate_unlabeled(rows),
    }


def run_selection(
    seeded: Any,
    hits_by_query: dict[str, list[Any]],
    e2_queries: list[Any],
    production_queries: list[Any],
    production_hits_by_query: dict[str, list[Any]],
    *,
    e2_k: int = 10,
    production_k: int = PRODUCTION_K,
    render_sightings: bool = False,
    per_topic: int = 1,
    estimator: tuple[str, Any] | None = None,
) -> dict[str, Any]:
    """Score all four arms over BOTH query sets from the replayed cache.

    Two ks on purpose: the E2 set at its own default of 10, so this table
    stays comparable with the committed E2 one, and the production set at 5,
    which is the window production's briefing assembler actually asks for
    (briefing.py:1376).
    """
    if estimator is None:
        estimator = bake_off().resolve_token_estimator()
    return {
        'shape': SELECTION_SHAPE,
        'sighting_crediting': 'rendered' if render_sightings else 'uncredited',
        'per_topic_cap': per_topic,
        'token_estimator': estimator[0],
        'e2': {
            arm: score_arm(
                arm, e2_queries, hits_by_query, seeded,
                k=e2_k, labeled=True, render_sightings=render_sightings,
                per_topic=per_topic, estimator=estimator,
            )
            for arm in ARM_KEYS
        },
        'production': {
            arm: score_arm(
                arm, production_queries, production_hits_by_query, seeded,
                k=production_k, labeled=False, render_sightings=render_sightings,
                per_topic=per_topic, estimator=estimator,
            )
            for arm in ARM_KEYS
        },
    }


# ---------------------------------------------------------------------------
# The artifact: plans/read-transform-selection-report.{json,md}
# ---------------------------------------------------------------------------
#
# A SIBLING pair, deliberately not an extension of
# `plans/e2-storage-shape-bakeoff-report.*`.  That artifact is the record of
# a different experiment (which WRITE shape), its `_check_arms` asserts the
# E2 arm set exactly, and task 3560 is concurrently landing the aliasing
# disclosure into it.  Writing read-side arms there would break its guards
# and collide with 3560.
#
# The E2 module's artifact discipline is reused wholesale, because every part
# of it was paid for: render BEFORE writing (:2316) so a render failure
# leaves both files untouched rather than a stale .md beside a fresh .json;
# `_atomic_write_text` (:2322); `_cell`/`_NO_MEASUREMENT` (:1955/:1824) so a
# None cannot print as 0.00; `fixture_provenance` (:1844) for the commit
# stamps.

from fused_memory.memory_metadata import (  # noqa: E402
    RESERVED_VOCABULARY_KEYS as _LIVE_RESERVED_KEYS,
)

#: The live reserved set, IMPORTED rather than restated.  The V2 disclosure
#: turns on `contested` not being in it; a hard-coded list here would go
#: stale silently the day a sixth key is reserved, and the disclosure would
#: keep asserting an impossibility that had quietly become possible.
RESERVED_VOCABULARY_KEYS = _LIVE_RESERVED_KEYS

#: Quoted verbatim from PRD V2 — the esc-5712 protection.
V2_PROTECTION = 'contested children are never suppressed'

#: Every metric an arm must carry before a row may be rendered.  A ``None``
#: VALUE is a legitimate no-measurement; an ABSENT KEY is a broken run.
REQUIRED_ARM_METRICS: tuple[str, ...] = (
    'claim_recall',
    'tokens_per_query',
    'topic_diversity',
    'baseline_retention_at_k',
    'window_displacement',
    'canonical_aliased_in_top_k',
    'canonical_unaliased_in_top_k',
)

SELECTION_REPORT_SCHEMA = 'read-transform-selection/1'

_PLANS_DIR = _SCRIPTS_DIR.parent.parent / 'plans'

#: The SIBLING pair.  Deliberately not `e2-storage-shape-bakeoff-report.*`:
#: that artifact records a different experiment (which WRITE shape), its
#: `_check_arms` asserts the E2 arm set exactly, and task 3560 is
#: concurrently landing the aliasing disclosure into it.
DEFAULT_SELECTION_JSON = _PLANS_DIR / 'read-transform-selection-report.json'
DEFAULT_SELECTION_MD = _PLANS_DIR / 'read-transform-selection-report.md'


class IncompleteSelectionError(RuntimeError):
    """A partial table must never render as a complete one.

    Mirrors ``IncompleteReportError``/``_check_arms`` (:1887-1920), including
    its reason for reporting a missing and an unknown arm TOGETHER: the
    unknown name is the actionable half, and an error naming only the absence
    sends a reader hunting for an arm that ran perfectly well.
    """


def _check_selection_arms(scored: dict[str, Any]) -> None:
    for half in ('e2', 'production'):
        arms = scored.get(half) or {}
        missing = [arm for arm in ARM_KEYS if arm not in arms]
        unknown = [arm for arm in arms if arm not in ARM_KEYS]
        if missing or unknown:
            problems = []
            if missing:
                problems.append(f'no measurement for {missing}')
            if unknown:
                problems.append(f'unknown arm(s) {unknown}')
            raise IncompleteSelectionError(
                f'{half}: {"; ".join(problems)}. Expected exactly '
                f'{list(ARM_KEYS)}. Refusing to emit a decision table with a '
                f'missing row: a reader cannot tell an absent arm from one '
                f'that scored the same as its neighbour.'
            )
        for arm in ARM_KEYS:
            means = arms[arm].get('means') or {}
            for metric in REQUIRED_ARM_METRICS:
                if metric not in means:
                    raise IncompleteSelectionError(
                        f"{half}: arm '{arm}' is missing metric '{metric}'. "
                        'A None VALUE is a measured no-measurement and is '
                        'fine; an absent KEY means the run did not produce '
                        'it, which is not the same fact.'
                    )


def _rank_key(arm: str, e2: dict[str, Any]) -> tuple:
    """Deterministic ordering over MEASURED columns only.

    No threshold, no tuning knob: the rule is stated in the artifact and
    applied as written (PRD gate G6/D10 — a column is never moved to make a
    recommendation come out).  A ``None`` sorts LAST rather than as a zero.
    """
    def worst_first(value: Any, *, higher_is_better: bool) -> tuple[int, float]:
        if value is None:
            return (1, 0.0)
        return (0, -float(value) if higher_is_better else float(value))

    return (
        worst_first(e2.get('canonical_unaliased_in_top_k'), higher_is_better=True),
        worst_first(e2.get('claim_recall'), higher_is_better=True),
        worst_first(e2.get('tokens_per_query'), higher_is_better=False),
        worst_first(e2.get('window_displacement'), higher_is_better=False),
        arm,
    )


def _recommend(arms: dict[str, Any]) -> dict[str, Any]:
    """Which read transform should 3111 land?

    LANDABILITY FIRST, then measurement.  An arm that would need the
    ``contested`` key to satisfy PRD V2 cannot ship the esc-5712 protection
    today at any score, because the key has no writer and no adjudication
    surface — so it is excluded from the recommendation and the exclusion is
    printed, rather than being silently outranked.
    """
    eligible = [
        arm for arm in ARM_KEYS
        if not arms[arm]['spec']['requires_contested_key_for_v2']
    ]
    excluded = [arm for arm in ARM_KEYS if arm not in eligible]
    winner = min(eligible, key=lambda arm: _rank_key(arm, arms[arm]['e2']))
    return {
        'arm': winner,
        'label': ARM_LABELS[winner],
        'eligible': eligible,
        'excluded_pending_contested_key': excluded,
        'rule': (
            'Landability first: any arm that would need a `contested` key to '
            'satisfy PRD V2 is excluded outright, because that key does not '
            'exist. Among the rest, rank by UNALIASED canonical '
            'discoverability, then claim recall, then tokens/query (lower '
            'better), then window displacement (lower better). A None sorts '
            'last, never as a zero. The rule is fixed before the numbers are '
            'read and is not re-tuned to move a column.'
        ),
        'rationale': (
            f'{ARM_LABELS[winner]} is the highest-ranked arm that task 3111 '
            f'can actually land today: it satisfies the ordering above while '
            f'requiring no vocabulary key that has no writer. Arms excluded '
            f'pending a `contested` key: '
            f'{[ARM_LABELS[a] for a in excluded] or "none"}.'
        ),
    }


def build_selection_report(scored: dict[str, Any]) -> dict[str, Any]:
    """Assemble the decision record, or refuse.

    Pure over *scored*; every disclosure is derived from the arm specs and
    the live vocabulary set rather than restated as prose constants, so the
    document cannot drift from the code it describes.
    """
    from dataclasses import asdict  # noqa: PLC0415

    _check_selection_arms(scored)
    arms = {
        arm: {
            'label': ARM_LABELS[arm],
            'spec': (
                asdict(ARM_SPECS[arm]) if arm in ARM_SPECS
                else {
                    'key': arm,
                    'label': ARM_LABELS[arm],
                    'drops_ranked_records': False,
                    'requires_contested_key_for_v2': False,
                    'displaces_at_window_edge': False,
                }
            ),
            'e2': dict(scored['e2'][arm]['means']),
            'production': dict(scored['production'][arm]['means']),
            'e2_query_count': scored['e2'][arm].get('query_count'),
            'production_query_count': scored['production'][arm].get('query_count'),
        }
        for arm in ARM_KEYS
    }
    return {
        'schema': SELECTION_REPORT_SCHEMA,
        'shape': scored.get('shape', SELECTION_SHAPE),
        'sighting_crediting': scored.get('sighting_crediting', 'uncredited'),
        'per_topic_cap': scored.get('per_topic_cap', 1),
        'token_estimator': scored.get('token_estimator'),
        'e2_k': scored['e2'][ARM_KEYS[0]].get('k'),
        'production_k': scored['production'][ARM_KEYS[0]].get('k'),
        'production_queries': scored.get('production_queries') or {},
        'arms': arms,
        'recommendation': _recommend(arms),
        'reserved_vocabulary_keys': sorted(RESERVED_VOCABULARY_KEYS),
        'v2_protection': V2_PROTECTION,
        'contested_is_a_fixture_field': True,
        'unlabeled_reason': _UNLABELED_REASON,
    }


def _pair_cell(aliased: Any, unaliased: Any) -> str:
    """The canonical column, rendered so a number cannot be read bare.

    The two rates are printed in ONE cell, each carrying its own word, so
    that any line quoting a canonical rate necessarily also says which
    semantics produced it.  Splitting them into two numeric columns would
    let a rate be copied out of the table with the aliasing silently
    stripped off — which is exactly how b_grouped's 0.97 came to be read as
    a retrieval property.
    """
    cell = bake_off()._cell
    return f'{cell(aliased)} (aliased) / {cell(unaliased)} (unaliased)'


def render_selection_markdown(report: dict[str, Any]) -> str:
    """Render the decision document for whoever lands task 3111.

    ONE table, one row per arm, carrying both query sets — not one table per
    set — so an arm cannot be read in isolation from its cost on real
    traffic.
    """
    bake = bake_off()
    cell = bake._cell
    dash = bake._NO_MEASUREMENT
    arms = report['arms']
    rec = report['recommendation']
    keys = ', '.join(f'`{k}`' for k in report['reserved_vocabulary_keys'])

    lines: list[str] = []
    add = lines.append

    add('# Read-transform selection over the ratified C write shape')
    add('')
    add(f'**Write shape:** `{report["shape"]}` (ratified by E2 gate ζ). '
        'This document does not re-litigate the write shape; it chooses the '
        'READ transform layered on top of it.')
    add(f'**Token estimator:** `{report["token_estimator"]}` — the same one '
        'the committed E2 table resolved, so the token columns are directly '
        'comparable across the two artifacts.')
    add(f'**Windows:** authored set at k={report["e2_k"]}; production set at '
        f'k={report["production_k"]}, because the briefing assembler fires at '
        '`limit=5` (`orchestrator/src/orchestrator/agents/briefing.py`:1376) '
        'and a wider window would measure a read no production caller gets.')
    add('')
    add('## The decision table')
    add('')
    add(f'A `{dash}` cell is **no measurement**, never a measured zero.')
    add('')
    add('| arm | claim recall | canonical in top-k | tokens/query | topic '
        'diversity | baseline retention | displacement | prod tokens/query | '
        'prod displacement | drops ranked records | needs `contested` for V2 |')
    add('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for arm in ARM_KEYS:
        block = arms[arm]
        e2, prod, spec = block['e2'], block['production'], block['spec']
        add(
            f'| {block["label"]} '
            f'| {cell(e2.get("claim_recall"))} '
            f'| {_pair_cell(e2.get("canonical_aliased_in_top_k"), e2.get("canonical_unaliased_in_top_k"))} '
            f'| {cell(e2.get("tokens_per_query"))} '
            f'| {cell(e2.get("topic_diversity"))} '
            f'| {cell(e2.get("baseline_retention_at_k"))} '
            f'| {cell(e2.get("window_displacement"))} '
            f'| {cell(prod.get("tokens_per_query"))} '
            f'| {cell(prod.get("window_displacement"))} '
            f'| {"yes" if spec["drops_ranked_records"] else "no"} '
            f'| {"yes" if spec["requires_contested_key_for_v2"] else "no"} |'
        )
    add('')
    add(f'The production half carries no claim-recall or canonical column at '
        f'all: those cells would be `{dash}` for every arm. '
        f'{report["unlabeled_reason"]}')
    add('')

    # --- (a) ---------------------------------------------------------
    add('## Disclosure (a): the `record_id` aliasing')
    add('')
    add('A grouped read emits its document under the **canonical\'s own** '
        '`record_id` (`bake_off_storage_shape.py`:934-935). Any metric that '
        'credits the canonical on `hit.record_id == canonical_record_id` '
        'therefore scores a fold as "canonical found" — whether or not the '
        'canonical\'s own stored record ever ranked. That is a property of '
        'the TRANSFORM, not of retrieval.')
    add('')
    add('This is the mechanism behind the committed E2 table\'s '
        '`b_grouped` canonical-in-top-5 of 0.97: it is an aliased rate, and '
        'reading it as "retrieval finds the canonical 97% of the time" is '
        'not what was measured. Both rates are printed above, in one cell '
        'each, so neither can be quoted without its semantics.')
    add('')
    add('* **aliased** — the legacy `record_id`-match semantics, preserved '
        'so this table stays comparable with the committed one.')
    add('* **unaliased** — the canonical\'s OWN stored record actually '
        'ranked, read from the transform\'s emitted provenance.')
    add('')

    # --- (b) ---------------------------------------------------------
    add('## Disclosure (b): the sighting-crediting knob')
    add('')
    add(f'**Setting for this run: `{report["sighting_crediting"]}`.**')
    add('')
    add('In the landed grouped read this is not a knob at all but a '
        'hard-coded policy (`bake_off_storage_shape.py`:927): a sighting is '
        'collapsed to a bare count, its body is not rendered, and its claims '
        'are not credited. Arm (2) exposes it as a dial.')
    add('')
    add('The consequence is arithmetic, not a verdict on any transform: with '
        'sightings uncredited the claim-recall **ceiling** is exactly '
        '`(claims - sightings + contested) / claims`. An arm cannot exceed '
        'that ceiling however well it retrieves, so a recall column read '
        'without the knob setting beside it is not interpretable.')
    add('')

    # --- (c) ---------------------------------------------------------
    add('## Disclosure (c): which arms suppress')
    add('')
    add('"Suppressing read" is **two independent facts**, kept in two '
        'columns above because collapsing them would mislead task 3111:')
    add('')
    for arm in ARM_KEYS:
        spec = arms[arm]['spec']
        bits = []
        bits.append('drops ranked records from the window'
                    if spec['drops_ranked_records']
                    else 'drops no ranked record')
        if spec['displaces_at_window_edge']:
            bits.append('displaces the k-th record at an already-full window')
        bits.append('would need a `contested` key for PRD V2'
                    if spec['requires_contested_key_for_v2']
                    else 'needs no `contested` key')
        add(f'* **{arms[arm]["label"]}** — ' + '; '.join(bits) + '.')
    add('')
    add('An arm can suppress and still be landable today (the '
        'topic-diversity cap drops records, yet computes its cap from '
        '`metadata[\'topic\']` alone and never asks whether a record is '
        'contested). Reporting one merged "suppressing" boolean would have '
        'said the cap is blocked on a key that does not exist.')
    add('')

    # --- (d) ---------------------------------------------------------
    add('## Disclosure (d): PRD V2\'s protection is unimplementable today')
    add('')
    add(f'PRD V2 requires that **{report["v2_protection"]}** — the esc-5712 '
        'protection. That protection cannot be implemented at all right now, '
        'for any arm, and this is a precondition on task 3111 rather than a '
        'property of any transform measured here.')
    add('')
    add('`contested` is a hand-labelled **fixture** field of the bake-off '
        '(`ArmClaim.contested`:196, `contested_record_ids`:2567, '
        '`SeededArm.contested_ids`:2593). It never appears in any '
        '`ArmRecord.metadata`, it has **no writer**, and it has no '
        'adjudication surface anywhere in the running system.')
    add('')
    add('The live reserved vocabulary is '
        '`RESERVED_VOCABULARY_KEYS` '
        '(`fused-memory/src/fused_memory/memory_metadata.py`:601), and it is '
        f'exactly {{{keys}}} — verified against the imported frozenset, not '
        'transcribed. `contested` is not among them. Any arm that suppresses '
        'records therefore ships without the esc-5712 protection until a '
        '`contested` key is designed, reserved, written and adjudicated.')
    add('')

    # --- recommendation ----------------------------------------------
    add('## Recommendation')
    add('')
    add(f'**Task 3111 should land: {rec["label"]}.**')
    add('')
    add(f'*Selection rule (fixed before the numbers were read):* {rec["rule"]}')
    add('')
    add(rec['rationale'])
    add('')
    if rec['excluded_pending_contested_key']:
        excluded = ', '.join(
            ARM_LABELS[a] for a in rec['excluded_pending_contested_key']
        )
        add(f'Excluded outright, at any score: **{excluded}** — see '
            'disclosure (d). This is an exclusion on landability, not a '
            'measurement verdict; the arm\'s row above is still reported in '
            'full so a later reader with a `contested` key can re-decide.')
        add('')

    # --- the production set ------------------------------------------
    production = report.get('production_queries') or {}
    if production.get('templates'):
        add('## The production query set')
        add('')
        add('Sampled READ-ONLY from the reconciliation write journal '
            '(`fused-memory/scripts/harvest_production_queries.py`, '
            '`mode=ro` + `PRAGMA query_only`). The four briefing-assembler '
            'templates below are the high-volume core; the shares are '
            'measured counts over that journal, not estimates.')
        add('')
        add('| query template | match | observed | share of all search traffic |')
        add('| --- | --- | --- | --- |')
        for entry in production['templates']:
            share = entry.get('traffic_share')
            add(f'| `{entry["template"]}` '
                f'| {entry.get("match") or dash} '
                f'| {entry.get("observed_count", dash):,} '
                f'| {dash if share is None else f"{share * 100:.2f}%"} |')
        add('')
        family = production.get('family_share')
        if family is not None:
            add(f'Together the four templates are **{family * 100:.2f}%** of '
                'all search traffic in the journal, so an arm\'s cost on them '
                'is not a corner case — it is the modal read. The rest of '
                'the sample is a bounded deterministic tail draw from the '
                'long tail of one-off queries.')
            add('')
        add('These queries fire at `limit=5` in production '
            '(`briefing.py`:1376), which is the window the production half '
            'of the table above is scored at.')
        add('')

    # --- the honest tail ---------------------------------------------
    add('## Why the production columns carry no labels')
    add('')
    add('The production query set is sampled read-only from the '
        'reconciliation write journal '
        '(`fused-memory/scripts/harvest_production_queries.py`). It is '
        '**unlabeled by construction**: the journal records which query was '
        'issued, never which memory should have come back. Claim recall and '
        'canonical discoverability are not computable over it, so they are '
        'reported as no-measurement rather than as a zero, and no arm is '
        'credited or penalised for them on this set. Inventing a label here '
        'would fabricate the very ground truth the measurement exists to '
        'establish.')
    return '\n'.join(lines) + '\n'


def write_artifacts(
    report: dict[str, Any],
    json_path: str | Path,
    md_path: str | Path,
    *,
    fixture_paths: list[str | Path] | None = None,
) -> tuple[Path, Path]:
    """Render FIRST, then write both files atomically.

    The order is the E2 module's (:2316) and is load-bearing: a renderer that
    raises must leave a stale ``.md`` beside a stale ``.json``, not a fresh
    JSON beside prose describing the previous run.
    """
    bake = bake_off()
    stamped = dict(report)
    if fixture_paths:
        stamped['fixture_provenance'] = bake.fixture_provenance(list(fixture_paths))
    markdown = render_selection_markdown(stamped)
    import json  # noqa: PLC0415

    body = json.dumps(stamped, indent=2, sort_keys=True, ensure_ascii=False) + '\n'
    json_target, md_target = Path(json_path), Path(md_path)
    bake._atomic_write_text(json_target, body)
    bake._atomic_write_text(md_target, markdown)
    return json_target, md_target


# ---------------------------------------------------------------------------
# The driver: materialize offline, replay the cache, score, emit
# ---------------------------------------------------------------------------
#
# Everything below the fetch is pure, which is the whole point of the cache:
# the ONLY step that needs Qdrant or an embedder is
# `fetch_production_rankings`, run once to extend the committed cache with
# the production query text.  Re-deriving this table afterwards — by task
# 3111, or by anyone re-checking it — needs neither.


PRODUCTION_CACHE_KEY = 'production'
DEFAULT_FETCH_CACHE = (
    _SCRIPTS_DIR.parent / 'tests' / 'fixtures' / 'e2_fetch_cache.json'
)
DEFAULT_PRODUCTION_QUERIES = (
    _SCRIPTS_DIR.parent / 'tests' / 'fixtures' / 'production_query_sample.jsonl'
)


class ProductionFetchError(RuntimeError):
    """The production half of the cache is absent, stale or truncated."""


def production_query_provenance(
    queries: list[ProductionQuery],
    *,
    harvest_sidecar: str | Path | None = None,
) -> dict[str, Any]:
    """The traffic shares this table's external-validity claim rests on.

    Copied FROM the committed sample rather than recomputed, so the artifact
    and the fixture cannot disagree: if the sample is re-harvested and the
    report is not regenerated, the committed-artifact test says so instead of
    the table quietly describing a corpus that is no longer in the tree.
    """
    import json  # noqa: PLC0415

    templates = [
        {
            'template': query.template,
            'text': query.text,
            'match': query.match,
            'observed_count': query.observed_count,
            'traffic_share': query.traffic_share,
        }
        for query in queries
        if query.source == 'briefing_template' and query.template
    ]
    block: dict[str, Any] = {
        'templates': templates,
        'family_share': round(
            sum(entry['traffic_share'] for entry in templates), 6,
        ) if templates else None,
        'sampled_query_count': len(queries),
        'tail_query_count': len(queries) - len(templates),
        'observed_limit': queries[0].observed_limit if queries else None,
        'unlabeled': True,
        'unlabeled_reason': _UNLABELED_REASON,
    }
    if harvest_sidecar is not None and Path(harvest_sidecar).exists():
        sidecar = json.loads(Path(harvest_sidecar).read_text(encoding='utf-8'))
        block['harvest'] = {
            key: sidecar[key]
            for key in (
                'harvested_at', 'journal_path', 'total_search_ops',
                'literal_share', 'family_share', 'tail_share', 'tail_count',
                'tail_distinct', 'search_limit', 'seed',
            )
            if key in sidecar
        }
    return block


def materialize_selection_arm(
    *,
    shape: str = SELECTION_SHAPE,
    alpha_fixture: str | Path | None = None,
    registry: str | Path | None = None,
    arm_claims: str | Path | None = None,
    distractor_slab: str | Path | None = None,
    project_suffix: str | None = None,
) -> tuple[Any, list[Any]]:
    """Rebuild the seeded arm OFFLINE, plus the E2 query set.

    Deterministic over the committed fixtures — that determinism is exactly
    what lets the fetch cache store record ids and nothing else.
    """
    bake = bake_off()
    clusters = bake.load_calibration_clusters(
        alpha_fixture or bake.DEFAULT_ALPHA_FIXTURE_PATH,
    )
    topics = bake.load_registry_topics(registry or bake.DEFAULT_REGISTRY_PATH)
    claims = bake.load_arm_claims(arm_claims or bake.DEFAULT_ARM_CLAIMS_PATH)
    queries = bake.load_query_set(bake.DEFAULT_QUERY_SET_PATH)
    distractors = bake.load_distractor_slab(
        distractor_slab or bake.DEFAULT_DISTRACTOR_SLAB_PATH,
    )
    bake.cross_validate_fixtures(
        clusters=clusters, topics=topics, claims=claims, queries=queries,
    )
    records = bake.materialize_arm(shape, clusters, claims, topics, distractors)
    seeded = bake._index_arm(
        shape,
        bake.arm_project_id(shape, suffix=project_suffix),
        bake.ephemeral_collections(suffix=project_suffix)[shape],
        records,
        claims,
    )
    return seeded, queries


def load_production_fetches(
    path: str | Path, seeded: Any, *, expect_query_ids: list[str] | None = None,
) -> dict[str, list[Any]]:
    """Rehydrate the production rankings, or refuse by name.

    The refusals are the bake-off's own — a cached record this run did not
    materialize raises through ``_rehydrate_ranking``, and the corpus
    fingerprint is checked here — because a production half replayed against
    a drifted corpus is exactly as unpublishable as an E2 half.
    """
    import json  # noqa: PLC0415

    bake = bake_off()
    target = Path(path)
    try:
        doc = json.loads(target.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise ProductionFetchError(
            f'no fetch cache at {target}. Produce it with --dump-fetches on a '
            f'live bake-off run, then extend it with --extend-cache-live.'
        ) from exc
    block = doc.get(PRODUCTION_CACHE_KEY)
    if not isinstance(block, dict) or seeded.shape not in block:
        raise ProductionFetchError(
            f'fetch cache at {target} carries no production rankings for shape '
            f'{seeded.shape!r}. The E2 cache cannot cover them: production '
            f'queries are new text and need their own embed+ANN pass. Run '
            f'`--extend-cache-live` once.'
        )
    shape_block = block[seeded.shape]
    fingerprint = bake.corpus_fingerprint(seeded.records)
    cached_fingerprint = shape_block.get('corpus_fingerprint')
    if cached_fingerprint != fingerprint:
        raise ProductionFetchError(
            f'fetch cache at {target}: the production rankings for '
            f'{seeded.shape!r} were fetched against corpus '
            f'{cached_fingerprint!r}, but this run materialized '
            f'{fingerprint!r}. The cache and the fixtures describe different '
            f'corpora; re-fetch rather than publishing a stale ranking as a '
            f'fresh measurement.'
        )
    cached = shape_block.get('queries') or {}
    if expect_query_ids is not None:
        missing = [q for q in expect_query_ids if q not in cached]
        if missing:
            raise ProductionFetchError(
                f'fetch cache at {target} is TRUNCATED for the production set: '
                f'{len(missing)} of {len(expect_query_ids)} queries have no '
                f'cached ranking, e.g. {missing[:5]}. Scoring the arm anyway '
                f'would measure a subset while the report claimed the whole '
                f'set.'
            )
    return {
        query_id: bake._rehydrate_ranking(
            seeded, entries, shape=seeded.shape, kind='production', key=query_id,
        )
        for query_id, entries in sorted(cached.items())
    }


async def fetch_production_rankings(
    production_queries: list[ProductionQuery],
    *,
    shape: str = SELECTION_SHAPE,
    limit: int | None = None,
    project_suffix: str | None = None,
    seed_concurrency: int | None = None,
) -> dict[str, Any]:
    """The ONE live pass this script makes: seed the arm, search, tear down.

    A deliberately narrow sibling of ``run_bake_off``'s driver — one shape,
    no probes, no measurement — carrying the same three non-negotiables,
    each of which exists because of a specific way this can go wrong:

    * the durable write queue is repointed at a per-run temp dir, so this
      never recovers the live server's in-flight rows into a collection it is
      about to destroy (``run_bake_off``:3934-3954 states that failure in
      full);
    * the pinned embedder key is cleared so mem0 falls back to the real
      ``OPENAI_API_KEY`` — a stub constant vector would make every ranking
      here meaningless;
    * teardown runs in a ``finally``, because the leak that matters is the
      mid-run one.
    """
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
    from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

    bake = bake_off()
    limit = bake.DEFAULT_SEARCH_LIMIT if limit is None else limit
    seed_concurrency = (
        bake.SEED_CONCURRENCY if seed_concurrency is None else seed_concurrency
    )
    seeded, _ = materialize_selection_arm(
        shape=shape, project_suffix=project_suffix,
    )
    collections = bake.ephemeral_collections(suffix=project_suffix)
    collection = collections[shape]

    config = FusedMemoryConfig().model_copy(deep=True)
    config.mem0.collection_prefix = bake.ephemeral_collection_prefix()
    openai_provider = getattr(config.embedder.providers, 'openai', None)
    if openai_provider is not None:
        openai_provider.api_key = None
    qdrant_url = config.mem0.qdrant_url
    queue_dir = tempfile.mkdtemp(prefix='read-transform-queue-')
    config.queue.data_dir = queue_dir

    memory = MemoryService(config)
    try:
        bake.drop_collections([collection], qdrant_url=qdrant_url)
        await memory.initialize()
        await bake.seed_arm(memory.mem0, seeded, concurrency=seed_concurrency)
        fetched = await bake.fetch_arm(
            memory.mem0, seeded, list(production_queries), [], limit=limit,
        )
    finally:
        await memory.close()
        bake.drop_collections([collection], qdrant_url=qdrant_url)
        shutil.rmtree(queue_dir, ignore_errors=True)

    return {
        'shape': shape,
        'corpus_fingerprint': bake.corpus_fingerprint(seeded.records),
        'search_limit': int(limit),
        'embedder_model': config.embedder.model,
        'queries': {
            query_id: bake._serialize_ranking(hits)
            for query_id, hits in sorted(fetched['queries'].items())
        },
    }


def extend_fetch_cache(path: str | Path, fetched: dict[str, Any]) -> Path:
    """Merge the production rankings into the committed cache, byte-stably.

    A SEPARATE top-level key rather than more entries under ``arms``:
    ``load_fetches`` enforces that every E2 query has a ranking, and folding
    unlabeled production queries in beside them would make that guard pass
    for the wrong reason — and would put label-free rows into the block the
    E2 replay treats as its labeled query set.
    """
    import json  # noqa: PLC0415

    bake = bake_off()
    target = Path(path)
    doc = json.loads(target.read_text(encoding='utf-8'))
    block = doc.setdefault(PRODUCTION_CACHE_KEY, {})
    block[fetched['shape']] = {
        'corpus_fingerprint': fetched['corpus_fingerprint'],
        'search_limit': fetched['search_limit'],
        'embedder_model': fetched['embedder_model'],
        'queries': fetched['queries'],
    }
    bake._atomic_write_text(
        target, json.dumps(doc, indent=2, sort_keys=True) + '\n',
    )
    return target


def score_from_cache(
    *,
    fetch_cache: str | Path = DEFAULT_FETCH_CACHE,
    production_queries: str | Path = DEFAULT_PRODUCTION_QUERIES,
    e2_k: int = 10,
    production_k: int = PRODUCTION_K,
    render_sightings: bool = False,
    per_topic: int = 1,
) -> dict[str, Any]:
    """Score all four arms over both query sets, entirely offline."""
    bake = bake_off()
    seeded, e2_queries = materialize_selection_arm()
    replayed = bake.load_fetches(
        fetch_cache,
        {seeded.shape: seeded},
        expect_query_ids=[query.query_id for query in e2_queries],
        expect_limit=e2_k,
    )
    prod_queries = load_production_queries(production_queries)
    prod_hits = load_production_fetches(
        fetch_cache, seeded,
        expect_query_ids=[query.query_id for query in prod_queries],
    )
    scored = run_selection(
        seeded,
        replayed[seeded.shape]['queries'],
        e2_queries,
        prod_queries,
        prod_hits,
        e2_k=e2_k,
        production_k=production_k,
        render_sightings=render_sightings,
        per_topic=per_topic,
    )
    scored['production_queries'] = production_query_provenance(
        prod_queries,
        harvest_sidecar=Path(production_queries).with_suffix(
            '.provenance.json',
        ),
    )
    return scored


def build_parser() -> Any:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description=(
            'Read-transform selection over the ratified C write shape: score '
            'three candidate read transforms against the flat-read baseline '
            'and emit the decision table task 3111 reads.'
        ),
    )
    parser.add_argument('--fetch-cache', default=str(DEFAULT_FETCH_CACHE),
                        help='replayed ranking cache (from --dump-fetches)')
    parser.add_argument('--production-queries',
                        default=str(DEFAULT_PRODUCTION_QUERIES),
                        help='harvested production query sample (JSONL)')
    parser.add_argument('--json-out', default=str(DEFAULT_SELECTION_JSON),
                        help='where to write the machine-readable report')
    parser.add_argument('--md-out', default=str(DEFAULT_SELECTION_MD),
                        help='where to write the operator-facing report')
    parser.add_argument('--k', type=int, default=10,
                        help='reader window for the authored E2 query set')
    parser.add_argument('--production-k', type=int, default=PRODUCTION_K,
                        help="reader window for the production set "
                             "(production's own limit is 5)")
    parser.add_argument('--render-sightings', action='store_true',
                        help='credit sightings by rendering their bodies')
    parser.add_argument('--per-topic', type=int, default=1,
                        help='records per topic the diversity cap keeps')
    parser.add_argument('--extend-cache-live', action='store_true',
                        help='ONE live pass: seed the arm and fetch the '
                             'production queries into the cache, then exit')
    parser.add_argument('--project-suffix', default=None,
                        help='override the ephemeral project id suffix')
    return parser


def main(argv: list[str] | None = None) -> int:
    import asyncio  # noqa: PLC0415
    import sys  # noqa: PLC0415

    args = build_parser().parse_args(argv)

    if args.extend_cache_live:
        queries = load_production_queries(args.production_queries)
        fetched = asyncio.run(fetch_production_rankings(
            queries, project_suffix=args.project_suffix, limit=args.k,
        ))
        target = extend_fetch_cache(args.fetch_cache, fetched)
        print(f'extended {target} with {len(fetched["queries"])} production '
              f'rankings at limit {fetched["search_limit"]}')
        return 0

    try:
        scored = score_from_cache(
            fetch_cache=args.fetch_cache,
            production_queries=args.production_queries,
            e2_k=args.k,
            production_k=args.production_k,
            render_sightings=args.render_sightings,
            per_topic=args.per_topic,
        )
        report = build_selection_report(scored)
        json_path, md_path = write_artifacts(
            report, args.json_out, args.md_out,
            fixture_paths=[
                bake_off().DEFAULT_ALPHA_FIXTURE_PATH,
                bake_off().DEFAULT_REGISTRY_PATH,
                bake_off().DEFAULT_ARM_CLAIMS_PATH,
                bake_off().DEFAULT_QUERY_SET_PATH,
                bake_off().DEFAULT_DISTRACTOR_SLAB_PATH,
                Path(args.production_queries),
                Path(args.fetch_cache),
            ],
        )
    except (
        ProductionFetchError, ProductionQuerySetError,
        IncompleteSelectionError, bake_off().FetchCacheError,
        bake_off().FixtureError,
    ) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1

    print(f'wrote {json_path}')
    print(f'wrote {md_path}')
    print(f'recommendation: {report["recommendation"]["label"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
