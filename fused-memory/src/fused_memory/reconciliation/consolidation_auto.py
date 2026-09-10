"""Deterministic auto-consolidation: the pure predicate and the canonical builder.

Task 5237. Implements PRD ``plans/memory-auto-consolidation-prd.md`` contract
C2 (:func:`evaluate_auto_predicate` — rules on whether a proposed near-duplicate
cluster may be consolidated with no human in the loop) and contract C3
(:func:`build_auto_canonical` — the ONE home of auto-consolidated canonical
text).

Import-leaf rule
----------------
This module imports **only the standard library**, and must keep doing so.
:class:`~fused_memory.config.schema.ConsolidationAutoConfig` is referenced under
``TYPE_CHECKING`` only, so importing this module never loads config, pydantic,
yaml or the mem0 SDK. That is a STRICTLY STRONGER property than its sibling leaf
``reconciliation/consolidation_gate.py`` holds, and it is deliberate: PRD D4
records a MEASURED hard import cycle from a careless import of exactly this kind
(``config/schema.py`` -> ``memory_metadata`` -> ``backends.mem0_client`` ->
``config.schema``, raising ``ImportError: cannot import name
'FusedMemoryConfig'``). ``tests/test_consolidation_auto.py`` pins the property in
a fresh interpreter; if you add a runtime ``fused_memory`` import here, this
module's import cost becomes a whole config build and the cycle above is one
edit away.

Two properties a reader must not get wrong
------------------------------------------
1. **The predicate NEVER reads a store.** Every fact it rules on — each member
   record, the count of canonicals for the topic, the open gate, the existing
   canonical slugs — is SUPPLIED by the caller. There is no service parameter
   and no search. A predicate that could read would be a predicate whose verdict
   depended on when it ran.

2. **The correction-banner arm is REFUSAL-ONLY.** A match adds a FAIL. A
   non-match certifies NOTHING: per PRD D14 an incumbent that is stale but
   carries no banner is undetectable by code, and that case stays with the human
   sitting. Never read "no banner matched" as "this record is current".
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - a runtime import here would cost the leaf
    from fused_memory.config.schema import ConsolidationAutoConfig

__all__ = [
    'CORRECTION_BANNER_RE',
    'CORRECTION_METADATA_KEYS',
    'UNREADABLE',
    'AutoOutcome',
    'AutoProposal',
    'AutoReason',
    'AutoReasonCode',
    'AutoVerdict',
    'build_auto_canonical',
    'evaluate_auto_predicate',
]


class AutoOutcome(StrEnum):
    """What the caller may do with a proposal.

    ``PASS`` mints a canonical and folds the members. ``PASS_TAG_ONLY`` tags
    unstamped members onto a topic that already has a canonical, touching that
    canonical's content and metadata not at all. ``NOOP`` means there is
    nothing to do (already consolidated, or a human already owns the topic).
    ``FAIL`` means a hazard fired and the cluster goes to a human.
    """

    PASS = 'pass'
    PASS_TAG_ONLY = 'pass_tag_only'
    NOOP = 'noop'
    FAIL = 'fail'


class AutoReasonCode(StrEnum):
    """Every reason the predicate can give, as a closed vocabulary.

    A closed enum rather than free prose so a caller SERIALISES ``code.value``
    and branches on it, instead of parsing a sentence (PRD §6's structured-data
    heuristic). ``tests/test_consolidation_auto.py`` pins that every member here
    has a fixture that actually produces it, so a code can never ship
    unreachable.
    """

    already_gated = 'already_gated'
    member_unreadable = 'member_unreadable'
    member_not_found = 'member_not_found'
    member_already_canonical = 'member_already_canonical'
    member_different_topic = 'member_different_topic'
    member_carries_correction_metadata = 'member_carries_correction_metadata'
    member_carries_correction_banner = 'member_carries_correction_banner'
    canonical_carries_correction = 'canonical_carries_correction'
    mixed_category = 'mixed_category'
    canonical_category_mismatch = 'canonical_category_mismatch'
    multiple_canonicals = 'multiple_canonicals'
    canonical_count_unavailable = 'canonical_count_unavailable'
    canonical_count_contradicted = 'canonical_count_contradicted'
    slug_near_collision = 'slug_near_collision'
    incumbent_canonical_stripped = 'incumbent_canonical_stripped'
    already_consolidated = 'already_consolidated'
    no_retained_members = 'no_retained_members'


@dataclass(frozen=True)
class AutoReason:
    """One reason, naming the records it is about.

    ``ids`` carries the memory ids (or, for ``already_gated``, the gate task id)
    the reason concerns, so a caller reports WHICH record tripped a rule without
    re-deriving it from ``detail``. ``detail`` is human-facing only and is never
    a parse target.
    """

    code: AutoReasonCode
    ids: tuple[str, ...] = ()
    detail: str = ''


@dataclass(frozen=True)
class AutoVerdict:
    """The predicate's whole answer.

    ``retain_ids`` are the members the caller may act on, in the proposal's
    order. ``stripped_ids`` are members the predicate REMOVED from that set and
    is disclosing — today only an incumbent canonical of the proposal's own
    topic. Disclosure is the point: a silently shortened member list would let
    the executor skip a record the proposal named with nothing recording why.
    """

    outcome: AutoOutcome
    reasons: tuple[AutoReason, ...]
    retain_ids: tuple[str, ...]
    stripped_ids: tuple[str, ...]
    predicate_version: str


@dataclass(frozen=True)
class AutoProposal:
    """A cluster put forward for consolidation.

    Structured rather than a blob the predicate parses: the caller builds this
    from a ledger row, so this module contains no JSON handling and no ad-hoc
    parser (PRD §6's structured-data heuristic). The CLAIM is deliberately
    absent — claim shape is the emit boundary's business
    (``server/consolidation.py::validate_consolidate_args``), and the predicate
    re-derives none of it.
    """

    topic: str
    member_ids: tuple[str, ...]
    category: str


@dataclass(frozen=True)
class _Unreadable:
    """Type of the :data:`UNREADABLE` sentinel. Not for instantiation elsewhere."""

    def __repr__(self) -> str:
        return 'UNREADABLE'


#: The read did not answer — distinct from ``None``, which means the record is
#: genuinely not there.
#:
#: Both states must be expressible because
#: ``MemoryService.get_memory_by_id`` distinguishes them: it returns ``None``
#: for a missing record and RAISES ``TimeoutError`` when the backend read fails.
#: A two-state ``dict[id, record | None]`` mapping would collapse a failed read
#: into "not found", and the predicate would then rule on a cluster it could not
#: actually see. Both are FAIL codes, but they are different facts and a human
#: reading the refusal needs to know which one happened.
UNREADABLE = _Unreadable()

#: Metadata keys whose presence marks a record as corrected or superseded.
#:
#: The three PRD C2 keys — ``corrects``, ``superseded_by``,
#: ``stale_pre_fix_state`` — are the CONTRACT CORE. The other three are MEASURED
#: siblings, and they are here because the 5180 canonical the PRD cites as its
#: own motivating specimen carries NONE of the three: its correction metadata is
#: ``corrected_by_run`` / ``corrected_at`` / ``x_corrected_by_sitting``. The
#: dark_factory census (``plans/memory-metadata-census-report.md``) counts
#: ``superseded_by`` 32, ``corrects_memory_id`` 17, ``corrects`` 6,
#: ``stale_pre_fix_state`` 2.
#:
#: None of these is registered vocabulary today — ``memory_metadata`` censuses
#: all of them as unknown keys — which is why they are enumerated here as
#: observed corpus facts rather than imported from a registry that does not
#: carry them. The arm is REFUSAL-ONLY, so widening it can only make the
#: predicate more conservative and can never grant a pass.
CORRECTION_METADATA_KEYS: frozenset[str] = frozenset({
    'corrects',
    'superseded_by',
    'stale_pre_fix_state',
    'corrects_memory_id',
    'corrected_by_run',
    'corrected_at',
})

#: Body text marking a record as corrected, superseded or retracted.
#:
#: Two alternatives, searched over the WHOLE body and never ``\A``-anchored: a
#: real specimen (mem0 ``0090d639-c325-490c-a432-866b60a26ba7``) opens with an
#: ordinary paragraph and carries its ``[CORRECTION ...]`` banner mid-body, so
#: an anchored pattern would miss it. The bracketed arm matches the stamp shape
#: wherever it appears; the line-leading arm catches the un-bracketed idiom
#: pinned in-repo as
#: ``scripts/amend_stale_resume_cwd_records.py::_CORRECTED_RECORD_PREIMAGE``
#: (``SUPERSEDED 2026-08-30 (by Stage-1 memory consolidation, task 4610)...``).
#:
#: ``AMENDMENT`` is DELIBERATELY ABSENT. 53 live dark_factory records open
#: ``AMENDMENT to ...``, and that idiom is benign ACCRETION — the read path
#: treats it as a first-class child kind (``server/grouped_read.py``) — so
#: matching it would refuse the majority of real clusters.
#:
#: REFUSAL-ONLY: a match adds a FAIL, a non-match certifies nothing (PRD D14).
CORRECTION_BANNER_RE: re.Pattern[str] = re.compile(
    r'\[\s*(?:CORRECTION|SUPERSEDED|RETRACTED|RETRACTION)\b'
    r'|(?m:^(?:CORRECTION|Correction|SUPERSEDED|CORPUS-HYGIENE WARNING)\b)',
)


def build_auto_canonical(claim: str, topic: str, n: int, run_id: str) -> str:
    """Render the canonical body for an auto-consolidated cluster (PRD C3).

    The ONE home of auto-consolidated canonical text: task delta's executor
    calls this and writes nothing of its own, so there is exactly one shape to
    audit and exactly one place to change it.

    The LLM supplies only *claim*. Its ``rationale`` and ``seeded_from`` fields
    are reviewer-facing and never enter canonical text (PRD D2), and *claim*
    leads the body verbatim because PRD §2 measured that a template canonical
    whose claim leads retrieves within 0.015 cosine of a hand-written one.

    The member list deliberately does NOT appear in the text. The live
    ``metadata.topic`` scroll IS the member list; *run_id* and the ledger row
    are the pointers back to the decision that wrote it. Naming the members
    here would give one fact two homes, and the copy in the text would be the
    one that goes stale (PRD D6).

    There is no runtime length cap, by decision D5: with every input at its own
    cap — a 200-char claim, a 100-char slug, N=20 and a 36-char uuid run id —
    the result is 464 characters, so a cap would be unreachable dead code, and
    dead code that looks like a safety bound is worse than none. The bound is
    asserted by ``tests/test_consolidation_auto.py`` instead.

    No truncation, no conditionals, no defaults: any branch here would be a
    second canonical shape.
    """
    return (
        f'{claim}\n\n'
        f'Index canonical for topic `{topic}` over {n} short peers; the live '
        f'metadata.topic scroll is the member list (auto-consolidated, run '
        f'{run_id}).'
    )


def _metadata(record: object) -> Mapping[str, Any]:
    """The metadata of a member record, as a mapping, whatever the record is.

    The ONE place a member record is destructured. A record may be
    :data:`UNREADABLE`, ``None``, or a mapping whose ``metadata`` is missing or
    itself ``None`` — and none of those may raise inside a function whose whole
    job is to RETURN a verdict. A predicate that throws on a malformed record is
    a predicate the caller cannot report on, which is exactly the case the
    ``member_unreadable`` code exists to make reportable.
    """
    if not isinstance(record, Mapping):
        return {}
    metadata = record.get('metadata')
    return metadata if isinstance(metadata, Mapping) else {}


def _content(record: object) -> str:
    """The body text of a member record, as a string, whatever the record is.

    Same no-raise discipline as :func:`_metadata`: a record with no ``content``,
    or a ``content`` that is not text, scans as empty rather than exploding. An
    empty scan certifies nothing — the banner arm is refusal-only — so failing
    to find a banner in a malformed body costs no safety the arm ever offered.
    """
    if not isinstance(record, Mapping):
        return ''
    content = record.get('content')
    return content if isinstance(content, str) else ''


def _is_incumbent(record: object, topic: str) -> bool:
    """Is *record* the canonical THIS topic already has?

    One home for the definition, because three rules turn on it: the incumbent
    is STRIPPED rather than retained, it is exempt from the member-level
    canonical and banner hazards (one record must not produce two codes for one
    fact), and it is the subject of the canonical-level hazards. A canonical of
    a DIFFERENT topic is not an incumbent — that is the
    ``member_already_canonical`` hazard.
    """
    metadata = _metadata(record)
    return metadata.get('canonical') is True and metadata.get('topic') == topic


def _member_categories(
    proposal: AutoProposal,
    members: Mapping[str, Mapping[str, Any] | None | _Unreadable],
) -> list[tuple[str, str]]:
    """``(member_id, category)`` for every readable NON-incumbent member with one.

    One home, because two rules read it: ``mixed_category`` fires when these
    span more than one value, and ``canonical_category_mismatch`` compares the
    incumbent against the single value they share. Two independent walks would
    let those two rules disagree about which records they were talking about.

    A member with NO category is skipped rather than counted as a distinct one:
    an unstamped record predates the vocabulary rather than contradicting it,
    and refusing on absence would refuse most older clusters.
    """
    pairs: list[tuple[str, str]] = []
    for member_id in proposal.member_ids:
        record = members.get(member_id)
        if record is None or isinstance(record, _Unreadable):
            continue
        if _is_incumbent(record, proposal.topic):
            continue
        category = _metadata(record).get('category')
        if isinstance(category, str):
            pairs.append((member_id, category))
    return pairs


def _slug_jaccard(left: str, right: str) -> float:
    """Token overlap of two topic slugs: ``|A n B| / |A u B|``.

    ``topic_slug.TOPIC_SLUG_RE`` already guarantees a slug is lowercase
    alphanumeric segments joined by single hyphens, so ``str.split('-')`` IS the
    tokenizer. Writing a second one here would be a second rule about what a
    slug is, and the two would drift.
    """
    left_tokens = set(left.split('-'))
    right_tokens = set(right.split('-'))
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _member_hazards(
    proposal: AutoProposal,
    members: Mapping[str, Mapping[str, Any] | None | _Unreadable],
) -> list[AutoReason]:
    """Every member-level refusal, COLLECTED — the member half of rung 2.

    Collecting rather than stopping at the first offender is the contract: one
    human sitting must be able to name every problem, or the same cluster comes
    back once per hazard. That property is structural here — each check appends
    to the same list and nothing returns early.

    An id the caller did not supply a read for reads as ``member_not_found``.
    That is deliberate fail-closed behaviour: the caller's contract is a read
    per proposed id, and a missing key means the predicate cannot see a record
    the proposal named.

    The INCUMBENT is exempt from the two CORRECTION scans, and only those. It is
    not thereby unscrutinised — a corrected incumbent is refused under
    ``canonical_carries_correction``, which is a different fact about a
    different record — but reporting it here as well would make one problem look
    like two.
    """
    reasons: list[AutoReason] = []

    for member_id in proposal.member_ids:
        record = members.get(member_id)

        if isinstance(record, _Unreadable):
            reasons.append(
                AutoReason(
                    code=AutoReasonCode.member_unreadable,
                    ids=(member_id,),
                    detail=(
                        f'the read for {member_id} did not answer; a backend that '
                        'failed to reply is not a record that is fine'
                    ),
                ),
            )
            continue

        if record is None:
            reasons.append(
                AutoReason(
                    code=AutoReasonCode.member_not_found,
                    ids=(member_id,),
                    detail=f'{member_id} was proposed but no such record exists',
                ),
            )
            continue

        metadata = _metadata(record)
        topic = metadata.get('topic')
        foreign_topic = topic is not None and topic != proposal.topic

        if metadata.get('canonical') is True and foreign_topic:
            reasons.append(
                AutoReason(
                    code=AutoReasonCode.member_already_canonical,
                    ids=(member_id,),
                    detail=(
                        f'{member_id} is the canonical of topic `{topic}`, not a '
                        f'member of `{proposal.topic}`; folding it in would destroy '
                        "that topic's index entry"
                    ),
                ),
            )

        if foreign_topic:
            reasons.append(
                AutoReason(
                    code=AutoReasonCode.member_different_topic,
                    ids=(member_id,),
                    detail=(
                        f'{member_id} is already stamped with topic `{topic}`; '
                        f're-stamping it to `{proposal.topic}` would silently move '
                        'it off a scroll nothing else sweeps'
                    ),
                ),
            )

        if not _is_incumbent(record, proposal.topic):
            present = sorted(key for key in CORRECTION_METADATA_KEYS if key in metadata)
            if present:
                reasons.append(
                    AutoReason(
                        code=AutoReasonCode.member_carries_correction_metadata,
                        ids=(member_id,),
                        detail=(
                            f'{member_id} carries correction metadata '
                            f'({", ".join(present)}), so it is superseded rather '
                            'than merely duplicated'
                        ),
                    ),
                )

            if CORRECTION_BANNER_RE.search(_content(record)):
                reasons.append(
                    AutoReason(
                        code=AutoReasonCode.member_carries_correction_banner,
                        ids=(member_id,),
                        detail=(
                            f"{member_id}'s body carries a correction or supersession "
                            'banner; consolidating it would fold a retracted claim '
                            'into a live canonical'
                        ),
                    ),
                )

    categorised = _member_categories(proposal, members)
    distinct = {category for _, category in categorised}
    if len(distinct) > 1:
        reasons.append(
            AutoReason(
                code=AutoReasonCode.mixed_category,
                ids=tuple(member_id for member_id, _ in categorised),
                detail=(
                    'the proposed members span more than one category '
                    f'({", ".join(sorted(distinct))}); one canonical cannot index two'
                ),
            ),
        )

    return reasons


def _canonical_hazards(
    proposal: AutoProposal,
    members: Mapping[str, Mapping[str, Any] | None | _Unreadable],
    canonical_count: int | None,
) -> list[AutoReason]:
    """Refusals about the topic's own canonical — the canonical half of rung 2.

    HONEST BOUNDARY, stated rather than papered over: the incumbent-level codes
    are reportable only when the proposal NAMES the incumbent. C2 gives this
    function per-member reads plus a count and no way to read a record the
    proposal did not name, so a canonical outside the member list is covered by
    *canonical_count* alone. That is not a gap in practice — PRD B2/B4 both put
    the incumbent in the member list, because an LLM re-proposing a topic
    enumerates its canonical — and PRD D14 already assigns the residue, a stale
    incumbent carrying no banner, to the human sitting rather than to code.
    """
    reasons: list[AutoReason] = []

    # ONE home for "who is this topic's incumbent", computed before the count
    # chain because that chain now reads it too. The per-member loop below
    # iterates this tuple rather than re-deriving `_is_incumbent`, so the three
    # rules keyed on the incumbent cannot drift apart (SPOT).
    incumbents = tuple(
        member_id
        for member_id in proposal.member_ids
        if _is_incumbent(members.get(member_id), proposal.topic)
    )

    if canonical_count is None:
        # Fail closed. Zero is the MINT path, so reading "I could not find out"
        # as "there is none" is exactly how a topic gets a second canonical.
        reasons.append(
            AutoReason(
                code=AutoReasonCode.canonical_count_unavailable,
                detail=(
                    f'the number of canonicals for topic `{proposal.topic}` could '
                    'not be determined; an unknown count is not a count of zero'
                ),
            ),
        )
    elif canonical_count > 1:
        # Canonical uniqueness ships in WARN mode (`memory_metadata.enforce` is
        # False, task 3626), so the count is probed rather than assumed.
        reasons.append(
            AutoReason(
                code=AutoReasonCode.multiple_canonicals,
                detail=(
                    f'topic `{proposal.topic}` already has {canonical_count} '
                    'canonicals; which one is authoritative is not decidable here'
                ),
            ),
        )
    elif canonical_count == 0 and incumbents:
        # The same reasoning as the `None` arm, one case over. There the count
        # was missing; here it is present and CONTRADICTED — it says the topic
        # has no canonical while a member the proposal named IS that canonical.
        # Zero is the MINT path, so trusting it would mint a second canonical
        # for a topic whose incumbent is visible in this very member list.
        #
        # Reachable WITHOUT a caller bug: `canonical_count` and the per-member
        # reads are two separate, non-atomic reads of the store. A count scoped
        # differently, or taken before a canonical landed, disagrees with a
        # record the caller can nonetheless see. Fail closed for the reason the
        # `None` arm does — a count that cannot be trusted must never be read as
        # zero, and here it demonstrably cannot be.
        named = ', '.join(incumbents)
        reasons.append(
            AutoReason(
                code=AutoReasonCode.canonical_count_contradicted,
                ids=incumbents,
                detail=(
                    f'topic `{proposal.topic}` is reported to have no canonical, but '
                    f'{named} is stamped as its canonical; the count and the '
                    'per-member reads disagree, so the canonical state of the topic '
                    'is not decidable here'
                ),
            ),
        )

    categories = {category for _, category in _member_categories(proposal, members)}
    sole_category = next(iter(categories)) if len(categories) == 1 else None

    # Only an incumbent of THIS topic contradicts THIS topic's count. A member
    # that is the canonical of a DIFFERENT topic is `member_already_canonical`
    # (`_member_hazards`), and the two codes stay disjoint — `_is_incumbent`'s
    # docstring draws that same line.
    for member_id in incumbents:
        record = members.get(member_id)
        metadata = _metadata(record)
        present = sorted(key for key in CORRECTION_METADATA_KEYS if key in metadata)
        banner = CORRECTION_BANNER_RE.search(_content(record)) is not None
        if present or banner:
            found = ', '.join(present) if present else 'a correction banner in its body'
            reasons.append(
                AutoReason(
                    code=AutoReasonCode.canonical_carries_correction,
                    ids=(member_id,),
                    detail=(
                        f'{member_id}, the canonical of topic `{proposal.topic}`, is '
                        f'itself corrected or superseded ({found}); consolidating '
                        'around it would build on a retracted index entry'
                    ),
                ),
            )

        category = metadata.get('category')
        if sole_category is not None and isinstance(category, str) and category != sole_category:
            reasons.append(
                AutoReason(
                    code=AutoReasonCode.canonical_category_mismatch,
                    ids=(member_id,),
                    detail=(
                        f'{member_id} indexes topic `{proposal.topic}` under category '
                        f'{category}, but its members are {sole_category}'
                    ),
                ),
            )

    return reasons


def _slug_hazards(
    proposal: AutoProposal,
    existing_canonical_slugs: Sequence[str],
    threshold: float,
) -> list[AutoReason]:
    """Refusals about the topic slug itself — the slug half of rung 2.

    Two slugs that mean the same thing split one topic across two canonicals,
    and nothing sweeps the split afterwards.
    """
    reasons: list[AutoReason] = []

    for slug in existing_canonical_slugs:
        # LOAD-BEARING SKIP. On every re-emission the topic ALREADY has a
        # canonical, so its own slug is necessarily in this list and its
        # self-Jaccard is 1.0. Without the skip the hazard fires on every
        # tag-only refresh, PASS_TAG_ONLY becomes unreachable, and the predicate
        # refuses precisely the case it was built for (PRD B2). The skip lives
        # here rather than in the caller because task delta's executor and the
        # task-theta migration script assemble this list independently, and
        # either could forget it.
        if slug == proposal.topic:
            continue

        ratio = _slug_jaccard(slug, proposal.topic)
        # `>=`, not `>`: the fail-closed reading of C2's "above a config
        # threshold". At the exact boundary a human gate costs one sitting,
        # while a wrong auto-mint splits a topic permanently.
        if ratio >= threshold:
            reasons.append(
                AutoReason(
                    code=AutoReasonCode.slug_near_collision,
                    detail=(
                        f'proposed topic `{proposal.topic}` overlaps existing topic '
                        f'`{slug}` at Jaccard {ratio:.2f} (threshold {threshold}); '
                        'they may be one topic under two names'
                    ),
                ),
            )

    return reasons


def evaluate_auto_predicate(
    proposal: AutoProposal,
    *,
    members: Mapping[str, Mapping[str, Any] | None | _Unreadable],
    canonical_count: int | None,
    open_gate_id: str | None,
    existing_canonical_slugs: Sequence[str],
    config: ConsolidationAutoConfig,
) -> AutoVerdict:
    """Rule on whether *proposal* may be consolidated with no human in the loop.

    PRD contract C2. The caller supplies EVERY fact and this function performs
    no I/O: *members* is the caller's per-id read (``None`` for a genuine miss,
    :data:`UNREADABLE` when the read did not answer), *canonical_count* is how
    many canonicals the topic already has (``None`` when the caller could not
    find out), *open_gate_id* is a human gate already open on the topic, and
    *existing_canonical_slugs* is the topic vocabulary to check for a near
    collision. There is no service parameter, because a verdict that could
    depend on a live read would be a verdict that depends on WHEN it ran.

    Evaluation order (BINDING — the rungs are tried in this sequence and the
    first that fires decides):

    1. ``already_gated`` -> NOOP. A topic a human already owns collects no
       second filing.
    2. Hazards -> FAIL, collecting EVERY offender rather than stopping at the
       first, so one human sitting names every problem. Among them: a
       *canonical_count* of zero CONTRADICTED by a named member that is this
       topic's canonical — two non-atomic reads disagreeing, which leaves the
       topic's canonical state undecidable here.
    3. An incumbent canonical of THIS topic named in the member list is
       STRIPPED from the retained set and disclosed.
    4. ``already_consolidated`` -> NOOP, judged over the RETAINED set.
    5. A retained member not yet stamped with the topic, beside exactly one
       canonical -> PASS_TAG_ONLY.
    6. No canonical at all, and at least one RETAINED member to index -> PASS.
       A mint over an empty cluster is refused instead: the executor is never
       told to write a canonical that would index nothing.

    The order is the contract, not an optimisation (PRD D4). Two cases fix it:
    a NEW member carrying a correction banner must FAIL even when the topic has
    a healthy live canonical — so hazards outrank the tag-only rung — and an
    incumbent canonical appearing in the member list is STRIPPED rather than
    failed, or PRD B2's re-emission-with-regrowth (the majority verdict over
    time) could never pass. Both refusals named above live INSIDE an existing
    rung rather than adding a seventh, so that binding order is untouched.

    What this function deliberately does NOT check: the benign shape codes
    ``member_count_out_of_range``, ``invalid_slug`` and
    ``claim_not_index_shaped``. Those are refused at the emit boundary by C1
    (``server/consolidation.py::validate_consolidate_args``) and are not
    re-derived here — one rule, one enforcement point (PRD D6). The predicate
    never even sees the claim.
    """
    # Rung 1. Evaluated before the hazard arm on purpose: when a human already
    # holds the topic, what the members look like is not this function's
    # business, and reporting a hazard here would invite a second filing on a
    # cluster someone is already sitting with (task 3524's DECIDE-FIRST seam).
    if open_gate_id is not None:
        return AutoVerdict(
            outcome=AutoOutcome.NOOP,
            reasons=(
                AutoReason(
                    code=AutoReasonCode.already_gated,
                    ids=(open_gate_id,),
                    detail=(
                        f'topic `{proposal.topic}` is already gated by task '
                        f'{open_gate_id}; the human sitting owns it'
                    ),
                ),
            ),
            retain_ids=(),
            stripped_ids=(),
            predicate_version=config.predicate_version,
        )

    # Rung 2. Hazards outrank every outcome rung below (PRD D4): a new member
    # carrying a correction banner must refuse the cluster even when the topic
    # has a healthy live canonical, because otherwise the tag-only rung would
    # stamp a retracted record into that topic's scroll. A FAIL is not
    # actionable, so it carries no retained or stripped ids — the reasons name
    # every record involved, which is what the human sitting reads.
    # One flat concatenation, so collecting every offender is structural rather
    # than remembered: no arm can short-circuit another.
    hazards = [
        *_member_hazards(proposal, members),
        *_canonical_hazards(proposal, members, canonical_count),
        *_slug_hazards(proposal, existing_canonical_slugs, config.slug_collision_jaccard),
    ]
    if hazards:
        return AutoVerdict(
            outcome=AutoOutcome.FAIL,
            reasons=tuple(hazards),
            retain_ids=(),
            stripped_ids=(),
            predicate_version=config.predicate_version,
        )

    reasons: list[AutoReason] = []

    # Rung 3. The proposal may name the canonical the topic already has — an
    # LLM reading a topic scroll enumerates it along with the members. Strip it
    # from the retained set (the executor would otherwise tag or fold the
    # canonical into itself) and DISCLOSE it: a silently shortened member list
    # drops a record the proposal named with nothing recording why.
    retain_ids: list[str] = []
    stripped_ids: list[str] = []
    for member_id in proposal.member_ids:
        if _is_incumbent(members.get(member_id), proposal.topic):
            stripped_ids.append(member_id)
            reasons.append(
                AutoReason(
                    code=AutoReasonCode.incumbent_canonical_stripped,
                    ids=(member_id,),
                    detail=(
                        f'{member_id} is the canonical topic `{proposal.topic}` '
                        'already has; it is excluded from the members to act on '
                        'and its content and metadata are left untouched'
                    ),
                ),
            )
        else:
            retain_ids.append(member_id)

    # Rungs 4 and 5 split on the RETAINED set, after the strip above — the
    # incumbent is accounted for separately rather than counted as a member
    # that happens to be stamped.
    every_member_stamped = all(
        _metadata(members.get(member_id)).get('topic') == proposal.topic
        for member_id in retain_ids
    )

    if canonical_count == 1:
        if every_member_stamped:
            # Rung 4. Nothing to write. NOOP rather than a tag-only pass is the
            # difference between a cycle that does nothing and one that
            # rewrites a settled topic every time Stage 1 notices it again.
            reasons.append(
                AutoReason(
                    code=AutoReasonCode.already_consolidated,
                    ids=tuple(retain_ids),
                    detail=(
                        f'topic `{proposal.topic}` already has its canonical and '
                        f'every one of the {len(retain_ids)} proposed members is '
                        'already stamped with it'
                    ),
                ),
            )
            return AutoVerdict(
                outcome=AutoOutcome.NOOP,
                reasons=tuple(reasons),
                retain_ids=tuple(retain_ids),
                stripped_ids=tuple(stripped_ids),
                predicate_version=config.predicate_version,
            )

        # Rung 5. Stamp the unstamped members onto the existing topic and touch
        # the incumbent canonical's content and metadata not at all (PRD D14).
        return AutoVerdict(
            outcome=AutoOutcome.PASS_TAG_ONLY,
            reasons=tuple(reasons),
            retain_ids=tuple(retain_ids),
            stripped_ids=tuple(stripped_ids),
            predicate_version=config.predicate_version,
        )

    if canonical_count == 0:
        if not retain_ids:
            # The write-bearing guard. Never tell the executor to mint a
            # canonical that would index nothing.
            #
            # Reachability, stated honestly because it is the non-obvious part:
            # a STRIPPED incumbent can no longer empty the retained set here —
            # that fires `canonical_count_contradicted` back at rung 2 — so the
            # surviving path is a proposal that named no members at all. That is
            # the emit boundary's `member_count_out_of_range` (C1) and this
            # predicate deliberately does not re-derive it (PRD D6), but an aged
            # ledger row or a mis-wired caller can still put one in front of us,
            # and a predicate whose contract is fail-closed must not answer it
            # with a write-bearing PASS. Heuristic 10: the invariant enforced
            # redundantly at the point where it would otherwise be silently
            # violated — the same stance as the loud `AssertionError` below.
            #
            # `reasons` is provably empty on this path (no strip can have
            # happened), but pass it through rather than dropping it, so the
            # guard stays correct if a future rung discloses something above it.
            reasons.append(
                AutoReason(
                    code=AutoReasonCode.no_retained_members,
                    detail=(
                        f'topic `{proposal.topic}` has no canonical and this '
                        'proposal retains no member to index; minting a canonical '
                        'over an empty cluster is not something to do'
                    ),
                ),
            )
            return AutoVerdict(
                outcome=AutoOutcome.FAIL,
                reasons=tuple(reasons),
                retain_ids=(),
                stripped_ids=(),
                predicate_version=config.predicate_version,
            )

        # Rung 6. Mint the canonical and fold the members. Reached with members
        # already stamped too: task theta's migration stamps a topic before any
        # canonical exists, and a NOOP there would leave a scroll with members
        # and no index entry that nothing else sweeps.
        return AutoVerdict(
            outcome=AutoOutcome.PASS,
            reasons=tuple(reasons),
            retain_ids=tuple(retain_ids),
            stripped_ids=tuple(stripped_ids),
            predicate_version=config.predicate_version,
        )

    # Unreachable: rung 2 refuses `canonical_count is None` as
    # `canonical_count_unavailable` and `canonical_count > 1` as
    # `multiple_canonicals`, so only 0 and 1 arrive here. Loud rather than a
    # silent default — if either hazard is ever removed, this says so at the
    # exact point the verdict would otherwise be invented.
    raise AssertionError(
        f'unreachable: canonical_count={canonical_count!r} for topic '
        f'`{proposal.topic}` should have been refused by the hazard arm as '
        'canonical_count_unavailable or multiple_canonicals',
    )
