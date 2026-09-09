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
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    'CORRECTION_BANNER_RE',
    'CORRECTION_METADATA_KEYS',
    'UNREADABLE',
    'AutoOutcome',
    'AutoProposal',
    'AutoReason',
    'AutoReasonCode',
    'AutoVerdict',
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
    slug_near_collision = 'slug_near_collision'
    incumbent_canonical_stripped = 'incumbent_canonical_stripped'
    already_consolidated = 'already_consolidated'


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
