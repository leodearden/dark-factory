"""Preservation-specimen corroboration guard for Stage 1 — task 4223.

Some tasks are left ``in-progress`` with a null claimant and a null heartbeat
*deliberately*, because that state is the evidence something else is waiting
on.  Stage 1's stranded-task heuristic cannot see intent: it reads the absence
of a live-workflow signal, and ``prompts/stage1.py`` tells it that absence
means "the genuinely stranded case that legitimately needs operator
attention".  So it re-emits a moderate/actionable stranded finding for such a
task on roughly every cycle, and downstream that finding becomes a request to
destroy the very thing it is looking at.

THE MEASURED DAMAGE.  dark_factory task 3105 is the sole preserved live
validation specimen for gate task 3546's soak/flip checklist.  Its re-flag
twice became an operator-gate ``submit_task`` asking for the specimen to be
reset out of ``in-progress`` — task 5080 (filed 2026-09-04T09:18:52Z) and task
5104 (filed 2026-09-04T14:09:16Z, born-at-L2 critical).  Both were declined
and cancelled by hand, without touching 3105.  Its twin specimen, task 3371,
was not so lucky: it stopped being a specimen when its pinning escalation lost
open status in an orphan-reaper cascade on 2026-08-08.

WHY A SUPPRESSION LIST DID NOT FIX IT.  A ``stage1_flag_suppression`` record
for task 3105 already exists (mem0 63905117-add7-4d47-89f2-90bb34157afc) and
was WIDENED to blanket on 2026-09-07 by removing its ``flag_types`` key.
Recurrence continued anyway, for two independent reasons this module is shaped
around:

1. ``stages/memory_consolidator.py``'s remediation early-return sits ABOVE the
   whole filter chain, so nothing wired beside ``filter_entity_standing_decisions``
   runs on a remediation pass — and the two post-widening recurrences (run
   f16954ae on 2026-08-26, run 720ebf37 on 2026-09-11) were both remediation
   runs.  This guard is therefore wired ABOVE that early-return.
2. ``flag_type`` is free-form LLM output.  This ONE false positive has already
   worn three distinct namings — ``task_stranded_no_claimant``,
   ``task_stranded_no_claimant_heartbeat``, ``stranded_merge_phase_liveness``
   — as mem0 4dcf7de6 states outright ("the THIRD occurrence of this false
   positive under a THIRD distinct flag_type naming").  An enumerated
   ``(task_id, flag_type)`` list cannot keep up, which is why the scoped record
   had to be widened in the first place.

So the verdict is DERIVED from the preservation evidence rather than looked up
in a hand-authored list.  That also closes the gap 5080 and 5104 were filed in:
a specimen is protected on the first cycle it is documented, not on the first
cycle somebody remembers to write a suppression record for it.

COMPOSITE TASK IDS, AND THE DEFECT THIS AVOIDS.  A flag's OWN ``task_id`` is
routinely comma-joined — 29 of 235 live ``stage1_flag_marker`` ledger rows are,
and task 3105 appears as ``'3105,5080'``, ``'3105,5080,5104'`` and
``'3105,4223'`` — so every candidate id is DECOMPOSED before corroboration and
the counters key on the corroborated COMPONENT, not on the raw string.  That is
a correctness requirement rather than a refinement, and the reason is visible in
the sibling: ``flag_dedup.filter_suppressed._keep`` looks its flag's task_id up
VERBATIM while only the suppression ROW side is decomposed (see
``_decompose_suppression_task_id``'s docstring, which scopes itself to that
side), so a suppression row for ``'3105'`` still does not match a finding
carrying ``'3105,4223'``.  Inheriting that gap would make this guard
zero-recall on exactly the task it exists to protect.  The flag-side splitter
followed here is ``_cluster_growth_candidate_task_ids`` (task 3476).

LEAF CONTRACT.  This module imports only from
``standing_decision_constants`` (for the one genuinely shared fact, the
``investigation_outcome`` mem0 kind) and ``services.memory_service`` (for the
canonical raw-payload content extractor).  It reaches nothing in ``stages/``,
``middleware/`` or ``prompts/``; the consolidator calls in.  It performs
detection only — it drops flags, never writes tasks or memories.

WHAT "DISTINCTIVE" MEANS HERE, AND WHY IT IS CHECKED.  Both token families are
matched as casefolded SUBSTRINGS, so their safety rests entirely on holding
phrases that appear in no unrelated text — the same precondition
``standing_decision_constants.GROUNDS_TOKEN_FAMILIES``' comment makes of its
own stems, and for the same reason: a member that fires outside its class does
not merely add noise, it silently DROPS that finding.  The bias is
deliberately toward UNDER-suppression — a miss costs one more cycle of a false
positive the rotation has been absorbing since August, while an over-broad
member costs a hidden finding.  ``tests/reconciliation/test_preservation_specimen_guard.py``
enforces this behaviourally, running each matcher against live-but-unrelated
flag_types and unrelated recon prose, rather than asserting on this comment.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fused_memory.reconciliation.standing_decision_constants import (
    MEM0_KIND_INVESTIGATION_OUTCOME,
)

# The canonical ``data`` -> ``memory`` -> ``content`` raw-payload key fallback.
# IMPORTED rather than re-spelled, exactly as ``server/grouped_read.py`` does:
# a scroll payload and a search item do not put the text under the same key,
# which is why guessing one is wrong, and a second copy of the order would be
# one more place for the two to drift (INV-5).
from fused_memory.services.memory_service import _mem0_content

# Optional escalation dependency, mirroring ``flag_dedup``'s block: the
# reconciliation package must import cleanly where the escalation package is not
# installed, and ``maybe_escalate_preservation_suppression_storm`` no-ops when
# ``Escalation`` is None.  ONE combined block, so all five names bind or fail
# together and any single identity check suffices at runtime; every name is
# still listed in that guard because only an identity check on the name itself
# narrows an optionally-imported symbol for the type checker.
try:
    from escalation.dedupe import (  # type: ignore[import-untyped]
        DedupeConfig,
        compute_content_fingerprint,
        content_fingerprint_key,
        submit_or_dedupe,
    )
    from escalation.models import Escalation  # type: ignore[import-untyped]
except ImportError:
    Escalation = None  # type: ignore[assignment,misc]
    DedupeConfig = None  # type: ignore[assignment,misc]
    compute_content_fingerprint = None  # type: ignore[assignment]
    content_fingerprint_key = None  # type: ignore[assignment]
    submit_or_dedupe = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

__all__ = [
    'CATEGORY_PRESERVATION_SPECIMEN_STORM',
    'MEM0_KIND_INVESTIGATION_OUTCOME',
    'PRESERVATION_SUPPRESSION_STORM_THRESHOLD_PER_CYCLE',
    'PRESERVATION_TOKEN_FAMILY',
    'STRANDED_ACTION_TARGET_FAMILY',
    'STRANDED_ACTION_VERB_FAMILY',
    'STRANDED_FLAG_TOKEN_FAMILY',
    'UNRESOLVED_CORROBORATION_SUBJECT',
    'GRAPHITI_TASK_ENTITY_TEMPLATE',
    'PreservationSuppressionResult',
    'cites_preservation',
    'filter_preservation_specimen_flags',
    'flag_asserts_stranded',
    'maybe_escalate_preservation_suppression_storm',
]


# ── Constants ────────────────────────────────────────────────────────────────

#: Phrases that identify text as a PRESERVATION CITATION — a statement that a
#: task's odd state is deliberate because the task is being kept as evidence.
#:
#: Curated from the live corpus, not invented.  ``'validation specimen'`` alone
#: covers all five of task 3105's ``investigation_outcome`` rows and both the
#: Graphiti edge and node summary, across four different surrounding phrasings
#: ("sole preserved live ...", "preserved dark-factory ...", "preserved SOLE
#: live ...", "sole remaining preserved live ...") — which is exactly why the
#: family holds the invariant NOUN PHRASE rather than any one full sentence.
#:
#: Every member is multi-word by rule.  The bare words this class is written in
#: — ``preserved``, ``specimen``, ``intentional``, ``deliberate`` — all occur
#: throughout unrelated recon prose, so a single-word member would suppress far
#: past this class.  ``'hand-recover'`` was considered and REJECTED on the same
#: test the ``GROUNDS_TOKEN_FAMILIES`` comment applies to ``'count'``/``'scope'``:
#: it adds zero recall on the live corpus (both rows carrying it already match
#: ``'validation specimen'``) while "task 3066 was hand-recovered by the
#: operator" is ordinary operations prose that would falsely corroborate.  The
#: four-word ``'hand-recovery evidence loss'`` is kept instead — it names the
#: specific harm that makes a specimen worth preserving and cannot be reached
#: by a bare mention of a recovery.
PRESERVATION_TOKEN_FAMILY: tuple[str, ...] = (
    'validation specimen',
    'preservation specimen',
    'preserved specimen',
    'hand-recovery evidence loss',
)

#: Stems identifying a flag_type as a member of the stranded/reset class.
#:
#: ONE stem, not an enumerated list of namings: ``'strand'`` is a substring of
#: ``task_stranded_no_claimant``, ``task_stranded_no_claimant_heartbeat`` and
#: ``stranded_merge_phase_liveness`` alike.  That is the whole point — the
#: naming drifts every few cycles and the stem does not.  Checked against the
#: live flag_type vocabulary: no unrelated flag_type contains it.
STRANDED_FLAG_TOKEN_FAMILY: tuple[str, ...] = ('strand',)

#: Second, orthogonal channel: what the finding ASKS FOR.  A flag_type nobody
#: has seen yet still makes the stranded claim if its suggested action is to
#: move the task out of ``in-progress``.  Matched as a CONJUNCTION with
#: :data:`STRANDED_ACTION_TARGET_FAMILY` — each half alone is far too common
#: ("reset the priority override", "the task is in-progress") to gate a drop.
STRANDED_ACTION_VERB_FAMILY: tuple[str, ...] = (
    'reset',
    'redispatch',
    're-dispatch',
    'unstick',
    'un-stick',
)

#: The status the stranded class always wants moved away from.  Both spellings
#: occur in the corpus ("reset it in-progress->pending", "reset it out of
#: in progress").
STRANDED_ACTION_TARGET_FAMILY: tuple[str, ...] = ('in-progress', 'in progress')

#: Flag fields consulted for the ACTION channel.  ``suggested_action`` is the
#: canonical field (``FINDING_ITEM_SCHEMA``); ``description`` is read too
#: because the LLM routinely states the requested reset there and leaves
#: ``suggested_action`` terse.
_STRANDED_ACTION_FIELDS: tuple[str, ...] = ('suggested_action', 'description')


#: Per-cycle, per-task suppression ceiling for the INV-4 storm escape.  One
#: preserved specimen legitimately draws one stranded finding a cycle, very
#: occasionally two (the same claim under two namings), so MORE THAN this many
#: drops for a single task in a single cycle is not the shape this guard was
#: built for: the citation is over-broad, stale, or the task's situation has
#: changed.  Strict ``>``, matching ``SUPPRESSION_STORM_THRESHOLD_PER_CYCLE``'s
#: value and reasoning without importing it — the two gates are independent and
#: must be free to move apart.
#:
#: Defined HERE, not in ``standing_decision_constants``: that module's stated
#: purpose is the entity-standing-decision batch, and this is a different
#: suppression class.  INV-5 asks that a fact have one home, not that every
#: constant share one file.
PRESERVATION_SUPPRESSION_STORM_THRESHOLD_PER_CYCLE: int = 5

#: Escalation category for BOTH records this module files, single-sourced here
#: rather than spelled at each use site.  ``Escalation.category`` is free-form
#: prose validated by nothing at submit time, so a categorized detector is only
#: correct while its filer and its readers spell the category identically — the
#: fold gate (``DedupeConfig.infra_dedupe_categories``) and any operator query
#: both read it back.  A shared constant makes that agreement structural.
CATEGORY_PRESERVATION_SPECIMEN_STORM: str = 'reconciliation_preservation_specimen_storm'

#: ``Escalation.task_id`` of the unresolved-corroboration report.  That record's
#: subject is the GUARD, not any one task — "my corroboration reads are
#: failing" is a subsystem fact, and filing it per task would mint one record
#: per candidate the cycle a backend goes down.  A stable sentinel subject keeps
#: it findable via ``get_by_task`` and folding into a single parent across
#: cycles, with the affected task ids carried in the detail.
UNRESOLVED_CORROBORATION_SUBJECT: str = 'preservation_specimen_guard'

#: Finding-category components of the two records' dedupe fingerprints — the
#: second axis of ``compute_content_fingerprint``.  Distinct values are what
#: keep a storm record and an unresolved-corroboration report from folding into
#: each other when both fire in one cycle.
_STORM_FINDING_CATEGORY: str = 'preservation_specimen_suppression_storm'
_UNRESOLVED_FINDING_CATEGORY: str = 'preservation_specimen_corroboration_unreadable'


# ── Pure helpers ─────────────────────────────────────────────────────────────


def _contains_any(text: Any, family: tuple[str, ...]) -> bool:
    """Return True iff *text* is a non-empty ``str`` containing a *family* member.

    Casefolded substring test, the spelling
    ``flag_dedup._flag_type_in_grounds_family`` uses.  Total over malformed
    input: a non-``str`` (``None``, ``int``, ``bytes``, ``list``) is ``False``,
    never an exception — every value reaching this helper comes out of a
    free-form LLM-authored dict or a raw Qdrant payload.

    Pure, sync, no I/O.
    """
    if not isinstance(text, str) or not text:
        return False
    folded = text.casefold()
    return any(member in folded for member in family)


def cites_preservation(text: Any) -> bool:
    """Return True iff *text* states that a task is a deliberately preserved specimen.

    The single preservation matcher for BOTH corroboration channels — the Mem0
    ``investigation_outcome`` prose and the Graphiti node summary / edge facts
    — so the two can never drift apart on what counts as a citation (INV-5).

    Matches any member of :data:`PRESERVATION_TOKEN_FAMILY` as a casefolded
    substring.  See that constant's comment for why the family holds only
    multi-word phrases and what was deliberately left out of it.

    Pure, sync, no I/O.
    """
    return _contains_any(text, PRESERVATION_TOKEN_FAMILY)


def flag_asserts_stranded(flag: Any) -> bool:
    """Return True iff *flag* makes the "this task is stranded" claim.

    Two orthogonal channels, OR'd, because neither is sufficient alone:

    1. **Vocabulary** — ``flag['flag_type']`` contains a
       :data:`STRANDED_FLAG_TOKEN_FAMILY` stem.  Catches the namings already
       observed, and any future one built on the same word.
    2. **Requested action** — the flag's ``suggested_action``/``description``
       contains BOTH a :data:`STRANDED_ACTION_VERB_FAMILY` verb and a
       :data:`STRANDED_ACTION_TARGET_FAMILY` status.  Catches a naming built on
       a different word entirely, which the vocabulary channel must eventually
       miss because ``flag_type`` is free-form LLM output.

    The conjunction in channel 2 is load-bearing.  ``'reset'`` alone matches
    ``stale_priority_override``'s perfectly reasonable "reset the priority
    override back to its default"; ``'in-progress'`` alone matches any finding
    that merely mentions a task's status.  Requiring both keeps the channel
    inside the class it is named for.

    This is the discriminator that BOUNDS the guard: a preserved task's genuine,
    unrelated findings still reach Stage 2 because they answer False here.

    Total over malformed input — *flag* may be any object, and a non-mapping,
    an absent key, or a non-``str`` value all yield ``False`` rather than
    raising.  ``items_flagged`` entries are free-form LLM-authored dicts.

    Pure, sync, no I/O.
    """
    if not isinstance(flag, dict):
        return False
    if _contains_any(flag.get('flag_type'), STRANDED_FLAG_TOKEN_FAMILY):
        return True
    return any(
        _contains_any(flag.get(field), STRANDED_ACTION_VERB_FAMILY)
        and _contains_any(flag.get(field), STRANDED_ACTION_TARGET_FAMILY)
        for field in _STRANDED_ACTION_FIELDS
    )



def _flag_task_ids(flag: dict[str, Any]) -> tuple[str, ...]:
    """Every task id *flag* names, in order, deduped; ``()`` when none is usable.

    Stage 1 routinely emits COMPOSITE task_ids — 29 of 235 live
    ``stage1_flag_marker`` ledger rows are comma-joined, and task 3105 is flagged
    as ``'3105,5080'``, ``'3105,5080,5104'`` and ``'3105,4223'`` — so a verbatim
    lookup would find nothing for exactly the task this guard exists to protect.
    Components are stripped, so an LLM-authored ``'3105, 4223'`` resolves too,
    and a separator-only value (``','``) yields no candidates rather than a junk
    id.

    This follows :func:`~fused_memory.reconciliation.flag_dedup._cluster_growth_candidate_task_ids`
    (task 3476), the existing FLAG-side splitter — deliberately not
    ``_decompose_suppression_task_id``, whose docstring scopes it to the
    suppression ROW side only.

    Accepts the two value shapes ``items_flagged`` actually carries: a ``str``
    (split on ``','``) and an ``int`` straight off a task dict.  Everything else
    — ``None``, a list, a nested dict, a ``bool``, a non-positive int — is NOT a
    task id and yields ``()``, so a malformed value can never be stringified
    into a backend query.

    Pure, sync, no I/O.
    """
    task_id = flag.get('task_id')
    if isinstance(task_id, int) and not isinstance(task_id, bool) and task_id > 0:
        return (str(task_id),)
    if not isinstance(task_id, str):
        return ()
    seen: set[str] = set()
    components: list[str] = []
    for part in task_id.split(','):
        component = part.strip()
        if component and component not in seen:
            seen.add(component)
            components.append(component)
    return tuple(components)



@dataclass(frozen=True)
class PreservationSuppressionResult:
    """Outcome of :func:`filter_preservation_specimen_flags`.

    - ``kept_flags`` — the flags that survived, input order preserved, to be
      assigned back to ``report.items_flagged``.
    - ``suppressed_by_task`` — ``{task_id: count}`` of flags dropped, attributed
      per corroborated task.  Its ``values()`` sum is the per-cycle
      ``preservation_specimen_suppressed`` stat, and it drives the storm escape.
    - ``citations_by_task`` — ``{task_id: citation_ref}``, the Mem0 memory id or
      Graphiti uuid each suppression actually relied on.  A suppression is never
      anonymous: without this, "one flag suppressed" gives an operator no way to
      check whether the citation is real, current, or over-broad.
    - ``unresolved_task_ids`` — tasks whose corroboration could NOT be read this
      cycle.  Their flags were KEPT (see the fail-open policy in
      :func:`filter_preservation_specimen_flags`), and this field is what stops
      that keep from being byte-identical to a clean "no citation exists"
      (INV-11).
    """

    kept_flags: list[dict[str, Any]]
    suppressed_by_task: dict[str, int]
    citations_by_task: dict[str, str]
    unresolved_task_ids: tuple[str, ...]


#: Canonical Graphiti entity label for a task.  ``get_entity`` TRIES this shape
#: as an EXACT, case-sensitive match first, and on a HIT it lands on the task's
#: own node, with edges gathered topologically from that node's uuid.
#:
#: On a MISS it does not return empty.  It falls back to a purely SEMANTIC
#: gather — ``search_nodes(query='Task <id>', max_nodes=5)`` plus
#: ``graphiti.search(query='Task <id>', num_results=edge_limit)`` — whose own
#: docstring warns it "can surface edges whose fact merely mentions the
#: entity's name (or is contextually related) without that edge being
#: RELATES_TO-incident on this node at all".  ``'Task 3105'`` and ``'Task
#: 5231'`` embed almost identically, so task 3105's preservation edge is
#: exactly the kind of edge that fuzzy branch surfaces for ANY other task.
#:
#: Both branches return the SAME ``{'nodes', 'edges'}`` shape, so a caller
#: cannot tell them apart by shape alone.  That is why
#: :func:`_entity_is_scoped_to_task` re-derives this label and requires it
#: before any citation is trusted.
GRAPHITI_TASK_ENTITY_TEMPLATE: str = 'Task {task_id}'

#: The Graphiti sub-collections consulted for a preservation citation, in
#: preference order, as ``(collection key, text field)``.
#:
#: EDGES FIRST, deliberately.  An edge fact is the atomic, individually
#: addressable statement (``get_edge(uuid)``) and is durable; a node summary is
#: a derived digest that ``refresh_entity_summary`` regenerates.  Citing the
#: edge is what lets a later reader check the evidence rather than take the
#: suppression on trust.  The summary is still consulted, because it can carry
#: the fact in a graph whose edges word it differently.
_GRAPHITI_CITATION_SOURCES: tuple[tuple[str, str], ...] = (
    ('edges', 'fact'),
    ('nodes', 'summary'),
)


def _mem0_row_text(row: dict[str, Any]) -> str:
    """The human-readable prose of one ``get_memories_by_metadata`` row.

    That reader returns ``{'id', 'created_at', 'metadata'}`` where ``metadata``
    is the FULL raw Qdrant payload, so the text is wherever
    :func:`~fused_memory.services.memory_service._mem0_content` says it is —
    ``data``, then ``memory``, then ``content``.  Pure, sync, no I/O.
    """
    payload = row.get('metadata')
    return _mem0_content(payload) if isinstance(payload, dict) else ''


def _first_citation(
    rows: Any,
    text_of: Callable[[dict[str, Any]], Any],
    ref_key: str,
) -> str | None:
    """Return *ref_key* of the first row in *rows* whose text cites preservation.

    The single definition of "a citable preservation record", shared by both
    channels so they cannot drift on what counts (INV-5).  *text_of* extracts
    the prose (payload fallback for Mem0, a plain field for Graphiti) and
    *ref_key* names the field holding the citable id (``'id'`` / ``'uuid'``).

    ``None`` when nothing matches, when *rows* is not a list/tuple of mappings,
    or when the row that matches carries no usable ref.  That last case is
    deliberate and mirrors ``curator_gate_resolution_sweep``'s refusal to flag a
    gate it cannot cite: a suppression that cannot name its evidence is exactly
    the anonymous drop ``citations_by_task`` exists to prevent.

    Total over malformed input — every row here comes off a raw backend read.
    Pure, sync, no I/O.
    """
    if not isinstance(rows, (list, tuple)):
        return None
    for row in rows:
        if not isinstance(row, dict) or not cites_preservation(text_of(row)):
            continue
        ref = row.get(ref_key)
        if isinstance(ref, str) and ref:
            return ref
    return None


def _is_degraded_entity_result(entity: Any) -> bool:
    """Return True iff *entity* is ``get_entity``'s degraded-fallback result.

    ``services/memory_service.py::_degrade_or_reraise`` does NOT re-raise a
    rate-limit/quota error: it returns
    ``{'nodes': [], 'edges': [], 'degraded': True, 'failed_stores': [...]}``.
    So that outcome arrives through the SUCCESS path, carrying collections
    byte-identical to a genuine "this task has no preservation fact" — which is
    why the Graphiti channel needs this screen in ADDITION to its ``except``.

    Either marker alone suffices.  The two keys are written together today, so
    demanding both would make the verdict depend on a coincidence of the
    producer rather than on what either key means.  ``failed_stores`` counts
    only when it is a NON-EMPTY list/tuple: the empty list is what a healthy
    result carries, and treating it as degradation would report every clean
    cycle as unreadable.

    Sniffs dict KEYS, not attributes: ``get_entity``'s degraded dict and
    ``search()``'s attribute-carrying ``SearchResults`` are deliberately
    different shapes (``_graphiti_degraded_entity_result``'s own docstring).

    Total over malformed input — every value here comes off a raw backend read.
    Pure, sync, no I/O.
    """
    if not isinstance(entity, dict):
        return False
    if entity.get('degraded') is True:
        return True
    failed_stores = entity.get('failed_stores')
    return isinstance(failed_stores, (list, tuple)) and bool(failed_stores)


def _entity_is_scoped_to_task(entity: Any, task_id: str) -> bool:
    """Return True iff *entity* carries *task_id*'s OWN node, matched by name.

    THE SECOND THING A ``get_entity`` RESULT CAN BE.  Like
    :func:`_is_degraded_entity_result`, this screens an outcome that arrives
    through the success path wearing the shape of a good answer.  There the
    imposter was a degraded read; here it is ``get_entity``'s SEMANTIC
    fallback, taken whenever no node is named exactly
    ``GRAPHITI_TASK_ENTITY_TEMPLATE.format(task_id=task_id)`` (see that
    constant's comment).  A fuzzy result is a set of nodes and edges that are
    merely TEXTUALLY NEAR the query string — it is evidence about the corpus,
    never about this task.

    WHAT GOES WRONG WITHOUT IT.  Task 3105's preservation edge is prose
    containing the words a fuzzy search for any other ``'Task N'`` ranks
    highly, so an ordinary undocumented stranded task would borrow 3105's
    citation and have its finding SUPPRESSED — the hidden-finding harm this
    module's docstring says it is biased against, and worse than a plain miss
    because the drop is logged with a citation uuid belonging to a DIFFERENT
    task, so the audit trail actively misleads the reader who checks it.

    Presence of the exact-named node is the available proxy for "the exact
    branch was taken": that branch is the only one that resolves nodes BY that
    name, and a node genuinely carrying the name is about this task whichever
    branch surfaced it.  Multiple nodes may share the name (the duplicate-name
    pathology ``get_entity`` unions edges across) — one suffices.

    Total over malformed input — every value here comes off a raw backend
    read.  Pure, sync, no I/O.
    """
    if not isinstance(entity, dict):
        return False
    nodes = entity.get('nodes')
    if not isinstance(nodes, (list, tuple)):
        return False
    expected = GRAPHITI_TASK_ENTITY_TEMPLATE.format(task_id=task_id)
    return any(isinstance(node, dict) and node.get('name') == expected for node in nodes)


def _graphiti_citation(entity: Any) -> str | None:
    """Return the uuid of the first edge fact / node summary citing preservation.

    Consults :data:`_GRAPHITI_CITATION_SOURCES` in order.  ``None`` for a
    non-mapping result, a missing collection, or no match.  Pure, sync, no I/O.

    Requires BOTH pre-screens, because this matcher reads TEXT only and so
    cannot tell any of ``get_entity``'s three same-shaped outcomes apart:

    - :func:`_is_degraded_entity_result` — else a Graphiti outage is silently
      converted into a "no citation" verdict.
    - :func:`_entity_is_scoped_to_task` — else a semantic-fallback result is
      silently converted into a citation for a task it does not describe.

    The two screens fail in OPPOSITE directions, which is why neither
    substitutes for the other.
    """
    if not isinstance(entity, dict):
        return None
    for collection, text_field in _GRAPHITI_CITATION_SOURCES:
        citation = _first_citation(
            entity.get(collection), lambda row, f=text_field: row.get(f), 'uuid',
        )
        if citation is not None:
            return citation
    return None


@dataclass(frozen=True)
class _Corroboration:
    """One task's corroboration verdict.

    - ``citation`` — the citing ref, or ``None`` when no channel produced one.
    - ``degraded`` — at least one channel could not be READ this cycle.

    The two are carried separately because ``citation is None`` alone cannot
    tell "no citation exists" from "we could not look", and collapsing them is
    exactly the silent fail-soft INV-11 forbids.  ``degraded`` matters only when
    ``citation is None``: a verdict reached on a surviving channel is resolved,
    however the other channel fared.
    """

    citation: str | None
    degraded: bool


async def _corroborate_preservation(
    memory_service: Any, project_id: str, task_id: str, *, log: logging.Logger,
) -> _Corroboration:
    """Corroborate *task_id* as a preserved specimen across both channels.

    Two channels, OR'd, cheapest-first.

    1. **Mem0** — a deterministic ``get_memories_by_metadata`` scroll filtered on
       ``{kind, task_id, actionable: False}``.  Qdrant ANDs equality conditions,
       so the ``actionable: False`` term is what makes a retrieved row a RECORDED
       not-actionable adjudication rather than any passing mention of the task.
    2. **Graphiti** — ``get_entity('Task <id>')``, consulted ONLY when channel 1
       produced no citation.

    Neither channel is a semantic ``search``: its top-N cutoff silently drops
    low-similarity matches, and a silent miss here re-opens the destructive path.

    The fallback is not redundant.  A Graphiti edge exists as soon as the
    preservation fact is recorded, whereas an ``investigation_outcome`` row only
    exists after some stage has ALREADY investigated a flag — so channel 2 is
    what protects a newly documented specimen on its first cycle, precisely the
    window tasks 5080 and 5104 were filed in.  Channel 1 runs first because its
    metadata filter is the cheaper and more authoritative signal, which makes
    the fallback free on the common path.

    Each channel is guarded independently, so one being down never costs the
    other its verdict.  A failed read logs WARNING naming the project, task and
    channel, and marks the verdict ``degraded`` — it is NEVER counted as
    evidence either way.  ``asyncio.CancelledError``/``KeyboardInterrupt``/
    ``SystemExit`` propagate unchanged: a shutdown is not a backend blip.

    ``try``/``except`` is NOT full coverage of channel 2.  ``get_entity`` has
    THREE outcomes: it can answer, it can raise, and on a rate-limit/quota
    error it can ABSORB the failure and return the degraded superset dict
    (``memory_service.py::_degrade_or_reraise``).  That third outcome arrives
    through the success path with empty ``nodes``/``edges``, so left unscreened
    it would read as a clean "this task has no preservation fact" and the
    specimen would go unprotected on a cycle reported as fully resolved.  The
    success path is therefore screened by :func:`_is_degraded_entity_result`
    as well, BEFORE the citation matcher runs — a read that failed is not
    evidence in either direction.

    NOR IS A READ THAT ANSWERED ABOUT SOMETHING ELSE.  ``get_entity`` resolves
    the exact label only when a node carries it; otherwise it returns a
    SEMANTIC gather over whatever merely reads like ``'Task <id>'``, in the
    same ``{'nodes', 'edges'}`` shape.  Trusting that would let one documented
    specimen's edge corroborate every OTHER task whose label embeds near it,
    silently suppressing genuine stranded findings and citing a uuid that
    belongs to a different task.  :func:`_entity_is_scoped_to_task` is
    therefore the second pre-screen, and its failure is a resolved negative
    rather than a degradation — see the comment at that call site.
    """
    degraded = False

    try:
        rows = await memory_service.get_memories_by_metadata(
            project_id=project_id,
            filters={
                'kind': MEM0_KIND_INVESTIGATION_OUTCOME,
                'task_id': task_id,
                'actionable': False,
            },
        )
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        # Includes a propagated Qdrant read-timeout, which arrives here as an
        # exception precisely so it cannot be mistaken for an empty scroll.
        log.warning(
            'preservation_specimen_guard: mem0 corroboration read failed for task '
            '%s in project %s — verdict unresolved, flag kept',
            task_id, project_id, exc_info=True,
        )
        degraded = True
    else:
        citation = _first_citation(rows, _mem0_row_text, 'id')
        if citation is not None:
            return _Corroboration(citation=citation, degraded=False)

    try:
        entity = await memory_service.get_entity(
            GRAPHITI_TASK_ENTITY_TEMPLATE.format(task_id=task_id), project_id,
        )
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        log.warning(
            'preservation_specimen_guard: graphiti corroboration read failed for '
            'task %s in project %s — verdict unresolved, flag kept',
            task_id, project_id, exc_info=True,
        )
        return _Corroboration(citation=None, degraded=True)

    if _is_degraded_entity_result(entity):
        log.warning(
            'preservation_specimen_guard: graphiti corroboration read degraded for '
            'task %s in project %s — verdict unresolved, flag kept',
            task_id, project_id,
        )
        return _Corroboration(citation=None, degraded=True)

    if not _entity_is_scoped_to_task(entity, task_id):
        # A resolved NEGATIVE, not a degraded read: the lookup answered, and
        # what it answered is that this task has no node of its own.  Marking
        # it unresolved would file an `unresolved_corroboration` report for
        # every ordinary task absent from the graph — i.e. for most stranded
        # flags — drowning the signal that channel exists to carry.
        log.debug(
            'preservation_specimen_guard: graphiti returned no node named %r for task '
            '%s in project %s (semantic fallback) — not evidence about this task, '
            'flag kept',
            GRAPHITI_TASK_ENTITY_TEMPLATE.format(task_id=task_id), task_id, project_id,
        )
        return _Corroboration(citation=None, degraded=degraded)

    return _Corroboration(citation=_graphiti_citation(entity), degraded=degraded)


# ── Orchestrator ─────────────────────────────────────────────────────────────


async def filter_preservation_specimen_flags(
    memory_service: Any,
    project_id: str,
    flags: list[dict[str, Any]] | None,
    *,
    log: logging.Logger = logger,
) -> PreservationSuppressionResult:
    """Drop stranded-class recon flags for tasks documented as preserved specimens.

    Narrow, then corroborate, then drop:

    1. Candidates are the tasks named by flags that pass
       :func:`flag_asserts_stranded`.  Nothing else is ever looked up, which is
       what keeps a preserved task's UNRELATED findings reaching Stage 2 — the
       guard drops the adjudicated class, not the task.
    2. Each candidate is corroborated ONCE by
       :func:`_corroborate_preservation`, positive and negative verdicts alike,
       so a task appearing on N flags costs one corroboration pass.
    3. A corroborated task's stranded flags are dropped and the citation
       recorded.

    An empty candidate set short-circuits before any I/O, so a cycle with
    nothing in this class — the overwhelming majority — costs zero backend
    calls.  The corroboration loop is sequential rather than an
    ``asyncio.gather``: the candidate population is tiny, and a serial loop
    keeps per-task error attribution exact (the reason
    ``curator_gate_resolution_sweep`` gives for its own), which is what lets a
    failure name the task it belongs to.

    **Fail OPEN on the drop, but never silently.**  A read failure NEVER
    suppresses: it keeps the flag and names the task in ``unresolved_task_ids``.
    The asymmetry is deliberate.  Suppressing on a backend blip would hide every
    stranded finding fleet-wide — the hidden-finding cost the under-suppression
    bias forbids — whereas keeping the flag costs at most one more cycle of a
    false positive the rotation has absorbed since August.  But a BARE fail-open
    would make "corroboration unreadable" byte-identical to "no citation
    exists", and the destructive recommendation would then flow on unremarked;
    disclosing the unresolved tasks in the result (surfaced as a stat and
    reported through the storm escape) is what keeps the degradation visible at
    the point of consumption.  A per-task failure costs that task's verdict
    only, never the batch.

    Returns a :class:`PreservationSuppressionResult`; the caller assigns
    ``kept_flags`` back to ``items_flagged``.  Suppression is NOT resolution —
    the caller excludes suppressed flags' signatures from marker acknowledgment
    so recurrence history survives until the preservation citation is retired.
    """
    batch = list(flags or ())
    # One pass fixes each flag's candidate task ids (``()`` = not a candidate),
    # so neither the discriminator nor the decomposition is recomputed below.
    candidacy: list[tuple[dict[str, Any], tuple[str, ...]]] = [
        (
            flag,
            _flag_task_ids(flag)
            if isinstance(flag, dict) and flag_asserts_stranded(flag)
            else (),
        )
        for flag in batch
    ]
    candidates = sorted({tid for _, task_ids in candidacy for tid in task_ids})
    if not candidates:
        return PreservationSuppressionResult(
            kept_flags=batch, suppressed_by_task={}, citations_by_task={},
            unresolved_task_ids=(),
        )

    verdicts: dict[str, _Corroboration] = {}
    unresolved: list[str] = []
    for task_id in candidates:
        verdict = await _corroborate_preservation(
            memory_service, project_id, task_id, log=log,
        )
        verdicts[task_id] = verdict
        if verdict.citation is None and verdict.degraded:
            unresolved.append(task_id)

    kept: list[dict[str, Any]] = []
    suppressed_by_task: dict[str, int] = {}
    citations_by_task: dict[str, str] = {}
    kept_task_ids: set[str] = set()
    for flag, task_ids in candidacy:
        cited = next(
            (
                (tid, verdict.citation)
                for tid in task_ids
                if (verdict := verdicts.get(tid)) is not None and verdict.citation is not None
            ),
            None,
        )
        if cited is None:
            kept.append(flag)
            kept_task_ids.update(task_ids)
            continue
        # Counters key on the CORROBORATED COMPONENT, never the raw composite:
        # task 3105 is flagged as '3105,5080', '3105,5080,5104' and '3105,4223',
        # and keying on the string would scatter one task's suppressions across
        # three counters — the storm threshold would never trip and the citation
        # audit trail would fragment.
        cited_task, citation = cited
        suppressed_by_task[cited_task] = suppressed_by_task.get(cited_task, 0) + 1
        citations_by_task[cited_task] = citation
        log.info(
            'preservation_specimen_guard: suppressed flag_type=%r (task_id=%r) for '
            'task %s in project %s — corroborated as a preserved specimen by %s',
            flag.get('flag_type'), flag.get('task_id'), cited_task, project_id, citation,
        )

    # Disclose a degraded verdict only for a task that is actually LEFT
    # UNPROTECTED by it — one whose flags were kept.  A component whose read
    # failed alongside a sibling that DID corroborate changed no outcome, and
    # reporting it would raise a false alarm about an unprotected specimen.
    return PreservationSuppressionResult(
        kept_flags=kept,
        suppressed_by_task=suppressed_by_task,
        citations_by_task=citations_by_task,
        unresolved_task_ids=tuple(t for t in unresolved if t in kept_task_ids),
    )


# ── Storm escape (INV-4) ─────────────────────────────────────────────────────


def _file_or_fold(
    escalation_queue: Any,
    project_id: str,
    subject: str,
    finding_category: str,
    summary: str,
    detail: str,
    config: Any,
    log: logging.Logger,
) -> bool:
    """Submit one L1 record for *subject*; return True iff a NEW one was minted.

    Everything that can fail is inside the try, so a fingerprint, id-gen,
    constructor, submit or fold failure costs this subject its filing and is
    logged WARNING — never the rest of the cycle.  A fold logs INFO, so the
    recurrence is visible in the log stream and not only as a counter on disk.
    """
    try:
        # Unreachable through the caller, which checks the whole import block
        # first; restated here so this helper enforces its own precondition and
        # a missing package surfaces as the logged WARNING below rather than an
        # AttributeError. Inside the try deliberately — loud, not silent.
        if Escalation is None or compute_content_fingerprint is None or submit_or_dedupe is None:
            raise RuntimeError('escalation package unavailable')
        fingerprint = compute_content_fingerprint(
            CATEGORY_PRESERVATION_SPECIMEN_STORM,
            finding_category,
            [f'{project_id}:{subject}'],
        )
        # Fail closed rather than file with a falsy key: find_dedupe_parent
        # short-circuits on one, so the record would silently become a second
        # visible pending record every cycle instead of folding.
        if not fingerprint:
            raise ValueError(f'empty dedupe_fingerprint for subject={subject}')
        esc = Escalation(
            id=escalation_queue.make_id(subject),
            task_id=subject,
            agent_role='reconciliation-stage1',
            severity='blocking',
            category=CATEGORY_PRESERVATION_SPECIMEN_STORM,
            summary=summary,
            detail=detail,
            level=1,
            dedupe_fingerprint=fingerprint,
        )
        outcome = submit_or_dedupe(escalation_queue, esc, config)
    except Exception as exc:
        log.warning(
            'maybe_escalate_preservation_suppression_storm: failed to escalate '
            'subject=%s (fingerprint, id-gen, construction, submit, or fold): %s',
            subject, exc, extra={'project_id': project_id},
        )
        return False

    if outcome.get('status') == 'dedup_skipped':
        log.info(
            'maybe_escalate_preservation_suppression_storm: subject=%s folded into '
            'parent_id=%s (child_id=%s) — the condition is recurring',
            subject, outcome.get('parent_id'), outcome.get('child_id'),
            extra={'project_id': project_id},
        )
        return False
    # Tested with != rather than == 'queued' so observed_submit_response's
    # auto-resolved/dismissed branch (a record WAS minted) still counts.
    return True


async def maybe_escalate_preservation_suppression_storm(
    escalation_queue: Any,
    project_id: str,
    run_id: str,
    result: PreservationSuppressionResult,
    *,
    threshold: int = PRESERVATION_SUPPRESSION_STORM_THRESHOLD_PER_CYCLE,
    log: logging.Logger = logger,
) -> list[str]:
    """File (or fold) the two L1 records that keep this guard audible (INV-4).

    1. **Suppression storm** — one record per task whose per-cycle suppression
       count exceeds *threshold* (strict ``>``), naming the count and the
       citation the drops leaned on.  A citation hiding a flood of findings in
       one cycle is a signal it is over-broad or stale, and the guard's whole
       job is dropping findings, so this is the escape that stops it doing so
       unaccountably.
    2. **Unreadable corroboration** — one record for the cycle when
       ``result.unresolved_task_ids`` is non-empty.  The guard fails OPEN, so a
       subsystem whose reads are all failing otherwise degrades in total
       silence: every stranded recommendation flows on, the suppressed stat
       reads 0, and nothing distinguishes that from a healthy quiet cycle.

    **Filed through** :func:`escalation.dedupe.submit_or_dedupe`, NOT gated on
    ``has_open_l1`` (task 3522).  Stage 1 re-evaluates every cycle, so a
    condition that fires once tends to fire every cycle — exactly the recurring-
    detector shape for which that skip was retired: it suppressed every cycle
    after the first, pinning ``dedupe_count`` at 0 so an operator saw no
    difference between one storm and forty.  Folding keeps ONE pending record
    per subject and increments ``dedupe_count`` on it, which is the steward's
    recurrence signal.  The fold key is the ``(category, finding_category,
    project:subject)`` fingerprint — deliberately NOT the count or run_id, which
    drift every cycle and would mint a fresh record per breach — and the window
    is UNBOUNDED, so a condition persisting for days still folds into its
    original parent.  Accepted cost: a folded record keeps the PARENT's summary,
    so the count named there is the FIRST breach's.

    Best-effort throughout: returns ``[]`` immediately when the ``escalation``
    package is unavailable, and any per-subject failure is logged and excluded.
    Returns the subjects that received a NEW record this cycle — folds excluded,
    with recurrence carried by ``dedupe_count`` and the fold INFO log.
    """
    # Any ONE identity check suffices at runtime (all five names bind or fail
    # together); each is named so the type checker narrows it below.
    if (
        Escalation is None
        or DedupeConfig is None
        or compute_content_fingerprint is None
        or content_fingerprint_key is None
        or submit_or_dedupe is None
    ):
        return []

    config = DedupeConfig(
        infra_dedupe_enabled=True,
        infra_dedupe_window_secs=float('inf'),
        infra_dedupe_categories=(CATEGORY_PRESERVATION_SPECIMEN_STORM,),
        key_fn=content_fingerprint_key,
    )

    escalated: list[str] = []

    for task_id, count in result.suppressed_by_task.items():
        if count <= threshold:
            continue
        citation = result.citations_by_task.get(task_id, 'unknown')
        filed = _file_or_fold(
            escalation_queue,
            project_id,
            task_id,
            _STORM_FINDING_CATEGORY,
            f'Preservation citation for task {task_id} suppressed {count} recon '
            f'flag(s) in a single cycle (> {threshold})',
            '\n'.join([
                f'project_id: {project_id}',
                f'run_id: {run_id}',
                f'task_id: {task_id}',
                f'citation: {citation}',
                f'suppressed_this_cycle: {count}',
                f'threshold: {threshold}',
            ]),
            config,
            log,
        )
        if filed:
            escalated.append(task_id)

    if result.unresolved_task_ids:
        unresolved = ', '.join(result.unresolved_task_ids)
        filed = _file_or_fold(
            escalation_queue,
            project_id,
            UNRESOLVED_CORROBORATION_SUBJECT,
            _UNRESOLVED_FINDING_CATEGORY,
            f'Preservation corroboration unreadable for '
            f'{len(result.unresolved_task_ids)} task(s) this cycle — stranded '
            f'findings are flowing through unfiltered',
            '\n'.join([
                f'project_id: {project_id}',
                f'run_id: {run_id}',
                f'unresolved_task_ids: {unresolved}',
                'The guard fails OPEN on the drop, so these tasks\' flags were '
                'KEPT. That is correct, but it means a preserved specimen is '
                'currently unprotected: check Qdrant/FalkorDB health and the '
                'WARNING log lines from this module naming the failing channel.',
            ]),
            config,
            log,
        )
        if filed:
            escalated.append(UNRESOLVED_CORROBORATION_SUBJECT)

    return escalated
