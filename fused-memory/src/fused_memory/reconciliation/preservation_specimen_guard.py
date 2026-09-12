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

logger = logging.getLogger(__name__)

__all__ = [
    'MEM0_KIND_INVESTIGATION_OUTCOME',
    'PRESERVATION_TOKEN_FAMILY',
    'STRANDED_ACTION_TARGET_FAMILY',
    'STRANDED_ACTION_VERB_FAMILY',
    'STRANDED_FLAG_TOKEN_FAMILY',
    'GRAPHITI_TASK_ENTITY_TEMPLATE',
    'PreservationSuppressionResult',
    'cites_preservation',
    'filter_preservation_specimen_flags',
    'flag_asserts_stranded',
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


#: Canonical Graphiti entity label for a task.  ``get_entity`` resolves this
#: shape by EXACT, case-sensitive match, so it lands on the task's own node
#: instead of scattering across fuzzy neighbours.
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


def _graphiti_citation(entity: Any) -> str | None:
    """Return the uuid of the first edge fact / node summary citing preservation.

    Consults :data:`_GRAPHITI_CITATION_SOURCES` in order.  ``None`` for a
    non-mapping result, a missing collection, or no match.  Pure, sync, no I/O.
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


async def _corroborate_preservation(
    memory_service: Any, project_id: str, task_id: str,
) -> str | None:
    """Return the citation corroborating *task_id* as a preserved specimen, or ``None``.

    Two channels, OR'd, cheapest-first.

    1. **Mem0** — a deterministic ``get_memories_by_metadata`` scroll filtered on
       ``{kind, task_id, actionable: False}``.  Qdrant ANDs equality conditions,
       so the ``actionable: False`` term is what makes a retrieved row a RECORDED
       not-actionable adjudication rather than any passing mention of the task.
    2. **Graphiti** — ``get_entity('Task <id>')``, consulted ONLY when channel 1
       found nothing.

    Neither channel is a semantic ``search``: its top-N cutoff silently drops
    low-similarity matches, and a silent miss here re-opens the destructive path.

    The fallback is not redundant.  A Graphiti edge exists as soon as the
    preservation fact is recorded, whereas an ``investigation_outcome`` row only
    exists after some stage has ALREADY investigated a flag — so channel 2 is
    what protects a newly documented specimen on its first cycle, which is
    precisely the window tasks 5080 and 5104 were filed in.  Channel 1 runs
    first because its metadata filter is the cheaper and more authoritative
    signal, which makes the fallback free on the common path.
    """
    rows = await memory_service.get_memories_by_metadata(
        project_id=project_id,
        filters={
            'kind': MEM0_KIND_INVESTIGATION_OUTCOME,
            'task_id': task_id,
            'actionable': False,
        },
    )
    citation = _first_citation(rows, _mem0_row_text, 'id')
    if citation is not None:
        return citation

    entity = await memory_service.get_entity(
        GRAPHITI_TASK_ENTITY_TEMPLATE.format(task_id=task_id), project_id,
    )
    return _graphiti_citation(entity)


# ── Orchestrator ─────────────────────────────────────────────────────────────


async def filter_preservation_specimen_flags(
    memory_service: Any,
    project_id: str,
    flags: list[dict[str, Any]] | None,
    *,
    log: logging.Logger = logger,
) -> PreservationSuppressionResult:
    """Drop stranded-class recon flags for tasks documented as preserved specimens.

    Each flag's task is corroborated by :func:`_corroborate_preservation`; a
    corroborated task's flag is dropped and the citation recorded.

    Returns a :class:`PreservationSuppressionResult`; the caller assigns
    ``kept_flags`` back to ``items_flagged``.  Suppression is NOT resolution —
    the caller excludes suppressed flags' signatures from marker acknowledgment
    so recurrence history survives.
    """
    kept: list[dict[str, Any]] = []
    suppressed_by_task: dict[str, int] = {}
    citations_by_task: dict[str, str] = {}

    for flag in flags or ():
        task_id = str(flag.get('task_id') or '')
        if not task_id:
            kept.append(flag)
            continue

        citation = await _corroborate_preservation(memory_service, project_id, task_id)
        if citation is None:
            kept.append(flag)
            continue

        suppressed_by_task[task_id] = suppressed_by_task.get(task_id, 0) + 1
        citations_by_task[task_id] = citation
        log.info(
            'preservation_specimen_guard: suppressed flag_type=%r for task %s in '
            'project %s — corroborated as a preserved specimen by %s',
            flag.get('flag_type'), task_id, project_id, citation,
        )

    return PreservationSuppressionResult(
        kept_flags=kept,
        suppressed_by_task=suppressed_by_task,
        citations_by_task=citations_by_task,
        unresolved_task_ids=(),
    )
