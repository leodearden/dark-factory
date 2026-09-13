#!/usr/bin/env python3
"""One-shot migration: reify mem0 record ``b0057f3d`` → an ``entity_standing_decision``.

Leaf η of ``plans/stage1-entity-standing-decision-prd.md`` (task 2900).

WHAT THIS MIGRATES
------------------
PRD decision 5 scopes the backfill to exactly one record. ``b0057f3d`` is an
unratified LLM convention — a ``recurring_flag_standing_decision`` mem0 memory
that *zero* code reads (PRD §3) — recording that a recurring complaint about
reify's 'orchestrator' entity was investigated and dismissed on structural
grounds. This script promotes it to the machine-consulted form: one ACTIVE row
in α's SQLite recon ledger, written through β's
``standing_decision_writer.write_entity_standing_decision``, which γ's Hook-A
filter and δ's Hook-B annotation already consume. The mem0 originals are then
stamped **evidence-only** so a later reader can tell the superseded convention
from a live record.

LIVE PROVENANCE (measured 2026-09-13, not merely at decompose)
--------------------------------------------------------------
* ``get_memory_by_id('reify', SOURCE_MEMORY_ID)`` resolves, with
  ``metadata.kind = 'recurring_flag_standing_decision'`` and an
  ``metadata.entity_uuid``; the metadata scroll for that kind returns exactly
  one record, so ``b0057f3d`` is still the sole record of its kind.
* The entity-scoped scroll returns SIX records: the source, one
  ``stage1_finding_correction``, two ``flag_correction`` and two
  ``stage1_flag_suppression``.

That last pair is why selection is not entity-scoped alone — see
:func:`select_evidence_only_targets`.

WHAT THIS DOES *NOT* DO — and why ``--apply`` is an operator action
--------------------------------------------------------------------
Mutating operations belong to the fused-memory MCP server process, the single
unsandboxed owner of the store; an in-sandbox script that builds its own
``MemoryService`` gets no such protection. ``assert_store_mutation_allowed``
(``utils/store_mutation_preflight``) is the application-level refusal that
bounds the blast radius, and it is called here BEFORE the scan and BEFORE the
first mutation. Running ``--apply`` from an agent sandbox is therefore expected
to be refused, loudly, having changed nothing. The dry run works anywhere.

OPEN QUESTION 6 — the evidence-only stamping shape — IS RESOLVED HERE
---------------------------------------------------------------------
The batch convention (see ``standing_decision_writer.py`` and
``standing_decision_constants.py``) is to record an open question's resolution
as a comment at the code site rather than by editing ``plans/``. Question 6 is
answered by :func:`build_evidence_only_patch`; read its docstring for the
shape and the reasoning.
"""

from __future__ import annotations

from typing import Any

from fused_memory.reconciliation.standing_decision_constants import (
    GROUNDS_STRUCTURAL_SIZE_CONFLATION,
)
from fused_memory.utils.validation import is_full_uuid

# ---------------------------------------------------------------------------
# Pinned identity — ONE memory id, and no copy of the entity uuid
# ---------------------------------------------------------------------------

#: The corpus this migration touches. Standing decisions are project-scoped and
#: the PRD scopes the backfill to reify's 'orchestrator' entity.
PROJECT_ID = 'reify'

#: The sole record PRD decision 5 admits to the backfill. This is the ONLY
#: memory id pinned for the source; the entity uuid it concerns is DERIVED from
#: the fetched record by :func:`resolve_source_entity_uuid` rather than pinned
#: alongside it, so there is no second copy to drift out of agreement with the
#: record it claims to describe.
SOURCE_MEMORY_ID = 'b0057f3d-dc53-4cf8-9d1f-9959bd0897bd'

#: The kind the source record must still carry. A source that changed kind is a
#: different record than the one this migration was written against, and must
#: stop the run rather than silently seed a ledger row from it.
SOURCE_KIND = 'recurring_flag_standing_decision'

#: The closed allowlist of UNRATIFIED ad-hoc kinds eligible for an evidence-only
#: stamp — exactly the two the PRD names as "unratified LLM conventions; zero
#: code reads them" (§3). Every other kind sharing the entity is out of scope,
#: including ``stage1_flag_suppression``, which IS machine-read.
DEMOTED_AD_HOC_KINDS = frozenset({SOURCE_KIND, 'stage1_finding_correction'})

#: The escalation that opened this work. A FOREIGN ref: β's
#: ``resolve_evidence_refs`` stamps every non-mem0 ref ``locally_resolved=False``
#: without a lookup, which is the correct reading — the escalation queue is not
#: this project's mem0 corpus.
ESCALATION_EVIDENCE_ID = 'esc-2867-1'

#: Human-authored mem0 evidence cited in ``b0057f3d``'s own PROSE. Pinned
#: explicitly because it is not metadata-discoverable: it carries no
#: ``entity_uuid``, so no scroll of this entity returns it. Its author is
#: ``claude-interactive`` — exactly β's arm-1 human-authorship predicate — so
#: citing it keeps the row's provenance standing on its own feet even though
#: this migration uses the ``authorized_by`` operator bypass.
HUMAN_EVIDENCE_MEMORY_IDS = ('ef1f1b1b-219c-40fb-a933-53631646df96',)

#: The grounds the migrated row asserts. Imported, never re-spelled (INV-5):
#: α's ``upsert_entity_standing_decision`` validates it against ``GROUNDS_ENUM``
#: and γ's fallback matcher keys its token family off the same value.
GROUNDS = GROUNDS_STRUCTURAL_SIZE_CONFLATION


class BackfillSourceInvalid(ValueError):
    """The pinned source record is absent, or no longer the record η targets.

    Module-typed so a caller can catch THIS migration's defect without also
    swallowing an unrelated ``ValueError`` from the ledger's own validation.
    Subclasses ``ValueError`` because the failure is a bad input value — the
    fetched payload — not a broken environment.

    Every message names the observed defect and carries a ``hint:`` clause
    saying what an operator should check, so the refusal is actionable from the
    log line alone rather than by reading this module.
    """


def resolve_source_entity_uuid(record: Any) -> str:
    """Derive the decided entity's uuid from the FETCHED source record.

    This is the single point at which the migration learns which entity it is
    about (plan design decision 3). Validating loudly here rather than pinning
    a second constant is what makes "the row describes the record it was built
    from" a checked property instead of a hope: a source that vanished, changed
    kind, or lost its ``entity_uuid`` stops the run before a ledger row exists.

    Canonicality is judged by the shared
    ``utils.validation.is_full_uuid`` — which rejects the undashed 32-hex and
    truncated forms a bare ``uuid.UUID`` parse would accept — so this module
    holds no second opinion about what a uuid looks like (INV-5).

    Args:
        record: The ``get_memory_by_id`` result for :data:`SOURCE_MEMORY_ID`,
            or ``None`` / a ``{'found': False}`` envelope when it did not
            resolve.

    Returns:
        The canonical entity uuid the standing decision is about.

    Raises:
        BackfillSourceInvalid: On any of the three defects above.
    """
    if not isinstance(record, dict) or record.get('found') is False:
        raise BackfillSourceInvalid(
            f'source memory {SOURCE_MEMORY_ID!r} did not resolve in project '
            f'{PROJECT_ID!r} (got {record!r}); hint: confirm the record still '
            'exists — a consolidation may have retired it, in which case η '
            'needs a new source id, not a re-run.'
        )

    metadata = record.get('metadata') or {}
    kind = metadata.get('kind')
    if kind != SOURCE_KIND:
        raise BackfillSourceInvalid(
            f'source memory {SOURCE_MEMORY_ID!r} carries kind {kind!r}, not '
            f'{SOURCE_KIND!r}; hint: the pinned record is no longer the standing '
            'decision this migration was written against — re-verify the source '
            'before writing a ledger row from it.'
        )

    entity_uuid = metadata.get('entity_uuid')
    if not is_full_uuid(entity_uuid):
        raise BackfillSourceInvalid(
            f'source memory {SOURCE_MEMORY_ID!r} has entity_uuid {entity_uuid!r}, '
            'which is not a canonical 36-char dashed UUID; hint: the ledger row '
            'and every γ/δ lookup key on this value verbatim, so a truncated or '
            'undashed spelling would write a row no hook can ever find.'
        )
    return entity_uuid


def select_evidence_only_targets(
    records: list[Any], entity_uuid: str
) -> list[str]:
    """Pick the mem0 records to stamp evidence-only, from a live scroll.

    The predicate is a CONJUNCTION, and both halves are load-bearing:

    * **kind ∈** :data:`DEMOTED_AD_HOC_KINDS`. The entity-scoped scroll also
      returns two ``stage1_flag_suppression`` records carrying the same
      ``entity_uuid``, and that kind is RATIFIED and machine-read —
      ``flag_dedup.filter_suppressed`` consumes it. Demoting those would be a
      live behaviour change, not a bookkeeping stamp, so entity scope alone
      cannot gate the write.
    * **entity_uuid == the source's**. Measured 2026-09-13: reify holds five
      ``stage1_finding_correction`` records and only one concerns this entity.
      Kind alone would stamp four records about entities this decision says
      nothing about.

    SELECTED, never a hardcoded id list, for the same measurement: the research
    doc's second pinned id (``12c3a5ce``) turned out to belong to a different
    entity, so a list transcribed at decompose time would have stamped the
    wrong record. A live predicate re-derives the truth on every run.

    Malformed records — no ``metadata``, no ``kind``, no ``entity_uuid``, no
    ``id`` — are EXCLUDED rather than raising: this scroll is a live corpus, and
    one unexpected neighbour must not be able to abort a migration that has
    nothing to do with it. Exclusion is the safe direction (an unstamped
    original is recoverable by a re-run; a wrongly-stamped one is not).

    Args:
        records: The mem0 records returned by an entity-scoped metadata scroll.
        entity_uuid: The source record's entity, from
            :func:`resolve_source_entity_uuid`.

    Returns:
        The matching memory ids, SORTED — the migration's stamp order, and its
        report's row order, must not depend on the order Qdrant happened to
        return.
    """
    return sorted(
        record['id']
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get('id'), str)
        and isinstance(record.get('metadata'), dict)
        and record['metadata'].get('kind') in DEMOTED_AD_HOC_KINDS
        and record['metadata'].get('entity_uuid') == entity_uuid
    )
