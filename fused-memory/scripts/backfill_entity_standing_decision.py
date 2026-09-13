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

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fused_memory.reconciliation.standing_decision_constants import (
    GROUNDS_STRUCTURAL_SIZE_CONFLATION,
    STATE_ACTIVE,
)
from fused_memory.reconciliation.standing_decision_writer import (
    LedgerUnavailable,
    write_entity_standing_decision,
)
from fused_memory.utils.store_mutation_preflight import (
    StoreMutationUnavailable,
    assert_store_mutation_allowed,
)
from fused_memory.utils.validation import is_full_uuid

logger = logging.getLogger(__name__)

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
    # The ``isinstance`` half is redundant at runtime — ``is_full_uuid`` already
    # rejects every non-str — and carries its weight as the narrowing a plain
    # ``bool`` predicate cannot give a type checker reading the ``str`` return.
    if not isinstance(entity_uuid, str) or not is_full_uuid(entity_uuid):
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


#: Evidence-ref types. β's ``resolve_evidence_refs`` treats ``'mem0'`` as the
#: one locally-resolvable type and stamps every other type
#: ``locally_resolved=False`` without a lookup.
EVIDENCE_TYPE_MEM0 = 'mem0'
EVIDENCE_TYPE_ESCALATION = 'escalation'


def build_evidence_refs(stamp_ids: list[str], entity_uuid: str) -> list[dict[str, str]]:
    """Assemble the row's cited provenance: bare ``{type, id}`` refs.

    Deliberately resolves NOTHING. β's ``resolve_evidence_refs`` is what stamps
    ``locally_resolved`` (mem0 refs looked up against this project, foreign refs
    marked ``False`` without a lookup), and it does so exactly once for both the
    gate and the row payload. A builder that pre-stamped resolution would be a
    second opinion about whether an id exists — one formed without touching the
    store (INV-5).

    The SOURCE record leads, so the row's provenance reads as "this record,
    plus its corroboration" rather than an unordered bag. The remaining stamp
    targets follow in their (sorted) selection order, then the pinned
    human-authored records, then the opening escalation.

    :data:`HUMAN_EVIDENCE_MEMORY_IDS` and :data:`ESCALATION_EVIDENCE_ID` are
    cited unconditionally, including on an idempotent re-run that selects no
    stamp targets at all: the row's provenance is a property of the decision,
    not of how much work a previous run happened to leave undone.

    Args:
        stamp_ids: The evidence-only stamp targets from
            :func:`select_evidence_only_targets`.
        entity_uuid: The decided entity, naming which decision this provenance
            belongs to. No ref embeds it: the ledger row carries the entity in
            its own indexed column, which is the field γ/δ query on, so
            repeating it inside a ref would be a second copy nothing reads.

    Returns:
        Bare refs in citation order, each id appearing exactly once.
    """
    del entity_uuid  # see Args — the row's own column is the entity's home.
    ordered: list[str] = [SOURCE_MEMORY_ID, *stamp_ids, *HUMAN_EVIDENCE_MEMORY_IDS]
    seen: set[str] = set()
    refs: list[dict[str, str]] = []
    for memory_id in ordered:
        if memory_id in seen:
            continue
        seen.add(memory_id)
        refs.append({'type': EVIDENCE_TYPE_MEM0, 'id': memory_id})
    refs.append({'type': EVIDENCE_TYPE_ESCALATION, 'id': ESCALATION_EVIDENCE_ID})
    return refs


#: PRD Open Question 6, RESOLVED (plan design decision 1, recorded at this code
#: site per the batch convention rather than by editing ``plans/``).
#:
#: The four keys the evidence-only stamp writes. ``x_`` is Tier-C: ``memory_
#: metadata.classify_unknown_keys`` exempts the prefix, so the stamp adds no
#: ``code=unknown_key`` census line — exactly the affordance
#: ``docs/task-authoring.md`` §Tier-C points ad-hoc annotations at.
EVIDENCE_ONLY_STATUS_KEY = 'x_standing_decision_status'
EVIDENCE_ONLY_ENTITY_KEY = 'x_standing_decision_entity_uuid'
EVIDENCE_ONLY_GROUNDS_KEY = 'x_standing_decision_grounds'
EVIDENCE_ONLY_MIGRATED_AT_KEY = 'x_standing_decision_migrated_at'

#: The value that marks a record superseded by a ledger row. An operator
#: enumerates the whole demoted population with
#: ``get_memories_by_metadata('reify', {EVIDENCE_ONLY_STATUS_KEY: this})``.
EVIDENCE_ONLY_STATUS = 'evidence_only'


def build_evidence_only_patch(
    *, entity_uuid: str, grounds: str, migrated_at: str
) -> dict[str, str]:
    """Build the metadata patch that demotes one ad-hoc original to evidence.

    FLAT SCALARS, not one nested dict, and that is the substance of Open
    Question 6's resolution: Qdrant payload filters do exact-match on scalars,
    so four flat keys leave the demoted population ENUMERABLE —
    ``get_memories_by_metadata(PROJECT_ID, {EVIDENCE_ONLY_STATUS_KEY:
    EVIDENCE_ONLY_STATUS})`` returns every record this migration touched, which
    is the operator affordance the demotion exists to provide. A single nested
    ``{'standing_decision': {...}}`` value would be unqueryable, and the stamp
    would be legible only to a human reading one record at a time.

    ``entity_uuid`` and ``grounds`` ride as SEPARATE fields rather than δ's
    joined ``f'{uuid}:{grounds}'`` ``standing_decision_id``: structured data
    instead of a meaningful string, and no second site re-deriving that join.

    The patch is METADATA-ONLY and merge-mode at the call site, so the record's
    content and its existing metadata survive untouched — the stamp is an
    annotation, not a rewrite.

    Args:
        entity_uuid: The decided entity, from :func:`resolve_source_entity_uuid`.
        grounds: The ledger row's grounds — :data:`GROUNDS`.
        migrated_at: When this run wrote the row, ISO-8601 UTC. Echoed verbatim
            so a reader can line the stamp up against the row's ``decided_at``.

    Returns:
        The four-key patch.
    """
    return {
        EVIDENCE_ONLY_STATUS_KEY: EVIDENCE_ONLY_STATUS,
        EVIDENCE_ONLY_ENTITY_KEY: entity_uuid,
        EVIDENCE_ONLY_GROUNDS_KEY: grounds,
        EVIDENCE_ONLY_MIGRATED_AT_KEY: migrated_at,
    }


@dataclass(frozen=True)
class BackfillPlan:
    """Everything this migration decided, before it touched anything.

    Frozen: the plan is computed once from a snapshot of the corpus and the
    ledger, then applied. Nothing between those two moments may edit it, so the
    report a dry run prints is exactly the work an ``--apply`` run would do.

    :param entity_uuid: The decided entity, derived from the source record.
    :param grounds: The ledger row's grounds — always :data:`GROUNDS` today,
        carried on the plan so the report and the write read the same value.
    :param needs_ledger_write: Whether an ACTIVE row still has to be written.
    :param stamp_targets: The originals still missing an evidence-only stamp.
    :param evidence_refs: The bare provenance refs the row will cite.
    """

    entity_uuid: str
    grounds: str
    needs_ledger_write: bool
    stamp_targets: tuple[str, ...]
    evidence_refs: tuple[dict[str, str], ...]


def is_already_stamped(record: Any) -> bool:
    """True iff *record* carries this migration's evidence-only status.

    Keys on :data:`EVIDENCE_ONLY_STATUS_KEY` alone rather than on all four
    stamp fields: that key is what an operator's enumerating query filters on,
    so it is the field whose presence MEANS demoted. A record carrying it is
    not re-stamped, which is what makes the stamping leg idempotent.
    """
    metadata = record.get('metadata') if isinstance(record, dict) else None
    if not isinstance(metadata, dict):
        return False
    return metadata.get(EVIDENCE_ONLY_STATUS_KEY) == EVIDENCE_ONLY_STATUS


def plan_backfill(
    source_record: Any, scrolled_records: list[Any], active_row: Any
) -> BackfillPlan:
    """Decide the whole migration from a snapshot. Pure, sync, no store.

    Every live read is the caller's responsibility (:func:`run_backfill` owns
    them), which is what leaves the entire decision surface directly
    unit-testable — and what lets a dry run rehearse the real decision rather
    than a simplified echo of it.

    Both legs are INDEPENDENTLY idempotent, and the plan is where that lives:

    * the ledger write is planned unless an ACTIVE row already exists. A row in
      any OTHER state is planned again, deliberately: α's
      ``get_active_entity_standing_decision`` gates on ``state='active'``, so a
      TTL-expired or revoked row leaves γ/δ blind. Reading "a row exists" as
      "the work is done" would silently leave a lapsed decision unenforced.
    * a stamp is planned per record still missing :func:`is_already_stamped`.

    So a re-run after a partial failure completes exactly what the first run
    missed, and a re-run after a complete one plans nothing at all.

    Args:
        source_record: ``get_memory_by_id`` for :data:`SOURCE_MEMORY_ID`.
        scrolled_records: The entity-scoped metadata scroll.
        active_row: ``get_active_entity_standing_decision`` for this entity, or
            ``None``. Any row whose ``state`` is not ``active`` is treated as
            absent for the write decision.

    Returns:
        The frozen plan.

    Raises:
        BackfillSourceInvalid: Propagated from
            :func:`resolve_source_entity_uuid` — an unusable source stops the
            run before anything is planned.
    """
    entity_uuid = resolve_source_entity_uuid(source_record)
    selected = select_evidence_only_targets(scrolled_records, entity_uuid)
    stamped_ids = {
        record['id']
        for record in scrolled_records
        if isinstance(record, dict) and is_already_stamped(record)
    }
    return BackfillPlan(
        entity_uuid=entity_uuid,
        grounds=GROUNDS,
        needs_ledger_write=getattr(active_row, 'state', None) != STATE_ACTIVE,
        stamp_targets=tuple(
            memory_id for memory_id in selected if memory_id not in stamped_ids
        ),
        evidence_refs=tuple(build_evidence_refs(selected, entity_uuid)),
    )


#: Who authorized this write. β's ``authorized_by`` arm SKIPS the evidence gate
#: and records an operator-authorization provenance entry naming this value, so
#: the row says on its face that a migration — not an LLM mid-cycle — wrote it.
#:
#: The bypass is used rather than leaned on: the cited evidence includes the
#: human-authored record in :data:`HUMAN_EVIDENCE_MEMORY_IDS`, so the row would
#: satisfy arm 1 on its own merits if that record were metadata-discoverable.
AUTHORIZED_BY = 'backfill_entity_standing_decision (task 2900 η)'

#: Stamped on every ``update_memory`` so the amendment is attributable in the
#: mem0 history a content-amendment audit reads.
WRITE_SOURCE = 'backfill_entity_standing_decision'
WRITE_REASON = (
    'superseded by an entity_standing_decision ledger row (PRD decision 5); '
    'retained as evidence only'
)

#: A record outcome that means the run did not fully land. Graded by
#: :func:`resolve_exit_code` off the SAME set the report uses, so the exit code
#: and the artifact can never disagree about whether a run was clean.
ERROR_OUTCOMES = frozenset({'stamp_error'})


async def run_backfill(memory_service: Any, *, apply: bool) -> dict[str, Any]:
    """Read the live state, plan, then (when applying) write the row and stamps.

    ORDER IS LOAD-BEARING and NOT reversible: **the ledger row is written
    first, and a stamp failure never rolls it back.** The row is the
    authoritative machine-consulted artifact — PRD decision 6 makes the ledger
    kind the sole form γ and δ read — while the stamps are advisory provenance
    on records nothing reads. Stamping first would leave originals asserting
    they are superseded by a row that does not exist: a strictly worse
    intermediate state than an un-stamped original, because it misleads the one
    audience the stamp exists for. Both legs are independently idempotent
    (:func:`plan_backfill`), so a re-run completes whatever the first missed.

    This function does NOT run the store-mutation preflight; :func:`main` does,
    once per run, ahead of both the scan and the first mutation. Probing here
    would put a ``RuntimeError`` subclass inside the per-record ``except``
    below, downgrading a run-wide environment denial into N ``stamp_error``
    rows in a report that otherwise reads as a completed sweep.

    Args:
        memory_service: A live service with ``recon_ledger`` attached.
        apply: ``False`` (the default everywhere) rehearses every read and
            decision and withholds only the writes.

    Returns:
        A JSON-serializable report: the decided entity, a ``ledger`` disposition
        (``would_write`` / ``written`` / ``already_migrated``) and one row per
        stamp target (``would_stamp`` / ``stamped`` / ``stamp_error``).
    """
    # The ledger is read for the idempotence check BEFORE β ever sees the
    # service, so an unwired one would surface here as an opaque
    # ``AttributeError`` on ``None`` — precisely the failure β's typed
    # LedgerUnavailable was introduced to replace for η's direct callers. Raise
    # β's own error rather than minting a second one, so both the migration's
    # read and its write report an unwired ledger identically (INV-5).
    ledger = getattr(memory_service, 'recon_ledger', None)
    if ledger is None:
        raise LedgerUnavailable(
            'backfill_entity_standing_decision: no recon_ledger wired on '
            f'memory_service for project {PROJECT_ID!r} — the migration cannot '
            'tell whether the row already exists, and must not guess; hint: '
            'attach a ReconLedgerStore before calling (main() does this from '
            'the configured reconciliation data_dir).'
        )

    source_record = await memory_service.get_memory_by_id(PROJECT_ID, SOURCE_MEMORY_ID)
    entity_uuid = resolve_source_entity_uuid(source_record)
    scrolled = await memory_service.get_memories_by_metadata(
        PROJECT_ID, {'entity_uuid': entity_uuid}
    )
    active_row = await ledger.get_active_entity_standing_decision(
        PROJECT_ID, entity_uuid
    )
    plan = plan_backfill(source_record, scrolled or [], active_row)

    report: dict[str, Any] = {
        'apply': apply,
        'project_id': PROJECT_ID,
        'source_memory_id': SOURCE_MEMORY_ID,
        'entity_uuid': plan.entity_uuid,
        'grounds': plan.grounds,
        'evidence_refs': [dict(ref) for ref in plan.evidence_refs],
        'ledger': 'already_migrated',
        'records': [],
    }

    if plan.needs_ledger_write and not apply:
        report['ledger'] = 'would_write'
    elif plan.needs_ledger_write:
        # Loud by omission: β raises (LedgerUnavailable, a graphiti sampling
        # error, α's grounds/expiry validation) rather than returning a status,
        # and those propagate from here UNCAUGHT — before the stamp loop below
        # has begun, so no original is ever marked superseded by a row that
        # does not exist.
        written = await write_entity_standing_decision(
            memory_service,
            project_id=PROJECT_ID,
            entity_uuid=plan.entity_uuid,
            grounds=plan.grounds,
            evidence=[dict(ref) for ref in plan.evidence_refs],
            authorized_by=AUTHORIZED_BY,
        )
        report['ledger'] = 'written'
        report['decided_at'] = written['decided_at']
        report['expires_at'] = written['expires_at']
        report['edge_count_at_decision'] = written['edge_count_at_decision']

    migrated_at = report.get('decided_at') or datetime.now(UTC).isoformat()
    patch = build_evidence_only_patch(
        entity_uuid=plan.entity_uuid, grounds=plan.grounds, migrated_at=migrated_at
    )
    for memory_id in plan.stamp_targets:
        report['records'].append(
            {'memory_id': memory_id, 'outcome': 'would_stamp', 'patch': dict(patch)}
            if not apply
            else await _stamp_one(memory_service, memory_id, patch)
        )
    return report


async def _stamp_one(
    memory_service: Any, memory_id: str, patch: dict[str, str]
) -> dict[str, Any]:
    """Demote one original to evidence-only, reporting its outcome in isolation.

    Every failure is captured PER RECORD and none rolls the ledger row back:
    the row is already the authoritative artifact, and a re-run stamps whatever
    is still missing. Two failure shapes are caught, because ``update_memory``
    has two: it RAISES on a vocabulary rejection but REPORTS a not-found (and
    other refusals) by returning an ``{'error_type': ...}`` envelope — so a
    caller that guarded only against exceptions would score a refused write as
    a stamp.
    """
    try:
        response = await memory_service.update_memory(
            memory_id=memory_id,
            project_id=PROJECT_ID,
            metadata_patch=dict(patch),
            metadata_mode='merge',
            reason=WRITE_REASON,
            agent_id=WRITE_SOURCE,
            _source=WRITE_SOURCE,
        )
    except Exception as exc:
        return {
            'memory_id': memory_id,
            'outcome': 'stamp_error',
            'error_type': type(exc).__name__,
            'error': str(exc),
        }
    if isinstance(response, dict) and response.get('error_type'):
        return {
            'memory_id': memory_id,
            'outcome': 'stamp_error',
            'error_type': response.get('error_type'),
            'error': response.get('error'),
        }
    return {'memory_id': memory_id, 'outcome': 'stamped'}


def resolve_exit_code(report: dict[str, Any]) -> int:
    """0 on a clean run, 1 when any record did not land as planned.

    An honest non-zero is the one signal an automated caller can act on: the
    stamps are per-record and partial failure is a real shape, so a bare 0
    would let a half-stamped corpus read as a completed migration.
    """
    return 1 if any(
        row.get('outcome') in ERROR_OUTCOMES for row in report.get('records') or []
    ) else 0


def build_parser() -> argparse.ArgumentParser:
    """The CLI. Dry run is the DEFAULT, and ``--apply`` is the only way past it."""
    parser = argparse.ArgumentParser(
        description=(
            'One-shot migration of reify mem0 record b0057f3d into an '
            'entity_standing_decision ledger row, stamping the ad-hoc mem0 '
            'originals evidence-only (PRD leaf η, task 2900).'
        ),
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Commit the ledger row and the stamps. Without it the run is a '
             'full rehearsal that reads and decides everything but writes '
             'nothing. Must be run from the fused-memory MCP server host: an '
             'in-sandbox --apply is refused by the store-mutation preflight.',
    )
    parser.add_argument(
        '--json-out', default=None,
        help='Write the JSON report here in addition to stdout.',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to a fused-memory config; sets CONFIG_PATH for this run.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build a live service, run the migration, print the report, exit graded."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        stream=sys.stderr,
    )
    args = build_parser().parse_args(argv)

    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    # Fail-CLOSED capability preflight — ONE probe per run, ahead of the scan
    # AND of the first mutation, which is what makes "--apply is an operator
    # action" enforceable rather than advisory. Kept out of `run_backfill` so a
    # run-wide environment denial cannot be swallowed by that function's
    # per-record error handling and re-reported as N stamp failures.
    if args.apply:
        try:
            assert_store_mutation_allowed(
                operation='backfill_entity_standing_decision --apply'
            )
        except StoreMutationUnavailable:
            logger.error(
                'backfill_entity_standing_decision: --apply NOT started '
                "(fail-closed) — this process cannot write mem0's history "
                'directory, so a stamp would patch a record and then fail to '
                'record the change. Nothing was scrolled, no ledger row was '
                'written and no record was stamped. Re-run from the '
                'fused-memory MCP server host (the unsandboxed owner of the '
                'store). To obtain the report safely from anywhere, re-run '
                'without --apply.'
            )
            raise

    async def _run_live() -> dict[str, Any]:
        # Deferred so importing this module — which the tests do, by path —
        # never constructs a backend.
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
        from fused_memory.reconciliation.recon_ledger import (  # noqa: PLC0415
            ReconLedgerStore,
        )
        from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

        config = FusedMemoryConfig()
        memory = MemoryService(config)
        ledger = ReconLedgerStore(
            Path(config.reconciliation.data_dir) / 'reconciliation.db'
        )
        try:
            await memory.initialize()
            await ledger.initialize()
            memory.set_recon_ledger(ledger)
            return await run_backfill(memory, apply=args.apply)
        finally:
            await ledger.close()
            if hasattr(memory, 'close'):
                await memory.close()

    report = asyncio.run(_run_live())

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        Path(args.json_out).write_text(rendered + '\n', encoding='utf-8')
    print(rendered)
    if not args.apply:
        print('DRY RUN — nothing was modified. Re-run with --apply to commit.')
    return resolve_exit_code(report)


if __name__ == '__main__':
    sys.exit(main())
