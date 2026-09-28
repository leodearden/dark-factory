"""Drop Stage-1 deletion-anomaly flags whose swept records a documented recon sweep took.

Stage 1 reads the cycle's buffered ``memory_deleted`` events and reports a Mem0
record deleted seconds after its ledger stamp as an "evidentiary anchor
deletion pattern".  The deletion log shows THAT a record was deleted, never
WHY; the tombstone says why.  In solar_challenge_platform run 09f2829f (finding
c4639ec8) all three "new occurrences" carried ``stage1_cycle_summary_trim`` /
``stage2_cycle_summary_trim`` tombstones from one run: the capped cycle-summary
pool evicting its oldest mirror, as designed, re-flagged every cycle.

**Visibility limit.**  The gate recognises an id as swept only if the id
carries a tombstone or was deleted in THIS cycle's buffered events.  An
untombstoned deletion from an EARLIER cycle is neither, so it reads exactly
like the run ids and present records a flag also names, and is ignored with
them.  A flag re-aggregating such an id alongside benign-tombstoned ones is
therefore dropped.  "Absent from Mem0" cannot close the gap: a run id is
absent too, and would keep every flag.  The cycle whose buffer held that
deletion does see it, and keeps any flag naming it; the Stage-1 prompt tells
the LLM to report such an id in a flag of its own.

This module imports only downward (flag_dedup, recon_self_model, the event
model); none of those may import it back.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Any

from fused_memory.models.reconciliation import EventType, ReconciliationEvent
from fused_memory.reconciliation.flag_dedup import FlagTypeFamily, extract_flag_uuids
from fused_memory.reconciliation.recon_self_model import MEM0_TOMBSTONE_DELETERS

logger = logging.getLogger(__name__)

#: The finding's spellings as measured in reconciliation.db's stage reports.
EVIDENTIARY_ANCHOR_DELETION_FAMILY = FlagTypeFamily(
    name='evidentiary_anchor_deletion',
    spellings=frozenset({
        'mem0_evidentiary_anchor_deletion_pattern',
        'mem0_evidentiary_anchor_full_mirror_loss',
        'cycle_summary_evidentiary_anchor_mirror_deletion_confirmed',
    }),
    drift_token='anchor',
)

_BENIGN_SWEEP_DELETERS: frozenset[str] = frozenset(MEM0_TOMBSTONE_DELETERS)


def _deleted_memory_ids(events: Iterable[ReconciliationEvent]) -> set[str]:
    """Lowercased ids the ``memory_deleted`` events name, cascaded children included."""
    ids: set[str] = set()
    for event in events:
        if event.type != EventType.memory_deleted:
            continue
        children = event.payload.get('cascaded_child_ids')
        named = [event.payload.get('memory_id')]
        if isinstance(children, (list, tuple)):
            named.extend(children)
        ids.update(m.lower() for m in named if isinstance(m, str) and m)
    return ids


async def _read_tombstone(
    memory_service: Any,
    project_id: str,
    memory_id: str,
) -> dict[str, Any] | None:
    """*memory_id*'s tombstone, or ``None`` when there is no readable one."""
    try:
        tombstone = await memory_service.get_mem0_deletion_tombstone(project_id, memory_id)
    except Exception as exc:
        logger.warning(
            'reconciliation.benign_sweep_deletion_tombstone_read_error '
            'project_id=%s memory_id=%s error=%s — treated as untombstoned',
            project_id, memory_id, exc,
        )
        return None
    return tombstone if isinstance(tombstone, dict) else None


def _swept_record(memory_id: str, tombstone: dict[str, Any] | None) -> dict[str, Any]:
    """The provenance of one swept id, classified by who (if anyone) tombstoned it."""
    deleter = tombstone.get('deleter') if tombstone is not None else None
    if tombstone is None:
        classification = 'untombstoned'
    elif isinstance(deleter, str) and deleter in _BENIGN_SWEEP_DELETERS:
        classification = 'benign'
    else:
        classification = 'undocumented_deleter'
    return {
        'memory_id': memory_id,
        'deleter': deleter,
        'deleting_run_id': tombstone.get('deleting_run_id') if tombstone is not None else None,
        'classification': classification,
    }


async def filter_benign_sweep_deletion_flags(
    memory_service: Any,
    project_id: str,
    flags: list[dict[str, Any]],
    *,
    events: Iterable[ReconciliationEvent],
) -> list[dict[str, Any]]:
    """Drop deletion-pattern flags whose every swept id a documented sweep tombstoned.

    A flag's SWEPT ids are the UUIDs it names that EITHER carry a Mem0
    tombstone OR were deleted by one of *events*' ``memory_deleted`` events.
    They are chosen structurally, not from prose: a deleted id cannot pass
    citation verification, so it appears only in the description, next to run
    ids and present records that are neither tombstoned nor deleted.  A swept
    id is ``benign`` when its tombstone's ``deleter`` is in
    :data:`~fused_memory.reconciliation.recon_self_model.MEM0_TOMBSTONE_DELETERS`,
    ``undocumented_deleter`` when it names any other deleter, and
    ``untombstoned`` otherwise.  The events arm is what makes an untombstoned
    deletion visible at all.

    **Fail direction is KEEP.**  A flag is dropped only when it has at least
    one swept id and every one is benign.  A single unexplained swept id, or
    no swept id at all, keeps it; a raising or non-dict tombstone read counts
    as untombstoned.  Only deletions visible to the gate can keep a flag: see
    the module docstring's visibility limit.  Every kept candidate gains ``sweep_deletion_provenance``
    (``{'swept': [...], 'decision': 'kept_unexplained_deletion' |
    'kept_no_swept_ids'}``), so Stage 2 does not re-investigate its benign ids.
    Non-candidates are never looked up or annotated, each distinct id is read
    once per call, and falsy *memory_service* / *project_id* pass through.

    Returns a new list in input order; the input list is never mutated.
    """
    if not memory_service or not project_id:
        return list(flags)

    named_by_pos: dict[int, set[str]] = {
        i: extract_flag_uuids(flag)
        for i, flag in enumerate(flags)
        if EVIDENTIARY_ANCHOR_DELETION_FAMILY.matches(flag.get('flag_type'))
    }
    EVIDENTIARY_ANCHOR_DELETION_FAMILY.log_drift(
        flags, log_event='reconciliation.benign_sweep_deletion_filter_possible_drift',
    )
    if not named_by_pos:
        return list(flags)

    deleted_ids = _deleted_memory_ids(events)
    wanted = sorted(set().union(*named_by_pos.values()))
    tombstones = dict(zip(
        wanted,
        await asyncio.gather(*[
            _read_tombstone(memory_service, project_id, memory_id) for memory_id in wanted
        ]),
        strict=True,
    ))

    kept: list[dict[str, Any]] = []
    for i, flag in enumerate(flags):
        named = named_by_pos.get(i)
        if named is None:
            kept.append(flag)
            continue
        swept = [
            _swept_record(memory_id, tombstones[memory_id])
            for memory_id in sorted(named)
            if tombstones[memory_id] is not None or memory_id in deleted_ids
        ]
        if swept and all(s['classification'] == 'benign' for s in swept):
            logger.info(
                'reconciliation.benign_sweep_deletion_flag_dropped '
                'task_id=%s flag_type=%s memory_ids=%s deleters=%s deleting_run_ids=%s '
                '— expected: every swept id was taken by a documented recon sweep',
                flag.get('task_id'),
                flag.get('flag_type'),
                [s['memory_id'] for s in swept],
                [s['deleter'] for s in swept],
                [s['deleting_run_id'] for s in swept],
            )
            continue
        flag.setdefault(
            'sweep_deletion_provenance',
            {
                'swept': swept,
                'decision': 'kept_unexplained_deletion' if swept else 'kept_no_swept_ids',
            },
        )
        kept.append(flag)
    return kept
