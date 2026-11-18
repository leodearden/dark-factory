"""The write boundary and two-store lockstep behind ``scripts/normalize_topic_slugs.py`` (task 4878).

The migration spans TWO stores
------------------------------
Most of the work patches Mem0 record metadata.  But
``metadata.x_recon_consolidation_gate.topic`` is a Tier-C block on **task**
metadata — read by ``reconciliation/consolidation_gate.py`` and enforced on
the ``done`` transition by ``middleware/task_interceptor.py`` — a genuinely
different store, reached through the MCP server rather than a backend call.

Renaming the memories without moving that block would leave a gate task's
closure check scrolling a slug no record carries any more: the uncloseable-gate
trap re-created at a NEW address, which is strictly worse than leaving the
snake_case slug alone because the operator would then also have to discover
that this migration caused it.  So the two halves are planned as a pair and
refused as a pair.

On the live path that promise rests on the GATE CENSUS, not on an operator's
memory of which topics are gated: ``census_consolidation_gates`` reads every
gate block out of each named task store, and the script's ``run`` refuses
``--apply`` unless the census covers every swept project.  A store the census
could not read is a gap it names, never a silent "no gates here".

Every slug moves all-or-nothing
-------------------------------
When any member write of a GATED slug fails, ``rename_group`` undoes the
members that moved, because a gate left on a vacated slug is uncloseable.

An UNGATED slug is undone when its attempt leaves a member on the legacy
value, because a split slug cannot be finished by re-running: the collision
guard sees the members on the new slug as incumbents and refuses the
stragglers as ``slug_collision``.  So the members this run moved are rolled
back, the report lists the group under ``ungated_partial_rename`` (and the
residue probe under ``legacy_slug_residue``), and a re-run finishes the move
once the failure is resolved.  A member that vanished, or that another writer
moved to a third value, is no longer on the legacy slug: it splits nothing and
triggers no undo.

The residual is a member the rollback leaves on the new slug: one whose undo
itself fails (named in ``undo_failures``), or one another writer had already
put there between plan and apply, which this run never moved and so never
undoes.  The group's ``on_new_slug_memory_ids`` names every such member, and
while it is non-empty the slug is still split and a re-run refuses it as
``slug_collision``.  The hand remedy is in
``fused-memory/scripts/README_normalize_topic_slugs.md``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fused_memory.maintenance.topic_slug_gates import GateGroup
from fused_memory.maintenance.topic_slug_plan import Rename
from fused_memory.reconciliation.consolidation_gate import GATE_METADATA_KEY

logger = logging.getLogger(__name__)

#: Written to every ``update_memory`` so the write journal attributes each
#: rename to this sweep rather than to a generic ``mcp_tool``.  The amendment
#: storm alarm reads this field; a bulk run under the default source would
#: look exactly like the runaway rewrite that alarm exists to catch.
WRITE_SOURCE = 'normalize_topic_slugs'

#: Recorded on the write journal row beside the patch.
WRITE_REASON = 'corpus-wide topic-slug normalization (task 4878)'

#: Outcomes whose rows still carry the LEGACY topic value at probe time and
#: were put there by this run's own rehearsal.  The residue probe subtracts
#: only these.  A failed write also still carries it — and that is genuine
#: residue, correctly left in.
REHEARSED_OUTCOMES = frozenset({'would_rename'})

#: Outcomes that actually moved (or intended to move) a slug.  Anything else
#: never touched one, so there is nothing to drain and nothing to probe.
MOVED_OUTCOMES = frozenset({'renamed', 'would_rename', 'update_failed', 'rename_error'})

#: Outcomes after which THIS run's rename of a member landed, or would in a
#: dry run.
LANDED_OUTCOMES: frozenset[str] = frozenset({'renamed', 'would_rename'})

#: Outcomes after which a member sits on the new slug, or would in a dry run:
#: moved there by this run, or found there already.  Every other outcome is a
#: member that did not stand.
STOOD_OUTCOMES: frozenset[str] = LANDED_OUTCOMES | {'already_normalized'}

#: Outcomes after which a member still carries the LEGACY value: its rename was
#: attempted and did not land.  A record that is gone (``memory_not_found``) or
#: that now carries a third value (``topic_moved_since_plan``) did not stand
#: either, but it is not on the legacy slug, so it splits nothing.
LEFT_ON_LEGACY_OUTCOMES: frozenset[str] = MOVED_OUTCOMES - STOOD_OUTCOMES


# ---------------------------------------------------------------------------
# The single write boundary
# ---------------------------------------------------------------------------

async def rename_one(memory_service, rename: Rename, *, apply: bool) -> dict:
    """Move one record's topic, or report precisely why it was not moved.

    The only function here that writes.  Order is copied from
    ``retro_stamp_topics.stamp_one`` and is load-bearing:

    1. **live re-read** (``get_memory_by_id``).  The plan's ids come off a
       scroll that may be minutes old on a corpus orchestrators are writing
       to.  Reading first is what turns a record consolidated away since the
       scroll into a report line instead of a Qdrant ``set_payload`` that
       acknowledges a write to nothing.
    2. **decide** against the LIVE value, not the planned one.  A topic that
       moved since the plan makes this row stale; applying anyway would
       overwrite a live value with a fold of a value that no longer exists —
       a data loss the report would score as a success.  A live topic that
       already equals the target ends the call: a no-op ``update_memory``
       would still journal a write op, still count toward the
       content-amendment storm alarm, and would inflate the renamed count
       with records that gained nothing.
    3. **write**, metadata-only, patch = exactly ``{'topic': new}`` under
       ``metadata_mode='merge'``.  Narrow on purpose: a wider patch would let
       a normalization silently clobber ``canonical``, ``supersedes`` or the
       consolidation bookkeeping across the whole target population at once.
       Never a delete-and-recreate — that would mint a new memory id,
       orphaning every reference to the old one and destroying ``created_at``,
       the field consolidation uses to pick a canonical.

    Every await is individually guarded.  The artifact is written after the
    loop, so an exception escaping here would discard everything the sweep had
    already learned; one backend hiccup must cost one report row instead.

    Args:
        memory_service: The injected ``MemoryService``.
        rename: The frozen plan row.
        apply: ``False`` performs the read and the decision and issues no
            write, so the rehearsal exercises the SAME path ``--apply`` runs
            rather than predicting it.

    Returns:
        A report-ready dict always carrying ``outcome``, plus whatever that
        outcome needs to be actionable without a follow-up query
        (``existing_topic``, ``error``, ``error_type``, ``response``).
    """
    base = {
        'project_id': rename.project_id,
        'memory_id': rename.memory_id,
        'old_topic': rename.old_topic,
        'new_topic': rename.new_topic,
    }

    try:
        record = await memory_service.get_memory_by_id(
            project_id=rename.project_id, memory_id=rename.memory_id,
        )
    except Exception as exc:
        return {
            **base,
            'outcome': 'rename_error',
            'error': f'{type(exc).__name__}: {exc}',
        }
    if record is None:
        return {**base, 'outcome': 'memory_not_found'}

    metadata = dict(record.get('metadata') or {})
    existing_topic = metadata.get('topic')

    if existing_topic == rename.new_topic:
        return {**base, 'outcome': 'already_normalized', 'existing_topic': existing_topic}
    if existing_topic != rename.old_topic:
        return {
            **base,
            'outcome': 'topic_moved_since_plan',
            'existing_topic': existing_topic,
        }

    if not apply:
        return {**base, 'outcome': 'would_rename', 'existing_topic': existing_topic}

    try:
        response = await memory_service.update_memory(
            memory_id=rename.memory_id,
            project_id=rename.project_id,
            metadata_patch={'topic': rename.new_topic},
            metadata_mode='merge',
            reason=WRITE_REASON,
            _source=WRITE_SOURCE,
        )
    except Exception as exc:
        return {
            **base,
            'outcome': 'rename_error',
            'error': f'{type(exc).__name__}: {exc}',
        }
    # ``update_memory`` reports a not-found (and some other rejections) by
    # RETURNING a structured envelope rather than raising, so a caller that
    # only guarded against exceptions would score a refused write as a rename
    # -- and the residue probe would then find the legacy slug still populated
    # with no explanation anywhere in the report.
    if isinstance(response, dict) and response.get('error_type'):
        return {
            **base,
            'outcome': 'update_failed',
            'error_type': response.get('error_type'),
            'error': response.get('error'),
            'response': response,
        }
    return {**base, 'outcome': 'renamed', 'response': response}


# ---------------------------------------------------------------------------
# The two-store lockstep — memories and the consolidation gate move together
# ---------------------------------------------------------------------------

class WriteRejectedError(RuntimeError):
    """An MCP tool call that returned an error envelope inside a success frame."""


def assert_write_accepted(result: object, *, tool: str = 'update_task') -> None:
    """Raise :class:`WriteRejectedError` on a tool-level rejection.

    Restated locally from ``migrate_task_metadata_to_x_namespace.py`` for the
    same reason it exists there: an accepted ``update_task`` returns
    ``{'id', 'message', 'updated', 'updated_task'}`` and a REFUSED one returns
    ``{'success': False, 'error': <code>, ...}`` — inside the SAME successful
    JSON-RPC envelope.  Nothing raises.  A caller guarding only against
    exceptions therefore scores a refused gate move as a completed one, which
    is precisely the state this lockstep exists to make unreachable.

    The rejection most likely to fire here is the ``done_provenance``
    write-authority floor, which refuses any whole-blob metadata replace of a
    done/merged task.  This writer never sends one — see :func:`rename_group`
    — but the guard is what makes that a checked claim rather than a hope.
    """
    if not isinstance(result, dict):
        return
    if result.get('success') is False or result.get('error'):
        raise WriteRejectedError(
            f'{tool} was REJECTED by the server (the JSON-RPC envelope was '
            f'still a success, so this would otherwise be invisible): '
            f'error={result.get("error")!r} '
            f'error_type={result.get("error_type")!r} '
            f'hint={result.get("hint")!r}'
        )


def _gate_row(group: GateGroup, outcome: str, **extra) -> dict:
    return {
        'project_id': group.project_id,
        'gate_task_id': group.gate_task_id,
        'gate_project_root': group.gate_project_root,
        'old_topic': group.old_topic,
        'new_topic': group.new_topic,
        'memory_ids': [r.memory_id for r in group.renames],
        'outcome': outcome,
        **extra,
    }


async def _undo_group(memory_service, results: list[dict], *, rescore_as: str) -> list[dict]:
    """Put back exactly the writes THIS run made, and report what would not go.

    An undo is just a rename in the opposite direction, so it runs through
    :func:`rename_one` rather than a second write path — same live re-read,
    same narrow metadata-only patch, same guard posture.

    Only rows scored ``renamed`` are reversed.  A record that was
    ``already_normalized`` arrived at the new value before this sweep touched
    it; "undoing" it would be this script inventing a write of its own, and
    would move a record the operator never asked it to move.  Each reversed
    row is re-scored *rescore_as*, the bucket of the hold that reversed it.
    """
    failures: list[dict] = []
    for result in results:
        if result.get('outcome') != 'renamed':
            continue
        undo = await rename_one(
            memory_service,
            Rename(
                project_id=result['project_id'],
                memory_id=result['memory_id'],
                old_topic=result['new_topic'],
                new_topic=result['old_topic'],
            ),
            apply=True,
        )
        if undo.get('outcome') != 'renamed':
            failures.append({
                'memory_id': result['memory_id'],
                'outcome': undo.get('outcome'),
                'error': undo.get('error') or undo.get('error_type'),
            })
        result['undone'] = undo.get('outcome') == 'renamed'
        result['outcome'] = rescore_as
    return failures


async def _hold_split_ungated_group(
    memory_service, group: GateGroup, results: list[dict],
) -> dict | None:
    """Roll back an ungated slug that this run left split, and report the hold.

    The entry is described from the PRE-undo rows: once the moved members are
    re-scored, nothing reads as moved.  An attempt that split nothing — every
    member stood, or a straggler vanished or moved elsewhere — keeps its
    writes and returns ``None``.

    ``on_new_slug_memory_ids`` is the state the hold leaves behind, measured
    rather than inferred: a member another writer had already put on the new
    slug, which the undo never touches, plus every member whose undo failed.
    Non-empty, it means the slug is still split.
    """
    entry = describe_partial_ungated_group(group, results)
    if entry is None:
        return None
    undo_failures = await _undo_group(
        memory_service, results, rescore_as='ungated_partial_rename')
    on_new_slug = sorted(
        r['memory_id'] for r in results
        if r.get('outcome') == 'already_normalized' or r.get('undone') is False
    )
    logger.warning(
        'UNGATED SLUG HELD project=%s topic=%s -> %s: %d member(s) left on the '
        'legacy value, so %d moved member(s) were rolled back; %d member(s) '
        'still on the new slug, %d undo failure(s).',
        group.project_id, group.old_topic, group.new_topic,
        len(entry['failed_memory_ids']),
        sum(1 for r in results if r.get('undone')),
        len(on_new_slug),
        len(undo_failures),
    )
    return {**entry, 'on_new_slug_memory_ids': on_new_slug, 'undo_failures': undo_failures}


@dataclass(frozen=True)
class GroupOutcome:
    """What :func:`rename_group` did to one slug, one field per destination.

    Attributes:
        results: The member rows, in plan order.
        gate_row: A gated group's gate row — patched, would-patch or held.
        held_entry: The ``ungated_partial_rename`` entry of an ungated group
            this run left split and rolled back.
    """

    results: list[dict]
    gate_row: dict | None = None
    held_entry: dict | None = None


async def rename_group(
    memory_service,
    group: GateGroup,
    *,
    apply: bool,
    client: Any = None,
) -> GroupOutcome:
    """Move one slug in both stores, or leave it exactly where it started.

    Order, and why it is this way round:

    1. **the memory renames**, all of them, through :func:`rename_one`;
    2. **the gate patch**, and only once every one of them succeeded.

    The gate is second because it is the single atomic write of the pair: N
    record patches cannot be made atomic, so the one call that can be is the
    commit point.  If it is refused, step 1 is UNDONE, so the pair's net state
    is unchanged and the group is reported as ``gate_lockstep_failed``.  A
    failed half never stands alone in either direction — memories on the new
    slug with the gate on the old, or the gate on the new with the records on
    the old, are the SAME uncloseable-gate failure, just seen from two sides.

    (The undo is why "hold the whole group" is achievable at all with
    memory-first ordering.  It is deliberately narrow: it reverses only the
    writes this run made, and a reversal that itself fails is named on the row
    — an operator must never have to infer that a half-applied state exists.)

    An UNGATED group is the common case and short-circuits after step 1: it
    needs no client, so a sweep that touches no gate never opens a socket.
    It is undone only when step 1 left it split (a member moved while another
    is still on the legacy value), because a split slug cannot be finished by
    re-running; see the module docstring's "Every slug moves all-or-nothing".
    A GATED group with no client is the same verdict as a refused gate —
    unable to move it is not permission to leave it behind.

    Returns:
        A :class:`GroupOutcome`: the member rows, a gated group's gate row,
        and a held ungated group's ``ungated_partial_rename`` entry.  Rows
        whose write was undone are re-scored ``gate_lockstep_failed`` or
        ``ungated_partial_rename`` so the report never claims a rename that no
        longer stands; rows that never wrote keep their own diagnosis, which
        is the actionable one.
    """
    results = [
        await rename_one(memory_service, rename, apply=apply)
        for rename in group.renames
    ]
    if group.gate_task_id is None:
        return GroupOutcome(
            results,
            held_entry=await _hold_split_ungated_group(memory_service, group, results),
        )

    failed = [r for r in results if r.get('outcome') not in STOOD_OUTCOMES]
    if failed:
        undo_failures = await _undo_group(
            memory_service, results, rescore_as='gate_lockstep_failed')
        logger.warning(
            'GATE LOCKSTEP HELD project=%s topic=%s -> %s: %d of %d memory writes '
            'did not stand, so gate task %s was NOT patched.',
            group.project_id, group.old_topic, group.new_topic,
            len(failed), len(results), group.gate_task_id,
        )
        return GroupOutcome(results, gate_row=_gate_row(
            group, 'gate_lockstep_failed',
            half='memory',
            error=f'{len(failed)} of {len(results)} memory renames did not stand',
            failed_outcomes=sorted({str(r.get('outcome')) for r in failed}),
            undo_failures=undo_failures,
            note=(
                'the gate patch was withheld, so the gate still points at the '
                'legacy slug — resolve the memory-side failures and re-run'
            ),
        ))

    if not apply:
        return GroupOutcome(
            results, gate_row=_gate_row(group, 'would_patch_gate', patch=group.gate_block))

    error: str | None = None
    if client is None:
        error = (
            'no MCP client: the gate half of this pair could not be attempted, '
            'and an unmovable gate is refused exactly like a refused one'
        )
    else:
        payload = {
            'id': group.gate_task_id,
            'project_root': group.gate_project_root,
            # A NARROW merge of one Tier-C block, never a whole-blob replace:
            # `update_task` refuses any replace carrying `done_provenance`, and
            # a replace here would also stake the task's entire metadata on
            # this script having read it back correctly.
            'metadata': {GATE_METADATA_KEY: dict(group.gate_block or {})},
            'metadata_mode': 'merge',
        }
        try:
            response = await client.call_tool('update_task', payload)
            assert_write_accepted(response, tool='update_task')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        else:
            return GroupOutcome(results, gate_row=_gate_row(
                group, 'gate_patched', patch=group.gate_block, response=response))

    undo_failures = await _undo_group(
        memory_service, results, rescore_as='gate_lockstep_failed')
    logger.warning(
        'GATE LOCKSTEP HELD project=%s topic=%s -> %s: gate task %s was not '
        'patched (%s); %d memory write(s) undone, %d undo failure(s).',
        group.project_id, group.old_topic, group.new_topic, group.gate_task_id,
        error, len(group.renames), len(undo_failures),
    )
    return GroupOutcome(results, gate_row=_gate_row(
        group, 'gate_lockstep_failed',
        half='gate', error=error, undo_failures=undo_failures,
        note=(
            'the memory half was undone so the slug is unchanged in BOTH '
            'stores; a gate left pointing at a slug no record carries is '
            'permanently uncloseable, which is worse than not migrating'
        ),
    ))


def describe_partial_ungated_group(group: GateGroup, results: list[dict]) -> dict | None:
    """The report entry for an ungated slug that only partly moved, else ``None``.

    Pure, and read from the rows BEFORE any rollback.  A gated group is never
    described here: its gate row already reports a held pair.
    ``failed_outcomes`` has the gate row's key and shape, so the two read the
    same way.  A slug is split when a member stood on the new slug while
    another is "failed": its rename did not land, so it still carries the
    legacy value.  A vanished or concurrently moved member splits nothing and
    is reported only in its own bucket.

    ``moved_memory_ids`` are the members whose rename this run landed (or, in
    a dry run, would land): the ones the hold rolls back.  A member another
    writer had already put on the new slug stood without being moved, so it
    is not among them.
    """
    if group.gate_task_id is not None:
        return None
    stood = [r for r in results if r.get('outcome') in STOOD_OUTCOMES]
    failed = [r for r in results if r.get('outcome') in LEFT_ON_LEGACY_OUTCOMES]
    if not stood or not failed:
        return None
    return {
        'reason': 'ungated_partial_rename',
        'project_id': group.project_id,
        'old_topic': group.old_topic,
        'new_topic': group.new_topic,
        'moved_memory_ids': sorted(
            r['memory_id'] for r in stood if r.get('outcome') in LANDED_OUTCOMES),
        'failed_memory_ids': sorted(r['memory_id'] for r in failed),
        'failed_outcomes': sorted({str(r.get('outcome')) for r in failed}),
        'note': (
            'a member of this slug was left on the legacy value, so the '
            'members this run moved are rolled back to it (in a dry run the '
            'hold is predicted, not yet made).  With on_new_slug_memory_ids '
            'empty the slug is whole on the legacy value and a re-run finishes '
            'it once the failure is resolved; non-empty, the slug is still '
            'split and a re-run refuses it as slug_collision until the members '
            'it names are put back by hand (see the module docstring)'
        ),
    }
