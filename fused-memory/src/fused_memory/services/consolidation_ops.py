"""The RETAIN ARM of a consolidation — mint, peer tag, closure — as one op.

What lives here is the half of ``consolidate_memories`` that ADDS: write
the canonical, stamp the topic onto the peers that are kept in place, and
list the topic's resulting closure. The delete arm — supersedes, child
reparenting, corroboration, tombstones — stays in the tool. They are two
mechanisms with independent axes of change, and separating them is what
lets the retain arm be reached without the delete arm's machinery.

Reached without a server, a tool closure, or an event-loop-bound
``mcp.tool()`` registration, which is the whole point: the auto-
consolidation executor has no MCP boundary in front of it and must run
the SAME code ``server/tools.py::consolidate_memories`` runs, not a
second implementation of it.

ONE LOOP, ONE CLASSIFIER, ONE SCROLL (INV-5). Tag-only is a MODE of
:func:`execute_retain_consolidation` — chosen by ``canonical_content is
None`` — never a sibling function, so the peer-tag loop has exactly one
home and the two paths cannot drift. :func:`patch_memory_metadata` is
likewise the single home of ``update_memory``'s split contract — a
refusal is RETURNED, every other failure is RAISED — for all three of
the op's patch sites: the retain tag here, and the child reparent and
supersedes narrowing the tool keeps. :func:`read_topic_closure` is the
single home of the closure scroll, exposed separately only so each
caller can place it where its own ordering requires; a caller with a
delete arm must re-read it AFTER the fold.

THE MINT DELIBERATELY BYPASSES THE TOOL-LEVEL WRITE GUARDS. It goes
through ``MemoryService.add_memory``, so it never meets the near-
duplicate or topic-cluster guards in ``server/near_duplicate_guard.py``,
which are reached only from the ``add_memory`` TOOL body. That is correct
by construction — a canonical is near its peers by definition, and a
topic under consolidation is the very cluster shape the topic guard
bounces, so routing the mint through the tool would make the ratified
index canonical unwritable. Pinned by
``tests/test_consolidation_ops.py::TestTheMintBypassesTheToolLevelGuards``
so the tool path cannot be quietly reintroduced.

IMPORT RULE: this module must NEVER import ``fused_memory.server.tools``.
That module imports this one, so the reverse edge is a cycle. It is also
why ``TOPIC_MEMBER_LIMIT`` and :func:`patch_memory_metadata` live here
and are imported BY the tool rather than the other way round.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fused_memory.server.consolidation import build_consolidation_result
from fused_memory.server.mem0_update_authz import resolve_mem0_update_authorization
from fused_memory.services.topic_anchor import select_canonical_payload

logger = logging.getLogger(__name__)

__all__ = [
    'TOPIC_MEMBER_LIMIT',
    'TopicClosure',
    'execute_retain_consolidation',
    'patch_memory_metadata',
    'read_topic_closure',
]

#: How many topic members ``consolidate_memories`` lists back as the
#: post-consolidation closure.
#:
#: A topic is a duplicate CLUSTER, so a conforming one is single digits and
#: this bound is never reached; it exists so a topic that has become a
#: dumping ground cannot return an unbounded payload. Reaching it is itself
#: a finding, which is why the envelope discloses ``topic_members_truncated``
#: rather than letting a capped listing read as the whole closure.
TOPIC_MEMBER_LIMIT = 200


@dataclass(frozen=True)
class TopicClosure:
    """What a topic's deterministic scroll saw, and whether it saw all of it.

    ``available`` is the third answer an empty ``members`` cannot give on
    its own: a scroll that could not be READ is not a topic with no
    members, and conflating them is the overclaim this op exists to end.
    """

    members: list[Any]
    total: int
    truncated: bool
    available: bool


# ONE call-and-classify block for EVERY metadata patch this op makes:
# the retain-arm tag, the child reparent, and the canonical's
# supersedes correction. Extracted rather than copied because the
# contract it encodes is non-obvious and identical at all three sites
# (INV-5: two copies would have to stay in lockstep, and a drift
# between them would be silent, since both halves would still
# "work").
#
# THE CONTRACT, in one place: `update_memory` reports MemoryNotFound
# and its authorization refusals by RETURNING {'error_type': ...},
# while every OTHER failure goes through `_journaled_backend_call`,
# which logs and RE-RAISES. Code that guarded only exceptions would
# record a refusal as a success; code that guarded only the returned
# shape would let one Qdrant timeout escape to `@mcp_tool_errors`,
# which flattens the whole envelope to {'error', 'error_type'} —
# destroying the per-id dispositions of records that are ALREADY
# IRREVERSIBLY DELETED and skipping their tombstone write. So both
# shapes are handled, and they collapse to the same per-id verdict.
#
# Returns None on success, or the normalized {'error', 'error_type'}
# failure dict each arm decorates with its own keys (`id`, or
# `child_id`/`from`/`to`).
async def patch_memory_metadata(
    memory_service: Any,
    *,
    memory_id: str,
    project_id: str,
    patch: dict[str, Any],
    agent_id: str | None,
    session_id: str | None,
    causation_id: str | None,
    source: str,
) -> dict[str, Any] | None:
    try:
        outcome = await memory_service.update_memory(
            memory_id=memory_id,
            project_id=project_id,
            # No content, ever: this op stamps metadata and writes new
            # records; it never rewrites an existing record's text, so
            # nothing is re-embedded and no vector moves.
            content=None,
            metadata_patch=patch,
            metadata_mode='merge',
            agent_id=agent_id,
            session_id=session_id,
            causation_id=causation_id,
            _source=source,
        )
    except Exception as exc:
        # `Exception`, not `BaseException`: a process going away
        # (CancelledError/KeyboardInterrupt/SystemExit) must never be
        # recorded as a per-id disposition.
        return {'error': str(exc), 'error_type': type(exc).__name__}
    if isinstance(outcome, dict) and outcome.get('error_type'):
        return {
            'error': outcome.get('error'),
            'error_type': outcome.get('error_type'),
        }
    return None


async def read_topic_closure(
    memory_service: Any,
    *,
    project_id: str,
    topic: str,
    run_id: str | None = None,
) -> TopicClosure:
    """Scroll every member of *topic*. The ONE home of the closure listing.

    Callers place it where their own ordering contract requires — this arm
    reads it once the canonical exists, while a caller that also DELETES
    must re-read it after the fold, or it reports a reaped record as a live
    topic member in the same envelope that reports it deleted.
    """
    # (6) The closure listing comes from the deterministic scroll, NOT
    # `search`. A ranked top-N read can silently omit the canonical this
    # call just wrote — the exact failure that made the original
    # incident's "re-derive via search" correction route dispatch back
    # into the superseded members it was collapsing.
    #
    # A scroll that could not be read degrades to NOT AVAILABLE rather
    # than to an empty list, because `[]` alone reads as "this topic has
    # no members" — the overclaim this op exists to eliminate, and
    # doubly wrong on a call that just wrote a canonical into that very
    # topic. The count is inside the same guard: it only runs when the
    # listing was already capped, so losing it leaves rows that cannot be
    # qualified, and publishing them with `truncated=False` would assert
    # completeness this call cannot support.
    topic_members_available = True
    try:
        topic_members = await memory_service.get_memories_by_metadata(
            project_id=project_id,
            filters={'topic': topic},
            limit=TOPIC_MEMBER_LIMIT,
        )
        returned = len(topic_members) if isinstance(topic_members, list) else 0
        if returned >= TOPIC_MEMBER_LIMIT:
            total = await memory_service.count_memories_by_metadata(
                project_id=project_id, filters={'topic': topic}
            )
        else:
            total = returned
    except Exception:
        logger.warning(
            'consolidate_memories: the topic-closure listing for %r could '
            'not be read; the fold itself stands and is reported in full',
            topic,
            exc_info=True,
            extra={'project_id': project_id, 'run_id': run_id},
        )
        topic_members = []
        returned = 0
        total = 0
        topic_members_available = False

    return TopicClosure(
        members=topic_members,
        total=total,
        truncated=total > returned,
        available=topic_members_available,
    )


async def _resolve_incumbent_canonical(
    memory_service: Any, *, project_id: str, topic: str
) -> str | None:
    """The id of *topic*'s existing canonical, or ``None`` if it names none.

    A canonical-filtered scroll, not the closure listing: that listing is
    capped at ``TOPIC_MEMBER_LIMIT`` and can miss the incumbent on a crowded
    topic. A read failure propagates to the caller.
    """
    rows = await memory_service.get_memories_by_metadata(
        project_id=project_id,
        filters={'topic': topic, 'canonical': True},
        limit=TOPIC_MEMBER_LIMIT,
    )
    incumbent = select_canonical_payload(
        rows, allowed_categories=None, include_planned=True
    )
    incumbent_id = (incumbent or {}).get('id')
    return incumbent_id if isinstance(incumbent_id, str) and incumbent_id else None


async def execute_retain_consolidation(
    memory_service: Any,
    *,
    project_id: str,
    topic: str,
    canonical_content: str | None,
    retain_ids: list[str],
    category: str | None,
    agent_id: str | None,
    run_id: str | None,
    extra_canonical_meta: dict[str, Any] | None,
    session_id: str | None = None,
    causation_id: str | None = None,
    source: str = 'mcp_tool',
) -> dict[str, Any]:
    """Mint the canonical, tag the retained peers, list the topic's closure.

    The retain arm of ``consolidate_memories``, callable without a server
    or a tool closure so the auto-consolidation executor reaches the SAME
    code the tool does. Returns ``build_consolidation_result``'s envelope
    with every delete-arm disposition empty, or a ``{'error',
    'error_type'}`` refusal.

    ``canonical_content=None`` selects TAG-ONLY (PRD D14): no canonical is
    minted and the topic's existing one is reported instead. It is a MODE
    of this function rather than a sibling, so the peer-tag loop below has
    exactly one home and cannot drift between the two.

    Every refusal — authorization, ``CanonicalWriteFailed``,
    ``TagOnlyIncumbentNotFound`` — is returned before any peer is read or
    patched.
    """
    # AUTHORIZE FIRST, above every read and every write: an unauthorized
    # caller is turned away before anything is done on its behalf and
    # before it learns anything about the system.
    #
    # `content_amend=False` is load-bearing. This function writes new
    # records and stamps metadata; it never rewrites an existing record's
    # text. Requesting an arm it does not use would make the deliberately
    # wider metadata bar a back door into a silent-rewrite primitive — the
    # one thing the resolver's two-arm split exists to prevent.
    #
    # DUPLICATED with the tool's own gate, deliberately. The tool's call is
    # unconditional because it also covers the child reparent and the
    # supersedes narrowing, which are not this arm's patches; this one
    # exists because the auto-consolidation executor has no tool boundary
    # in front of it. Running it twice on the tool path costs nothing
    # measurable — the resolver is pure, synchronous and three `getattr`
    # hops — and the refusal shape is identical, so a caller reads one
    # vocabulary whichever gate turned it away.
    #
    # The LIVE `memory_service` goes in, positionally. Binding
    # `memory_service.config` or any leaf of it to a local would make the
    # five green-tier `mem0_update.*` leaves restart-only in disguise
    # (PRD C4).
    decision = resolve_mem0_update_authorization(
        memory_service,
        agent_id=agent_id,
        content_amend=False,
        metadata_patch=True,
    )
    if not decision.allowed:
        return {
            'error': decision.error,
            'error_type': decision.error_type,
            'agent_id': agent_id,
        }

    # (4) The canonical FIRST: no delete or metadata patch may precede an
    # ESTABLISHED canonical, which the mint arm establishes by writing it
    # and tag-only by resolving it. The ordering IS the anti-ratchet
    # property, and it is asymmetric on purpose:
    #
    #   delete-then-write, on a failed write  -> net LOSS, unrecoverable
    #                                            (no write path reaches a
    #                                            deleted point id)
    #   write-then-delete, on a failed delete -> net ADD, reportable in
    #                                            `failed_deletes` and
    #                                            re-runnable
    #
    # This order makes the first outcome impossible and the second
    # visible.
    canonical_meta: dict[str, Any] = {}
    canonical_id: str
    if canonical_content is not None:
        # `CanonicalUniquenessViolation` and `MemoryMetadataValidationError`
        # are left to propagate to `@mcp_tool_errors`: the op refused before
        # touching anything, so there is no partial state to describe and
        # the flattened {'error', 'error_type'} envelope is the whole truth.
        canonical_meta = dict(extra_canonical_meta or {})
        canonical_meta.update({'topic': topic, 'canonical': True})
        written = await memory_service.add_memory(
            content=canonical_content,
            category=category,
            project_id=project_id,
            agent_id=agent_id,
            session_id=session_id,
            metadata=canonical_meta,
            causation_id=causation_id,
            _source=source,
        )
        # `AddMemoryResponse` is a pydantic model whose `memory_ids` can come
        # back EMPTY without raising — a write that landed nothing while
        # reporting no failure. Indexing it blindly would either raise an
        # IndexError flattened into an unreadable error, or (worse, had this
        # been a dict lookup) carry a None canonical id into the delete loop
        # and reap a live cluster in favour of a record that does not exist.
        minted_id = written.memory_ids[0] if written.memory_ids else None
        if not minted_id:
            return {
                'error': (
                    'consolidate_memories: the canonical write returned no memory '
                    'id, so nothing was deleted. The supersedes are untouched — '
                    're-run once the write path is healthy.'
                ),
                'error_type': 'CanonicalWriteFailed',
                'topic': topic,
                'supersedes': list(canonical_meta.get('supersedes') or []),
            }
        canonical_id = minted_id
    else:
        # TAG-ONLY (PRD D14). NOTHING is written to the incumbent:
        # `content_amend=False` is load-bearing, and an incumbent that
        # arrives inside `retain_ids` is refused by the loop's
        # `RetainedPeerIsCanonical` branch rather than demoted — the
        # caller's predicate is meant to strip it and disclose the strip.
        try:
            incumbent_id = await _resolve_incumbent_canonical(
                memory_service, project_id=project_id, topic=topic
            )
            reason = 'its topic names no canonical'
        except Exception as exc:
            logger.warning(
                'consolidate_memories: tag-only could not read the canonical '
                'for %r; refusing before any peer is touched',
                topic,
                exc_info=True,
                extra={'project_id': project_id, 'run_id': run_id},
            )
            incumbent_id = None
            reason = f'its canonical could not be read ({exc})'
        if not incumbent_id:
            return {
                'error': (
                    f'consolidate_memories: tag-only was asked to report the '
                    f'existing canonical for {topic!r}, but {reason}. No peer '
                    'was read or tagged and nothing was minted, so the call '
                    'can simply be re-run once the topic has a canonical, or '
                    'with canonical_content to mint one.'
                ),
                'error_type': 'TagOnlyIncumbentNotFound',
                'topic': topic,
            }
        canonical_id = incumbent_id

    # (5a) THE RETAIN ARM — the ratified default (gate 3200). Each peer
    # is TAGGED IN PLACE: it keeps its Qdrant point id, so every citation,
    # parent pointer and supersedes edge already aimed at it stays valid.
    # That id stability is the entire reason the arm exists; a
    # delete-and-rewrite peer would drop every inbound reference and
    # could not be restored, since no write path reaches a deleted point.
    #
    # `content=None` is the same property one level down: a peer whose
    # claim did not change must not be re-embedded, so its vector does
    # not move either. Metadata is MERGED and nothing is deleted — the
    # peer's own source/run_id/parent are not this op's to discard.
    #
    # `topic` ONLY. Never `canonical` (exactly one per (project, topic) —
    # a second claimant would make the next consolidation of this topic
    # refuse outright) and never `parent_id` (these are PEERS of the
    # canonical, not children of it).
    #
    # Task 3523 is the live seam here: `update_memory` does not run
    # `_apply_memory_metadata_validation`, so this slug is not
    # re-validated at the patch seam. It is validated at op entry
    # instead, which bounds the hole for this caller without closing it.
    #
    # THAT SAME SEAM IS WHY NOT SETTING `canonical` IS NOT ENOUGH. The
    # patch is a server-side Qdrant payload MERGE, so a peer that ALREADY
    # carries `canonical: True` keeps it and is now paired with the new
    # `topic` — a second claimant for (project, T), minted without a
    # rejection, a census line, or a `retain_failures` entry, because
    # `_apply_canonical_uniqueness` is reached only from the `add_memory`
    # path. Not setting a key and ensuring it is unset are different
    # claims, and only the second one holds the invariant.
    #
    # This is REACHABLE THROUGH THE DEFAULT ARM, not through misuse: the
    # ratchet this op exists to end is precisely "a cluster ends up
    # containing the consolidator's own prior canonicals", so a cluster
    # under consolidation routinely contains one, and retaining it is the
    # natural call when its content is still correct and cited. The damage
    # lands on the NEXT pass, which is the worst time to find it: the
    # follow-up consolidation of T fails its own canonical write.
    retained: list[str] = []
    retain_failures: list[dict[str, Any]] = []
    for retain_id in retain_ids:
        # (5a-i) PROVE THE PEER IS NOT ALREADY A CANONICAL, and FAIL
        # CLOSED — the same posture as the child listing, for the same
        # reason: a check that did not ANSWER is not a check that said
        # "not canonical". Refusing costs one peer's tag, which a caller
        # can retry; tagging on an unproven check mints a duplicate
        # canonical that nothing downstream will catch.
        #
        # REFUSE rather than demote. Patching `canonical: False` would
        # also hold the invariant, but it would silently rewrite a claim
        # this op was not asked to touch — and a prior canonical in the
        # retain list is usually an AUTHORING MISTAKE: that record is what
        # the caller should have put in `supersedes`. Surfacing it as a
        # named failure is the recoverable outcome; quietly demoting it
        # is not.
        try:
            peer_record = await memory_service.get_memory_by_id(
                project_id=project_id, memory_id=retain_id
            )
        except Exception as exc:
            retain_failures.append({
                'id': retain_id,
                'error': (
                    f'refused to tag {retain_id}: it could not be read '
                    f'({exc}), so it cannot be shown to be a non-canonical '
                    'peer'
                ),
                'error_type': 'RetainCheckFailed',
            })
            continue
        # A peer that does not resolve is NOT refused here: it falls
        # through to `update_memory`, which reports MemoryNotFound in the
        # structured shape below. One vocabulary for one condition.
        if (peer_record or {}).get('metadata', {}).get('canonical') is True:
            retain_failures.append({
                'id': retain_id,
                'error': (
                    f'refused to tag {retain_id}: it is already the '
                    f'canonical for its topic, and tagging it with '
                    f'{topic!r} would make a second canonical for that '
                    'topic — supersede it instead of retaining it'
                ),
                'error_type': 'RetainedPeerIsCanonical',
            })
            continue
        # `topic` ONLY, through the shared classifier above: a raise and a
        # returned rejection are THE SAME EVENT here (this peer was not
        # tagged) and are recorded identically. One failure costs one
        # peer, never the arm — the peers that CAN be tagged are, because
        # re-running to catch the rest would re-write the canonical, the
        # +1-per-pass ratchet.
        failure = await patch_memory_metadata(
            memory_service,
            memory_id=retain_id,
            project_id=project_id,
            patch={'topic': topic},
            agent_id=agent_id,
            session_id=session_id,
            causation_id=causation_id,
            source=source,
        )
        if failure:
            retain_failures.append({'id': retain_id, **failure})
            continue
        retained.append(retain_id)

    closure = await read_topic_closure(
        memory_service, project_id=project_id, topic=topic, run_id=run_id
    )

    return build_consolidation_result(
        canonical_id=canonical_id,
        topic=topic,
        canonical_supersedes=list(canonical_meta.get('supersedes') or []),
        deleted=[],
        failed_deletes=[],
        survivors=[],
        survivor_check_failed=[],
        retained=retained,
        retain_failures=retain_failures,
        reparented=[],
        reparent_failures=[],
        topic_members=closure.members,
        topic_members_total=closure.total,
        topic_members_truncated=closure.truncated,
        topic_members_available=closure.available,
    )
