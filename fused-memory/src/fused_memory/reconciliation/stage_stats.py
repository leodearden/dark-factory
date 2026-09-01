"""Derive a reconciliation stage's write counters from the write journal.

This module is the write-shaped source of truth for stage stats. Stage agents
still self-report counts like ``memories_added: 3`` in their final summary,
but Mem0 silently deduplicates and returns an empty ``memory_ids`` list on
filtered writes, so self-reported counts drift from what actually happened.
``derive_stage_stats`` recomputes the canonical counters directly from the
write journal for a given stage's own ``agent_id`` — the LLM's self-report is
no longer on the read path for these keys (see ``stats_verifier``, which
applies these computed values as overrides and keeps the LLM's originals
under ``stats['_reported']`` purely as a divergence signal for the judge).

Correct counting depends on every stage write being stamped with the
canonical ``agent_id=f'recon-stage-{stage_id}'`` (see ``stages/base.py``).
Harness-authored writes (e.g. ``task_knowledge_sync.py``) always carry this
tag, but writes made by the LLM CLI agent carry it only because the agent's
prompt instructs it to — a stage that omits or mistypes its ``agent_id``
silently yields 0 for every counter here rather than an error.
``stats_verifier.verify_and_rewrite_stats`` logs an INFO diagnostic when a
write op carries a ``recon-stage-*`` agent_id that matches no known stage,
which backstops (but does not fully cover) this dependency.

The counters here answer "did the write LAND", which is a different fact from
``write_ops.success`` ("was the enqueue ACCEPTED"). A write can be accepted at
enqueue and still never reach the backend, because the durable queue retries
it asynchronously and may ultimately dead-letter it — long after the stage
that issued it has finished and reported. ``derive_stage_stats`` therefore
applies ``_landed`` (reading ``write_ops.terminal_status``, stamped back by
``DurableWriteQueue``'s terminal hook) on top of the existing ``success``
gates, so an operation whose writes all died reports 0 rather than green.

That gate is applied PER LEG, not per op, because ``terminal_status``
describes only an op's DURABLE-QUEUE leg — and a write_op row does not always
map 1:1 to a queue item. ``MemoryService.add_memory`` mints ONE
``write_op_id`` spanning a queued Graphiti leg AND a synchronous Mem0 leg,
then journals a single Layer-1 row under that shared id; the queue's terminal
hook is keyed on the same id, so a dead Graphiti leg stamps ``'dead'`` on the
row that also carries Mem0's returned ``memory_ids``. The general rule, which
any future dual-leg operation inherits: a counter may be suppressed by
``terminal_status`` only when the queue item is the very thing that counter
counts. So a dead queue leg suppresses ``graphiti_writes_queued`` (its whole
subject is that queued write) but never ``memories_added`` (whose evidence,
non-empty ``memory_ids``, could only have come back inline from the completed
synchronous call).

Excluding them would be truthful but SILENT, so a third fact is reported
alongside: ``writes_dead_lettered`` answers "did the write reach the queue's
dead-letter state". The pair is self-explaining — ``graphiti_writes_queued: 0``
next to ``writes_dead_lettered: 12`` means the stage's writes DIED, which is a
failure, not the no-op that a bare ``0`` would suggest. The two overlap on two
legitimate cases (a post-execute dead-letter, and a partially-landed dual
write); ``_COMPUTED_STAT_KEYS`` documents why that is deliberate.
"""

from __future__ import annotations

import json
from typing import Any

from fused_memory.services.durable_queue import POST_EXECUTE_DEAD_PREFIX

# Operation → stat-key mapping. This is the canonical set of write-journal
# operations this module knows how to tally into a stage's stats.
_OP_TO_STAT: dict[str, str] = {
    'add_memory': 'memories_added',
    'delete_memory': 'memories_deleted',
    'add_episode': 'episodes_added',
    'delete_episode': 'episodes_deleted',
    'update_edge': 'edges_updated',
    'refresh_entity_summary': 'entity_summaries_refreshed',
    'merge_entities': 'entities_merged',
    'rebuild_entity_summaries': 'entity_summaries_rebuilt',
    'replay_dead_letters': 'dead_letters_replayed',
    'replay_to_graphiti': 'episodes_replayed',
}

# The complete set of counter keys derive_stage_stats produces. Includes two
# VIRTUAL keys derived from op state rather than mapped 1:1 in _OP_TO_STAT:
#
#   'graphiti_writes_queued' — the residual of add_memory ops where memory_ids
#       is empty but stores includes graphiti.
#   'writes_dead_lettered'   — ops the durable queue drove to its dead-letter
#       state (terminal_status == 'dead').
#
# The two facts these keys report are deliberately distinct. Every counter
# other than 'writes_dead_lettered' answers "did the write LAND" (see
# ``_landed``); 'writes_dead_lettered' answers "did the write reach the
# queue's dead-letter state". They overlap on TWO legitimate cases, and
# neither overlap is double-counting:
#
#   1. A post-execute dead-letter — it landed AND dead-lettered: the write
#      persisted, and the item still needs operator attention because it must
#      not be blind-replayed.
#   2. A partially-landed dual write — 'memories_added: 1' next to
#      'writes_dead_lettered: 1', where add_memory's synchronous Mem0 leg
#      persisted but its queued Graphiti twin died. Both answers are honestly
#      yes, and reporting both is the only signal that the op was PARTIAL;
#      suppressing the dead-letter tally to keep the keys mutually exclusive
#      would leave it looking like a clean success.
#
# KNOWN LIMITATION: a dead-letter whose Layer-1 row was never journalled has
# no agent_id/operation, so it is invisible here — the same agent_id
# dependency already documented in this module's docstring.
#
# Every one of these keys is always present in the returned dict, 0-default,
# so downstream override logic has a complete, deterministic key set to
# iterate regardless of what ops actually occurred.
_COMPUTED_STAT_KEYS: frozenset[str] = frozenset(_OP_TO_STAT.values()) | {
    'graphiti_writes_queued',
    'writes_dead_lettered',
}


def _parse_result_summary(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _landed(op: dict) -> bool:
    """Return False only for a write the durable queue gave up on.

    Reads ``write_ops.terminal_status``, which ``DurableWriteQueue``'s terminal
    hook stamps back once an item reaches a terminal state. The domain is
    ``NULL | 'completed' | 'dead'``.

    Deliberately a DENYLIST on the single literal ``'dead'``, not an allowlist
    on ``{None, 'completed'}``. A NULL means the write's outcome is genuinely
    unknown — every row predating the terminal-status migration is NULL (it
    deliberately does not backfill), as is every operation that never touches
    the durable queue at all (``delete_memory``, ``update_edge``, task writes).
    Those, and any future or unrecognised status value, keep their current
    counting behaviour, so a partially-applied migration or a new queue state
    can never silently zero a stage's real counters.

    ONE EXCEPTION, where ``'dead'`` still counts as landed: a *post-execute*
    dead-letter. ``'dead'`` alone means only "the queue exhausted its
    attempts" and does NOT imply the write never happened — when the backend
    write succeeded and only the callback or completion commit kept failing,
    ``DurableWriteQueue`` prefixes ``terminal_error`` with
    ``POST_EXECUTE_DEAD_PREFIX`` precisely so consumers can separate the two
    cases (see that constant's defining comment in ``services.durable_queue``).
    Excluding those would UNDERCOUNT writes that genuinely persisted — the
    opposite error from the over-counting this gate exists to fix. Such an op
    is still tallied under ``writes_dead_lettered``: it landed, and it also
    still needs operator attention because it must not be blind-replayed.

    The ``isinstance`` guard is explicit because ``terminal_error`` is
    ``str | None`` — a dead-letter carrying no error string is reachable, is
    treated as NOT landed, and must not raise ``AttributeError``.

    Answers "did this op's DURABLE-QUEUE leg land", so ``derive_stage_stats``
    applies it PER BRANCH, not once per op, and never inside the ``_count_*``
    helpers — those stay purely about ``result_summary`` shape. It gates the
    ``_count_graphiti_queued`` branch (whose subject IS the queued write) and
    the generic branch (whose ops map 1:1 to one queue item, or never touch
    the queue at all). It deliberately does NOT gate the ``_count_add_memory``
    branch: that row's ``memory_ids`` come from the synchronous Mem0 leg, which
    no queue-leg outcome can contradict.
    """
    if op.get('terminal_status') != 'dead':
        return True
    error = op.get('terminal_error')
    return isinstance(error, str) and error.startswith(POST_EXECUTE_DEAD_PREFIX)


def _count_update_edge(op: dict) -> bool:
    """Return True only if the update_edge op was verified by a server-side readback.

    MemoryService.update_edge performs a ``get_edge_text`` round-trip after the
    save and sets ``verified=True`` in ``result_summary`` only when the returned
    fact matches. A missing ``verified`` key (legacy ops written before Guard 2
    was deployed) or ``verified=False`` (readback failed or returned a different
    fact) are both treated as unverified and excluded from ``edges_updated``.

    The strict ``is True`` identity check (not truthy) prevents strings like
    ``'true'`` or ``1`` from accidentally counting.

    Answers only "what SHAPE of write was this". Whether the write LANDED is
    the orthogonal ``_landed`` gate, applied by the caller.
    """
    if not op.get('success', 1):
        return False
    rs = _parse_result_summary(op.get('result_summary'))
    return rs.get('verified') is True


def _count_add_memory(op: dict) -> bool:
    """Return True if the add_memory op actually produced a stored memory.

    Mem0 may dedup a write and return an empty ``memory_ids`` list. Only ops
    where ``memory_ids`` is a non-empty list count toward ``memories_added``.
    Graphiti-only async-enqueued writes (``memory_ids=[]``, ``stores=['graphiti']``)
    are tracked separately under ``graphiti_writes_queued`` via ``_count_graphiti_queued``.
    An op that reached neither store with a returned ID is a no-op here.

    Answers only "what SHAPE of write was this". Whether the write LANDED is
    the orthogonal ``_landed`` gate, applied by the caller.
    """
    if not op.get('success', 1):
        return False
    rs = _parse_result_summary(op.get('result_summary'))
    memory_ids = rs.get('memory_ids')
    return bool(isinstance(memory_ids, list) and memory_ids)


def _count_graphiti_queued(op: dict) -> bool:
    """Return True if the add_memory op was a graphiti-only async enqueue.

    Graphiti writes are enqueued asynchronously and return no ``memory_ids``
    inline. An op counts as a graphiti-only enqueue iff it succeeded,
    ``memory_ids`` is empty (so it was not counted toward ``memories_added``),
    and ``stores``/``stores_written`` contains ``'graphiti'``. These are
    tallied separately under ``graphiti_writes_queued`` rather than
    ``memories_added`` to avoid inflating the memories count.

    Answers only "what SHAPE of write was this". Whether the enqueued write
    ultimately LANDED is the orthogonal ``_landed`` gate, applied by the
    caller — this helper's subject is precisely the queued writes that gate
    most affects.
    """
    if not op.get('success', 1):
        return False
    rs = _parse_result_summary(op.get('result_summary'))
    memory_ids = rs.get('memory_ids')
    if isinstance(memory_ids, list) and memory_ids:
        # Non-empty IDs: this op is a memories_added, not a graphiti enqueue.
        return False
    stores = rs.get('stores') or rs.get('stores_written')
    return bool(isinstance(stores, list) and 'graphiti' in stores)


def derive_stage_stats(ops: list[dict], stage_agent_id: str) -> dict[str, int]:
    """Tally a stage's own write-journal ops into canonical write counters.

    Filters ``ops`` down to entries where ``layer == 'write_op'`` (backend_ops
    are a second audit layer of the same write and would double-count) AND
    ``agent_id == stage_agent_id`` (each stage's CLI agent stamps every write
    with ``agent_id=f'recon-stage-{stage_id}'`` — filtering on this is exact,
    unlike timestamp-window bucketing, and correctly excludes ops from other
    stages/agents that happen to fall in the same time window).

    Ops the durable queue dead-lettered are then dropped by ``_landed``: they
    were accepted at enqueue (``success=1``) but never reached the backend, so
    counting them would report a 0%-landing-rate stage as green.

    That gate is applied PER BRANCH, never inside the ``_count_*`` helpers
    (which stay purely about ``result_summary`` shape). It is NOT applied once
    at the top of the loop, because ``terminal_status`` is a per-leg fact: the
    ``add_memory`` arm's row spans a queued Graphiti leg and a synchronous
    Mem0 leg under one shared ``write_op_id``, so a whole-op gate would let a
    dead Graphiti leg zero a ``memories_added`` the Mem0 leg already proved.
    Hence ``_count_add_memory`` counts ungated, while the
    ``_count_graphiti_queued`` arm and the generic branch below — whose ops
    map 1:1 to a single queue item, or never touch the queue at all — are
    gated.

    Always returns every key in ``_COMPUTED_STAT_KEYS``, 0-default, even when
    no ops matched.
    """
    counts: dict[str, int] = dict.fromkeys(_COMPUTED_STAT_KEYS, 0)

    for op in ops:
        if op.get('layer') != 'write_op':
            continue
        if op.get('agent_id') != stage_agent_id:
            continue

        operation = op.get('operation')
        if not isinstance(operation, str):
            continue
        stat_key = _OP_TO_STAT.get(operation)
        if stat_key is None:
            continue

        if op.get('terminal_status') == 'dead':
            # Gated on the RAW status, not on _landed, and sitting ABOVE every
            # branch so a partially-landed dual write is still surfaced. A
            # post-execute dead-letter DID land (and so still counts toward its
            # landed counter below) but is reported here too, because it needs
            # operator attention and must not be blind-replayed. Also
            # deliberately NOT gated on `success` — reaching the dead-letter
            # queue is an outcome fact independent of whether the enqueue was
            # accepted, and AND-ing them would make an odd combination vanish
            # from the counter instead of being reported.
            counts['writes_dead_lettered'] += 1

        if operation == 'add_memory':
            if _count_add_memory(op):
                # NOT gated on _landed. This row spans two legs sharing one
                # write_op_id, and terminal_status describes ONLY the queued
                # Graphiti leg. A non-empty memory_ids can only have come back
                # inline from the SYNCHRONOUS Mem0 call, so it is standalone
                # proof that write persisted — a dead Graphiti leg must not
                # veto it.
                counts[stat_key] += 1
            elif _landed(op) and _count_graphiti_queued(op):
                # This branch's subject IS the queued Graphiti leg, so here
                # terminal_status maps 1:1 onto what is being counted.
                # Graphiti-only async enqueue: no inline ID, but store accepted
                # the write. Track separately so it doesn't inflate memories_added.
                counts['graphiti_writes_queued'] += 1
            # Either path accounts for the op — skip the generic counter.
            continue

        # Every other operation's Layer-1 row maps 1:1 to a single queue item
        # (add_episode) or never touches the queue at all (update_edge,
        # delete_memory, task writes — terminal_status stays NULL forever), so
        # the whole-op gate is exact for them.
        if not _landed(op):
            continue

        if operation == 'update_edge':
            if not _count_update_edge(op):
                continue
        elif not op.get('success', 1):
            continue
        counts[stat_key] += 1

    return counts
