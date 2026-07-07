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
"""

from __future__ import annotations

import json
from typing import Any

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

# The complete set of counter keys derive_stage_stats produces. Includes the
# virtual 'graphiti_writes_queued' key derived from ops rather than mapped 1:1
# in _OP_TO_STAT (the residual of add_memory ops where memory_ids is empty but
# stores includes graphiti). Every one of these keys is always present in the
# returned dict, 0-default, so downstream override logic has a complete,
# deterministic key set to iterate regardless of what ops actually occurred.
_COMPUTED_STAT_KEYS: frozenset[str] = frozenset(_OP_TO_STAT.values()) | {
    'graphiti_writes_queued',
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


def _count_update_edge(op: dict) -> bool:
    """Return True only if the update_edge op was verified by a server-side readback.

    MemoryService.update_edge performs a ``get_edge_text`` round-trip after the
    save and sets ``verified=True`` in ``result_summary`` only when the returned
    fact matches. A missing ``verified`` key (legacy ops written before Guard 2
    was deployed) or ``verified=False`` (readback failed or returned a different
    fact) are both treated as unverified and excluded from ``edges_updated``.

    The strict ``is True`` identity check (not truthy) prevents strings like
    ``'true'`` or ``1`` from accidentally counting.
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

        if operation == 'add_memory':
            if _count_add_memory(op):
                counts[stat_key] += 1
            elif _count_graphiti_queued(op):
                # Graphiti-only async enqueue: no inline ID, but store accepted
                # the write. Track separately so it doesn't inflate memories_added.
                counts['graphiti_writes_queued'] += 1
            # Either path accounts for the op — skip the generic counter.
            continue
        elif operation == 'update_edge':
            if not _count_update_edge(op):
                continue
        elif not op.get('success', 1):
            continue
        counts[stat_key] += 1

    return counts
