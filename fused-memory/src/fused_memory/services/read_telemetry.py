"""The single home for the search-telemetry shape written to ``write_ops.result_summary``.

A search journal row used to carry ``{'count': N}``, which records that a read
happened and nothing about what it showed.  This module owns the widened shape
(task 3212 item 1) so that the THREE producers emit one contract rather than
three variants (INV-5):

  - ``fused_memory/server/tools.py::search`` — summarises the GROUPED payload,
    because grouping (task 3129) is applied at the MCP boundary and the grouped
    list is literally what the agent was shown.
  - ``fused_memory/services/memory_service.py::MemoryService.search`` — its
    causation-id self-journal, covering the whole reconciliation read path.  No
    grouping happens below the MCP boundary, so it summarises its raw list.
  - ``fused_memory/reconciliation/context_assembler.py::ContextAssembler._ctx_task_event``
    — memory_hints executions, which journalled nothing at all before.

CONSUMERS (why the detail exists at all):

  - leaf eta's write-after-miss metric (task 3213) — this is its SOLE data
    source.  Its question is "was the agent SHOWN the thing it then re-wrote?",
    which a bare count cannot answer and per-result ids can.
  - leaf theta's retro corpus (task 3214) reads the same rows.

SIZE UNIT.  ``content_size`` is ``len(content)`` — a count of CHARACTERS, not a
tokenizer estimate.  The unit is emitted as a first-class ``size_unit`` key
rather than baked into a field name, so a consumer reads the unit
programmatically and a future switch to a token estimate changes that key's
VALUE instead of silently redefining an existing field.  ``schema_version`` is
stamped alongside for the same reason.

SEAM NOTE (PRD §8).  When task 3127's triage acknowledgement lands, its
``routed`` / ``canonical_id`` fields join this dict — PRODUCED there, only
LOGGED here.  Do not implement them in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SEARCH_TELEMETRY_SCHEMA_VERSION = 1

# CHARACTERS, not tokens — see the module docstring.  Emitted as a key so a
# reader never has to infer it.
SEARCH_TELEMETRY_SIZE_UNIT = 'chars'

# Derived from what callers actually ask for, not a round-number guess:
# orchestrator/agents/briefing.py passes limit=5, ReconciliationConfig's
# context_search_limit defaults to 5, and the MCP tool's own default is 10.  So
# 50 is 5-10x headroom over every observed use while bounding a pathological row
# to roughly 6 KB instead of roughly 24 KB.  Truncation is never silent — see
# ``results_truncated`` below.
SEARCH_TELEMETRY_MAX_RESULTS = 50

# Keys inside a grouped entry's 'grouped' block whose members were folded INTO
# the parent hit and so were shown to the agent as part of it.
_FOLDED_CHILD_KEYS = ('matched_children', 'amendments')


def _field(item: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` off a ``MemoryResult`` attribute or a ``model_dump()`` key."""
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _folded_child_ids(item: Any) -> list[str]:
    """Collect ids folded into a grouped entry, in the order they appear.

    A grouped block is produced by ``server/grouped_read.py``; this function is
    telemetry, not that block's validator, so a malformed block yields no ids
    rather than costing the row its summary.
    """
    block = _field(item, 'grouped')
    if not isinstance(block, Mapping):
        return []
    seen: dict[str, None] = {}
    for key in _FOLDED_CHILD_KEYS:
        members = block.get(key)
        if not isinstance(members, Sequence) or isinstance(members, str | bytes):
            continue
        for member in members:
            child_id = member.get('id') if isinstance(member, Mapping) else None
            if isinstance(child_id, str) and child_id:
                seen.setdefault(child_id, None)
    return list(seen)


def _summarize_one(item: Any) -> dict[str, Any]:
    content = _field(item, 'content', '')
    metadata = _field(item, 'metadata') or {}
    source_store = _field(item, 'source_store')
    entry: dict[str, Any] = {
        'id': _field(item, 'id'),
        'relevance_score': _field(item, 'relevance_score'),
        'content_size': len(content) if isinstance(content, str) else 0,
        # str() rather than the raw StrEnum member: MemoryResult.model_dump()
        # hands back the enum in python mode while the attribute read hands back
        # the same enum, and both must serialise to the identical JSON as the
        # other producers' plain strings.
        'source_store': str(source_store) if source_store is not None else None,
        # Per-store truth, stamped into metadata by task 3658.  Absent means
        # None — never invented — and Graphiti reports store_score=None by
        # contract, which passes through verbatim rather than being coerced to
        # 0.0 (a coerced 0.0 is indistinguishable from a genuinely worst hit).
        'store_rank': metadata.get('store_rank') if isinstance(metadata, Mapping) else None,
        'store_score': metadata.get('store_score') if isinstance(metadata, Mapping) else None,
        # A pinned result was PROMOTED into the window rather than earning its
        # slot by rank, so its relevance_score is not meaningful.  Logged so a
        # consumer can tell the two apart.
        'topic_anchored': bool(_field(item, 'topic_anchored', False)),
    }
    folded = _folded_child_ids(item)
    if folded:
        entry['folded_child_ids'] = folded
    return entry


def summarize_search_results(
    results: Any,
    *,
    max_results: int = SEARCH_TELEMETRY_MAX_RESULTS,
) -> dict[str, Any]:
    """Summarise search results into the journalled ``result_summary`` shape.

    Pure and synchronous — no I/O, no logging, no clock.  Accepts ``MemoryResult``
    objects or their ``model_dump()`` dicts interchangeably and produces a
    byte-identical summary for either, which is what lets one implementation
    serve the MCP boundary (grouped dicts), ``MemoryService`` (raw results) and
    the reconciliation hint site.

    Returns::

        {'schema_version': int,
         'count': int,              # the TRUE total, never rewritten by the cap
         'size_unit': 'chars',
         'results': [{'id', 'relevance_score', 'content_size', 'source_store',
                      'store_rank', 'store_score', 'topic_anchored',
                      'folded_child_ids'?}],
         'results_logged': int,     # how many per-result entries were kept
         'results_truncated': bool} # True iff entries were dropped by the cap

    ``folded_child_ids`` appears only for a grouped entry that actually folded
    children, so an ungrouped row carries no empty-list noise.
    """
    items = list(results) if results else []
    kept = items[:max_results] if max_results >= 0 else items
    return {
        'schema_version': SEARCH_TELEMETRY_SCHEMA_VERSION,
        'count': len(items),
        'size_unit': SEARCH_TELEMETRY_SIZE_UNIT,
        'results': [_summarize_one(item) for item in kept],
        'results_logged': len(kept),
        'results_truncated': len(kept) < len(items),
    }
