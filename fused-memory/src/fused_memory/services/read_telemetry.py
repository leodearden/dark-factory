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

DEGRADATION IS OWNED HERE, not bolted on per producer.  ``degraded`` /
``failed_stores`` is the field that decides whether a short or empty result
list was a real MISS or a store outage, so a consumer that cannot read it
reaches the wrong verdict rather than no verdict.  Left to the producers it
acquired three different rules — one added it only when degraded, one always,
one never — so a degraded hint search journalled a row byte-indistinguishable
from a healthy search that genuinely found nothing.  Every entry point here
therefore takes a ``failed_stores`` argument and stamps BOTH keys
unconditionally: an absent key is indistinguishable from a producer that never
records degradation, the same reasoning that keeps ``journal_drops`` present at
zero.

BOUNDS ARE DISCLOSED, never silent.  Three sizes here are capped and each cap is
stamped beside the value it bounds: the per-result list by
``SEARCH_TELEMETRY_MAX_RESULTS`` (``results_logged`` / ``results_truncated``),
the journalled query text by ``SEARCH_TELEMETRY_MAX_QUERY_CHARS``
(``query_truncated``), and the envelope as a whole by ``telemetry_error`` when
the summariser itself faulted.  ``write_ops`` is the table this same task adds a
prune for because it reached 16 GB, so an unbounded caller-supplied string has
no business in it.

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

# ``query`` arrives over MCP and nothing upstream bounds its length — only
# ``limit`` is clamped — so one pathological caller could otherwise write
# arbitrarily large strings into the table this task is simultaneously adding a
# prune for.  4096 chars is roughly 15x the longest query the real callers
# issue, so the "a metric computed from half a query measures the wrong thing"
# rationale that retired the old ``query[:200]`` still holds for every realistic
# query while the unbounded tail is gone.  Truncation is never silent — see
# ``query_truncated``.
SEARCH_TELEMETRY_MAX_QUERY_CHARS = 4096

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


def _degradation(failed_stores: Any) -> dict[str, Any]:
    """The ``degraded`` / ``failed_stores`` pair every entry point stamps.

    ``degraded`` is DERIVED from a non-empty ``failed_stores`` rather than taken
    as a second argument: ``MemoryService.search`` computes it exactly that way
    (``degraded = bool(failed_stores)``), so accepting both would let a producer
    hand over a pair that cannot occur and would give one fact two sources of
    truth.
    """
    names = [str(store) for store in failed_stores] if failed_stores else []
    return {'degraded': bool(names), 'failed_stores': names}


def summarize_search_query(
    query: str,
    *,
    max_chars: int = SEARCH_TELEMETRY_MAX_QUERY_CHARS,
) -> dict[str, Any]:
    """Build the query keys of a journalled search ``params`` dict.

    Returns ``{'query': str, 'query_truncated': bool}``.  The text is kept in
    full up to ``max_chars`` — a retrieval metric computed from half a query
    measures the wrong thing, which is why the old blanket ``query[:200]`` was
    dropped — and the cap is disclosed exactly the way the result cap is, so a
    consumer never has to guess whether it is reading a whole query.

    The caller owns the rest of ``params`` (``limit``, the ``caller_*``
    identities), which differ per producer; only the bounded text is shared.
    """
    text = query if isinstance(query, str) else str(query)
    return {'query': text[:max_chars], 'query_truncated': len(text) > max_chars}


def fallback_search_summary(
    count: int,
    *,
    failed_stores: Any = None,
) -> dict[str, Any]:
    """The envelope a producer journals when ``summarize_search_results`` RAISED.

    A summariser fault must never turn a working search into an error, but the
    row it leaves behind must not read as a healthy one either.  A bare
    ``{'count': N}`` did exactly that: a consumer reading
    ``summary['schema_version']`` KeyErrors, and one reading
    ``summary.get('results', [])`` silently scores a two-result search as "the
    agent was shown nothing" — the same false negative the journal-drop counter
    exists to make visible.  So the full envelope is emitted, marked by
    ``telemetry_error``, and the facts that survive a summariser fault (the
    count, the degradation) are carried verbatim.

    ``telemetry_error`` is fault-only, so an absent key reads False on the
    healthy path — unlike ``degraded``, which is a per-search FACT a consumer
    must be able to read on every row.
    """
    return {
        'schema_version': SEARCH_TELEMETRY_SCHEMA_VERSION,
        'count': count,
        'size_unit': SEARCH_TELEMETRY_SIZE_UNIT,
        'results': [],
        'results_logged': 0,
        # Entries were dropped iff there were any to drop.  An empty search
        # loses nothing to the fault, and claiming otherwise would add a second
        # false signal on top of the one this function exists to remove.
        'results_truncated': count > 0,
        'telemetry_error': True,
        **_degradation(failed_stores),
    }


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
    failed_stores: Any = None,
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
         'results_logged': int,      # how many per-result entries were kept
         'results_truncated': bool,  # True iff entries were dropped by the cap
         'degraded': bool,           # a selected store raised or timed out
         'failed_stores': [str]}     # which ones; empty on a healthy search

    ``folded_child_ids`` appears only for a grouped entry that actually folded
    children, so an ungrouped row carries no empty-list noise.

    ``failed_stores`` is passed IN rather than read off ``results`` because the
    MCP producer summarises the GROUPED list — a plain ``list[dict]`` carrying
    no degrade metadata — while the fact lives on the ``SearchResults`` beside
    it.  One argument is what keeps all three producers on one rule.
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
        **_degradation(failed_stores),
    }
