"""Shared Mem0 prior-marker dedup helper for reconciliation modules.

Both ``harness._replay_deferred_writes`` and ``flag_dedup.dedup_flags``
implement the same search→metadata-equality-filter→log-on-failure→return-match
pattern.  This module extracts that pattern into a single function so both
callers get consistent behaviour: symmetric str coercion, defensive metadata
guard, category-constrained search, and search-failure degrades to None/[].

Public API
----------
- ``find_prior_memories(memory_service, *, project_id, task_id, kind, query, ...)``
  — returns all matching MemoryResults as a list (empty list on failure/no match).
- ``find_prior_memory(memory_service, *, project_id, task_id, kind, query, ...)``
  — returns the first matching MemoryResult or None; thin wrapper around
  ``find_prior_memories``.
"""
from __future__ import annotations

import logging
from typing import Any

_module_logger = logging.getLogger(__name__)


async def find_prior_memories(
    memory_service: Any,
    *,
    project_id: str,
    task_id: str | int,
    kind: dict[str, Any],
    query: str,
    categories: list[str] | None = None,
    limit: int = 50,
    log: logging.Logger | None = None,
) -> list[Any]:
    """Search Mem0 for ALL prior memories matching *task_id* and all *kind* key/values.

    All equality comparisons coerce both sides to ``str`` so that a prior
    written with ``task_id`` as an ``int`` matches a caller passing ``task_id``
    as a ``str``.  The defensive ``(r.metadata or {})`` guard handles mock
    objects and partial records where ``r.metadata`` may be ``None``.

    Args:
        memory_service: Mem0 service with an async ``search`` method.
        project_id: Project scope passed to ``memory_service.search``.
        task_id: Task identifier; compared symmetrically as ``str`` on both sides.
        kind: Extra metadata-equality filters (e.g. ``{'transition': 'done'}``).
              Values may be any type — they are str-coerced on both sides of
              comparison.  ALL keys must match (AND semantics).  A key that is
              absent from the row's metadata never matches, even when the filter
              value is ``''``; only a row that explicitly stores the key with a
              matching (str-coerced) value satisfies the filter.
        query: Embedding query string forwarded to ``memory_service.search``.
        categories: Optional category list forwarded to ``memory_service.search``
                    to constrain the corpus before similarity ranking.
        limit: Result cap forwarded to ``memory_service.search``.
        log: Logger to use for WARNING messages on search failure.  Defaults to
             this module's logger; callers pass their own so WARNINGs appear
             under the caller's logger namespace (preserving caplog-based tests).

    Returns:
        A list of all ``MemoryResult`` objects whose metadata satisfies
        ``task_id`` AND all ``kind`` key/value pairs, in search-result order.
        Returns an empty list if no match is found or if the search fails.
    """
    _log = log or _module_logger
    try:
        results = await memory_service.search(
            query=query,
            project_id=project_id,
            categories=categories,
            limit=limit,
        )
    except Exception as e:
        # Singular 'find_prior_memory' phrasing is the historical canonical grep term.  Both
        # the find_prior_memory (wrapper) and find_prior_memories (this function) share this
        # code path, so the message cannot match both function names anyway.  Keeping the
        # singular phrasing preserves ops dashboard / alert-rule grep continuity across all
        # historical log archives.  Do not change without auditing log-grep consumers.
        _log.warning('find_prior_memory search failed for task %s: %s', task_id, e)
        return []

    task_id_str = str(task_id)
    matches = []
    for r in results:
        meta = r.metadata or {}
        if str(meta.get('task_id', '')) != task_id_str:
            continue
        if all(k in meta and str(meta[k]) == str(v) for k, v in kind.items()):
            matches.append(r)

    return matches


async def find_prior_memory(
    memory_service: Any,
    *,
    project_id: str,
    task_id: str | int,
    kind: dict[str, Any],
    query: str,
    categories: list[str] | None = None,
    limit: int = 50,
    log: logging.Logger | None = None,
) -> Any | None:
    """Search Mem0 for a prior memory matching *task_id* and all *kind* key/values.

    Thin wrapper around :func:`find_prior_memories` that returns the first match
    or ``None``.  Preserves the existing first-match-or-None contract so callers
    that only need one result (e.g. ``harness._replay_deferred_writes``) are
    unaffected.

    Args:
        memory_service: Mem0 service with an async ``search`` method.
        project_id: Project scope passed to ``memory_service.search``.
        task_id: Task identifier; compared symmetrically as ``str`` on both sides.
        kind: Extra metadata-equality filters.  See :func:`find_prior_memories`.
        query: Embedding query string forwarded to ``memory_service.search``.
        categories: Optional category list forwarded to ``memory_service.search``.
        limit: Result cap forwarded to ``memory_service.search``.
        log: Logger to use for WARNING messages on search failure.

    Returns:
        The first ``MemoryResult`` whose metadata satisfies ``task_id`` AND all
        ``kind`` key/value pairs, or ``None`` if no match or search fails.
    """
    matches = await find_prior_memories(
        memory_service,
        project_id=project_id,
        task_id=task_id,
        kind=kind,
        query=query,
        categories=categories,
        limit=limit,
        log=log,
    )
    return matches[0] if matches else None
