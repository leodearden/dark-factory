"""Shared Mem0 prior-marker dedup helper for reconciliation modules.

Both ``harness._replay_deferred_writes`` and ``flag_dedup.dedup_flags``
implement the same search→metadata-equality-filter→log-on-failure→return-match
pattern.  This module extracts that pattern into a single function so both
callers get consistent behaviour: symmetric str coercion, defensive metadata
guard, category-constrained search, and search-failure degrades to None.

Public API
----------
- ``find_prior_memory(memory_service, *, project_id, task_id, kind, query, ...)``
  — returns the first matching MemoryResult or None.
"""
from __future__ import annotations

import logging
from typing import Any

_module_logger = logging.getLogger(__name__)


async def find_prior_memory(
    memory_service: Any,
    *,
    project_id: str,
    task_id: str | int,
    kind: dict[str, str],
    query: str,
    categories: list[str] | None = None,
    limit: int = 50,
    log: logging.Logger | None = None,
) -> Any | None:
    """Search Mem0 for a prior memory matching *task_id* and all *kind* key/values.

    All equality comparisons coerce both sides to ``str`` so that a prior
    written with ``task_id`` as an ``int`` matches a caller passing ``task_id``
    as a ``str``.  The defensive ``(r.metadata or {})`` guard handles mock
    objects and partial records where ``r.metadata`` may be ``None``.

    Args:
        memory_service: Mem0 service with an async ``search`` method.
        project_id: Project scope passed to ``memory_service.search``.
        task_id: Task identifier; compared symmetrically as ``str`` on both sides.
        kind: Extra metadata-equality filters (e.g. ``{'transition': 'done'}``).
              All values are str-coerced; ALL keys must match (AND semantics).
        query: Embedding query string forwarded to ``memory_service.search``.
        categories: Optional category list forwarded to ``memory_service.search``
                    to constrain the corpus before similarity ranking.
        limit: Result cap forwarded to ``memory_service.search``.
        log: Logger to use for WARNING messages on search failure.  Defaults to
             this module's logger; callers pass their own so WARNINGs appear
             under the caller's logger namespace (preserving caplog-based tests).

    Returns:
        The first ``MemoryResult`` whose metadata satisfies ``task_id`` AND all
        ``kind`` key/value pairs, or ``None`` if no match or search fails.
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
        _log.warning('find_prior_memory search failed for task %s: %s', task_id, e)
        return None

    task_id_str = str(task_id)
    for r in results:
        meta = r.metadata or {}
        if str(meta.get('task_id', '')) != task_id_str:
            continue
        if all(str(meta.get(k, '')) == str(v) for k, v in kind.items()):
            return r

    return None
