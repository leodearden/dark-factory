"""Async fetchers for task state via fused-memory MCP HTTP endpoint.

Replaces the legacy ``.taskmaster/tasks/tasks.json`` readers after the
2026-05-02 SQLite cutover made fused-memory the sole owner of task state.

The dashboard's per-task wire shape is preserved here so consumers
(``active_tasks``, ``orchestrator``, ``burndown``, ``merge_queue``)
do not need to be re-keyed.

Network errors are caught and surfaced as ``{'offline': True, 'error': ...}``;
the caller turns that into a per-project skip plus a Tasks-tab banner.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping
from typing import Any

import httpx

from dashboard.config import DashboardConfig
from dashboard.data.memory import _sessions, mcp_tool_call

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-project_root TTL cache for fetch_tasks
# (mirrors app._load_task_cards / merge_queue.load_task_titles pattern)
#
# Code-duplication note: this is now the third copy of the
# {TTL constant, dict[str, tuple[float, list[dict]]], _clear() hook,
# store-only-on-success, list()-copy} pattern alongside
# app._task_cards_cache and merge_queue._task_titles_cache.  Extracting a
# shared TTLCache helper would require changes to app.py and merge_queue.py,
# which fall outside this task's module lock.  Both caller caches are now
# primarily redundant for MCP de-duplication (the 20 s inner TTL handles it);
# they remain for legacy shaping-cost avoidance and are outside this task's
# scope to remove.
# ---------------------------------------------------------------------------

# Within the PRD's recommended 15-30 s staleness window.  Slightly longer than
# the 10 s caller caches (_TASK_CARDS_TTL_SECONDS / _TASK_TITLES_TTL_SECONDS)
# because fetch_tasks is the dominant full-tree seam — a monitoring view
# tolerates brief staleness; the inner TTL dominates net MCP cadence.
#
# Caller-cache stacking: app._load_task_cards (10 s) and
# merge_queue.load_task_titles (10 s) both cache fetch_tasks output on top of
# this inner cache.  Worst-case combined staleness ≈ caller TTL + inner TTL
# ≈ 10 s + 20 s = 30 s — at the PRD's upper bound; intentional for a
# monitoring view where brief staleness is preferable to MCP hammering.
_FETCH_TASKS_TTL_SECONDS = 20.0
_fetch_tasks_cache: dict[str, tuple[float, list[dict]]] = {}


def _fetch_tasks_cache_clear() -> None:
    """Clear the per-project_root fetch_tasks TTL cache (test/admin hook)."""
    _fetch_tasks_cache.clear()


def _shape_task(task: dict) -> dict | None:
    """Trim an MCP get_tasks row to the dashboard's persistent shape.

    MCP returns top-level ids as strings and includes testStrategy/subtasks
    that the dashboard does not render. Cast id at the boundary; drop those.
    ``updatedAt`` is preserved as ``updated_at`` — it is the recency key for
    ordering done tasks and the ``completed`` display timestamp.
    """
    raw_id = task.get('id')
    if raw_id is None:
        return None
    try:
        tid = int(raw_id)
    except (TypeError, ValueError):
        return None

    raw_deps = task.get('dependencies') or []
    deps: list[int] = []
    for d in raw_deps:
        try:
            deps.append(int(d))
        except (TypeError, ValueError):
            continue

    metadata = task.get('metadata')
    if not isinstance(metadata, dict):
        metadata = {}

    return {
        'id': tid,
        'title': task.get('title') or '',
        'description': task.get('description') or '',
        'details': task.get('details') or '',
        'status': task.get('status'),
        'priority': task.get('priority'),
        'dependencies': deps,
        'metadata': metadata,
        'updated_at': task.get('updatedAt'),
    }


async def fetch_tasks(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: str | bytes | os.PathLike[str],
) -> list[dict] | dict:
    """Fetch the dashboard-shaped task list for *project_root* via MCP.

    Returns a ``list[dict]`` on success, or an offline marker
    ``{'offline': True, 'error': str}`` if every configured server fails.

    Results are cached per *project_root* for ``_FETCH_TASKS_TTL_SECONDS``
    (~20 s) to avoid hammering the MCP server on every render.  All statuses
    (including done tasks and their completed timestamps) are cached unchanged.
    Offline/error markers are never cached so a transient failure does not pin
    empty results for the TTL window.

    **Copy isolation (list-level only):** returns a shallow ``list()`` copy on
    every call, so list-level mutations (``result.clear()``, ``result.append()``)
    do not affect the cached entry.  Inner task dicts are shared references —
    mutating a field in place (e.g. ``result[0]['status'] = 'x'``) WILL corrupt
    the cached entry and other callers' views within the TTL window.  Current
    callers (active_tasks, shape_escalations) build fresh rows and do not mutate
    source dicts.  Switch to ``copy.deepcopy(cached[1])`` if element-level
    isolation becomes necessary.

    **Graceful degradation:** during an MCP outage that begins while a valid
    cache entry exists, callers receive the stale cached list (not the offline
    marker) for up to ``_FETCH_TASKS_TTL_SECONDS`` before the entry expires and
    a fresh attempt is made.  This delays outage detection by up to ~20 s —
    intentional for a monitoring view (stale data preferable to a blank tab).

    **Data consistency:** ``fetch_statuses`` is uncached and returns live data;
    callers that combine a cached task tree (this function) with a live status
    map (``fetch_statuses``) in the same render may observe transiently
    inconsistent rows for up to ~20 s (e.g. a task listed as in-progress in
    the tree but already done per the status map).  The pre-existing 10 s
    caller caches had this property at a narrower window; the 20 s inner cache
    widens it uniformly across all callers.
    """
    errors: list[str] = []
    project_root_str = str(project_root)
    now = time.monotonic()
    cached = _fetch_tasks_cache.get(project_root_str)
    if cached is not None and (now - cached[0]) < _FETCH_TASKS_TTL_SECONDS:
        return list(cached[1])
    for url in config.fused_memory_urls:
        try:
            result = await mcp_tool_call(
                client, url, 'get_tasks', {'project_root': project_root_str},
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError,
                ValueError) as e:
            logger.debug('fetch_tasks failed for %s: %s', url, e)
            errors.append(f'{url}: {e}')
            _sessions.pop(url.rstrip('/'), None)
            continue

        if 'error' in result and 'tasks' not in result:
            errors.append(f'{url}: {result.get("error")}')
            continue

        raw_tasks = result.get('tasks') or []
        shaped: list[dict] = []
        for task in raw_tasks:
            row = _shape_task(task)
            if row is not None:
                shaped.append(row)
        _fetch_tasks_cache[project_root_str] = (now, shaped)
        return list(shaped)

    return {'offline': True, 'error': '; '.join(errors)}


async def fetch_external_statuses(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    deps: list[str],
) -> dict[str, str]:
    """Fetch a ``{dep_id: status}`` map for a list of external dep strings via MCP.

    Calls ``get_external_statuses`` which returns a BARE ``{dep: status}`` map
    (NOT wrapped in a ``'statuses'`` key, unlike ``get_statuses``).

    Short-circuits to ``{}`` when *deps* is empty (no MCP call issued).
    Returns ``{}`` on any network/parse failure (fail-safe: leaves entries at
    the ``'unknown'`` sentinel rather than fabricating statuses or crashing).
    """
    if not deps:
        return {}

    for url in config.fused_memory_urls:
        try:
            result = await mcp_tool_call(
                client, url, 'get_external_statuses', {'deps': deps},
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError,
                ValueError) as e:
            logger.debug('fetch_external_statuses failed for %s: %s', url, e)
            _sessions.pop(url.rstrip('/'), None)
            continue

        if not isinstance(result, dict):
            logger.debug('fetch_external_statuses: unexpected result type %s from %s', type(result), url)
            continue

        if 'error' in result or not result:
            # Structured error dict or empty result (e.g. parse failure) — try next URL.
            # The fail-safe {} is returned below if all URLs exhaust.
            logger.debug('fetch_external_statuses: soft failure from %s: %s', url, result)
            continue

        return result  # bare {dep: status} map

    return {}


async def fetch_statuses(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: str | bytes | os.PathLike[str],
) -> Mapping[Any, Any]:
    """Fetch a compact ``{int(id): status}`` map for *project_root* via MCP.

    Used by the burndown collector — ~95% smaller than ``fetch_tasks``.
    Returns ``{'offline': True, 'error': str}`` if every server fails.
    """
    errors: list[str] = []
    project_root_str = str(project_root)
    for url in config.fused_memory_urls:
        try:
            result = await mcp_tool_call(
                client, url, 'get_statuses', {'project_root': project_root_str},
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError,
                ValueError) as e:
            logger.debug('fetch_statuses failed for %s: %s', url, e)
            errors.append(f'{url}: {e}')
            _sessions.pop(url.rstrip('/'), None)
            continue

        if 'error' in result and 'statuses' not in result:
            errors.append(f'{url}: {result.get("error")}')
            continue

        raw = result.get('statuses') or {}
        out: dict[int, str] = {}
        for raw_id, status in raw.items():
            try:
                out[int(raw_id)] = status
            except (TypeError, ValueError):
                continue
        return out

    return {'offline': True, 'error': '; '.join(errors)}
