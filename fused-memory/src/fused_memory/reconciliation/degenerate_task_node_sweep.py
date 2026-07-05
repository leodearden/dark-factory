"""Degenerate "tasks N" placeholder Graphiti node sweep — task 2107.

Stage 1 (``MemoryConsolidator``) has no deterministic sweep that keeps
"tasks N" placeholder entity nodes in sync for terminal tasks. Those nodes
("tasks 142", "tasks 144", "tasks 148", ...) are an unintended byproduct of
graphiti-core's LLM-driven entity extraction (``GraphitiBackend.add_episode``)
rather than anything this codebase names or creates deliberately — a
repo-wide search for ``tasks \\d+`` / ``f"tasks`` node-naming returns zero
source matches. When such a node ends up with zero valid edges (no facts
were ever attached, or its facts were all invalidated), it is a pure
placeholder with nothing to backfill from: the real rationale for a
task's completion or cancellation already lives in Mem0 (task evidence
written elsewhere). This module's helpers DELETE those degenerate
placeholders rather than attempting to backfill them.

Design decisions (captured in plan.json):

- Sweep covers BOTH done and cancelled terminal tasks — cancelled is the
  newly-required coverage; done closes the same class of degenerate node.
  Active tasks are never swept: their nodes may be legitimately mid-ingestion
  (edge_count could be transiently 0).
- Detection uses ``GraphitiBackend.find_duplicate_entity_nodes(name)``, which
  is the only exact-name finder that surfaces ``edge_count`` (despite its
  dedup-oriented name/docstring — see graphiti_client.py:1209). A returned
  node is degenerate iff ``edge_count == 0``; a node with >= 1 valid edge is
  NEVER touched (safety invariant, pinned by tests).
- Node-name template defaults to ``("tasks {id}",)`` — lowercase plural,
  matching the evidence exactly — and is matched case-sensitively via
  ``find_duplicate_entity_nodes``'s exact ``{name: $name}`` Cypher match, so
  legitimate canonical "Task {id}"-style nodes are never touched. Templates
  are parameterized so future observed casings/singular forms can be added
  without a code change.
- Best-effort throughout (modelled on
  ``MemoryService._dedup_episode_nodes``, memory_service.py:473): a
  transient backend error scanning or deleting one task id/name must not
  abort the sweep for the remaining ids — it is tallied into
  ``stats['errors']`` and the loop continues.
"""

from __future__ import annotations

import logging

from fused_memory.reconciliation.task_filter import FilteredTaskTree

logger = logging.getLogger(__name__)

# Node-name templates checked per terminal task id. Lowercase plural "tasks
# {id}" matches the observed degenerate-node evidence exactly ("tasks 142",
# "tasks 144", "tasks 148"). Parameterized (rather than hardcoded in the sweep
# loop) so a future observed casing/singular form can be added without
# touching the sweep logic itself.
DEFAULT_TASK_NODE_NAME_TEMPLATES: tuple[str, ...] = ('tasks {id}',)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def extract_terminal_task_ids(tree: FilteredTaskTree | None) -> list[str]:
    """Return deduped str task ids from tree.done_tasks + tree.cancelled_tasks.

    active_tasks is never a source — only terminal (done/cancelled) tasks are
    swept for degenerate placeholder nodes. Ids are coerced to str via the
    same int-parse rule as ``task_filter.id_key``, but — unlike ``id_key``,
    which defaults unparseable/missing ids to ``0`` for sorting — a dict that
    is non-dict, missing its ``'id'`` key, or carries an unparseable ``'id'``
    is silently SKIPPED rather than contributing a spurious ``'0'``: an
    accidental "task 0" id could cause the sweep to delete an unrelated node.

    Order is done_tasks first, then cancelled_tasks, each in list order;
    duplicates (an id appearing more than once across both lists) are
    dropped, keeping the first-seen occurrence.

    Returns [] when tree is None or neither done_tasks nor cancelled_tasks
    contains a task with a parseable id.

    Pure: no I/O, no side effects.
    """
    if tree is None:
        return []

    seen: set[str] = set()
    result: list[str] = []
    for task in (*tree.done_tasks, *tree.cancelled_tasks):
        if not isinstance(task, dict):
            continue
        if 'id' not in task:
            continue
        try:
            tid = str(int(task['id']))
        except (TypeError, ValueError):
            continue
        if tid not in seen:
            seen.add(tid)
            result.append(tid)
    return result
