"""Canonicalization for Graphiti task-entity node names.

graphiti-core's LLM entity extraction (invoked by GraphitiBackend.add_episode)
sometimes mints task-entity nodes with non-canonical names — e.g. 'task 132'
(lowercase) or 'tasks 153' (lowercase, plural) — instead of the canonical
'Task N' form. ``canonicalize_task_node_name`` is the pure detector used by
the post-write normalization hook (MemoryService._normalize_task_node_names)
to find and correct these without touching any other entity node.

This module is a dependency-free leaf (mirrors utils/validation.py) so it can
be imported from both the services write path and any future reconciliation
sweep without import cycles.
"""
from __future__ import annotations

import re

# Anchored (whole-string) match only — a bare 'task(s) N', optionally padded
# with whitespace, case-insensitive. Anchoring means names that merely mention
# a task ('Task 42 orchestrator', 'reify task 12') or resemble but aren't a
# task-node name ('subtask 5', 'multitask 3', 'taskforce 9') never match, so
# they are never renamed.
_TASK_NODE_NAME_PATTERN = re.compile(r'^\s*tasks?\s+(\d+)\s*$', re.IGNORECASE)


def canonicalize_task_node_name(name: str) -> str | None:
    """Return the canonical 'Task N' form of a bare task-node *name*, or None.

    Matches only a bare 'task N' or 'tasks N' string (any case, tolerant of
    leading/trailing/internal whitespace around the number) and returns
    ``f'Task {digits}'`` with the digits preserved verbatim — never
    int-normalized, so the result never invents or reformats a task number
    (e.g. 'task 0132' maps to 'Task 0132', not 'Task 132').

    Returns None for anything that is not a bare task-node name — including
    non-task entity names ('Alice'), a bare 'task' with no number, and names
    that merely contain a task reference ('Task 42 orchestrator', 'reify task
    12', 'subtask 5', 'multitask 3', 'taskforce 9'). Callers use this None
    return to leave those nodes untouched.

    Already-canonical input maps to itself (e.g. 'Task 42' -> 'Task 42'),
    making the function idempotent — safe to re-apply to its own output.

    Args:
        name: The Entity node's current name.

    Returns:
        The canonical 'Task N' string, or None if *name* is not a bare
        task-node name.
    """
    match = _TASK_NODE_NAME_PATTERN.match(name)
    if match is None:
        return None
    return f'Task {match.group(1)}'
