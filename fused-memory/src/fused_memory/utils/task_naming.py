"""Canonicalization for Graphiti task-entity node names.

graphiti-core's LLM entity extraction (invoked by GraphitiBackend.add_episode)
sometimes mints task-entity nodes with non-canonical names — e.g. 'task 132'
(lowercase), 'tasks 153' (lowercase, plural) or 'task #1153' (hash-spelled) —
instead of the canonical 'Task N' form. ``canonicalize_task_node_name`` is the
pure detector used by the post-write normalization hook
(MemoryService._normalize_task_node_names) to find and correct these without
touching any other entity node.

This module owns NO pattern of its own. The task-label vocabulary lives in
utils/canonical_labels.py, the single normative site (INV-5 / PRD decision 5),
and this module is a thin adapter over its ``parse_node_name``. That is not
cosmetic: this module used to carry its own compiled copy, which had already
drifted from the one in utils/cross_project_refs.py — 'task #1153' was a
task-node name to a human and to that module's mention scanner, but not to
the copy here. tests/test_task_naming.py asserts the absence of a second copy
structurally, so re-introducing one fails the suite.

This module is a dependency-free leaf (canonical_labels is itself a leaf) so
it can be imported from both the services write path and any future
reconciliation sweep without import cycles.
"""
from __future__ import annotations

from fused_memory.utils.canonical_labels import parse_node_name


def canonicalize_task_node_name(name: str) -> str | None:
    """Return the canonical 'Task N' form of a bare task-node *name*, or None.

    Matches only a bare 'task N' or 'tasks N' string (any case, tolerant of
    leading/trailing/internal whitespace around the number) and returns
    ``f'Task {digits}'`` with the digits preserved verbatim — never
    int-normalized, so the result never invents or reformats a task number
    (e.g. 'task 0132' maps to 'Task 0132', not 'Task 132').

    The separator between the word and the number may be whitespace, '#' or
    ':', so the variant spellings 'task #1153' and 'Task: 132' canonicalize
    too. Those are two of the PRD's 53 measured task-node variant splits, and
    ``_normalize_task_node_names`` renaming such a node onto 'Task 1153' is
    the intended collapse. A separator is REQUIRED, so 'task132' is still not
    a match.

    Returns None for anything that is not a bare task-node name — including
    non-task entity names ('Alice'), a bare 'task' with no number, and names
    that merely contain a task reference ('Task 42 orchestrator', 'reify task
    12', 'subtask 5', 'multitask 3', 'taskforce 9'). Callers use this None
    return to leave those nodes untouched.

    Also returns None for a PROJECT-QUALIFIED name such as 'reify:132', which
    ``parse_node_name`` does parse. A qualifier is a different-project signal
    and must never be normalized away to 'Task 132': that collapse is exactly
    the cross-project misattribution utils/cross_project_refs.py exists to
    detect, and doing it here would have the normalization hook cause the very
    bug the split hook repairs.

    Already-canonical input maps to itself (e.g. 'Task 42' -> 'Task 42'),
    making the function idempotent — safe to re-apply to its own output.

    Args:
        name: The Entity node's current name.

    Returns:
        The canonical 'Task N' string, or None if *name* is not a bare
        task-node name.
    """
    referent = parse_node_name(name)
    if referent is None or referent.project_id:
        return None
    return referent.node_name
