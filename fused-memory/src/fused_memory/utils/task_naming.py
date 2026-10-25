"""Which task-node family does a Graphiti entity name belong to?

graphiti-core's LLM entity extraction (invoked by GraphitiBackend.add_episode)
sometimes mints task-entity nodes with non-canonical names — e.g. 'task 132'
(lowercase), 'tasks 153' (lowercase, plural) or 'task #1153' (hash-spelled) —
instead of the canonical 'Task N' form. Every one of those spellings names the
SAME task, and this module is the one place that judgement is made.

It answers the question in two views over a single acceptance rule.
``task_node_referent`` returns the family's structured IDENTITY — a frozen,
hashable ``Referent`` carrying the digits verbatim — and
``canonicalize_task_node_name`` returns that identity's canonical NAME. The
name view is expressed in terms of the identity view, so the two can never
disagree about what is a task node. The post-write normalization hook
(MemoryService._normalize_task_node_names) needs both halves: the identity to
key a family on and to probe the backend with, the name to rename onto.

``group_task_node_families`` lifts the identity view to a BATCH of nodes,
partitioning them into families. It is the one site both family consumers read
— that same normalization hook, and maintenance/task_family_census — so
neither has to re-decide what belongs together.

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

from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

from fused_memory.utils.canonical_labels import Referent, parse_node_name

#: Bound to Mapping rather than fixed to dict so grouping never narrows what a
#: caller gets back: the node rows the backend hands in come out of
#: ``group_task_node_families`` as the very same objects, same type.
_NodeT = TypeVar('_NodeT', bound=Mapping[str, Any])


def task_node_referent(name: str) -> Referent | None:
    """Return the task family a bare task-node *name* belongs to, or None.

    THE acceptance rule for this module, which
    :func:`canonicalize_task_node_name` renders as a string: a name belongs to
    a family only if it parses as a whole task label AND that label is
    unqualified (own-project).

    Matches only a bare 'task N' or 'tasks N' string (any case, tolerant of
    leading/trailing/internal whitespace around the number). The separator
    between the word and the number may be whitespace, '#' or ':', so the
    variant spellings 'task #1153' and 'Task: 132' belong to the same families
    as 'Task 1153' and 'Task 132'. Those are two of the PRD's 53 measured
    task-node variant splits, and collapsing such a node onto the canonical
    node is the intended repair. A separator is REQUIRED, so 'task132' is still
    not a match.

    The returned referent carries the digits VERBATIM as ``.number`` — never
    int-normalized, so it never invents or reformats a task number, and
    ``Referent(number='0132')`` stays a DIFFERENT family from
    ``Referent(number='132')``. Those verbatim digits are what the normalizer
    probes the backend with (``find_entity_nodes_by_name_substring``): they are
    case-free, so one query reaches 'Task', 'task', 'TASK' and 'tasks'
    spellings alike, and highly selective, so the prefilter stays cheap.
    Recovering them by splitting the canonical name back apart would be an
    ad-hoc parser over a meaningful string; the referent is the structured
    carrier of exactly this pair and cannot drift from the name it renders.

    Returns None for anything that is not a bare task-node name — including
    non-task entity names ('Alice'), a bare 'task' with no number, and names
    that merely contain a task reference ('Task 42 orchestrator', 'reify task
    12', 'subtask 5', 'multitask 3', 'taskforce 9'). Callers use this None
    return to leave those nodes untouched.

    Also returns None for a PROJECT-QUALIFIED name such as 'reify:132', which
    ``parse_node_name`` does parse into a foreign referent. A qualifier is a
    different-project signal and must never be normalized away to 'Task 132':
    that collapse is exactly the cross-project misattribution
    utils/cross_project_refs.py exists to detect, and doing it here would have
    the normalization hook cause the very bug the split hook repairs. Refusing
    it at this one site is what keeps every consumer — the name view, the
    family grouping, and the normalizer — from having to remember the rule.

    Args:
        name: The Entity node's current name.

    Returns:
        The own-project ``Referent`` naming the family, or None if *name* is
        not a bare task-node name.
    """
    referent = parse_node_name(name)
    if referent is None or referent.project_id:
        return None
    return referent


def canonicalize_task_node_name(name: str) -> str | None:
    """Return the canonical 'Task N' form of a bare task-node *name*, or None.

    The canonical-NAME view of :func:`task_node_referent`, which holds the
    acceptance rule and the rationale for it; this function adds only the
    rendering. Matching names yield ``f'Task {digits}'`` with the digits
    preserved verbatim (so 'task 0132' maps to 'Task 0132', not 'Task 132'),
    and everything the referent view rejects — non-task names, bare 'task',
    names that merely mention a task, and project-qualified names like
    'reify:132' — yields None here too.

    Already-canonical input maps to itself (e.g. 'Task 42' -> 'Task 42'),
    making the function idempotent — safe to re-apply to its own output.

    Args:
        name: The Entity node's current name.

    Returns:
        The canonical 'Task N' string, or None if *name* is not a bare
        task-node name.
    """
    referent = task_node_referent(name)
    return referent.node_name if referent is not None else None


def group_task_node_families(nodes: Iterable[_NodeT]) -> dict[Referent, list[_NodeT]]:
    """Partition *nodes* into task families, dropping everything that is not one.

    THE single site of the family-grouping rule, which is why a four-line loop
    is a named function. Two consumers need exactly this operation and must
    agree on it:

    - ``MemoryService._normalize_task_node_names`` narrows a backend substring
      probe down to one family with it. The probe is a deliberately dumb
      ``CONTAINS`` match, so it also returns 'Task 6051', 'Task 1605' and
      'reify:605'; this function is where that recall is turned back into
      precision, using :func:`task_node_referent`'s one acceptance rule rather
      than a second copy of the label vocabulary inside a Cypher predicate.
    - ``maintenance/task_family_census`` partitions a whole graph with it to
      count the families that are still fragmented.

    Order is preserved twice over, and the second one is load-bearing rather
    than incidental: families appear in first-seen order, and WITHIN a family
    the nodes keep their input order, so the backend's survivor-first ordering
    (highest provenance_rank — valid RELATES_TO plus Episodic MENTIONS — then
    oldest, then uuid) reaches the normalizer intact and ``members[0]`` is still
    the merge survivor.

    A node with a missing, empty or non-task ``'name'`` is skipped rather than
    raising: this runs on a best-effort post-commit path where one malformed
    row must not abandon every other family in the batch.

    Args:
        nodes: Node mappings, each expected to carry a ``'name'`` key. The
            mappings are returned as-is, never copied or rewritten.

    Returns:
        A dict keyed by the family's :class:`Referent`, each value the list of
        member nodes in input order. Empty when no node names a task.
    """
    families: dict[Referent, list[_NodeT]] = {}
    for node in nodes:
        name = node.get('name')
        if not name:
            continue
        referent = task_node_referent(name)
        if referent is None:
            continue
        families.setdefault(referent, []).append(node)
    return families
