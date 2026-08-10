"""Detection of project-qualified cross-project task references in episode text.

CONVENTION (normative): a cross-project task reference is named
``<project_id>:<task_number>`` in the graph — never a bare ``Task N``. Bare
``Task N`` is reserved for the task owned by the entity's own group.

graphiti-core's LLM entity extraction (invoked by GraphitiBackend.add_episode)
does not honour that convention: given an episode body that says
``dark_factory:2500``, it normalizes the reference to the canonical bare form
``Task 2500`` and then name-dedupes it against the CURRENT group's own,
unrelated ``Task 2500`` node. Entity names are partitioned only by ``group_id``
(= ``project_id``) and every project's task ids are independently-incrementing
integers, so this silently attaches one project's facts to another project's
task entity. ``find_cross_project_task_refs`` is the pure detector used by the
post-write split hook (MemoryService._split_cross_project_task_nodes, task
3335) to find and repair those collapses.

It reads the VERBATIM episode body rather than the extracted node names or
edge facts, because the qualifier is exactly what extraction discards — that
is the bug — so neither the resulting node name nor the LLM-restated fact can
be relied on to still contain it. The content, by contrast, reaches
``graphiti.add_episode(episode_body=...)` byte-for-byte from the MCP argument.

This module is now an ADAPTER. Since task 3667 the label vocabulary and every
precision narrowing live in utils/canonical_labels.py, the single normative
site (INV-5 / PRD decision 5), and this module owns no compiled pattern of its
own — tests/test_cross_project_refs.py asserts that structurally. All this
module adds is its own public types and the FOREIGN-only filter that makes
those types mean what they have always meant. Consult canonical_labels for the
narrowings; do NOT re-derive them here, and do not add a second copy.

PRECISION OVER RECALL, still. The consumer performs destructive edge surgery,
so a false positive misattributes a local fact — the same bug in the opposite
direction. Four narrowings apply (a qualifier naming the current group is not
cross-project; a qualifier that is itself task VOCABULARY — 'Task: 2500',
'subtask: 2500' — is never a project id; a task number mentioned both
qualified and bare in one episode is ambiguous and is reported separately
rather than guessed at; plus the optional known-projects allowlist). For most
shape-valid prose the decisive narrowing lives in the consumer: it splits only
when the episode actually touched a node literally named ``Task N``, which a
stray prose match ('Total: 42 items') essentially never coincides with.

That delegation has a LIMIT, and it is why the task-vocabulary rejection has
to live upstream in canonical_labels rather than downstream in the consumer.
The consumer's guard is decisive only for qualifiers which do not themselves
spell a local task. For the literal string 'Task: 2500' the node named ``Task
2500`` is PRECISELY what extraction produces, so the guard is guaranteed to
CO-OCCUR rather than to filter — it cannot save you, and the split would move
a local task's facts onto a bogus 'task:2500' entity. Do not re-derive the
confidence that the consumer catches everything shape-valid; it catches
everything shape-valid EXCEPT this.

This module is a dependency-free leaf — it imports only utils/canonical_labels
(itself a leaf) — so it can be imported from both the services write path and
any future reconciliation sweep without import cycles.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

from fused_memory.utils.canonical_labels import Referent, scan_content


@dataclass(frozen=True)
class CrossProjectRef:
    """One project-qualified task reference found in an episode body.

    Frozen because it is evidence for destructive graph surgery: a consumer
    must not be able to rewrite which node gets split.
    """

    #: Canonical (lowercase, underscore-separated) project id of the qualifier.
    project_id: str
    #: The task number's digits VERBATIM — never int-normalized, mirroring
    #: canonicalize_task_node_name, so the scan never invents or reformats a
    #: task number ('0250' stays '0250').
    task_number: str

    # Derived names are properties rather than fields so they cannot drift out
    # of sync with the two source-of-truth values above.
    @property
    def entity_name(self) -> str:
        """The qualified entity name the reference SHOULD have in the graph."""
        return f'{self.project_id}:{self.task_number}'

    @property
    def task_node_name(self) -> str:
        """The bare node name extraction collapses this reference onto.

        Exactly the canonical form ``canonicalize_task_node_name`` produces,
        which is what lets the consumer key off node names the earlier
        normalization sub-pass has already canonicalized.
        """
        return f'Task {self.task_number}'


@dataclass(frozen=True)
class CrossProjectRefScan:
    """The result of scanning one episode body."""

    #: Refs safe to act on, in first-seen order, de-duplicated on
    #: (project_id, task_number).
    refs: list[CrossProjectRef] = field(default_factory=list)
    #: Refs whose task number ALSO appears as a bare 'task N' mention in the
    #: same content. Such content is genuinely ambiguous about which facts
    #: belong to the foreign task and which to the local one, so the consumer
    #: must refuse to split it and say so loudly rather than guess.
    ambiguous: list[CrossProjectRef] = field(default_factory=list)


def _foreign_refs(referents: Sequence[Referent]) -> list[CrossProjectRef]:
    """Map FOREIGN referents to CrossProjectRefs, preserving order.

    The ``if r.project_id`` filter is what makes this module's output mean
    'cross-project'. Own-project referents — which canonical_labels reports
    both for bare mentions ('task 5181') and for SELF-qualified references
    ('reify:5181' scanned in group reify) — are dropped here by construction.
    That is exactly this module's long-standing behaviour, now expressed as
    one filter instead of two separate rules: bare mentions were never
    reported as refs, and a self-qualified reference was explicitly dropped
    because extraction collapsing it onto 'Task 5181' inside its own graph is
    correct, not a bug.
    """
    return [
        CrossProjectRef(project_id=r.project_id, task_number=r.number)
        for r in referents
        if r.project_id
    ]


def find_cross_project_task_refs(
    content: str,
    *,
    group_id: str,
    known_project_ids: Collection[str] | None = None,
) -> CrossProjectRefScan:
    """Scan *content* for task references qualified with a FOREIGN project id.

    Args:
        content: The verbatim episode body, exactly as it was handed to
            graphiti-core. Falsy content yields an empty scan.
        group_id: The group the episode was written into (= the local
            ``project_id``). References qualified with this project are not
            cross-project and are dropped: extraction collapsing ``reify:5181``
            onto ``Task 5181`` inside the reify graph is correct, not a bug.
            Both sides are canonicalized before the comparison, so case and
            hyphen/underscore spelling differences still count as local.
        known_project_ids: Optional registry of known project ids (any
            collection; a ``{project_id: project_root}`` mapping works, since
            iterating it yields its keys). When non-empty, refs naming a
            project outside it are dropped. When None or empty the filter is
            PERMISSIVE — see :func:`canonical_labels._canonical_allowlist`.

    Returns:
        A :class:`CrossProjectRefScan`. ``refs`` is safe to act on;
        ``ambiguous`` must not be split silently.

    A path-shaped *group_id* yields an empty scan: without a trustworthy local
    project id, local and foreign references cannot be told apart, and the
    conservative answer is to repair nothing.
    """
    scan = scan_content(content, group_id=group_id, known_project_ids=known_project_ids)
    return CrossProjectRefScan(
        refs=_foreign_refs(scan.refs),
        ambiguous=_foreign_refs(scan.ambiguous),
    )
