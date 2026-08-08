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

PRECISION OVER RECALL. The consumer performs destructive edge surgery, so a
false positive misattributes a local fact — the same bug in the opposite
direction. Three narrowings live here (a qualifier naming the current group is
not cross-project; a qualifier that is itself task VOCABULARY — 'Task: 2500',
'subtask: 2500' — is never a project id; a task number mentioned both
qualified and bare in one episode is ambiguous and is reported separately
rather than guessed at) plus an optional fourth (the known-projects
allowlist). For most shape-valid prose the decisive narrowing lives in the
consumer: it splits only when the episode actually touched a node literally
named ``Task N``, which a stray prose match ('Total: 42 items') essentially
never coincides with.

That delegation has a LIMIT, and it is why the task-vocabulary rejection has
to live here rather than downstream. The consumer's guard is decisive only for
qualifiers which do not themselves spell a local task. For the literal string
'Task: 2500' the node named ``Task 2500`` is PRECISELY what extraction
produces, so the guard is guaranteed to CO-OCCUR rather than to filter — it
cannot save you, and the split would move a local task's facts onto a bogus
'task:2500' entity. Do not re-derive the confidence that the consumer catches
everything shape-valid; it catches everything shape-valid EXCEPT this.

This module is a dependency-free leaf — it imports only stdlib and
utils/validation (itself a leaf) — so it can be imported from both the
services write path and any future reconciliation sweep without import
cycles. Mirrors utils/task_naming.py, whose shape it copies.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass, field

from fused_memory.utils.validation import (
    PathShapedProjectIdError,
    canonicalize_project_id,
)

# A project-qualified task reference: '<qualifier>:<digits>'.
#
# - The lookbehind rejects a qualifier glued to a preceding word character or
#   path/URL punctuation, which is what keeps source locations
#   ('graphiti_client.py:2091' — '.py' is preceded by '.'), URL authorities
#   ('example.com:8080'), colon-chained tokens ('a:b:3') and digit-prefixed
#   junk ('2500dark_factory:2500') out.
# - The qualifier must start with a letter, so clock times ('12:30') never
#   match, and must be at least 3 characters, so short non-project tokens
#   ('w6:2', 'py:3', 'a:1') never match.
# - '\s*' around the colon tolerates the spacing humans actually write.
# - '(?!\d)' anchors the number's right edge so a truncated prefix is never
#   captured.
_QUALIFIED_REF_PATTERN = re.compile(r'(?<![\w:/.-])([A-Za-z][A-Za-z0-9_-]{2,})\s*:\s*(\d+)(?!\d)')

# An UNqualified mention of a task number in the same prose — the unanchored
# variant of task_naming._TASK_NODE_NAME_PATTERN. The lookbehind reuses that
# module's refusal to match word-glued lookalikes ('subtask 5', 'multitask 3',
# 'reify-task 12'), so only a genuine standalone 'task N' counts.
_BARE_TASK_MENTION_PATTERN = re.compile(r'(?<![\w:-])tasks?\s+(\d+)(?!\d)', re.IGNORECASE)

# A COLON-spelled local task mention ('Task: 2500'), which the qualified
# pattern would otherwise capture as a project named 'task'. Shares
# _BARE_TASK_MENTION_PATTERN's lookbehind so the 'task: N' substring inside
# 'subtask: N' / 'multitask: N' is not matched.
_BARE_TASK_COLON_PATTERN = re.compile(r'(?<![\w:-])tasks?\s*:\s*(\d+)(?!\d)', re.IGNORECASE)

# Task-vocabulary words are never project ids. This is the ONE rejection the
# consumer's decisive 'episode touched a node named Task N' guard cannot make
# for us: for 'Task: 2500' that node is exactly what extraction produces, so
# the guard is guaranteed to co-occur rather than to filter, and the split
# would move a LOCAL task's facts onto a bogus 'task:2500' entity.
#
# Matched with fullmatch() against the CANONICALIZED qualifier, so every
# spelling ('Task', 'TASK', 'sub-task', 'Sub-Tasks') collapses onto this one
# check, while a real project id that merely starts with 'task' ('taskmaster')
# is not rejected.
_TASK_VOCABULARY_QUALIFIER = re.compile(r'(sub_?)?tasks?')


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


def _canonical_allowlist(known_project_ids: Collection[str] | None) -> frozenset[str] | None:
    """Canonicalize *known_project_ids*, or return None for permissive mode.

    Returns None when the registry is missing or empty. That is PERMISSIVE by
    design and mirrors ``validate_known_project_id``'s documented
    empty-registry mode: failing closed would silently disable this protection
    entirely on any deployment that never wires the registry, which is the
    exact silent-degradation failure mode this detector exists to eliminate.

    A path-shaped registry key is skipped rather than normalized (normalizing a
    mangled path would mint a new, wrong canonical key — RCA §4) and rather
    than raised, so one bad entry never disables the whole allowlist.
    """
    if not known_project_ids:
        return None
    canonical = set()
    for raw in known_project_ids:
        try:
            canonical.add(canonicalize_project_id(raw))
        except PathShapedProjectIdError:
            continue
    return frozenset(canonical)


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
            PERMISSIVE — see :func:`_canonical_allowlist`.

    Returns:
        A :class:`CrossProjectRefScan`. ``refs`` is safe to act on;
        ``ambiguous`` must not be split silently.

    A path-shaped *group_id* yields an empty scan: without a trustworthy local
    project id, local and foreign references cannot be told apart, and the
    conservative answer is to repair nothing.
    """
    if not content:
        return CrossProjectRefScan()
    try:
        local_project = canonicalize_project_id(group_id)
    except PathShapedProjectIdError:
        return CrossProjectRefScan()

    allowlist = _canonical_allowlist(known_project_ids)
    bare_mentions = {m.group(1) for m in _BARE_TASK_MENTION_PATTERN.finditer(content)}
    # Colon-spelled local mentions count as bare mentions too. This is a
    # SEPARATE defence from the qualifier rejection below, guarding a different
    # failure: rejection stops 'task' becoming a project id at all, while this
    # stops a genuinely FOREIGN ref being split when the same number is also
    # named as a local task in colon form ('dark_factory:2500 and Task: 2500').
    bare_mentions |= {m.group(1) for m in _BARE_TASK_COLON_PATTERN.finditer(content)}

    refs: list[CrossProjectRef] = []
    ambiguous: list[CrossProjectRef] = []
    seen: set[tuple[str, str]] = set()
    for match in _QUALIFIED_REF_PATTERN.finditer(content):
        qualifier, task_number = match.group(1), match.group(2)
        try:
            project_id = canonicalize_project_id(qualifier)
        except PathShapedProjectIdError:
            # Defensive: the pattern cannot capture a '/' or a leading '-', so
            # this is unreachable today. Kept so a future pattern relaxation
            # skips the candidate instead of raising into the write path.
            continue
        if _TASK_VOCABULARY_QUALIFIER.fullmatch(project_id):
            continue
        if project_id == local_project:
            continue
        if allowlist is not None and project_id not in allowlist:
            continue
        key = (project_id, task_number)
        if key in seen:
            continue
        seen.add(key)
        ref = CrossProjectRef(project_id=project_id, task_number=task_number)
        if task_number in bare_mentions:
            ambiguous.append(ref)
        else:
            refs.append(ref)
    return CrossProjectRefScan(refs=refs, ambiguous=ambiguous)
