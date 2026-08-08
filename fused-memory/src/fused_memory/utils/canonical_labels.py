"""THE canonical task-label vocabulary — one normative site, no second copy.

This module owns the single description of what a canonical task label looks
like: the anchored node-name form ('Task 132'), the unanchored prose mention
('blocked on task 132'), the project-qualified cross-project form
('dark_factory:2500'), and the narrowings that keep colon-bearing noise out.

INV-5 (no lockstep duplication) / PRD resolved decision 5. Before task 3667
the vocabulary existed as separate compiled copies in utils/task_naming.py
and utils/cross_project_refs.py, which had already drifted: task_naming's
``^\\s*tasks?\\s+(\\d+)\\s*$`` requires whitespace between the word and the
digits and is therefore STRUCTURALLY unable to see 'task #1153' — one of the
PRD's 53 measured task-node variant splits — while cross_project_refs had
grown a second, colon-spelled mention pattern task_naming never got. Both of
those modules now call this one and carry NO compiled pattern of their own;
their test suites assert that structurally. Do not re-introduce a second copy
of the label pattern anywhere: a vocabulary that exists twice drifts, and the
drift is invisible until a destructive consumer acts on the stale half.

This module is a dependency-free leaf — it imports only stdlib and
utils/validation (itself a leaf) — so it can be imported from both the
services write path and any future reconciliation sweep without import
cycles. Mirrors utils/task_naming.py and utils/cross_project_refs.py, whose
shape it copies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from fused_memory.utils.validation import (
    PathShapedProjectIdError,
    canonicalize_project_id,
)

#: The registry of referent kinds and the bare node-name label each renders.
#: Deliberately holds only 'task' today; 'escalation' is the PRD's next entry
#: and is NOT added speculatively. Extending the vocabulary means adding an
#: entry here, which is the whole point of routing every caller through one
#: module.
_KIND_LABELS: dict[str, str] = {'task': 'Task'}

# ANCHORED (whole-string) task-node name: a bare 'task(s) N', optionally padded
# with whitespace, case-insensitive. Anchoring means names that merely mention a
# task ('Task 42 orchestrator', 'reify task 12') or resemble but aren't a
# task-node name ('subtask 5', 'multitask 3', 'taskforce 9') never match, so
# they are never renamed.
#
# The separator alternation REQUIRES at least one of whitespace, '#' or ':'.
# That is load-bearing: 'task #1153' and 'Task: 132' — two of the PRD's 53
# measured variant splits, structurally invisible to this pattern's ancestor
# '^\s*tasks?\s+(\d+)\s*$' — now match, while 'task132' still does NOT. The
# obvious spelling 'tasks?\s*#?\s*(\d+)' would have matched 'task132' too,
# silently widening what _normalize_task_node_names renames.
_TASK_NODE_NAME_PATTERN = re.compile(r'^\s*tasks?(?:\s*[#:]\s*|\s+)(\d+)\s*$', re.IGNORECASE)

# The ANCHORED twin of _QUALIFIED_REF_PATTERN: a whole-string cross-project node
# name, '<project_id>:<task_number>'. Shares that pattern's letter-start and
# >=3-character qualifier rules, so clock times ('12:30') and short non-project
# tokens ('w6:2', 'py:3') never parse as a project. Case-SENSITIVE start class
# with no IGNORECASE flag needed, since [A-Za-z] already spans both cases.
_QUALIFIED_NODE_NAME_PATTERN = re.compile(r'^\s*([A-Za-z][A-Za-z0-9_-]{2,})\s*:\s*(\d+)\s*$')

# Task-vocabulary words are never project ids. Matched with fullmatch() against
# the CANONICALIZED qualifier, so every spelling ('Task', 'TASK', 'sub-task',
# 'Sub-Tasks') collapses onto this one check, while a real project id that
# merely starts with 'task' ('taskmaster') is not rejected.
#
# Moved verbatim from cross_project_refs, where it guards the ONE rejection the
# split consumer's decisive 'episode touched a node named Task N' guard cannot
# make: for 'Task: 2500' that node is exactly what extraction produces, so the
# guard is guaranteed to CO-OCCUR rather than to filter, and the split would
# move a LOCAL task's facts onto a bogus 'task:2500' entity.
_TASK_VOCABULARY_QUALIFIER = re.compile(r'(sub_?)?tasks?')


@dataclass(frozen=True, kw_only=True)
class Referent:
    """One thing a canonical label refers to: a kind, a project, a number.

    Frozen because a referent is evidence for destructive graph surgery — a
    consumer must not be able to rewrite which node it names.

    Keyword-only so ``number`` can be the one REQUIRED field while ``kind``
    and ``project_id`` carry defaults, making the overwhelmingly common
    own-project task referent spell as ``Referent(number='132')``.
    """

    #: The number's digits VERBATIM — never int-normalized, so a referent
    #: never invents or reformats a task number ('0132' stays '0132', and is
    #: a DIFFERENT referent from '132').
    number: str
    #: Which registry entry this referent belongs to; see :data:`_KIND_LABELS`.
    kind: str = 'task'
    #: EMPTY means own-project / unqualified. A non-empty value is always
    #: ``canonicalize_project_id``-canonical and denotes a DIFFERENT project.
    #: Encoding local-ness in the referent itself is what lets
    #: :attr:`node_name` discriminate 'Task 132' from 'reify:132' from the
    #: referent alone, with no ambient group_id.
    project_id: str = ''

    def __post_init__(self) -> None:
        if self.kind not in _KIND_LABELS:
            raise ValueError(
                f'unregistered referent kind {self.kind!r}; registered kinds are '
                f'{sorted(_KIND_LABELS)}. Add it to canonical_labels._KIND_LABELS '
                'rather than rendering an unlabelled referent.'
            )

    # Derived name is a property rather than a field so it cannot drift out of
    # sync with the source-of-truth fields above (the CrossProjectRef.entity_name
    # precedent).
    @property
    def node_name(self) -> str:
        """The graph node name this referent denotes.

        ``'<project_id>:<number>'`` when the referent names a foreign project
        — the qualifier is a different-project signal and is NEVER normalized
        away to the bare form, because that collapse is precisely the bug
        utils/cross_project_refs.py exists to detect. Otherwise the canonical
        bare label for the kind, e.g. ``'Task 132'``.
        """
        if self.project_id:
            return f'{self.project_id}:{self.number}'
        return f'{_KIND_LABELS[self.kind]} {self.number}'


def parse_node_name(name: str) -> Referent | None:
    """Parse an entity *name* as a whole task label, or return None.

    ANCHORED by design: this answers "is this entity NAME a task label?", not
    "does this text mention a task" (:func:`scan_content` answers that). So a
    name that merely contains a task reference — 'Task 42 orchestrator',
    'reify task 12' — returns None and its node is left untouched.

    Two forms parse:

    - The bare local form 'task(s) N', separated by whitespace, '#' or ':',
      in any case and tolerant of padding, yielding an OWN-project referent
      ('task #1153' -> ``Task 1153``).
    - The project-qualified form '<project_id>:N', yielding a FOREIGN referent
      whose qualifier is canonicalized but never normalized AWAY ('reify:132'
      stays ``reify:132``; flattening it to ``Task 132`` is precisely the
      cross-project collapse utils/cross_project_refs.py exists to detect).

    Digits are preserved VERBATIM, never int-normalized, so a referent never
    invents or reformats a task number ('task 0132' -> ``Task 0132``, and
    ``Referent('0132') != Referent('132')``).

    Returns None for a path-shaped qualifier rather than canonicalizing it: a
    mangled path must never be silently mapped into a new, wrong project key
    (RCA §4). Also None for a task-VOCABULARY qualifier ('subtask: 2500'),
    which is local task vocabulary, not a project id.
    """
    local = _TASK_NODE_NAME_PATTERN.match(name)
    if local is not None:
        # Ordering matters: the local pattern is tried FIRST, which is what
        # makes a vocabulary-qualified name like 'Task: 132' resolve to the
        # local referent it is, rather than to a project named 'task'.
        return Referent(kind='task', number=local.group(1))

    qualified = _QUALIFIED_NODE_NAME_PATTERN.match(name)
    if qualified is None:
        return None
    qualifier, number = qualified.group(1), qualified.group(2)
    try:
        project_id = canonicalize_project_id(qualifier)
    except PathShapedProjectIdError:
        # Defensive: the pattern cannot capture a '/' or a leading '-', so this
        # is unreachable today. Kept so a future pattern relaxation refuses the
        # name instead of raising into a caller's write path.
        return None
    if _TASK_VOCABULARY_QUALIFIER.fullmatch(project_id):
        # 'subtask: 2500' / 'sub-task: 2500'. The local pattern already claimed
        # every spelling that IS a local label ('Task: 132'), so anything
        # reaching here is a word-glued lookalike, not a task label at all.
        return None
    return Referent(kind='task', project_id=project_id, number=number)
