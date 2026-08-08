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

from dataclasses import dataclass

#: The registry of referent kinds and the bare node-name label each renders.
#: Deliberately holds only 'task' today; 'escalation' is the PRD's next entry
#: and is NOT added speculatively. Extending the vocabulary means adding an
#: entry here, which is the whole point of routing every caller through one
#: module.
_KIND_LABELS: dict[str, str] = {'task': 'Task'}


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
