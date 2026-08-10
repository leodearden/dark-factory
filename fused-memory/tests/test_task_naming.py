"""Unit tests for fused_memory.utils.task_naming.canonicalize_task_node_name.

canonicalize_task_node_name is a pure, dependency-free helper used by the
post-add_episode node-name-normalization hook (MemoryService.
_normalize_task_node_names, task 2110) to detect and correct non-canonical
task-entity node names (e.g. 'task 132', 'tasks 153') minted by graphiti-core's
LLM entity extraction, without ever touching legitimate non-task-node names.
"""
from __future__ import annotations

import re

import pytest

from fused_memory.utils import task_naming
from fused_memory.utils.task_naming import canonicalize_task_node_name


class TestCanonicalizeTaskNodeNameMatches:
    """Bare 'task N' / 'tasks N' node names (any case, extra whitespace) canonicalize
    to 'Task N', preserving the matched digits verbatim."""

    @pytest.mark.parametrize(
        ('name', 'expected'),
        [
            ('task 132', 'Task 132'),
            ('tasks 153', 'Task 153'),
            ('TASK 42', 'Task 42'),
            ('Task  7', 'Task 7'),
            (' tasks 9 ', 'Task 9'),
            ('Task 42', 'Task 42'),  # idempotence: already-canonical maps to itself
        ],
    )
    def test_canonicalizes_to_expected(self, name, expected):
        assert canonicalize_task_node_name(name) == expected

    def test_idempotent_on_already_canonical_name(self):
        """Applying canonicalize_task_node_name to its own output is a no-op change
        (same string in, same string out) — required for the hook's canonical!=name
        guard to treat already-canonical nodes as untouched."""
        once = canonicalize_task_node_name('task 132')
        assert once is not None  # narrows str | None -> str for the chained call below
        twice = canonicalize_task_node_name(once)
        assert once == 'Task 132'
        assert twice == 'Task 132'


class TestCanonicalizeTaskNodeNameNonMatches:
    """Non-task-node names, and anything beyond a bare 'task(s) N', return None
    so the caller leaves the node name untouched — no false positives."""

    @pytest.mark.parametrize(
        'name',
        [
            'Alice',
            '',
            'task',  # no number
            'subtask 5',
            'multitask 3',
            'taskforce 9',
            'Task 42 orchestrator',
            'reify task 12',
        ],
    )
    def test_returns_none(self, name):
        assert canonicalize_task_node_name(name) is None


class TestCanonicalizeTaskNodeNameVariantSpellings:
    """Variant spellings of the same task label now canonicalize (task 3667).

    Before the canonical_labels extraction this function owned its own
    compiled pattern, ``^\\s*tasks?\\s+(\\d+)\\s*$``. The ``\\s+`` between the
    word and the digits means it was STRUCTURALLY unable to see 'task #1153' —
    that name returned None and the node was left alone, keeping it split from
    the canonical 'Task 1153' node. It is one of the PRD's 53 measured
    task-node variant splits, and collapsing it is exactly what makes this
    extraction a fix rather than only a refactor: the single consumer,
    MemoryService._normalize_task_node_names, now renames such a node onto the
    canonical form.
    """

    @pytest.mark.parametrize(
        ('name', 'expected'),
        [
            ('task #1153', 'Task 1153'),
            ('Task #1153', 'Task 1153'),
            ('task#1153', 'Task 1153'),
            ('TASK # 1153', 'Task 1153'),
            # A task-vocabulary qualifier is local task vocabulary, never a
            # project id — the same rule cross_project_refs enforces.
            ('Task: 132', 'Task 132'),
        ],
    )
    def test_variant_spellings_canonicalize(self, name, expected):
        assert canonicalize_task_node_name(name) == expected

    def test_glued_digits_still_do_not_match(self):
        """Pins that the added '#'/':' alternation did not widen the pattern
        beyond those two separators. 'task132' was a non-match before the
        extraction and must stay one — the obvious spelling
        'tasks?\\s*#?\\s*(\\d+)' would have matched it, silently changing what
        the normalization hook renames."""
        assert canonicalize_task_node_name('task132') is None

    def test_variant_spelling_result_is_idempotent(self):
        """Required by the hook's ``canonical != name`` guard: re-applying the
        function to its own output must be a no-op, or the hook would rename
        the same node on every pass."""
        once = canonicalize_task_node_name('task #1153')
        assert once == 'Task 1153'
        assert once is not None  # narrows str | None -> str for the chained call
        assert canonicalize_task_node_name(once) == 'Task 1153'


class TestNoSecondCopyOfTheLabelPattern:
    """INV-5 (no lockstep duplication), the invariant task 3667 exists to
    enforce: the task-label vocabulary lives ONLY in utils/canonical_labels.py.

    A vocabulary that exists in two places drifts, and the drift is invisible
    until a destructive consumer acts on the stale half — which is precisely
    how 'task #1153' came to be a task-node name to a human, to
    cross_project_refs' mention scanner, and not to this module.

    Asserted structurally over the imported module's attributes (not as a
    source grep and not as a docstring pin), so re-introducing a compiled copy
    fails the suite rather than merely contradicting a comment.
    """

    def test_module_exposes_no_compiled_pattern(self):
        compiled = [
            name for name, value in vars(task_naming).items() if isinstance(value, re.Pattern)
        ]
        assert compiled == []
