"""Unit tests for fused_memory.utils.task_naming.canonicalize_task_node_name.

canonicalize_task_node_name is a pure, dependency-free helper used by the
post-add_episode node-name-normalization hook (MemoryService.
_normalize_task_node_names, task 2110) to detect and correct non-canonical
task-entity node names (e.g. 'task 132', 'tasks 153') minted by graphiti-core's
LLM entity extraction, without ever touching legitimate non-task-node names.
"""
from __future__ import annotations

import pytest

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
