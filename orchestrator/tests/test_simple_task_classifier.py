"""Tests for the author-declared simple-path gate predicates.

Covers ``has_simple_task_blocker`` (contradiction veto) and
``is_declared_simple_task`` (the composite predicate that replaces the old
title-regex / file-count / priority classifier).
"""

from __future__ import annotations

import pytest

from orchestrator.agents.triage import has_simple_task_blocker, is_declared_simple_task


def _task(
    *,
    description: str = 'Add a docstring to foo.',
    priority: str | None = None,
    complexity: str | None = None,
    files: list[str] | None = None,
) -> dict:
    metadata: dict = {}
    if complexity is not None:
        metadata['complexity'] = complexity
    if files is not None:
        metadata['files'] = files
    return {
        'description': description,
        'priority': priority,
        'metadata': metadata if metadata else {},
    }


class TestHasSimpleTaskBlocker:
    @pytest.mark.parametrize(
        'description',
        [
            'Run integration test against staging',
            'Database migration to v3',
            'Refactor architecture for new module',
            'Design a new caching layer',
            'Implement the new feature requested in PRD',
        ],
    )
    def test_returns_true_for_blocker_tokens(self, description: str):
        assert has_simple_task_blocker(description) is True

    @pytest.mark.parametrize(
        'description',
        [
            'Quick docstring fix',
            'Add a docstring to foo.',
            '',
        ],
    )
    def test_returns_false_for_clean_descriptions(self, description: str):
        assert has_simple_task_blocker(description) is False


class TestIsDeclaredSimpleTask:
    def test_declared_simple_with_clean_description_routes_to_simple_path(self):
        """complexity='simple' + no blocker → True (routes to simple path)."""
        task = _task(complexity='simple', description='Add a docstring to foo.')
        assert is_declared_simple_task(task) is True

    def test_no_complexity_field_routes_to_full_path(self):
        """No complexity field → False (routes to full path)."""
        task = _task(description='Add a docstring to foo.')
        assert is_declared_simple_task(task) is False

    def test_complexity_complex_routes_to_full_path(self):
        """complexity='complex' → False (routes to full path)."""
        task = _task(complexity='complex', description='Add a docstring to foo.')
        assert is_declared_simple_task(task) is False

    def test_complexity_other_value_routes_to_full_path(self):
        """Any value other than 'simple' → False."""
        task = _task(complexity='medium', description='Add a docstring to foo.')
        assert is_declared_simple_task(task) is False

    def test_declared_simple_with_blocker_routes_to_full_path(self):
        """complexity='simple' + blocker token → False (veto fires → full path)."""
        task = _task(complexity='simple', description='Database migration to v3')
        assert is_declared_simple_task(task) is False

    def test_declared_simple_high_priority_routes_to_simple_path(self):
        """complexity='simple' + priority='high' → True (priority guard dropped)."""
        task = _task(complexity='simple', description='Add a docstring to foo.', priority='high')
        assert is_declared_simple_task(task) is True

    def test_declared_simple_many_files_routes_to_simple_path(self):
        """complexity='simple' + many files → True (file-count cap dropped)."""
        task = _task(
            complexity='simple',
            description='Add a docstring to foo.',
            files=['a.py', 'b.py', 'c.py', 'd.py', 'e.py', 'f.py'],
        )
        assert is_declared_simple_task(task) is True

    def test_missing_metadata_routes_to_full_path(self):
        """Missing metadata entirely → False."""
        task = {'description': 'Add a docstring to foo.'}
        assert is_declared_simple_task(task) is False

    def test_empty_metadata_routes_to_full_path(self):
        """Empty metadata dict (no complexity field) → False."""
        task = {'description': 'Add a docstring to foo.', 'metadata': {}}
        assert is_declared_simple_task(task) is False
