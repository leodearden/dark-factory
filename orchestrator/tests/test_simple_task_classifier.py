"""Tests for ``classify_simple_task`` — Lever C's pre-PLAN gate."""

from __future__ import annotations

import pytest

from orchestrator.agents.triage import classify_simple_task


def _task(
    *,
    title: str = 'Document foo',
    description: str = 'Add a docstring to foo.',
    files: list[str] | None = None,
    priority: str | None = None,
) -> dict:
    return {
        'title': title,
        'description': description,
        'priority': priority,
        'metadata': {'files': files or ['mod_a/foo.py']},
    }


class TestClassifySimpleTask:
    @pytest.mark.parametrize(
        'title',
        [
            'Document foo',
            'document the public API',
            'comment on the regex',
            'docstring for parser',
            'rename old_name to new_name',
            'fix typo in description',
            'tighten error messages',
            'clarify the contract',
            'inline a one-shot helper',
            'extract _helper from process()',
            'add note about thread safety',
            'update comment in file.py',
            'cleanup unused imports',
            'small refactor — split helper',
            'simplify the matcher',
            'deduplicate the helper functions',
        ],
    )
    def test_accepts_simple_titles(self, title: str):
        assert classify_simple_task(_task(title=title)) is True

    @pytest.mark.parametrize(
        'title',
        [
            'Implement new feature',
            'Add database migration',
            'Build integration tests',
            'Refactor authentication system',
            '',
        ],
    )
    def test_rejects_non_simple_titles(self, title: str):
        assert classify_simple_task(_task(title=title)) is False

    def test_rejects_when_more_than_two_files(self):
        task = _task(files=['a.py', 'b.py', 'c.py'])
        assert classify_simple_task(task) is False

    def test_accepts_two_files(self):
        task = _task(files=['a.py', 'b.py'])
        assert classify_simple_task(task) is True

    def test_accepts_one_file(self):
        task = _task(files=['a.py'])
        assert classify_simple_task(task) is True

    def test_accepts_empty_files(self):
        # 0 <= 2 — the agent will explore briefly to identify targets
        task = _task(files=[])
        assert classify_simple_task(task) is True

    def test_rejects_files_not_a_list(self):
        task = _task()
        task['metadata']['files'] = 'not-a-list'
        assert classify_simple_task(task) is False

    def test_rejects_high_priority(self):
        task = _task(priority='high')
        assert classify_simple_task(task) is False

    @pytest.mark.parametrize(
        'priority',
        ['low', 'medium', 'critical', 'polish', None],
    )
    def test_accepts_other_priorities(self, priority):
        assert classify_simple_task(_task(priority=priority)) is True

    @pytest.mark.parametrize(
        'description',
        [
            'Refactor architecture for new module',
            'Run integration test against staging',
            'Database migration to v3',
            'Design a new caching layer',
            'Implement the new feature requested in PRD',
        ],
    )
    def test_rejects_hard_blocker_in_description(self, description: str):
        assert classify_simple_task(_task(description=description)) is False

    def test_accepts_short_description(self):
        assert classify_simple_task(_task(description='Quick docstring fix')) is True

    def test_handles_missing_metadata(self):
        # Empty/missing metadata -> empty files list -> 0 <= 2
        task = {'title': 'Document foo', 'description': 'd'}
        assert classify_simple_task(task) is True

    def test_handles_missing_title(self):
        task = {'description': 'd', 'metadata': {'files': []}}
        assert classify_simple_task(task) is False
