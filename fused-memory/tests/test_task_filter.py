"""Tests for reconciliation/task_filter.py — FilteredTaskTree, filter_task_tree, format_filtered_task_tree."""

from __future__ import annotations

from fused_memory.reconciliation.task_filter import (
    FilteredTaskTree,
    filter_task_tree,
    format_filtered_task_tree,
)


def _make_task(tid: int, status: str, title: str | None = None, deps: list | None = None) -> dict:
    return {
        'id': tid,
        'title': title or f'Task {tid}',
        'status': status,
        'dependencies': deps or [],
    }


class TestFilterTaskTree:
    """Tests for filter_task_tree()."""

    def test_partitions_active_done_cancelled_and_other(self):
        """filter_task_tree partitions 8 tasks correctly: 5 active, 1 done, 1 cancelled, 1 other."""
        tasks_data = {
            'tasks': [
                _make_task(1, 'pending'),
                _make_task(2, 'in-progress'),
                _make_task(3, 'blocked'),
                _make_task(4, 'deferred'),
                _make_task(5, 'review'),
                _make_task(6, 'done'),
                _make_task(7, 'cancelled'),
                _make_task(8, 'stalled'),  # unknown status → other
            ]
        }
        result = filter_task_tree(tasks_data)

        assert isinstance(result, FilteredTaskTree)
        active_statuses = {t['status'] for t in result.active_tasks}
        assert active_statuses == {'pending', 'in-progress', 'blocked', 'deferred', 'review'}
        assert len(result.active_tasks) == 5
        assert result.done_count == 1
        assert result.cancelled_count == 1
        assert result.other_count == 1
        assert result.total_count == 8

    def test_handles_empty_and_malformed_inputs(self):
        """filter_task_tree returns empty FilteredTaskTree for empty/malformed inputs."""
        # Empty dict
        result = filter_task_tree({})
        assert result.active_tasks == []
        assert result.done_count == 0
        assert result.cancelled_count == 0
        assert result.other_count == 0
        assert result.total_count == 0

        # tasks is None
        result = filter_task_tree({'tasks': None})
        assert result.active_tasks == []
        assert result.total_count == 0

        # tasks is a list with non-dict elements
        result = filter_task_tree({'tasks': ['not-a-dict', 42, None]})
        assert result.active_tasks == []
        assert result.total_count == 0

        # tasks has task with missing status → other_count
        result = filter_task_tree({'tasks': [{'id': 1}]})
        assert result.active_tasks == []
        assert result.other_count == 1
        assert result.total_count == 1

    def test_sorts_active_by_priority_and_id_desc(self):
        """filter_task_tree sorts active tasks by _STATUS_PRIORITY then ID descending."""
        tasks_data = {
            'tasks': [
                _make_task(10, 'deferred'),
                _make_task(20, 'pending'),
                _make_task(5, 'review'),
                _make_task(15, 'blocked'),
                _make_task(8, 'in-progress'),
                _make_task(3, 'in-progress'),
                _make_task(12, 'pending'),
                _make_task(7, 'blocked'),
            ]
        }
        result = filter_task_tree(tasks_data)

        # All active tasks are present
        assert len(result.active_tasks) == 8

        # Verify sort by priority groups first
        # Priority: in-progress(0) < blocked(1) < review(2) < pending(3) < deferred(4+)
        statuses = [t['status'] for t in result.active_tasks]
        # in-progress tasks come first
        assert statuses[0] == 'in-progress'
        assert statuses[1] == 'in-progress'
        # blocked tasks come after in-progress
        assert statuses[2] == 'blocked'
        assert statuses[3] == 'blocked'
        # review comes after blocked
        assert statuses[4] == 'review'
        # pending comes after review
        assert statuses[5] == 'pending'
        assert statuses[6] == 'pending'
        # deferred comes last
        assert statuses[7] == 'deferred'

        # Within same priority, higher IDs sort first
        in_progress_ids = [t['id'] for t in result.active_tasks if t['status'] == 'in-progress']
        assert in_progress_ids == sorted(in_progress_ids, reverse=True)

        blocked_ids = [t['id'] for t in result.active_tasks if t['status'] == 'blocked']
        assert blocked_ids == sorted(blocked_ids, reverse=True)

        pending_ids = [t['id'] for t in result.active_tasks if t['status'] == 'pending']
        assert pending_ids == sorted(pending_ids, reverse=True)


class TestFormatFilteredTaskTree:
    """Tests for format_filtered_task_tree()."""

    def _make_tree(self, active_count: int = 3, done_count: int = 5, cancelled_count: int = 2, other_count: int = 1) -> FilteredTaskTree:
        active = [_make_task(i + 1, 'pending', f'Task title {i + 1}') for i in range(active_count)]
        total = active_count + done_count + cancelled_count + other_count
        return FilteredTaskTree(
            active_tasks=active,
            done_count=done_count,
            cancelled_count=cancelled_count,
            other_count=other_count,
            total_count=total,
        )

    def test_includes_active_list_and_summary(self):
        """format_filtered_task_tree includes each task and the summary line."""
        tree = self._make_tree(active_count=3, done_count=5, cancelled_count=2, other_count=1)
        output = format_filtered_task_tree(tree)

        # Each active task must appear
        for i in range(1, 4):
            assert f'Task title {i}' in output

        # Summary line with em dash
        assert '5 done, 2 cancelled \u2014 omitted' in output

    def test_caps_at_max_tasks_and_under_budget(self):
        """format_filtered_task_tree caps at max_tasks and keeps output under max_chars."""
        # 500 active tasks with plausible-length titles
        active = [
            _make_task(i, 'pending', f'This is a moderately long title for task number {i} in the queue')
            for i in range(1, 501)
        ]
        tree = FilteredTaskTree(
            active_tasks=active,
            done_count=340,
            cancelled_count=20,
            other_count=0,
            total_count=860,
        )

        output = format_filtered_task_tree(tree)

        # Output must be under budget
        assert len(output) <= 50_000

        # Task 51+ must NOT appear (max_tasks=50 cap)
        assert 'task number 51' not in output, 'task number 51 should be excluded by max_tasks=50 cap'
        assert 'task number 100' not in output, 'task number 100 should be excluded by max_tasks=50 cap'

        # Task 1 MUST appear (first tasks are included)
        assert 'task number 1' in output, 'task number 1 should be present in the output'

        # The header must mention the max_tasks omission
        assert '450 more active omitted by max_tasks cap' in output, (
            'Expected "450 more active omitted by max_tasks cap" in header'
        )

    def test_max_chars_clamp_truncation_count(self):
        """trimmed_count in truncation notice uses len(active-after-max_tasks) not total_active."""
        # 200 tasks in tree, but max_tasks=50 will cap to first 50.
        # With max_chars=300 (tiny), only a few lines will fit.
        # trimmed_count must be relative to the 50-task cap, NOT 200.
        active = [
            _make_task(i, 'pending', 'A' * 80)  # ~80-char title per task
            for i in range(1, 201)
        ]
        tree = FilteredTaskTree(
            active_tasks=active,
            done_count=10,
            cancelled_count=5,
            other_count=0,
            total_count=215,
        )
        output = format_filtered_task_tree(tree, max_tasks=50, max_chars=300)

        # Output must be within the char budget
        assert len(output) <= 300, f'Output length {len(output)} exceeds 300 chars'

        # The truncation notice must say 'N more active (truncated for budget)'
        # where N is relative to the 50-task slice, NOT all 200 tasks.
        # Since max_chars=300 is tiny, 0 or very few task lines will fit,
        # so trimmed_count should be close to 50 (e.g., '50 more' or '49 more').
        # The WRONG value would be close to 200 (e.g., '198 more active' or '200 more active').
        assert 'truncated for budget' in output, 'Expected truncation notice in output'
        # Extract the trimmed count from the notice
        import re
        match = re.search(r'and (\d+) more active \(truncated for budget\)', output)
        assert match is not None, f'Could not find truncation notice in: {output!r}'
        trimmed_count = int(match.group(1))
        assert trimmed_count <= 50, (
            f'trimmed_count={trimmed_count} should be <= 50 (the max_tasks cap), '
            f'not based on total_active=200'
        )

    def test_deps_none_normalized_to_empty_list(self):
        """format_filtered_task_tree renders deps=[] when dependencies key is explicitly None."""
        task_with_none_deps = {
            'id': 42,
            'title': 'Task with null deps',
            'status': 'pending',
            'dependencies': None,  # Explicitly None (not missing)
        }
        tree = FilteredTaskTree(
            active_tasks=[task_with_none_deps],
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=1,
        )
        output = format_filtered_task_tree(tree)

        # Must render deps=[] not deps=None
        assert 'deps=[]' in output, f'Expected deps=[] in output, got: {output!r}'
        assert 'deps=None' not in output, f'Unexpected deps=None in output: {output!r}'

    def test_max_chars_smaller_than_header_does_not_crash(self):
        """format_filtered_task_tree does not crash when max_chars is smaller than the header."""
        active = [_make_task(i, 'pending') for i in range(1, 6)]
        tree = FilteredTaskTree(
            active_tasks=active,
            done_count=2,
            cancelled_count=1,
            other_count=0,
            total_count=8,
        )
        # max_chars=50 is smaller than any realistic header+summary
        output = format_filtered_task_tree(tree, max_chars=50)

        # Must not raise; output must be a string
        assert isinstance(output, str)
        # Must contain the header marker
        assert '### Active Task Tree' in output
        # Must contain the summary line
        assert '\u2014 omitted' in output

    def test_empty_active_and_empty_tree(self):
        """format_filtered_task_tree handles empty FilteredTaskTree gracefully."""
        # Completely empty tree
        empty_tree = FilteredTaskTree(
            active_tasks=[],
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=0,
        )
        output = format_filtered_task_tree(empty_tree)
        assert '0 active' in output
        assert 'No active tasks.' in output

        # Tree with done/cancelled but 0 active
        done_only = FilteredTaskTree(
            active_tasks=[],
            done_count=10,
            cancelled_count=3,
            other_count=0,
            total_count=13,
        )
        output2 = format_filtered_task_tree(done_only)
        # Summary line should still be present
        assert '10 done, 3 cancelled \u2014 omitted' in output2
        assert 'No active tasks.' in output2
